from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from .dataset import DatasetConfig, create_splits
from .preprocessing.segmentation import SegmentationConfig


def build_model(backbone: str, num_classes: int) -> nn.Module:
    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    log_interval: int,
    epoch: int,
):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train {epoch}")
    for batch_idx, (inputs, targets) in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1).detach().cpu().numpy().tolist()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy().tolist())

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            acc = accuracy_score(all_targets, all_preds)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.4f}"})

    epoch_loss = running_loss / max(1, len(loader))
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Val", total=len(loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            preds = outputs.argmax(dim=1).detach().cpu().numpy().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / max(1, len(loader))
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    seg_cfg_raw = cfg.get("segmentation", {})
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    logging_cfg = cfg.get("logging", {})
    early_cfg = train_cfg.get("early_stopping", {})

    dataset_cfg = DatasetConfig(
        root_dir=data_cfg["root_dir"],
        crops=data_cfg["crops"],
        img_size=data_cfg["img_size"],
        train_val_split=data_cfg["train_val_split"],
    )

    seg_cfg = SegmentationConfig(
        hsv_lower=tuple(seg_cfg_raw.get("hsv_lower", [25, 40, 40])),
        hsv_upper=tuple(seg_cfg_raw.get("hsv_upper", [90, 255, 255])),
        kernel_size=seg_cfg_raw.get("kernel_size", 5),
        background_blur=seg_cfg_raw.get("background_blur", True),
    )

    models_dir = Path(train_cfg.get("output_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = models_dir / "class_mapping.json"

    train_ds, val_ds, class_to_idx = create_splits(
        dataset_cfg=dataset_cfg,
        seg_cfg=seg_cfg,
        save_mapping_path=mapping_path,
    )

    num_classes = len(class_to_idx)
    print(f"Discovered {num_classes} classes, {len(train_ds)} train, {len(val_ds)} val")

    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    device_str = train_cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(model_cfg["backbone"], num_classes=num_classes)
    model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_model_path = models_dir / "best_model.pt"
    log_interval = logging_cfg.get("log_interval", 50)
    # Early stopping state
    patience = early_cfg.get("patience", 0)
    min_delta = early_cfg.get("min_delta", 0.0)
    target_val_acc = early_cfg.get("target_val_acc", None)
    epochs_without_improvement = 0

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            log_interval=log_interval,
            epoch=epoch,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        improved = (val_acc - best_val_acc) > min_delta

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": model_cfg["backbone"],
                    "num_classes": num_classes,
                    "class_to_idx": class_to_idx,
                },
                best_model_path,
            )
            print(f"Saved new best model to {best_model_path} (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1

        # Criterion 1: reach target validation accuracy
        if target_val_acc is not None and best_val_acc >= target_val_acc:
            print(
                f"Early stopping: reached target val_acc {target_val_acc:.4f} "
                f"(best_val_acc={best_val_acc:.4f})"
            )
            break

        # Criterion 2: no improvement for 'patience' epochs
        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"Early stopping: no improvement for {patience} epochs "
                f"(best_val_acc={best_val_acc:.4f})"
            )
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(args.config)

