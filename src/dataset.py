from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .preprocessing.pipeline import PreprocessingConfig, preprocess_image_for_model
from .preprocessing.segmentation import SegmentationConfig
from .utils.transforms import apply_simple_augmentations_chw


@dataclass
class DatasetConfig:
    root_dir: str
    crops: List[str]
    img_size: int
    train_val_split: float


def discover_images_and_labels(
    cfg: DatasetConfig,
) -> Tuple[List[str], List[int], dict]:
    """
    Walk through crop folders and build a flat list of image paths and integer labels.

    Label naming scheme: 'Crop_DiseaseFolder', e.g. 'Cottons_Aphids', 'Rice_Healthy'.
    """
    root = Path(cfg.root_dir)
    paths: List[str] = []
    labels: List[int] = []
    class_to_idx: dict[str, int] = {}

    for crop in cfg.crops:
        crop_dir = root / crop
        if not crop_dir.is_dir():
            continue
        for disease_dir in sorted(crop_dir.iterdir()):
            if not disease_dir.is_dir():
                continue

            class_name = f"{crop}_{disease_dir.name}"
            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(class_to_idx)

            idx = class_to_idx[class_name]
            for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
                for img_path in disease_dir.glob(ext):
                    paths.append(str(img_path))
                    labels.append(idx)

    return paths, labels, class_to_idx


class LeafDiseaseDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        preprocessing_cfg: PreprocessingConfig,
        augment: bool = False,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessing_cfg = preprocessing_cfg
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        label = self.labels[idx]

        debug_name = None
        if self.preprocessing_cfg.save_debug and self.augment is False:
            # Save only from the validation set usually
            debug_name = Path(path).stem

        img_chw = preprocess_image_for_model(
            path,
            cfg=self.preprocessing_cfg,
            debug_name=debug_name,
        )
        if img_chw is None:
            # In case of a corrupted image, skip by returning a zero tensor
            img_chw = np.zeros(
                (3, self.preprocessing_cfg.img_size, self.preprocessing_cfg.img_size),
                dtype=np.float32,
            )

        if self.augment:
            img_chw = apply_simple_augmentations_chw(img_chw)

        tensor = torch.from_numpy(img_chw).float()
        return tensor, torch.tensor(label, dtype=torch.long)


def create_splits(
    dataset_cfg: DatasetConfig,
    seg_cfg: SegmentationConfig,
    save_mapping_path: Path,
):
    """
    Create train/val datasets and save class mapping to JSON.
    """
    image_paths, labels, class_to_idx = discover_images_and_labels(dataset_cfg)

    # Save mapping for inference
    save_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with save_mapping_path.open("w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    X_train, X_val, y_train, y_val = train_test_split(
        image_paths,
        labels,
        train_size=dataset_cfg.train_val_split,
        stratify=labels,
        random_state=42,
    )

    base_preproc_cfg = PreprocessingConfig(
        img_size=dataset_cfg.img_size,
        segmentation=seg_cfg,
    )

    train_ds = LeafDiseaseDataset(
        X_train,
        y_train,
        preprocessing_cfg=base_preproc_cfg,
        augment=True,
    )
    val_preproc_cfg = PreprocessingConfig(
        img_size=dataset_cfg.img_size,
        segmentation=seg_cfg,
        save_debug=False,
    )
    val_ds = LeafDiseaseDataset(
        X_val,
        y_val,
        preprocessing_cfg=val_preproc_cfg,
        augment=False,
    )

    return train_ds, val_ds, class_to_idx

