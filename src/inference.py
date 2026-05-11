from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import yaml
from torchvision import models

from .preprocessing.pipeline import PreprocessingConfig, preprocess_image_for_model
from .preprocessing.segmentation import SegmentationConfig


def build_model_for_inference(
    backbone: str, num_classes: int, state_dict: dict
) -> nn.Module:
    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_llm_recommendations(
    disease_name: str,
    model_name: str = "llama3.1:8b",
    ollama_url: str | None = None,
) -> str:
    """
    Get treatment recommendations for a detected disease using Ollama LLM.
    No cloud API keys: set OLLAMA_BASE_URL if Ollama runs on another host/port.

    Args:
        disease_name: The name of the detected disease
        model_name: The Ollama model to use (default: llama3.1:8b)
        ollama_url: Base URL for Ollama (default: env OLLAMA_BASE_URL or http://127.0.0.1:11434)

    Returns:
        A string containing treatment recommendations
    """
    base = (ollama_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    prompt = f"""You are an agricultural expert. A plant disease has been detected: {disease_name}.

Start your response with a title line in this format: "[Crop]'s [Disease Name] Disease Treatment Recommendations"
Example: "Cotton's Bacterial Blight Disease Treatment Recommendations"

Then provide these 4 sections with exactly 2-3 bullet points each. Keep it concise:

**Immediate Actions:**
- List 2-3 immediate actions (isolate, remove infected plants, monitor spread)

**Chemical Treatments:**
- List 2-3 chemical treatments (product names or active ingredients with rates)

**Organic/Natural Treatment Options:**
- List 2-3 organic/natural remedies

**Cultural Practices:**
- List 2-3 cultural practices (crop rotation, planting, sanitation, etc.)

Use the exact section headers above. Be specific and practical."""

    try:
        response = requests.post(
            f"{base}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,  # Increased timeout to 120 seconds for LLM generation
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "Unable to generate recommendations at this time.")
    except requests.exceptions.Timeout:
        return f"Request to Ollama timed out. The model '{model_name}' may be taking longer than expected. Please try again or check if Ollama is running properly."
    except requests.exceptions.ConnectionError:
        return f"Cannot connect to Ollama at {base}. Please ensure Ollama is running: 'ollama serve'"
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {str(e)}\nPlease ensure Ollama is running and the model '{model_name}' is available."
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"


def main(config_path: str, image_path: str, llm_model: str = "llama3.1:8b", enable_llm: bool = True):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    seg_cfg_raw = cfg.get("segmentation", {})
    train_cfg = cfg["training"]

    seg_cfg = SegmentationConfig(
        hsv_lower=tuple(seg_cfg_raw.get("hsv_lower", [25, 40, 40])),
        hsv_upper=tuple(seg_cfg_raw.get("hsv_upper", [90, 255, 255])),
        kernel_size=seg_cfg_raw.get("kernel_size", 5),
        background_blur=seg_cfg_raw.get("background_blur", True),
    )

    preproc_cfg = PreprocessingConfig(
        img_size=data_cfg["img_size"],
        segmentation=seg_cfg,
        save_debug=False,
    )

    models_dir = Path(train_cfg.get("output_dir", "models"))
    best_model_path = models_dir / "best_model.pt"
    mapping_path = models_dir / "class_mapping.json"

    if not best_model_path.is_file():
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Class mapping not found at {mapping_path}")

    checkpoint = torch.load(best_model_path, map_location="cpu")
    backbone = checkpoint["backbone"]
    num_classes = checkpoint["num_classes"]
    state_dict = checkpoint["model_state_dict"]

    with mapping_path.open("r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model_for_inference(backbone, num_classes, state_dict)
    device_str = train_cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model.to(device)

    img_chw = preprocess_image_for_model(image_path, preproc_cfg, debug_name=None)
    if img_chw is None:
        raise RuntimeError(f"Failed to load/validate image: {image_path}")

    x = torch.from_numpy(img_chw).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_class = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx])

    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")
    
    if enable_llm:
        print("\n" + "="*60)
        print("GENERATING TREATMENT RECOMMENDATIONS...")
        print("="*60 + "\n")
        
        # Get LLM recommendations for the detected disease
        recommendations = get_llm_recommendations(pred_class, model_name=llm_model)
        print("TREATMENT RECOMMENDATIONS:")
        print("-" * 60)
        print(recommendations)
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to single leaf image",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="llama3.1:8b",
        help="Ollama model to use for recommendations (default: llama3.1:8b)",
    )
    parser.add_argument(
        "--disable_llm",
        action="store_true",
        help="Disable LLM recommendations",
    )
    args = parser.parse_args()
    main(args.config, args.image_path, llm_model=args.llm_model, enable_llm=not args.disable_llm)

