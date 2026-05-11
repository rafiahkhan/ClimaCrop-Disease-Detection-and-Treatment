from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .image_io import (
    load_image_bgr,
    validate_image,
    resize_with_aspect_ratio,
    bgr_to_rgb,
    normalize_imagenet,
)
from .segmentation import (
    SegmentationConfig,
    segment_and_optionally_blur,
    save_debug_segmentation,
)


@dataclass
class PreprocessingConfig:
    img_size: int = 224
    segmentation: Optional[SegmentationConfig] = None
    save_debug: bool = False
    debug_dir: str = "segmented_samples"


def preprocess_image_for_model(
    path: str | Path,
    cfg: PreprocessingConfig,
    debug_name: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Complete preprocessing pipeline used by both Dataset and inference:
      1. Load image
      2. Validate
      3. (Optional) leaf segmentation + background blur
      4. Resize to square with kept aspect ratio
      5. Convert BGR → RGB
      6. Normalize with ImageNet mean/std

    Returns:
        np.ndarray of shape (3, H, W) ready for PyTorch (CHW),
        or None if loading/validation fails.
    """
    img_bgr = load_image_bgr(path)
    if not validate_image(img_bgr):
        return None

    # Segmentation + blur
    if cfg.segmentation is not None:
        processed_bgr, mask = segment_and_optionally_blur(img_bgr, cfg.segmentation)
        if cfg.save_debug and debug_name is not None:
            out_path = Path(cfg.debug_dir) / f"{debug_name}.jpg"
            save_debug_segmentation(img_bgr, processed_bgr, mask, out_path)
        img_bgr = processed_bgr

    # Resize and normalize
    img_bgr = resize_with_aspect_ratio(img_bgr, cfg.img_size)
    img_rgb = bgr_to_rgb(img_bgr)
    img_norm = normalize_imagenet(img_rgb)  # HWC, float32

    # To CHW for PyTorch
    chw = np.transpose(img_norm, (2, 0, 1))
    return chw

