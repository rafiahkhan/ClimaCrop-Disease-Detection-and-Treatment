from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SegmentationConfig:
    hsv_lower: tuple[int, int, int] = (25, 40, 40)
    hsv_upper: tuple[int, int, int] = (90, 255, 255)
    kernel_size: int = 5
    background_blur: bool = True


def create_leaf_mask_hsv(
    img_bgr: np.ndarray, cfg: SegmentationConfig
) -> np.ndarray:
    """
    Simple unsupervised leaf segmentation using HSV thresholds.

    This is NOT perfect semantic segmentation, but it is usually good enough
    to:
      - keep the leaf area sharp
      - blur the non-leaf background
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg.hsv_lower, dtype=np.uint8)
    upper = np.array(cfg.hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    k = cfg.kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # Clean up noise: first close (fill small holes), then open (remove noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Ensure mask is 0/255 uint8
    mask = (mask > 0).astype(np.uint8) * 255
    return mask


def apply_background_blur(
    img_bgr: np.ndarray, mask: np.ndarray, blur_kernel: int = 21
) -> np.ndarray:
    """
    Blur only the background (where mask == 0), keep leaf region sharp.
    """
    if mask.ndim == 2:
        mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask_3c = mask

    blurred = cv2.GaussianBlur(img_bgr, (blur_kernel, blur_kernel), 0)

    # Normalize mask to [0,1] float
    alpha = (mask_3c.astype(np.float32) / 255.0)
    result = alpha * img_bgr.astype(np.float32) + (1.0 - alpha) * blurred.astype(
        np.float32
    )
    return result.astype(np.uint8)


def segment_and_optionally_blur(
    img_bgr: np.ndarray, cfg: SegmentationConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full segmentation step:
      1. Get leaf mask
      2. Optionally blur background

    Returns:
        processed_bgr, mask
    """
    mask = create_leaf_mask_hsv(img_bgr, cfg)
    if cfg.background_blur:
        processed = apply_background_blur(img_bgr, mask)
    else:
        processed = img_bgr.copy()
    return processed, mask


def save_debug_segmentation(
    original_bgr: np.ndarray,
    processed_bgr: np.ndarray,
    mask: np.ndarray,
    out_path: str | Path,
) -> None:
    """
    Save a side-by-side visualization: original | processed | mask.
    Useful to visually verify segmentation quality.
    """
    h, w = original_bgr.shape[:2]
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_color = cv2.resize(mask_color, (w, h), interpolation=cv2.INTER_NEAREST)

    processed_resized = cv2.resize(processed_bgr, (w, h), interpolation=cv2.INTER_AREA)

    concat = cv2.hconcat([original_bgr, processed_resized, mask_color])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), concat)

