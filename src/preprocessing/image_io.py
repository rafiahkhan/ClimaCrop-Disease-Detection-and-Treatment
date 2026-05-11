import cv2
import numpy as np
from pathlib import Path
from typing import Optional


def load_image_bgr(path: str | Path) -> Optional[np.ndarray]:
    """
    Load an image from disk in BGR format (as used by OpenCV).

    Returns:
        np.ndarray of shape (H, W, 3) in BGR format, or None if loading fails.
    """
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def validate_image(img: np.ndarray, min_size: int = 64) -> bool:
    """
    Simple sanity checks for an image.
    """
    if img is None:
        return False
    if img.ndim != 3 or img.shape[2] != 3:
        return False
    h, w = img.shape[:2]
    if h < min_size or w < min_size:
        return False
    return True


def resize_with_aspect_ratio(
    img: np.ndarray,
    target_size: int,
) -> np.ndarray:
    """
    Resize image keeping aspect ratio, then center-crop/pad to square.
    This is suitable for classification backbones like EfficientNet/ResNet.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img

    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create square canvas and paste resized image in the center
    canvas = np.zeros((target_size, target_size, 3), dtype=resized.dtype)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def normalize_imagenet(img_rgb: np.ndarray) -> np.ndarray:
    """
    Normalize an RGB image with ImageNet mean/std.
    Expects float32 in [0, 1].
    """
    img = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return img

