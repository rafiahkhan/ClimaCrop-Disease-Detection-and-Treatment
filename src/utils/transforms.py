from __future__ import annotations

import numpy as np
import cv2
import random


def apply_simple_augmentations_chw(
    tensor_chw: np.ndarray,
    rotation_deg: int = 15,
    hflip_prob: float = 0.5,
) -> np.ndarray:
    """
    Apply simple geometric augmentations directly on a CHW tensor:
    - Random horizontal flip
    - Small random rotation (within ±rotation_deg)

    This avoids external augmentation libraries and keeps behaviour stable
    across environments.
    """
    img = tensor_chw

    # CHW -> HWC for OpenCV
    img_hwc = np.transpose(img, (1, 2, 0))

    # Random horizontal flip
    if random.random() < hflip_prob:
        img_hwc = cv2.flip(img_hwc, 1)

    # Random small rotation around image center
    angle = random.uniform(-rotation_deg, rotation_deg)
    h, w = img_hwc.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_hwc = cv2.warpAffine(
        img_hwc,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # HWC -> CHW
    img_chw = np.transpose(img_hwc, (2, 0, 1))
    return img_chw

