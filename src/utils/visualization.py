from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def save_tensor_as_image(
    tensor_chw: np.ndarray,
    out_path: str | Path,
    denormalize: bool = True,
) -> None:
    """
    Save a CHW tensor as an RGB image using matplotlib.
    If denormalize=True, apply inverse ImageNet normalization.
    """
    img = tensor_chw.copy()
    if denormalize:
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]
        img = img * std + mean

    img = np.clip(img, 0.0, 1.0)
    img_hwc = np.transpose(img, (1, 2, 0))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, img_hwc)


def show_tensor(
    tensor_chw: np.ndarray,
    title: Optional[str] = None,
    denormalize: bool = True,
) -> None:
    img = tensor_chw.copy()
    if denormalize:
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]
        img = img * std + mean

    img = np.clip(img, 0.0, 1.0)
    img_hwc = np.transpose(img, (1, 2, 0))

    plt.figure(figsize=(4, 4))
    plt.imshow(img_hwc)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()

