"""Image preprocessing helpers for Phase 2 baselines.

The Phase 2 baseline experiments use aspect-ratio preserving resize with
padding, followed by ImageNet normalization for TorchVision backbones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image_rgb(image_path: str | Path) -> Image.Image:
    """Load an image and convert it to RGB."""
    return Image.open(image_path).convert("RGB")


def resize_with_padding(
    image: Image.Image,
    size: tuple[int, int] = (224, 224),
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Resize an image to fit within `size` while preserving aspect ratio.

    The remaining area is center-padded with `fill` so leaf morphology is not
    distorted by a hard crop.
    """
    target_width, target_height = size
    width, height = image.size

    if width == 0 or height == 0:
        raise ValueError("Image has invalid dimensions")

    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resized = image.resize((resized_width, resized_height),
                           Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, fill)

    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def image_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert an RGB PIL image to a float32 CHW numpy array in [0, 1]."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def imagenet_normalize(array_chw: np.ndarray) -> np.ndarray:
    """Apply ImageNet normalization to a CHW numpy array."""
    if array_chw.ndim != 3 or array_chw.shape[0] != 3:
        raise ValueError("Expected CHW array with 3 channels")

    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    return (array_chw - mean) / std


def preprocess_image(
    image_path: str | Path,
    size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Load, resize, and normalize one image for backbone input."""
    image = load_image_rgb(image_path)
    resized = resize_with_padding(image, size=size)
    return imagenet_normalize(image_to_numpy(resized))
