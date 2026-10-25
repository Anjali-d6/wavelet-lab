"""Classical contrast and gamma enhancement methods."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from skimage import exposure

from src.image_io import validate_image_array


def _adjust_luminance(
    image: np.ndarray, operation: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    source = validate_image_array(image)
    if source.ndim == 2:
        return operation(source)
    rgb = source.astype(np.float32)
    luminance = np.clip(
        np.rint(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]),
        0,
        255,
    ).astype(np.uint8)
    enhanced = operation(luminance)
    delta = enhanced.astype(np.float32) - luminance.astype(np.float32)
    return np.clip(np.rint(rgb + delta[:, :, None]), 0, 255).astype(np.uint8)


def enhance_gamma(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Apply gamma correction to intensity; gamma above one brightens the image."""
    if not np.isfinite(gamma) or not 0.1 <= gamma <= 4:
        raise ValueError("Gamma must be between 0.1 and 4.")

    def apply(channel: np.ndarray) -> np.ndarray:
        normalized = channel.astype(np.float32) / 255
        return np.clip(np.rint(np.power(normalized, 1 / gamma) * 255), 0, 255).astype(
            np.uint8
        )

    return _adjust_luminance(image, apply)


def enhance_clahe(image: np.ndarray, clip_limit: float = 0.01) -> np.ndarray:
    """Apply contrast-limited adaptive histogram equalization to image luminance."""
    if not np.isfinite(clip_limit) or not 0.001 <= clip_limit <= 0.1:
        raise ValueError("CLAHE clip limit must be between 0.001 and 0.1.")

    def apply(channel: np.ndarray) -> np.ndarray:
        equalized = exposure.equalize_adapthist(
            channel, kernel_size=None, clip_limit=clip_limit, nbins=256
        )
        return np.clip(np.rint(equalized * 255), 0, 255).astype(np.uint8)

    return _adjust_luminance(image, apply)
