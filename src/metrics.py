"""Image statistics and reference-based quality measurements."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.image_io import validate_image_array


def grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a validated image to float grayscale values in the 0-255 range."""
    array = validate_image_array(image)
    if array.ndim == 2:
        return array.astype(np.float64)
    rgb = array[:, :, :3].astype(np.float64)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def image_statistics(image: np.ndarray) -> dict[str, float | int]:
    """Return mean intensity, standard deviation, RMS contrast, and dimensions."""
    gray = grayscale(image)
    return {
        "mean_intensity": float(np.mean(gray)),
        "intensity_std": float(np.std(gray)),
        "rms_contrast": float(np.std(gray)),
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
    }


def reference_metrics(
    image: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    """Calculate PSNR and SSIM only for equal-sized aligned arrays."""
    candidate = validate_image_array(image)
    target = validate_image_array(reference)
    if candidate.shape != target.shape:
        raise ValueError("Reference and result must have identical dimensions and channels.")
    smallest_dimension = min(candidate.shape[:2])
    window_size = min(7, smallest_dimension if smallest_dimension % 2 else smallest_dimension - 1)
    if window_size < 3:
        raise ValueError("SSIM requires reference images at least 3 pixels wide and tall.")
    channel_axis = -1 if candidate.ndim == 3 else None
    return {
        "psnr": (
            float("inf")
            if np.array_equal(target, candidate)
            else float(peak_signal_noise_ratio(target, candidate, data_range=255))
        ),
        "ssim": float(
            structural_similarity(
                target,
                candidate,
                data_range=255,
                channel_axis=channel_axis,
                win_size=window_size,
            )
        ),
    }
