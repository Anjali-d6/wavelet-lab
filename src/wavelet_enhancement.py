"""Two-dimensional wavelet image enhancement and optional soft-threshold denoising."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pywt

from src.image_io import validate_image_array


def maximum_level(shape: tuple[int, int], wavelet_name: str) -> int:
    """Return the maximum useful DWT level for an image and wavelet."""
    try:
        wavelet = pywt.Wavelet(wavelet_name)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported wavelet: {wavelet_name}") from exc
    return max(1, pywt.dwt_max_level(min(shape), wavelet.dec_len))


def _soft_threshold(details: tuple[np.ndarray, ...], strength: float) -> tuple[np.ndarray, ...]:
    if strength == 0:
        return details
    diagonal = details[2]
    sigma = float(np.median(np.abs(diagonal)) / 0.6745) if diagonal.size else 0.0
    threshold = strength * sigma * math.sqrt(2 * math.log(max(diagonal.size, 2)))
    return tuple(pywt.threshold(coeff, threshold, mode="soft") for coeff in details)


def _enhance_luminance(
    luminance: np.ndarray,
    *,
    wavelet_name: str,
    level: int,
    strength: float,
    detail_scale: float,
    denoise: bool,
    denoise_strength: float,
) -> np.ndarray:
    unit = luminance.astype(np.float64) / 255.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        coefficients = pywt.wavedec2(
            unit, wavelet=wavelet_name, mode="symmetric", level=level
        )
    approximation = coefficients[0]
    average = float(np.mean(approximation))
    contrast_gain = 1.0 + 0.6 * strength
    adjusted = [((approximation - average) * contrast_gain + average)]
    detail_multiplier = min(2.0, max(0.5, 1 + strength * (detail_scale - 1)))
    for horizontal, vertical, diagonal in coefficients[1:]:
        details = (horizontal, vertical, diagonal)
        if denoise:
            details = _soft_threshold(details, denoise_strength)
        adjusted.append(tuple(detail * detail_multiplier for detail in details))
    reconstructed = pywt.waverec2(
        adjusted, wavelet=wavelet_name, mode="symmetric"
    )
    reconstructed = reconstructed[: luminance.shape[0], : luminance.shape[1]]
    return np.clip(np.rint(reconstructed * 255), 0, 255).astype(np.uint8)


def enhance_wavelet(
    image: np.ndarray,
    *,
    wavelet_name: str = "haar",
    level: int = 1,
    strength: float = 0.5,
    detail_scale: float = 1.25,
    denoise: bool = False,
    denoise_strength: float = 0.5,
) -> np.ndarray:
    """Enhance luminance by scaling DWT approximation contrast and detail coefficients."""
    source = validate_image_array(image)
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Enhancement strength must be between 0 and 1.")
    if not np.isfinite(detail_scale) or not 0.5 <= detail_scale <= 2:
        raise ValueError("Detail scaling must be between 0.5 and 2.")
    if not np.isfinite(denoise_strength) or not 0 <= denoise_strength <= 1:
        raise ValueError("Denoising strength must be between 0 and 1.")
    if not isinstance(level, int) or isinstance(level, bool) or level < 1:
        raise ValueError("Decomposition level must be a positive integer.")
    allowed_level = maximum_level(source.shape[:2], wavelet_name)
    if level > allowed_level:
        raise ValueError(
            f"Level {level} is too high for this image and wavelet; use level {allowed_level} or lower."
        )

    if source.ndim == 2:
        return _enhance_luminance(
            source,
            wavelet_name=wavelet_name,
            level=level,
            strength=strength,
            detail_scale=detail_scale,
            denoise=denoise,
            denoise_strength=denoise_strength,
        )

    rgb = source.astype(np.float32)
    luminance = np.clip(
        np.rint(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]),
        0,
        255,
    ).astype(np.uint8)
    enhanced_luminance = _enhance_luminance(
        luminance,
        wavelet_name=wavelet_name,
        level=level,
        strength=strength,
        detail_scale=detail_scale,
        denoise=denoise,
        denoise_strength=denoise_strength,
    )
    delta = enhanced_luminance.astype(np.float32) - luminance.astype(np.float32)
    return np.clip(np.rint(rgb + delta[:, :, None]), 0, 255).astype(np.uint8)
