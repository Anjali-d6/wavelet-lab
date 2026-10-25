"""Isolated per-file batch processing with error reporting."""

from __future__ import annotations

from time import perf_counter
from collections.abc import Callable
from typing import Any

from src.classical_enhancement import enhance_clahe, enhance_gamma
from src.image_io import ImageValidationError, decode_image
from src.metrics import image_statistics
from src.wavelet_enhancement import enhance_wavelet


def process_batch(
    uploads: list[tuple[str, bytes]],
    *,
    method: str,
    parameters: dict[str, Any],
    max_files: int = 20,
    max_total_pixels: int = 40_000_000,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Process each upload independently and return a status record per file."""
    if len(uploads) > max_files:
        raise ValueError(f"Choose no more than {max_files} files at a time.")
    processors = {
        "Wavelet": enhance_wavelet,
        "CLAHE": enhance_clahe,
        "Gamma": enhance_gamma,
        "Gamma Correction": enhance_gamma,
    }
    if method not in processors:
        raise ValueError(f"Unsupported batch method: {method}")
    results: list[dict[str, Any]] = []
    processed_pixels = 0
    for filename, data in uploads:
        started = perf_counter()
        try:
            original = decode_image(data, filename)
            pixel_count = original.shape[0] * original.shape[1]
            if processed_pixels + pixel_count > max_total_pixels:
                raise ValueError(
                    f"Batch cumulative pixel limit ({max_total_pixels:,} pixels) exceeded."
                )
            enhanced = processors[method](original, **parameters)
            processed_pixels += pixel_count
            results.append(
                {
                    "filename": filename,
                    "success": True,
                    "image": enhanced,
                    "error": "",
                    "processing_time_ms": (perf_counter() - started) * 1000,
                    **image_statistics(enhanced),
                }
            )
        except (ImageValidationError, ValueError, OSError) as exc:
            results.append(
                {
                    "filename": filename,
                    "success": False,
                    "image": None,
                    "error": str(exc),
                    "processing_time_ms": (perf_counter() - started) * 1000,
                }
            )
        if progress_callback is not None:
            progress_callback(len(results), len(uploads), filename)
    return results
