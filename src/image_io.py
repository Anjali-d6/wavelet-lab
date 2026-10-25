"""Validated, in-memory image decoding for uploaded files."""

from __future__ import annotations

from io import BytesIO
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageValidationError(ValueError):
    """Raised when an uploaded image is not safe or supported."""


SUPPORTED_FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_PIXELS = 20_000_000


def decode_image(
    data: bytes,
    filename: str,
    *,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> np.ndarray:
    """Decode a supported upload as uint8 grayscale or RGB without saving it."""
    if not isinstance(data, bytes) or not data:
        raise ImageValidationError("The uploaded file is empty or invalid.")
    if len(data) > max_bytes:
        raise ImageValidationError(
            f"File is larger than the {max_bytes // (1024 * 1024)} MB upload limit."
        )

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    expected_format = SUPPORTED_FORMATS.get(f".{extension}")
    if expected_format is None:
        raise ImageValidationError("Use a PNG, JPG, JPEG, or TIFF image.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format != expected_format:
                    raise ImageValidationError(
                        "The file extension does not match the image contents."
                    )
                width, height = image.size
                if width < 1 or height < 1 or width * height > max_pixels:
                    raise ImageValidationError(
                        f"Image exceeds the {max_pixels:,}-pixel limit."
                    )
                image.load()
                if image.mode in ("1", "L", "I", "F", "I;16"):
                    return np.asarray(image.convert("L"), dtype=np.uint8).copy()
                return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except ImageValidationError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ImageValidationError(
            "The image could not be decoded. Check that it is a valid PNG, JPG, or TIFF."
        ) from exc


def validate_image_array(image: np.ndarray) -> np.ndarray:
    """Validate an image array and return a safe uint8 copy."""
    array = np.asarray(image)
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (3, 4)):
        raise ImageValidationError("Expected a grayscale, RGB, or RGBA image.")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ImageValidationError("Image dimensions must be greater than zero.")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ImageValidationError("Image pixels must contain finite numeric values.")
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype == np.uint8:
        return array.copy()
    if np.issubdtype(array.dtype, np.floating) and array.size and array.min() >= 0 and array.max() <= 1:
        array = array * 255
    return np.clip(array, 0, 255).astype(np.uint8)
