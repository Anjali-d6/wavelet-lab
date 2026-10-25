from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from src.image_io import ImageValidationError, decode_image, validate_image_array


def encoded(array: np.ndarray, fmt: str = "PNG") -> bytes:
    stream = BytesIO()
    Image.fromarray(array).save(stream, format=fmt)
    return stream.getvalue()


@pytest.mark.parametrize(
    ("array", "filename", "shape"),
    [
        (np.arange(64, dtype=np.uint8).reshape(8, 8), "gray.png", (8, 8)),
        (np.zeros((9, 11, 3), dtype=np.uint8), "rgb.jpg", (9, 11, 3)),
        (np.zeros((5, 7, 4), dtype=np.uint8), "rgba.png", (5, 7, 3)),
        (np.zeros((4, 6), dtype=np.uint8), "tiny.tiff", (4, 6)),
    ],
)
def test_decode_supported_images(array, filename, shape):
    format_name = "TIFF" if filename.endswith("tiff") else ("JPEG" if filename.endswith("jpg") else "PNG")
    result = decode_image(encoded(array, format_name), filename)
    assert result.shape == shape
    assert result.dtype == np.uint8


def test_rejects_invalid_data_and_extension_mismatch():
    with pytest.raises(ImageValidationError):
        decode_image(b"not an image", "bad.png")
    with pytest.raises(ImageValidationError, match="extension"):
        decode_image(encoded(np.zeros((3, 3), dtype=np.uint8)), "wrong.jpg")


def test_rejects_upload_and_pixel_limits():
    data = encoded(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ImageValidationError, match="upload limit"):
        decode_image(data, "large.png", max_bytes=1)
    with pytest.raises(ImageValidationError, match="pixel limit"):
        decode_image(data, "many.png", max_pixels=10)


def test_validate_float_and_invalid_array():
    assert validate_image_array(np.ones((2, 2), dtype=np.float32)).max() == 255
    with pytest.raises(ImageValidationError):
        validate_image_array(np.array([[np.nan]]))
