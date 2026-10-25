import numpy as np
import pytest

from src.classical_enhancement import enhance_clahe, enhance_gamma
from src.metrics import image_statistics
from src.wavelet_enhancement import enhance_wavelet, maximum_level


@pytest.mark.parametrize("shape", [(1, 1), (7, 9), (32, 24), (32, 24, 3)])
@pytest.mark.parametrize("wavelet", ["haar", "db2", "sym2"])
def test_wavelet_output_shape_and_range(shape, wavelet):
    image = np.tile(np.arange(shape[1], dtype=np.uint8), (shape[0], 1))
    if len(shape) == 3:
        image = np.repeat(image[:, :, None], 3, axis=2)
    result = enhance_wavelet(
        image, wavelet_name=wavelet, level=1, strength=0.5, detail_scale=1.3
    )
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0 and result.max() <= 255


def test_wavelet_validation_and_level_limit():
    image = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="positive integer"):
        enhance_wavelet(image, level=0)
    with pytest.raises(ValueError, match="too high"):
        enhance_wavelet(image, wavelet_name="db3", level=5)
    with pytest.raises(ValueError, match="Unsupported wavelet"):
        maximum_level((16, 16), "bad-wavelet")


def test_wavelet_enhances_low_contrast_image_with_valid_independent_output():
    base = np.linspace(112, 144, 96, dtype=np.float64)
    image = np.tile(np.rint(base).astype(np.uint8), (96, 1))
    image[::4, :] += 1

    result = enhance_wavelet(
        image,
        wavelet_name="haar",
        level=2,
        strength=0.5,
        detail_scale=1.25,
    )

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 255
    assert not np.array_equal(result, image)
    assert not np.shares_memory(result, image)
    assert image_statistics(result)["intensity_std"] > image_statistics(image)["intensity_std"]


def test_wavelet_supports_rgb_and_zero_strength():
    image = np.full((24, 32, 3), 120, dtype=np.uint8)
    image[:, 8:24, :] = 132
    result = enhance_wavelet(image, strength=0)

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert np.isfinite(result).all()


@pytest.mark.parametrize("operation", [enhance_gamma, enhance_clahe])
def test_classical_methods_preserve_shape(operation):
    image = np.full((24, 28, 3), 80, dtype=np.uint8)
    result = operation(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_classical_parameter_validation():
    with pytest.raises(ValueError):
        enhance_gamma(np.zeros((3, 3), dtype=np.uint8), gamma=0)
    with pytest.raises(ValueError):
        enhance_clahe(np.zeros((3, 3), dtype=np.uint8), clip_limit=0)
