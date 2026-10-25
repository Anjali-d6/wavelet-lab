import numpy as np
import pytest

from src.visualization import comparison_figure, difference_preview, histogram_figure


def test_histogram_and_comparison_figures_use_actual_rows():
    image = np.tile(np.arange(32, dtype=np.uint8), (16, 1))
    histogram = histogram_figure([("Original", image), ("Output", image + 1)])
    assert len(histogram.data) == 2
    assert histogram.data[0].name == "Original"

    rows = [
        {"method": "Original", "mean_intensity": 15, "intensity_std": 9, "processing_time_ms": 0},
        {"method": "Wavelet", "mean_intensity": 16, "intensity_std": 10, "processing_time_ms": 1.25},
    ]
    comparison = comparison_figure(rows)
    assert len(comparison.data) == 3
    assert list(comparison.data[0].y) == [15, 16]
    assert list(comparison.data[1].y) == [9, 10]


def test_difference_preview_is_valid_and_handles_equal_arrays():
    original = np.zeros((10, 12), dtype=np.uint8)
    enhanced = original.copy()
    enhanced[:, 6:] = 20
    preview = difference_preview(original, enhanced)
    assert preview.shape == original.shape
    assert preview.dtype == np.uint8
    assert preview.max() == 255
    assert np.count_nonzero(difference_preview(original, original)) == 0


def test_difference_preview_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="matching dimensions"):
        difference_preview(np.zeros((8, 8)), np.zeros((9, 8)))
