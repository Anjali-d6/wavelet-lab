from io import BytesIO
import zipfile

import numpy as np
from PIL import Image
import pytest

from src.batch_processing import process_batch
from src.export_utils import comparison_zip, encode_image, metrics_csv, safe_stem
from src.metrics import image_statistics, reference_metrics


def encoded(array: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def test_statistics_and_reference_metrics():
    image = np.full((10, 12), 100, dtype=np.uint8)
    stats = image_statistics(image)
    assert stats["mean_intensity"] == 100
    assert stats["intensity_std"] == 0
    assert stats["width"] == 12 and stats["height"] == 10
    scores = reference_metrics(image, image.copy())
    assert scores["psnr"] == float("inf")
    assert scores["ssim"] == pytest.approx(1)
    with pytest.raises(ValueError, match="identical dimensions"):
        reference_metrics(image, np.zeros((9, 12), dtype=np.uint8))
    with pytest.raises(ValueError, match="at least 3 pixels"):
        reference_metrics(np.zeros((1, 2), dtype=np.uint8), np.zeros((1, 2), dtype=np.uint8))


def test_exports_reopen_and_zip_contents_are_limited():
    image = np.full((8, 9, 3), 120, dtype=np.uint8)
    for image_format in ("PNG", "JPEG"):
        with Image.open(BytesIO(encode_image(image, image_format))) as reopened:
            assert reopened.size == (9, 8)
    csv_bytes = metrics_csv([{"method": "test", "mean": 12}])
    assert b"method,mean" in csv_bytes
    archive_bytes = comparison_zip(
        {
            "../unsafe.png": encode_image(image),
            "unsafe.png": encode_image(image),
        },
        metrics=[{"method": "test"}],
        parameters={"level": 1},
    )
    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        names = set(archive.namelist())
        assert names == {"unsafe.png", "unsafe_2.png", "metrics.csv", "parameters.json"}
        assert all(".." not in name and "/" not in name for name in names)
    assert safe_stem("../../unsafe filename.png") == "unsafe_filename"


def test_batch_keeps_processing_after_bad_file():
    good = encoded(np.full((8, 8), 100, dtype=np.uint8))
    results = process_batch(
        [("good.png", good), ("bad.png", b"broken")],
        method="Gamma", parameters={"gamma": 1.2},
    )
    assert results[0]["success"] is True
    assert results[1]["success"] is False


def test_batch_accepts_dashboard_gamma_method_name():
    image = encoded(np.full((8, 8), 100, dtype=np.uint8))
    result = process_batch(
        [("image.png", image)],
        method="Gamma Correction",
        parameters={"gamma": 1.2},
    )
    assert result[0]["success"] is True


def test_batch_enforces_cumulative_pixel_limit():
    good = encoded(np.full((8, 8), 100, dtype=np.uint8))
    results = process_batch(
        [("first.png", good), ("second.png", good)],
        method="Gamma",
        parameters={"gamma": 1.2},
        max_total_pixels=100,
    )
    assert results[0]["success"] is True
    assert results[1]["success"] is False
    assert "cumulative pixel limit" in results[1]["error"]
