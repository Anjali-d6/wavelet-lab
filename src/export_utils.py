"""In-memory image, CSV, and ZIP exports."""

from __future__ import annotations

from io import BytesIO, StringIO
import csv
import json
import re
import zipfile

import numpy as np
from PIL import Image

from src.image_io import validate_image_array


def safe_stem(filename: str) -> str:
    """Return a short, path-safe filename stem."""
    leaf = filename.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", leaf).strip("_-")
    return (cleaned or "image")[:80]


def encode_image(image: np.ndarray, image_format: str = "PNG") -> bytes:
    """Encode an image as PNG or JPEG and return its bytes."""
    array = validate_image_array(image)
    normalized_format = image_format.upper()
    if normalized_format not in ("PNG", "JPEG", "JPG"):
        raise ValueError("Image export format must be PNG or JPEG.")
    if normalized_format == "JPG":
        normalized_format = "JPEG"
    if normalized_format == "JPEG" and array.ndim == 2:
        pil_image = Image.fromarray(array, mode="L")
    elif normalized_format == "JPEG":
        pil_image = Image.fromarray(array[:, :, :3], mode="RGB")
    else:
        pil_image = Image.fromarray(array)
    buffer = BytesIO()
    pil_image.save(buffer, format=normalized_format, quality=95)
    return buffer.getvalue()


def metrics_csv(rows: list[dict[str, object]]) -> bytes:
    """Serialize measurement rows into UTF-8 CSV bytes."""
    if not rows:
        raise ValueError("At least one metrics row is required.")
    fields: list[str] = list(dict.fromkeys(key for row in rows for key in row))
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def comparison_zip(
    files: dict[str, bytes],
    *,
    metrics: list[dict[str, object]] | None = None,
    parameters: dict[str, object] | None = None,
) -> bytes:
    """Create a ZIP containing only named result files and optional metadata."""
    if not files:
        raise ValueError("At least one file is required for the ZIP export.")
    output = BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        used_names: set[str] = set()
        for name, content in files.items():
            extension = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if extension not in (".png", ".jpg", ".jpeg"):
                extension = ".png"
            base_name = safe_stem(name)
            safe_name = base_name + extension
            suffix = 2
            while safe_name in used_names:
                safe_name = f"{base_name}_{suffix}{extension}"
                suffix += 1
            used_names.add(safe_name)
            archive.writestr(safe_name, content)
        if metrics:
            archive.writestr("metrics.csv", metrics_csv(metrics))
        if parameters:
            archive.writestr(
                "parameters.json",
                json.dumps(parameters, indent=2, sort_keys=True).encode("utf-8"),
            )
    return output.getvalue()
