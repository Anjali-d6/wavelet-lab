"""Interactive, data-backed Plotly charts for the Wavelet Lab dashboard."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.metrics import grayscale

NAVY = "#17324d"
TEAL = "#188c8c"
GOLD = "#d49339"


def _template(dark: bool) -> str:
    return "plotly_dark" if dark else "plotly_white"


def histogram_figure(
    images: Sequence[tuple[str, np.ndarray]], *, dark: bool = False
) -> go.Figure:
    """Build an interactive normalized grayscale intensity histogram."""
    figure = go.Figure()
    colors = (
        ["#8bbcff", "#48d4c2", "#ffca72", "#b6a0ff", "#8fd19e"]
        if dark
        else [NAVY, TEAL, GOLD, "#7187b3", "#6b9b72"]
    )
    for index, (label, image) in enumerate(images):
        values = grayscale(image).ravel()
        counts, edges = np.histogram(values, bins=64, range=(0, 256), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        figure.add_trace(
            go.Scatter(
                x=centers,
                y=counts,
                mode="lines",
                name=label,
                line={"color": colors[index % len(colors)], "width": 2},
                hovertemplate="Intensity: %{x:.1f}<br>Density: %{y:.5f}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(
        template=_template(dark),
        title="Grayscale intensity distribution",
        xaxis_title="Grayscale intensity (0–255)",
        yaxis_title="Normalized frequency",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 24, "r": 20, "t": 72, "b": 24},
        height=340,
    )
    return figure


def comparison_figure(
    rows: Sequence[dict[str, object]], *, dark: bool = False
) -> go.Figure:
    """Build interactive means, contrast, and processing-time comparisons."""
    if not rows:
        raise ValueError("At least one measurement row is needed for a chart.")

    names = [str(row["method"]) for row in rows]
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Mean intensity", "Contrast (standard deviation)", "Processing time"),
        horizontal_spacing=0.1,
    )
    metrics = (
        ("mean_intensity", "Intensity (0–255)", "#8bbcff" if dark else NAVY),
        ("contrast_std", "Standard deviation (0–127.5)", "#48d4c2" if dark else TEAL),
        ("processing_time_ms", "Milliseconds", "#ffca72" if dark else GOLD),
    )
    for column, (key, unit, color) in enumerate(metrics, start=1):
        values = [
            float(row.get(key, row.get("intensity_std", 0)) or 0)
            for row in rows
        ]
        figure.add_trace(
            go.Bar(
                x=names,
                y=values,
                marker_color=color,
                name=unit,
                showlegend=False,
                text=[f"{value:.2f}" for value in values],
                textposition="outside",
                hovertemplate="%{x}<br>" + unit + ": %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
        figure.update_yaxes(title_text=unit, rangemode="tozero", row=1, col=column)
        figure.update_xaxes(tickangle=-20, row=1, col=column)
    figure.update_layout(
        template=_template(dark),
        height=370,
        margin={"l": 18, "r": 18, "t": 75, "b": 75},
        bargap=0.4,
        showlegend=False,
    )
    return figure


def difference_preview(original: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    """Create a grayscale absolute-difference preview with per-image display scaling."""
    if original.shape != enhanced.shape:
        raise ValueError("Difference images must have matching dimensions and channels.")
    difference = np.abs(
        grayscale(original).astype(np.float32) - grayscale(enhanced).astype(np.float32)
    )
    peak = float(difference.max())
    if peak == 0:
        return np.zeros(difference.shape, dtype=np.uint8)
    return np.clip(np.rint(difference * (255 / peak)), 0, 255).astype(np.uint8)
