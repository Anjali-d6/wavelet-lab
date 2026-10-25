"""Reusable Streamlit components for the Wavelet Lab dashboard."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd
import streamlit as st

from src.classical_enhancement import enhance_clahe, enhance_gamma
from src.export_utils import encode_image, metrics_csv, safe_stem
from src.image_io import ImageValidationError, decode_image
from src.metrics import image_statistics
from src.visualization import difference_preview
from src.wavelet_enhancement import enhance_wavelet, maximum_level

PAGES = [
    "Home Dashboard",
    "Enhance Image",
    "Compare Methods",
    "Quality Analytics",
    "Batch Processing",
    "Methodology & Help",
]
METHODS = ("Wavelet", "CLAHE", "Gamma Correction")
WAVELETS = ("haar", "db2", "db3", "sym2", "sym3")
NAVY = "#17324d"
TEAL = "#188c8c"


def apply_theme(dark: bool) -> None:
    """Apply shared layout, type, color, and responsive styling."""
    background = "#101c2a" if dark else "#f3f7fa"
    surface = "#172738" if dark else "#ffffff"
    sidebar = "#142436" if dark else "#eaf1f5"
    text = "#e8f0f5" if dark else "#183247"
    muted = "#afc0cc" if dark else "#607789"
    border = "#31475a" if dark else "#dce6ec"
    st.markdown(
        f"""
        <style>
        :root {{
          --wl-bg: {background}; --wl-surface: {surface}; --wl-sidebar: {sidebar};
          --wl-text: {text}; --wl-muted: {muted}; --wl-border: {border};
          --wl-teal: {TEAL}; --wl-navy: {NAVY};
        }}
        .stApp {{ background: var(--wl-bg); color: var(--wl-text); }}
        [data-testid="stHeader"] {{ background: var(--wl-bg); }}
        [data-testid="stSidebar"] {{ background: var(--wl-sidebar); border-right: 1px solid var(--wl-border); }}
        [data-testid="stSidebar"] [role="radiogroup"] label {{ border-radius: 8px; padding: .25rem .4rem; }}
        [data-testid="stMain"], [data-testid="stMainBlockContainer"] {{ color: var(--wl-text); }}
        [data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--wl-surface); }}
        [data-testid="stMarkdownContainer"] p, [data-testid="stWidgetLabel"] p,
        [data-testid="stCaptionContainer"] {{ color: var(--wl-text); }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color: var(--wl-muted); }}
        [data-baseweb="select"] > div, [data-baseweb="input"] > div,
        [data-testid="stFileUploaderDropzone"] {{ background: var(--wl-surface); color: var(--wl-text); }}
        [data-testid="stMetric"] {{ background: var(--wl-surface); border: 1px solid var(--wl-border); border-radius: 12px; padding: .85rem 1rem; }}
        [data-testid="stVerticalBlockBorderWrapper"] {{ border-color: var(--wl-border); border-radius: 14px; }}
        .wl-brand {{ color: var(--wl-text); font-size: 1.1rem; font-weight: 750; letter-spacing: -.02em; }}
        .wl-kicker {{ color: var(--wl-teal); font-size: .75rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }}
        .wl-title {{ color: var(--wl-text); font-size: clamp(1.7rem, 4vw, 2.5rem); font-weight: 760; letter-spacing: -.04em; line-height: 1.12; margin: .35rem 0 .55rem; }}
        .wl-subtitle {{ color: var(--wl-muted); font-size: 1rem; line-height: 1.6; max-width: 850px; }}
        .wl-panel-title {{ color: var(--wl-text); font-size: 1.04rem; font-weight: 700; margin-bottom: .2rem; }}
        .wl-disclaimer {{ border-left: 3px solid #d49339; border-radius: 5px; padding: .8rem 1rem; background: var(--wl-surface); color: var(--wl-muted); font-size: .9rem; }}
        .wl-step {{ border-top: 2px solid var(--wl-teal); background: var(--wl-surface); border-radius: 0 0 10px 10px; padding: .9rem 1rem; min-height: 85px; }}
        div.stButton > button[kind="primary"] {{ border-radius: 9px; font-weight: 650; min-height: 2.7rem; }}
        @media (max-width: 760px) {{
          [data-testid="stHorizontalBlock"] {{ gap: .65rem; }}
          .wl-title {{ font-size: 1.75rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def heading(title: str, description: str) -> None:
    st.markdown('<div class="wl-kicker">Wavelet Lab · Image research workspace</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="wl-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="wl-subtitle">{description}</div>', unsafe_allow_html=True)
    st.write("")


def navigate_to(page: str) -> None:
    st.session_state["navigation"] = page


def upload_image(key: str = "shared_image_upload") -> tuple[np.ndarray | None, str | None]:
    upload = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key=key,
        help="PNG, JPEG, and TIFF are supported. Uploads are validated and processed in memory.",
    )
    if upload is not None:
        data = upload.getvalue()
        fingerprint = f"{upload.name}:{hashlib.sha256(data).hexdigest()}"
        if st.session_state.get("upload_fingerprint") != fingerprint:
            try:
                image = decode_image(data, upload.name)
                st.session_state.decoded_image = image
                st.session_state.upload_fingerprint = fingerprint
                st.session_state.upload_filename = upload.name
                st.session_state.upload_size = len(data)
                for name in ("single_result", "comparison_results", "analytics_result"):
                    st.session_state.pop(name, None)
            except ImageValidationError as exc:
                st.error(str(exc))
                return None, None
    image = st.session_state.get("decoded_image")
    filename = st.session_state.get("upload_filename")
    if image is not None and filename:
        channels = "Grayscale" if image.ndim == 2 else f"RGB · {image.shape[2]} channels"
        size = st.session_state.get("upload_size", 0) / (1024 * 1024)
        st.caption(
            f"**{filename}** · {image.shape[1]} × {image.shape[0]} px · "
            f"{channels} · {size:.2f} MB"
        )
        st.button(
            "Clear image and derived results",
            key=f"{key}_clear",
            on_click=clear_cached_image,
            help="Discard the in-memory source image and its single-image, comparison, and analytics results.",
        )
    return image, filename


def wavelet_controls(prefix: str, image: np.ndarray) -> dict[str, object]:
    """Render configurable wavelet settings inside a caller's form."""
    wavelet = st.selectbox(
        "Wavelet family",
        WAVELETS,
        key=f"{prefix}_wavelet",
        help="Haar, Daubechies, and Symlets have different spatial/frequency behavior.",
    )
    max_level = min(maximum_level(image.shape[:2], wavelet), 5)
    level = st.slider(
        "Decomposition level",
        min_value=1,
        max_value=max_level,
        value=1,
        key=f"{prefix}_level",
        help="Higher levels represent coarser image structures; the maximum depends on image size and wavelet.",
    )
    strength = st.slider(
        "Enhancement strength",
        0.0,
        1.0,
        0.5,
        0.05,
        key=f"{prefix}_strength",
        help="Controls the approximation-band contrast adjustment.",
    )
    detail_scale = st.slider(
        "Detail coefficient scale",
        0.5,
        2.0,
        1.25,
        0.05,
        key=f"{prefix}_detail_scale",
        help="Scales horizontal, vertical, and diagonal detail coefficients.",
    )
    denoise = st.checkbox(
        "Soft-threshold wavelet details",
        value=False,
        key=f"{prefix}_denoise",
        help="Noise reduction can also remove fine structures.",
    )
    denoise_strength = st.slider(
        "Denoising strength",
        0.0,
        1.0,
        0.5,
        0.05,
        key=f"{prefix}_denoise_strength",
        disabled=not denoise,
        help="Sets soft-threshold strength based on diagonal-detail noise estimation.",
    )
    return {
        "wavelet_name": wavelet,
        "level": level,
        "strength": strength,
        "detail_scale": detail_scale,
        "denoise": denoise,
        "denoise_strength": denoise_strength,
    }


def method_controls(
    method: str, prefix: str, image: np.ndarray
) -> dict[str, object]:
    if method == "Wavelet":
        return wavelet_controls(prefix, image)
    if method == "CLAHE":
        return {
            "clip_limit": st.slider(
                "CLAHE clip limit", 0.001, 0.1, 0.01, 0.001,
                key=f"{prefix}_clahe", help="Limits local histogram amplification.",
            )
        }
    return {
        "gamma": st.slider(
            "Gamma correction", 0.5, 2.5, 1.2, 0.05,
            key=f"{prefix}_gamma", help="Values above one brighten image luminance.",
        )
    }


def process_image(
    image: np.ndarray, method: str, parameters: dict[str, object]
) -> np.ndarray:
    if method == "Wavelet":
        return enhance_wavelet(image, **parameters)
    if method == "CLAHE":
        return enhance_clahe(image, **parameters)
    if method == "Gamma Correction":
        return enhance_gamma(image, **parameters)
    raise ValueError(f"Unsupported enhancement method: {method}")


def timed_process(
    image: np.ndarray, method: str, parameters: dict[str, object]
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    result = process_image(image, method, parameters)
    return result, (time.perf_counter() - started) * 1000


def metric_row(name: str, image: np.ndarray, elapsed_ms: float) -> dict[str, object]:
    stats = image_statistics(image)
    return {
        "method": name,
        "mean_intensity": stats["mean_intensity"],
        "contrast_std": stats["intensity_std"],
        "processing_time_ms": round(elapsed_ms, 2),
        "dimensions": f"{stats['width']} × {stats['height']}",
    }


def show_metric_cards(
    original: np.ndarray,
    enhanced: np.ndarray,
    elapsed_ms: float,
) -> None:
    before = image_statistics(original)
    after = image_statistics(enhanced)
    columns = st.columns(5)
    columns[0].metric("Original mean", f"{before['mean_intensity']:.2f}", help="Grayscale/luminance average on a 0–255 scale.")
    columns[1].metric("Enhanced mean", f"{after['mean_intensity']:.2f}", help="Grayscale/luminance average on a 0–255 scale.")
    columns[2].metric("Original contrast", f"{before['intensity_std']:.2f}", help="Grayscale standard deviation; this is descriptive, not diagnostic.")
    columns[3].metric("Enhanced contrast", f"{after['intensity_std']:.2f}", help="Grayscale standard deviation; more contrast is not necessarily better.")
    columns[4].metric("Processing time", f"{elapsed_ms:.2f} ms", help="Measured processing duration for this run.")
    st.caption(f"Image dimensions: {before['width']} × {before['height']} px. Statistics use grayscale intensity/luminance.")


def difference_png(original: np.ndarray, enhanced: np.ndarray) -> bytes:
    """Encode a display-scaled pixel difference for an explicit visualization export."""
    return encode_image(difference_preview(original, enhanced), "PNG")


def show_image_workspace(
    original: np.ndarray,
    enhanced: np.ndarray,
    *,
    method: str,
    elapsed_ms: float,
    filename: str,
    parameters: dict[str, object],
) -> None:
    st.markdown('<div class="wl-panel-title">Image comparison</div>', unsafe_allow_html=True)
    st.caption("Both previews use the same width scaling. Select Fullscreen on either image for closer inspection.")
    view = st.radio(
        "Comparison view",
        ["Side by side", "Original", "Enhanced", "Difference"],
        horizontal=True,
        key="comparison_view",
        label_visibility="collapsed",
    )
    if view == "Side by side":
        left, right = st.columns(2, gap="medium")
        with left:
            st.image(original, caption="Original", width="stretch")
        with right:
            st.image(enhanced, caption=f"{method} result", width="stretch")
    elif view == "Original":
        st.image(original, caption="Original image", width="stretch")
    elif view == "Enhanced":
        st.image(enhanced, caption=f"{method} result", width="stretch")
    else:
        st.image(
            difference_preview(original, enhanced),
            caption="Absolute difference preview — display-scaled; not a medical finding",
            width="stretch",
        )

    st.warning("Enhancement may introduce artifacts or obscure subtle features. Contrast changes do not establish diagnostic benefit.")
    show_metric_cards(original, enhanced, elapsed_ms)
    rows = [
        metric_row("Original", original, 0),
        metric_row(method, enhanced, elapsed_ms),
    ]
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {"mean_intensity": "{:.2f}", "contrast_std": "{:.2f}", "processing_time_ms": "{:.2f}"}
        ),
        hide_index=True,
        width="stretch",
    )
    stem = safe_stem(filename)
    st.markdown('<div class="wl-panel-title">Export results</div>', unsafe_allow_html=True)
    first, second, third = st.columns(3)
    with first:
        st.download_button(
            "Enhanced PNG",
            encode_image(enhanced, "PNG"),
            f"{stem}_enhanced.png",
            "image/png",
            width="stretch",
        )
    with second:
        st.download_button(
            "Enhanced JPEG",
            encode_image(enhanced, "JPEG"),
            f"{stem}_enhanced.jpg",
            "image/jpeg",
            width="stretch",
        )
    with third:
        st.download_button(
            "Difference PNG",
            difference_png(original, enhanced),
            f"{stem}_difference_preview.png",
            "image/png",
            width="stretch",
            help="This is an absolute pixel difference normalized for display.",
        )
    st.download_button(
        "Download measurements (CSV)",
        metrics_csv(rows),
        f"{stem}_metrics.csv",
        "text/csv",
    )
    st.caption(f"Parameters used: {parameters}")


def render_upload_and_settings(
    prefix: str, method_selector: bool = True
) -> tuple[np.ndarray | None, str | None, str, dict[str, object], bool]:
    image, filename = upload_image()
    if image is None:
        st.info("Upload an image to configure and run enhancement.")
        return None, filename, "Wavelet", {}, False
    method = (
        st.selectbox("Enhancement method", METHODS, key=f"{prefix}_method")
        if method_selector
        else "Wavelet"
    )
    with st.form(f"{prefix}_enhancement_form"):
        st.markdown('<div class="wl-panel-title">Processing settings</div>', unsafe_allow_html=True)
        parameters = method_controls(method, prefix, image)
        submitted = st.form_submit_button(
            "Run enhancement",
            type="primary",
            width="stretch",
            help="Processing runs only when you submit these settings.",
        )
    st.button(
        "Reset settings to defaults",
        key=f"{prefix}_reset_settings",
        on_click=reset_settings,
        args=(prefix,),
        help="Restore the default method and parameter values.",
    )
    return image, filename, method, parameters, submitted


def reset_settings(prefix: str) -> None:
    for suffix in (
        "method", "wavelet", "level", "strength", "detail_scale",
        "denoise", "denoise_strength", "clahe", "gamma",
    ):
        st.session_state.pop(f"{prefix}_{suffix}", None)


def clear_cached_image() -> None:
    for key in (
        "shared_image_upload",
        "decoded_image",
        "upload_fingerprint",
        "upload_filename",
        "upload_size",
        "single_result",
        "comparison_results",
        "analytics_result",
    ):
        st.session_state.pop(key, None)

