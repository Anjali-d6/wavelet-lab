"""Streamlit presentation layer for Wavelet Lab."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.batch_processing import process_batch
from src.export_utils import comparison_zip, encode_image, metrics_csv, safe_stem
from src.image_io import ImageValidationError, decode_image
from src.metrics import image_statistics, reference_metrics
from src.visualization import comparison_figure, histogram_figure

from src.dashboard_components import (
    METHODS,
    PAGES,
    apply_theme,
    heading,
    metric_row,
    method_controls,
    navigate_to,
    render_upload_and_settings,
    show_image_workspace,
    timed_process,
    upload_image,
)


def render_enhance() -> None:
    heading("Enhance image", "Configure a method, run it on your image, and inspect measurements without changing the source upload.")
    focus = st.toggle("Expand image workspace", key="focus_image_workspace")

    if focus:
        with st.expander("Image input and processing settings", expanded=True):
            image, filename, method, parameters, submitted = render_upload_and_settings("single")
        result_column = st.container()
    else:
        settings, result_column = st.columns([0.34, 0.66], gap="large")
        with settings:
            st.markdown('<div class="wl-panel-title">Image and settings</div>', unsafe_allow_html=True)
            image, filename, method, parameters, submitted = render_upload_and_settings("single")

    if image is None or not filename:
        with result_column:
            st.markdown('<div class="wl-panel-title">Enhancement workspace</div>', unsafe_allow_html=True)
            st.info("Your original and processed image previews will appear here.")
        return

    current_key = (
        st.session_state.get("upload_fingerprint"),
        method,
        tuple(sorted(parameters.items())),
    )
    if submitted:
        try:
            with st.spinner("Processing image…"):
                result, elapsed = timed_process(image, method, parameters)
            st.session_state.single_result = {
                "key": current_key,
                "image": result,
                "elapsed_ms": elapsed,
                "method": method,
                "parameters": parameters.copy(),
            }
            st.success(f"{method} enhancement completed in {elapsed:.2f} ms.")
        except (ValueError, RuntimeError, OSError) as exc:
            st.session_state.pop("single_result", None)
            st.error(f"Enhancement failed: {exc}")

    with result_column:
        cached = st.session_state.get("single_result")
        if cached and cached["key"] == current_key:
            show_image_workspace(
                image,
                cached["image"],
                method=cached["method"],
                elapsed_ms=cached["elapsed_ms"],
                filename=filename,
                parameters=cached["parameters"],
            )
        else:
            st.info("Choose settings and select **Run enhancement** to generate a result.")


def render_home() -> None:
    heading(
        "Medical image enhancement, made inspectable",
        "A compact research workspace for exploring wavelet-based enhancement, classical methods, and image statistics.",
    )
    st.button(
        "Open enhancement workspace",
        type="primary",
        on_click=navigate_to,
        args=("Enhance Image",),
    )
    st.write("")
    steps = st.columns(5)
    for column, (number, label) in zip(
        steps,
        [("01", "Upload"), ("02", "Configure"), ("03", "Enhance"), ("04", "Analyze"), ("05", "Export")],
    ):
        with column:
            st.markdown(
                f'<div class="wl-step"><span class="wl-kicker">{number}</span><br><strong>{label}</strong></div>',
                unsafe_allow_html=True,
            )
    st.write("")
    st.markdown('<div class="wl-panel-title">Research tools</div>', unsafe_allow_html=True)
    cards = st.columns(4, gap="medium")
    features = [
        ("Wavelet enhancement", "DWT approximation and detail-coefficient processing.", "Enhance Image"),
        ("Method comparison", "Compare supported methods on the same image.", "Compare Methods"),
        ("Quality analytics", "Inspect intensity, contrast, and processing time.", "Quality Analytics"),
        ("Batch workflow", "Process multiple images with per-file status.", "Batch Processing"),
    ]
    for column, (title, description, page) in zip(cards, features):
        with column:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(description)
                st.button("Open", key=f"home_{page}", on_click=navigate_to, args=(page,))
    st.markdown('<div class="wl-panel-title">How the DWT workflow works</div>', unsafe_allow_html=True)
    st.write(
        "The 2D Discrete Wavelet Transform separates luminance into a coarse approximation "
        "and horizontal, vertical, and diagonal details. Wavelet Lab applies controlled "
        "coefficient operations, reconstructs the image, and reports measurements from the actual result."
    )
    state = st.session_state
    completed = sum(
        key in state for key in ("single_result", "comparison_results", "analytics_result", "batch_results")
    )
    st.caption(f"Current session: {completed} result workspace(s) with saved results.")
    st.markdown(
        '<div class="wl-disclaimer"><strong>Educational use only.</strong> This application is not '
        "a clinically validated diagnostic tool. Enhancement can create artifacts or obscure "
        "subtle details; a contrast change is not evidence of clinical improvement.</div>",
        unsafe_allow_html=True,
    )


def _comparison_key(
    fingerprint: str, methods: list[str], settings: dict[str, dict[str, object]]
) -> tuple[object, ...]:
    return (
        fingerprint,
        tuple(methods),
        tuple((method, tuple(sorted(settings[method].items()))) for method in methods),
    )


def render_compare() -> None:
    heading("Compare methods", "Run selected enhancement methods against the same validated image and compare actual outputs.")
    left, main = st.columns([0.34, 0.66], gap="large")
    with left:
        image, filename = upload_image()
        if image is None:
            st.info("Upload an image to configure a comparison.")
            return
        st.markdown('<div class="wl-panel-title">Comparison setup</div>', unsafe_allow_html=True)
        with st.form("comparison_form"):
            methods = st.multiselect(
                "Methods to compare",
                METHODS,
                default=list(METHODS),
                help="Each selected method processes the same uploaded image.",
            )
            settings: dict[str, dict[str, object]] = {}
            for method in methods:
                with st.expander(f"{method} parameters", expanded=(method == "Wavelet")):
                    settings[method] = method_controls(
                        method, f"compare_{method.lower().replace(' ', '_')}", image
                    )
            submitted = st.form_submit_button(
                "Run comparison", type="primary", width="stretch"
            )

    current_key = _comparison_key(
        st.session_state.get("upload_fingerprint", ""), methods, settings
    )
    if submitted:
        if not methods:
            st.warning("Select at least one enhancement method.")
        else:
            results: list[dict[str, Any]] = []
            progress = st.progress(0, text="Preparing comparison")
            try:
                for index, method in enumerate(methods):
                    with st.spinner(f"Running {method}…"):
                        result, elapsed = timed_process(image, method, settings[method])
                    results.append(
                        {"method": method, "image": result, "elapsed_ms": elapsed,
                         "parameters": settings[method].copy()}
                    )
                    progress.progress((index + 1) / len(methods), text=f"Completed {method}")
                st.session_state.comparison_results = {
                    "key": current_key,
                    "results": results,
                    "source": image.copy(),
                }
                st.success(f"Comparison complete: {len(results)} method(s) processed.")
            except (ValueError, RuntimeError, OSError) as exc:
                st.session_state.pop("comparison_results", None)
                st.error(f"Comparison failed: {exc}")

    with main:
        saved = st.session_state.get("comparison_results")
        if not saved or saved["key"] != current_key:
            st.info("Select methods and settings, then choose **Run comparison**.")
            return
        outputs = saved["results"]
        source = saved["source"]
        display_rows = [metric_row("Original", source, 0)] + [
            metric_row(item["method"], item["image"], item["elapsed_ms"])
            for item in outputs
        ]
        comparison_items = [
            {"method": "Original", "image": source, "elapsed_ms": 0},
            *outputs,
        ]
        for start in range(0, len(comparison_items), 3):
            row_items = comparison_items[start : start + 3]
            cards = st.columns(len(row_items), gap="medium")
            for column, item in zip(cards, row_items):
                with column:
                    with st.container(border=True):
                        st.markdown(f"**{item['method']}**")
                        st.image(item["image"], width="stretch")
                        st.caption(
                            "Source image" if item["method"] == "Original"
                            else f"{item['elapsed_ms']:.2f} ms · {item['parameters']}"
                        )
        st.markdown('<div class="wl-panel-title">Detailed result</div>', unsafe_allow_html=True)
        selected = st.selectbox("Inspect result", [item["method"] for item in outputs])
        detail = next(item for item in outputs if item["method"] == selected)
        st.image(detail["image"], caption=f"{selected} · Fullscreen available", width="stretch")
        st.download_button(
            f"Download {selected} PNG",
            encode_image(detail["image"]),
            f"{safe_stem(filename or 'image')}_{safe_stem(selected)}.png",
            "image/png",
        )
        st.dataframe(
            pd.DataFrame(display_rows).style.format(
                {"mean_intensity": "{:.2f}", "contrast_std": "{:.2f}", "processing_time_ms": "{:.2f}"}
            ),
            hide_index=True,
            width="stretch",
        )
        dark = st.session_state.get("dark_theme", False)
        st.plotly_chart(comparison_figure(display_rows, dark=dark), width="stretch", key="comparison_plot")
        zip_files = {
            f"{safe_stem(filename or 'image')}_{safe_stem(item['method'])}.png":
            encode_image(item["image"])
            for item in outputs
        }
        st.download_button(
            "Download comparison ZIP",
            comparison_zip(zip_files, metrics=display_rows, parameters=settings),
            f"{safe_stem(filename or 'image')}_comparison.zip",
            "application/zip",
        )
        st.download_button(
            "Download comparison CSV",
            metrics_csv(display_rows),
            f"{safe_stem(filename or 'image')}_comparison.csv",
            "text/csv",
        )
        st.caption("These measurements describe the output; they do not establish that one method is universally superior.")


def render_analytics() -> None:
    heading("Quality analytics", "Explore intensity distributions and descriptive measurements. Reference-based scores are optional and require an aligned reference.")
    left, main = st.columns([0.34, 0.66], gap="large")
    with left:
        image, filename = upload_image()
        if image is None:
            st.info("Upload an image to begin quality analysis.")
            return
        reference_upload = st.file_uploader(
            "Optional aligned reference",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="analytics_reference",
            help="PSNR/SSIM will be calculated only if dimensions and channels match the result.",
        )
        with st.form("analytics_form"):
            method = st.selectbox("Method to analyze", METHODS, key="analytics_method")
            parameters = method_controls(method, "analytics", image)
            submitted = st.form_submit_button(
                "Run analytics", type="primary", width="stretch"
            )

    reference_key = None
    if reference_upload is not None:
        reference_data = reference_upload.getvalue()
        reference_key = (
            reference_upload.name,
            hashlib.sha256(reference_data).hexdigest(),
        )
    result_key = (
        st.session_state.get("upload_fingerprint"),
        method,
        tuple(sorted(parameters.items())),
        reference_key,
    )
    if submitted:
        try:
            with st.spinner("Calculating measurements…"):
                result, elapsed = timed_process(image, method, parameters)
            rows = [
                metric_row("Original", image, 0),
                metric_row(method, result, elapsed),
            ]
            if reference_upload is not None:
                reference = decode_image(reference_data, reference_upload.name)
                rows[1].update(reference_metrics(result, reference))
            st.session_state.analytics_result = {
                "key": result_key, "result": result, "rows": rows,
                "elapsed_ms": elapsed, "parameters": parameters.copy(),
            }
            st.success("Quality analysis completed. Measurements are descriptive only.")
        except (ImageValidationError, ValueError, RuntimeError, OSError) as exc:
            st.session_state.pop("analytics_result", None)
            st.error(f"Analysis failed: {exc}")

    with main:
        saved = st.session_state.get("analytics_result")
        if not saved or saved["key"] != result_key:
            st.info("Configure an analysis and select **Run analytics** to see actual measurements.")
            return
        result = saved["result"]
        rows = saved["rows"]
        before = image_statistics(image)
        after = image_statistics(result)
        cards = st.columns(5)
        cards[0].metric("Mean intensity", f"{before['mean_intensity']:.2f}", help="Original grayscale/luminance mean.")
        cards[1].metric("Enhanced mean", f"{after['mean_intensity']:.2f}")
        cards[2].metric("Contrast · original", f"{before['intensity_std']:.2f}", help="Grayscale standard deviation.")
        cards[3].metric("Contrast · enhanced", f"{after['intensity_std']:.2f}", help="Grayscale standard deviation; descriptive only.")
        cards[4].metric("Processing time", f"{saved['elapsed_ms']:.2f} ms")
        dark = st.session_state.get("dark_theme", False)
        st.plotly_chart(
            histogram_figure([("Original", image), (method, result)], dark=dark),
            width="stretch",
            key="analytics_histogram",
        )
        st.plotly_chart(
            comparison_figure(rows, dark=dark),
            width="stretch",
            key="analytics_metrics_plot",
        )
        st.markdown('<div class="wl-panel-title">Measured values</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(rows).style.format(
                {
                    "mean_intensity": "{:.2f}",
                    "contrast_std": "{:.2f}",
                    "processing_time_ms": "{:.2f}",
                    "psnr": "{:.2f}",
                    "ssim": "{:.4f}",
                },
                na_rep="—",
            ),
            hide_index=True,
            width="stretch",
        )
        delta_contrast = after["intensity_std"] - before["intensity_std"]
        direction = "increased" if delta_contrast > 0 else "decreased" if delta_contrast < 0 else "was unchanged"
        delta_mean = after["mean_intensity"] - before["mean_intensity"]
        mean_direction = "increased" if delta_mean > 0 else "decreased" if delta_mean < 0 else "was unchanged"
        st.info(
            f"Measured grayscale standard deviation {direction} by {abs(delta_contrast):.2f}; "
            f"mean intensity {mean_direction} by {abs(delta_mean):.2f}. "
            "These are descriptive image statistics, not evidence of better diagnostic quality."
        )
        st.caption(
            "PSNR and SSIM are included only when a same-sized reference image with matching channels is supplied."
        )
        st.download_button(
            "Download metrics CSV",
            metrics_csv(rows),
            f"{safe_stem(filename or 'image')}_metrics.csv",
            "text/csv",
        )
        st.download_button(
            "Download enhanced PNG",
            encode_image(result),
            f"{safe_stem(filename or 'image')}_enhanced.png",
            "image/png",
        )


def batch_fingerprint(uploads: list[Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (upload.name, hashlib.sha256(upload.getvalue()).hexdigest())
        for upload in uploads
    )


def render_batch() -> None:
    heading("Batch processing", "Configure one method, process files independently, and review per-image outcomes and exports.")
    left, main = st.columns([0.36, 0.64], gap="large")
    with left:
        uploads = st.file_uploader(
            "Select images",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
            key="batch_upload",
            help="Maximum 20 files and 40 million cumulative pixels per batch.",
        )
        if uploads:
            upload_rows = [
                {
                    "filename": item.name,
                    "size_mb": round(len(item.getvalue()) / (1024 * 1024), 2),
                    "validation": "Validated on processing" if len(item.getvalue()) <= 10 * 1024 * 1024 else "Exceeds 10 MB limit",
                }
                for item in uploads
            ]
            st.dataframe(pd.DataFrame(upload_rows), hide_index=True, width="stretch")
        settings_image = np.zeros((256, 256), dtype=np.uint8)
        if uploads:
            try:
                settings_image = decode_image(uploads[0].getvalue(), uploads[0].name)
            except ImageValidationError:
                st.caption("Wavelet level is based on a default preview size; invalid uploads will be reported individually.")
        with st.form("batch_form"):
            method = st.selectbox("Batch method", METHODS, key="batch_method")
            parameters = method_controls(method, "batch", settings_image)
            submitted = st.form_submit_button(
                "Start processing", type="primary",
                width="stretch", disabled=not uploads or len(uploads) > 20,
            )
        if uploads and len(uploads) > 20:
            st.warning("This batch exceeds the 20-file limit.")

    current_key = (
        batch_fingerprint(uploads or []),
        method,
        tuple(sorted(parameters.items())),
    )
    if submitted and uploads:
        try:
            progress = st.progress(0, text="Starting batch")
            status = st.empty()

            def report_progress(completed: int, total: int, filename: str) -> None:
                progress.progress(completed / total, text=f"Processed {completed} of {total}")
                status.caption(f"Latest file: {filename}")

            results = process_batch(
                [(item.name, item.getvalue()) for item in uploads],
                method=method,
                parameters=parameters,
                progress_callback=report_progress,
            )
            st.session_state.batch_results = {"key": current_key, "results": results}
            successes = sum(item["success"] for item in results)
            st.success(f"Batch complete: {successes} succeeded, {len(results) - successes} failed.")
        except ValueError as exc:
            st.error(f"Batch could not be started: {exc}")

    with main:
        stored = st.session_state.get("batch_results")
        if not stored or stored["key"] != current_key:
            st.info("Select files and settings, then choose **Start processing**.")
            return
        results = stored["results"]
        successes = sum(item["success"] for item in results)
        cards = st.columns(3)
        cards[0].metric("Files processed", len(results))
        cards[1].metric("Succeeded", successes)
        cards[2].metric("Failed", len(results) - successes)
        rows: list[dict[str, object]] = []
        archive_files: dict[str, bytes] = {}
        for index, item in enumerate(results, start=1):
            row: dict[str, object] = {
                "filename": item["filename"],
                "status": "Success" if item["success"] else "Failed",
                "processing_time_ms": round(item["processing_time_ms"], 2),
            }
            if item["success"]:
                row.update(
                    {
                        "mean_intensity": item["mean_intensity"],
                        "contrast_std": item["intensity_std"],
                        "dimensions": f"{item['width']} × {item['height']}",
                    }
                )
                output = encode_image(item["image"])
                output_name = f"{safe_stem(item['filename'])}_{index}_enhanced.png"
                archive_files[output_name] = output
                with st.expander(f"Preview · {item['filename']}"):
                    st.image(item["image"], width="stretch")
                    st.download_button(
                        f"Download {item['filename']}",
                        output, output_name, "image/png", key=f"batch_download_{index}",
                    )
            else:
                row["error"] = item["error"]
            rows.append(row)
        st.dataframe(
            pd.DataFrame(rows).style.format(
                {"processing_time_ms": "{:.2f}", "mean_intensity": "{:.2f}", "contrast_std": "{:.2f}"}
            ),
            hide_index=True,
            width="stretch",
        )
        if successes:
            first, second = st.columns(2)
            with first:
                st.download_button(
                    "Download successful outputs (ZIP)",
                    comparison_zip(archive_files, metrics=rows, parameters={"method": method, **parameters}),
                    "wavelet_lab_batch.zip",
                    "application/zip",
                    width="stretch",
                )
            with second:
                st.download_button(
                    "Download batch measurements (CSV)",
                    metrics_csv(rows),
                    "wavelet_lab_batch_metrics.csv",
                    "text/csv",
                    width="stretch",
                )


def render_methodology() -> None:
    heading("Methodology & help", "A concise guide to the implemented processing workflow, controls, measurements, and limitations.")
    stages = [
        ("01 · Image input", "Validate file type and decoded data, then load supported grayscale, RGB, or RGBA images into memory."),
        ("02 · DWT decomposition", "PyWavelets separates grayscale/luminance into approximation and horizontal, vertical, and diagonal detail coefficients."),
        ("03 · Coefficient processing", "The wavelet method adjusts approximation contrast and bounded detail scaling; optional soft-thresholding denoises details."),
        ("04 · Reconstruction", "The inverse DWT reconstructs the image, restores its original dimensions, and clips to a valid 8-bit range."),
        ("05 · Comparison methods", "CLAHE and gamma correction operate on luminance for color inputs; all methods use the same decoded image."),
        ("06 · Analysis and export", "Calculate grayscale intensity mean/standard deviation and processing time; PSNR/SSIM need an aligned reference."),
    ]
    for title, explanation in stages:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(explanation)
    st.markdown('<div class="wl-panel-title">Parameter glossary</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(
            [
                {"Parameter": "Wavelet family", "Meaning": "Filter family used for DWT decomposition and reconstruction."},
                {"Parameter": "Decomposition level", "Meaning": "Number of transform scales; constrained by image dimensions."},
                {"Parameter": "Enhancement strength", "Meaning": "Controls approximation-coefficient contrast gain."},
                {"Parameter": "Detail coefficient scale", "Meaning": "Scales wavelet horizontal, vertical, and diagonal details."},
                {"Parameter": "Denoising strength", "Meaning": "Controls soft thresholding estimated from diagonal detail coefficients."},
                {"Parameter": "Contrast standard deviation", "Meaning": "Global grayscale spread; descriptive only and not a diagnostic score."},
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.markdown("**Supported formats:** PNG, JPEG/JPG, TIFF. Each upload is limited to 10 MB and 20 million pixels.")
    st.markdown("**Troubleshooting:** reduce the decomposition level for small images; lower enhancement/detail scaling if artifacts appear; disable denoising if fine detail is lost.")
    st.markdown(
        '<div class="wl-disclaimer"><strong>Educational and research use only.</strong> This tool is not '
        "clinically validated and must not be used for diagnosis or treatment. Processing can "
        "introduce artifacts or obscure subtle features.</div>",
        unsafe_allow_html=True,
    )


def run_dashboard() -> None:
    """Render the shared navigation shell and the selected dashboard page."""
    dark = st.session_state.get("dark_theme", False)
    apply_theme(dark)
    with st.sidebar:
        st.markdown('<div class="wl-brand">WAVELET LAB</div>', unsafe_allow_html=True)
        st.caption("Medical image enhancement")
        st.divider()
        page = st.radio("Workspace", PAGES, key="navigation", label_visibility="visible")
        st.divider()
        st.toggle("Dark appearance", key="dark_theme", value=dark)
        st.caption("Uploads stay in this session. Do not use unauthorized sensitive images.")

    if page == "Home Dashboard":
        render_home()
    elif page == "Enhance Image":
        render_enhance()
    elif page == "Compare Methods":
        render_compare()
    elif page == "Quality Analytics":
        render_analytics()
    elif page == "Batch Processing":
        render_batch()
    else:
        render_methodology()
