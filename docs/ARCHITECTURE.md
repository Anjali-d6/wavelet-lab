# Architecture

The Streamlit entry point in `app.py` delegates page routing and page rendering
to `src/dashboard.py`; reusable controls, image previews, and metric components
live in `src/dashboard_components.py`. Interactive charts are composed in
`src/visualization.py` using Plotly. The dashboard uses small processing modules under `src/`.
`image_io.py` decodes and validates uploads in memory; `wavelet_enhancement.py`
implements DWT enhancement; `classical_enhancement.py` contains CLAHE and gamma;
`metrics.py` calculates image measurements; `batch_processing.py` isolates each
batch file; and `export_utils.py` creates in-memory downloads.

Uploads and results are held in the active Streamlit session only. No database,
external API, permanent upload directory, or globally shared image cache is used.
The app limits each upload to 10 MB and 20 million pixels, and each batch to 20
files and 40 million total pixels. The processing pipeline is CPU-only.
