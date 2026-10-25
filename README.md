# Medical Image Enhancement Using Wavelet Transform

An educational Streamlit app to explore 2D discrete wavelet transform (DWT)
enhancement alongside gamma correction and CLAHE. It is intended for learning
and image-processing evaluation, not clinical diagnosis.

## Features

- Wavelet detail scaling and approximation-contrast adjustment; optional soft-threshold denoising.
- Grayscale and RGB inputs; luminance-based processing helps preserve color.
- PNG, JPG/JPEG, and TIFF uploads; each upload is limited to 10 MB and 20 million pixels. Batches allow up to 20 images and 40 million total pixels.
- Single-image enhancement, method comparison, histograms, quality statistics, and batch processing.
- PNG/JPEG downloads, comparison ZIPs, and CSV measurements.
- Responsive light/dark research dashboard with interactive Plotly analytics.
- Images are processed in memory by the app without calls to external image APIs or persistent image storage. A hosted app receives uploads on its server for processing; do not use patient-identifiable data on a public demo.

## Technology and structure

Python 3.13, Streamlit, NumPy, PyWavelets, Pillow, scikit-image, Pandas, Plotly, and pytest.

```text
app.py
src/                   Dashboard pages/components, charts, image I/O, enhancement, metrics, batch, exports
tests/                 Synthetic-image automated tests
docs/                  Architecture, algorithm, deployment, troubleshooting
.streamlit/config.toml Streamlit configuration
.github/workflows/     Automated test workflow
Dockerfile             Optional non-root container
render.yaml            Render Docker service configuration
```

## Install and run locally

Install Python 3.13. In PowerShell, from the project folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).
No account, API key, or environment variable is required.

## Use the application

1. Choose **Enhance Image** or **Compare Methods** from the dashboard navigation.
2. Upload a PNG, JPG/JPEG, or TIFF image; the workspace shows its name, size, dimensions, and channels.
3. Choose a supported method and adjust settings inside the form.
4. Select **Run enhancement**; controls do not start processing until the form is submitted.
5. Inspect aligned side-by-side previews, fullscreen images, or a normalized difference view.
6. Review measured statistics and export images, a difference preview, CSV, or comparison ZIP.
7. Use **Quality Analytics** for interactive Plotly histograms and measurements; supply a matching reference to calculate PSNR/SSIM.
8. Use **Batch Processing** for up to 20 images. Each invalid file is reported without stopping other files.
9. Toggle the dark appearance in the sidebar when preferred.

## Wavelet method and metrics

The 2D DWT separates an image into coarse approximation coefficients and
horizontal, vertical, and diagonal detail coefficients. The app adjusts those
coefficients and reconstructs the image using the inverse transform. Read
[docs/ALGORITHM.md](docs/ALGORITHM.md) for details and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for the module structure.

Mean grayscale intensity describes average brightness. Intensity standard
deviation is also reported as RMS contrast. PSNR and SSIM are reference-based
metrics only and require equal dimensions/channels. Brightness or contrast
changes do not demonstrate diagnostic improvement.

## Tests

```powershell
python -m compileall -q app.py src tests
python -m pytest -q
```

GitHub Actions runs these checks on pushes and pull requests.

## Docker

```powershell
docker build -t wavelet-lab .
docker run --rm -p 8501:8501 wavelet-lab
```

The container runs as a non-root user and accepts the host-assigned `PORT`.
Large images and batch operations consume CPU and memory; hosting limits vary.

## Deploy online

To deploy with Streamlit Community Cloud:

1. On GitHub, create a new **empty** repository. Do not initialize it with a
   README, `.gitignore`, or license.
2. In PowerShell, initialize and review this project folder. Keep `.gitignore`
   and the hidden `.streamlit` and `.github` folders; do not force-add ignored
   local environments, secrets, outputs, or private images.

   ```powershell
   Set-Location C:\Users\ganja\Downloads\project
   git init
   git branch -M main
   git add .
   git status --short
   ```

   Review the staged files before continuing. Then commit and push, replacing
   the URL with your GitHub username and repository name:

   ```powershell
   git commit -m "Prepare Wavelet Lab for deployment"
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   git push -u origin main
   ```

3. At [share.streamlit.io](https://share.streamlit.io), select **Create app**,
   choose the repository and `main` branch, and set the main file path to
   `app.py`.
4. In **Advanced settings**, select Python 3.13 and leave **Secrets** empty.
   Select **Deploy** and check the app URL and deployment logs.
5. Test with a synthetic or otherwise non-sensitive image. Commits to the
   selected branch trigger redeployment.

This project requires no API keys or environment variables. A hosted app
receives uploaded images on its server for processing; do not use patient or
other sensitive images. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for
detailed steps and the optional Docker/Render instructions.

## Limitations and disclaimer

Enhancement may amplify noise, clip details, or hide subtle structures. The app
is an educational project, not a clinically validated medical device or
diagnostic service. It is not represented as HIPAA, GDPR, or otherwise
regulatory-compliant. Do not use its results for diagnosis, treatment, or
clinical decisions.
