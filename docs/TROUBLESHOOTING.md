# Troubleshooting

- **Python or package installation fails:** Use Python 3.13 and run
  `python -m pip install -r requirements.txt` in a new virtual environment.
- **Streamlit command is missing:** Run `python -m streamlit run app.py`.
- **Upload rejected:** Use a valid PNG, JPG/JPEG, or TIFF file no larger than
  10 MB and 20 million pixels. The filename extension must match its contents.
- **Wavelet level error:** Choose a lower decomposition level or a shorter
  wavelet for the image dimensions.
- **Unexpected artifacts:** Reduce enhancement strength/detail scale or disable
  denoising. Enhancement may amplify noise or remove subtle features.
- **Deployment build fails:** Check that the host uses Python 3.13, installs
  `requirements.txt`, and exposes the port assigned in `PORT`; inspect build
  logs for the first dependency or startup error.
- **CI fails:** Open the GitHub Actions job log. Tests create synthetic data and
  need no secrets or external datasets.
