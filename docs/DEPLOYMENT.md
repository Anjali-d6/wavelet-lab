# Deployment

## Recommended: Streamlit Community Cloud

### Upload this project to GitHub

The project files are in the `project` folder. That folder does not currently
have its own Git repository; do not upload or publish its parent folder.

1. Sign in at [github.com](https://github.com) and choose **New repository**.
2. Give the repository a name such as `wavelet-lab`. Choose **Private** if you
   do not intend to publish the source. Leave **Add a README**, **.gitignore**,
   and **license** unchecked so the new repository is empty.
3. Open the new repository, choose **Add file → Upload files**, and upload the
   *contents* of the project's `project` folder, preserving their directories.
   Include `app.py`, `requirements.txt`, `src`, `tests`, `docs`, `.streamlit`,
   and `.github`. Do not upload the parent folder, `.venv`, Python caches,
   environment files, or private images. `.gitignore` excludes local virtual
   environments, secrets, outputs, and sample images.
4. Enter a commit message such as `Add Wavelet Lab app` and select **Commit
   changes** on the `main` branch. If the browser upload cannot preserve
   directories or dot-prefixed folders, use GitHub Desktop to publish the
   contents of the project folder instead.

### Create the Streamlit app

1. Sign in at [share.streamlit.io](https://share.streamlit.io) using the
   GitHub account that can access the repository.
2. Select **Create app → Yup, I have an app**.
3. Choose the repository, `main` branch, and `app.py` as the main file path.
4. Open **Advanced settings**, select **Python 3.13** (the version tested
   locally), and leave **Secrets** empty. This app needs no API key, secret, or
   environment variable.
5. Select **Save**, then **Deploy**. Wait for the build to finish and check the
   app URL. If startup fails, inspect the deployment logs before changing
   dependencies.
6. Test the deployed app with a synthetic or otherwise non-sensitive image.
   Commits pushed to the selected branch trigger redeployment.

All third-party Python packages used by the app are declared in the root
`requirements.txt`; no `packages.txt` or external system package is needed.
Images are processed in memory by the app, without calls to an image-processing
API or other external image service. When hosted on Community Cloud, a browser
upload is sent to the Streamlit-hosted app server for processing. Do not use
real patient images, identifiable health information, or other sensitive
medical data on a public demo; hosting is not represented as a compliant
clinical or protected-health-information environment.

No credentials or environment variables are required for local development or
this app's default deployment. Hosting resource limits vary; processing large
images may be slow.

## Docker / Render

Build and run locally:

```powershell
docker build -t wavelet-lab .
docker run --rm -p 8501:8501 wavelet-lab
```

For Render, create a Docker web service from the repository and use the included
`render.yaml` as a starting point. Render assigns a port via `PORT`; the image
binds to `0.0.0.0` and reads this variable. Review the hosting plan's current
memory, CPU, and pricing limits before deployment. Deployment requires your own
hosting account authorization and is not performed by this project.

Never commit secrets, local environment files, or private input images. The
container runs as a non-root user. GitHub Actions runs tests without secrets.
