# Deployment

## Published demo

Production URL: <https://healthcaredl-resume.vercel.app>

Published on November 11, 2026. Verified without authentication: the explorer, health endpoint, model/data/privacy pages, static assets, JSON prediction, browser form submission, and mobile layout. The demo is hosted under the `healthcaredl-resume` Vercel project. The theme update was verified as deployment `dpl_3KcTQ8zzq8K39wBZqHNZFFULnkYv`; subsequent deployment history is available in the project dashboard.

Before the report and dataset-expansion updates, the original symptom model was rerun from scratch with seed 20, a 400-epoch maximum, and patience 40. It completed 152 epochs and selected epoch 112, reproducing the previous weights and learning history exactly. Held-out accuracy and macro F1 remained 1.000 on the same 61 test patterns. All 79 automated checks passed before redeployment; the public health endpoint, model page, and neural prediction were verified afterward.

The subsequent explorer update adds four one-click fictional examples, a count-specific minimum-input explanation, and separately labeled model scores and report-derived test accuracy on matched results. All 90 automated checks passed before this deployment. Desktop and mobile accessibility checks found no WCAG A/AA violations in the checked result states and no horizontal overflow. The model weights and training report are unchanged by this interface update.

After publishing, unauthenticated checks confirmed the health endpoint and JSON match/insufficient/urgent states. All four example buttons returned their expected matches in a mobile browser on the public production alias, with the percentages visible and no horizontal overflow or JavaScript errors.

The theme update adds a shared light/dark switch, device-preference detection, browser-local persistence, cross-tab synchronization, and themed controls, charts, and result states. All 96 automated checks passed. Accessibility audits of 24 states across both themes found no WCAG A/AA violations or horizontal overflow. Theme initialization uses an external script compatible with the existing content security policy and applies the preference before the stylesheet loads. The privacy page documents the single stored theme preference.

The report workspace adds local PDF/image blood-report extraction and a separate browser-based chest CT research classifier. All 144 automated checks passed before the source update, including real image/scanned-PDF OCR and browser/PyTorch numerical agreement. The imaging training stack and raw CT archive are excluded from the web function. The server still requires only Flask and NumPy; WebAssembly, model, and language assets are served from `public/static/`. Selected medical files are not uploaded. The new workflow is reviewed on `development` before merging into `main` for production.

The dataset expansion trains the CT head on 4,169 unique images using separate publisher patient folders, and adds 913 explicitly synthetic symptom-training variants. Both model bundles and report-page metrics change together. All 150 automated checks passed, and both updated models reproduced their checkpoint tensors and metrics in separate CPU runs. The raw archive and imaging training modules remain excluded from the function. Source updates stay on `development` until the user merges their PR.

Vercel installs the runtime dependencies declared in `pyproject.toml`; these must stay aligned with `requirements.txt`. The application was deployed from the local working tree through the CLI. Source updates are pushed to `development` and reviewed through a pull request into `main`; merging to `main` lets the connected Vercel project deploy the reviewed source.

## Vercel — primary target

The application follows [Vercel's Flask deployment](https://vercel.com/docs/frameworks/backend/flask): a top-level `app` in `main.py`, a Python 3.12 runtime, and assets under `public/static/` so the platform can serve them from its CDN. A custom domain is optional; Vercel supplies a deployment address and managed HTTPS.

The server reads the compact CT metadata from `artifacts/chest_ct_runtime.json`. Keep it identical to `public/static/models/chest-ct/report.json`; training exports both, and the asset checker verifies equality. Files under `public/` are served separately by the CDN and must not be assumed to exist in the function filesystem. Docker also copies the compact metadata into its runtime image.

The runtime requirements contain only Flask and NumPy. Training remains in PyTorch. Training exports every dense-layer parameter to `artifacts/inference.npz`; serving executes the same matrix operations and ReLU activations with dropout inactive. Automated tests compare its logits against the PyTorch model on all 304 unique patterns. This does not replace the model with a classical classifier.

`vercel.json` declares the Flask framework, function duration, deployment exclusions, and response headers. `.vercelignore` also keeps development environments, raw datasets, notebooks, original unsafe tables, and training-only code out of uploads. Keep the runtime artifacts and condition catalog together. No secrets or database connection are required for the public educational demo.

### Publish through Vercel

1. Connect a Vercel account through the available integration, or import the project repository in the Vercel dashboard. Repository deployment requires the new source files to be committed and pushed first.
2. Use the repository root as the root directory and the **Flask** framework preset. Leave build/install overrides empty unless your project settings already override the repository defaults.
3. Deploy a preview and verify `/health`, `/`, `/model`, `/data`, `/privacy`, and an example `POST /api/predict` before promoting it.
4. Use the Vercel-assigned HTTPS address or attach a domain later. Configure deployment protection if the audience should be private, and review platform log access, retention, firewall, and budget settings for the chosen account.

The CLI is an alternative after account authentication:

```bash
npx vercel@latest login
npx vercel@latest
# Promote only the reviewed version:
npx vercel@latest --prod
```

The documentation is not evidence that a live deployment occurred. An actual provider response and reachable deployment URL are required to report publishing as completed. No Vercel credentials are stored in this repository.

### Runtime checks

```bash
python -m pip install -r requirements.txt
python tools/check_artifacts.py
python main.py
```

Training uses `requirements-training.txt`; development and browser tests use `requirements-dev.txt`. Avoid installing the training stack inside the web function. Vercel's Python bundle limits and platform behavior can change; the small export is intended to avoid needing large-function features.

## Optional Docker / WSGI deployment

A separate deployment path is provided for a Linux host. It uses Gunicorn, a non-root user, an inference-only image, a health check, a read-only filesystem, no Linux capabilities, and temporary storage mounted only under `/tmp`.

```bash
docker compose up --build -d
curl http://127.0.0.1:8080/health
```

The application port binds to loopback on the host. Configure an HTTPS reverse proxy before exposing it publicly. Set `HEALTHDL_HTTPS=1` only once HTTPS is in place. The app does not blindly trust forwarded client headers; adapt proxy trust only to a verified topology.

Gunicorn access logs are disabled so request paths, query strings, and client addresses are not intentionally recorded by that server. Warnings/errors remain available for operations. Reverse proxies, host providers, and observability tools need their own retention and redaction settings. Resource limits and process counts should be tuned to the host.

```bash
python -m pip install -r requirements-server.txt
gunicorn --config deploy/gunicorn.conf.py main:app
```

## Release verification

Run the automated suite and `tools/check_artifacts.py`, confirm the data/model documentation matches the selected artifact bundle, and check an example prediction using non-personal data. The included CI workflow performs data, training, inference, route, and Chromium checks.

Clinical review and independent patient evaluation remain separate evidence requirements. Deploying an educational demonstration does not complete those requirements.
