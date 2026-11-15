# Deployment

The whole application — FastAPI API **and** the dashboard UI — runs on Vercel as
a single Python Function. Local development and production therefore behave
identically: same entrypoint, same paths, same-origin, no CORS to configure.

```
Vercel project
└── FastAPI app (dashboard.app:app)   ← one Function, every request
    ├── /                → dashboard/static/index.html
    ├── /static/*        → dashboard/static/*   (app.js, style.css, config.js, favicon.svg)
    ├── /figures/*       → results/figures/*    (publication figures)
    └── /api/*           → JSON APIs + live PyTorch inference
```

Vercel loads the app from `[tool.vercel] entrypoint = "dashboard.app:app"` in
`pyproject.toml` and installs dependencies from that file, using the committed
`uv.lock`. The lock pins **`torch==2.15.0+cpu`**; the default PyPI wheel would
drag in ~4 GB of CUDA libraries.

## Required Vercel project settings

| Setting | Value | Why |
| --- | --- | --- |
| Framework Preset | **FastAPI** | Runs `dashboard.app:app` as the site's single Function. |
| Build Command | *(empty — preset default)* | The Python preset installs dependencies itself. |
| Output Directory | *(empty — preset default)* | Nothing static is published; FastAPI serves the UI. |
| Root Directory | repository root | `pyproject.toml` lives there. |
| **Settings → Functions → Large Functions** | **enabled** | Mandatory, see below. |

### Large Functions is mandatory

A cold install of this app is ~1.2 GB (torch CPU alone is ~750 MB). Vercel's
standard Python Function bundle limit is **500 MB**, so the build fails on size
without Large Functions. Large Functions (public beta, Fluid compute) raises the
Python bundle limit to **5 GB**; enable it in Settings → Functions before
deploying. The bundle also contains the repo's own files (`data/`, `models/`,
`results/`), which the API reads at runtime — a few tens of MB.

## Environment variables

None are required for the Vercel deployment. The API and UI share an origin.

| Variable | When it is needed | Purpose |
| --- | --- | --- |
| `ALLOWED_ORIGINS` | Only for the split deployment below, or if you call the API from another origin | Comma-separated exact origins allowed to call the API. Unset → loopback origins only, which is enough for local tools. |
| `VITE_API_BASE_URL` | Only for the split deployment below | Backend base URL baked into `public/config.js` by `scripts/build-frontend.mjs`. Not used when Vercel runs the FastAPI app. |

`.env.example` documents both. Never commit real values; `.env` is gitignored.

## Local development

```bash
pip install -r requirements.txt   # or: uv sync --frozen
python serve.py                   # http://127.0.0.1:8000
```

`dashboard/static/config.js` defaults to `apiBaseUrl: ""` (same origin), so the
UI calls whichever host served it.

## Verify a deployment

```bash
curl -s https://<your-app>.vercel.app/api/health | python -m json.tool
# → {"status": "ok", "model_checkpoint_found": true, "cors_mode": "local-dev", ...}

curl -s -o /dev/null -w "%{http_code}\n" https://<your-app>.vercel.app/          # 200 (UI)
curl -s -o /dev/null -w "%{http_code}\n" https://<your-app>.vercel.app/static/app.js
curl -s https://<your-app>.vercel.app/api/buildings | head -c 200                 # 4 buildings
curl -s https://<your-app>.vercel.app/api/figures-list | head -c 200              # 8 figures
curl -s https://<your-app>.vercel.app/api/stream/tick?building_id=building_A      # live torch inference
```

Then load the UI: KPI cards must show **94.5%** and round **15** (the values the
API returns). If they show 94.7% / round 10, the page is rendering the static
placeholders from `index.html` and the API is not being reached.

## Cold starts

The Function imports torch and loads `models/federated_global.pt` on start, so
the first request after an idle period can take 10–20 s. `maxDuration` is set to
60 s in `vercel.json`. Live streaming polls every 2.5 s while the stream toggle
is on, which keeps the instance warm.

## Alternative: split deployment (static frontend + separate Python host)

Use this only if a Vercel bundle of ~1.2 GB is unacceptable. The dashboard UI is
plain static files, so it can be published on Vercel while the API runs on any
host with room for ML wheels:

1. Vercel: Build Command `node scripts/build-frontend.mjs`, Output Directory
   `public`, and set `VITE_API_BASE_URL=https://<backend-host>` (no trailing
   slash). `scripts/build-frontend.mjs` copies `dashboard/static` into `public/`
   and writes `public/config.js` from that variable.
2. Backend host: start `uv run uvicorn dashboard.app:app --host 0.0.0.0 --port $PORT`
   after `pip install uv && uv sync --frozen`, and set
   `ALLOWED_ORIGINS=https://<your-app>.vercel.app` (exact origin, comma-separated
   for several).
3. `vercel.json` must then drop the `functions` entry and add
   `"buildCommand"`, `"outputDirectory"` and the `/static/*` rewrite, because the
   FastAPI app is no longer serving the UI.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Build fails with a bundle size error | Large Functions is not enabled (1.2 GB > 500 MB). |
| `ModuleNotFoundError: No module named 'fastapi'` | Dependencies were resolved from a `pyproject.toml` without `[project] dependencies`, or the FastAPI preset was not selected. |
| Build downloads ~4 GB | The CUDA torch wheel was selected — install from `uv.lock`, or keep the `--extra-index-url` line in `requirements.txt`. |
| UI loads but charts/KPIs are placeholders (94.7%, round 10) while `/api/*` 404s | The app is not the deployed Function: the project is still building the static-only frontend. Set Framework Preset to FastAPI and clear Build Command / Output Directory. |
| First request takes ~20 s | Cold start: torch import plus model load. |
| CORS error in the browser | Split deployment with `ALLOWED_ORIGINS` not exactly matching the frontend origin. |
