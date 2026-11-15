# Deployment

Split deployment: the dashboard frontend is a static site on **Vercel**, and the
FastAPI + PyTorch backend runs on a **Python host with room for ML wheels**
(e.g. Render). Nothing about the ML, federated learning or API code changes —
only *where* each piece runs.

```
        ┌────────────────────────────┐         ┌──────────────────────────────┐
        │  Vercel (static)           │  HTTPS  │  Python host (ASGI)          │
        │  dashboard/static → public │ ──────▶ │  uvicorn dashboard.app:app   │
        │  index.html / style.css    │  CORS   │  torch + pandas + sklearn    │
        │  app.js / config.js        │         │  /api/*  /figures/*          │
        └────────────────────────────┘         └──────────────────────────────┘
             VITE_API_BASE_URL ────────────────────────▶ ALLOWED_ORIGINS
```

Why split: `torch` is ~750 MB installed (CPU wheel) and Vercel caps Python
function bundles at 500 MB, so the backend cannot be a Vercel Function. A normal
Python host has no such limit.

## Environment variables

| Variable | Where it is set | Required | Purpose |
| --- | --- | --- | --- |
| `VITE_API_BASE_URL` | Vercel → Project → Settings → Environment Variables | Production only | Base URL of the backend, **no trailing slash** (e.g. `https://your-backend.example.com`). Injected into `public/config.js` at build time. Leave unset for local dev. |
| `ALLOWED_ORIGINS` | Backend host → Environment | Production only | Comma-separated **exact** frontend origins allowed to call the API (e.g. `https://your-frontend.vercel.app`). Leave unset locally. |
| `PORT` | Injected by the backend host | Auto | Port uvicorn must bind to. Never hardcode it. |

No deployment URL or secret belongs in the repository. `.env.example` documents
both variables; copy it to `.env` for local overrides (`.env` is gitignored).

## 1. Backend (FastAPI + PyTorch)

Any host that runs a long-lived Python process, exposes `$PORT`, and allows a
~1.2 GB dependency install works. Render is documented here because its Python
runtime takes a build and a start command directly (Render's own FastAPI guide
uses the same `$PORT` convention).

Create a **Web Service** from this GitHub repo:

| Setting | Value |
| --- | --- |
| Runtime | Python 3 |
| Build command | `pip install uv && uv sync --frozen` |
| Start command | `uv run uvicorn dashboard.app:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/api/health` |
| Environment | `ALLOWED_ORIGINS=<your Vercel origin>` |

`uv sync --frozen` installs from the committed `uv.lock`, which pins
`torch==2.15.0+cpu` — the CPU wheel. Do not switch to plain
`pip install -r requirements.txt` on a host with a small disk quota unless you
have checked the resolved torch build; `requirements.txt` points pip at the same
CPU index, but `uv.lock` is the reproducible path.

Memory: importing torch and loading the autoencoder needs roughly 300–400 MB
RSS. A 512 MB instance is the minimum and can be tight; use a 2 GB instance for
comfort. The free tier also sleeps after inactivity, so the first request after
a pause can take ~30 s.

`serve.py` is the local entrypoint and is **not** used by the host — the host
runs uvicorn directly against the ASGI app.

## 2. Frontend (Vercel)

In Vercel, import the same repository and set:

| Setting | Value |
| --- | --- |
| Framework preset | Other (none) — `vercel.json` sets `framework: null` |
| Build command | `node scripts/build-frontend.mjs` (already in `vercel.json`) |
| Output directory | `public` (already in `vercel.json`) |
| Environment variable | `VITE_API_BASE_URL=https://<your-backend-host>` |

`scripts/build-frontend.mjs` copies `dashboard/static/` into `public/` and writes
`public/config.js` from `VITE_API_BASE_URL`. It fails the build on a malformed
URL instead of shipping a broken frontend, and writes `apiBaseUrl: ""` when the
variable is absent.

`vercel.json` rewrites `/static/*` to `/*` because `index.html` and `style.css`
reference assets through `/static/…` (that is how FastAPI serves them locally).

## 3. CORS

The API sets CORS from `ALLOWED_ORIGINS`:

- Production: only the exact origins listed are allowed. No wildcard, and
  credentials are disabled — the API is unauthenticated, so browsers never need
  to send cookies cross-origin.
- Unset: loopback origins (`http://localhost:<port>`, `http://127.0.0.1:<port>`)
  are allowed, which only helps local tooling. Requests from the FastAPI-served
  UI are same-origin and need no CORS at all.
- Vercel **preview** deployments have per-branch URLs. Add each one you actually
  use to `ALLOWED_ORIGINS`, comma-separated. Wildcards are deliberately not
  supported; list real origins.

Symptom of a missing origin: the browser console reports a CORS error while the
network tab shows the request, and `curl` against the API works fine.

## 4. Local development

Nothing extra is required — same-origin, no environment variables:

```bash
pip install -r requirements.txt   # or: uv sync --frozen
python serve.py                   # http://127.0.0.1:8000
```

The committed `dashboard/static/config.js` defaults to `apiBaseUrl: ""`, so the
UI calls the API it was served from. To develop the frontend against a separate
backend, run the backend with `ALLOWED_ORIGINS=http://localhost:3000` and build
or serve the static files with `VITE_API_BASE_URL=http://127.0.0.1:8000`.

## 5. Verify a deployment

Run against your deployed hosts (values from your own environment, not from the
repo):

```bash
# 1. Backend is up and the model checkpoint was found
curl -s "$BACKEND_URL/api/health" | python -m json.tool
# → {"status": "ok", "model_checkpoint_found": true, "cors_mode": "allowlist", ...}

# 2. API data endpoints answer
curl -s "$BACKEND_URL/api/summary" | head -c 200; echo
curl -s "$BACKEND_URL/api/buildings" | head -c 200; echo

# 3. CORS allows the frontend origin (must print the origin)
curl -s -D- -o /dev/null -H "Origin: $FRONTEND_URL" "$BACKEND_URL/api/summary" \
  | grep -i access-control-allow-origin

# 4. CORS rejects an unknown origin (must print nothing)
curl -s -D- -o /dev/null -H "Origin: https://not-allowed.example" "$BACKEND_URL/api/summary" \
  | grep -i access-control-allow-origin

# 5. Frontend built with the right backend URL
curl -s "$FRONTEND_URL/config.js"
# → window.__APP_CONFIG__ = { "apiBaseUrl": "https://your-backend..." };
```

Then open the frontend and confirm KPI cards, buildings, telemetry and figures
load — figures are served from the backend host (their URLs are derived from the
request, so they follow `VITE_API_BASE_URL`).

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `ModuleNotFoundError: No module named 'fastapi'` on the backend | Dependencies were resolved from `pyproject.toml` — it must keep its `[project] dependencies`, or install with `uv sync --frozen`. |
| Backend build downloads ~4 GB | The CUDA torch wheel was selected. Use `uv sync --frozen`, or keep the `--extra-index-url` line in `requirements.txt`. |
| Build killed / OOM at runtime | Instance below ~512 MB RAM while importing torch; use a larger instance. |
| CORS error in the browser, `curl` works | `ALLOWED_ORIGINS` on the backend does not exactly match the frontend origin (scheme, host and port all count). |
| Frontend calls `http://localhost:8000` in production | `VITE_API_BASE_URL` was not set when the Vercel build ran; redeploy after adding it. |
| Figures 404 on the deployed frontend | The backend host is missing `results/figures/` — commit them or generate them with `python run.py`. |
