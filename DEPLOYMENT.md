# Deployment

The dashboard frontend is a static site on Vercel and the FastAPI + PyTorch API
runs on a separate Python host. This is what `main` deploys today and it must
keep working; the "everything on Vercel" variant lives on a separate branch so it
can be tried without touching the live site.

```
Vercel (static, branch main)                 Python host (FastAPI + PyTorch)
public/index.html, app.js, config.js  ────▶  uvicorn dashboard.app:app
VITE_API_BASE_URL ─────────────────────────▶  ALLOWED_ORIGINS (CORS allowlist)
```

## 1. Frontend on Vercel (live)

`vercel.json` pins this configuration, so Vercel ignores framework detection:

| Setting | Value |
| --- | --- |
| Framework Preset | Other / none (`"framework": null`) |
| Build Command | `node scripts/build-frontend.mjs` |
| Output Directory | `public` |
| Rewrite | `/static/*` → `/*` (the HTML/CSS reference assets through `/static/`) |

`scripts/build-frontend.mjs` copies `dashboard/static/` into `public/` and writes
`public/config.js` from `VITE_API_BASE_URL`. It fails the build on a malformed
URL and writes `apiBaseUrl: ""` when the variable is absent.

Without a backend the UI shell still renders — but the KPIs, charts and figures
are the static placeholders from `index.html` (94.7%, round 10) and `/api/*`
returns 404 from Vercel. That is the current state; the page is not broken, it is
just not connected.

## 2. Backend on a Python host (needed to make the live site functional)

Any host that runs a long-lived Python process, exposes `$PORT` and allows a
~1.2 GB dependency install works. Render is documented here because its Python
runtime takes a build and start command directly.

Create a **Web Service** from this repo:

| Setting | Value |
| --- | --- |
| Runtime | Python 3 |
| Build command | `pip install uv && uv sync --frozen` |
| Start command | `uv run uvicorn dashboard.app:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/api/health` |
| Environment | `ALLOWED_ORIGINS=https://<your-app>.vercel.app` |

`uv sync --frozen` installs from `uv.lock`, which pins `torch==2.15.0+cpu`. The
default PyPI wheel pulls ~4 GB of CUDA libraries; `requirements.txt` points pip at
the same CPU index as a fallback, but the lock is the reproducible path.

Memory: importing torch and loading `models/federated_global.pt` needs roughly
300–400 MB RSS. 512 MB is the practical minimum, 2 GB is comfortable. Free tiers
that sleep make the first request slow (~30 s).

Then set `VITE_API_BASE_URL=https://<backend-host>` (no trailing slash) in Vercel
→ Settings → Environment Variables and **redeploy**: the value is baked into
`config.js` at build time, so a restart is not enough.

## 3. Optional: everything on Vercel (branch `vercel-fastapi`)

Vercel can run the whole FastAPI app as one Function — UI, `/static`, `/figures`
and `/api` all on one origin, so no CORS and no `VITE_API_BASE_URL`.

It is kept on the **`vercel-fastapi` branch**, never on `main`, because:

- it requires **Settings → Functions → Large Functions** (public beta, 5 GB
  bundle cap). A cold install is ~1.2 GB (torch CPU alone ~750 MB) and the
  standard Python bundle limit is 500 MB, so without that toggle the build fails;
- switching the live project's Framework Preset to FastAPI would replace the
  static deployment, which is exactly the risk this layout avoids.

Try it without touching production:

1. In Vercel, create a **new project** from the same repo and select the
   `vercel-fastapi` branch. It gets its own URL; the live project still builds
   `main`.
2. In that new project enable Large Functions, then deploy.
3. Check `https://<new-project>.vercel.app/api/health` and the UI (KPIs should
   read 94.5% / round 15, not the 94.7% / 10 placeholders).
4. If it fails or is too slow, delete the project. Production was never touched.

Do not promote that variant into the existing project unless you accept
replacing the static deployment. If you do switch the existing project, the
reversal is: Framework Preset back to **Other**, Build Command
`node scripts/build-frontend.mjs`, Output Directory `public`, then redeploy
`main` — all values come from `main`'s `vercel.json`.

## Environment variables

| Variable | Where | Required | Purpose |
| --- | --- | --- | --- |
| `VITE_API_BASE_URL` | Vercel, build time | Yes for the split deployment | Backend base URL, no trailing slash. Baked into `public/config.js`. |
| `ALLOWED_ORIGINS` | Backend host | Yes in production | Comma-separated **exact** frontend origins allowed to call the API. Unset → loopback origins only. |
| `PORT` | Injected by the backend host | Auto | Port uvicorn binds to. |

No deployment URL or secret is committed. `.env.example` documents both
variables; copy it to `.env` for local overrides (`.env` is gitignored).

## CORS

- Production: only the exact origins in `ALLOWED_ORIGINS` are allowed. No
  wildcard; credentials are disabled because the API is unauthenticated.
- Unset: loopback origins (`http://localhost:<port>`, `http://127.0.0.1:<port>`)
  are allowed, which only helps local tooling.
- Vercel preview deployments have per-branch URLs — add each one you use,
  comma-separated. Wildcards are deliberately not supported.

A CORS failure shows up as a console error while the network tab shows the
request and `curl` works fine.

## Local development

Same entrypoint as production, same-origin, no configuration:

```bash
pip install -r requirements.txt   # or: uv sync --frozen
python serve.py                   # http://127.0.0.1:8000
```

`dashboard/static/config.js` defaults to `apiBaseUrl: ""`, so the UI calls the
host that served it.

## Verify a deployment

```bash
# Backend
curl -s "$BACKEND_URL/api/health" | python -m json.tool
# → {"status": "ok", "model_checkpoint_found": true, ...}

curl -s "$BACKEND_URL/api/buildings" | head -c 200; echo
curl -s "$BACKEND_URL/api/figures-list" | head -c 200; echo

# CORS allow / deny
curl -s -D- -o /dev/null -H "Origin: $FRONTEND_URL" "$BACKEND_URL/api/summary" | grep -i access-control-allow-origin
curl -s -D- -o /dev/null -H "Origin: https://not-allowed.example" "$BACKEND_URL/api/summary" | grep -i access-control-allow-origin

# Frontend was built against the right backend
curl -s "$FRONTEND_URL/config.js"
```

In the UI: KPI **94.5%** and round **15** mean the API is answering; **94.7%** and
round **10** are the static placeholders, i.e. the frontend is not reaching the
API.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| UI shows placeholders, `/api/*` 404s on the Vercel URL | No backend connected yet: `VITE_API_BASE_URL` unset, or not redeployed after setting it. |
| `ModuleNotFoundError: No module named 'fastapi'` on the backend | Dependencies resolved from a `pyproject.toml` without `[project] dependencies`. Install with `uv sync --frozen`. |
| Backend build downloads ~4 GB | The CUDA torch wheel was selected; use the lock, or keep the `--extra-index-url` line in `requirements.txt`. |
| Build killed / OOM at runtime | Instance below ~512 MB RAM while importing torch. |
| CORS error in the browser, `curl` works | `ALLOWED_ORIGINS` does not exactly match the frontend origin (scheme, host and port all count). |
| Figures 404 | Backend host has no `results/figures/`; commit them or run `python run.py`. |
| Full-stack-on-Vercel build fails on bundle size | Large Functions not enabled (~1.2 GB vs 500 MB limit). |
