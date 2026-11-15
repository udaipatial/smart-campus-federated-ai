// Runtime configuration for the dashboard frontend.
//
// apiBaseUrl: "" means "same origin" — used for local development, where the
// FastAPI app serves this UI itself, and for any deployment that proxies /api.
//
// For a split deployment (static frontend on Vercel, FastAPI elsewhere) the
// Vercel build overwrites this file via scripts/build-frontend.mjs using the
// VITE_API_BASE_URL environment variable. Never hardcode a deployment URL here.
window.__APP_CONFIG__ = window.__APP_CONFIG__ || { apiBaseUrl: '' };
