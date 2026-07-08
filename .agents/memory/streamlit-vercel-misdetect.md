---
name: G0DM0D3 misdetected as Vercel project
description: Why this Streamlit repo has leftover pnpm/Node artifacts and how to fix Streamlit Cloud entrypoint issues.
---

Replit's importer classified this repo as a Vercel/Next.js project because a
legacy Next.js app was preserved under `.migration-backup/` for reference. The
real app is a Streamlit/Python app (`app.py` at repo root). This produced
unrelated scaffold — `pnpm-workspace.yaml`, root `package.json`,
`artifacts/api-server/`, `artifacts/mockup-sandbox/`, `lib/`, `scripts/` — and
two registered platform artifacts ("API Server", "Canvas") that are dead
weight for this product. Left in place pending explicit user confirmation to
remove (never delete `.migration-backup/` — user requires it kept).

**Why:** don't be misled by importer/platform auto-classification — always
verify what actually runs (check for `app.py` + `.streamlit/config.toml` vs
`package.json` + `next.config.js`) before assuming project type from scaffold.

**How to apply:** if asked to deploy/run this repo and it looks like a Next.js
app, check for `.migration-backup/` first — the Streamlit app at the root is
the real, current product.

Streamlit Cloud requires an entrypoint literally named `streamlit_app.py`. Fix
for "main module file does not exist" errors: add a thin `streamlit_app.py`
that `compile()`/`exec()`s the real entry file's source with correct
`__file__`/`__name__`, rather than duplicating app logic — keeps one source of
truth while satisfying the platform's naming convention.
