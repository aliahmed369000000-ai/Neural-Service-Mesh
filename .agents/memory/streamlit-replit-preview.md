---
name: Streamlit app in Replit preview
description: What makes a Streamlit app render correctly in Replit's proxied iframe preview.
---

Replit's preview pane loads the app inside a proxied iframe from a different origin than
localhost. A Streamlit app can start fine (workflow shows "running", no errors) yet the
preview stays blank until:

- `.streamlit/config.toml` sets `enableCORS = false` and `enableXsrfProtection = false`.
- The server binds to `0.0.0.0` on the port the workflow/preview expects (port 5000 is the
  Replit default for the main preview).

**Why:** Streamlit's default CORS/XSRF protections reject cross-origin requests, and the
proxy iframe is cross-origin from the app's own domain — so without these settings the app
silently fails to render even though the process is healthy.

**How to apply:** When setting up or debugging a Streamlit project on Replit, check
`.streamlit/config.toml` for these settings and the run command's `--server.port` /
`--server.address` flags first, before assuming a code-level bug.
