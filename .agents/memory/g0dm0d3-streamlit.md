---
name: G0DM0D3 Streamlit port
description: Arabic Streamlit AI mesh app — entrypoint, workflow, and root-file duplication history.
---

- Canonical running entrypoint is `streamlit_app.py` (13+ tabs: CKG/quran/QA/chat/agents/creative/training/memory/health/API/system/G0DM0D3/ULTRAPLINIAN). Workflow: `streamlit run streamlit_app.py --server.port 5000 --server.headless true`.
- `app.py` at repo root is a separate, older, simpler single-file variant (5 tabs) that predates the streamlit_app.py rewrite — NOT a stale duplicate to delete blindly, but also not the active entrypoint. It was used as a *feature source* to backport pieces (VISION_MODELS/IMAGE_MIMES/_extract_file, ULTRAPLINIAN tab UI) into streamlit_app.py.
- `.migration-backup/` is an unrelated archived import scaffold (old Next.js/Vercel misdetection artifacts) — irrelevant to this app; do not source features from it.
- OpenRouter integration in streamlit_app.py lives in sidebar (`_or_api_key`/`_or_model` session state) + `_or_stream()`/`_or_chat()` helpers; before a 2026-07-10 change it was wired only into render_godmode(), NOT render_chat() — verify actual wiring by grepping `_or_api_key` usage rather than trusting prior claims about what "already exists".
- **Why this matters:** a user claimed Harm Badge + 👍/👎 feedback already existed in streamlit_app.py "so don't duplicate them" — grep found zero matches. Always verify such claims against the current file before skipping planned work; don't silently comply with an inaccurate premise.
- Streamlit file-uploader widgets retain their last selection across reruns if given a fixed `key`; must bump a version counter in session state and interpolate it into the key after consuming/clearing files, or old files reappear.
