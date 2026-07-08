# G0DM0D3 — Neural Service Mesh

تطبيق دردشة Streamlit (Python) يتيح محادثة نماذج ذكاء اصطناعي متعددة عبر OpenRouter، مع شخصيات/أنماط مخصصة (GODMODE، Hall of Fame combos)، وكلاء متخصصين، تحليل لغة عربية، سجل ذاكرة محادثات، وإخفاء صياغة (Parseltongue) وتحويلات مخرجات (STM) وسباق نماذج متوازي (ULTRAPLINIAN).

## Run & Operate

- Workflow "Streamlit App": `streamlit run app.py --server.port 5000 --server.headless true`
- Streamlit Cloud entry point: `streamlit_app.py` (wrapper that execs `app.py` — Streamlit Cloud expects this filename by convention; do not duplicate app logic there)
- Required secret/session input: OpenRouter API key (entered by the user in the sidebar at runtime, not stored as an env secret)
- Python deps: see `requirements.txt` (streamlit, requests, numpy, pypdf)
- Local conversation history: SQLite at `memory/conversations.db`

## Stack

- Python 3.12, Streamlit
- `ai/` package: feature modules (godmode personas/AutoTune/STM, agent_categories, arabic_nlp, parseltongue, ultraplinian, llm_fallback, nsm_agent_core)
- OpenRouter as the LLM backend (multi-model access through one API)

## Where things live

- `app.py` — the single canonical Streamlit entry point; all UI tabs (chat, agents, Arabic NLP, memory log, ULTRAPLINIAN) live here
- `streamlit_app.py` — thin Streamlit Cloud wrapper around `app.py`, kept in sync automatically since it executes `app.py`'s source directly
- `ai/godmode.py` — GODMODE system prompt, Hall of Fame prompt combos, AutoTune (temperature/top_p heuristics), STM (Session Transform Modules) post-processing
- `ai/parseltongue.py` — input obfuscation techniques (leetspeak, unicode homoglyphs, zero-width, mixed-case, phonetic)
- `ai/ultraplinian.py` — parallel multi-model racing engine (5 tiers × 10 models, up to 51 total) with compound scoring (raw quality + Borda rank + Jaccard cluster-vote)
- `.migration-backup/` — the original legacy project tree (a prior Next.js/TypeScript version of this same product, plus old Streamlit/Python prototypes). Kept **read-only for reference by explicit user request — do not delete or modify.**
- `artifacts/api-server/`, `artifacts/mockup-sandbox/`, `pnpm-workspace.yaml`, root `package.json`, `lib/`, `scripts/` — Node/pnpm scaffold auto-created by Replit's importer after it mistakenly detected this as a Vercel/Next.js project (because of the `.migration-backup/` Next.js tree). **Not used by the real app.** Left in place pending explicit user confirmation to remove.

## Architecture decisions

- `app.py` is the single source of truth for the Streamlit UI; the Next.js version in `.migration-backup/` was ported feature-by-feature into new `ai/*.py` modules rather than rewritten from scratch, to preserve original logic/behavior.
- Streamlit Cloud requires a `streamlit_app.py` entry file; rather than duplicating `app.py`'s logic, `streamlit_app.py` compiles and `exec`s `app.py`'s source with the correct `__file__`/`__name__`, so there is exactly one place to edit app logic.
- ULTRAPLINIAN's compound score = 0.45×raw quality + 0.30×Borda rank score + 0.25×Jaccard n-gram cluster-vote agreement, computed only from non-streaming full responses (needed for scoring), fetched in parallel via a capped `ThreadPoolExecutor` (max 20 workers) to bound concurrent OpenRouter calls/cost.

## Product

- Free-form multi-persona chat with file/image attachments and vision-model support
- Specialized category agents, Arabic NLP analysis tab, persistent conversation history browser
- Parseltongue: optional prompt obfuscation toggle in the sidebar for triggering-word rewriting
- STM: optional post-processing filters applied to assistant replies (e.g. hedge reduction, direct mode, casual tone)
- ULTRAPLINIAN tab: pick a model tier and count (2–10 by default, up to 51 across tiers), race them on one question in parallel, and see a ranked, scored comparison with an auto-selected winner

## User preferences

- Never delete files — especially `.migration-backup/` (legacy Next.js + old Streamlit code kept for reference).
- Be mindful of API/free-tier usage: avoid unnecessary or repeated heavy operations (e.g. don't auto-run ULTRAPLINIAN races or make live LLM calls without an explicit user action).
- Push completed work to GitHub (`origin` → `aliahmed369000000-ai/Neural-Service-Mesh`) after finishing verified changes.

## Gotchas

- This repo was auto-classified as a Vercel project once because `.migration-backup/` contains a full Next.js app; ignore that scaffold (`artifacts/`, `pnpm-workspace.yaml`, root `package.json`) — the real app is Streamlit/Python at the repo root.
- Always restart the "Streamlit App" workflow after editing `app.py` or any `ai/*.py` module — Streamlit's own hot-reload can miss changes to imported modules.
- `ULTRAPLINIAN` and live chat both require the user to paste an OpenRouter key into the sidebar each session; there is no server-side default key configured.
- When rendering message history, image attachments have their raw bytes stripped before being persisted (memory-saving) — always guard `img.get("raw_bytes")` before calling `st.image`, never assume it exists on replayed history.
