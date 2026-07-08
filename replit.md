# Neural Service Mesh (NSM)

نظام ذكاء اصطناعي عربي متكامل مبني على Streamlit، يجمع بين وكيل ذكي وشبكة نماذج لغوية ورسم معرفي وذاكرة محادثة.

## Run & Operate

- `streamlit run streamlit_app.py --server.port 5000` — run the Streamlit app

## Stack

- Python 3.11 + Streamlit
- Multi-provider LLM fallback (Anthropic, Cloudflare, Gemini, Groq, OpenAI, OpenRouter, Together)
- Cognitive Knowledge Graph (CKG) with Arabic NLP
- NSM Chat with conversation memory

## Where things live

- `streamlit_app.py` — main Streamlit UI
- `ai/` — AI modules (LLM fallback, agents, knowledge graph, etc.)
- `knowledge/` — knowledge store, Quran QA engine, episodic memory
- `nsm_chat_plus.py` — generative chat layer
- `nsm_memory.py` — conversation memory
- `.streamlit/config.toml` — server config (port 5000, headless)

## User preferences

- Arabic responses preferred (اللغة العربية الفصحى)
- Maintain existing file structure

## Gotchas

- Must inject Streamlit secrets → os.environ before any module imports (done in streamlit_app.py)
- `st.rerun()` must be used instead of `experimental_rerun`

## Required Secrets

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude (primary) |
| `CF_API_TOKEN` | Cloudflare Workers AI |
| `CF_ACCOUNT_ID` | Cloudflare account |
| `GOOGLE_API_KEY` | Gemini |
| `OPENROUTER_API_KEY` | OpenRouter |
| `GROQ_API_KEY` | Groq |
| `OPENAI_API_KEY` | OpenAI |
| `TOGETHER_API_KEY` | Together.xyz |
