---
name: Anthropic model names 2026
description: Correct Anthropic Claude model IDs sourced from Claude.ai system prompt (That.md). The old codebase used "claude-sonnet-5" which is fictional.
---

## Rule
Always use model IDs from the `ANTHROPIC_MODELS` dict in `ai/llm_fallback.py`, not hardcoded strings.

The correct model IDs (sourced from Claude.ai system prompt, 2026):
- `claude-sonnet-4-6`          — Sonnet 4, default (balance of quality/speed)
- `claude-opus-4-8`            — Opus 4, most capable
- `claude-haiku-4-5-20251001`  — Haiku 4, fastest/lightest
- `claude-sonnet-4-20250514`   — Sonnet 4 stable, for production artifacts

**Why:** The old code used `claude-sonnet-5` which is not a valid Anthropic model ID and would return API errors. That.md (a Claude.ai system prompt export) contains the authoritative model names.

**How to apply:** When updating Anthropic model names, cross-reference That.md or the official Anthropic API docs. Never invent model version strings.
