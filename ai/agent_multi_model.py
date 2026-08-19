"""
ai/agent_multi_model.py
=======================
🆕 تبديل ذكي بين مزودي LLM (Multi-Model Fallback).

عند فشل مزود (rate limit, error, timeout) ينتقل تلقائيًا إلى مزود بديل:
  Groq → OpenRouter → Gemini → fallback JSON

الاستخدام:
    from ai.agent_multi_model import MultiModelCaller
    caller = MultiModelCaller()
    response = caller.chat(system="أنت مساعد...", user="اشرح لي...")
"""
from __future__ import annotations
import json
import logging
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger("nsm.multi_model")

EVENT_MODEL_SWITCH = "model_switch"
EVENT_MODEL_FALLBACK = "model_fallback"


class MultiModelCaller:
    """استدعاء LLM مع تبديل ذكي بين المزودين."""

    # ترتيب المزودين (الأفضل أولاً)
    PROVIDERS = [
        {
            "name": "groq",
            "models": [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
            ],
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "key_env": "GROQ_API_KEY",
            "max_tokens": 4096,
        },
        {
            "name": "openrouter",
            "models": [
                "meta-llama/llama-3.1-70b-instruct",
                "meta-llama/llama-3.1-8b-instruct",
                "google/gemma-2-9b-it",
            ],
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "key_env": "OPENROUTER_API_KEY",
            "max_tokens": 4096,
        },
        {
            "name": "gemini",
            "models": [
                "gemini-1.5-flash",
                "gemini-1.5-pro",
            ],
            "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            "key_env": "GEMINI_API_KEY",
            "max_tokens": 8192,
        },
    ]

    def __init__(self, max_retries: int = 3, timeout: int = 30):
        self.max_retries = max_retries
        self.timeout = timeout
        self._switch_log: List[Dict[str, Any]] = []

    def _get_api_key(self, env_var: str) -> Optional[str]:
        key = os.getenv(env_var, "").strip()
        return key if key else None

    def _call_groq(self, provider: Dict, model: str, system: str,
                   user: str, history: List[Dict]) -> Optional[Dict[str, Any]]:
        """استدعاء Groq."""
        api_key = self._get_api_key(provider["key_env"])
        if not api_key:
            return None

        messages = [{"role": "system", "content": system}] + history + [
            {"role": "user", "content": user},
        ]

        body = {
            "model": model,
            "messages": messages,
            "max_tokens": provider["max_tokens"],
            "temperature": 0.15,
        }

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            provider["url"],
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            choices = result.get("choices", [])
            if choices:
                return {"ok": True, "content": choices[0]["message"]["content"],
                        "model": model, "provider": "groq"}
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError,
                KeyError, TimeoutError) as e:
            logger.debug(f"Groq error ({model}): {e}")

        return None

    def _call_openrouter(self, provider: Dict, model: str, system: str,
                         user: str, history: List[Dict]) -> Optional[Dict[str, Any]]:
        """استدعاء OpenRouter."""
        api_key = self._get_api_key(provider["key_env"])
        if not api_key:
            return None

        messages = [{"role": "system", "content": system}] + history + [
            {"role": "user", "content": user},
        ]

        body = {
            "model": model,
            "messages": messages,
            "max_tokens": provider["max_tokens"],
            "temperature": 0.15,
        }

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            provider["url"],
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            choices = result.get("choices", [])
            if choices:
                return {"ok": True, "content": choices[0]["message"]["content"],
                        "model": model, "provider": "openrouter"}
        except Exception as e:
            logger.debug(f"OpenRouter error ({model}): {e}")

        return None

    def _call_gemini(self, provider: Dict, model: str, system: str,
                     user: str, history: List[Dict]) -> Optional[Dict[str, Any]]:
        """استدعاء Gemini."""
        api_key = self._get_api_key(provider["key_env"])
        if not api_key:
            return None

        contents = []
        if system:
            contents.append({
                "role": "user",
                "parts": [{"text": system}],
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "فهمت. أنا جاهز."}],
            })
        for h in history:
            contents.append({
                "role": "user" if h["role"] == "user" else "model",
                "parts": [{"text": h["content"]}],
            })
        contents.append({
            "role": "user",
            "parts": [{"text": user}],
        })

        url = provider["url"].format(model=model)
        body = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": provider["max_tokens"],
                "temperature": 0.15,
            },
        }

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{url}?key={api_key}",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            candidates = result.get("candidates", [])
            if candidates:
                text = candidates[0]["content"]["parts"][0]["text"]
                return {"ok": True, "content": text,
                        "model": model, "provider": "gemini"}
        except Exception as e:
            logger.debug(f"Gemini error ({model}): {e}")

        return None

    def chat(
        self,
        system: str = "",
        user: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        emit_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """استدعاء LLM مع تبديل تلقائي بين المزودين."""
        history = history or []
        last_error = ""

        for provider in self.PROVIDERS:
            api_key = self._get_api_key(provider["key_env"])
            if not api_key:
                continue

            for model in provider["models"]:
                result = None
                for attempt in range(self.max_retries):
                    if provider["name"] == "groq":
                        result = self._call_groq(provider, model, system, user, history)
                    elif provider["name"] == "openrouter":
                        result = self._call_openrouter(provider, model, system, user, history)
                    elif provider["name"] == "gemini":
                        result = self._call_gemini(provider, model, system, user, history)

                    if result and result.get("ok"):
                        if emit_fn:
                            emit_fn(EVENT_MODEL_SWITCH, metadata={
                                "provider": result["provider"],
                                "model": result["model"],
                                "attempts": attempt + 1,
                            })
                        return result

                    last_error = f"{provider['name']}/{model} attempt {attempt + 1} failed"

                # التبديل للمزود التالي
                if emit_fn:
                    emit_fn(EVENT_MODEL_FALLBACK, metadata={
                        "from": provider["name"],
                        "reason": last_error,
                    })

        return {"ok": False, "error": last_error or "No providers available",
                "provider": "none", "model": "none"}

    def get_available_providers(self) -> List[str]:
        """جلب قائمة المزودين المتاحين (الذين لديهم API keys)."""
        available = []
        for provider in self.PROVIDERS:
            if self._get_api_key(provider["key_env"]):
                available.append(provider["name"])
        return available
