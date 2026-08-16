"""
ai/agent_function_calling.py
============================
🆕 Function Calling رسمي — ترقية _call_api من "JSON يدوي هش" إلى
أداة tool-calling معتمدة من المزودين.

المشكلة السابقة: الوكيل كان يطلب "رد JSON فقط" ويحلل النص يدويًا
(regex/quoting fixes) — هش وقابل للكسر. هذه الوحدة تستخدم صيغة
OpenAI tools الرسمية حيث يدعمها المزود:

| المزود   | الصيغة الرسمية              | ملاحظة |
|-----------|------------------------------|---------|
| OpenRouter/OpenAI | tools + tool_choice | native |
| Groq      | tools (Llama/Mixtral/Gemma) | native |
| Gemini    | functionDeclarations      | native |
| Cloudflare| لا يدعم — fallback JSON     | fallback |

الاستخدام:
    from ai.agent_function_calling import execute_with_tools
    res = execute_with_tools(
        tools=TOOL_REGISTRY_SPEC_LIST,
        system="أنت مساعد...",
        user_input="افحص المشروع",
    )
    # res = {"tool_calls": [...], "text": "...", "done": True}
    # tool_call = {"name": "shell", "arguments": {"cmd": "ls"}}
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_MAX_FN_ROUNDS = 12


def _groq_tool_call(system: str, user_input: str, tools: List[Dict[str, Any]],
                    history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Groq — يدعم tool calling الرسمي لنماذج متعددة."""
    import urllib.request

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    models = ["llama-3.1-8b-instant", "llama3-groq-70b-8192-tool-use-preview",
              "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    errors: List[str] = []
    for model in models:
        url = "https://api.groq.com/openai/v1/chat/completions"
        messages: List[Dict] = [{"role": "system", "content": system}] + [
            {"role": "user", "content": user_input},
        ] + history
        body = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 4096,
            "temperature": 0.15,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            choice = data["choices"][0]["message"]
            if choice.get("tool_calls"):
                calls = []
                for tc in choice["tool_calls"]:
                    try:
                        calls.append({"name": tc["function"]["name"],
                                      "arguments": json.loads(tc["function"]["arguments"])})
                    except Exception:
                        continue
                if calls:
                    return {"tool_calls": calls, "text": "", "done": False,
                            "model": model, "stop_reason": choice.get("finish_reason")}
            return {"tool_calls": [], "text": (choice.get("content") or "").strip(),
                    "done": True, "model": model,
                    "stop_reason": choice.get("finish_reason")}
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                errors.append(f"Groq/{model} محجوب ({e.code})")
                break
            errors.append(f"Groq/{model}: HTTP {e.code}")
        except Exception as e:
            errors.append(f"Groq/{model}: {e}")
    return {"tool_calls": [], "text": "", "done": True,
            "error": " | ".join(errors[:3])}


def _gemini_tool_call(system: str, user_input: str, tools: List[Dict[str, Any]],
                      history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Gemini — functionDeclarations رسمية."""
    import urllib.request

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return {"tool_calls": [], "text": "", "done": True, "error": "لا GOOGLE_API_KEY"}
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "gemini-2.5-flash:generateContent")
    fn_decls = [t["function"] for t in tools if t.get("type") == "function"]
    contents: List[Dict] = [{"role": "user", "parts": [{"text": user_input}]}]
    for h in history:
        if h["role"] == "assistant":
            contents.append({"role": "model", "parts": [{"text": h["content"]}]})
        elif h["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": h["content"]}]})
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        "tools": [{"functionDeclarations": fn_decls}] if fn_decls else [],
        "generationConfig": {"maxOutputTokens": 8192, "temperature": 0.15},
    }
    try:
        req = urllib.request.Request(
            f"{url}?key={api_key}", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        parts = data["candidates"][0]["content"]["parts"]
        calls, text = [], ""
        for p in parts:
            if "functionCall" in p:
                calls.append({"name": p["functionCall"]["name"],
                              "arguments": p["functionCall"].get("args", {})})
            elif "text" in p:
                text += p["text"]
        return {"tool_calls": calls, "text": text.strip(),
                "done": not calls,
                "stop_reason": data["candidates"][0].get("finishReason")}
    except Exception as e:
        return {"tool_calls": [], "text": "", "done": True, "error": str(e)[:300]}


def _openai_tool_call(system: str, user_input: str, tools: List[Dict[str, Any]],
                      history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """OpenRouter/OpenAI — الصيغة الرسمية tools."""
    import urllib.request

    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        return {"tool_calls": [], "text": "", "done": True, "error": "لا OPENROUTER_API_KEY"}
    url = "https://openrouter.ai/api/v1/chat/completions"
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_input}] + history
    body = {"model": "openai/gpt-4o-mini",
            "messages": messages, "tools": tools,
            "tool_choice": "auto", "max_tokens": 4096, "temperature": 0.15}
    try:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read())
        choice = data["choices"][0]["message"]
        if choice.get("tool_calls"):
            calls = []
            for tc in choice["tool_calls"]:
                try:
                    calls.append({"name": tc["function"]["name"],
                                  "arguments": json.loads(tc["function"]["arguments"])})
                except Exception:
                    continue
            return {"tool_calls": calls, "text": "", "done": not calls,
                    "stop_reason": choice.get("finish_reason")}
        return {"tool_calls": [], "text": (choice.get("content") or "").strip(),
                "done": True, "stop_reason": choice.get("finish_reason")}
    except Exception as e:
        return {"tool_calls": [], "text": "", "done": True, "error": str(e)[:300]}


def execute_with_tools(
    tools: List[Dict[str, Any]],
    system: str,
    user_input: str,
    *,
    execute_fn: Optional[Any] = None,
    max_rounds: int = _MAX_FN_ROUNDS,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """حلقة function calling رسمية: يستدعي الأدوات حتى ينتهي النموذج.

    execute_fn(name, args) -> str — تنفّذ الأداة وتعيد ملاحظة نصية.
    إن لم تُمرَّ، تُجمع tool_calls فقط ويعاد history مكتمل (للتدريس).
    """
    history = list(history or [])
    if not tools:
        return {"tool_calls": [], "text": "", "done": True,
                "error": "لا أدوات معرّفة"}
    # ترتيب المحاولات: Groq أولاً (الأرخص والأسرع)، ثم Gemini، ثم OpenRouter
    providers = [_groq_tool_call, _gemini_tool_call, _openai_tool_call]
    round_n = 0
    text_out, tool_calls_out = "", []
    while round_n < max_rounds:
        round_n += 1
        res: Optional[Dict[str, Any]] = None
        last_errors = []
        for prov in providers:
            try:
                res = prov(system, user_input, tools, history)
            except Exception as e:
                last_errors.append(str(e)[:100])
                continue
            if res and not res.get("error"):
                break
            if res and res.get("error"):
                last_errors.append(res["error"][:100])
        if res is None or res.get("error"):
            return {"tool_calls": tool_calls_out, "text": "", "done": True,
                    "error": " | ".join(last_errors[:3])}
        if res.get("tool_calls"):
            tool_calls_out.extend(res["tool_calls"])
            if execute_fn is None:
                return {"tool_calls": tool_calls_out, "text": "",
                        "done": False, "history": history + [{"role": "assistant",
                                "content": json.dumps(res.get("text", ""), ensure_ascii=False)}]}
            # تنفيذ الأدوات وإضافة ملاحظاتها
            history.append({"role": "assistant",
                            "content": json.dumps(res.get("text", ""), ensure_ascii=False)})
            for call in res["tool_calls"]:
                try:
                    obs = execute_fn(call["name"], call["arguments"])
                except Exception as e:
                    obs = f"استثناء أثناء {call['name']}: {e}"
                history.append({"role": "tool", "name": call["name"],
                                "content": str(obs)[:6000]})
            continue
        # النموذج أنهى بأداة واحدة؟ لا — لا توجد tool_calls ⇒ رد نهائي
        text_out = res.get("text", "")
        return {"tool_calls": tool_calls_out, "text": text_out, "done": True,
                "model": res.get("model", ""), "stop_reason": res.get("stop_reason")}
    return {"tool_calls": tool_calls_out, "text": text_out, "done": False,
            "error": f"استُنفدت {max_rounds} جولات"}
