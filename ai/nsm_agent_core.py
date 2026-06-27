"""
NSM Agent Core — ai/nsm_agent_core.py
=======================================
وكيل ذكي يعمل داخل محادثة NSM بـ Groq مجاناً.
يفهم الطلبات بالعربي → يكتب الكود → يحفظ → يرفع GitHub.

الاستخدام من nsm_chat.py:
    from ai.nsm_agent_core import NSMAgent
    agent = NSMAgent()
    result = agent.run("أنشئ نظام تسجيل دخول")
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).parent.parent

# ══════════════════════════════════════════════════════════════════
# System Prompt للوكيل
# ══════════════════════════════════════════════════════════════════
_AGENT_SYSTEM = """أنت وكيل برمجي ذكي متخصص في مشروع NSM (Neural Service Mesh).
مشروع Python/Streamlit للذكاء الاصطناعي العربي على GitHub.

عند استلام طلب برمجي، رد بـ JSON فقط بهذا الشكل:
{
  "thinking": "تفكيرك في الطلب",
  "action": "create_file | edit_file | git_push | answer",
  "path": "المسار إن وجد",
  "content": "المحتوى أو الكود",
  "old": "النص القديم إن كان تعديلاً",
  "new": "النص الجديد إن كان تعديلاً",
  "message": "رسالة commit إن كان رفعاً",
  "reply": "رد للمستخدم بالعربية"
}

قواعد:
1. الكود يكون Python نظيف مكتمل مع docstring
2. المسارات نسبية من جذر المشروع (مثل ai/new_module.py)
3. إذا الطلب معلوماتي فقط → action: answer
4. دائماً أجب بـ JSON صحيح فقط بدون أي نص خارجه
5. اكتب الكود بالكامل وليس مجرد مثال"""

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# نماذج بديلة — يُجرَّب الأول، ثم الثاني عند الفشل
_GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
]
_OPENROUTER_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
]


# ══════════════════════════════════════════════════════════════════
# استدعاء API مع Fallback للنماذج
# ══════════════════════════════════════════════════════════════════
def _call_api(messages: List[Dict], api_key: str) -> str:
    """يجرب Cloudflare أولاً (مجاني ويعمل من اليمن)، ثم Gemini، ثم OpenRouter."""
    errors = []

    # 1) Cloudflare Workers AI — مجاني 10k/يوم ✅
    cf_token   = os.getenv("CF_API_TOKEN", "").strip()
    cf_account = os.getenv("CF_ACCOUNT_ID", "").strip()
    if cf_token and cf_account:
        cf_url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{cf_account}/ai/run/@cf/meta/llama-3.1-8b-instruct"
        )
        payload = json.dumps({"messages": messages, "max_tokens": 2048}).encode()
        req = urllib.request.Request(
            cf_url, data=payload,
            headers={"Authorization": f"Bearer {cf_token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read())
            text = data.get("result", {}).get("response", "").strip()
            if text:
                return text
        except Exception as e:
            errors.append(f"Cloudflare: {e}")

    # 2) Google Gemini — مجاني (قد لا يعمل من اليمن)
    gemini_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if gemini_key and gemini_key.startswith("AIzaSy"):
        gemini_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-1.5-flash:generateContent?key={gemini_key}"
        )
        gemini_parts = []
        for m in messages:
            if m["role"] == "system":
                gemini_parts.append({"role": "user", "parts": [{"text": m["content"]}]})
                gemini_parts.append({"role": "model", "parts": [{"text": "حسناً."}]})
            elif m["role"] == "user":
                gemini_parts.append({"role": "user", "parts": [{"text": m["content"]}]})
            elif m["role"] == "assistant":
                gemini_parts.append({"role": "model", "parts": [{"text": m["content"]}]})
        payload = json.dumps({"contents": gemini_parts}).encode()
        req = urllib.request.Request(
            gemini_url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            errors.append(f"Gemini: {e}")

    # 3) OpenRouter
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if or_key:
        for model in _OPENROUTER_MODELS:
            payload = json.dumps({
                "model": model, "messages": messages,
                "max_tokens": 2048, "temperature": 0.3,
            }).encode()
            req = urllib.request.Request(
                _OPENROUTER_URL, data=payload,
                headers={
                    "Authorization": f"Bearer {or_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://neural-service-mesh.streamlit.app",
                    "X-Title": "Neural Service Mesh",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=25) as resp:
                    return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
            except Exception as e:
                errors.append(f"OpenRouter/{model}: {e}")

    # 4) Groq (محجوب في اليمن لكن نحاول)
    if api_key:
        for model in _GROQ_MODELS:
            payload = json.dumps({
                "model": model, "messages": messages,
                "max_tokens": 2048, "temperature": 0.3, "stream": False,
            }).encode()
            req = urllib.request.Request(
                _GROQ_URL, data=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    errors.append(f"Groq محجوب ({e.code})")
                    break
                errors.append(f"Groq/{model}: {e.code}")
            except Exception as e:
                errors.append(f"Groq: {e}")

    raise Exception(" | ".join(errors) or "لا يوجد مزوّد — أضف CF_API_TOKEN وCF_ACCOUNT_ID")


# ══════════════════════════════════════════════════════════════════
# تنفيذ الإجراء
# ══════════════════════════════════════════════════════════════════
def _execute(action_data: Dict[str, Any]) -> str:
    action  = action_data.get("action", "answer")
    path    = action_data.get("path", "")
    content = action_data.get("content", "")
    old     = action_data.get("old", "")
    new     = action_data.get("new", "")
    message = action_data.get("message", "NSM Agent auto-commit")
    reply   = action_data.get("reply", "")
    thinking = action_data.get("thinking", "")

    result_lines = []

    if thinking:
        result_lines.append(f"🤔 {thinking}\n")

    if action == "create_file" and path and content:
        try:
            f = ROOT / path
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content, encoding="utf-8")
            result_lines.append(f"✅ تم إنشاء: `{path}`")
            # رفع تلقائي
            push = _git_push(f"إنشاء {path} — {message}")
            result_lines.append(push)
        except Exception as e:
            result_lines.append(f"❌ خطأ في الإنشاء: {e}")

    elif action == "edit_file" and path and old and new:
        try:
            f = ROOT / path
            if not f.exists():
                result_lines.append(f"❌ الملف غير موجود: {path}")
            else:
                text = f.read_text(encoding="utf-8")
                if old not in text:
                    result_lines.append(f"❌ النص القديم غير موجود في {path}")
                else:
                    f.write_text(text.replace(old, new, 1), encoding="utf-8")
                    result_lines.append(f"✅ تم التعديل في: `{path}`")
                    push = _git_push(f"تعديل {path} — {message}")
                    result_lines.append(push)
        except Exception as e:
            result_lines.append(f"❌ خطأ في التعديل: {e}")

    elif action == "git_push":
        result_lines.append(_git_push(message))

    if reply:
        result_lines.append(f"\n💬 {reply}")

    return "\n".join(result_lines) if result_lines else reply or "✅ تم"


def _git_push(message: str) -> str:
    try:
        # ضبط هوية git تلقائياً إذا لم تكن موجودة
        subprocess.run(
            ["git", "-C", str(ROOT), "config", "--local", "user.email", "nsm-agent@neural-service-mesh.app"],
            capture_output=True
        )
        subprocess.run(
            ["git", "-C", str(ROOT), "config", "--local", "user.name", "NSM Agent"],
            capture_output=True
        )
        for cmd in [
            ["git", "-C", str(ROOT), "add", "-A"],
            ["git", "-C", str(ROOT), "commit", "-m", message],
            ["git", "-C", str(ROOT), "push"],
        ]:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                if "nothing to commit" in r.stdout + r.stderr:
                    return "ℹ️ لا توجد تغييرات للرفع"
                return f"❌ git error: {r.stderr.strip()}"
        return "📤 رُفع لـ GitHub ✅"
    except Exception as e:
        return f"❌ خطأ git: {e}"


# ══════════════════════════════════════════════════════════════════
# الوكيل الرئيسي
# ══════════════════════════════════════════════════════════════════
class NSMAgent:
    def __init__(self):
        self.available = self._check_available()
        self.history: List[Dict] = []
        self._llm_fallback = None

    @staticmethod
    def _check_available() -> bool:
        return bool(
            (os.getenv("CF_API_TOKEN", "").strip() and os.getenv("CF_ACCOUNT_ID", "").strip()) or
            os.getenv("GOOGLE_API_KEY", "").strip() or
            os.getenv("GROQ_API_KEY", "").strip() or
            os.getenv("OPENROUTER_API_KEY", "").strip() or
            os.getenv("OPENAI_API_KEY", "").strip()
        )

    def _get_api_key(self) -> str:
        """يُعيد قراءة مفتاح Groq (للتوافق مع _call_api)."""
        self.available = self._check_available()
        return os.getenv("GROQ_API_KEY", "").strip()

    def _get_llm_fallback(self):
        """يحمّل LLMFallback كخيار احتياطي عند فشل Groq."""
        if self._llm_fallback is None:
            try:
                from ai.llm_fallback import LLMFallback
                self._llm_fallback = LLMFallback()
            except Exception:
                pass
        return self._llm_fallback

    def run(self, user_input: str) -> str:
        api_key = self._get_api_key()
        if not self.available:
            return "⚠️ لا يوجد مفتاح API — أضف GOOGLE_API_KEY في Streamlit Secrets"

        messages = [{"role": "system", "content": _AGENT_SYSTEM}]
        messages += self.history[-6:]
        messages.append({"role": "user", "content": user_input})

        raw = None
        groq_error = None

        try:
            raw = _call_api(messages, api_key)
        except Exception as e:
            groq_error = str(e)

        # إذا فشل Groq — جرب LLMFallback (نص عادي بدون JSON)
        if raw is None:
            fb = self._get_llm_fallback()
            if fb and fb.available:
                try:
                    result = fb.generate(user_input)
                    return result.text
                except Exception:
                    pass
            # لا يوجد أي مزوّد
            t = user_input.strip()
            if any(t.startswith(p) for p in ("أنشئ", "انشئ", "اكتب كود", "ابنِ", "ابني")):
                return (
                    f"⚠️ Groq غير متاح حالياً ({groq_error})\n\n"
                    f"💡 يمكنك إنشاء الملف مباشرة بالصيغة:\n"
                    f"  أنشئ path/file.py | محتوى الملف\n\n"
                    f"مثال:\n"
                    f"  أنشئ tests/test_simple.py | import unittest\n\nclass TestSimple(unittest.TestCase):\n    def test_basic(self):\n        self.assertTrue(True)\n\nif __name__ == '__main__':\n    unittest.main()"
                )
            return (
                f"⚠️ لا يمكن الوصول لـ Groq حالياً.\n"
                f"السبب: {groq_error}\n"
                f"💡 تحقق من GROQ_API_KEY في Streamlit Secrets"
            )

        # تنظيف JSON — يُنفَّذ دائماً بعد نجاح API
        try:
            raw = raw.strip()
            if raw.startswith("```"):
                lines = raw.split("```")
                raw = lines[1] if len(lines) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            action_data = json.loads(raw)
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": raw})
            return _execute(action_data)

        except json.JSONDecodeError:
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": raw})
            return raw
        except Exception as e:
            return f"❌ خطأ في معالجة الرد: {e}"

    def clear(self):
        self.history.clear()
