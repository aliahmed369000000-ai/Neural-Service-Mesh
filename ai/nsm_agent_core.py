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
# قائمة نماذج بديلة — يُجرَّب الأول، ثم الثاني عند الفشل
_GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
]


# ══════════════════════════════════════════════════════════════════
# استدعاء Groq مع Fallback للنماذج
# ══════════════════════════════════════════════════════════════════
def _call_groq(messages: List[Dict], api_key: str) -> str:
    last_err = None
    for model in _GROQ_MODELS:
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.3,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            _GROQ_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 401:
                raise Exception(f"❌ GROQ_API_KEY غير صالح (401 Unauthorized)")
            elif e.code == 403:
                last_err = f"403 Forbidden على {model}"
                continue  # جرب النموذج التالي
            elif e.code == 429:
                raise Exception(f"❌ تجاوزت حد الطلبات — انتظر قليلاً (429 Rate Limit)")
            else:
                last_err = f"HTTP {e.code} على {model}: {body[:200]}"
                continue
        except Exception as e:
            last_err = str(e)
            continue

    raise Exception(f"❌ فشلت كل نماذج Groq. آخر خطأ: {last_err}")


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
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.available = bool(self.api_key)
        self.history: List[Dict] = []
        self._llm_fallback = None  # يُحمَّل عند الحاجة

    def _get_api_key(self) -> str:
        key = os.getenv("GROQ_API_KEY", "").strip()
        if key:
            self.api_key = key
            self.available = True
        return self.api_key

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
        if not api_key:
            return "⚠️ GROQ_API_KEY غير موجود في Secrets — أضفه من لوحة Streamlit"

        messages = [{"role": "system", "content": _AGENT_SYSTEM}]
        messages += self.history[-6:]
        messages.append({"role": "user", "content": user_input})

        raw = None
        groq_error = None

        try:
            raw = _call_groq(messages, api_key)
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
            # لا يوجد أي مزوّد — أنشئ ملفاً بسيطاً محلياً إذا كان الطلب إنشاء
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

            # تنظيف JSON
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
        # معالجة الرد
        try:
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
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
