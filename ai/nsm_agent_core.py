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
_MODEL = "llama-3.1-8b-instant"


# ══════════════════════════════════════════════════════════════════
# استدعاء Groq
# ══════════════════════════════════════════════════════════════════
def _call_groq(messages: List[Dict], api_key: str) -> str:
    payload = json.dumps({
        "model": _MODEL,
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
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


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

    def run(self, user_input: str) -> str:
        if not self.available:
            return "⚠️ GROQ_API_KEY غير موجود في Secrets"

        # بناء الرسائل
        messages = [{"role": "system", "content": _AGENT_SYSTEM}]
        messages += self.history[-6:]  # آخر 3 تبادلات
        messages.append({"role": "user", "content": user_input})

        try:
            raw = _call_groq(messages, self.api_key)

            # تنظيف JSON
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            action_data = json.loads(raw)

            # حفظ في التاريخ
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": raw})

            return _execute(action_data)

        except json.JSONDecodeError:
            # Groq أجاب نصاً عادياً
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": raw})
            return raw
        except Exception as e:
            return f"❌ خطأ في الوكيل: {e}"

    def clear(self):
        self.history.clear()
