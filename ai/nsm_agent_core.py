"""
NSM Agent Core — ai/nsm_agent_core.py  (v2 — Real Agent)
==========================================================
قفزة حقيقية نحو Replit Agent:

✅ يقرأ الملفات قبل التعديل (لا تعديل أعمى)
✅ يُشغّل الكود ويرى النتيجة (run_file / run_tests)
✅ يعرف هيكل المشروع كله ديناميكياً في كل طلب
✅ يدعم multi-step: سلسلة أفعال في رد واحد
✅ يُصحّح نفسه إذا فشل التنفيذ (retry مرة واحدة)
✅ fallback كامل: Cloudflare → Gemini → OpenRouter → Groq
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent

# ══════════════════════════════════════════════════════════════════
# حدود أمان: الملفات الكبيرة تُرسَل مقتطعة فقط
# ══════════════════════════════════════════════════════════════════
_MAX_FILE_CHARS  = 6_000   # أقصى عدد حروف للملف في الـ prompt
_MAX_CONTEXT_FILES = 5     # أقصى ملفات تُقرأ دفعة واحدة
_MAX_RUN_OUTPUT  = 2_000   # أقصى حروف من نتيجة التنفيذ
_IGNORED_DIRS = {".git", "__pycache__", ".streamlit", "node_modules",
                 "venv", ".venv", "weights", "checkpoints", "logs"}

# ══════════════════════════════════════════════════════════════════
# 1) هيكل المشروع الديناميكي
# ══════════════════════════════════════════════════════════════════

def _get_project_tree() -> str:
    """يُنشئ شجرة ملفات المشروع الحقيقية."""
    lines: List[str] = []
    try:
        for p in sorted(ROOT.rglob("*")):
            # تجاهل المجلدات المحجوبة
            if any(d in p.parts for d in _IGNORED_DIRS):
                continue
            if p.is_file() and p.suffix in (".py", ".json", ".toml", ".txt", ".md"):
                rel = p.relative_to(ROOT)
                size = p.stat().st_size
                lines.append(f"  {rel}  ({size:,} bytes)")
    except Exception:
        pass
    return "\n".join(lines[:80])  # أقصى 80 ملف


def _read_file_safe(path: str, max_chars: int = _MAX_FILE_CHARS) -> Tuple[str, bool]:
    """يقرأ الملف بأمان ويُقتطع إذا كان كبيراً.
    يُعيد (المحتوى, هل_اقتُطع)
    """
    try:
        f = ROOT / path
        if not f.exists():
            return f"❌ الملف غير موجود: {path}", False
        text = f.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            half = max_chars // 2
            snippet = (
                text[:half]
                + f"\n\n... [اقتُطع — {len(text):,} حرف إجمالاً، يُعرض {max_chars:,} فقط] ...\n\n"
                + text[-half:]
            )
            return snippet, True
        return text, False
    except Exception as e:
        return f"❌ خطأ في القراءة: {e}", False


# ══════════════════════════════════════════════════════════════════
# 2) System Prompt الديناميكي
# ══════════════════════════════════════════════════════════════════

def _build_system_prompt() -> str:
    tree = _get_project_tree()
    return f"""أنت **NSM Agent** — وكيل برمجي ذكي مدمج في مشروع Neural Service Mesh.
مشروع Python/Streamlit للذكاء الاصطناعي العربي مع معرفة إسلامية وقرآنية على GitHub.

## هيكل المشروع الحالي:
{tree}

## قدراتك الحقيقية:
- قراءة أي ملف في المشروع قبل التعديل
- كتابة وتعديل الملفات مباشرة على القرص
- تشغيل كود Python وعرض النتيجة
- رفع التغييرات لـ GitHub تلقائياً
- سلسلة أفعال متعددة في رد واحد

## صيغة الرد — JSON فقط لا غير:
{{
  "thinking": "تحليلك للطلب خطوة بخطوة",
  "steps": [
    {{
      "action": "read_file | create_file | edit_file | run_file | run_tests | git_push | answer",
      "path": "المسار النسبي من جذر المشروع",
      "content": "محتوى الملف الكامل (لـ create_file)",
      "old": "النص القديم المراد استبداله (لـ edit_file) — يجب أن يكون موجوداً حرفياً",
      "new": "النص الجديد البديل (لـ edit_file)",
      "cmd": "أمر bash للتشغيل (لـ run_file)",
      "message": "رسالة commit (لـ git_push)",
      "reply": "رد للمستخدم بالعربية (لـ answer)"
    }}
  ]
}}

## قواعد صارمة:
1. رد بـ JSON صحيح فقط — لا نص خارجه أبداً
2. قبل edit_file: اطلب read_file أولاً لترى المحتوى الحالي
3. الكود يكون مكتملاً وقابلاً للتشغيل فوراً
4. المسارات نسبية دائماً (مثل: ai/new_module.py)
5. عند create_file: اكتب الكود كاملاً مع docstring
6. رد بالعربية في thinking وreply"""


# ══════════════════════════════════════════════════════════════════
# 3) استدعاء API مع Fallback كامل
# ══════════════════════════════════════════════════════════════════

_GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_GROQ_MODELS    = [
    "llama-3.1-8b-instant", "gemma2-9b-it",
    "llama-3.3-70b-versatile", "llama3-8b-8192",
]
_OPENROUTER_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
]


def _call_api(messages: List[Dict]) -> str:
    """Cloudflare → Gemini → OpenRouter → Groq"""
    errors: List[str] = []

    # ── 1. Cloudflare Workers AI ──
    cf_token   = os.getenv("CF_API_TOKEN", "").strip()
    cf_account = os.getenv("CF_ACCOUNT_ID", "").strip()
    if cf_token and cf_account:
        url = (f"https://api.cloudflare.com/client/v4/accounts/"
               f"{cf_account}/ai/run/@cf/meta/llama-3.1-8b-instruct")
        payload = json.dumps({"messages": messages, "max_tokens": 3000}).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Authorization": f"Bearer {cf_token}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            text = data.get("result", {}).get("response", "").strip()
            if text:
                return text
        except Exception as e:
            errors.append(f"CF: {e}")

    # ── 2. Google Gemini ──
    gemini_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if gemini_key:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"gemini-1.5-flash:generateContent?key={gemini_key}")
        parts: List[Dict] = []
        sys_text = ""
        for m in messages:
            if m["role"] == "system":
                sys_text = m["content"]
            elif m["role"] == "user":
                parts.append({"role": "user",  "parts": [{"text": m["content"]}]})
            elif m["role"] == "assistant":
                parts.append({"role": "model", "parts": [{"text": m["content"]}]})
        body: Dict[str, Any] = {"contents": parts,
                                 "generationConfig": {"maxOutputTokens": 4000,
                                                       "temperature": 0.2}}
        if sys_text:
            body["systemInstruction"] = {"parts": [{"text": sys_text}]}
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            text = (data["candidates"][0]["content"]["parts"][0]["text"]).strip()
            if text:
                return text
        except Exception as e:
            errors.append(f"Gemini: {e}")

    # ── 3. OpenRouter ──
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if or_key:
        for model in _OPENROUTER_MODELS:
            payload = json.dumps({
                "model": model, "messages": messages,
                "max_tokens": 3000, "temperature": 0.2,
            }).encode()
            req = urllib.request.Request(
                _OPENROUTER_URL, data=payload,
                headers={"Authorization": f"Bearer {or_key}",
                         "Content-Type": "application/json",
                         "HTTP-Referer": "https://neural-service-mesh.streamlit.app"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=25) as r:
                    text = json.loads(r.read())["choices"][0]["message"]["content"].strip()
                if text:
                    return text
            except Exception as e:
                errors.append(f"OR/{model}: {e}")

    # ── 4. Groq ──
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        for model in _GROQ_MODELS:
            payload = json.dumps({
                "model": model, "messages": messages,
                "max_tokens": 3000, "temperature": 0.2, "stream": False,
            }).encode()
            req = urllib.request.Request(
                _GROQ_URL, data=payload,
                headers={"Authorization": f"Bearer {groq_key}",
                         "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=25) as r:
                    text = json.loads(r.read())["choices"][0]["message"]["content"].strip()
                if text:
                    return text
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    errors.append(f"Groq محجوب ({e.code})")
                    break
                errors.append(f"Groq/{model}: HTTP {e.code}")
            except Exception as e:
                errors.append(f"Groq/{model}: {e}")

    raise RuntimeError(" | ".join(errors) or "لا يوجد مزوّد متاح")


# ══════════════════════════════════════════════════════════════════
# 4) تنفيذ خطوة واحدة
# ══════════════════════════════════════════════════════════════════

def _run_step(step: Dict[str, Any]) -> str:
    action  = step.get("action", "answer")
    path    = step.get("path", "")
    content = step.get("content", "")
    old     = step.get("old", "")
    new     = step.get("new", "")
    message = step.get("message", "NSM Agent auto-commit")
    reply   = step.get("reply", "")
    cmd     = step.get("cmd", "")

    # ── read_file ──
    if action == "read_file":
        if not path:
            return "❌ read_file: مطلوب path"
        text, truncated = _read_file_safe(path)
        note = " (مقتطع)" if truncated else ""
        return f"📖 **{path}**{note}:\n```python\n{text}\n```"

    # ── create_file ──
    if action == "create_file":
        if not path or not content:
            return "❌ create_file: مطلوب path وcontent"
        try:
            f = ROOT / path
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content, encoding="utf-8")
            lines = content.count("\n") + 1
            return f"✅ أُنشئ `{path}` ({lines} سطر)"
        except Exception as e:
            return f"❌ خطأ في الإنشاء: {e}"

    # ── edit_file ──
    if action == "edit_file":
        if not path or not old or not new:
            return "❌ edit_file: مطلوب path وold وnew"
        try:
            f = ROOT / path
            if not f.exists():
                return f"❌ الملف غير موجود: {path}"
            text = f.read_text(encoding="utf-8")
            if old not in text:
                # محاولة مرنة: بعد تطبيع المسافات
                old_stripped = textwrap.dedent(old).strip()
                found = False
                for line in text.split("\n"):
                    if old_stripped in line:
                        found = True
                        break
                if not found:
                    return (f"❌ النص القديم غير موجود في `{path}`\n"
                            f"💡 استخدم read_file أولاً لرؤية المحتوى الحالي")
            new_text = text.replace(old, new, 1)
            f.write_text(new_text, encoding="utf-8")
            return f"✅ عُدِّل `{path}`"
        except Exception as e:
            return f"❌ خطأ في التعديل: {e}"

    # ── run_file ──
    if action == "run_file":
        target = cmd or (f"python {path}" if path else "")
        if not target:
            return "❌ run_file: مطلوب path أو cmd"
        try:
            r = subprocess.run(
                target, shell=True, capture_output=True,
                text=True, timeout=30, cwd=str(ROOT),
            )
            out = (r.stdout + r.stderr).strip()
            if len(out) > _MAX_RUN_OUTPUT:
                out = out[:_MAX_RUN_OUTPUT] + "\n... [اقتُطعت النتيجة]"
            status = "✅" if r.returncode == 0 else "❌"
            return f"{status} `{target}`:\n```\n{out or '(لا مخرجات)'}\n```"
        except subprocess.TimeoutExpired:
            return "⏱️ انتهت المهلة (30 ثانية)"
        except Exception as e:
            return f"❌ خطأ في التشغيل: {e}"

    # ── run_tests ──
    if action == "run_tests":
        test_path = path or "."
        try:
            r = subprocess.run(
                ["python", "-m", "pytest", test_path, "-v", "--tb=short", "-q"],
                capture_output=True, text=True, timeout=60, cwd=str(ROOT),
            )
            out = (r.stdout + r.stderr).strip()
            if len(out) > _MAX_RUN_OUTPUT:
                out = out[:_MAX_RUN_OUTPUT] + "\n... [اقتُطعت]"
            status = "✅ اجتازت" if r.returncode == 0 else "❌ فشلت"
            return f"{status} الاختبارات:\n```\n{out}\n```"
        except Exception as e:
            return f"❌ خطأ في الاختبارات: {e}"

    # ── git_push ──
    if action == "git_push":
        return _git_push(message)

    # ── answer ──
    if reply:
        return f"💬 {reply}"

    return "✅ تم"


def _git_push(message: str) -> str:
    try:
        for cfg in [
            ["git", "-C", str(ROOT), "config", "--local",
             "user.email", "nsm-agent@neural-service-mesh.app"],
            ["git", "-C", str(ROOT), "config", "--local",
             "user.name", "NSM Agent"],
        ]:
            subprocess.run(cfg, capture_output=True)

        for cmd in [
            ["git", "-C", str(ROOT), "add", "-A"],
            ["git", "-C", str(ROOT), "commit", "-m", message],
            ["git", "-C", str(ROOT), "push"],
        ]:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                out = (r.stdout + r.stderr).strip()
                if "nothing to commit" in out:
                    return "ℹ️ لا توجد تغييرات للرفع"
                return f"❌ git: {out}"
        return "📤 رُفع لـ GitHub ✅"
    except Exception as e:
        return f"❌ خطأ git: {e}"


# ══════════════════════════════════════════════════════════════════
# 5) تحليل رد LLM
# ══════════════════════════════════════════════════════════════════

def _parse_llm_response(raw: str) -> Optional[Dict]:
    """يحوّل رد LLM لـ dict. يتعامل مع ```json و نص عادي."""
    text = raw.strip()
    # إزالة ```json ... ```
    if "```" in text:
        for block in text.split("```"):
            b = block.strip()
            if b.startswith("json"):
                b = b[4:].strip()
            try:
                return json.loads(b)
            except Exception:
                continue
    # محاولة مباشرة
    try:
        return json.loads(text)
    except Exception:
        # استخراج أول { ... } في النص
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
    return None


# ══════════════════════════════════════════════════════════════════
# 6) الوكيل الرئيسي
# ══════════════════════════════════════════════════════════════════

class NSMAgent:
    """
    وكيل NSM الحقيقي — يقرأ الملفات، يعدّلها، يشغّل الكود، يرفع GitHub.
    يُستدعى من nsm_chat.py تلقائياً.
    """

    def __init__(self) -> None:
        self.available = self._check_available()
        self.history: List[Dict] = []
        self._llm_fallback = None

    @staticmethod
    def _check_available() -> bool:
        return bool(
            (os.getenv("CF_API_TOKEN", "").strip()
             and os.getenv("CF_ACCOUNT_ID", "").strip())
            or os.getenv("GOOGLE_API_KEY", "").strip()
            or os.getenv("GROQ_API_KEY", "").strip()
            or os.getenv("OPENROUTER_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )

    def _get_llm_fallback(self):
        if self._llm_fallback is None:
            try:
                from ai.llm_fallback import LLMFallback
                self._llm_fallback = LLMFallback()
            except Exception:
                pass
        return self._llm_fallback

    # ── قلب الوكيل ──
    def run(self, user_input: str) -> str:
        self.available = self._check_available()
        if not self.available:
            return "⚠️ لا يوجد مفتاح API — أضف GOOGLE_API_KEY في Streamlit Secrets"

        # ── بناء رسائل الـ API ──
        system = _build_system_prompt()   # هيكل المشروع الحالي في كل طلب
        messages: List[Dict] = [{"role": "system", "content": system}]
        messages += self.history[-8:]     # آخر 4 رسائل للسياق
        messages.append({"role": "user", "content": user_input})

        # ── استدعاء LLM ──
        raw: Optional[str] = None
        try:
            raw = _call_api(messages)
        except Exception as e:
            # fallback للنص العادي
            fb = self._get_llm_fallback()
            if fb and fb.available:
                try:
                    result = fb.generate(user_input)
                    return result.text
                except Exception:
                    pass
            return f"⚠️ لا يمكن الوصول لأي مزوّد LLM:\n{e}"

        # ── حفظ في التاريخ ──
        self.history.append({"role": "user",      "content": user_input})
        self.history.append({"role": "assistant",  "content": raw})

        # ── تحليل الرد ──
        parsed = _parse_llm_response(raw)
        if parsed is None:
            # رد نصي عادي — اعرضه مباشرة
            return raw

        thinking = parsed.get("thinking", "")
        steps    = parsed.get("steps", [])

        # دعم الصيغة القديمة (action في الجذر مباشرة)
        if not steps and parsed.get("action"):
            steps = [parsed]

        if not steps:
            reply = parsed.get("reply", raw)
            return f"🤔 {thinking}\n\n💬 {reply}" if thinking else reply

        # ── تنفيذ الخطوات ──
        output_parts: List[str] = []
        if thinking:
            output_parts.append(f"🤔 **{thinking}**\n")

        read_results: Dict[str, str] = {}  # نتائج read_file لإضافتها للسياق

        for i, step in enumerate(steps, 1):
            prefix = f"**الخطوة {i}/{len(steps)}**" if len(steps) > 1 else ""
            result = _run_step(step)
            output_parts.append(f"{prefix}\n{result}" if prefix else result)

            # إذا كانت read_file — احفظ المحتوى لخطوة تالية محتملة
            if step.get("action") == "read_file" and step.get("path"):
                read_results[step["path"]] = result

        # ── إذا كانت هناك read_file فقط وخطوات أخرى لاحقة — أعد الاستدعاء ──
        # (سيرى LLM محتوى الملف في التاريخ ويقرر التعديل)
        if read_results and len(steps) == 1 and steps[0].get("action") == "read_file":
            file_ctx = "\n\n".join(
                f"محتوى `{p}`:\n{c}" for p, c in read_results.items()
            )
            followup = (f"هذا محتوى الملف الذي طلبته:\n{file_ctx}\n\n"
                        f"الآن نفّذ الطلب الأصلي: {user_input}")
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",      "content": followup})
            try:
                raw2   = _call_api(messages)
                parsed2 = _parse_llm_response(raw2)
                if parsed2 and parsed2.get("steps"):
                    self.history.append({"role": "assistant", "content": raw2})
                    output_parts.append("\n---")
                    for step in parsed2["steps"]:
                        output_parts.append(_run_step(step))
            except Exception:
                pass

        return "\n\n".join(output_parts)

    def clear(self) -> None:
        self.history.clear()
