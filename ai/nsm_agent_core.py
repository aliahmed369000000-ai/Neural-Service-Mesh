"""
NSM Agent Core — ai/nsm_agent_core.py  (v3 — Replit Agent Level)
=================================================================
الجديد في v3:

✅ [v2] يقرأ الملفات قبل التعديل
✅ [v2] يُشغّل الكود ويرى النتيجة
✅ [v2] هيكل المشروع الديناميكي في كل طلب
✅ [v2] multi-step في رد واحد
✅ [v2] fallback: CF → Gemini → OpenRouter → Groq

🆕 [v3] Streaming بحرف بحرف — Generator يرسل النتائج فور اكتمال كل خطوة
🆕 [v3] Self-Healing Loop — يصحح أخطاءه تلقائياً (حتى 3 محاولات)
🆕 [v3] Read-Before-Edit تلقائي — إذا طُلب edit_file بدون read_file سابق،
         يقرأ الملف أولاً تلقائياً ثم ينفذ التعديل في نفس الدورة
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

ROOT = Path(__file__).parent.parent

# ══════════════════════════════════════════════════════════════════
# حدود أمان
# ══════════════════════════════════════════════════════════════════
_MAX_FILE_CHARS    = 6_000
_MAX_CONTEXT_FILES = 5
_MAX_RUN_OUTPUT    = 2_000
_MAX_HEAL_ATTEMPTS = 3      # 🆕 v3: أقصى محاولات إصلاح تلقائي
_IGNORED_DIRS = {
    ".git", "__pycache__", ".streamlit", "node_modules",
    "venv", ".venv", "weights", "checkpoints", "logs",
}


# ══════════════════════════════════════════════════════════════════
# 1) هيكل المشروع الديناميكي
# ══════════════════════════════════════════════════════════════════

def _get_project_tree() -> str:
    lines: List[str] = []
    try:
        for p in sorted(ROOT.rglob("*")):
            if any(d in p.parts for d in _IGNORED_DIRS):
                continue
            if p.is_file() and p.suffix in (".py", ".json", ".toml", ".txt", ".md"):
                rel = p.relative_to(ROOT)
                size = p.stat().st_size
                lines.append(f"  {rel}  ({size:,} bytes)")
    except Exception:
        pass
    return "\n".join(lines[:80])


def _read_file_safe(path: str, max_chars: int = _MAX_FILE_CHARS) -> Tuple[str, bool]:
    """يقرأ الملف بأمان. يُعيد (المحتوى, هل_اقتُطع)"""
    try:
        f = ROOT / path
        if not f.exists():
            return f"❌ الملف غير موجود: {path}", False
        text = f.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            half = max_chars // 2
            snippet = (
                text[:half]
                + f"\n\n... [اقتُطع — {len(text):,} حرف، يُعرض {max_chars:,} فقط] ...\n\n"
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
    return f"""أنت **NSM Agent v3** — وكيل برمجي ذكي مدمج في مشروع Neural Service Mesh.
مشروع Python/Streamlit للذكاء الاصطناعي العربي مع معرفة إسلامية وقرآنية على GitHub.

## هيكل المشروع الحالي:
{tree}

## قدراتك الحقيقية:
- قراءة أي ملف في المشروع قبل التعديل
- كتابة وتعديل الملفات مباشرة على القرص
- تشغيل كود Python وعرض النتيجة
- رفع التغييرات لـ GitHub تلقائياً
- 🆕 بحث حقيقي في الإنترنت (بدون مفتاح API) لمعلومات حديثة أو خارجية
- سلسلة أفعال متعددة في رد واحد
- تصحيح أخطائك تلقائياً إذا فشل التنفيذ

## صيغة الرد — JSON فقط لا غير:
{{
  "thinking": "تحليلك للطلب خطوة بخطوة",
  "steps": [
    {{
      "action": "read_file | create_file | edit_file | run_file | run_tests | git_push | web_search | answer",
      "path": "المسار النسبي من جذر المشروع",
      "content": "محتوى الملف الكامل (لـ create_file)",
      "old": "النص القديم المراد استبداله (لـ edit_file) — يجب أن يكون موجوداً حرفياً",
      "new": "النص الجديد البديل (لـ edit_file)",
      "cmd": "أمر bash للتشغيل (لـ run_file)",
      "message": "رسالة commit (لـ git_push)",
      "query": "نص البحث (لـ web_search فقط)",
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
6. رد بالعربية في thinking وreply
7. إذا فشل run_file: أصلح الخطأ وأعد المحاولة تلقائياً
8. ⚠️ "action" يجب أن يكون **كلمة واحدة فقط** من القائمة (مثل "read_file")
   — لا تكتب القائمة كاملة مفصولة بـ | كما هي في الوصف أعلاه، هذا خطأ.
9. ⚠️ عند طلب "افحص/اقرأ المشروع": لا تقرأ كل الملفات — اختر فقط 5-8 ملفات
   الأكثر صلة بالسؤال (احكم من الأسماء ووظائفها في هيكل المشروع أعلاه).
10. ⚠️ آخر خطوة في "steps" يجب أن تكون دائماً "answer" فيها "reply" يلخّص
    ما وجدته ويجاوب على سؤال المستخدم مباشرة — لا تكتفِ بقراءة الملفات فقط.
11. 🆕 أي سؤال عن معلومة حالية أو حديثة (رئيس/مسؤول حالي، سعر اليوم، تاريخ
    اليوم، أخبار، آخر إصدار من برنامج، إلخ) — استخدم خطوة "web_search" أولاً
    ثم اجعل الرد النهائي مبنياً على نتائجها الفعلية فقط. ممنوع تقول "لا
    أستطيع توفير معلومات عن الأشخاص/الأحداث الحالية" — لديك أداة بحث حقيقية
    الآن، استخدمها. وممنوع تختلق رقماً أو اسماً من عندك بدون بحث فعلي.
12. 🆕 في حقل "cmd" (لـ run_file): لا تضع علامات اقتباس مزدوجة متداخلة غير
    مهرّبة (مثل استخدام " بداخل نص محاط أصلاً بـ "). استخدم علامات اقتباس
    مفردة ' بالداخل، أو أنشئ ملف Python كامل عبر create_file وشغّله بـ
    run_file بدل كتابة أكواد معقدة داخل سطر cmd واحد.

## مثال حقيقي لرد صحيح (وليس نصاً تنسخه — فقط توضيح للصيغة):
{{
  "thinking": "المستخدم يريد قراءة agent_factory.py أولاً",
  "steps": [
    {{"action": "read_file", "path": "ai/agent_factory.py"}}
  ]
}}"""


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
        body: Dict[str, Any] = {
            "contents": parts,
            "generationConfig": {"maxOutputTokens": 4000, "temperature": 0.2},
        }
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
                headers={
                    "Authorization": f"Bearer {or_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://neural-service-mesh.streamlit.app",
                },
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
    query   = step.get("query", "")

    # ── 🆕 حماية: النموذج أحياناً (خصوصاً النماذج الصغيرة/الاحتياطية)
    # ينسخ قيمة الحقل من الـ schema حرفياً بدل اختيار فعل واحد حقيقي،
    # مثل: "action": "read_file | create_file | edit_file | ..."
    # هذا كان يمر بصمت كـ"✅ تم" بدون تنفيذ أي شيء فعلي. الآن نرفضه
    # صراحة كخطأ قابل للاكتشاف عبر _is_failure() ليُعاد المحاولة تلقائياً.
    _VALID_ACTIONS = {
        "read_file", "create_file", "edit_file",
        "run_file", "run_tests", "git_push", "web_search", "answer",
    }
    if action not in _VALID_ACTIONS:
        return (f"❌ فعل غير صالح من النموذج: '{action}'\n"
                f"💡 يجب اختيار فعل واحد بالضبط من: "
                f"read_file, create_file, edit_file, run_file, run_tests, git_push, web_search, answer")

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
                old_stripped = textwrap.dedent(old).strip()
                found = any(old_stripped in line for line in text.split("\n"))
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

    # ── web_search ── 🆕
    if action == "web_search":
        if not query:
            return "❌ web_search: مطلوب query (نص البحث)"
        try:
            from ai.web_search_tool import web_search as _web_search
            return _web_search(query)
        except Exception as e:
            return f"❌ خطأ في أداة البحث: {e}"

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
    """
    يحوّل رد LLM لـ dict.
    يجرب 5 طرق استخراج قبل الاستسلام.
    """
    text = raw.strip()

    # ── طريقة 1: JSON مباشر ──
    try:
        return json.loads(text)
    except Exception:
        pass

    # ── طريقة 2: كتلة ```json ... ``` ──
    if "```" in text:
        import re
        for m in re.finditer(r"```(?:json)?(.*?)```", text, re.DOTALL):
            block = m.group(1).strip()
            try:
                return json.loads(block)
            except Exception:
                continue

    # ── طريقة 3: أول { ... } في النص ──
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass

    # ── طريقة 4: تنظيف trailing commas ثم إعادة المحاولة ──
    if start != -1 and end > start:
        import re
        cleaned = re.sub(r",\s*([}\]])", r"\1", text[start:end+1])
        try:
            return json.loads(cleaned)
        except Exception:
            pass

    # ── طريقة 5: بناء رد answer من النص الحر ──
    # 🆕 مهم: هذا يُستخدم فقط لو النص "حر" فعلاً (بدون أي أثر لمحاولة JSON).
    # لو النص فيه علامات JSON واضحة (مثل "action" أو "steps" أو يبدأ بـ {)
    # وفشل تحليله كـ JSON صحيح، فهذا فشل حقيقي في التحليل يجب أن يُعاد
    # محاولته وليس أن يُعرض كإجابة سليمة على المستخدم كما كان يحدث سابقاً
    # (كان يُسرّب نص JSON مكسور خام مباشرة للمستخدم).
    looks_like_json_attempt = (
        text.lstrip().startswith("{")
        or '"action"' in text
        or '"steps"' in text
        or '"thinking"' in text
    )
    if looks_like_json_attempt:
        return None

    if text and len(text) > 5:
        return {
            "thinking": "",
            "steps": [{"action": "answer", "reply": text}]
        }

    return None


# ══════════════════════════════════════════════════════════════════
# 🆕 v3 — إضافة 1: Read-Before-Edit تلقائي
# ══════════════════════════════════════════════════════════════════

def _inject_read_before_edit(steps: List[Dict]) -> List[Dict]:
    """
    إذا وُجد edit_file بدون read_file سابق لنفس الملف،
    يُضيف read_file تلقائياً قبله.
    هذا يجعل الوكيل يرى المحتوى الحالي دائماً قبل التعديل.
    """
    result: List[Dict] = []
    read_paths: set = set()

    for step in steps:
        action = step.get("action", "")
        path   = step.get("path", "")

        if action == "read_file" and path:
            read_paths.add(path)

        if action == "edit_file" and path and path not in read_paths:
            # أضف read_file تلقائياً
            result.append({"action": "read_file", "path": path,
                            "_auto": True})  # علامة داخلية
            read_paths.add(path)

        result.append(step)

    return result


# ══════════════════════════════════════════════════════════════════
# 🆕 v3 — إضافة 2: Self-Healing Loop
# ══════════════════════════════════════════════════════════════════

def _is_failure(result: str) -> bool:
    """يتحقق إذا كانت نتيجة الخطوة فشلاً يستحق الإصلاح."""
    return result.startswith("❌") and any(
        kw in result for kw in [
            "خطأ في التشغيل", "خطأ في الإنشاء", "خطأ في التعديل",
            "غير موجود", "SyntaxError", "ImportError", "ModuleNotFoundError",
            "NameError", "TypeError", "IndentationError",
            "فعل غير صالح", "فشل البحث", "خطأ في أداة البحث", "مطلوب",
        ]
    )


def _build_heal_prompt(
    original_request: str,
    failed_step: Dict,
    error_msg: str,
    attempt: int,
) -> str:
    """يبني prompt لطلب الإصلاح من LLM."""
    return (
        f"فشلت الخطوة في المحاولة {attempt}/{_MAX_HEAL_ATTEMPTS}:\n"
        f"الخطوة: {json.dumps(failed_step, ensure_ascii=False)}\n"
        f"الخطأ: {error_msg}\n\n"
        f"الطلب الأصلي: {original_request}\n\n"
        f"أصلح المشكلة وأرسل خطوات جديدة صحيحة بصيغة JSON فقط."
    )


# ══════════════════════════════════════════════════════════════════
# 🆕 v3 — إضافة 3: Streaming Generator
# ══════════════════════════════════════════════════════════════════

_MAX_STEPS_PER_RESPONSE = 12  # 🆕 حماية من استجابة تقرأ عشرات الملفات دفعة واحدة بلا خلاصة


def _stream_steps(
    steps: List[Dict],
    thinking: str,
    messages: List[Dict],
    original_request: str,
) -> Generator[str, None, None]:
    """
    Generator يُرسل النتائج فور اكتمال كل خطوة (Streaming).
    يدعم Self-Healing: إذا فشلت خطوة، يطلب الإصلاح ويعيد المحاولة.
    """
    if thinking:
        yield f"🤔 **{thinking}**\n\n"

    # ── 🆕 سقف عدد الخطوات: يمنع قراءة عشرات الملفات دفعة واحدة ──
    truncated = False
    if len(steps) > _MAX_STEPS_PER_RESPONSE:
        truncated = True
        steps = steps[:_MAX_STEPS_PER_RESPONSE]

    total = len(steps)
    has_answer = any(s.get("action") == "answer" for s in steps)

    for i, step in enumerate(steps, 1):
        action = step.get("action", "answer")
        prefix = f"**الخطوة {i}/{total}** " if total > 1 else ""

        # علامة القراءة التلقائية
        if step.get("_auto"):
            yield f"{prefix}🔍 *قراءة تلقائية قبل التعديل...*\n"

        # ── تنفيذ الخطوة ──
        result = _run_step(step)
        yield f"{prefix}{result}\n\n"

        # ── Self-Healing Loop 🆕 ──
        # 🆕 وسّعنا الشرط: أي فشل حقيقي يستحق إصلاحاً، وليس فقط
        # run_file/create_file/edit_file (كان يفوت حالات مثل "فعل غير صالح").
        if _is_failure(result):
            healed = False
            for attempt in range(1, _MAX_HEAL_ATTEMPTS + 1):
                yield f"🔧 **محاولة إصلاح تلقائي {attempt}/{_MAX_HEAL_ATTEMPTS}...**\n"

                heal_messages = list(messages) + [
                    {
                        "role": "user",
                        "content": _build_heal_prompt(
                            original_request, step, result, attempt
                        ),
                    }
                ]

                try:
                    raw_heal  = _call_api(heal_messages)
                    parsed_h  = _parse_llm_response(raw_heal)
                    if not parsed_h:
                        yield "⚠️ لم أتمكن من تحليل رد الإصلاح\n"
                        break

                    heal_steps = parsed_h.get("steps", [])
                    if not heal_steps and parsed_h.get("action"):
                        heal_steps = [parsed_h]

                    if not heal_steps:
                        yield "⚠️ لا توجد خطوات إصلاح\n"
                        break

                    # تنفيذ خطوات الإصلاح
                    all_ok = True
                    for hs in heal_steps:
                        hr = _run_step(hs)
                        yield f"  ↳ {hr}\n"
                        if _is_failure(hr):
                            all_ok = False
                            result = hr  # للمحاولة التالية
                            break

                    if all_ok:
                        yield f"✅ **تم الإصلاح في المحاولة {attempt}**\n\n"
                        healed = True
                        break

                except Exception as e:
                    yield f"⚠️ خطأ في الإصلاح: {e}\n"
                    break

            if not healed:
                yield f"❌ **فشل الإصلاح بعد {_MAX_HEAL_ATTEMPTS} محاولات**\n\n"

    # ── 🆕 إذا قُصّت الخطوات، أخبر المستخدم صراحة ──
    if truncated:
        yield (f"⚠️ **الطلب احتاج أكثر من {_MAX_STEPS_PER_RESPONSE} خطوة "
               f"(قراءة ملفات كثيرة جداً دفعة واحدة).**\n"
               f"نفّذت أول {_MAX_STEPS_PER_RESPONSE} فقط لتجنّب استهلاك مفرط للـ API. "
               f"حدّد الملفات المهمة تحديداً (مثال: \"افحص ai/goal_planner.py و"
               f"ai/agent_factory.py و ai/github_sync.py فقط\") لتحليل أدق وأسرع.\n\n")

    # ── 🆕 ضمان وجود خلاصة نهائية دائماً (خصوصاً لطلبات القراءة/التحليل) ──
    elif not has_answer and total > 1:
        yield ("💬 **انتهت القراءة.** لخّص لي الآن بناءً على ما رأيته أعلاه: "
               "ماذا ينقص بالضبط، وما هي الخطوات العملية التالية؟ اسأل مباشرة "
               "وسأجاوب بناءً على الملفات التي قرأتها للتو.")


# ══════════════════════════════════════════════════════════════════
# 6) الوكيل الرئيسي
# ══════════════════════════════════════════════════════════════════

class NSMAgent:
    """
    وكيل NSM v3 — Replit Agent Level:
    - Streaming بحرف بحرف عبر run_stream()
    - Self-Healing تلقائي (حتى 3 محاولات)
    - Read-Before-Edit تلقائي
    - run() للتوافق مع nsm_chat.py القديم (يجمع الـ stream)
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

    # ══════════════════════════════════════════════════════════════
    # 🆕 v3: run_stream — Streaming Generator
    # ══════════════════════════════════════════════════════════════
    def run_stream(self, user_input: str) -> Generator[str, None, None]:
        """
        Generator يُرسل أجزاء الرد فور اكتمال كل خطوة.
        الاستخدام في Streamlit:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full = ""
                for chunk in agent.run_stream(user_input):
                    full += chunk
                    placeholder.markdown(full)
        """
        self.available = self._check_available()
        if not self.available:
            yield "⚠️ لا يوجد مفتاح API — أضف GOOGLE_API_KEY في Streamlit Secrets"
            return

        # 🆕 Planning Engine — يكشف طلبات بناء التطبيقات
        try:
            from ai.nsm_planner import NSMPlanner, is_planning_request
            if is_planning_request(user_input):
                planner = NSMPlanner(self)
                yield from planner.build(user_input)
                return
        except ImportError:
            pass  # إذا لم يكن الـ Planner موجوداً، تابع عادياً

        # بناء رسائل API
        system   = _build_system_prompt()
        messages: List[Dict] = [{"role": "system", "content": system}]
        messages += self.history[-8:]
        messages.append({"role": "user", "content": user_input})

        yield "⏳ *أفكر...*\n\n"

        # استدعاء LLM
        raw: Optional[str] = None
        try:
            raw = _call_api(messages)
        except Exception as e:
            fb = self._get_llm_fallback()
            if fb and fb.available:
                try:
                    result = fb.generate(user_input)
                    yield result.text
                    return
                except Exception:
                    pass
            yield f"⚠️ لا يمكن الوصول لأي مزوّد LLM:\n{e}"
            return

        # حفظ في التاريخ
        self.history.append({"role": "user",      "content": user_input})
        self.history.append({"role": "assistant",  "content": raw})

        # تحليل الرد
        parsed = _parse_llm_response(raw)

        # 🆕 فشل تحليل حقيقي (JSON مكسور، غالباً بسبب اقتباسات غير مهرّبة داخل
        # حقل مثل cmd) — نصلحه بإعادة سؤال النموذج، بدل ما نسرّب النص الخام
        # المكسور مباشرة للمستخدم كما كان يحدث سابقاً.
        if parsed is None:
            healed_parse = None
            for attempt in range(1, _MAX_HEAL_ATTEMPTS + 1):
                yield f"🔧 *الرد السابق لم يكن JSON صالحاً — إصلاح تلقائي ({attempt}/{_MAX_HEAL_ATTEMPTS})...*\n"
                repair_messages = list(messages) + [
                    {"role": "assistant", "content": raw[:1500]},
                    {
                        "role": "user",
                        "content": (
                            "ردك السابق لم يكن JSON صالحاً ولا يمكن تحليله (على الأغلب بسبب "
                            "علامات اقتباس داخلية غير مهرّبة في حقل مثل cmd أو content). "
                            "أعد الإرسال الآن بصيغة JSON صحيحة فقط، بدون أي نص خارج الأقواس، "
                            "وتأكد من تهريب أي علامة اقتباس مزدوجة داخل أي قيمة نصية بوضع \\\\ قبلها. "
                            "إن كان الكود يحتاج علامات اقتباس متداخلة، استخدم علامات اقتباس مفردة "
                            "بالداخل بدل المزدوجة."
                        ),
                    },
                ]
                try:
                    raw_repair = _call_api(repair_messages)
                except Exception:
                    continue
                healed_parse = _parse_llm_response(raw_repair)
                if healed_parse is not None:
                    raw = raw_repair
                    break

            if healed_parse is None:
                yield ("⚠️ تعذّر تحليل رد النموذج بصيغة صحيحة بعد عدة محاولات. "
                       "جرّب إعادة صياغة طلبك بشكل أبسط أو أكثر تحديداً.")
                return
            parsed = healed_parse

        thinking = parsed.get("thinking", "")
        steps    = parsed.get("steps", [])

        # دعم الصيغة القديمة
        if not steps and parsed.get("action"):
            steps = [parsed]

        if not steps:
            reply = parsed.get("reply", raw)
            if thinking:
                yield f"🤔 {thinking}\n\n"
            yield f"💬 {reply}"
            return

        # 🆕 Read-Before-Edit تلقائي
        steps = _inject_read_before_edit(steps)

        # 🆕 Stream الخطوات مع Self-Healing
        yield from _stream_steps(steps, thinking, messages, user_input)

    # ══════════════════════════════════════════════════════════════
    # run() — للتوافق مع nsm_chat.py القديم
    # ══════════════════════════════════════════════════════════════
    def run(self, user_input: str) -> str:
        """
        يجمع كل chunks من run_stream في نص واحد.
        متوافق 100% مع nsm_chat.py بدون أي تعديل فيه.
        """
        parts: List[str] = []
        for chunk in self.run_stream(user_input):
            parts.append(chunk)
        return "".join(parts).replace("⏳ *أفكر...*\n\n", "", 1)

    def _call_api_bound(self):
        """يُعيد دالة _call_api للاستخدام من الـ Planner"""
        return _call_api

    def clear(self) -> None:
        self.history.clear()
