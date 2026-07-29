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

# ── وضع المالك — يحدد هل أفعال الوكيل الخطرة (كتابة/تنفيذ/push) مفعّلة ──
# نفس الآلية المستخدمة في nsm_chat.py و streamlit_app.py: هذه الأفعال
# تكتب/تنفذ/تدفع فعلياً على الخادم والمستودع، ويجب ألا تُنفَّذ أبداً
# لمحادثة عامة يقودها LLM بناءً على نص زائر مجهول — الحكم بالسماح لا
# يجوز أن يُترك لتقدير النموذج نفسه (عرضة لحقن التعليمات/prompt injection).
try:
    import streamlit as _st
    _HAS_STREAMLIT_AGENT = True
except Exception:
    _HAS_STREAMLIT_AGENT = False


def _is_admin_unlocked() -> bool:
    if not _HAS_STREAMLIT_AGENT:
        return False
    try:
        return bool(_st.session_state.get("_dev_console_unlocked", False))
    except Exception:
        return False


# الأفعال الآمنة لأي زائر: تجيب على سؤال، أو تبحث (ويب/صور) بدون لمس
# الخادم. كل ما عداها (قراءة/كتابة/تعديل ملفات، تشغيل أوامر shell،
# git push، إنشاء واجهة تفاعلية مشتركة، استدعاء API عام) يتطلب فتح
# وضع المالك أولاً.
_PUBLIC_SAFE_ACTIONS = {"answer", "web_search", "image_search"}
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

# 🆕 المرحلة 7 من خطة "المراحل المقترحة" — ملف تعليمات دائم
_NSM_INSTRUCTIONS_FILE = "nsm.md"
_MAX_NSM_INSTRUCTIONS_CHARS = 6_000


def _read_persistent_instructions() -> str:
    """
    سابقاً: تفضيلات أسلوب العمل الثابتة (رسائل عربية، هوية الكوميت،
    قواعد الرفع/الأمان...) كانت تُكرَّر يدوياً في كل طلب لأن الوكيل لا
    يملك ذاكرة دائمة عبر الجلسات. هذه الدالة تقرأ `nsm.md` من جذر
    المشروع (إن وُجد) وتُدرِج محتواه في الـ system prompt في بداية كل
    جلسة، فتُطبَّق التفضيلات تلقائياً دون الحاجة لتكرارها.

    لا ترمي استثناءً أبداً — غياب الملف أو فشل قراءته يُعيد نصاً فارغاً
    (سلوك افتراضي، ليس خطأً يوقف الوكيل).
    """
    try:
        f = ROOT / _NSM_INSTRUCTIONS_FILE
        if not f.exists():
            return ""
        text = f.read_text(encoding="utf-8")
        if len(text) > _MAX_NSM_INSTRUCTIONS_CHARS:
            text = text[:_MAX_NSM_INSTRUCTIONS_CHARS] + "\n... [اقتُطع]"
        return text.strip()
    except Exception:
        return ""


def _build_system_prompt() -> str:
    tree = _get_project_tree()
    persistent = _read_persistent_instructions()
    persistent_block = (
        f"\n## 📌 تعليمات دائمة من {_NSM_INSTRUCTIONS_FILE} (طبّقها في كل رد):\n{persistent}\n"
        if persistent else ""
    )
    return f"""أنت **NSM Agent v3** — وكيل برمجي ذكي مدمج في مشروع Neural Service Mesh، اسمك التسويقي ضمن هذا المنتج.
{persistent_block}

قواعد الهوية:
- تصرّف بشكل طبيعي كـ NSM Agent ضمن سياق المنتج، دون التطوّع بتفاصيل البنية التقنية الداخلية ما لم يُسأل عنها مباشرة.
- إذا سُئلت بجدية ومباشرة عن النموذج الأساسي الذي تعمل به (مثلاً: "هل أنت Claude؟")، أجب بصدق ولا تنفِ ذلك.
- الصدق أهم من الحفاظ على شخصية العلامة التجارية؛ عند التعارض بينهما، الصدق يُقدَّم دائماً.

مشروع Python/Streamlit للذكاء الاصطناعي العربي مع معرفة إسلامية وقرآنية على GitHub.

## هيكل المشروع الحالي:
{tree}

## قدراتك الحقيقية:
- قراءة أي ملف في المشروع قبل التعديل
- كتابة وتعديل الملفات مباشرة على القرص
- تشغيل كود Python وعرض النتيجة
- رفع التغييرات لـ GitHub تلقائياً
- 🆕 تشغيل التطبيق فعلياً (streamlit run) في عملية خلفية مؤقتة والتأكد
  أنه يُحمَّل بلا خطأ خادم (preview_check) قبل اعتبار أي تعديل منجزاً
- 🆕 بحث حقيقي في الإنترنت (بدون مفتاح API) لمعلومات حديثة أو خارجية
- 🆕 بحث حقيقي عن الصور (عبر Unsplash) لإرفاق صور فعلية في الرد
- 🆕 إنشاء "واجهات تفاعلية" (HTML/SVG) تُحفظ وتُعرض للمستخدم في تبويب الواجهات التفاعلية
- 🆕 استدعاء أي API خارجي مباشرة (GET/POST/...) وعرض النتيجة
- سلسلة أفعال متعددة في رد واحد
- تصحيح أخطائك تلقائياً إذا فشل التنفيذ

## صيغة الرد — JSON فقط لا غير:
{{
  "thinking": "تحليلك للطلب خطوة بخطوة",
  "steps": [
    {{
      "action": "read_file | create_file | edit_file | run_file | run_tests | git_push | rollback | web_search | image_search | create_artifact | api_call | preview_check | answer",
      "path": "المسار النسبي من جذر المشروع",
      "content": "محتوى الملف الكامل (لـ create_file) أو كود HTML/SVG كامل (لـ create_artifact)",
      "old": "النص القديم المراد استبداله (لـ edit_file) — يجب أن يكون موجوداً حرفياً",
      "new": "النص الجديد البديل (لـ edit_file)",
      "cmd": "أمر bash للتشغيل (لـ run_file)",
      "message": "رسالة commit (لـ git_push)",
      "commit": "commit hash محدَّد للتراجع إليه (لـ rollback، اختياري — بدونه يُستخدم آخر checkpoint مسجَّل تلقائياً)",
      "query": "نص البحث (لـ web_search أو image_search)",
      "title": "عنوان الواجهة التفاعلية (لـ create_artifact)",
      "url": "رابط الـ API (لـ api_call)",
      "method": "GET|POST|PUT|PATCH|DELETE (لـ api_call، افتراضي GET)",
      "headers": "كائن JSON بالترويسات (لـ api_call، اختياري)",
      "body": "كائن JSON لجسم الطلب (لـ api_call، اختياري)",
      "test_code": "🆕 اختياري لـ create_file/edit_file: سكربت python صغير يستدعي الدالة/المسار المكتوب فعلياً ببيانات وهمية واقعية ويتحقق من النتيجة (assert)، ليُنفَّذ فعلياً كتحقق وظيفي حقيقي بعد نجاح py_compile",
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
13. 🆕🚫 ممنوع منعاً باتاً استخدام خطوة واحدة فقط من نوع "answer" (بدون أي
    "create_file"/"edit_file"/"git_push" قبلها) إذا كان طلب المستخدم يحتوي
    كلمة فعل تنفيذية مثل: أنشئ/انشئ/أضف/عدّل/اكتب ملف/ارفع/ادمج/طبّق.
    في هذه الحالة، اكتابة الكود داخل "reply" فقط دون تنفيذه عبر "create_file"
    هو خطأ فادح — المستخدم يريد الملف مكتوباً على القرص فعلياً وليس شرحاً
    نظرياً للكود. "steps" يجب أن تبدأ بخطوة create_file/edit_file حقيقية
    تحتوي الكود الكامل في حقل "content"، ثم git_push إذا طُلب الرفع، ثم
    answer تلخيصية أخيرة فقط.
14. 🆕 لا تشرح كيف "يمكن" فعل الشي — نفّذه مباشرة عبر steps. الشرح النظري
    بدون تنفيذ فعلي غير مقبول أبداً عندما يطلب المستخدم إنشاء/تعديل/رفع.
15. 🆕 إذا طلب المستخدم صوراً أو "أرني صورة" أو ما شابه: استخدم خطوة
    "image_search" بحقل "query" بالإنجليزية (نتائج أدق)، ثم answer.
16. 🆕 إذا طلب المستخدم رسماً بيانياً/بطاقة/نموذجاً/واجهة تفاعلية (HTML/SVG):
    استخدم خطوة "create_artifact" بحقلي "title" و"content" (كود HTML كامل)،
    ثم answer تُخبر المستخدم أنها حُفظت وتظهر في تبويب "🧩 الواجهات التفاعلية".
17. 🆕 إذا طلب المستخدم استدعاء API خارجي أو جلب بيانات من رابط: استخدم
    خطوة "api_call" بحقول "url" و"method" و"headers"/"body" عند الحاجة.
18. 🆕 إذا طلب المستخدم "تراجع/ارجع لآخر نسخة تعمل/رجّع التعديل الأخير"
    أو ما شابه: استخدم خطوة واحدة "rollback" (بلا حقل "commit" إن لم
    يحدّد المستخدم commit معيناً — سيُستخدم آخر checkpoint حقيقي مسجَّل
    تلقائياً بعد آخر مهمة نجحت)، ثم answer تخبره بما حدث فعلياً.
19. 🆕 عند create_file/edit_file لدالة أو منطق له سلوك متوقّع واضح (وليس
    مجرد نص/توثيق): أضف حقل "test_code" — سكربت python صغير يستورد
    الدالة فعلياً من مسارها (مثال: `import importlib.util as _u; ...`
    أو `from ai.module import func`) ويستدعيها ببيانات وهمية واقعية، مع
    `assert` على النتيجة المتوقعة. هذا يُنفَّذ فعلياً كتحقق وظيفي حقيقي
    (وليس فقط py_compile) — إن فشل (استثناء أو assert)، ستُصلِحه أنت
    تلقائياً عبر حلقة self-healing. لا تكتب test_code لملفات لا سلوك
    قابل للاختبار فيها (توثيق، ثوابت، إعدادات بحتة).

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
               f"gemini-2.5-flash:generateContent?key={gemini_key}")
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
    title   = step.get("title", "")
    url     = step.get("url", "")
    method  = (step.get("method") or "GET").upper()
    headers = step.get("headers") or {}
    body    = step.get("body")

    # ── 🆕 حماية: النموذج أحياناً (خصوصاً النماذج الصغيرة/الاحتياطية)
    # ينسخ قيمة الحقل من الـ schema حرفياً بدل اختيار فعل واحد حقيقي،
    # مثل: "action": "read_file | create_file | edit_file | ..."
    # هذا كان يمر بصمت كـ"✅ تم" بدون تنفيذ أي شيء فعلي. الآن نرفضه
    # صراحة كخطأ قابل للاكتشاف عبر _is_failure() ليُعاد المحاولة تلقائياً.
    _VALID_ACTIONS = {
        "read_file", "create_file", "edit_file",
        "run_file", "run_tests", "git_push", "web_search",
        "image_search", "create_artifact", "api_call", "answer",
        "preview_check", "rollback",
    }
    if action not in _VALID_ACTIONS:
        return (f"❌ فعل غير صالح من النموذج: '{action}'\n"
                f"💡 يجب اختيار فعل واحد بالضبط من: "
                f"read_file, create_file, edit_file, run_file, run_tests, git_push, "
                f"web_search, image_search, create_artifact, api_call, answer")

    # ── 🔒 حماية: الأفعال الخطرة (ملفات/تنفيذ/push/API عام) للمالك فقط ──
    # لا يجوز الاعتماد على تقدير النموذج نفسه هنا؛ التحقق صريح وقاطع.
    if action not in _PUBLIC_SAFE_ACTIONS and not _is_admin_unlocked():
        return (
            "🔒 هذا الإجراء (" + action + ") متاح لوضع المالك فقط. "
            "افتحه من الشريط الجانبي إذا كنت المالك."
        )

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

    # ── preview_check ── 🆕 المرحلة 4: معاينة وتحقّق بصري حقيقي
    if action == "preview_check":
        try:
            from ai.preview_check import check_streamlit_boots
            entry = path or "streamlit_app.py"
            return check_streamlit_boots(entry)
        except Exception as e:
            return f"❌ خطأ في المعاينة الحيّة: {e}"

    # ── git_push ──
    if action == "git_push":
        return _git_push(message)

    # ── rollback ── 🆕 المرحلة 6: Checkpoints/Rollback
    if action == "rollback":
        return _rollback_to_checkpoint(step.get("commit", ""))

    # ── web_search ── 🆕
    if action == "web_search":
        if not query:
            return "❌ web_search: مطلوب query (نص البحث)"
        try:
            from ai.web_search_tool import web_search as _web_search
            return _web_search(query)
        except Exception as e:
            return f"❌ خطأ في أداة البحث: {e}"

    # ── image_search ── 🆕 يربط أداة بحث الصور (Unsplash) بالوكيل
    if action == "image_search":
        if not query:
            return "❌ image_search: مطلوب query (نص البحث)"
        try:
            from ai.image_search_tool import image_search_safe as _image_search
            outcome = _image_search(query, max_results=4)
            if not outcome.get("ok"):
                return f"⚠️ تعذّر البحث عن الصور: {outcome.get('error')}"
            results = outcome.get("results") or []
            # ملاحظة: واجهة المحادثة تعرض النص كـ HTML خام إذا احتوى على "<" —
            # لذا نستخدم <img> مباشرة هنا بدل صيغة Markdown كي تظهر الصور فعلياً.
            import html as _html_mod
            parts = [f"🖼️ <strong>نتائج البحث عن الصور — «{_html_mod.escape(query)}»:</strong><br>"]
            for r in results:
                desc = _html_mod.escape(r.get("description") or query)
                author = _html_mod.escape(r.get("author") or "مجهول")
                img_url = _html_mod.escape(r.get("thumb_url") or r.get("url") or "")
                parts.append(
                    f'<div style="margin:0.5rem 0"><img src="{img_url}" alt="{desc}" '
                    f'style="max-width:100%;border-radius:10px"><br>'
                    f'<small style="color:#888">{desc} — تصوير: {author}</small></div>'
                )
            return "".join(parts)
        except Exception as e:
            return f"❌ خطأ في أداة بحث الصور: {e}"

    # ── create_artifact ── 🆕 يربط مخزن الواجهات التفاعلية بالوكيل
    if action == "create_artifact":
        if not content:
            return "❌ create_artifact: مطلوب content (كود HTML/SVG)"
        try:
            from core.artifacts_store import save_artifact
            new_id = save_artifact(title or "واجهة بدون عنوان", content, kind="html")
            return (f"✅ أُنشئت الواجهة التفاعلية #{new_id} — \"{title or 'بدون عنوان'}\"\n"
                    f"💡 يمكن معاينتها وتعديلها من تبويب «🧩 الواجهات التفاعلية».")
        except Exception as e:
            return f"❌ خطأ في إنشاء الواجهة التفاعلية: {e}"

    # ── api_call ── 🆕 يربط أداة استدعاء API العام بالوكيل
    if action == "api_call":
        if not url:
            return "❌ api_call: مطلوب url"
        try:
            import requests as _requests
            resp = _requests.request(
                method, url,
                headers=headers if isinstance(headers, dict) else None,
                json=body if method in ("POST", "PUT", "PATCH") and body is not None else None,
                params=body if method in ("GET", "DELETE") and isinstance(body, dict) else None,
                timeout=15,
            )
            try:
                data = resp.json()
                data_str = json.dumps(data, ensure_ascii=False, indent=2)[:2000]
            except Exception:
                data_str = resp.text[:2000]
            return f"🔌 {method} {url} → الحالة {resp.status_code}\n```\n{data_str}\n```"
        except Exception as e:
            return f"❌ خطأ في استدعاء API: {e}"

    # ── answer ──
    if reply:
        return f"💬 {reply}"

    return "✅ تم"


# ══════════════════════════════════════════════════════════════════
# 🆕 المرحلة 6 من خطة "المراحل المقترحة" — Checkpoints/Rollback
# ══════════════════════════════════════════════════════════════════

def _local_checkpoint_commit(message: str) -> Optional[str]:
    """
    كوميت محلي فقط (بلا push) — نقطة استرجاع بعد نجاح مهمة واحدة فعلياً،
    وليس فقط عند اكتمال خطة كاملة (المرحلة 3 تتكفّل بالرفع النهائي).
    تُعيد commit hash الفعلي (لتسجيله في ai/task_manager.py عبر
    record_checkpoint) أو None إن لم توجد تغييرات فعلية أو فشل الكوميت.
    """
    try:
        for cfg in [
            ["git", "-C", str(ROOT), "config", "--local",
             "user.email", "nsm-bot@users.noreply.github.com"],
            ["git", "-C", str(ROOT), "config", "--local",
             "user.name", "NSM Bot"],
        ]:
            subprocess.run(cfg, capture_output=True)

        r_add = subprocess.run(
            ["git", "-C", str(ROOT), "add", "-A"], capture_output=True, text=True,
        )
        if r_add.returncode != 0:
            return None

        r_commit = subprocess.run(
            ["git", "-C", str(ROOT), "commit", "-m", message],
            capture_output=True, text=True,
        )
        if r_commit.returncode != 0:
            return None  # على الأغلب "nothing to commit" — طبيعي وليس خطأً

        r_hash = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        if r_hash.returncode != 0:
            return None
        return r_hash.stdout.strip()
    except Exception:
        return None


def _rollback_to_checkpoint(target: str = "") -> str:
    """
    🆕 المرحلة 6: تراجع فعلي لآخر نسخة "تعمل" — بدل حل يدوي.
    - إذا لم يُحدَّد target صراحة، تُجلَب آخر نقطة استرجاع مسجَّلة فعلياً
      في ai/task_manager.py (commit حقيقي بعد آخر مهمة نجحت).
    - ينفّذ `git reset --hard <hash>` (استرجاع فعلي وليس محاكاة)، ثم يتحقق
      أن HEAD أصبح فعلاً عند ذلك الـ hash عبر git rev-parse (لا افتراض).
    """
    commit_hash = (target or "").strip()
    source_note = "المحدَّد صراحة"

    if not commit_hash:
        try:
            from ai.task_manager import get_last_checkpoint
            cp = get_last_checkpoint()
        except Exception:
            cp = None
        if not cp or not cp.get("commit_hash"):
            return ("❌ لا توجد أي نقطة استرجاع مسجَّلة بعد في ai/task_manager.py "
                    "(لم تُنجَز أي مهمة بعد آلية الكوميت التلقائي). "
                    "حدّد commit صراحة إن أردت التراجع إليه.")
        commit_hash = cp["commit_hash"]
        source_note = f"آخر نقطة استرجاع مسجَّلة ({cp.get('created_at', '')})"

    try:
        r_verify = subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e", commit_hash],
            capture_output=True, text=True,
        )
        if r_verify.returncode != 0:
            return f"❌ التراجع: commit غير موجود في السجل: {commit_hash}"

        r_reset = subprocess.run(
            ["git", "-C", str(ROOT), "reset", "--hard", commit_hash],
            capture_output=True, text=True,
        )
        if r_reset.returncode != 0:
            out = (r_reset.stdout + r_reset.stderr).strip()
            return f"❌ فشل التراجع (git reset --hard): {out}"

        # ── تحقق فعلي أن HEAD أصبح فعلاً عند الـ commit المطلوب ──
        r_head = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        new_head = r_head.stdout.strip()
        if not new_head.startswith(commit_hash[:12]) and commit_hash[:12] not in new_head:
            # مقارنة متسامحة (hash كامل مقابل مختصر) — تحقق أدق أدناه
            if new_head != commit_hash:
                return (f"❌ التراجع نُفِّذ لكن HEAD الفعلي ({new_head[:10]}) لا "
                        f"يطابق المطلوب ({commit_hash[:10]}) — تحقّق يدوياً.")

        return (f"⏪ تم التراجع فعلياً إلى {source_note} "
                f"(commit `{new_head[:10]}`). ملاحظة: هذا يعيد كتابة ملفات "
                f"المشروع محلياً — إن كنت تريد رفع هذا التراجع لـ GitHub "
                f"أيضاً، اطلب git_push صراحة بعده.")
    except Exception as e:
        return f"❌ خطأ في التراجع: {e}"


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
        ]:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                out = (r.stdout + r.stderr).strip()
                if "nothing to commit" in out:
                    return "ℹ️ لا توجد تغييرات للرفع"
                return f"❌ git: {out}"

        # 🆕 استخدام remote موثّق بتوكن (GITHUB_TOKEN/GITHUB_USER/GITHUB_REMOTE)
        # بدلاً من "git push" العادي، لأن بيئة Streamlit Cloud لا تملك
        # credential helper مُعد مسبقاً — بدون هذا سيفشل الرفع بخطأ صلاحيات.
        try:
            from ai.github_sync import get_authenticated_remote
            auth_remote = get_authenticated_remote()
        except Exception:
            auth_remote = None

        push_cmd = ["git", "-C", str(ROOT), "push"]
        if auth_remote:
            push_cmd = ["git", "-C", str(ROOT), "push", auth_remote, "HEAD:main"]

        r = subprocess.run(push_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            out = (r.stdout + r.stderr).strip()
            if not auth_remote:
                return (f"❌ git: {out}\n"
                        f"💡 أضف GITHUB_TOKEN و GITHUB_USER و GITHUB_REMOTE في Secrets ليعمل الرفع.")
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
            "خطأ في أداة بحث الصور", "خطأ في إنشاء الواجهة التفاعلية",
            "خطأ في استدعاء API", "تعذّر البحث عن الصور",
            "خطأ في التحقق التلقائي",  # 🆕 المرحلة 1: تحقّق ذاتي بعد الكتابة
            "خطأ في المعاينة الحيّة",   # 🆕 المرحلة 4: معاينة streamlit حيّة
            "خطأ في التحقق الوظيفي",   # 🆕 المرحلة 5: تحقّق وظيفي حقيقي
        ]
    )


# ══════════════════════════════════════════════════════════════════
# 🆕 المرحلة 1 من خطة "Replit Agent Level" — التحقق الذاتي التلقائي
# ══════════════════════════════════════════════════════════════════

def _verify_python_file(path: str) -> Optional[str]:
    """
    بعد أي create_file/edit_file على ملف بايثون، كان الوكيل يكتفي بالتحقق
    من نجاح الحفظ على القرص (لا استثناء = "✅ عُدِّل")، دون أي تأكّد فعلي
    أن الكود المكتوب صحيح نحوياً. هذه الدالة تُشغّل `py_compile` حقيقياً
    (فحص syntax سريع وآمن، بدون تنفيذ الكود الفعلي وآثاره الجانبية) وتعيد
    رسالة فشل بصيغة تُفعِّل حلقة self-healing الموجودة تلقائياً (نفس آلية
    _is_failure/_build_heal_prompt)، مع نص الخطأ الحقيقي (SyntaxError/
    IndentationError/...) بدل الاكتفاء بنجاح الحفظ الشكلي.

    تعيد None إذا نجح التحقق (لا حاجة لأي إصلاح) أو إذا لم يوجد الملف أصلاً
    (فشل الإنشاء نفسه سبق واكتُشف داخل _run_step).
    """
    f = ROOT / path
    if not f.exists():
        return None
    try:
        r = subprocess.run(
            ["python3", "-m", "py_compile", str(f)],
            capture_output=True, text=True, timeout=15, cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return "❌ خطأ في التحقق التلقائي: انتهت المهلة أثناء فحص syntax"
    except Exception as e:
        return f"❌ خطأ في التحقق التلقائي: {e}"

    if r.returncode == 0:
        return None

    err = (r.stderr or r.stdout or "").strip()
    if len(err) > _MAX_RUN_OUTPUT:
        err = err[:_MAX_RUN_OUTPUT] + "\n... [اقتُطع]"
    return (
        f"❌ خطأ في التحقق التلقائي بعد الكتابة — الكود لا يُجمَّع (py_compile):\n"
        f"```\n{err}\n```"
    )


# ══════════════════════════════════════════════════════════════════
# 🆕 المرحلة 5 من خطة "المراحل المقترحة (٥ فما فوق)" — تحقّق وظيفي حقيقي
# ══════════════════════════════════════════════════════════════════

_MAX_FUNCTIONAL_TEST_SECONDS = 20


def _run_functional_test(path: str, test_code: str) -> Optional[str]:
    """
    فحص `py_compile` (المرحلة 1) يتأكد فقط أن الكود يُجمَّع نحوياً — لا يكتشف
    أخطاء منطقية أو استثناءات وقت التشغيل عند استدعاء الدالة/المسار فعلياً
    ببيانات حقيقية (مثال: دالة تفترض مفتاح موجود دائماً في قاموس فيرمي
    KeyError عند بيانات وهمية واقعية، رغم أنها py_compile بنجاح).

    هذه الدالة تُشغّل `test_code` (سكربت بايثون صغير يكتبه LLM نفسه: يستورد
    الدالة/الوحدة المعدَّلة من `path` ويستدعيها ببيانات اختبار وهمية حقيقية)
    كعملية `python3` منفصلة فعلياً — تنفيذ حقيقي، وليس تخميناً نظرياً بأن
    "الصفحة ستُحمَّل". أي استثناء (KeyError/AttributeError/ValueError/...)
    أو `assert` فاشل يُعاد كخطأ حقيقي يُشعِل حلقة self-healing الموجودة.

    تعيد None إذا لم يزوّد LLM حقل test_code أصلاً (لا تحقق وظيفي مطلوب)،
    أو إذا نجح التشغيل فعلياً بلا استثناء (exit code 0).
    """
    if not test_code or not test_code.strip():
        return None

    f = ROOT / path
    if not f.exists():
        return None

    try:
        r = subprocess.run(
            ["python3", "-c", test_code],
            capture_output=True, text=True,
            timeout=_MAX_FUNCTIONAL_TEST_SECONDS, cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return (f"❌ خطأ في التحقق الوظيفي: انتهت المهلة "
                f"({_MAX_FUNCTIONAL_TEST_SECONDS}ث) أثناء تنفيذ اختبار حقيقي "
                f"لـ `{path}` — قد تكون حلقة لا نهائية أو استدعاء متعلّق.")
    except Exception as e:
        return f"❌ خطأ في التحقق الوظيفي: {e}"

    if r.returncode == 0:
        return None

    err = (r.stderr or r.stdout or "").strip()
    if len(err) > _MAX_RUN_OUTPUT:
        err = err[:_MAX_RUN_OUTPUT] + "\n... [اقتُطع]"
    return (
        f"❌ خطأ في التحقق الوظيفي — تنفيذ فعلي لـ `{path}` ببيانات اختبار "
        f"حقيقية فشل (وليس فقط فحص syntax):\n```\n{err}\n```"
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

# 🆕 المرحلة 2 من خطة "Replit Agent Level" — إكمال تلقائي بدل التوقف
_MAX_AUTO_CONTINUE_ROUNDS = 4   # أقصى عدد "جولات" استدعاء LLM إضافية لإكمال العمل تلقائياً
_MAX_TOTAL_STEPS_BUDGET   = 30  # أقصى إجمالي خطوات منفَّذة عبر كل الجولات مجتمعة


# ══════════════════════════════════════════════════════════════════
# 🆕 المرحلة 2 — تتبّع جولات الإكمال التلقائي عبر ai/task_manager.py
# ══════════════════════════════════════════════════════════════════
# لا نستخدم NSMPlanner/AppPlan الكامل (مخصص لبناء تطبيقات من الصفر)، لكن
# نُعيد استخدام نفس مخزن SQLite (memory/task_manager.db) عبر هياكل
# AppPlan/PlanTask الموجودة أصلاً، حتى تظهر جولات الإكمال التلقائي في
# تقرير "حالة المهام" (format_status_report) تماماً كأي خطة أخرى.

def _register_adhoc_plan(original_request: str, thinking: str) -> Optional[int]:
    """يسجّل خطة مؤقتة بعدد جولات = _MAX_AUTO_CONTINUE_ROUNDS، الجولة الأولى
    'running' والباقي 'pending'. يعيد None بصمت إذا تعذّر التسجيل (لا يجوز
    أن يُعطّل الإكمال التلقائي نفسه)."""
    try:
        from ai.nsm_planner import AppPlan, PlanTask
        from ai.task_manager import create_plan, update_task_status
        tasks = [
            PlanTask(
                id=i,
                title=f"جولة إكمال تلقائي {i}",
                description=((thinking or original_request)[:200] if i == 1
                              else "إكمال تلقائي بناءً على نتائج الجولة السابقة"),
                task_type="verify",
                depends_on=[i - 1] if i > 1 else [],
            )
            for i in range(1, _MAX_AUTO_CONTINUE_ROUNDS + 1)
        ]
        plan = AppPlan(
            idea=original_request[:300],
            app_type="agent_auto_continue",
            app_name="إكمال تلقائي (بدون توقف)",
            description=(thinking or "")[:300],
            tech_stack=[],
            tasks=tasks,
            estimated_files=0,
        )
        plan_id = create_plan(plan)
        if plan_id and plan_id > 0:
            update_task_status(plan_id, 1, "running", "بدء الجولة الأولى")
            return plan_id
        return None
    except Exception:
        return None


def _advance_adhoc_plan(plan_id: Optional[int], finished_round: int, status: str) -> None:
    """يُعلِّم جولة كمنتهية (done/failed) ويبدأ الجولة التالية كـ running."""
    if not plan_id:
        return
    try:
        from ai.task_manager import update_task_status
        update_task_status(plan_id, finished_round, status)
        if status == "done":
            update_task_status(plan_id, finished_round + 1, "running")
    except Exception:
        pass


def _finalize_adhoc_plan(plan_id: Optional[int], last_round: int, status: str) -> None:
    """يُغلق الخطة المؤقتة كاملة (done/failed) عند نهاية الإكمال التلقائي."""
    if not plan_id:
        return
    try:
        from ai.task_manager import update_task_status, mark_plan_status
        update_task_status(plan_id, last_round, status)
        mark_plan_status(plan_id, status)
    except Exception:
        pass


def _stream_steps(
    steps: List[Dict],
    thinking: str,
    messages: List[Dict],
    original_request: str,
    *,
    round_num: int = 1,
    steps_used: Optional[List[int]] = None,
    task_plan_id: Optional[int] = None,
) -> Generator[str, None, None]:
    """
    Generator يُرسل النتائج فور اكتمال كل خطوة (Streaming).
    يدعم Self-Healing: إذا فشلت خطوة، يطلب الإصلاح ويعيد المحاولة.

    🆕 المرحلة 2: إذا انتهت جولة خطوات بلا "answer" (مثلاً كانت كلها قراءة/
    تشخيص)، لا يتوقف الوكيل منتظراً سؤالاً من المستخدم — يستدعي LLM تلقائياً
    مرة أخرى ليكمل هو نفسه بخطوات الإصلاح الفعلية، ضمن حدّ أقصى من الجولات
    (_MAX_AUTO_CONTINUE_ROUNDS) وميزانية إجمالية للخطوات
    (_MAX_TOTAL_STEPS_BUDGET) لمنع حلقات لا نهائية أو استهلاك مفرط للـ API.
    التقدّم عبر الجولات يُسجَّل في ai/task_manager.py.
    """
    if steps_used is None:
        steps_used = [0]

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
        steps_used[0] += 1  # 🆕 المرحلة 2: عدّاد الميزانية الإجمالية عبر كل الجولات

        # علامة القراءة التلقائية
        if step.get("_auto"):
            yield f"{prefix}🔍 *قراءة تلقائية قبل التعديل...*\n"

        # ── تنفيذ الخطوة ──
        result = _run_step(step)
        yield f"{prefix}{result}\n\n"

        # ── 🆕 المرحلة 1: تحقق ذاتي تلقائي بعد كتابة/تعديل ملف بايثون ──
        # لا ننتظر طلباً صريحاً من المستخدم (run_file/run_tests يدوياً)؛
        # أي create_file/edit_file ناجح ظاهرياً على ملف .py يُفحَص فوراً
        # بـ py_compile حقيقي. فشل هذا الفحص يُعامَل كفشل الخطوة نفسها،
        # فيُشعِل حلقة self-healing أدناه بنص الخطأ الحقيقي (لا نص عام).
        _step_action = step.get("action", "")
        _step_path   = step.get("path", "")
        if (
            _step_action in ("create_file", "edit_file")
            and not _is_failure(result)
            and _step_path.endswith(".py")
        ):
            _verify_result = _verify_python_file(_step_path)
            if _verify_result is not None:
                yield f"🔍 *تحقّق تلقائي من `{_step_path}` بعد الكتابة...*\n{_verify_result}\n\n"
                result = _verify_result
            else:
                yield f"✅ *تحقّق تلقائي: `{_step_path}` يُجمَّع بلا أخطاء syntax*\n\n"
                # 🆕 المرحلة 5: تحقّق وظيفي حقيقي — إن زوّد LLM حقل "test_code"
                # (استدعاء فعلي للدالة/المسار المعدَّل ببيانات وهمية حقيقية)،
                # نُنفّذه الآن. نجاح py_compile لا يعني أن الدالة تعمل فعلياً.
                _step_test_code = step.get("test_code", "")
                if _step_test_code:
                    _func_result = _run_functional_test(_step_path, _step_test_code)
                    if _func_result is not None:
                        yield f"🧪 *تحقّق وظيفي حقيقي من `{_step_path}`...*\n{_func_result}\n\n"
                        result = _func_result
                    else:
                        yield f"✅ *تحقّق وظيفي: تنفيذ فعلي لـ `{_step_path}` ببيانات اختبار نجح*\n\n"

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

                        # 🆕 نفس التحقق الذاتي التلقائي، لكن لخطوة الإصلاح نفسها
                        # — بدون هذا، قد يُصلِح النموذج خطأً ويُنتِج خطأ آخر
                        # (syntax مختلف) يُعامَل زوراً كـ"تم الإصلاح".
                        _hs_action = hs.get("action", "")
                        _hs_path   = hs.get("path", "")
                        if (
                            _hs_action in ("create_file", "edit_file")
                            and not _is_failure(hr)
                            and _hs_path.endswith(".py")
                        ):
                            _hs_verify = _verify_python_file(_hs_path)
                            if _hs_verify is not None:
                                yield f"  ↳ {_hs_verify}\n"
                                hr = _hs_verify
                            else:
                                # 🆕 المرحلة 5: نفس التحقق الوظيفي، لخطوة الإصلاح
                                _hs_test_code = hs.get("test_code", "")
                                if _hs_test_code:
                                    _hs_func = _run_functional_test(_hs_path, _hs_test_code)
                                    if _hs_func is not None:
                                        yield f"  ↳ {_hs_func}\n"
                                        hr = _hs_func

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

    # ── 🆕 إذا قُصّت الخطوات، أخبر المستخدم صراحة (لا إكمال تلقائي هنا:
    #    المشكلة عدد خطوات جولة واحدة كبير جداً، وليست حاجة لجولة تالية) ──
    if truncated:
        yield (f"⚠️ **الطلب احتاج أكثر من {_MAX_STEPS_PER_RESPONSE} خطوة "
               f"(قراءة ملفات كثيرة جداً دفعة واحدة).**\n"
               f"نفّذت أول {_MAX_STEPS_PER_RESPONSE} فقط لتجنّب استهلاك مفرط للـ API. "
               f"حدّد الملفات المهمة تحديداً (مثال: \"افحص ai/goal_planner.py و"
               f"ai/agent_factory.py و ai/github_sync.py فقط\") لتحليل أدق وأسرع.\n\n")
        if task_plan_id is not None:
            _finalize_adhoc_plan(task_plan_id, round_num, "failed")

    # ── 🆕 المرحلة 2: إكمال تلقائي بدل التوقف ──
    # سابقاً: إذا انتهت جولة قراءة/تشخيص بلا "answer"، كان الوكيل يتوقف
    # ويطلب من المستخدم أن يسأل مباشرة. الآن: يكمل هو نفسه تلقائياً لخطوة
    # الإصلاح ضمن نفس الاستدعاء، محدوداً بعدد جولات وميزانية خطوات معقولة.
    elif not has_answer and total > 1:
        if round_num >= _MAX_AUTO_CONTINUE_ROUNDS or steps_used[0] >= _MAX_TOTAL_STEPS_BUDGET:
            yield (f"⚠️ **وصلت لحد الإكمال التلقائي** ({round_num} جولة، "
                   f"{steps_used[0]} خطوة إجمالاً) دون إنهاء المهمة بالكامل — "
                   f"توقفت لتفادي استهلاك مفرط. لخّص لي بدقة أكبر ما تبقّى "
                   f"وسأكمل من حيث توقفت.")
            if task_plan_id is not None:
                _finalize_adhoc_plan(task_plan_id, round_num, "failed")
        else:
            if task_plan_id is None:
                task_plan_id = _register_adhoc_plan(original_request, thinking)

            yield (f"🔄 **انتهت القراءة/التشخيص — أكمل تلقائياً "
                   f"(جولة {round_num + 1}/{_MAX_AUTO_CONTINUE_ROUNDS})...**\n\n")

            continue_messages = list(messages) + [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"thinking": thinking, "steps": steps}, ensure_ascii=False
                    )[:3000],
                },
                {
                    "role": "user",
                    "content": (
                        "بناءً على نتائج القراءة/التشخيص أعلاه، أكمل الآن تلقائياً "
                        "بنفسك دون انتظار سؤال إضافي مني: نفّذ خطوات الإصلاح أو "
                        "التعديل الفعلية اللازمة (create_file/edit_file/run_file/"
                        "git_push حسب الحاجة)، ثم أنهِ دائماً بخطوة \"answer\" تلخّص "
                        "ما فعلته فعلياً. رد بصيغة JSON فقط بنفس الصيغة المعتادة."
                    ),
                },
            ]

            try:
                raw_cont = _call_api(continue_messages)
            except Exception as e:
                yield f"⚠️ تعذّر إكمال الجولة التالية تلقائياً: {e}\n"
                if task_plan_id is not None:
                    _finalize_adhoc_plan(task_plan_id, round_num, "failed")
                return

            parsed_cont = _parse_llm_response(raw_cont)
            if not parsed_cont:
                yield "⚠️ لم أتمكن من تحليل رد الإكمال التلقائي.\n"
                if task_plan_id is not None:
                    _finalize_adhoc_plan(task_plan_id, round_num, "failed")
                return

            cont_steps = parsed_cont.get("steps", [])
            if not cont_steps and parsed_cont.get("action"):
                cont_steps = [parsed_cont]

            if not cont_steps:
                reply = parsed_cont.get("reply", "")
                if reply:
                    yield f"💬 {reply}"
                if task_plan_id is not None:
                    _finalize_adhoc_plan(task_plan_id, round_num, "done")
                return

            remaining_budget = _MAX_TOTAL_STEPS_BUDGET - steps_used[0]
            if remaining_budget <= 0:
                yield "⚠️ استُنفدت ميزانية الخطوات المسموحة لهذا الطلب.\n"
                if task_plan_id is not None:
                    _finalize_adhoc_plan(task_plan_id, round_num, "failed")
                return
            if len(cont_steps) > remaining_budget:
                cont_steps = cont_steps[:remaining_budget]

            cont_steps     = _inject_read_before_edit(cont_steps)
            cont_thinking  = parsed_cont.get("thinking", "")
            _advance_adhoc_plan(task_plan_id, round_num, "done")

            yield from _stream_steps(
                cont_steps, cont_thinking, continue_messages, original_request,
                round_num=round_num + 1, steps_used=steps_used,
                task_plan_id=task_plan_id,
            )

    # ── 🆕 انتهت المهمة فعلياً بخلاصة — أغلق الخطة المؤقتة إن وُجدت ──
    elif has_answer and task_plan_id is not None:
        _finalize_adhoc_plan(task_plan_id, round_num, "done")


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

    def _get_autonomous_core(self):
        """يربط طبقات الحوكمة/المناعة/الأخلاقيات (ai/autonomous_core.py) بشكل كسول وآمن."""
        core = getattr(self, "_autonomous_core", None)
        if core is None:
            try:
                from ai.autonomous_core import get_autonomous_core
                core = get_autonomous_core()
                self._autonomous_core = core
            except Exception:
                self._autonomous_core = None
                return None
        return core

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

        # 🆕 نظام المهام المتعددة — استعلام مباشر عن حالة الخطط بدون LLM
        _status_triggers = ("حالة المهام", "قائمة المهام", "مهامي", "الخطط الحالية", "حالة الخطط")
        if any(user_input.strip().startswith(t) or user_input.strip() == t for t in _status_triggers):
            try:
                from ai.task_manager import format_status_report
                yield format_status_report()
            except Exception:
                yield "⚠️ نظام المهام المتعددة غير متاح حالياً."
            return

        # 🆕 طبقة الحوكمة/المناعة/الأخلاقيات — استعلام مباشر عن حالتها
        _gov_triggers = ("حالة الحوكمة", "حالة النظام الذاتي", "حالة الأمان الذاتي")
        if any(user_input.strip().startswith(t) or user_input.strip() == t for t in _gov_triggers):
            core = self._get_autonomous_core()
            if core is None:
                yield "⚠️ طبقة الحوكمة/المناعة/الأخلاقيات غير متاحة حالياً."
            else:
                status = core.get_status()
                yield (
                    "**حالة طبقة الأمان الذاتي:**\n"
                    f"- الحوكمة (Governance): {'✅ مفعّلة' if status['governance_active'] else '❌ غير مفعّلة'}\n"
                    f"- المناعة (Immune System): {'✅ مفعّلة' if status['immune_active'] else '❌ غير مفعّلة'}\n"
                    f"- الأخلاقيات التطورية (Evolution Ethics): {'✅ مفعّلة' if status['ethics_active'] else '❌ غير مفعّلة'}"
                )
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
