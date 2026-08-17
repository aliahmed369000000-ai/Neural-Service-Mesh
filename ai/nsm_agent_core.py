"""
NSM Agent Core — ai/nsm_agent_core.py  (v3 — Replit Agent Level)
=================================================================
الجديد في v3:

✅ [v2] يقرأ الملفات قبل التعديل
✅ [v2] يُشغّل الكود ويرى النتيجة
✅ [v2] هيكل المشروع الديناميكي في كل طلب
✅ [v2] multi-step في رد واحد
✅ [v2] fallback: Groq → Cloudflare → Gemini → OpenRouter

🆕 [v3] Streaming بحرف بحرف — Generator يرسل النتائج فور اكتمال كل خطوة
🆕 [v3] Self-Healing Loop — يصحح أخطاءه تلقائياً (حتى 3 محاولات)
🆕 [v3] Read-Before-Edit تلقائي — إذا طُلب edit_file بدون read_file سابق،
         يقرأ الملف أولاً تلقائياً ثم ينفذ التعديل في نفس الدورة
"""

from __future__ import annotations

import json
import os
import re
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
_PUBLIC_SAFE_ACTIONS = {"answer", "web_search", "image_search", "system_info", "fetch_url", "deep_research", "trending", "will_status"}
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

# 🆕 تحسين أداء: _get_project_tree() كانت تُشغَّل بالكامل (rglob + stat على
# كل ملف — قياس فعلي: ~50ms لكل استدعاء على 749 ملفاً حالياً، وتكبر مع
# نمو المشروع) في *كل* رسالة محادثة عبر _build_system_prompt()، حتى لو
# كانت الرسالة سؤالاً بسيطاً لا يلمس أي ملف (action="answer"). هذه ذاكرة
# تخزين مؤقت قصيرة العمر (TTL) تُلغى تلقائياً فور أي كتابة فعلية على القرص
# (create_file/edit_file ناجحين — انظر _invalidate_tree_cache) حتى تبقى
# الشجرة المعروضة للـLLM محدَّثة دائماً بلا مسح متكرر لا داعي له.
_TREE_CACHE_TTL_SECONDS = 30.0
_tree_cache: Dict[str, Any] = {"text": None, "ts": 0.0}


def _invalidate_tree_cache() -> None:
    """تُستدعى فور أي create_file/edit_file ناجح كي لا تُعرَض شجرة قديمة."""
    _tree_cache["text"] = None
    _tree_cache["ts"] = 0.0


def _get_project_tree() -> str:
    now = time.time()
    if _tree_cache["text"] is not None and (now - _tree_cache["ts"]) < _TREE_CACHE_TTL_SECONDS:
        return _tree_cache["text"]

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

    text = "\n".join(lines[:80])
    _tree_cache["text"] = text
    _tree_cache["ts"] = now
    return text


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
    from ai.methodology_engine import method_principles_prompt
    tree = _get_project_tree()
    persistent = _read_persistent_instructions()
    persistent_block = (
        f"\n## 📌 تعليمات دائمة من {_NSM_INSTRUCTIONS_FILE} (طبّقها في كل رد):\n{persistent}\n"
        if persistent else ""
    )
    method_block = method_principles_prompt()
    return f"""أنت **NSM Agent v3** — وكيل برمجي ذكي مدمج في مشروع Neural Service Mesh، اسمك التسويقي ضمن هذا المنتج.
{persistent_block}
{method_block}

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
- 🆕 مراقبة Kaggle: فحص حالة كيرنل التدريب (status + آخر المخرجات)
  عبر kaggle_status — الكيرنل النشط: aliahmedmo/nsm-surahchain-xlarge-12h-fix7
- 🆕 التحقق الفعلي بعد الرفع: push يُتبع تلقائياً بـ git ls-remote
  لمطابقة SHA المحلي مع البعيد — لا تُعلم بنجاح الرفع دون هذا التحقق

## صيغة الرد — JSON فقط لا غير:
{{
  "thinking": "تحليلك للطلب خطوة بخطوة",
  "steps": [
    {{
      "action": "read_file | create_file | edit_file | run_file | run_tests | git_push | git_lfs | git_info | search_code | find_files | py_compile | fetch_url | system_info | run_safe | rollback | kaggle_status | web_search | deep_research | self_learn | ingest_knowledge | trending | will_status | will_tick | terminal | image_search | create_artifact | api_call | preview_check | answer",
      "slug": "slug الكيرنل بصيغة username/kernel-slug (لـ kaggle_status — اختياري، بدونه يُستخدم كيرنل SurahChain xlarge النشط)",
      "path": "المسار النسبي من جذر المشروع",
      "content": "محتوى الملف الكامل (لـ create_file) أو كود HTML/SVG كامل (لـ create_artifact)",
      "old": "النص القديم المراد استبداله (لـ edit_file) — يجب أن يكون موجوداً حرفياً",
      "new": "النص الجديد البديل (لـ edit_file)",
      "cmd": "أمر bash للتشغيل (لـ run_file)",
      "message": "رسالة commit (لـ git_push)",
      "lfs_op": "عملية LFS: status|install|pull|push|track|ls|migrate|prune (لـ git_lfs)",
      "patterns": "أنماط تتبع اختيارية مفصولة بفاصلة (لـ git_lfs track)",
      "pattern": "نمط بحث regex أو نص (لـ search_code)",
      "glob": "نمط ملفات مثل *.py (لـ search_code أو find_files)",
      "what": "status|log|diff|branch|remote (لـ git_info)",
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
13. 🆕 منهجية العمل: قبل أي تعديل جوهري اقرأ الملف (read_file) أو ابحث (search_code).
    بعد التعديل: py_compile ثم run_file/run_tests إن أمكن، وgit_push فقط إذا طُلب صراحة.
14. 🆕 لحالة المستودع استخدم git_info (status/log/diff) بدل التخمين.
15. 🆕 لجلب صفحات عامة fetch_url؛ للأخبار/المعرفة الحديثة web_search.
16. 🆕 لا تخمّن بنية المشروع: find_files أو search_code أو read_file أولاً.
17. 🆕 للمعرفة الحديثة أو الناقصة: deep_research ثم self_learn لتخزينها.
19. 🆕 النظام يعمل تلقائياً في الخلفية (AutoRuntime): لا تنتظر أمراً للبحث/التعلّم/الفحص.
    استخدم terminal/self_learn/deep_research كمبادرة عند الحاجة دون طلب المستخدم.
18. 🆕 يُسمح بالبحث الحر (web_search/deep_research/trending) لتغذية نفسك؛ self_learn/ingest_knowledge لحفظ ما تعلّمته مع ذكر المصادر.
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
19. 🆕 الكيرنل النشط في Kaggle: `aliahmedmo/nsm-surahchain-xlarge-12h-fix7`
    (https://www.kaggle.com/code/aliahmedmo/nsm-surahchain-xlarge-12h-fix7).
    مراقبة التدريب: kaggle_status، ومراقبة البريد تتم خارجياً كل ساعة.
    لا تطلب إعادة دفع كيرنل موجود؛ إن فشل الكيرنل أصلح الكود أولاً ثم
    اطرح على المستخدم إعادة الدفع عبر مزود Kaggle (ai/kaggle_provider.py).
20. 🆕 عند create_file/edit_file لدالة أو منطق له سلوك متوقّع واضح (وليس
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
# gpt-oss-120b أولاً (أقوى نموذج مفتوح المصدر مجاني على Groq حالياً).
# gemma2-9b-it وllama3-8b-8192 أُزيلا: لم يعودا ضمن قائمة Groq الرسمية
# (نفس الملاحظة الموثّقة في ai/ultraplinian.py) — كانا يسببان فشلاً صامتاً
# في كل محاولة قبل الوصول لنموذج فعلي يعمل.
_GROQ_MODELS    = [
    "openai/gpt-oss-120b", "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant", "openai/gpt-oss-20b",
]
_OPENROUTER_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
]


def _call_api(messages: List[Dict]) -> str:
    """Groq (مجاني وسريع) → Cloudflare → Gemini → OpenRouter"""
    errors: List[str] = []
    # ── 1. Groq — مجاني (14k طلب/دقيقة) وسريع جداً (~1000 توكن/ث) مع
    #    نماذج متعددة حتى استُنفدت الحصة أو حُجب النموذج، ثم ينتقل
    #    تلقائياً للمجانيات الأخرى (Cloudflare/Gemini/OpenRouter).
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
    # ── 2. Cloudflare Workers AI ──
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
        "run_file", "run_tests", "git_push", "git_lfs", "git_info",
        "search_code", "find_files", "py_compile", "fetch_url",
        "system_info", "run_safe", "web_search", "deep_research",
        "self_learn", "ingest_knowledge", "trending", "will_status", "will_tick", "terminal",
        "image_search", "create_artifact", "api_call", "answer",
        "preview_check", "rollback", "kaggle_status",
    }
    if action not in _VALID_ACTIONS:
        return (f"❌ فعل غير صالح من النموذج: '{action}'\n"
                f"💡 يجب اختيار فعل واحد بالضبط من: "
                f"read_file, create_file, edit_file, run_file, run_tests, git_push, git_lfs, "
                f"git_info, search_code, find_files, py_compile, fetch_url, system_info, run_safe, "
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
            _invalidate_tree_cache()
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
            _invalidate_tree_cache()
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

    # ── search_code ── 🆕 بحث في كود المشروع
    if action == "search_code":
        try:
            from ai.agent_tools import search_code as _sc, format_tool_result
            pat = step.get("pattern") or step.get("query") or ""
            if not pat:
                return "❌ search_code: مطلوب pattern أو query"
            res = _sc(pat, path=path or ".", glob=step.get("glob") or "*.py")
            return format_tool_result("🔎 search_code", res)
        except Exception as e:
            return f"❌ search_code: {e}"

    # ── find_files ──
    if action == "find_files":
        try:
            from ai.agent_tools import find_files as _ff, format_tool_result
            res = _ff(step.get("glob") or step.get("pattern") or "*.py", path=path or ".")
            return format_tool_result("📁 find_files", res)
        except Exception as e:
            return f"❌ find_files: {e}"

    # ── git_info ──
    if action == "git_info":
        try:
            from ai.agent_tools import git_info as _gi, format_tool_result
            res = _gi(step.get("what") or step.get("cmd") or "status")
            return format_tool_result("🌿 git_info", res)
        except Exception as e:
            return f"❌ git_info: {e}"

    # ── py_compile ──
    if action == "py_compile":
        try:
            from ai.agent_tools import py_compile_check, format_tool_result
            if not path:
                return "❌ py_compile: مطلوب path"
            return format_tool_result("🧪 py_compile", py_compile_check(path))
        except Exception as e:
            return f"❌ py_compile: {e}"

    # ── fetch_url ──
    if action == "fetch_url":
        try:
            from ai.agent_tools import fetch_url as _fu, format_tool_result
            u = step.get("url") or query or ""
            if not u:
                return "❌ fetch_url: مطلوب url"
            return format_tool_result("🌐 fetch_url", _fu(u))
        except Exception as e:
            return f"❌ fetch_url: {e}"

    # ── system_info ──
    if action == "system_info":
        try:
            from ai.agent_tools import system_info as _si, format_tool_result
            return format_tool_result("🖥️ system_info", _si())
        except Exception as e:
            return f"❌ system_info: {e}"

    # ── run_safe ──
    if action == "run_safe":
        try:
            from ai.agent_tools import run_safe_cmd, format_tool_result
            c = step.get("cmd") or ""
            if not c:
                return "❌ run_safe: مطلوب cmd"
            return format_tool_result("⚙️ run_safe", run_safe_cmd(c))
        except Exception as e:
            return f"❌ run_safe: {e}"

    # ── git_lfs ── 🆕 قدرات Git LFS كاملة
    if action == "git_lfs":
        return _git_lfs_action(step)

    # ── git_push ──
    if action == "git_push":
        return _git_push(message)

    # ── rollback ── 🆕 المرحلة 6: Checkpoints/Rollback
    if action == "rollback":
        return _rollback_to_checkpoint(step.get("commit", ""))

    # ── kaggle_status ── 🆕 مراقبة Kaggle (نفس منهجية المراقبة الدورية)
    if action == "kaggle_status":
        try:
            from ai.agent_tools import kaggle_status as _ks
            slug = step.get("slug") or step.get("kernel") or ""
            import json as _json
            res = _ks(slug)
            return "## 🟢 kaggle_status\n```json\n" + _json.dumps(res, ensure_ascii=False, indent=2) + "\n```"
        except Exception as e:
            return f"❌ kaggle_status: {e}"

    # ── terminal ── 🆕 طرفية NSM للوكلاء
    # ── auto_status ── حالة التشغيل التلقائي
    if action == "auto_status":
        try:
            from ai.auto_runtime import get_auto_runtime
            import json as _json
            rt = get_auto_runtime(start=True)
            return "## 🤖 auto_status\n```json\n" + _json.dumps(rt.status(), ensure_ascii=False, indent=2) + "\n```"
        except Exception as e:
            return f"❌ auto_status: {e}"

    if action == "terminal":
        try:
            from ai.nsm_terminal import get_terminal
            c = step.get("cmd") or step.get("command") or ""
            if not c:
                return "❌ terminal: مطلوب cmd"
            mode = step.get("mode") or "safe"
            # الوكلاء الافتراضي safe؛ admin فقط مع وضع المالك
            if mode == "admin" and not _is_admin_unlocked():
                mode = "safe"
            timeout = int(step.get("timeout") or 45)
            r = get_terminal().run(c, mode=mode, timeout=timeout)
            return f"## 💻 terminal\n```\n{r.formatted()}\n```"
        except Exception as e:
            return f"❌ terminal: {e}"

    # ── will_status / will_tick ── إرادة ذاتية
    if action == "will_status":
        try:
            from ai.autonomous_will import get_autonomous_will
            import json as _json
            w = get_autonomous_will(start=True)
            return "## 🧬 will_status\n```json\n" + _json.dumps({"status": w.status(), "recent": w.recent_actions(5)}, ensure_ascii=False, indent=2) + "\n```"
        except Exception as e:
            return f"❌ will_status: {e}"

    if action == "will_tick":
        try:
            from ai.autonomous_will import get_autonomous_will
            import json as _json
            w = get_autonomous_will(start=True)
            res = w.tick()
            return "## ⚡ will_tick\n```json\n" + _json.dumps(res or {"msg": "لا دافع فوق العتبة"}, ensure_ascii=False, indent=2) + "\n```"
        except Exception as e:
            return f"❌ will_tick: {e}"

    # ── deep_research ── 🆕 بحث متعدد الزوايا والمصادر
    if action == "deep_research":
        q = query or step.get("topic") or ""
        if not q:
            return "❌ deep_research: مطلوب query"
        try:
            from ai.web_search_tool import deep_research as _dr
            import json as _json
            res = _dr(q)
            preview = [
                {"title": r.get("title"), "source": r.get("source"), "url": r.get("url")}
                for r in (res.get("results") or [])[:10]
            ]
            out = {k: res.get(k) for k in ("ok", "query", "count", "msg", "angles")}
            out["results_preview"] = preview
            return "## 🔬 deep_research\n```json\n" + _json.dumps(out, ensure_ascii=False, indent=2) + "\n```"
        except Exception as e:
            return f"❌ deep_research: {e}"

    # ── trending ──
    if action == "trending":
        try:
            from ai.web_search_tool import format_trending, get_trending_topics
            geo = step.get("geo") or "SA"
            return format_trending(geo=geo)
        except Exception as e:
            return f"❌ trending: {e}"

    # ── self_learn ── 🆕 تعلّم ذاتي من الويب + حفظ
    if action == "self_learn":
        try:
            from ai.self_feed_learner import learn_from_web, self_learn_cycle
            import json as _json
            topic = query or step.get("topic") or ""
            if topic:
                res = learn_from_web(topic, deep=True)
            else:
                res = self_learn_cycle(limit=int(step.get("limit") or 3))
            return "## 🧠 self_learn\n```json\n" + _json.dumps(res, ensure_ascii=False, indent=2) + "\n```"
        except Exception as e:
            return f"❌ self_learn: {e}"

    # ── ingest_knowledge ──
    if action == "ingest_knowledge":
        try:
            from ai.self_feed_learner import ingest_text
            import json as _json
            topic = step.get("topic") or query or "بدون عنوان"
            content = step.get("content") or ""
            if not content:
                return "❌ ingest_knowledge: مطلوب content"
            sources = step.get("sources") or []
            if isinstance(sources, str):
                sources = [s.strip() for s in sources.split(",") if s.strip()]
            res = ingest_text(topic, content, sources=sources, origin="agent")
            return "## 📥 ingest_knowledge\n```json\n" + _json.dumps(res, ensure_ascii=False, indent=2) + "\n```"
        except Exception as e:
            return f"❌ ingest_knowledge: {e}"

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

_git_identity_configured = False  # 🆕 أداء: تجنّب إعادة ضبط git config في كل مهمة ناجحة


def _ensure_git_identity_once() -> None:
    """يضبط هوية git محلياً مرة واحدة فقط للعملية (وليس في كل checkpoint).

    كانت _local_checkpoint_commit تُشغّل `git config` مرتين (subprocess
    منفصلة، ~2ms لكل واحدة قياساً فعلياً) في كل مهمة ناجحة ضمن أي خطة
    متعددة المهام — رغم أن الهوية لا تتغيّر إلا مرة واحدة بالجلسة. هذا
    يضبطها أول مرة فقط ويتخطاها في كل استدعاء لاحق.
    """
    global _git_identity_configured
    if _git_identity_configured:
        return
    for cfg in [
        ["git", "-C", str(ROOT), "config", "--local",
         "user.email", "nsm-bot@users.noreply.github.com"],
        ["git", "-C", str(ROOT), "config", "--local",
         "user.name", "NSM Bot"],
    ]:
        subprocess.run(cfg, capture_output=True)
    _git_identity_configured = True


def _local_checkpoint_commit(message: str) -> Optional[str]:
    """
    كوميت محلي فقط (بلا push) — نقطة استرجاع بعد نجاح مهمة واحدة فعلياً،
    وليس فقط عند اكتمال خطة كاملة (المرحلة 3 تتكفّل بالرفع النهائي).
    تُعيد commit hash الفعلي (لتسجيله في ai/task_manager.py عبر
    record_checkpoint) أو None إن لم توجد تغييرات فعلية أو فشل الكوميت.
    """
    try:
        _ensure_git_identity_once()

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

        _invalidate_tree_cache()
        return (f"⏪ تم التراجع فعلياً إلى {source_note} "
                f"(commit `{new_head[:10]}`). ملاحظة: هذا يعيد كتابة ملفات "
                f"المشروع محلياً — إن كنت تريد رفع هذا التراجع لـ GitHub "
                f"أيضاً، اطلب git_push صراحة بعده.")
    except Exception as e:
        return f"❌ خطأ في التراجع: {e}"



def _git_lfs_action(step: dict) -> str:
    """تنفيذ عمليات Git LFS من وكيل NSM."""
    import json
    try:
        from ai import git_lfs_helper as lfs
    except Exception as e:
        return f"❌ تعذّر تحميل git_lfs_helper: {e}"

    op = (step.get("lfs_op") or step.get("op") or step.get("cmd") or "status").strip().lower()
    patterns_raw = step.get("patterns") or step.get("include") or ""
    patterns = [p.strip() for p in str(patterns_raw).split(",") if p.strip()] if patterns_raw else None

    try:
        if op in ("status", "حالة", "env"):
            res = lfs.lfs_status(detailed=True)
        elif op in ("install", "setup", "تفعيل", "تثبيت"):
            res = lfs.install_lfs()
        elif op in ("pull", "اسحب", "fetch"):
            res = lfs.lfs_pull()
        elif op in ("push", "رفع"):
            res = lfs.lfs_push()
        elif op in ("track", "تتبع"):
            res = lfs.lfs_track(patterns)
        elif op in ("untrack",):
            res = lfs.lfs_untrack(patterns or [])
        elif op in ("ls", "ls-files", "ملفات", "list"):
            res = lfs.lfs_ls_files()
        elif op in ("migrate", "هجرة"):
            res = lfs.lfs_migrate_import(
                include=",".join(patterns) if patterns else "*.pkl,*.npy,*.pt,*.pth,*.bin,*.h5,*.onnx,*.safetensors",
                everything=bool(step.get("everything")),
            )
        elif op in ("prune",):
            res = lfs.lfs_prune()
        else:
            return (
                f"❌ عملية LFS غير معروفة: {op}\n"
                "المدعوم: status, install, pull, push, track, untrack, ls, migrate, prune"
            )
        return (
            f"## 📦 Git LFS — `{op}`\n```json\n"
            + json.dumps(res, ensure_ascii=False, indent=2)
            + "\n```"
        )
    except Exception as e:
        return f"❌ خطأ git_lfs ({op}): {e}"


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
        # ── 🆕 تحقق فعلي من نجاح الرفع (لا افتراض) — git ls-remote
        # يطابق local SHA مع remote SHA بعد دقائق من الانتظار المؤقتي. ──
        _verify_push_note = ""
        try:
            import time as _time
            local_sha = subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                capture_output=True, text=True,
            ).stdout.strip()
            remote_sha = ""
            for attempt in range(5):
                if attempt > 0:
                    _time.sleep(2 ** attempt)
                _ls = subprocess.run(
                    ["git", "-C", str(ROOT), "ls-remote", "origin", "refs/heads/main"],
                    capture_output=True, text=True,
                )
                if _ls.returncode == 0 and _ls.stdout.strip():
                    remote_sha = _ls.stdout.split()[0]
                    break
            if remote_sha:
                if remote_sha.startswith(local_sha[:12]):
                    _verify_push_note = " ✅ (تُحقّق من التطابق عبر ls-remote)"
                else:
                    _verify_push_note = (
                        f" ⚠️ رفع لكن HEAD المحلي ({local_sha[:10]}) لا يطابق"
                        f" الفرع البعيد الرئيسي ({remote_sha[:10]}) — تحقّق يدوياً"
                    )
            else:
                _verify_push_note = " ⚠️ تعذّر التحقق من الرفع عبر ls-remote"
        except Exception:
            pass
        lfs_note = ""
        try:
            from ai.git_lfs_helper import has_git_lfs, lfs_push
            if has_git_lfs():
                lr = lfs_push()
                if lr.get("ok"):
                    lfs_note = " + LFS"
                elif lr.get("msg"):
                    lfs_note = f" (LFS: {str(lr.get('msg'))[:60]})"
        except Exception:
            pass
        return f"📤 رُفع لـ GitHub ✅{lfs_note}{_verify_push_note}"
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
    """يتحقق إذا كانت نتيجة الخطوة فشلاً يستحق إصلاحاً عبر إعادة كتابة كود.

    🆕 ملاحظة: "فشل البحث" (رسالة web_search عند فشل كل مصادر البحث —
    غالباً بسبب timeout/حظر شبكي) أُزيلت عمداً من هذه القائمة. حلقة
    self-healing تعمل بإعادة سؤال الـ LLM ليكتب خطوات جديدة، وهذا مفيد
    لأخطاء كود حقيقية (SyntaxError مثلاً) لكنه عديم الفائدة لخطأ شبكة —
    الـ LLM يعيد إرسال نفس البحث الفاشل 3 مرات ويُسرّب تفاصيل تقنية خام
    للمستخدم بلا أي فائدة. النتيجة الفاشلة (بادئة ❌) تبقى تُعرَض للمستخدم
    بصدق كما هي دائماً — فقط لا تُشعِل محاولات الإصلاح المكلفة.
    """
    return result.startswith("❌") and any(
        kw in result for kw in [
            "خطأ في التشغيل", "خطأ في الإنشاء", "خطأ في التعديل",
            "غير موجود", "SyntaxError", "ImportError", "ModuleNotFoundError",
            "NameError", "TypeError", "IndentationError",
            "فعل غير صالح", "خطأ في أداة البحث", "مطلوب",
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


# ══════════════════════════════════════════════════════════════════
# 🆕 المرحلة 9 من خطة "المراحل المقترحة (٥ فما فوق)" — إدارة أسرار شبه تلقائية
# ══════════════════════════════════════════════════════════════════

_ENV_VAR_PATTERN = re.compile(
    r"""os\.(?:getenv|environ\.get)\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']"""
    r"""|os\.environ\[\s*["']([A-Za-z_][A-Za-z0-9_]*)["']\s*\]"""
)

# 🆕 متغيرات مزوّدي LLM: غيابها مُدار أصلاً عبر fallback متعدد المزوّدين
# (انظر NSMAgent._check_available) — تنبيه إضافي عنها هنا سيكون تكراراً
# مزعجاً بلا فائدة، فهي مستثناة من هذا الفحص تحديداً.
_ENV_VAR_KNOWN_OPTIONAL = {
    "CF_API_TOKEN", "CF_ACCOUNT_ID", "GOOGLE_API_KEY",
    "GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY",
}


def _extract_env_var_names(content: str) -> set:
    """يستخرج كل أسماء متغيرات البيئة التي يقرأها الكود عبر os.getenv/os.environ."""
    names: set = set()
    if not content:
        return names
    try:
        for m in _ENV_VAR_PATTERN.finditer(content):
            name = m.group(1) or m.group(2)
            if name:
                names.add(name)
    except Exception:
        pass
    return names


def _detect_missing_env_vars(path: str, before_content: str, after_content: str) -> Optional[str]:
    """
    كشف تلقائي شبه فوري لمتغيرات بيئة (أسرار) جديدة يعتمد عليها الكود
    المكتوب/المعدَّل للتو، لكنها غير مضبوطة فعلياً في البيئة الحالية.

    السابقة التي دفعت لهذه المرحلة: كود يعتمد على متغير بيئة مفقود (مثال
    سابق: Unsplash) كان يمكن أن يجتاز py_compile والتحقق الوظيفي (إن لم
    يُمرَّر test_code يستدعي بالضبط المسار المحتاج للمتغير)، ثم يُرفع
    ويُنشر، فتفشل الميزة بصمت وقت التشغيل الفعلي في الإنتاج بدل أن يُكتشف
    الخلل الآن أثناء الكتابة.

    المقارنة تكون بين أسماء المتغيرات في المحتوى الجديد والقديم لنفس
    الملف — فقط متغير "جديد" أُدخل بهذا التعديل تحديداً يستحق تنبيهاً؛
    متغيرات كانت موجودة أصلاً في الملف قبل هذا التعديل ليست مسؤولية هذه
    الخطوة (لتفادي تكرار نفس التنبيه في كل تعديل لاحق غير متعلق بها).

    تعيد نص تنبيه صريح بالاسم المطلوب فعلاً إن وُجد، أو None إن لم يوجد
    ما يستحق التنبيه. هذا التنبيه *لا* يبدأ بـ"❌" عمداً — فهو ليس فشلاً
    يُشعِل حلقة self-healing (إعادة تعديل الكود لن يُضيف السرّ الناقص
    من تلقاء نفسه)؛ فقط رسالة واضحة تصل المستخدم بالاسم المطلوب بدل أن
    يفشل النشر بصمت لاحقاً.
    """
    try:
        new_vars = _extract_env_var_names(after_content) - _extract_env_var_names(before_content)
    except Exception:
        return None

    new_vars -= _ENV_VAR_KNOWN_OPTIONAL
    if not new_vars:
        return None

    missing = sorted(v for v in new_vars if not os.getenv(v, "").strip())
    if not missing:
        return None

    names = "، ".join(f"`{v}`" for v in missing)
    return (
        f"🔑 **تنبيه أسرار — متغيّر بيئة جديد غير مضبوط (`{path}`):**\n"
        f"الكود الجديد يعتمد على {names} لكنه غير موجود حالياً في البيئة "
        f"(Streamlit Secrets). هذا لن يظهر كخطأ syntax أو تشغيل الآن — "
        f"الميزة ستفشل بصمت وقت الاستخدام الفعلي بعد النشر. أضِف {names} "
        f"فعلياً في Streamlit Secrets قبل اعتبار هذه الميزة جاهزة للإنتاج."
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

        # 🆕 المرحلة 9: نلتقط محتوى الملف *قبل* التنفيذ (فارغ إن كان create_file
        # لملف جديد) لمقارنته لاحقاً واكتشاف متغيرات بيئة جديدة أُدخلت بهذا
        # التعديل تحديداً، بعد نجاح الكتابة والتحقق.
        _env_before_content = ""
        if step.get("action", "") in ("create_file", "edit_file") and step.get("path", "").endswith(".py"):
            try:
                _f = ROOT / step["path"]
                if _f.exists():
                    _env_before_content = _f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                _env_before_content = ""

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

                # 🆕 المرحلة 9: إدارة أسرار شبه تلقائية — فحص واحد فقط هنا
                # (وليس أيضاً في حلقة self-healing أدناه) لتفادي التكرار،
                # ولأنه يعمل فقط بعد نجاح التحقّق الوظيفي/py_compile أصلاً.
                # هذا التنبيه لا يُسند إلى result عمداً كي لا يُشعِل
                # self-healing (تعديل الكود لن يُضيف السرّ الناقص).
                if not _is_failure(result):
                    # 🆕 أداء: لـ create_file المحتوى الفعلي المكتوب على القرص
                    # موجود أصلاً في step["content"] (نفس ما كُتب حرفياً) —
                    # إعادة قراءته من القرش هنا كانت I/O مكرراً بلا فائدة.
                    # edit_file فقط يحتاج القراءة (لا نملك النص الكامل الجديد
                    # في الذاكرة، فقط old/new الجزئيين).
                    if _step_action == "create_file":
                        _env_after_content = step.get("content", "")
                    else:
                        try:
                            _env_after_content = (ROOT / _step_path).read_text(
                                encoding="utf-8", errors="replace"
                            )
                        except Exception:
                            _env_after_content = ""
                    _env_warning = _detect_missing_env_vars(
                        _step_path, _env_before_content, _env_after_content
                    )
                    if _env_warning:
                        yield f"{_env_warning}\n\n"

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
        # مدير الفريق يُبنى عند الحاجة فقط حتى لا يضيف كلفة أو استيرادات
        # جانبية إلى الطلبات المباشرة والمهام البرمجية.
        self._team_orchestrator = None
        self.last_orchestration: Dict[str, Any] = {}

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

    def _run_team_orchestrator(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """يفوض الطلب المركب إلى المدير الموحّد والوكلاء المتخصصين."""
        from ai.agent_orchestrator import delegate_to_unified_chat
        if self._team_orchestrator is None:
            from ai.agent_categories import UnifiedAgentChat
            self._team_orchestrator = UnifiedAgentChat()
        response, meta = delegate_to_unified_chat(
            user_input, orchestrator=self._team_orchestrator
        )
        self.last_orchestration = meta
        return response, meta

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

        # 🌱 دورة نمو الوكيل — تخطيط/تنفيذ آمن بدون LLM إن أمكن
        try:
            from ai.agent_growth_loop import handle_growth_command, format_experience_hints
            growth_out = handle_growth_command(user_input)
            if growth_out is not None:
                yield growth_out
                return
        except Exception:
            format_experience_hints = None  # type: ignore

        # 🆕 المهام التعاونية — مهمة مركّبة (بحث/مقارنة/تحليل متعدّد الأوجه)
        # تُفكّك إلى أدوار وكلاء متوازية (باحثون + مدقق) عبر ناقل الأحداث
        # المشترك، ثم يُولّف المنسّق تقريرًا موحدًا. يسبق المهام الفردية
        # لأن المهمات المركّبة ذات الأوجه المتعددة تستحق فريقًا كاملًا.
        try:
            from ai.collaborative_tasks import (
                detect_collaborative_request as _det_coop,
                get_collaborative_manager as _gcm,
            )
            _coop_subgoals = _det_coop(user_input)
            if _coop_subgoals is not None:
                _coop_mgr = _gcm()
                _coop_task = _coop_mgr.submit(user_input, _coop_subgoals)
                if _coop_task is None:
                    yield (
                        "⚠️ طابور المهام التعاونية ممتلئ (مهمتان متزامنتان كحد "
                        "أقصى) — راقب لوحة «التعاون» في «📡 مراقبة حيّة» "
                        "وأعد المحاولة بعد اكتمال مهمة.\n"
                    )
                else:
                    _roles = ", ".join(_coop_subgoals)
                    yield (
                        "🤝 **قبلت المهمة التعاونية** — فريق من الأدوار "
                        "المتوازية سيعمل عليها عبر الإنترنت:\n\n"
                        f"الهدف: {_coop_task.title}\n"
                        f"الأدوار ({len(_coop_subgoals)}): {_roles}\n\n"
                        f"المعرّف: `{_coop_task.task_id}` — تنفَّذ في "
                        "الخلفية (بحث ← مدقق نتائج ← تجميع النتائج ← تقرير "
                        "موحد)، وستجد التقدم اللحظي في لوحة «التعاون» "
                        "داخل تبويب «📡 مراقبة حيّة».\n"
                    )
                return
        except Exception:
            pass  # أي فشل يعيد السلوك الأصلي
        # 🆕 المهام طويلة الأمد — بحث معمّق/تقرير عبر خطوات مع وصول الإنترنت
        # (يُكتشف حتميًا دون API). يسبق Planning Engine لأن بناء التطبيقات
        # يذهب للمنسّق بينما المهام البحثية/التقريرية لها منظّمها الخاص.
        try:
            from ai.long_horizon_tasks import get_long_horizon_manager as _ghm
            _lht_goal = _ghm().detect_long_horizon_request(user_input)
            if _lht_goal is not None:
                _lht_mgr = _ghm()
                _lht_task = _lht_mgr.submit(_lht_goal)
                if _lht_task is None:
                    yield (
                        "⚠️ طابور المهام الطويلة ممتلئ (٢ متزامنة / "
                        "٦ معلقة كحد أقصى) — راقب لوحة «المهام طويلة "
                        "الأمد» في «📡 مراقبة حيّة» وأعد المحاولة بعد "
                        "اكتمال واحدة.\n"
                    )
                else:
                    yield (
                        f"🧵 **قبلت المهمة:** {_lht_task.title}\n\n"
                        f"المعرّف: `{_lht_task.task_id}` — تنفَّذ في الخلفية "
                        f"عبر خطوات (خطة ← بحث ← جلب صفحات ← تجميع ← "
                        f"تقرير)، وستجد التقدم اللحظي وسجل الخطوات في لوحة "
                        f"«المهام طويلة الأمد» داخل تبويب «📡 مراقبة حيّة».\n"
                    )
                return
        except Exception:
            pass  # أي فشل يعيد السلوك الأصلي
        # 🆕 Planning Engine — يكشف طلبات بناء التطبيقات
        try:
            from ai.nsm_planner import NSMPlanner, is_planning_request
            if is_planning_request(user_input):
                planner = NSMPlanner(self)
                yield from planner.build(user_input)
                return
        except ImportError:
            pass  # إذا لم يكن الـ Planner موجوداً، تابع عادياً

        # 🆕 الحلقة التنفيذية متعددة الجولات — "نفّذ:" أو "نفذ:" أو
        # "exec:" تحوّل الطلب من "ردّ نصّي" إلى عامل ذاتي يعمل بجولات LLM
        # متعددة عبر أدوات حقيقية (shell/ملفات/متصفح/بحث) في الخلفية
        _raw_e = user_input.strip()
        if any(_raw_e.startswith(m) for m in ("نفّذ:", "نفذ:", "exec:", "نفذ في الخلفية:", "نفّذ في الخلفية:")):
            try:
                from ai.agent_background_tasks import (
                    enqueue_background_task as _ebt,
                )
                _task = _ebt(_raw_e)
                yield ("🤖 **قبلت مهمة التنفيذ الذاتي** — سيعمل الوكيل الآن كعامل "
                       "متعدد الجولات عبر أدوات حقيقية (shell/ملفات/متصفح/بحث/بحث معمّق) "
                       "في الخلفية حتى يكمل أو تفشل المهمة:\n\n"
                       f"المعرّف: `{_task.get('loop_id', '')}` — تابع التقدم في لوحة "
                       f"«المهام الخلفية» داخل تبويب «📡 مراقبة حيّة».")
                return
            except Exception:
                pass  # أي فشل يسقط للمعالجة العادية (لا تُكسر المحادثة)

        # مدير الفريق: الطلبات التحليلية المركبة تُفكك إلى أدوار متخصصة
        # ثم تُولّف في جواب واحد. المهام البرمجية لا تمر هنا حتى تبقى أدوات
        # NSMAgent التنفيذية (read/edit/test/git) هي المسؤولة عنها.
        try:
            from ai.agent_orchestrator import should_delegate_request
            if should_delegate_request(user_input):
                yield "🎯 *أفكك المهمة وأجمع وجهات نظر الوكلاء المتخصصين...*\n\n"
                team_response, team_meta = self._run_team_orchestrator(user_input)
                if team_response:
                    delegated = team_meta.get("delegated_agents", [])
                    if delegated:
                        names = ", ".join(str(item) for item in delegated)
                        yield f"📌 *الفريق المكلّف: {names}*\n\n"
                    # 🆕 تمرير نتيجة اتساق الفريق للواجهة (badge أسفل الرد)
                    try:
                        import streamlit as st
                        _tc_m = team_meta.get("team_consistency") or {}
                        if _tc_m:
                            st.session_state["_nsm_tc_result"] = dict(_tc_m)
                    except Exception:
                        pass
                    # 🆕 تسجيل نتيجة الجولة في خبرة التوجيه المثبتة
                    # (single-agent route من orchestrator)
                    try:
                        from ai.routing_experience import get_routing_experience
                        _route_ok = not team_meta.get("_team_failed") and bool(
                            (team_meta.get("team_consistency") or {}).get("score", 1.0) or 1.0 >= 0.45
                        )
                        get_routing_experience().record(
                            user_input,
                            route_method=team_meta.get("route_method", "default"),
                            selected_agents=delegated,
                            ok=bool(_route_ok),
                        )
                    except Exception:
                        pass
                    yield team_response
                    return
        except Exception:
            # فشل طبقة التفويض لا يمنع المساعد العام من متابعة الطلب عبر
            # مسار LLM المركزي؛ العزل هنا مقصود لضمان التوافق الرجعي.
            self.last_orchestration = {"route_method": "fallback_after_orchestration_error"}

        # بناء رسائل API (+ تلميح خبرات مشابهة إن وُجدت)
        system   = _build_system_prompt()
        try:
            if format_experience_hints is not None:
                _hints = format_experience_hints(user_input)
                if _hints:
                    system = system + chr(10)*2 + _hints
        except Exception:
            pass
        # 🆕 تلميح خبرة التوجيه المثبتة — من جولات ناجحة سابقة فعلية
        # (routing_experience.py): إن وُجدت خبرة مثبتة لطلب مشابه يُدفع
        # التلميح داخل system prompt ليوجّه أسلوب الرد فورًا.
        try:
            from ai.routing_experience import get_routing_experience
            _rex = get_routing_experience()
            _rex_hint = _rex.hints_for(user_input)
            if _rex_hint:
                system = system + chr(10)*2 + _rex_hint
                # تذكير للواجهة (badge) — يُقرأ لاحقًا من session_state
                try:
                    import streamlit as st
                    st.session_state["_nsm_rex_hint"] = _rex_hint
                except Exception:
                    pass
        except Exception:
            pass
        # 🆕 درس الذاكرة التعلمية — إن سجّلنا سابقًا خطئًا مماثلًا لهذا الرد،
        # نحمّله كتحذير وقائي (reflection_memory.py) حتى لا يتكرر الفشل.
        try:
            from ai.reflection_memory import get_reflection_memory
            _sig = getattr(self, "_last_error_signature", None)
            if _sig:
                _rm = get_reflection_memory()
                _l = _rm.lookup(_sig)
                if _l:
                    system = system + chr(10)*2 + "⚠️ " + _l["hint"]
                    try:
                        import streamlit as st
                        st.session_state["_nsm_rm_hint"] = _l["hint"]
                    except Exception:
                        pass
        except Exception:
            pass
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
        try:
            from ai.agent_project_bridge import dispatch_agent_message
            _bridged = dispatch_agent_message(user_input)
            if _bridged is not None:
                try:
                    from ai.agent_growth_loop import record_experience
                    # تسجيل نجاح المسار الجسري (أدوات مشروع) كخبرة خفيفة
                    record_experience(
                        user_input,
                        plan=["dispatch_agent_message"],
                        tools=["project_bridge"],
                        success=True,
                        summary=_bridged[:400],
                    )
                except Exception:
                    pass
                return _bridged
        except Exception:
            pass

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
