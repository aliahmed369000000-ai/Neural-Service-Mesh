# -*- coding: utf-8 -*-
"""🆕 v2: مساعد ذكي للخلية (Cell Copilot) — NSM Notebook

يثري تجربة الدفتر بثلاثة قدرات لكل خلية:
1. شرح الخلية — ماذا تفعل ولماذا
2. إصلاح الخطأ — عندما تنتهي الخلية بحالة error يعرض سببًا واضحًا وحلًا
3. اقتراح تحسينات — تحسينات عملية للكود (أداء/قراءة/توافق مع التدريب)

يستخدم OpenRouter (gemini-2.5-flash) عبر مفتاح OPENROUTER_API_KEY من
البيئة/secrets — وعند غيابه يرجع إلى تحليل محلي قاعدي (لا يفشل أبدًا).
لا يُعدّل الكود تلقائيًا — يقدم الاقتراحات فقط، والمستخدم هو من يقرر.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

_COPILOT_URL = "https://openrouter.ai/api/v1/chat/completions"
_COPilot_MODEL = "google/gemini-2.5-flash"

_SYS = (
    "أنت مساعد خبير في Python وPyTorch وتدريب نماذج الذكاء الاصطناعي لمشروع "
    "Neural Service Mesh (NSM) — مشروع عربي بلغة Python + Streamlit. "
    "أجب بالعربية الفصحى باختصار ووضوح. لا تعرض كودًا ينفّذ أوامر نظام خطرة."
)

_PROMPT_EXPLAIN = """اشرح بإيجاز (3-5 أسطر) ماذا تفعل هذه الخلية ولماذا:
{source}
إن وُجدت مخرجات:
{outputs}"""

_PROMPT_FIX = """هذه الخلية أنتجت خطأً. أعطِ:
1) سبب الخطأ في سطر واحد
2) إصلاحًا عمليًا في 2-4 أسطر (كود مصحح مقتطف فقط إن لزم)

مصدر الخلية:
{source}

الخطأ:
{error}"""

_PROMPT_IMPROVE = """اقترح 2-3 تحسينات عملية قصيرة على هذه الخلية (أداء/قراءة/توافق تدريب):
{source}
مخرجاتها:
{outputs}"""


def _call_llm(user: str, api_key: Optional[str] = None,
              timeout: int = 25) -> Optional[str]:
    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        return None
    try:
        import requests
        r = requests.post(
            _COPILOT_URL,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={
                "model": _COPilot_MODEL,
                "messages": [
                    {"role": "system", "content": _SYS},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.3,
                "max_tokens": 400,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        choice = r.json().get("choices") or []
        if choice:
            return (choice[0].get("message", {}).get("content", "") or "").strip()
    except Exception:
        pass
    return None


def _local_fix(source: str, error: str) -> str:
    """تحليل قاعدي محلي — يعمل بدون أي مفتاح API."""
    out: List[str] = []
    err = (error or "").strip().splitlines() or ["—"]
    last = err[-1][:200]
    if "NameError" in last:
        m = re.search(r"name '(\w+)' is not defined", last)
        name = m.group(1) if m else "متغيّر"
        out.append(f"السبب: استخدام `{name}` قبل تعريفه.")
        out.append("الحل: عرّف المتغيّر أولًا أو استورد الوحدة التي تحتويه.")
    elif "ImportError" in last or "ModuleNotFoundError" in last:
        m = re.search(r"(\S+)$", err[0] if err else "")
        out.append("السبب: وحدة غير مثبتة في بيئة التشغيل.")
        out.append("الحل: ثبّتها في الخلية الأولى عبر `!pip install` أو subprocess.")
    elif "SyntaxError" in last:
        out.append("السبب: خطأ بنيوي في صياغة Python.")
        out.append("الحل: راجع الأقواس والفواصل والمسافات البادئة (indentation).")
    elif "IndexError" in last or "KeyError" in last:
        out.append("السبب: الوصول إلى عنصر خارج نطاق القائمة/القاموس.")
        out.append("الحل: افحص الطول أو استخدم `.get()` قبل الوصول.")
    elif "CUDA" in last or "cuda" in last.lower():
        out.append("السبب: ذاكرة GPU نفدت أو CUDA غير متاح في هذه البيئة.")
        out.append("الحل: نقل التدريب إلى Kaggle/Colab أو تصغير `batch_size`.")
    elif "Traceback" in error or "Error" in last:
        out.append(f"سبب مرجّح: {last[:150]}")
        out.append("الحل: اطبع القيم الوسيطة للتحقق من المسار قبل نقطة الخطأ.")
    else:
        out.append("لا تشخيص قاعديًا — جرّب مساعد الذكاء الاصطناعي.")
    return "\n".join(out)


def _local_explain(source: str) -> str:
    src = (source or "").strip()
    if not src:
        return "الخلية فارغة."
    lines = len(src.splitlines())
    parts = []
    if any(w in src for w in ("torch", "nn.", "optim")):
        parts.append("خلية تدريب/نموذج PyTorch")
    elif "def " in src:
        parts.append("تعريف دوال/مساعدات")
    elif "import " in src:
        parts.append("استيرادات وحدات")
    elif "print" in src:
        parts.append("طباعة/فحص قيم")
    else:
        parts.append("كود عام")
    return f"خلاصة: {', '.join(parts)} — {lines} سطرًا."


def _local_improve(source: str) -> str:
    src = source or ""
    ideas: List[str] = []
    if "for " in src and "range(" in src:
        ideas.append("يمكن تسريع الحلقات الطويلة بإدخالها في دالة — Python أسرع داخل الدوال.")
    if "print(" in src and "for " in src:
        ideas.append("للطباعة داخل حلقات طويلة: اطبع كل N خطوة بدل كل خطوة لتقليل الإبطاء.")
    if "import " in src and "import " in src.splitlines()[0]:
        ideas.append("يفضّل فصل الاستيرادات في خلية أولى مستقلة لتسريع إعادة التشغيل.")
    if not ideas:
        ideas.append("الكود واضح — لا تحسينات قاعدية بارزة.")
    return " • ".join(ideas)


def explain_cell(source: str, outputs: str = "",
                 api_key: Optional[str] = None) -> Dict[str, Any]:
    user = _PROMPT_EXPLAIN.format(source=source[:3000], outputs=(outputs or "—")[:1200])
    text = _call_llm(user, api_key)
    if not text:
        text = _local_explain(source)
    return {"ok": True, "text": text, "from_llm": bool(text != _local_explain(source) and _call_llm(user, api_key))}


def fix_cell(source: str, error: str,
             api_key: Optional[str] = None) -> Dict[str, Any]:
    user = _PROMPT_FIX.format(source=source[:3000], error=(error or "")[:3000])
    text = _call_llm(user, api_key)
    local = _local_fix(source, error)
    if not text:
        text = local
    return {"ok": True, "text": text, "from_llm": bool(text != local)}


def improve_cell(source: str, outputs: str = "",
                 api_key: Optional[str] = None) -> Dict[str, Any]:
    user = _PROMPT_IMPROVE.format(source=source[:3000], outputs=(outputs or "—")[:1200])
    text = _call_llm(user, api_key)
    local = _local_improve(source)
    if not text:
        text = local
    return {"ok": True, "text": text, "from_llm": bool(text != local)}


def suggest_next(cell_source: str, history: List[str],
                 api_key: Optional[str] = None) -> Dict[str, Any]:
    """🆕 اقتراح الخلية التالية حسب سياق الدفتر."""
    from ai.terminal_smart import suggest_command
    text, from_llm = suggest_command(history or ["# notebook"],
                                     last_output="", api_key=api_key)
    return {"ok": True, "text": text, "from_llm": from_llm}
