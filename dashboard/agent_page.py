"""
دمج الوكيل (NSM_Agent_v7.html) داخل تطبيق Streamlit — بدون أي اعتماد على Gradio.

طريقة الاستخدام:
1. ضع NSM_Agent_v7.html في نفس مجلد هذا الملف (app.py / agent_page.py).
2. أضف مفتاح Groq المجاني في Secrets الخاصة بتطبيق Streamlit
   (Settings → Secrets في Streamlit Community Cloud) بالشكل التالي:

       GROQ_API_KEY = "gsk_..."

   هذا المفتاح هو المفتاح "الافتراضي" الذي سيُستخدم تلقائياً لأي زائر
   ليس لديه مفتاحه الخاص، فلا يحتاج أحد لإعداد أي شيء قبل التجربة.
   (يعمل أيضاً عبر متغير بيئة عادي GROQ_API_KEY إن لم تستخدم Streamlit Cloud).

3. في app.py استورد render_agent_page() ونادِها داخل تبويب "🤖 الوكيل".

تنبيه أمني: لأن هذا الملف يعمل بالكامل داخل المتصفح (بدون خادم وسيط)،
أي مفتاح يُحقَن هنا يصبح مرئياً لمن يفتح "عرض المصدر" في صفحة الوكيل —
أي أنه ليس سرّاً حقيقياً. لذلك استخدم فقط مفتاح Groq المجاني (لا تكلفة
عليه)، وتجنّب وضع مفاتيح Cloudflare/Gemini كافتراضية لأنها قد تُستخدم
بشكل مكلف من زوار مجهولين. لا تضع توكن GitHub هنا أبداً — يبقى ذلك
اختيارياً وشخصياً فقط من داخل تبويب GitHub بالوكيل نفسه.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

AGENT_HTML_FILENAME = "NSM_Agent_v7.html"
AGENT_HTML_PATH = Path(__file__).parent / AGENT_HTML_FILENAME
PLACEHOLDER = "__NSM_DEFAULT_GROQ_KEY__"


def _get_default_groq_key() -> str:
    """يقرأ المفتاح من Streamlit Secrets أولاً، ثم من متغيرات البيئة كبديل."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


def render_agent_page(height: int = 820) -> None:
    """يعرض صفحة الوكيل كاملة (Setup + Chat + GitHub) داخل تبويب Streamlit."""
    if not AGENT_HTML_PATH.exists():
        st.error(
            f"⚠️ لم يتم العثور على {AGENT_HTML_FILENAME} بجانب app.py — "
            "ارفعه لنفس مجلد المشروع على GitHub / Streamlit Cloud."
        )
        return

    html = AGENT_HTML_PATH.read_text(encoding="utf-8")
    html = html.replace(PLACEHOLDER, _get_default_groq_key())

    # scrolling=False لأن صفحة الوكيل تدير التمرير الداخلي لنفسها (لوحات/محادثة)
    components.html(html, height=height, scrolling=False)
