"""
دمج الوكيل (NSM_Agent_v6_repo_agent.html) داخل تطبيق Streamlit.

طريقة الاستخدام:
1. ضع NSM_Agent_v6_repo_agent.html في نفس مجلد هذا الملف (مثلاً dashboard/).
2. أضف مفتاح Groq المجاني في إعدادات Secrets الخاصة بتطبيق Streamlit
   (Settings → Secrets في Streamlit Cloud) بالشكل التالي:

       GROQ_API_KEY = "gsk_..."

   هذا المفتاح هو المفتاح "الافتراضي" الذي سيُستخدم تلقائياً لأي زائر
   ليس لديه مفتاحه الخاص، فلا يحتاج أحد لإعداد أي شيء قبل التجربة.

3. في app.py، استورد الدالة أدناه واستدعها من التبويب/الزر الذي تريد
   أن يفتح صفحة الوكيل (مثلاً أضف "🤖 الوكيل" إلى شريط التنقل الحالي
   بجانب "الرئيسية" و"البحث المعرفي"... إلخ، ثم نادِ render_agent_page()
   في فرع ذلك التبويب).

تنبيه أمني: لأن هذا ملف يعمل بالكامل داخل المتصفح (بدون خادم وسيط)،
أي مفتاح يُحقَن هنا يصبح مرئياً لمن يفتح "عرض المصدر" في صفحة الوكيل —
أي أنه ليس سرّاً حقيقياً. لذلك استخدم فقط مفتاح Groq المجاني (لا تكلفة
عليه)، وتجنّب وضع مفاتيح Cloudflare/Gemini كافتراضية لأنها قد تُستخدم
بشكل مكلف من زوار مجهولين. لا تضع توكن GitHub هنا أبداً — يبقى ذلك
اختيارياً وشخصياً فقط من داخل تبويب GitHub بالتطبيق.
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

AGENT_HTML_PATH = Path(__file__).parent / "NSM_Agent_v6_repo_agent.html"
PLACEHOLDER = "__NSM_DEFAULT_GROQ_KEY__"


def render_agent_page(height: int = 820) -> None:
    """يعرض صفحة الوكيل كاملة (Setup + Chat) داخل التطبيق."""
    if not AGENT_HTML_PATH.exists():
        st.error("لم يتم العثور على NSM_Agent_v6_repo_agent.html بجانب هذا الملف.")
        return

    html = AGENT_HTML_PATH.read_text(encoding="utf-8")

    default_key = st.secrets.get("GROQ_API_KEY", "")
    html = html.replace(PLACEHOLDER, default_key)

    # scrolling=False لأن التطبيق يدير التمرير الداخلي لنفسه (لوحات/محادثة)
    components.html(html, height=height, scrolling=False)
