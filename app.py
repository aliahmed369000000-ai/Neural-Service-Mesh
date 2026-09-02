import streamlit as st
import os
import asyncio
import threading
import logging
from pathlib import Path

# إعداد السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NSM-HF-App")

def run_mesh_node():
    """تشغيل عقدة NSM في خلفية تطبيق Streamlit."""
    try:
        from ai.node_launcher import main as node_main
        # ضبط المتغيرات إذا لم تكن موجودة
        if not os.getenv("NODE_ID"):
            os.environ["NODE_ID"] = "mesh_global_streamlit"
        
        # تشغيل العقدة في حلقة أحداث منفصلة
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(node_main())
    except Exception as e:
        logger.error(f"❌ Failed to run NSM Node: {e}")

# تشغيل عقدة الشبكة اختياري فقط؛ تشغيل aiohttp داخل Streamlit Cloud
# قد يحجز منفذاً إضافياً أو يعيد تشغيل التطبيق عند فشل العقدة.
# فعّله صراحةً عبر NSM_ENABLE_NODE=true في البيئات التي تحتاجه.
#
# تنبيه معماري مهم: حتى لو فُعِّل هذا المتغير، Streamlit Community Cloud
# يعرض للإنترنت منفذ خادم Streamlit نفسه فقط (لا يوجد بروكسي عام لأي منفذ
# إضافي يفتحه thread داخلي). هذا يعني أن نقاط النهاية الخاصة بعقدة aiohttp
# (/status, /health, /v2/status) لن تكون قابلة للوصول من رابط النشر العام
# لتطبيق Streamlit نفسه — الطلبات لتلك المسارات ستُعالَج (أو تُرفَض) من
# خادم Streamlit، لا من عملية aiohttp. لتشغيل عقدة قابلة للوصول فعلياً عبر
# HTTP علنياً، استخدم مساراً منفصلاً حيث تكون عملية node_launcher.py هي
# الخادم الرئيسي المُعرَّض (راجع Dockerfile / Procfile المُعدَّين لهذا الغرض،
# مثل Hugging Face Spaces)، لا الاعتماد على thread خلفي داخل Streamlit Cloud.
_enable_node = os.getenv("NSM_ENABLE_NODE", "false").strip().lower() in {"1", "true", "yes", "on"}
if _enable_node and "node_started" not in st.session_state:
    thread = threading.Thread(target=run_mesh_node, daemon=True, name="nsm-mesh-node")
    thread.start()
    st.session_state["node_started"] = True
    logger.info(
        "🚀 NSM Background Node Started via Streamlit "
        "(ملاحظة: منفذ aiohttp داخلي فقط، غير معروض علناً عبر Streamlit Cloud)."
    )

# استيراد وتشغيل التطبيق الأصلي
try:
    from streamlit_app import main
    main()
except ImportError:
    # إذا كان streamlit_app.py لا يحتوي على main() مباشرة، نقوم بتشغيل المنطق يدوياً
    import streamlit_app
    if hasattr(streamlit_app, 'main'):
        streamlit_app.main()
    else:
        # محاكاة التشغيل إذا كان الملف يعتمد على الترتيب التسلسلي
        pass
except Exception as e:
    st.error(f"Error loading NSM Dashboard: {e}")
