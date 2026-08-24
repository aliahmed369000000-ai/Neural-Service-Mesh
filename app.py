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

# بدء تشغيل العقدة في خيط منفصل (Thread) لضمان عدم حجز واجهة Streamlit
if "node_started" not in st.session_state:
    thread = threading.Thread(target=run_mesh_node, daemon=True)
    thread.start()
    st.session_state["node_started"] = True
    logger.info("🚀 NSM Background Node Started via Streamlit.")

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
