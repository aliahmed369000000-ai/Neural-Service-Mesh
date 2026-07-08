"""
streamlit_app.py — نقطة دخول Streamlit Cloud
================================================
Streamlit Cloud يبحث عن هذا الملف افتراضياً.
كل منطق التطبيق موجود في app.py (نقطة الدخول الأصلية في Replit).
"""
# تأكد من أن المسار الجذر في sys.path حتى تعمل استيرادات ai.*
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# تشغيل التطبيق الرئيسي — نفس السلوك تماماً كـ app.py
exec(  # noqa: S102
    compile(ROOT.joinpath("app.py").read_bytes(), "app.py", "exec"),
    {"__file__": str(ROOT / "app.py"), "__name__": "__main__"},
)
