"""ai/vision_analyzer.py: openai كان استيراداً إلزامياً (from openai import
OpenAI على مستوى الملف، بلا try/except) غير مُعلَن في requirements.txt
إطلاقاً (تحقّقت بالبحث المباشر في requirements.txt قبل هذا التعديل: صفر
نتائج). النتيجة: أي تثبيت نظيف عبر `pip install -r requirements.txt` كان
يفشل عند أول استيراد لـai/vision_analyzer.py (أو ai/video_sampler.py أو
ai/agent_loop.py التي تستورده انتقالياً، ويستوردها بدورها 19 سكربت
simulations/*.py) بـModuleNotFoundError — ليس بسبب بيئة ناقصة، بل تبعية
حقيقية مفقودة من الملف المعلن نفسه. هذا الاختبار يمنع تكرار الفجوة لو
أُزيل سطر openai من requirements.txt مستقبلاً بالخطأ.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_openai_declared_in_requirements():
    req = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"(?m)^openai\b", req), (
        "ai/vision_analyzer.py يستورد openai بلا try/except، لكنه غير معلَن "
        "في requirements.txt"
    )


def test_vision_analyzer_module_imports_cleanly():
    """الاستيراد نفسه ينجح الآن (openai مثبَّت في بيئة الاختبار الحالية) —
    يثبت أن المشكلة كانت في الإعلان (requirements.txt) لا في الكود نفسه."""
    import ai.vision_analyzer  # noqa: F401 — النجاح بلا استثناء هو الاختبار
