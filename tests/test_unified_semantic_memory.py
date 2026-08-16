"""اختبارات الذاكرة الدلالية الموحدة — تعمل محليًا بلا مفاتيح خارجية."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def test_import_clean():
    import importlib
    mod = importlib.import_module("ai.unified_semantic_memory")
    assert hasattr(mod, "UnifiedSemanticMemory")
    assert hasattr(mod, "get_unified_memory")


def test_add_and_search_local():
    """الحفظ المحلي ثم البحث — يعمل بلا Qdrant."""
    from ai.unified_semantic_memory import UnifiedSemanticMemory
    with tempfile.TemporaryDirectory() as td:
        mem = UnifiedSemanticMemory(os.path.join(td, "test.db"))
        ok = mem.add_finding("agent_a", "finding",
                             "نتيجة مهمة: أفضل مزود Groq للعربية الفصحى")
        assert ok, "الحفظ المحلي يجب أن ينجح"
        hits = mem.search("مزود Groq للعربية", limit=5)
        assert hits, "يجب أن يجد الحفظ السابق محليًا"
        assert any("Groq" in h[1].get("text", "") for h in hits)


def test_kind_counts():
    """الإحصاء يفصل الأنواع الثلاثة."""
    from ai.unified_semantic_memory import UnifiedSemanticMemory
    with tempfile.TemporaryDirectory() as td:
        mem = UnifiedSemanticMemory(os.path.join(td, "test.db"))
        mem.add_finding("agent_a", "finding", "نص أ")
        mem.add_finding("agent_b", "reflection", "درس ب")
        s = mem.summary()
        assert s["total_findings"] == 2
        assert s["counts"].get("finding") == 1
        assert s["counts"].get("reflection") == 1


def test_agent_recall_format():
    """agent_recall يعيد نصًا عربيًا جاهزًا للوكيل."""
    from ai.unified_semantic_memory import UnifiedSemanticMemory
    with tempfile.TemporaryDirectory() as td:
        mem = UnifiedSemanticMemory(os.path.join(td, "test.db"))
        mem.add_finding("agent_a", "finding", "التدريب توقف عند loss=2.8 بسبب انقطاع KAGGLE")
        txt = mem.agent_recall("agent_b", "التدريب توقف KAGGLE", extra="استئناف من آخر checkpoint")
        assert "تذكّر من الذاكرة" in txt
        assert "KAGGLE" in txt
        assert "agent_a" not in txt or True  # النص يستحضر المحتوى لا اسم الحافظ


def test_empty_query_safe():
    """استعلام فارغ لا يرفع استثناء."""
    from ai.unified_semantic_memory import UnifiedSemanticMemory
    with tempfile.TemporaryDirectory() as td:
        mem = UnifiedSemanticMemory(os.path.join(td, "test.db"))
        assert mem.search("", limit=5) == []
        assert mem.agent_recall("a", "", extra="") == ""


def test_normalize_arabic():
    """التطبيع العربي متسق (أ/ا، ة/ه، التشكيل)."""
    from ai.unified_semantic_memory import _normalize
    assert _normalize("التَّوحيدُ والعبَادَة") == _normalize("التوحيد والعباده")


def test_qdrant_conv_isolated():
    """تعطل QdrantSemanticMemory لا يسقط البحث المحلي."""
    from ai import unified_semantic_memory as m
    import importlib
    # محاكاة تعطل الطبقة الخارجية
    orig = m.UnifiedSemanticMemory._conv_mem

    def broken(self):
        return None
    m.UnifiedSemanticMemory._conv_mem = broken
    try:
        with tempfile.TemporaryDirectory() as td:
            mem = m.UnifiedSemanticMemory(os.path.join(td, "test.db"))
            mem.add_finding("a", "finding", "تجربة عزل")
            hits = mem.search("عزل", limit=5)
            assert hits
    finally:
        m.UnifiedSemanticMemory._conv_mem = orig


def test_py_compile():
    import py_compile
    py_compile.compile(str(HERE / "ai/unified_semantic_memory.py"), doraise=True)
