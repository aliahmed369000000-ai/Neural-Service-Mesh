"""
البحث الدلالي في الذاكرة بعد إعادة التصميم إلى MemoryManager (كوميت 19832b7):
state.semantic_memory/search_semantic حُذفا؛ الحقائق في
state.memory_manager.ltm_semantic والبحث عبر memory_manager.search().
أُعيدت كتابة الاختبار بتأكيدات حقيقية بدل الطباعة فقط.
"""
from ai import agent_hibernation as ah

AGENT_ID = "test_semantic_agent"


def _cleanup():
    f = ah.SLEEP_DIR / f"{AGENT_ID}_sleep.json"
    if f.exists():
        f.unlink()


def test_memory_semantic_search():
    context = [
        {"role": "user", "content": "الهدف هو تحسين أداء الشبكة العصبية عبر تقليل زمن الاستجابة."},
        {"role": "assistant", "content": "تم تنفيذ ذلك عبر ضغط المتجهات بنسبة 75%."},
        {"role": "user", "content": "ما هو SHA للالتزام الأخير؟"},
        {"role": "assistant", "content": "SHA هو a1b2c3d."},
    ]
    try:
        assert ah.hibernate_agent(AGENT_ID, context, {"steps": []}, force_summarize=True)
        state = ah.wake_up_agent(AGENT_ID)
        assert state is not None
        mm = state.memory_manager
        assert len(mm.ltm_semantic) >= 1, "لم تُستخرج حقائق دلالية"
        results = mm.search("أريد رمز SHA")["semantic"]
        assert any("a1b2c3d" in (r.get("content") or "") for r in results)
    finally:
        _cleanup()
