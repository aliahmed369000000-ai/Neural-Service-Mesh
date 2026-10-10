"""
اختبار الذاكرة الهرمية بعد إعادة التصميم إلى MemoryManager (كوميت 19832b7):
AgentState لم يعد يحمل episodic_memory/semantic_memory/search_episodic مباشرة؛
الذاكرة طويلة الأجل في state.memory_manager (ltm_episodic/ltm_semantic/search).
الاختبار القديم كان يشير للحقول المحذوفة ويفشل بـAttributeError، وكان يطبع
"نجاح/فشل" بلا أي assert فعلي؛ أُعيدت كتابته بتأكيدات حقيقية.
"""
from ai import agent_hibernation as ah

AGENT_ID = "test_hierarchical_agent"


def _cleanup():
    f = ah.SLEEP_DIR / f"{AGENT_ID}_sleep.json"
    if f.exists():
        f.unlink()


def test_hierarchical_memory():
    context = [{"role": "system", "content": "أنت وكيل ذكي."}]
    for i in range(20):  # >15 رسالة لتفعيل التلخيص التلقائي
        context.append({"role": "user", "content": f"رسالة اختبار رقم {i}"})
        context.append({"role": "assistant", "content": f"رد اختبار رقم {i}"})
    try:
        assert ah.hibernate_agent(AGENT_ID, context, {"tasks": ["مهمة 1"]}, compress=True)
        state = ah.wake_up_agent(AGENT_ID)
        assert state is not None
        mm = state.memory_manager
        # ضُغطت الذاكرة العاملة وأُرشف القديم في الذاكرة الأحداثية
        assert len(state.context) < len(context)
        assert len(mm.ltm_episodic) >= 1
        assert mm.ltm_episodic[0].get("summary")
        # البحث يصل للذاكرة الأحداثية
        assert len(mm.search("رسالة")["episodic"]) >= 1
    finally:
        _cleanup()
