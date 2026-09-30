"""
اكتُشف أثناء تتبّع مسار plan_and_execute_goal في core/mesh_bundle.py حتى
ai/capability_marketplace.py: `restore()` (المضافة في commit 6c2abb4)
توثّق صراحة أن الهدف هو "الحفاظ على درجات الجودة/الكمون المتراكمة من
record_execution()"، لكن `record_execution()` نفسها — وكذلك
`deactivate_node()`/`reactivate_node()` — لم تكن تستدعي `_persist()`
إطلاقاً؛ فقط `advertise()` كانت تكتب على القرص. النتيجة العملية: أي
تحديث حقيقي لدرجة الجودة ناتج عن تنفيذ فعلي عبر plan_and_execute_goal
(أو أي مستدعٍ آخر لـrecord_execution) يبقى في الذاكرة فقط ويُفقَد عند
إعادة التشغيل التالية — فهرس القدرات نفسه يُستعاد (لأن العدد الإجمالي
للإعلانات لا يتغيّر)، لكن بقيم quality_score/execution_count الابتدائية
القديمة من advertise()، لا القيم المتعلَّمة فعلياً. اختبار
tests/test_capability_marketplace_and_multi_goal_wiring.py الموجود لا
يغطي هذه الحالة (يتحقق فقط من ثبات total_capabilities/registered_nodes
الإجماليَّين، لا من نجاة تحديثات record_execution تحديداً عبر إعادة
تشغيل).
"""
from __future__ import annotations

import shutil
import tempfile

import pytest

from ai.capability_marketplace import CapabilityMarketplace
from knowledge.knowledge_store import KnowledgeStore


@pytest.fixture()
def knowledge_store():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_marketplace_record_exec_ks_test_")
    try:
        yield KnowledgeStore(knowledge_dir=tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


class TestRecordExecutionPersistence:
    def test_learned_quality_and_count_survive_restart(self, knowledge_store):
        mkt1 = CapabilityMarketplace(knowledge_store=knowledge_store)
        mkt1.advertise(node_id="n1", node_name="Node1", capability="translate_text",
                        quality_score=0.8, avg_latency_ms=50.0)

        for _ in range(4):
            mkt1.record_execution("n1", "translate_text", success=True, latency_ms=10.0)
        mkt1.record_execution("n1", "translate_text", success=False, latency_ms=200.0)

        learned = mkt1.best_provider("translate_text")
        assert learned.execution_count == 5
        assert learned.quality_score == pytest.approx(0.8 + 4 * 0.01 - 0.05)

        # لا استدعاء _persist() يدوي هنا — record_execution وحدها يجب أن
        # تكون كافية بعد الإصلاح. (restore() نفسها غير تلقائية في __init__
        # بتصميم هذا الصنف — MeshBundle هي من تستدعيها صراحة بعد الإنشاء،
        # فنطابق نفس الاستخدام الحقيقي هنا.)
        mkt2 = CapabilityMarketplace(knowledge_store=knowledge_store)
        mkt2.restore()
        revived = mkt2.best_provider("translate_text")
        assert revived is not None, "الإعلان الأصلي لم يُستعَد إطلاقاً"
        assert revived.execution_count == 5, (
            f"execution_count المتعلَّم لم يُحفظ: توقّعنا 5، وجدنا {revived.execution_count}"
        )
        assert revived.quality_score == pytest.approx(learned.quality_score), (
            "quality_score المتعلَّم من record_execution لم يُحفظ — استُعيدت القيمة الابتدائية فقط"
        )
        assert revived.avg_latency_ms == pytest.approx(learned.avg_latency_ms, abs=0.01)

    def test_deactivate_and_reactivate_survive_restart(self, knowledge_store):
        mkt1 = CapabilityMarketplace(knowledge_store=knowledge_store)
        mkt1.advertise(node_id="n1", node_name="Node1", capability="cap_x")
        mkt1.deactivate_node("n1")
        assert mkt1.find_providers("cap_x") == []

        mkt2 = CapabilityMarketplace(knowledge_store=knowledge_store)
        mkt2.restore()
        assert mkt2.find_providers("cap_x") == [], "حالة is_active=False لم تُحفظ عبر إعادة التشغيل"

        mkt2.reactivate_node("n1")
        mkt3 = CapabilityMarketplace(knowledge_store=knowledge_store)
        mkt3.restore()
        assert len(mkt3.find_providers("cap_x")) == 1, "حالة is_active=True بعد إعادة التفعيل لم تُحفظ"
