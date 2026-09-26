"""
اختبار: ai/gap_detector.py — GapDetectionEngine كانت تحتفظ بكل الفجوات
المكتشَفة (self._detected_gaps) في الذاكرة فقط. _persist_gaps() كانت
مكتوبة بالكامل (تكتب إلى knowledge_store.write_custom) لكن:
  1. MeshBundle لم يكن يمرّر knowledge_store إطلاقاً (KnowledgeStore لم
     تكن تُبنى في أي مكان بالمشروع كله) — فـ_persist_gaps() كانت لا-عملية
     دائماً في الإنتاج.
  2. حتى مع knowledge_store، لا شيء كان يستدعي read_custom() عند الإقلاع
     لإعادة تحميل الفجوات المحفوظة في self._detected_gaps.
  3. mark_resolved() تُحدّث self.resolved في الذاكرة فقط، بلا أي مزامنة
     للتخزين — أي حالة "محلولة" تضيع فوراً عند إعادة التشغيل.

هذا الاختبار يبني MeshBundle حقيقية (bundle1) عبر نفس مسار الإنتاج
(core/mesh_bundle.py)، يضيف فجوة، يعلّمها محلولة، ثم يبني MeshBundle
*ثانية* (bundle2) بنفس التخزين (محاكاة إعادة تشغيل) ويتحقق أن الفجوة
ما تزال موجودة وأن حالة "resolved" محفوظة فعلياً.
"""
from __future__ import annotations

import shutil
import tempfile

import pytest

from ai.gap_detector import DetectedGap, GapDetectionEngine
from core.mesh_bundle import MeshBundle


@pytest.fixture()
def storage_paths():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_gap_persist_test_")
    try:
        yield tmp_dir, f"{tmp_dir}/mesh.db"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _fake_gap(gap_id_hint="x"):
    return DetectedGap(
        gap_type="routing",
        missing_service=f"Svc{gap_id_hint}",
        confidence=0.7,
        source_node={"node_id": "a"},
        target_node={"node_id": "b"},
        evidence=["test evidence"],
    )


def test_mesh_bundle_wires_a_real_knowledge_store_into_gap_detector(storage_paths):
    storage_dir, db_path = storage_paths
    bundle = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    assert bundle.gap_detector._knowledge is not None, (
        "GapDetectionEngine._knowledge ما زالت None — knowledge_store لم يُسلَك فعلياً"
    )
    assert bundle.gap_detector._knowledge is bundle.knowledge_store


def test_gap_survives_restart_and_resolution_is_persisted(storage_paths):
    storage_dir, db_path = storage_paths

    bundle1 = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    gap = _fake_gap()
    bundle1.gap_detector._detected_gaps.append(gap)
    bundle1.gap_detector._persist_gaps([gap])

    # محاكاة إعادة تشغيل حقيقية.
    bundle2 = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    restored = [g for g in bundle2.gap_detector._detected_gaps if g.gap_id == gap.gap_id]
    assert restored, "الفجوة المحفوظة لم تُستعَد بعد إعادة التشغيل"
    assert restored[0].resolved is False

    assert bundle2.gap_detector.mark_resolved(gap.gap_id) is True

    # إعادة تشغيل ثالثة: حالة "resolved" يجب أن تكون قد وُصلت للتخزين.
    bundle3 = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    restored3 = [g for g in bundle3.gap_detector._detected_gaps if g.gap_id == gap.gap_id]
    assert restored3, "الفجوة اختفت بعد mark_resolved + إعادة تشغيل"
    assert restored3[0].resolved is True, (
        "mark_resolved() لم يُزامن حالة resolved=True مع التخزين فعلياً"
    )


def test_loading_persisted_gaps_is_idempotent_and_survives_double_call():
    """_load_persisted_gaps() قد تُستدعى من __init__ ومن set_knowledge_store()
    معاً — يجب ألا تُكرّر نفس الفجوة في self._detected_gaps."""
    tmp_dir = tempfile.mkdtemp(prefix="nsm_gap_idempotent_test_")
    try:
        from knowledge.knowledge_store import KnowledgeStore
        ks = KnowledgeStore(knowledge_dir=f"{tmp_dir}/knowledge")
        gap = _fake_gap("dup")
        ks.write_custom("detected_gaps", {gap.gap_id: gap.to_dict()})

        engine = GapDetectionEngine(knowledge_store=ks)  # يحمّل مرة في __init__
        engine.set_knowledge_store(ks)  # ويُعاد استدعاؤها هنا يدوياً
        matching = [g for g in engine._detected_gaps if g.gap_id == gap.gap_id]
        assert len(matching) == 1, f"توقّعت نسخة واحدة فقط، وجدت {len(matching)}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
