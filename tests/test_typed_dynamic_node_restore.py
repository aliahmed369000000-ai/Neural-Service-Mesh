"""
تفاعل بين إصلاحين من جلستين منفصلتين:
  - services/generated_service_nodes.py: عُقد self_evolved صارت أصنافاً
    حقيقية (NormalizerNode/ValidatorNode/...) بدل PassThroughNode دائماً.
  - core/mesh_bundle.py._restore_dynamic_nodes: يُحيي عُقد self_evolved
    بعد إعادة تشغيل العملية.

قبل هذا التعديل، _restore_dynamic_nodes كانت تُحيي *كل* عقدة يتيمة كـ
PassThroughNode عام بغضّ النظر عن نوعها الأصلي — كان هذا صحيحاً ومقبولاً
تماماً حين كانت كل عقدة self_evolved أصلاً PassThroughNode، لكنه أصبح
سيُسقط المنطق الحقيقي (تطبيع/تحقّق/تجميع/...) بصمت عند أول إعادة تشغيل
لعقدة مولَّدة حديثاً. هذا الاختبار يبني عقدة NormalizerNode حقيقية،
يسجّلها، يعيد بناء MeshBundle (محاكاة إعادة تشغيل حقيقية على نفس
التخزين)، ويتحقق أنها تُحيا بنفس الصنف الحقيقي وأن process() الحقيقية
(وليس تمريراً بلا تغيير) لا تزال تعمل بعد الإحياء.
"""
from __future__ import annotations

import shutil
import tempfile

import pytest

from core.mesh_bundle import MeshBundle
from core.node import NodeState
from services.generated_service_nodes import NormalizerNode, ValidatorNode


@pytest.fixture()
def storage_paths():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_typed_node_restore_test_")
    try:
        yield tmp_dir, f"{tmp_dir}/mesh.db"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_normalizer_node_revived_with_real_class_after_restart(storage_paths):
    storage_dir, db_path = storage_paths

    bundle1 = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    node = NormalizerNode(name="self-evolved-normalizer")
    node.execute({"raw_data": {" X ": "  y  "}})  # تنفيذ حقيقي واحد قبل التخزين
    node_id = bundle1.registry.register(node)

    # عملية ثانية بنفس التخزين — محاكاة إعادة تشغيل حقيقية.
    bundle2 = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    revived = bundle2.registry.get(node_id)

    assert revived is not None
    assert isinstance(revived, NormalizerNode), (
        f"أُحييت كـ {type(revived).__name__} — نزلت إلى PassThroughNode بدل النوع الحقيقي"
    )
    assert revived.state == NodeState.ACTIVE
    assert revived._execution_count == 1

    # المنطق الحقيقي لا يزال يعمل بعد الإحياء (وليس تمريراً بلا تغيير).
    out = revived.execute({"raw_data": {" A ": "  B  ", "n": None}})
    assert out["normalized_data"] == {"a": "B"}
    assert out["validation_report"]["changes_made"] > 0


def test_validator_node_also_revived_with_real_class(storage_paths):
    storage_dir, db_path = storage_paths

    bundle1 = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    node = ValidatorNode(name="self-evolved-validator")
    node_id = bundle1.registry.register(node)

    bundle2 = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    revived = bundle2.registry.get(node_id)

    assert isinstance(revived, ValidatorNode)
    out = revived.execute({"data": {}, "rules": {"required": ["name"]}})
    assert out["valid"] is False
    assert any("name" in e for e in out["errors"])
