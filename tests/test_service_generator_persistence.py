"""
ai/service_generator.py::ServiceGeneratorEngine كانت تُبنى في
core/mesh_bundle.py بلا knowledge_store (بحث منفصل عن إصلاح
CapabilityMarketplace في commit 6c2abb4) — _persist_spec() مكتوبة
بالكامل لكنها كانت no-op دائماً (self._knowledge is None)، وحتى لو
أُصلح ذلك، لا توجد أي دالة استعادة مقابلة. كل GeneratedServiceSpec
(بما فيها svc_type/status/confidence/gap_context) تختفي عند إعادة
التشغيل، رغم أن العُقد الحية نفسها المُنشأة منها تُستعاد فعلياً بنوعها
الحقيقي (إصلاح سابق: core/mesh_bundle.py::_restore_dynamic_nodes +
services/generated_service_nodes.py).
"""
from __future__ import annotations

import shutil
import tempfile

import pytest

from ai.service_generator import ServiceGeneratorEngine
from knowledge.knowledge_store import KnowledgeStore


@pytest.fixture()
def knowledge_store():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_service_generator_ks_test_")
    try:
        yield KnowledgeStore(knowledge_dir=tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _make_gap():
    return {
        "source_node": {"name": "Src", "capability": "produce"},
        "target_node": {"name": "Tgt", "capability": "consume"},
        "missing_service": "Normalizer",
        "confidence": 0.85,
        "gap_type": "capability_gap",
    }


class TestServiceGeneratorPersistence:
    def test_spec_survives_new_engine_instance(self, knowledge_store):
        gen1 = ServiceGeneratorEngine(knowledge_store=knowledge_store)
        spec = gen1.generate_for_gap(_make_gap())
        assert spec is not None

        gen2 = ServiceGeneratorEngine(knowledge_store=knowledge_store)
        restored = gen2.get_spec(spec.spec_id)
        assert restored is not None
        assert restored.name == spec.name
        assert restored.svc_type == "normalizer"
        assert restored.confidence == pytest.approx(spec.confidence)
        assert any(s["spec_id"] == spec.spec_id for s in gen2.list_generated())

    def test_status_change_persists_across_restart(self, knowledge_store):
        gen1 = ServiceGeneratorEngine(knowledge_store=knowledge_store)
        spec = gen1.generate_for_gap(_make_gap())
        spec.status = "active"
        gen1._persist_spec(spec)  # instantiate_spec يفعل هذا فعلياً بعد ضبط status

        gen2 = ServiceGeneratorEngine(knowledge_store=knowledge_store)
        assert gen2.get_spec(spec.spec_id).status == "active"

    def test_set_knowledge_store_after_construction_also_restores(self, knowledge_store):
        gen1 = ServiceGeneratorEngine(knowledge_store=knowledge_store)
        spec = gen1.generate_for_gap(_make_gap())

        gen2 = ServiceGeneratorEngine()  # بلا knowledge_store وقت الإنشاء
        assert gen2.get_spec(spec.spec_id) is None
        gen2.set_knowledge_store(knowledge_store)
        assert gen2.get_spec(spec.spec_id) is not None

    def test_no_knowledge_store_still_works_in_memory_only(self):
        gen = ServiceGeneratorEngine()  # بلا knowledge_store — لا يجب أن ينكسر
        spec = gen.generate_for_gap(_make_gap())
        assert gen.get_spec(spec.spec_id) is spec
