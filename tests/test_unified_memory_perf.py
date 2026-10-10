# -*- coding: utf-8 -*-
"""
أداء الذاكرة الموحدة (ANN + Sharding) عبر LivingMeshNode.memory.

الاختبار القديم كان 'async def' (لا يعمل بلا pytest-asyncio) ويستدعي
_generate_simulated_embedding وsemantic_query — حُذفتا عمداً من living_mesh
(كوميت b0f6cc4: حذف قدرات وهمية). أُعيدت كتابته على الواجهة الفعلية:
UnifiedMemoryManager.store_experience/semantic_search/get_memory_stats،
بمتجهات حتمية يولّدها الاختبار نفسه، وبتأكيدات حقيقية (بما فيها أن
الاستعلام بمتجه خبرة محفوظة يجدها).
"""
import shutil
import tempfile
import time

import numpy as np

from ai.living_mesh import LivingMeshNode

DIM = 1536
NUM_EXPS = 50


def _embedding(seed: int) -> list:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM)
    return (v / np.linalg.norm(v)).tolist()


def test_performance():
    d = tempfile.mkdtemp(prefix="nsm_unified_mem_")
    try:
        node = LivingMeshNode(node_id="perf_tester", data_dir=d)

        start = time.time()
        for i in range(NUM_EXPS):
            exp = {
                "kind": "perf_test",
                "data": {"value": i, "content": f"خبرة تجريبية رقم {i} للتحقق من سرعة التخزين المجزأ"},
                "timestamp": time.time(),
            }
            node.memory.store_experience(exp, embedding=_embedding(i))
        save_duration = time.time() - start

        start = time.time()
        results = node.memory.semantic_search(_embedding(7), top_k=3)
        search_duration = time.time() - start

        stats = node.memory.get_memory_stats()
        assert stats["total_experiences"] >= NUM_EXPS
        assert stats["indexed_vectors"] >= 1
        assert results, "البحث الدلالي لم يُرجع نتائج"
        assert len(results) <= 3
        # متجه الاستعلام هو نفسه متجه الخبرة رقم 7 → يجب أن تكون ضمن النتائج
        assert any(
            (r.get("data") or r.get("experience", {}).get("data") or {}).get("value") == 7
            for r in results
        ), f"الخبرة المطابقة لم تُسترجَع: {results[:1]}"
        # حدود أداء واسعة جداً (تكشف الانهيار لا التذبذب)
        assert save_duration < 30 and search_duration < 5
    finally:
        shutil.rmtree(d, ignore_errors=True)
