"""
اختبار: RoutingEngine لم تكن مربوطة بمسار /process؛ AIDecisionLayer
(المستخدمة فعلياً في ExecutionEngine.run_between) كانت تختار المسار
بإحصاءات في الذاكرة فقط تبدأ فارغة عند كل إعادة تشغيل، ولا تقرأ ذاكرة
المسارات المحفوظة (MemoryEngine/SQLite) إطلاقاً.

يتحقق: (1) MeshBundle تبني RoutingEngine مربوطة بنفس المحركات وتمرّرها
لـAIDecisionLayer، (2) الاختيار يتبع ذاكرة المسارات المتعلَّمة فعلاً
(مسار نجح مراراً يُفضَّل على بديل فشل)، (3) استثناء RoutingEngine يرجع
للمنطق القديم بدل إسقاط الطلب، (4) أوزان RoutingEngine لا تُكتَب خارج
storage_dir (لا تلويث لمجلد العمل).
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from core.mesh_bundle import MeshBundle


@pytest.fixture()
def bundle():
    d = tempfile.mkdtemp(prefix="nsm_routing_wiring_")
    cwd = os.getcwd()
    try:
        yield MeshBundle(storage_dir=d, db_path=f"{d}/mesh.db"), d
    finally:
        os.chdir(cwd)
        shutil.rmtree(d, ignore_errors=True)


def _diamond(b):
    """root → (A | B) → target: مساران بطول متساوٍ."""
    ids = {}
    for name in ("rt-start", "rt-a", "rt-b", "rt-end"):
        ids[name] = f"{name}-node-id-0000"
        b.graph.add_node(ids[name], {"node_id": ids[name], "name": name})
    for s, t in (("rt-start", "rt-a"), ("rt-start", "rt-b"),
                 ("rt-a", "rt-end"), ("rt-b", "rt-end")):
        b.graph.add_edge(ids[s], ids[t])
    return ids


def _run(b, path, ok):
    b.memory_engine.learn_from_run({
        "run_id": "x", "status": "success" if ok else "failed",
        "total_duration_ms": 5.0, "path": path,
        "steps": [{"node_id": n, "node_name": n,
                   "status": "success" if ok else "failed", "duration_ms": 1.0}
                  for n in path],
    })


def test_routing_engine_is_wired_into_decision_layer(bundle):
    b, _ = bundle
    assert b.ai_decision._routing is b.routing_engine
    assert b.routing_engine._memory is b.memory_engine
    assert b.routing_engine._scoring is b.scoring_engine
    assert b.routing_engine._knowledge is b.knowledge_store


def test_choice_follows_learned_route_memory(bundle):
    b, _ = bundle
    ids = _diamond(b)
    good = [ids["rt-start"], ids["rt-b"], ids["rt-end"]]
    bad = [ids["rt-start"], ids["rt-a"], ids["rt-end"]]
    for _ in range(6):
        _run(b, good, True)
        _run(b, bad, False)
    assert b.ai_decision.choose_path(ids["rt-start"], ids["rt-end"]) == good


def test_routing_failure_falls_back_to_heuristics(bundle):
    b, _ = bundle
    ids = _diamond(b)

    def boom(*a, **k):
        raise RuntimeError("routing down")

    b.routing_engine.choose_route = boom
    path = b.ai_decision.choose_path(ids["rt-start"], ids["rt-end"])
    assert path and path[0] == ids["rt-start"] and path[-1] == ids["rt-end"]


def test_weights_are_isolated_under_storage_dir(bundle):
    b, d = bundle
    assert b.routing_engine._WEIGHTS_PATH.startswith(d)
    assert b.routing_engine._DEEP_NETWORK_DIR.startswith(d)
