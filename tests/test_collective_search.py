# -*- coding: utf-8 -*-
from __future__ import annotations
import asyncio, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ai import mesh_task_protocol as mt
from ai.mesh_job_orchestrator import MeshJobOrchestrator

def test_search_chunk_local():
    r = mt.execute_search_chunk({
        "query": "نصاب",
        "documents": [
            {"source_id": "a", "text": "الشبكات الموزعة والنصاب يرفعان الثقة."},
            {"source_id": "b", "text": "الطقس جميل اليوم في المدينة."},
        ],
        "top_k": 2,
    })
    assert r["ok"] and r["hit_count"] >= 1
    assert r["hits"][0]["source_id"] == "a"
    assert r["hits"][0]["source_hash"]
    print("✅ search_chunk", r["hits"][0]["snippet"][:50])

def test_collective_search_mock():
    class N:
        node_id = "n"
        def _get_active_peers_list(self, require_capabilities=None):
            return [{"id": "w1", "host": "h", "port": 1, "capabilities": ["text"]}]
        def get_reputation(self, p): return {"score": 0}
        def update_reputation(self, *a, **k): return {"score": 0}
        def _load_state(self): return {"nodes": {}, "job_decisions": {}}
        def _save_state(self, s): pass
        async def request_from_peer(self, host, port, kind, data, timeout=12.0):
            res = mt.dispatch_task(kind, data)
            return {"ok": True, "acked": True, "result": res, "mode": "rpc"}
    orch = MeshJobOrchestrator(N())
    async def run():
        return await orch.submit_collective_search(
            "تعليم ذكاء",
            [
                {"source_id": "edu", "text": "الذكاء الاصطناعي في التعليم يحسّن التخصيص."},
                {"source_id": "food", "text": "وصفات الطبخ التقليدية متنوعة."},
            ],
            top_k=2,
            n_workers=1,
            worker_list=[{"peer_id": "w1", "host": "h", "port": 1}],
            then_summarize=True,
            quorum=1,
        )
    rep = asyncio.run(run())
    assert rep["ok"] and rep["hit_count"] >= 1
    assert rep["hits"][0]["source_id"] == "edu"
    assert rep.get("summary") and rep["summary"].get("ok")
    print("✅ collective_search+summary", rep["hit_count"], "hits")

if __name__ == "__main__":
    test_search_chunk_local()
    test_collective_search_mock()
    print("🎉 search tests passed")
