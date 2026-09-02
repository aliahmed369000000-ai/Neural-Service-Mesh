# -*- coding: utf-8 -*-
from __future__ import annotations
import asyncio, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ai import mesh_task_protocol as mt
from ai.mesh_job_orchestrator import MeshJobOrchestrator

def test_summarize_local():
    r = mt.execute_summarize_chunk({
        "source_id": "s1",
        "text": "الذكاء الاصطناعي يغيّر التعليم. الشبكات الموزعة تزيد الموثوقية. الأمان أولوية.",
        "query": "شبكات",
    })
    assert r["ok"] and r["source_hash"] and r["summary"]
    print("✅ summarize_chunk", r["summary"][:60])

def test_collective_with_mock():
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
        return await orch.submit_collective_summary(
            "تعليم",
            [
                {"source_id": "a", "text": "التعليم يحتاج أدوات حديثة. الذكاء الاصطناعي يساعد المعلم."},
                {"source_id": "b", "text": "الشبكات الموزعة تحسّن التوافر. التحقق يمنع التزوير."},
            ],
            n_workers=1,
            redundancy=1,
            quorum=1,
            worker_list=[{"peer_id": "w1", "host": "h", "port": 1}],
        )
    rep = asyncio.run(run())
    assert rep["ok"] and rep["sources_ok"] == 2
    assert len(rep["provenance"]) == 2
    assert all(p.get("source_hash") for p in rep["provenance"])
    print("✅ collective_summary", rep["combined_summary"][:80])

if __name__ == "__main__":
    test_summarize_local()
    test_collective_with_mock()
    print("🎉 collective tests passed")
