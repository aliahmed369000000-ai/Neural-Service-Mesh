"""سجل تاريخي لدورات تشخيص العُقد."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from ai.nodes_diagnose_store import NodesDiagnoseStore
from core.mesh_bundle import MeshBundle


def test_store_log_and_trend(tmp_path):
    db = tmp_path / "diag.db"
    store = NodesDiagnoseStore(db_path=db)
    for i in range(3):
        store.log_cycle({
            "ts": f"2026-01-0{i+1}T00:00:00+00:00",
            "scanned": 10,
            "errors": 0,
            "avg_reputation": 0.5 + i * 0.1,
            "effective_low_rep_threshold": 0.15,
            "low_reputation": [{"node_id": "x"}] * (2 - min(i, 2)),
            "recovered": [{"node_id": "y"}] if i else [],
            "high_unread": [],
        })
    recent = store.get_recent(10)
    assert len(recent) == 3
    tr = store.trend(10)
    assert tr["points"] == 3
    assert len(tr["low_rep_count"]) == 3
    sm = store.summary()
    assert sm["cycles"] == 3


def test_mesh_bundle_persists_diagnose_history():
    tmp = tempfile.mkdtemp(prefix="nsm_diag_hist_")
    try:
        b = MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db")
        b.run_nodes_diagnose_cycle()
        b.run_nodes_diagnose_cycle()
        hist = b.get_nodes_diagnose_history(limit=10)
        assert hist["trend"]["points"] >= 2
        assert hist["summary"].get("cycles", 0) >= 2
        db = Path(tmp) / "nodes_diagnose_history.db"
        assert db.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
