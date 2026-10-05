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


def test_export_csv_and_spike(tmp_path):
    store = NodesDiagnoseStore(db_path=tmp_path / "d.db")
    for i in range(5):
        store.log_cycle({
            "ts": f"t{i}",
            "scanned": 5,
            "errors": 0,
            "avg_reputation": 0.4,
            "effective_low_rep_threshold": 0.15,
            "low_reputation": [{"node_id": "a"}] * 1,  # low count baseline
            "recovered": [],
            "high_unread": [],
        })
    csv = store.export_csv(10)
    assert "low_rep_count" in csv
    assert csv.count("\n") >= 6
    # spike when current is high vs avg ~1
    sp = store.spike_vs_average(5, lookback=10)
    assert sp["spike"] is True
    assert sp["current"] == 5
    sp2 = store.spike_vs_average(1, lookback=10)
    assert sp2["spike"] is False


def test_prune_old_keeps_last_n(tmp_path):
    store = NodesDiagnoseStore(db_path=tmp_path / "p.db")
    for i in range(20):
        store.log_cycle({
            "ts": f"t{i}",
            "scanned": 1,
            "errors": 0,
            "avg_reputation": 0.2,
            "effective_low_rep_threshold": 0.15,
            "low_reputation": [],
            "recovered": [],
            "high_unread": [],
        })
    deleted = store.prune_old(keep_last=10)
    assert deleted == 10
    assert len(store.get_recent(50)) == 10


def test_weekly_report_shape(tmp_path):
    store = NodesDiagnoseStore(db_path=tmp_path / "w.db")
    for i in range(4):
        store.log_cycle({
            "ts": f"t{i}",
            "scanned": 3,
            "errors": 0,
            "avg_reputation": 0.3,
            "effective_low_rep_threshold": 0.15,
            "low_reputation": [{"node_id": "x"}] * (i % 3),
            "recovered": [{"node_id": "y"}] if i else [],
            "high_unread": [],
        })
    rep = store.weekly_report(days=7)
    assert rep["cycles"] >= 1
    assert "avg_low_rep" in rep
    assert "total_recovered" in rep
    assert "approx_spike_events" in rep


def test_diagnose_history_keep_last_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NSM_DIAGNOSE_HISTORY_KEEP", "25")
    b = MeshBundle(storage_dir=str(tmp_path), db_path=str(tmp_path / "m.db"))
    assert b._diagnose_history_keep_last() == 25
    hist = b.get_nodes_diagnose_history(limit=5)
    assert hist.get("keep_last") == 25
    assert "weekly" in hist
