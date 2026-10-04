"""دورة تشخيص العُقد الدورية + تنبيهات العتبات."""
from __future__ import annotations

import shutil
import tempfile

import pytest

from core.mesh_bundle import MeshBundle


@pytest.fixture()
def bundle():
    tmp = tempfile.mkdtemp(prefix="nsm_diag_cycle_")
    try:
        yield MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_nodes_diagnose_cycle_stores_summary(bundle):
    out = bundle.run_nodes_diagnose_cycle()
    assert out["scanned"] >= 1
    assert "ts" in out
    summary = bundle.get_nodes_diagnose_summary()
    assert summary.get("layer") == "mesh-nodes-diagnose-cycle-v1"
    assert summary.get("scanned") == out["scanned"]
    assert isinstance(summary.get("nodes"), list)
    assert len(summary["nodes"]) >= 1


def test_summary_includes_nodes_diagnose(bundle):
    bundle.run_nodes_diagnose_cycle()
    s = bundle.summary()
    assert "nodes_diagnose" in s
    assert s["nodes_diagnose"].get("scanned", 0) >= 1


def test_system_hub_check_mesh_nodes():
    from ai.system_hub import check_mesh_nodes
    # may use process singleton — just ensure structure
    r = check_mesh_nodes()
    assert "ok" in r and "detail" in r
