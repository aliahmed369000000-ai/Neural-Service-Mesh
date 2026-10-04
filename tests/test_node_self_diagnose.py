"""تطوير العقد: self_diagnose + last_terminal_check + inbox_summary."""
from __future__ import annotations

import shutil
import tempfile

import pytest

from core.mesh_bundle import MeshBundle
from core.node_hands import LEFT


@pytest.fixture()
def bundle():
    tmp = tempfile.mkdtemp(prefix="nsm_diag_")
    try:
        yield MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _node(b):
    return b.registry.get(list(b.role_node_ids.values())[0])


def test_terminal_check_appears_in_node_status(bundle):
    n = _node(bundle)
    r = n.use_hand(LEFT, "terminal_run_safe", cmd="python3 --version")
    assert r.ok
    st = n.use_hand(LEFT, "node_status")
    assert st.ok
    ltc = st.output.get("last_terminal_check")
    assert isinstance(ltc, dict)
    assert "cmd" in ltc and "at" in ltc


def test_self_diagnose_bundle(bundle):
    n = _node(bundle)
    d = n.use_hand(LEFT, "self_diagnose")
    assert d.ok
    out = d.output
    assert out["layer"] in ("node-self-diagnose-v1", "node-self-diagnose-v2")
    assert "collective_health" in out
    assert "status" in out and "mesh_health" in out
    assert "neighbors" in out and "inbox" in out


def test_inbox_summary_structure(bundle):
    n = _node(bundle)
    r = n.use_hand(LEFT, "inbox_summary")
    assert r.ok
    assert r.output["node_id"] == n.node_id
    assert "unread_total" in r.output
    assert "by_topic" in r.output
