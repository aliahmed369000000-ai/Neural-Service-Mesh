import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_defines_28_unique_nodes_and_ports():
    manifest = json.loads((ROOT / "config" / "mesh_nodes_28.json").read_text(encoding="utf-8"))
    workers = manifest["workers"]
    assert manifest["node_count"] == 28
    assert len(workers) == 27
    ids = [manifest["seed"]["id"]] + [worker["id"] for worker in workers]
    offsets = [manifest["seed"]["port_offset"]] + [worker["port_offset"] for worker in workers]
    assert len(set(ids)) == 28
    assert len(set(offsets)) == 28
    assert offsets == list(range(28))


def test_launcher_uses_manifest_and_allows_28():
    source = (ROOT / "scripts" / "run_local_mesh.py").read_text(encoding="utf-8")
    assert "mesh_nodes_28.json" in source
    assert "MAX_NODES" in source
    assert "NODE_MANIFEST[\"workers\"]" in source
