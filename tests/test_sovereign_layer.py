def test_mesh_broadcast_and_recv():
    from ai.cosmic_mesh import mesh_broadcast, mesh_recv, mesh_status
    r = mesh_broadcast("test", {"x": 1})
    assert r["ok"] is True
    inbox = mesh_recv()
    assert inbox["n"] >= 1
    assert mesh_status()["node_id"]

def test_microkernel_boot():
    from ai.cognitive_microkernel import k_boot, k_status
    k_boot()
    st = k_status()
    assert st["boots"] >= 1

def test_asic_files_exist():
    from pathlib import Path
    assert Path("hardware/nsm_asic/matmul_pe.v").is_file()
