"""اختبارات منطق التخزين الموزّع و Content-ID (بدون شبكة / بدون aiohttp)."""
import hashlib
import base64
import json
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_content_id_roundtrip(tmp_path: Path = None):
    root = Path("/tmp/nsm_content_test")
    root.mkdir(parents=True, exist_ok=True)
    data = b"model.pth-weights-binary-demo"
    cid = sha256(data)
    path = root / f"{cid}.bin"
    path.write_bytes(data)
    loaded = path.read_bytes()
    assert sha256(loaded) == cid
    assert cid == sha256(base64.b64decode(base64.b64encode(data)))
    print("✅ content-id roundtrip", cid[:16])


def test_capability_filter():
    peers = [
        {"id": "a", "capabilities": ["GPU_HIGH", "storage"], "status": "online"},
        {"id": "b", "capabilities": ["CPU", "text"], "status": "online"},
        {"id": "c", "capabilities": ["GPU_HIGH"], "status": "offline"},
    ]
    need = {"GPU_HIGH"}
    filtered = [
        p for p in peers
        if p.get("status") == "online" and need.issubset(set(p.get("capabilities") or []))
    ]
    assert [p["id"] for p in filtered] == ["a"]
    print("✅ capability filter GPU_HIGH")


def test_checkpoint_hash_integrity():
    original = b"X" * 1000
    h = sha256(original)
    # simulate corruption
    corrupt = b"Y" + original[1:]
    assert sha256(corrupt) != h
    print("✅ checkpoint hash detects corruption")


if __name__ == "__main__":
    test_content_id_roundtrip()
    test_capability_filter()
    test_checkpoint_hash_integrity()
    print("🏆 storage protocol logic tests passed")
