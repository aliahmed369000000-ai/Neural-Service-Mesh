"""اختبارات وحدة لبروتوكول المهام الموزّعة — بدون شبكة."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ai import mesh_task_protocol as mt


def test_submodel_and_merge():
    r1 = mt.execute_submodel_train({"layer_name": "L1", "layer_index": 1, "steps": 5})
    r2 = mt.execute_submodel_train({"layer_name": "L2", "layer_index": 2, "steps": 5})
    assert r1["ok"] and r2["ok"]
    merged = mt.merge_submodel_results([r1, r2])
    assert merged["layers_count"] == 2
    print("✅ submodel", merged["mean_loss"])


def test_inference():
    r = mt.execute_inference({"prompt": "مرحبا بالشبكة", "model_hint": "llama"})
    assert r["ok"] and "output" in r
    print("✅ inference")


def test_eval_merge():
    samples = [{"x": 0.8, "y": 1.0}, {"x": 0.2, "y": 0.0}] * 5
    r = mt.execute_model_eval({"samples": samples, "metric": "both", "weights": [0.9] * 8})
    assert r["ok"] and "accuracy" in r and "loss" in r
    m = mt.merge_eval_results([r, r])
    assert m["n_samples"] == r["n_samples"] * 2
    print("✅ eval", r["accuracy"], r["loss"])


def test_map_reduce():
    r1 = mt.execute_map({"chunk_id": "c1", "lines": ["hello world", "hello mesh"], "op": "wordcount"})
    r2 = mt.execute_map({"chunk_id": "c2", "lines": ["world mesh network"], "op": "wordcount"})
    reduced = mt.reduce_map_results("wordcount", [r1, r2])
    assert reduced["counts"].get("hello") == 2
    assert reduced["counts"].get("world") == 2
    print("✅ map-reduce", reduced["counts"])


def test_sim():
    r = mt.execute_sim_chunk({"x0": 1.0, "t0": 0, "t1": 0.5, "dt": 0.05, "params": {"k": 0.5, "mode": "decay"}})
    assert r["ok"] and r["final"]["x"] < 1.0
    print("✅ sim", r["final"])


def test_keyspace():
    # ابنِ هدفاً معروفاً
    import hashlib
    n_secret = 12345
    target = hashlib.sha256(f"nsm{n_secret}".encode()).hexdigest()
    r = mt.execute_keyspace_scan({
        "start": 12300, "end": 12400, "prefix": "nsm", "target_hash": target, "max_checks": 5000
    })
    assert r["ok"] and r["found"] and r["found"]["n"] == n_secret
    print("✅ keyspace found", r["found"]["n"])


def test_dispatch():
    assert mt.dispatch_task(mt.KIND_INFERENCE, {"prompt": "x"})["ok"]
    assert mt.result_kind_for(mt.KIND_MAP) == mt.KIND_MAP_RESULT
    print("✅ dispatch")


if __name__ == "__main__":
    test_submodel_and_merge()
    test_inference()
    test_eval_merge()
    test_map_reduce()
    test_sim()
    test_keyspace()
    test_dispatch()
    print("🏆 mesh task protocol tests passed")
