def test_devops_cycle():
    from ai.autonomous_train_devops import devops_cycle, detect_plateau
    r = devops_cycle()
    assert "plateau" in r
    p = detect_plateau([1.0] * 20)
    assert p["plateau"] is True

def test_quantize_roundtrip():
    import numpy as np
    from ai.quantization_worker import quantize_array, dequantize
    x = np.random.randn(32, 16).astype("float32")
    pack = quantize_array(x, bits=8)
    recon = dequantize({k: pack[k] for k in ("scale", "zero_point")}, pack["q"])
    assert recon.shape == x.shape

def test_gateway_search():
    from ai.mcp_internal_gateway import search_ckg
    r = search_ckg("عدل")
    assert "ok" in r

def test_sensor_bridge():
    from ai.sensors_training_bridge import bridge_cycle
    r = bridge_cycle()
    assert "weak_for_training" in r
