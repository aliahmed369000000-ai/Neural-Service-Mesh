def test_predictive_campaign():
    from world_model.predictive_sim import full_campaign_sim
    r = full_campaign_sim("العلم نور")
    assert "avg_engagement" in r
    assert "platforms" in r

def test_active_retrain_plan():
    from ai.active_retrain_loop import plan_retrain
    p = plan_retrain(5)
    assert p["epochs"] == 5
    assert "steps_ar" in p

def test_mcp_monetization_open_dev(monkeypatch):
    import os
    monkeypatch.setenv("NSM_MCP_OPEN", "1")
    from mcp_server.monetization import authenticate_mcp_key
    ok, meta = authenticate_mcp_key(None)
    assert ok is True

def test_pipeline_ensemble_flag():
    import inspect
    from ai.reasoning_pipeline import ReasoningPipeline
    assert "use_deep_routing" in str(inspect.signature(ReasoningPipeline.__init__))
