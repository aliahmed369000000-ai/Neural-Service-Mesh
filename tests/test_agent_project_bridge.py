def test_dispatch_rl_status():
    from ai.agent_project_bridge import dispatch_agent_message
    r = dispatch_agent_message("حالة التعلم المعزز")
    assert r is not None
    assert "التعلم المعزز" in r or "RL" in r or "policy" in r.lower()

def test_unified_uses_bridge():
    from ai.agent_categories import UnifiedAgentChat
    u = UnifiedAgentChat()
    resp, meta = u.chat("حالة التعلم المعزز")
    assert meta.get("route_method") == "project_bridge"
    assert resp

def test_integration_status():
    from ai.agent_project_bridge import agent_integration_status
    st = agent_integration_status()
    assert st["bridge"] is True
    assert st["components"].get("model_training_agent") is True
