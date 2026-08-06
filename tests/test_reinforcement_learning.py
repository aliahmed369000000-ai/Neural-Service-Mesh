def test_rl_step_updates_policy():
    from ai.reinforcement_learning import RoutingPolicy, rl_step_on_weights, ACTION_KEYS
    p0 = RoutingPolicy()
    steps0 = p0.steps
    w = {k: 0.25 for k in ACTION_KEYS}
    r = rl_step_on_weights(w, quality={"overall": 0.8}, answer_text="إجابة جيدة كافية للاختبار", explore=True)
    assert "reward" in r
    assert r["reward"] > 0
    p1 = RoutingPolicy()
    assert p1.steps >= steps0 + 1

def test_rl_commands():
    from ai.reinforcement_learning import handle_rl_command
    assert handle_rl_command("تفعيل تعلم معزز")
    assert handle_rl_command("حالة التعلم المعزز")
