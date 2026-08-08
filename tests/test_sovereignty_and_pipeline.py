"""اختبارات دخان لحلقة السيادة وDeepRouting في pipeline."""
from __future__ import annotations


def test_environment_training_safety():
    from world_model.environment_model import EnvironmentModel
    env = EnvironmentModel(model_dir="./world_model")
    report = env.assess_training_safety("run_training_loop", estimated_vram_mb=512)
    assert "decision" in report
    assert "green_light" in report
    assert isinstance(report.get("risks"), list)


def test_reasoning_pipeline_accepts_deep_routing_flag():
    import inspect
    from ai.reasoning_pipeline import ReasoningPipeline
    sig = str(inspect.signature(ReasoningPipeline.__init__))
    assert "use_deep_routing" in sig


def test_sovereignty_status():
    from ai.sovereignty_loop import sovereignty_status
    st = sovereignty_status()
    assert "components" in st
    assert "reasoning_pipeline" in st["components"]


def test_knowledge_pulse_runs():
    from ai.sovereignty_loop import knowledge_pulse
    r = knowledge_pulse()
    assert "at" in r
    assert "n_events" in r
