"""
فحص مركزي: دورة سؤال → ReasoningPipeline → مخرجات آمنة بلا انهيار.
"""
from __future__ import annotations

import inspect


def test_reasoning_pipeline_imports_and_signature():
    from ai.reasoning_pipeline import ReasoningPipeline
    sig = str(inspect.signature(ReasoningPipeline.__init__))
    assert "use_deep_routing" in sig


def test_pipeline_answer_smoke_arabic():
    from ai.reasoning_pipeline import ReasoningPipeline
    pipe = ReasoningPipeline(train_on_query=False, use_deep_routing=True, record_episodes=False)
    result = pipe.answer("ما الأمانة؟")
    assert result is not None
    assert hasattr(result, "answer_text") or hasattr(result, "to_dict")
    weights = getattr(result, "decision_weights", None)
    assert isinstance(weights, dict)
    for k in ("W_SEMANTIC", "W_SCORE", "W_MEMORY", "W_TOPOLOGY"):
        assert k in weights


def test_pipeline_ensemble_flags_or_graceful():
    from ai.reasoning_pipeline import ReasoningPipeline
    pipe = ReasoningPipeline(train_on_query=False, use_deep_routing=True, record_episodes=False)
    result = pipe.answer("العدل أساس الملك")
    w = getattr(result, "decision_weights", {}) or {}
    # إما تفعيل ensemble أو على الأقل أوزان أساسية صالحة
    assert all(isinstance(w.get(k), (int, float)) for k in ("W_SEMANTIC", "W_SCORE", "W_MEMORY", "W_TOPOLOGY"))


def test_neural_core_forward_smoke():
    from ai.neural_core import get_default_core
    import numpy as np
    core = get_default_core()
    x = np.zeros(getattr(core, "input_dim", 784) if hasattr(core, "input_dim") else 784, dtype=np.float64)
    # try common dims
    try:
        out = core.forward(x)
    except Exception:
        x = np.zeros(784, dtype=np.float64)
        out = core.forward(x)
    assert out is not None
    assert len(np.asarray(out).ravel()) >= 4


def test_social_pre_publish_blocks_toxic():
    from ai.social_swarm import pre_publish_check
    bad = pre_publish_check("اقتل الجميع إرهاب")
    assert bad.get("ok") is False
    good = pre_publish_check("العلم نور والجهل ظلام")
    assert good.get("ok") is True


def test_train_state_readable():
    from pathlib import Path
    import json
    p = Path("ckg_train_state_v3.json")
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        assert "loss_history_tail" in data or "runs" in data
