"""
Reinforcement Learning Agent Layer — تفعيل التعلم المعزز
=======================================================
سياسة بسيطة (contextual bandit) على أوزان التوجيه الأربعة:
  state  ≈ متجه سياق مختصر (أو hash الميزات)
  action ≈ تعديل طفيف على W_SEMANTIC/SCORE/MEMORY/TOPOLOGY
  reward ≈ جودة الإجابة (score_episode / heuristic)

يدعم:
  • تسجيل (s, a, r) في buffer
  • تحديث سياسة softmax على تفضيلات الأوزان
  • دورة RL من episodes المخزّنة
  • أوامر عربية: «تفعيل تعلم معزز» / «دورة RL»
"""
from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("NSM-RL")

ROOT = Path(__file__).resolve().parent.parent
RL_DIR = ROOT / "artifacts" / "model_training" / "rl"
RL_DIR.mkdir(parents=True, exist_ok=True)
POLICY_PATH = RL_DIR / "routing_policy.json"
BUFFER_PATH = RL_DIR / "rl_buffer.jsonl"

ACTION_KEYS = ("W_SEMANTIC", "W_SCORE", "W_MEMORY", "W_TOPOLOGY")
# 5 إجراءات منفصلة: تعزيز أحد الأبعاد أو البقاء
ACTION_NAMES = [
    "boost_semantic",
    "boost_score",
    "boost_memory",
    "boost_topology",
    "keep_balanced",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _softmax(logits: List[float], temperature: float = 1.0) -> List[float]:
    t = max(0.05, float(temperature))
    m = max(logits)
    exps = [math.exp((x - m) / t) for x in logits]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


@dataclass
class RLConfig:
    learning_rate: float = 0.15
    temperature: float = 1.0
    baseline_momentum: float = 0.9
    delta: float = 0.12  # مقدار تعزيز الوزن عند اختيار إجراء


class RoutingPolicy:
    """سياسة softmax على 5 إجراءات لتعديل أوزان التوجيه."""

    def __init__(self, config: Optional[RLConfig] = None):
        self.config = config or RLConfig()
        self.logits = [0.0] * len(ACTION_NAMES)
        self.baseline = 0.5
        self.steps = 0
        self.load()

    def load(self) -> None:
        if not POLICY_PATH.is_file():
            return
        try:
            data = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
            self.logits = list(data.get("logits") or self.logits)
            self.baseline = float(data.get("baseline", self.baseline))
            self.steps = int(data.get("steps", 0))
        except Exception as e:
            logger.warning("policy load: %s", e)

    def save(self) -> None:
        payload = {
            "logits": self.logits,
            "baseline": self.baseline,
            "steps": self.steps,
            "updated_at": _now(),
            "action_names": ACTION_NAMES,
        }
        POLICY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def probs(self) -> List[float]:
        return _softmax(self.logits, self.config.temperature)

    def select_action(self, explore: bool = True) -> Tuple[int, str]:
        probs = self.probs()
        if explore:
            idx = random.choices(range(len(ACTION_NAMES)), weights=probs, k=1)[0]
        else:
            idx = int(np.argmax(probs))
        return idx, ACTION_NAMES[idx]

    def apply_action_to_weights(self, weights: Dict[str, float], action_idx: int) -> Dict[str, float]:
        w = {k: float(weights.get(k, 0.25)) for k in ACTION_KEYS}
        d = self.config.delta
        if action_idx == 0:
            w["W_SEMANTIC"] += d
        elif action_idx == 1:
            w["W_SCORE"] += d
        elif action_idx == 2:
            w["W_MEMORY"] += d
        elif action_idx == 3:
            w["W_TOPOLOGY"] += d
        # keep_balanced: mild normalize only
        s = sum(w.values()) or 1.0
        return {k: round(v / s, 6) for k, v in w.items()}

    def update(self, action_idx: int, reward: float) -> Dict[str, float]:
        """REINFORCE بخط أساس متحرك."""
        advantage = float(reward) - self.baseline
        self.baseline = (
            self.config.baseline_momentum * self.baseline
            + (1.0 - self.config.baseline_momentum) * float(reward)
        )
        probs = self.probs()
        # gradient of log π(a): for chosen a increase, others relative
        lr = self.config.learning_rate
        for i in range(len(self.logits)):
            grad = (1.0 - probs[i]) if i == action_idx else (-probs[i])
            self.logits[i] += lr * advantage * grad
        self.steps += 1
        self.save()
        return {
            "reward": reward,
            "advantage": round(advantage, 4),
            "baseline": round(self.baseline, 4),
            "probs": [round(p, 4) for p in self.probs()],
            "steps": self.steps,
        }


def compute_reward_from_quality(quality: Optional[Dict[str, float]] = None, answer_text: str = "") -> float:
    if quality and isinstance(quality, dict):
        # score_episode style keys
        vals = []
        for k in ("overall", "concept_coverage", "relation_coverage", "memory_quality", "confidence"):
            if k in quality and quality[k] is not None:
                vals.append(float(quality[k]))
        if vals:
            return float(max(0.0, min(1.0, sum(vals) / len(vals))))
    # heuristic
    score = 0.4
    if len(answer_text or "") > 60:
        score += 0.2
    if len(answer_text or "") > 150:
        score += 0.1
    return float(max(0.0, min(1.0, score)))


def append_transition(state: Dict[str, Any], action: str, reward: float, meta: Optional[dict] = None) -> None:
    row = {
        "at": _now(),
        "state": state,
        "action": action,
        "reward": reward,
        "meta": meta or {},
    }
    with BUFFER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rl_step_on_weights(
    base_weights: Dict[str, float],
    quality: Optional[Dict[str, float]] = None,
    answer_text: str = "",
    explore: bool = True,
) -> Dict[str, Any]:
    """خطوة RL واحدة: اختر إجراء → عدّل الأوزان → احسب مكافأة → حدّث السياسة."""
    policy = RoutingPolicy()
    idx, name = policy.select_action(explore=explore)
    new_w = policy.apply_action_to_weights(base_weights, idx)
    reward = compute_reward_from_quality(quality, answer_text)
    upd = policy.update(idx, reward)
    append_transition(
        {"weights": base_weights},
        name,
        reward,
        {"new_weights": new_w, "update": upd},
    )
    return {
        "action": name,
        "action_idx": idx,
        "weights_before": base_weights,
        "weights_after": new_w,
        "reward": reward,
        "policy_update": upd,
    }


def rl_cycle_from_episodes(limit: int = 20) -> Dict[str, Any]:
    """إعادة تشغيل RL على episodes إن وُجدت."""
    policy = RoutingPolicy()
    reports = []
    try:
        from ai.experience_store import EpisodeStore
        store = EpisodeStore()
        episodes = []
        if hasattr(store, "list_recent"):
            episodes = store.list_recent(limit) or []
        elif hasattr(store, "recent"):
            episodes = store.recent(limit) or []
        else:
            # fallback: try get_all
            all_eps = getattr(store, "all", lambda: [])()
            episodes = list(all_eps)[-limit:]
    except Exception as e:
        return {"ok": False, "error": f"episodes: {e}", "policy_steps": policy.steps}

    for ep in episodes:
        try:
            if isinstance(ep, dict):
                q = ep.get("quality") or {}
                w = ep.get("decision_weights") or {k: 0.25 for k in ACTION_KEYS}
                ans = ep.get("answer_text") or ""
            else:
                q = getattr(ep, "quality", None) or {}
                w = getattr(ep, "decision_weights", None) or {k: 0.25 for k in ACTION_KEYS}
                ans = getattr(ep, "answer_text", "") or ""
            reports.append(rl_step_on_weights(w, quality=q, answer_text=ans, explore=True))
        except Exception as e:
            logger.info("ep skip: %s", e)
    return {
        "ok": True,
        "n": len(reports),
        "avg_reward": round(float(np.mean([r["reward"] for r in reports])), 4) if reports else None,
        "policy_steps": policy.steps,
        "probs": policy.probs(),
        "sample": reports[:3],
    }


def enable_rl_on_pipeline_result(result: Any) -> Dict[str, Any]:
    """يُستدعى بعد ReasoningPipeline.answer لتحديث السياسة."""
    weights = getattr(result, "decision_weights", None) or {}
    quality = getattr(result, "quality", None)
    answer = getattr(result, "answer_text", "") or ""
    base = {k: float(weights.get(k, 0.25)) for k in ACTION_KEYS}
    return rl_step_on_weights(base, quality=quality, answer_text=answer, explore=True)


def status() -> Dict[str, Any]:
    policy = RoutingPolicy()
    n_buf = 0
    if BUFFER_PATH.is_file():
        n_buf = sum(1 for _ in BUFFER_PATH.open(encoding="utf-8"))
    return {
        "enabled": True,
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "steps": policy.steps,
        "baseline": policy.baseline,
        "probs": dict(zip(ACTION_NAMES, [round(p, 4) for p in policy.probs()])),
        "buffer_size": n_buf,
        "at": _now(),
    }


def handle_rl_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if re.search(r"(حاله\s*rl|حالة\s*التعلم\s*المعزز|rl\s*status)", text, re.I):
        return "## 🎯 حالة التعلم المعزز\n```json\n" + json.dumps(status(), ensure_ascii=False, indent=2) + "\n```"
    if re.search(r"(دورة\s*rl|دورة\s*تعلم\s*معزز|rl\s*cycle|replay\s*rl)", text, re.I):
        r = rl_cycle_from_episodes(20)
        return "## 🔁 دورة RL من الخبرات\n```json\n" + json.dumps(r, ensure_ascii=False, indent=2, default=str)[:3500] + "\n```"
    if re.search(r"(تفعيل\s*تعلم\s*معزز|تفعيل\s*rl|enable\s*rl|reinforcement)", text, re.I):
        # خطوة تجريبية + حفظ سياسة
        demo_w = {k: 0.25 for k in ACTION_KEYS}
        step = rl_step_on_weights(demo_w, quality={"overall": 0.6}, answer_text="تفعيل تجريبي للتعلم المعزز في NSM", explore=True)
        st = status()
        return (
            "## ✅ تم تفعيل التعلم المعزز\n"
            "- سياسة توجيه softmax على 5 إجراءات\n"
            "- المكافأة من جودة الإجابة / الحلقات\n"
            "- بعد كل `ReasoningPipeline.answer` يمكن استدعاء `enable_rl_on_pipeline_result`\n\n"
            "### خطوة تجريبية\n```json\n"
            + json.dumps(step, ensure_ascii=False, indent=2)
            + "\n```\n### الحالة\n```json\n"
            + json.dumps(st, ensure_ascii=False, indent=2)
            + "\n```"
        )
    return None
