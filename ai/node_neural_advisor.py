# -*- coding: utf-8 -*-
"""
ai/node_neural_advisor.py — الشبكة العصبية داخل قرارات العُقد
================================================================
قرارات العُقد وقت التشغيل (اختيار البديل عند الفشل، ترتيب المسارات، اقتراح
الخطوة التالية) كانت قواعد ثابتة بلا أي شبكة عصبية: `should_fallback` تُرجع
«أول جار» دائماً. هذه الوحدة تُدخل شبكة ai.neural_core.NeuralNetwork الحقيقية
في الحلقة، **وتتدرّب وحدها** من نتائج التنفيذ الفعلية دون أي استدعاء يدوي:

    خصائص العقدة (نجاح حديث، سلسلة فشل، زمن، خبرة، حالة، سمعة)
        → [8 → 16 → 8 → 1 sigmoid] → احتمال نجاح التنفيذ التالي

التعلّم: بعد كل تنفيذ فعلي (ExecutionEngine عبر learn_from_run، ونتائج السرب عبر
MeshBundle.record_swarm_result) تُحسب الخصائص كما كانت **قبل** النتيجة (بلا
تسريب للتسمية) وتُجرى خطوة backprop واحدة نحو 1.0/0.0.

الأمان وقت البدء البارد: قبل `min_observations` خطوة تُمزَج توقعات الشبكة مع
معدل نجاح مُلَيَّن (Laplace) بوزن يتصاعد خطياً، فلا تُتَّخذ قرارات من أوزان
عشوائية.

الحفظ: كل `save_every` ملاحظة تُكتب الأوزان والإحصاءات ذرّياً (tmp + replace)؛
ملف تالف لا يُسقِط الإقلاع بل يبدأ من الصفر مع تحذير.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ai.neural_core import NeuralNetwork

logger = logging.getLogger(__name__)

FEATURE_NAMES = (
    "recent_success",   # معدل نجاح آخر 20 تنفيذاً (مُلَيَّن)
    "fail_streak",      # سلسلة الفشل الحالية / 5
    "latency",          # EMA الزمن / 5000ms
    "experience",       # عدد التنفيذات / 50
    "is_active",
    "is_failed",
    "is_paused",
    "reputation",       # سمعة العقدة (0..1)، 0.5 إن لم تتوفر
)
N_FEATURES = len(FEATURE_NAMES)
_WINDOW = 20
_LATENCY_CAP_MS = 5000.0


class _NodeStats:
    __slots__ = ("window", "latency_ema", "fail_streak", "total")

    def __init__(self) -> None:
        self.window: Deque[int] = deque(maxlen=_WINDOW)
        self.latency_ema: Optional[float] = None
        self.fail_streak = 0
        self.total = 0

    def smoothed_success(self) -> float:
        return (sum(self.window) + 1.0) / (len(self.window) + 2.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"window": list(self.window), "latency_ema": self.latency_ema,
                "fail_streak": self.fail_streak, "total": self.total}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_NodeStats":
        s = cls()
        s.window.extend(int(x) for x in (d.get("window") or [])[-_WINDOW:])
        ema = d.get("latency_ema")
        s.latency_ema = float(ema) if isinstance(ema, (int, float)) else None
        s.fail_streak = max(0, int(d.get("fail_streak", 0)))
        s.total = max(0, int(d.get("total", 0)))
        return s


class NodeNeuralAdvisor:
    """مستشار عصبي لعُقد الشبكة: يتنبأ باحتمال نجاح التنفيذ التالي لكل عقدة."""

    def __init__(
        self,
        storage_path: Optional[os.PathLike] = None,
        *,
        min_observations: int = 30,
        learning_rate: float = 0.03,
        seed: int = 7,
        save_every: int = 25,
    ):
        self._path = Path(storage_path) if storage_path else None
        self._min_obs = max(1, int(min_observations))
        self._save_every = max(0, int(save_every))
        self._lock = threading.RLock()
        self._stats: Dict[str, _NodeStats] = {}
        self._observed = 0
        self._last_loss: Optional[float] = None
        self._net = NeuralNetwork(
            [N_FEATURES, 16, 8, 1], ["relu", "relu", "sigmoid"],
            learning_rate=learning_rate, loss="mse", optimizer="adam",
            name="node_neural_advisor", seed=seed,
        )
        if self._path is not None:
            self._load()

    # ── الخصائص ──────────────────────────────────────────────────────────
    @staticmethod
    def _features(stats: Optional[_NodeStats], state: Optional[str],
                  reputation: Optional[float]) -> np.ndarray:
        s = stats or _NodeStats()
        lat = 0.0 if s.latency_ema is None else min(1.0, s.latency_ema / _LATENCY_CAP_MS)
        rep = 0.5 if reputation is None else float(min(1.0, max(0.0, reputation)))
        return np.array([
            s.smoothed_success(),
            min(1.0, s.fail_streak / 5.0),
            lat,
            min(1.0, s.total / 50.0),
            1.0 if state == "active" else 0.0,
            1.0 if state == "failed" else 0.0,
            1.0 if state == "paused" else 0.0,
            rep,
        ], dtype=np.float64)

    # ── التنبؤ ───────────────────────────────────────────────────────────
    def is_trained(self) -> bool:
        with self._lock:
            return self._observed >= self._min_obs

    def predict(self, node_id: str, state: Optional[str] = None,
                reputation: Optional[float] = None) -> float:
        """احتمال نجاح التنفيذ التالي (0..1). قبل اكتمال التدريب يُمزَج مع المعدل المُلَيَّن."""
        with self._lock:
            stats = self._stats.get(node_id)
            x = self._features(stats, state, reputation)
            prior = (stats or _NodeStats()).smoothed_success()
            nn_p = float(self._net.forward(x)[0])
            w = min(1.0, self._observed / float(self._min_obs))
        p = w * nn_p + (1.0 - w) * prior
        return float(min(1.0, max(0.0, p)))

    def rank(self, node_ids: Iterable[str],
             state_of: Optional[Any] = None,
             reputation_of: Optional[Any] = None) -> List[Tuple[str, float]]:
        """يرتّب العُقد تنازلياً بالاحتمال (الترتيب الأصلي يكسر التعادل)."""
        scored: List[Tuple[str, float]] = []
        for nid in node_ids:
            st = state_of(nid) if callable(state_of) else None
            rep = reputation_of(nid) if callable(reputation_of) else None
            scored.append((nid, self.predict(nid, st, rep)))
        return sorted(scored, key=lambda t: -t[1])

    # ── التعلّم الآلي ────────────────────────────────────────────────────
    def observe(self, node_id: str, success: bool, latency_ms: Optional[float] = None,
                state: Optional[str] = None, reputation: Optional[float] = None) -> float:
        """يتدرّب على نتيجة تنفيذ فعلية. الخصائص تُحسب قبل تحديث الإحصاءات."""
        with self._lock:
            stats = self._stats.setdefault(node_id, _NodeStats())
            x = self._features(stats, state, reputation)
            loss = float(self._net.train_step(x, [1.0 if success else 0.0]))
            self._last_loss = loss

            stats.window.append(1 if success else 0)
            stats.total += 1
            stats.fail_streak = 0 if success else stats.fail_streak + 1
            if isinstance(latency_ms, (int, float)) and latency_ms >= 0:
                stats.latency_ema = (float(latency_ms) if stats.latency_ema is None
                                     else 0.7 * stats.latency_ema + 0.3 * float(latency_ms))
            self._observed += 1
            due = bool(self._save_every and self._path and self._observed % self._save_every == 0)
        if due:
            self.save()
        return loss

    # ── معلومات ──────────────────────────────────────────────────────────
    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "observations": self._observed,
                "trained": self._observed >= self._min_obs,
                "min_observations": self._min_obs,
                "nodes_tracked": len(self._stats),
                "last_loss": self._last_loss,
                "architecture": self._net.architecture_str()
                if hasattr(self._net, "architecture_str") else None,
                "params": self._net.param_count(),
            }

    # ── الحفظ/الاسترجاع ──────────────────────────────────────────────────
    def _net_path(self) -> Path:
        return self._path.with_suffix(".net.json")  # type: ignore[union-attr]

    def save(self) -> bool:
        if self._path is None:
            return False
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                net_tmp = self._net_path().with_suffix(".tmp")
                self._net.save(str(net_tmp))
                os.replace(net_tmp, self._net_path())
                payload = {
                    "version": 1, "observed": self._observed, "last_loss": self._last_loss,
                    "features": list(FEATURE_NAMES),
                    "stats": {k: v.to_dict() for k, v in self._stats.items()},
                }
                tmp = self._path.with_suffix(".tmp")
                tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, self._path)
            return True
        except Exception as e:  # الحفظ لا يُسقِط التنفيذ
            logger.warning("NodeNeuralAdvisor: save failed: %s", e)
            return False

    def _load(self) -> None:
        try:
            if self._path.exists() and self._net_path().exists():  # type: ignore[union-attr]
                data = json.loads(self._path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
                if list(data.get("features") or []) != list(FEATURE_NAMES):
                    raise ValueError("feature schema changed")
                net = NeuralNetwork.load(str(self._net_path()))
                if net.input_dim != N_FEATURES or net.output_dim != 1:
                    raise ValueError("network shape mismatch")
                self._net = net
                self._observed = max(0, int(data.get("observed", 0)))
                self._last_loss = data.get("last_loss")
                self._stats = {k: _NodeStats.from_dict(v)
                               for k, v in (data.get("stats") or {}).items()}
        except Exception as e:
            logger.warning("NodeNeuralAdvisor: ignoring unreadable saved state (%s); starting fresh", e)
            self._stats, self._observed, self._last_loss = {}, 0, None
