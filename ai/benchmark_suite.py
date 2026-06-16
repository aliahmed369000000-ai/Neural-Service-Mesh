"""
Benchmark Suite — مجموعة اختبار ثابتة لقياس أداء NeuralCore
================================================================
تحمل 20 مفهوماً ثابتاً من cognitive_graph.json مع target محسوب
مباشرة من strength كل مفهوم في CKG. تُستخدم قبل/بعد كل دورة تدريب
لاكتشاف التراجع (rollback) في run_training_cycle().
"""
from __future__ import annotations

import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CKG_PATH = "knowledge/cognitive_graph.json"

# المفاهيم الـ20 الثابتة للاختبار (اسم المفهوم كما هو في cognitive_graph.json)
BENCHMARK_CONCEPTS = [
    "الله", "الرب", "علم", "إيمان", "أرض",
    "كفر", "عذاب", "العليم", "بيان", "حقوق",
    "سماء", "قرآن", "نار", "أنبياء", "خلق",
    "هداية", "صدق", "يوم القيامة", "ضلال", "ورع",
]

# أقصى تردد معروف وقت كتابة هذا الكود (للمفهوم "الله") — يُستخدم لتطبيع log-frequency
_MAX_FREQ = 1943


def _build_vector_from_concept(name: str, data: dict) -> List[float]:
    """
    يبني متجه 7 عناصر من بيانات المفهوم في CKG.
    يطابق نفس الـ 7 features التي يستخدمها RichDataCollector:
    [strength, log_freq_norm, relation_count_norm, 0.5, 0.5, strength*0.8, 0.6]
    """
    strength = float(data.get("strength", 0.5))
    frequency = float(data.get("frequency", 1))
    # تطبيع log-frequency إلى [0,1] (max_freq=1943 للمفهوم "الله")
    log_freq = min(1.0, math.log1p(frequency) / math.log1p(_MAX_FREQ))
    return [
        round(strength, 4),
        round(log_freq, 4),
        round(strength * 0.9, 4),
        0.5,
        0.5,
        round(strength * 0.8, 4),
        0.6,
    ]


class BenchmarkSuite:
    """
    مجموعة اختبار ثابتة لقياس أداء NeuralCore.

    تحمل 20 مفهوماً من CKG مع target محسوب من strength كل مفهوم.
    تُستخدم قبل/بعد كل دورة تدريب لاكتشاف التراجع (rollback).
    """

    def __init__(self, ckg_path: str = CKG_PATH):
        self.ckg_path = ckg_path
        self._samples: List[Tuple[List[float], float, str]] = []
        # كل عنصر: (context_vector, target, concept_name)
        self._loaded = False
        self._load()

    def _load(self) -> None:
        """يحمّل CKG ويبني الـ samples."""
        path = Path(self.ckg_path)
        concepts: Dict[str, dict] = {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            concepts = data.get("concepts", {})
        except Exception as e:
            logger.warning(f"BenchmarkSuite: failed to load CKG from {path}: {e}")

        self._samples = []
        for name in BENCHMARK_CONCEPTS:
            concept_data = concepts.get(name)
            if concept_data is None:
                strength = 0.3
                concept_data = {"strength": strength}
            else:
                strength = float(concept_data.get("strength", 0.3))

            vector = _build_vector_from_concept(name, concept_data)
            target = round(strength * 0.95, 4)
            self._samples.append((vector, target, name))

        self._loaded = True
        logger.info(f"BenchmarkSuite: loaded {len(self._samples)} samples from {path}")

    def evaluate(self, core) -> dict:
        """
        يُشغّل core.forward() على كل sample ويحسب متوسط MSE.

        Parameters
        ----------
        core : NeuralCore

        Returns
        -------
        dict:
            {
                "score": float,       # متوسط MSE (أقل = أفضل)
                "n_samples": int,
                "per_concept": {name: mse, ...}
            }
        """
        import numpy as np

        per_concept: Dict[str, float] = {}
        mses: List[float] = []

        for vector, target, name in self._samples:
            x = np.array(vector, dtype=np.float64)
            out = core.forward(x)
            # target واحد لكل مفهوم: نقارنه بالعنصر الأول من الخرج (W_SEMANTIC)
            # كحدّ أدنى من الإشارة، ونستخدم MSE على ناتج الشبكة كاملاً مقابل
            # متجه target موزَّع بالتساوي لتقييم اتساق الخرج العام.
            target_vec = np.full(out.shape, target, dtype=np.float64)
            mse = float(np.mean((out - target_vec) ** 2))
            per_concept[name] = round(mse, 8)
            mses.append(mse)

        score = float(np.mean(mses)) if mses else 0.0

        return {
            "score": round(score, 8),
            "n_samples": len(self._samples),
            "per_concept": per_concept,
        }

    @property
    def n_samples(self) -> int:
        return len(self._samples)
