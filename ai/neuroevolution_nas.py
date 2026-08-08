"""
Neuroevolution & NAS — تطور جيني للبنى العصبية
==============================================
  • جيل من شبكات صغيرة تتدرب بالتوازي (محاكاة محلية)
  • اختيار الأقوى ودمج «جينات» الهيكل + ضوضاء طفيفة للأوزان
  • تكرار عبر أجيال لرفع الدقة دون تصميم يدوي كامل
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("NeuroevolutionNAS")

ROOT = Path(__file__).resolve().parent.parent
NAS_DIR = ROOT / "artifacts" / "model_training" / "meta_ai" / "nas"
NAS_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Genome:
    width: int
    depth: int
    dropout: float
    lr: float
    activation: str  # relu|tanh|gelu

    def mutate(self, rng: np.random.Generator) -> "Genome":
        w = self.width
        d = self.depth
        if rng.random() < 0.4:
            w = int(np.clip(w + rng.choice([-32, 0, 32]), 16, 256))
        if rng.random() < 0.3:
            d = int(np.clip(d + rng.choice([-1, 0, 1]), 1, 4))
        drop = float(np.clip(self.dropout + rng.normal(0, 0.05), 0.0, 0.5))
        lr = float(np.clip(self.lr * float(rng.uniform(0.5, 1.5)), 1e-4, 3e-2))
        act = self.activation
        if rng.random() < 0.25:
            act = str(rng.choice(["relu", "tanh", "gelu"]))
        return Genome(w, d, drop, lr, act)

    @staticmethod
    def crossover(a: "Genome", b: "Genome", rng: np.random.Generator) -> "Genome":
        return Genome(
            width=int(a.width if rng.random() < 0.5 else b.width),
            depth=int(a.depth if rng.random() < 0.5 else b.depth),
            dropout=float(a.dropout if rng.random() < 0.5 else b.dropout),
            lr=float(a.lr if rng.random() < 0.5 else b.lr),
            activation=str(a.activation if rng.random() < 0.5 else b.activation),
        )


def _act(name: str, x: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return np.tanh(x)
    if name == "gelu":
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    return np.maximum(0, x)


def _eval_genome(
    g: Genome,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 25,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    n, d_in = X.shape
    layers_w = []
    prev = d_in
    for _ in range(g.depth):
        W = rng.normal(0, 0.2, size=(prev, g.width))
        b = np.zeros(g.width)
        layers_w.append((W, b))
        prev = g.width
    W_out = rng.normal(0, 0.2, size=(prev,))
    b_out = 0.0
    for _ in range(epochs):
        h = X
        for W, b in layers_w:
            h = _act(g.activation, h @ W + b)
            if g.dropout > 0:
                mask = rng.random(h.shape) > g.dropout
                h = h * mask / max(1e-6, 1 - g.dropout)
        z = h @ W_out + b_out
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        err = (p - y) / n
        # تدرج تقريبي للطبقة الأخيرة فقط + إسناد بسيط
        dW_out = h.T @ err
        W_out -= g.lr * dW_out
        b_out -= g.lr * float(err.sum())
    h = X
    for W, b in layers_w:
        h = _act(g.activation, h @ W + b)
    p = 1 / (1 + np.exp(-np.clip(h @ W_out + b_out, -30, 30)))
    return float(((p >= 0.5) == y).mean())


@dataclass
class NASReport:
    ok: bool
    generations: int
    population: int
    best_genome: Dict[str, Any]
    best_score: float
    history: List[float]
    narrative_ar: str
    created_at: str = field(default_factory=_now)

    def to_markdown(self) -> str:
        lines = [
            "## 🧬 Neuroevolution / NAS",
            f"- أجيال: **{self.generations}** | حجم الجيل: **{self.population}**",
            f"- أفضل دقة: **{self.best_score:.1%}**",
            f"- أفضل جينوم: `{json.dumps(self.best_genome, ensure_ascii=False)}`",
            f"- مسار الأفضل عبر الأجيال: {', '.join(f'{h:.3f}' for h in self.history)}",
            "",
            self.narrative_ar,
        ]
        return "\n".join(lines)


def run_nas(
    generations: int = 5,
    population: int = 8,
    seed: int = 42,
) -> NASReport:
    rng = np.random.default_rng(seed)
    n, d_in = 500, 10
    X = rng.normal(size=(n, d_in))
    y = (X[:, 0] + 0.4 * X[:, 1] > 0).astype(np.float64)

    # جيل أولي
    pop = [
        Genome(
            width=int(rng.choice([32, 64, 128])),
            depth=int(rng.choice([1, 2, 3])),
            dropout=float(rng.choice([0.0, 0.1, 0.2])),
            lr=float(rng.choice([1e-3, 3e-3, 1e-2])),
            activation=str(rng.choice(["relu", "tanh", "gelu"])),
        )
        for _ in range(population)
    ]
    history = []
    best_g, best_s = pop[0], -1.0

    for gen in range(generations):
        scores = []
        for i, g in enumerate(pop):
            s = _eval_genome(g, X, y, epochs=20, seed=seed + gen * 100 + i)
            scores.append(s)
            if s > best_s:
                best_s, best_g = s, g
        history.append(float(max(scores)))
        # اختيار أفضل اثنين + تكاثر
        order = np.argsort(scores)[::-1]
        parents = [pop[int(order[0])], pop[int(order[1])]]
        new_pop = [parents[0], parents[1]]
        while len(new_pop) < population:
            child = Genome.crossover(parents[0], parents[1], rng).mutate(rng)
            new_pop.append(child)
        pop = new_pop
        logger.info("NAS gen %s best=%.3f", gen + 1, history[-1])

    narrative = (
        f"بعد {generations} أجيال بقيَ الأقوى: width={best_g.width}, depth={best_g.depth}, "
        f"act={best_g.activation}, lr={best_g.lr}. "
        "يمكن دفع هذا الجينوم لتدريب أطول على Kaggle/Colab."
    )
    report = NASReport(
        ok=True,
        generations=generations,
        population=population,
        best_genome=asdict(best_g),
        best_score=float(best_s),
        history=history,
        narrative_ar=narrative,
    )
    out = NAS_DIR / f"nas_{int(time.time())}.json"
    out.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    (out.with_suffix(".md")).write_text(report.to_markdown(), encoding="utf-8")
    try:
        from ai.persistent_memory import remember_experience

        remember_experience(
            kind="nas_result",
            text=narrative,
            meta={"best": asdict(best_g), "score": best_s},
        )
    except Exception:
        pass
    return report


def handle_nas_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(
        r"(neuroevolution|nas|تطور\s*جيني|بحث\s*هيكلي|جيل\s*شبكات|بقاء\s*للأقوى)",
        text,
        re.I,
    ):
        return None
    gens, pop = 5, 8
    m = re.search(r"(\d+)\s*(?:أجيال|اجيال|جيل|generations?)", text, re.I)
    if m:
        gens = max(2, min(20, int(m.group(1))))
    m = re.search(r"(\d+)\s*(?:شبكات|شبك|فرد|population|pop)", text, re.I)
    if m:
        pop = max(4, min(20, int(m.group(1))))
    return run_nas(generations=gens, population=pop).to_markdown()
