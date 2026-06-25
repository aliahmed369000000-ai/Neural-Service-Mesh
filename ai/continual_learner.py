"""
Continual Learner — EWC (Elastic Weight Consolidation)
=======================================================
يمنع Catastrophic Forgetting عند تدريب مهام جديدة.

المبدأ الرياضي (Kirkpatrick et al., 2017):
  L_total = L_new_task + λ * Σ F_i * (θ_i - θ*_i)²

  حيث:
    L_new_task  = خسارة المهمة الجديدة
    F_i         = Fisher Information لكل وزن (يقيس أهميته)
    θ*_i        = قيمة الوزن بعد المهمة السابقة
    λ           = قوة الحماية (lambda)

التكامل:
  - يستخدم EpisodeStore من ai/experience_store.py كمصدر بيانات الحلقات السابقة
  - يعمل مع مصفوفة E (784×128) من nsm_trainer.py
  - يمكن تشغيله قبل كل دورة تدريب جديدة

الاستخدام:
    from ai.continual_learner import EWCLearner
    ewc = EWCLearner(lambda_reg=400.0)
    ewc.compute_fisher(E_matrix, old_episodes)  # بعد المهمة الأولى
    penalty = ewc.penalty(E_matrix)             # أثناء تدريب المهمة الجديدة
    total_loss = new_task_loss + penalty
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_NOW = lambda: datetime.now(timezone.utc).isoformat()

# ── مسار حفظ Fisher Information ─────────────────────────────────────────────
_DEFAULT_FISHER_PATH = Path("memory/ewc_fisher.npz")
_DEFAULT_ANCHOR_PATH = Path("memory/ewc_anchor.npz")


# ════════════════════════════════════════════════════════════════════════════
# FisherSnapshot — لقطة من Fisher + Anchor weights بعد كل مهمة
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class FisherSnapshot:
    """لقطة واحدة: Fisher matrix + أوزان المرجعية + توصيف المهمة."""
    task_id: str
    task_label: str
    fisher: np.ndarray          # (784, 128) — نفس شكل E
    anchor_weights: np.ndarray  # (784, 128) — قيمة E عند إنهاء المهمة
    n_samples: int
    computed_at: str = field(default_factory=_NOW)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_label": self.task_label,
            "n_samples": self.n_samples,
            "computed_at": self.computed_at,
        }


# ════════════════════════════════════════════════════════════════════════════
# EWCLearner — المحرك الرئيسي
# ════════════════════════════════════════════════════════════════════════════

class EWCLearner:
    """
    يمنع Catastrophic Forgetting باستخدام Elastic Weight Consolidation.

    الاستخدام النموذجي:
        ewc = EWCLearner(lambda_reg=400.0)

        # بعد الانتهاء من المهمة A:
        ewc.compute_fisher(E, episodes_A, task_label="arabic_islamic")
        ewc.save()

        # أثناء تدريب المهمة B:
        for batch in task_B_batches:
            loss_new = compute_triplet_loss(E, batch)
            penalty  = ewc.penalty(E)          # حماية المعرفة القديمة
            total    = loss_new + penalty
            dE       = compute_grad(E, total)
            E       -= lr * dE
    """

    def __init__(
        self,
        lambda_reg: float = 400.0,
        fisher_path: Path | str = _DEFAULT_FISHER_PATH,
        anchor_path: Path | str = _DEFAULT_ANCHOR_PATH,
        max_snapshots: int = 5,
    ):
        self.lambda_reg   = lambda_reg
        self.fisher_path  = Path(fisher_path)
        self.anchor_path  = Path(anchor_path)
        self.max_snapshots = max_snapshots

        self._snapshots: List[FisherSnapshot] = []
        self._load_if_exists()

    # ── واجهة عامة ──────────────────────────────────────────────────────────

    def compute_fisher(
        self,
        E: np.ndarray,
        episodes: List[Any],
        task_label: str = "task",
        task_id: Optional[str] = None,
        n_samples: int = 200,
    ) -> FisherSnapshot:
        """
        يحسب Fisher Information matrix من حلقات التجارب السابقة.

        المعادلة:
            F_i = E[ (∂ log p(y|x) / ∂θ_i)² ]

        نحسبها بالتقريب:
            F_i ≈ (1/N) Σ (∂ L / ∂ E_i)²

        Args:
            E:          مصفوفة Embedding الحالية (784, 128)
            episodes:   قائمة حلقات من EpisodeStore
            task_label: وصف المهمة للتوثيق
            task_id:    معرّف فريد (اختياري)
            n_samples:  عدد الحلقات المستخدمة في الحساب

        Returns:
            FisherSnapshot يحتوي على Fisher + Anchor
        """
        import uuid
        tid = task_id or f"task_{uuid.uuid4().hex[:8]}"

        logger.info(f"[EWC] Computing Fisher for task='{task_label}' "
                    f"using {min(n_samples, len(episodes))} episodes …")

        fisher_accum = np.zeros_like(E, dtype=np.float64)
        used = 0

        sample_episodes = episodes[:n_samples] if len(episodes) > n_samples else episodes

        for ep in sample_episodes:
            ctx_vec = self._episode_to_vec(ep, E.shape[0])
            if ctx_vec is None:
                continue

            # Forward: z = E.T @ x
            z = E.T @ ctx_vec          # (128,)
            norm = np.linalg.norm(z) + 1e-8
            z_hat = z / norm           # L2 normalised

            # الخسارة التقريبية: 1 - max_cosine (نحاكي log-softmax)
            # التدرج: ∂L/∂E ≈ -x ⊗ z_hat  (نظرًا لـ L2 norm)
            grad_z = -z_hat                      # (128,)
            # chain rule عبر L2 norm
            grad_raw = (grad_z - z_hat * float(z_hat @ grad_z)) / norm
            dE = np.outer(ctx_vec, grad_raw)    # (784, 128)

            fisher_accum += dE ** 2
            used += 1

        if used > 0:
            fisher_accum /= used

        anchor = E.copy()

        snap = FisherSnapshot(
            task_id=tid,
            task_label=task_label,
            fisher=fisher_accum,
            anchor_weights=anchor,
            n_samples=used,
        )

        # نحتفظ بأحدث max_snapshots فقط
        self._snapshots.append(snap)
        if len(self._snapshots) > self.max_snapshots:
            self._snapshots = self._snapshots[-self.max_snapshots:]

        logger.info(f"[EWC] Fisher computed — {used} samples, "
                    f"max_F={fisher_accum.max():.4f}, mean_F={fisher_accum.mean():.6f}")
        return snap

    def penalty(self, E: np.ndarray) -> float:
        """
        يحسب عقوبة EWC الكلية (مجموع عبر كل المهام السابقة).

            penalty = λ * Σ_tasks Σ_i  F_i * (E_i - E*_i)²

        Args:
            E: أوزان Embedding الحالية (784, 128)

        Returns:
            float — قيمة العقوبة (تُضاف إلى خسارة المهمة الجديدة)
        """
        if not self._snapshots:
            return 0.0

        total = 0.0
        for snap in self._snapshots:
            diff = E - snap.anchor_weights        # (784, 128)
            total += float(np.sum(snap.fisher * (diff ** 2)))

        return self.lambda_reg * total

    def penalty_gradient(self, E: np.ndarray) -> np.ndarray:
        """
        تدرّج العقوبة بالنسبة لـ E — للاستخدام المباشر في خطوة التدريب.

            ∂penalty/∂E_i = 2λ * Σ_tasks F_i * (E_i - E*_i)

        Args:
            E: أوزان Embedding الحالية (784, 128)

        Returns:
            np.ndarray بنفس شكل E
        """
        if not self._snapshots:
            return np.zeros_like(E)

        grad = np.zeros_like(E, dtype=np.float64)
        for snap in self._snapshots:
            diff = E - snap.anchor_weights
            grad += snap.fisher * diff

        return 2.0 * self.lambda_reg * grad

    # ── تكامل مع EpisodeStore ────────────────────────────────────────────────

    @classmethod
    def from_episode_store(
        cls,
        E: np.ndarray,
        db_path: str | Path = "memory/experience.db",
        lambda_reg: float = 400.0,
        task_label: str = "nsm_base",
        n_samples: int = 200,
    ) -> "EWCLearner":
        """
        يبني EWCLearner مباشرةً من EpisodeStore الموجود.

        مثال:
            from ai.continual_learner import EWCLearner
            import numpy as np
            E = np.load("nsm_embedding.npz")["E"]
            ewc = EWCLearner.from_episode_store(E, task_label="arabic_islamic")
        """
        try:
            from ai.experience_store import EpisodeStore
            store = EpisodeStore(db_path=db_path)
            episodes = store.get_diverse_sample(limit=n_samples)
            logger.info(f"[EWC] Loaded {len(episodes)} episodes from store")
        except Exception as exc:
            logger.warning(f"[EWC] Could not load EpisodeStore: {exc} — using empty episodes")
            episodes = []

        ewc = cls(lambda_reg=lambda_reg)
        if episodes:
            ewc.compute_fisher(E, episodes, task_label=task_label, n_samples=n_samples)
            ewc.save()
        return ewc

    # ── حفظ / تحميل ─────────────────────────────────────────────────────────

    def save(
        self,
        fisher_path: Optional[Path | str] = None,
        anchor_path: Optional[Path | str] = None,
    ) -> bool:
        """يحفظ كل الـ Fisher snapshots على القرص."""
        fp = Path(fisher_path or self.fisher_path)
        ap = Path(anchor_path or self.anchor_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        ap.parent.mkdir(parents=True, exist_ok=True)

        if not self._snapshots:
            logger.warning("[EWC] No snapshots to save")
            return False

        try:
            fishers  = np.stack([s.fisher         for s in self._snapshots])  # (T, 784, 128)
            anchors  = np.stack([s.anchor_weights  for s in self._snapshots])  # (T, 784, 128)
            meta     = json.dumps([s.to_dict() for s in self._snapshots]).encode()

            np.savez_compressed(str(fp), fishers=fishers, meta=meta)
            np.savez_compressed(str(ap), anchors=anchors)

            logger.info(f"[EWC] Saved {len(self._snapshots)} snapshot(s) → {fp}, {ap}")
            return True
        except Exception as exc:
            logger.error(f"[EWC] Save failed: {exc}")
            return False

    def _load_if_exists(self) -> None:
        """يحمّل Fisher + Anchor من القرص إن وُجدا."""
        fp = self.fisher_path
        ap = self.anchor_path
        if not (fp.exists() and ap.exists()):
            return
        try:
            f_data = np.load(str(fp), allow_pickle=True)
            a_data = np.load(str(ap), allow_pickle=True)
            fishers = f_data["fishers"]                    # (T, 784, 128)
            anchors = a_data["anchors"]                    # (T, 784, 128)
            meta    = json.loads(f_data["meta"].tobytes())

            self._snapshots = []
            for i, m in enumerate(meta):
                snap = FisherSnapshot(
                    task_id=m["task_id"],
                    task_label=m["task_label"],
                    fisher=fishers[i],
                    anchor_weights=anchors[i],
                    n_samples=m["n_samples"],
                    computed_at=m["computed_at"],
                )
                self._snapshots.append(snap)

            logger.info(f"[EWC] Loaded {len(self._snapshots)} snapshot(s) from disk")
        except Exception as exc:
            logger.warning(f"[EWC] Could not load from disk: {exc}")
            self._snapshots = []

    # ── أدوات مساعدة ────────────────────────────────────────────────────────

    def _episode_to_vec(self, episode: Any, dim: int) -> Optional[np.ndarray]:
        """يستخرج context_vector من حلقة تجربة."""
        try:
            # Episode من ai/experience_store.py
            if hasattr(episode, "context_vector") and episode.context_vector:
                vec = np.array(episode.context_vector, dtype=np.float64)
                if len(vec) == dim:
                    return vec
                elif len(vec) > 0:
                    # تغيير الحجم إن اختلف
                    resized = np.zeros(dim, dtype=np.float64)
                    n = min(len(vec), dim)
                    resized[:n] = vec[:n]
                    return resized

            # fallback: تحويل السؤال إلى متجه
            if hasattr(episode, "question") and episode.question:
                return _text_to_vec_fallback(episode.question, dim)
        except Exception:
            pass
        return None

    def get_stats(self) -> Dict[str, Any]:
        """ملخص حالة EWC الحالية."""
        return {
            "lambda_reg": self.lambda_reg,
            "n_snapshots": len(self._snapshots),
            "max_snapshots": self.max_snapshots,
            "tasks": [s.to_dict() for s in self._snapshots],
            "fisher_path": str(self.fisher_path),
            "anchor_path": str(self.anchor_path),
        }

    def current_penalty_magnitude(self, E: np.ndarray) -> Dict[str, float]:
        """تفاصيل العقوبة لكل مهمة — مفيد للمراقبة."""
        result = {}
        for snap in self._snapshots:
            diff = E - snap.anchor_weights
            p = float(self.lambda_reg * np.sum(snap.fisher * (diff ** 2)))
            result[snap.task_label] = round(p, 6)
        result["total"] = round(sum(result.values()), 6)
        return result


# ════════════════════════════════════════════════════════════════════════════
# أداة مساعدة: text_to_vec بدون استيراد خارجي
# ════════════════════════════════════════════════════════════════════════════

def _fnv1a(s: str) -> int:
    h = 0x811c9dc5
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def _text_to_vec_fallback(text: str, dim: int = 784) -> np.ndarray:
    """نسخة محلية من text_to_vec لتجنب الاستيراد الدائري."""
    vec = np.zeros(dim, dtype=np.float64)
    t = text.strip().lower()
    for n in (1, 2, 3):
        for i in range(len(t) - n + 1):
            vec[_fnv1a(t[i:i+n]) % dim] += 1.0
    total = vec.sum()
    if total > 0:
        vec = np.log1p(vec * 10.0 / total) / math.log1p(10.0)
    return vec


# ════════════════════════════════════════════════════════════════════════════
# EWCTrainingLoop — حلقة تدريب متكاملة تجمع Triplet Loss + EWC Penalty
# ════════════════════════════════════════════════════════════════════════════

class EWCTrainingLoop:
    """
    حلقة تدريب كاملة تجمع:
      • Triplet Loss (من nsm_trainer)
      • EWC Penalty (من EWCLearner)

    مثال الاستخدام:
        loop = EWCTrainingLoop(E, ewc, lr=0.01)
        result = loop.train(new_train_data, epochs=30)
        E_updated = result["E"]
    """

    def __init__(
        self,
        E: np.ndarray,
        ewc: EWCLearner,
        lr: float = 0.01,
        margin: float = 0.4,
    ):
        self.E      = E.copy().astype(np.float64)
        self.ewc    = ewc
        self.lr     = lr
        self.margin = margin

    def train(
        self,
        train_data: List[Tuple[str, int]],
        epochs: int = 30,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        يُدرّب E مع الحماية من Catastrophic Forgetting.

        Args:
            train_data: قائمة (نص، رقم_الموضوع) — نفس تنسيق TRAIN_DATA في nsm_trainer
            epochs:     عدد دورات التدريب
            verbose:    طباعة التقدم

        Returns:
            dict يحتوي على E المحدَّثة وتاريخ الخسارة
        """
        import random

        # فصل البيانات حسب الموضوع
        by_topic: Dict[int, List[np.ndarray]] = {}
        for text, label in train_data:
            vec = _text_to_vec_fallback(text, self.E.shape[0])
            by_topic.setdefault(label, []).append(vec)

        topics = list(by_topic.keys())
        if len(topics) < 2:
            logger.warning("[EWC] Need at least 2 topics for Triplet Loss")
            return {"E": self.E, "history": [], "epochs": 0}

        history = []

        for epoch in range(epochs):
            triplet_loss_sum = 0.0
            ewc_penalty_sum  = 0.0
            steps = 0

            for topic in topics:
                ancs = by_topic[topic]
                neg_topics = [t for t in topics if t != topic]

                for anc_vec in ancs:
                    # Positive: نفس الموضوع
                    pos_vecs = [v for v in ancs if not np.array_equal(v, anc_vec)]
                    if not pos_vecs:
                        continue
                    pos_vec = random.choice(pos_vecs)

                    # Hard Negative: الأقرب من موضوع مختلف
                    neg_topic = random.choice(neg_topics)
                    neg_vecs  = by_topic[neg_topic]
                    neg_vec   = self._hard_negative(anc_vec, neg_vecs)

                    # حساب Triplet Gradient
                    t_loss, dE_triplet = self._triplet_grad(anc_vec, pos_vec, neg_vec)

                    # حساب EWC Gradient
                    dE_ewc   = self.ewc.penalty_gradient(self.E)
                    ewc_pen  = self.ewc.penalty(self.E)

                    # تحديث الأوزان
                    total_dE = dE_triplet + dE_ewc
                    self.E  -= self.lr * total_dE

                    triplet_loss_sum += t_loss
                    ewc_penalty_sum  += ewc_pen
                    steps += 1

            avg_t   = triplet_loss_sum / max(steps, 1)
            avg_ewc = ewc_penalty_sum  / max(steps, 1)
            record  = {"epoch": epoch + 1, "triplet_loss": round(avg_t, 6),
                       "ewc_penalty": round(avg_ewc, 6),
                       "total_loss": round(avg_t + avg_ewc, 6)}
            history.append(record)

            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                logger.info(f"[EWC] Epoch {epoch+1:3d}/{epochs} — "
                            f"triplet={avg_t:.4f}  ewc={avg_ewc:.4f}  "
                            f"total={avg_t + avg_ewc:.4f}")

        return {"E": self.E, "history": history, "epochs": epochs, "steps": steps}

    def _hard_negative(self, anchor: np.ndarray, candidates: List[np.ndarray]) -> np.ndarray:
        """يختار الـ Negative الأقرب إلى الـ Anchor (الأصعب)."""
        def _l2(x):
            return x / (np.linalg.norm(x) + 1e-8)

        z_a = _l2(self.E.T @ anchor)
        best, best_sim = candidates[0], -1.0
        for c in candidates:
            sim = float(z_a @ _l2(self.E.T @ c))
            if sim > best_sim:
                best, best_sim = c, sim
        return best

    def _triplet_grad(
        self,
        anc: np.ndarray,
        pos: np.ndarray,
        neg: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Triplet Loss مع التدرّج."""
        def _embed(x):
            raw = self.E.T @ x
            n   = np.linalg.norm(raw) + 1e-8
            return raw / n, n

        za, na = _embed(anc)
        zp, _  = _embed(pos)
        zn, _  = _embed(neg)

        d_pos = 1.0 - float(za @ zp)
        d_neg = 1.0 - float(za @ zn)
        loss  = max(0.0, self.margin + d_pos - d_neg)

        if loss < 1e-7:
            return loss, np.zeros_like(self.E)

        dza    = -zp + zn
        draw   = (dza - za * float(za @ dza)) / na
        dE     = np.outer(anc, draw)
        return loss, dE
