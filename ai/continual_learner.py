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
    ewc = EWCLearner(lambda_reg=20.0)
    ewc.compute_fisher(E_matrix, old_episodes)  # بعد المهمة الأولى
    penalty = ewc.penalty(E_matrix)             # أثناء تدريب المهمة الجديدة
    total_loss = new_task_loss + penalty

ملاحظة معايرة (محدّثة):
  كانت compute_fisher() تحتوي خللاً رياضياً جعل قيمة Fisher تساوي صفراً
  دائماً (grad_z=-z_hat ينعدم هندسياً عند إسقاطه على المستوى العمودي)،
  أي أن EWC لم يكن يحمي شيئاً فعلياً منذ كتابته. تم تصحيح الصيغة باستخدام
  حيلة "Empirical Fisher" المعروفة، وبالتبعية أُعيدت معايرة lambda_reg
  الافتراضية من 400.0 إلى 20.0 (القيمة القديمة كانت غير مؤثرة وقت الخلل،
  وتُسبب الآن انفجاراً عددياً (inf/nan) مع القيم الحقيقية غير الصفرية).
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
        ewc = EWCLearner(lambda_reg=20.0)

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
        lambda_reg: float = 20.0,
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
            # ★ إصلاح: الصيغة السابقة كانت grad_z = -z_hat، وهذا يجعل
            # ناتج الإسقاط على المستوى العمودي (tangent space) صفراً
            # رياضياً *دائماً* — لأن z_hat متجه وحدة، فإن
            # z_hat·(-z_hat) = -1 ثابتة، فيتلاشى الإسقاط تماماً لأي مدخل
            # (تم التحقق رقمياً: fisher_accum كانت تساوي صفراً دائماً،
            # أي أن EWC لم يكن يحمي أي شيء فعلياً منذ كتابته).
            # الإصلاح: نستخدم حيلة "Empirical Fisher" المعروفة (استعمال
            # أقوى بُعد في z_hat نفسه كـ pseudo-label بدل افتراض هدف
            # مضاد تماماً لـ z_hat) — هذا يُعطي تدرجاً غير متلاشٍ ومعبّراً
            # فعلاً عن حساسية كل وزن تجاه هذا المدخل، دون الحاجة لأي
            # تصنيف حقيقي (يطابق نفس فكرة الكود الأصلي بدون الخلل الرياضي).
            k = int(np.argmax(z_hat))
            one_hot = np.zeros_like(z_hat)
            one_hot[k] = 1.0
            grad_z = z_hat - one_hot              # (128,) — لا يتلاشى عمومياً
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
        lambda_reg: float = 20.0,
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


# ══════════════════════════════════════════════════════════════════════════════
# ConversationLearner — التعلم من المحادثات الحقيقية
# يُضاف للملف الأصلي دون تعديل EWCLearner
# ══════════════════════════════════════════════════════════════════════════════

import json
import re
import sqlite3
import time as _time
from pathlib import Path as _Path

_CONV_DB = "memory/nsm_learning.db"
_MAX_CACHED_AGE = 86400 * 7


def _init_conv_db(path: str):
    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(p) as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS learned_qa (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                question      TEXT NOT NULL,
                answer        TEXT NOT NULL,
                domain        TEXT DEFAULT 'عام',
                quality       REAL DEFAULT 0.5,
                usage_count   INTEGER DEFAULT 0,
                positive_fb   INTEGER DEFAULT 0,
                negative_fb   INTEGER DEFAULT 0,
                created_at    REAL NOT NULL,
                last_used     REAL NOT NULL,
                q_hash        TEXT UNIQUE
            );
            CREATE TABLE IF NOT EXISTS response_cache (
                q_hash    TEXT PRIMARY KEY,
                query     TEXT NOT NULL,
                response  TEXT NOT NULL,
                quality   REAL DEFAULT 0.5,
                hits      INTEGER DEFAULT 0,
                created   REAL NOT NULL,
                last_hit  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_domain ON learned_qa(domain, quality DESC);
            CREATE INDEX IF NOT EXISTS idx_usage  ON learned_qa(usage_count DESC);
        """)


def _qhash(text: str) -> str:
    h = 0x811c9dc5
    for ch in text.lower()[:100]:
        h ^= ord(ch); h = (h * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def _estimate_quality(question: str, answer: str) -> float:
    if not answer or len(answer) < 20:
        return 0.1
    score = 0.5
    if len(answer) > 80:    score += 0.15
    if len(question) > 10:  score += 0.10
    if re.search(r'[\u0600-\u06FF]{50,}', answer): score += 0.10
    if re.search(r'(لا أعرف|لست متأكد|خطأ|error)', answer, re.I): score -= 0.20
    return round(min(0.95, max(0.1, score)), 3)


class ConversationLearner:
    """
    يتعلم من كل محادثة حقيقية ويحفظها في SQLite.

    الاستخدام:
        learner = ConversationLearner()

        # بعد كل رد
        learner.learn(question, answer, domain="صلاة")

        # تقييم
        learner.feedback(question, is_positive=True)

        # قبل الإرسال للـ LLM — ابحث في الذاكرة أولاً
        recalled = learner.recall(question)
    """

    def __init__(self, db_path: str = _CONV_DB):
        self._db = db_path
        _init_conv_db(db_path)
        self._session = {"learned": 0, "recalled": 0, "feedback": 0}

    def learn(self, question: str, answer: str, domain: str = "عام",
              source: str = "conversation") -> float:
        """يتعلم من تفاعل. يعيد درجة الجودة."""
        if not question or not answer:
            return 0.0
        quality = _estimate_quality(question, answer)
        if quality < 0.3:
            return quality
        qh = _qhash(question)
        now = _time.time()
        try:
            with sqlite3.connect(self._db) as c:
                c.execute("""
                    INSERT INTO learned_qa
                    (question,answer,domain,quality,usage_count,created_at,last_used,q_hash)
                    VALUES(?,?,?,?,0,?,?,?)
                    ON CONFLICT(q_hash) DO UPDATE SET
                        answer      = CASE WHEN excluded.quality > quality
                                      THEN excluded.answer ELSE answer END,
                        quality     = MAX(quality, excluded.quality),
                        usage_count = usage_count + 1,
                        last_used   = excluded.last_used
                """, (question[:300], answer[:800], domain, quality, now, now, qh))
                # كاش
                c.execute("""
                    INSERT INTO response_cache(q_hash,query,response,quality,hits,created,last_hit)
                    VALUES(?,?,?,?,0,?,?)
                    ON CONFLICT(q_hash) DO UPDATE SET
                        response = CASE WHEN excluded.quality > quality
                                   THEN excluded.response ELSE response END,
                        quality  = MAX(quality, excluded.quality),
                        hits     = hits + 1, last_hit = excluded.last_hit
                """, (qh, question[:200], answer[:600], quality, now, now))
        except Exception as e:
            logger.debug(f"ConversationLearner.learn: {e}")
        self._session["learned"] += 1
        return quality

    def recall(self, query: str, min_quality: float = 0.6) -> Optional[dict]:
        """يبحث في الكاش والإجابات المتعلَّمة."""
        qh = _qhash(query)
        now = _time.time()
        try:
            with sqlite3.connect(self._db) as c:
                # كاش
                row = c.execute(
                    "SELECT response,quality FROM response_cache "
                    "WHERE q_hash=? AND quality>=? AND (?-last_hit)<?",
                    (qh, min_quality, now, _MAX_CACHED_AGE)
                ).fetchone()
                if row:
                    c.execute("UPDATE response_cache SET hits=hits+1,last_hit=? WHERE q_hash=?",
                              (now, qh))
                    self._session["recalled"] += 1
                    return {"answer": row[0], "quality": row[1], "source": "cache"}

                # بحث بالكلمات المفتاحية
                words = re.findall(r'[\u0600-\u06FF]{2,}|[a-zA-Z]{3,}', query)[:4]
                for w in words:
                    row2 = c.execute(
                        "SELECT answer,quality FROM learned_qa "
                        "WHERE question LIKE ? AND quality>=? "
                        "ORDER BY quality DESC LIMIT 1",
                        (f"%{w}%", min_quality)
                    ).fetchone()
                    if row2:
                        self._session["recalled"] += 1
                        return {"answer": row2[0], "quality": row2[1], "source": "learned"}
        except Exception as e:
            logger.debug(f"ConversationLearner.recall: {e}")
        return None

    def feedback(self, question: str, is_positive: bool) -> bool:
        """تطبيق تقييم 👍/👎"""
        qh = _qhash(question)
        col   = "positive_fb" if is_positive else "negative_fb"
        delta = 0.05 if is_positive else -0.05
        try:
            with sqlite3.connect(self._db) as c:
                c.execute(
                    f"UPDATE learned_qa SET {col}={col}+1, "
                    f"quality=MAX(0.1,MIN(1.0,quality+?)) WHERE q_hash=?",
                    (delta, qh)
                )
            self._session["feedback"] += 1
            return True
        except Exception:
            return False

    def infer_implicit_feedback(self, prev_answer: str, followup: str) -> Optional[bool]:
        """يستنتج التقييم الضمني من سؤال المتابعة"""
        NEG = ["لماذا","هل أنت متأكد","غير صحيح","خطأ","لا أعتقد","لا أوافق"]
        POS = ["شكراً","جيد","ممتاز","صحيح","رائع","أحسنت","مفيد"]
        fl  = followup.lower()
        if any(s in fl for s in NEG): return False
        if any(s in fl for s in POS): return True
        return None

    def learn_batch(self, history: List[Tuple[str, str]], domain: str = "عام") -> int:
        """تعلم دفعي من قائمة (سؤال، جواب)"""
        return sum(1 for q,a in history if self.learn(q, a, domain) > 0.3)

    def stats(self) -> dict:
        try:
            with sqlite3.connect(self._db) as c:
                total = c.execute("SELECT COUNT(*) FROM learned_qa").fetchone()[0]
                avg_q = c.execute("SELECT AVG(quality) FROM learned_qa").fetchone()[0] or 0
                by_domain = c.execute(
                    "SELECT domain,COUNT(*),AVG(quality) FROM learned_qa "
                    "GROUP BY domain ORDER BY COUNT(*) DESC LIMIT 5"
                ).fetchall()
            return {
                "total_learned": total,
                "avg_quality":   round(avg_q, 3),
                "session":       self._session,
                "by_domain":     [{"domain": d,"count": n,"quality": round(q,3)}
                                  for d,n,q in by_domain],
            }
        except Exception:
            return {"session": self._session}

    def expertise(self) -> Dict[str, float]:
        """مستوى الخبرة في كل مجال (0-1)"""
        try:
            with sqlite3.connect(self._db) as c:
                rows = c.execute(
                    "SELECT domain,COUNT(*),AVG(quality) FROM learned_qa GROUP BY domain"
                ).fetchall()
            if not rows: return {}
            mx = max(r[1] for r in rows)
            return {d: round(n/mx*0.5 + q*0.5, 3) for d,n,q in rows}
        except Exception:
            return {}
