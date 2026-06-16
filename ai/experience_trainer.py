"""
Experience Learning — الجزء 2: تقييم الجودة + ExperienceTrainer
===================================================================
Requirements #4-#7:
  - experience quality scoring (concept/relation coverage, memory recall
    quality, answer confidence)
  - ExperienceTrainer: replay top / recent / diverse episodes
  - NeuralCore يتحسّن من الخبرة المتراكمة بمرور الوقت
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ai.experience_store import Episode, EpisodeStore
from ai.neural_core import NeuralCore
from ai.core_history import CoreHistory, get_default_history, \
    EVENT_TRAINING_CYCLE, EVENT_ROLLBACK, EVENT_PROMOTION, EVENT_CONSOLIDATION

logger = logging.getLogger("ExperienceLearning")


# ════════════════════════════════════════════════════════════════════════
# 1) تقييم جودة التجربة (Requirement #4)
# ════════════════════════════════════════════════════════════════════════

def score_concept_coverage(matched_concepts: List[Dict[str, Any]],
                            max_expected: int = 5) -> float:
    """
    تغطية المفاهيم: نسبة عدد المفاهيم المطابقة إلى max_expected (مقصوصة عند 1.0).
    صفر إن لم يُطابَق أي مفهوم.
    """
    if not matched_concepts:
        return 0.0
    return float(min(1.0, len(matched_concepts) / max_expected))


def score_relation_coverage(related_concepts: List[Dict[str, Any]],
                             max_expected: int = 10) -> float:
    """
    تغطية العلاقات: نسبة عدد العلاقات المتبوعة (وأوزانها) إلى max_expected.
    تُحسب كـ (متوسط relation_weight) * (نسبة العدد إلى max_expected).
    """
    if not related_concepts:
        return 0.0
    count_ratio = min(1.0, len(related_concepts) / max_expected)
    avg_weight = float(np.mean([r.get("relation_weight", 0.0) for r in related_concepts]))
    return float(round(count_ratio * avg_weight, 6))


def score_memory_recall_quality(memory_hits: List[Dict[str, Any]],
                                 self_match_threshold: float = 0.999) -> float:
    """
    جودة استرجاع الذاكرة: متوسط التشابه للذكريات المسترجَعة،
    باستثناء الذكرى الذاتية (التي خُزِّنت في هذه الخطوة نفسها، تشابه≈1.0)
    إن وُجدت أكثر من ذكرى واحدة.
    صفر إن لم تُسترجَع أي ذكرى.
    """
    if not memory_hits:
        return 0.0
    sims = [h.get("similarity", 0.0) for h in memory_hits]
    if len(sims) > 1:
        # استبعاد أعلى تشابه (الذكرى الذاتية المخزَّنة للتو) إن كانت ~1.0
        if max(sims) >= self_match_threshold:
            sims_excl = [s for s in sims if s < self_match_threshold]
            if sims_excl:
                sims = sims_excl
    return float(round(np.mean(sims), 6))


def score_answer_confidence(decision_weights: Dict[str, float],
                             matched_concepts: List[Dict[str, Any]]) -> float:
    """
    ثقة الإجابة: مزيج من:
      - W_SEMANTIC (ثقة الشبكة الدلالية)
      - متوسط strength للمفاهيم المطابقة (إن وُجدت)
    confidence = 0.5 * W_SEMANTIC + 0.5 * avg(strength)
    إن لم توجد مفاهيم مطابقة: confidence = W_SEMANTIC * 0.5 (عقوبة لعدم التغطية)
    """
    w_sem = float(decision_weights.get("W_SEMANTIC", 0.0))
    if matched_concepts:
        avg_strength = float(np.mean([m.get("strength", 0.0) for m in matched_concepts]))
        conf = 0.5 * w_sem + 0.5 * avg_strength
    else:
        conf = 0.5 * w_sem
    return float(round(min(1.0, max(0.0, conf)), 6))


def score_episode(
    matched_concepts: List[Dict[str, Any]],
    related_concepts: List[Dict[str, Any]],
    memory_hits: List[Dict[str, Any]],
    decision_weights: Dict[str, float],
    max_expected_concepts: int = 5,
    max_expected_relations: int = 10,
) -> Dict[str, float]:
    """
    يحسب كل مكوّنات الجودة + overall_quality (متوسط الأربعة).
    Requirement #5: تُخزَّن هذه القيم داخل الـ Episode (في حقل `quality`).
    """
    concept_coverage = score_concept_coverage(matched_concepts, max_expected_concepts)
    relation_coverage = score_relation_coverage(related_concepts, max_expected_relations)
    memory_recall_quality = score_memory_recall_quality(memory_hits)
    answer_confidence = score_answer_confidence(decision_weights, matched_concepts)

    overall = float(np.mean([
        concept_coverage, relation_coverage, memory_recall_quality, answer_confidence
    ]))

    return {
        "concept_coverage": round(concept_coverage, 6),
        "relation_coverage": round(relation_coverage, 6),
        "memory_recall_quality": round(memory_recall_quality, 6),
        "answer_confidence": round(answer_confidence, 6),
        "overall_quality": round(overall, 6),
    }


# ════════════════════════════════════════════════════════════════════════
# 2) ExperienceTrainer (Requirements #3, #6, #7)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ReplayReport:
    """نتيجة جولة replay واحدة."""
    strategy: str
    episodes_used: int
    avg_loss_before: Optional[float]
    avg_loss_after: Optional[float]
    losses: List[float]
    episode_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "episodes_used": self.episodes_used,
            "avg_loss_before": self.avg_loss_before,
            "avg_loss_after": self.avg_loss_after,
            "improved": (
                self.avg_loss_after is not None and self.avg_loss_before is not None
                and self.avg_loss_after < self.avg_loss_before
            ),
            "episode_ids": self.episode_ids,
        }


class ExperienceTrainer:
    """
    يحوّل حلقات سابقة (Episodes) إلى إشارات تدريب لـ NeuralCore.

    إشارة التدريب (training signal) لكل حلقة:
      x      = episode.context_vector  (7,)  — نفس متجه السياق الذي وُلِّد منه القرار
      target = إعادة بناء هدف محسَّن من quality scores + target_used الأصلي:

        target_refined = normalize(
            0.5 * target_used_original  +  0.5 * quality_components
        )

      حيث quality_components = [concept_coverage, relation_coverage,
                                  memory_recall_quality, answer_confidence]

      الفكرة: الهدف الأصلي (target_used) كان مبنياً من strength/weight لحظة
      السؤال، لكن quality scores تعكس "كيف كانت التجربة فعلياً" (شمولية
      التغطية والثقة) — الدمج بينهما هو "إشارة التعلّم من الخبرة" الفعلية،
      المختلفة عن إشارة CKG الأولية وحدها.

    Parameters
    ----------
    core  : NeuralCore المراد تحسينه
    store : EpisodeStore (مصدر الحلقات)
    """

    def __init__(self, core: NeuralCore, store: EpisodeStore):
        self.core = core
        self.store = store
        self._cycle_count: int = 0
        self._history: CoreHistory = get_default_history()

    # ── بناء إشارة تدريب من حلقة واحدة ────────────────────────────────

    def _episode_to_signal(self, ep: Episode) -> Optional[tuple]:
        if not ep.context_vector:
            return None

        x = np.array(ep.context_vector, dtype=np.float64)

        quality_vec = np.array([
            ep.quality.get("concept_coverage", 0.0),
            ep.quality.get("relation_coverage", 0.0),
            ep.quality.get("memory_recall_quality", 0.0),
            ep.quality.get("answer_confidence", 0.0),
        ], dtype=np.float64)

        if ep.target_used is not None and len(ep.target_used) == 4:
            original = np.array(ep.target_used, dtype=np.float64)
        else:
            original = np.array([0.30, 0.35, 0.25, 0.10], dtype=np.float64)

        combined = 0.5 * original + 0.5 * quality_vec
        total = combined.sum()
        if total <= 0.0:
            target = original
        else:
            target = combined / total

        return x, target

    # ── replay عام ───────────────────────────────────────────────────

    def _replay(self, episodes: List[Episode], strategy: str) -> ReplayReport:
        signals = []
        used_ids = []
        for ep in episodes:
            sig = self._episode_to_signal(ep)
            if sig is None:
                continue
            signals.append(sig)
            used_ids.append(ep.episode_id)

        if not signals:
            return ReplayReport(strategy, 0, None, None, [], [])

        losses_before = []
        for x, target in signals:
            out = self.core.forward(x)
            from ai.neural_core import mse_loss
            l, _ = mse_loss(out, target)
            losses_before.append(l)

        losses_after = []
        for x, target in signals:
            l = self.core.train_step(x, target)
            losses_after.append(l)
            self.core.evolve_if_plateau()

        self.store.mark_replayed(used_ids)

        return ReplayReport(
            strategy=strategy,
            episodes_used=len(signals),
            avg_loss_before=round(float(np.mean(losses_before)), 8),
            avg_loss_after=round(float(np.mean(losses_after)), 8),
            losses=[round(float(l), 8) for l in losses_after],
            episode_ids=used_ids,
        )

    # ── الإستراتيجيات الثلاث (Requirement #6) ──────────────────────────

    def replay_top(self, limit: int = 20) -> ReplayReport:
        """يعيد تدريب NeuralCore على أعلى الحلقات جودة (overall_quality)."""
        episodes = self.store.get_top_by_quality(limit=limit)
        return self._replay(episodes, strategy="top")

    def replay_recent(self, limit: int = 20) -> ReplayReport:
        """يعيد تدريب NeuralCore على أحدث الحلقات."""
        episodes = self.store.get_recent(limit=limit)
        return self._replay(episodes, strategy="recent")

    def replay_diverse(self, limit: int = 20, seed: Optional[int] = None) -> ReplayReport:
        """يعيد تدريب NeuralCore على عينة متنوعة (clusters مختلفة)."""
        episodes = self.store.get_diverse_sample(limit=limit, seed=seed)
        return self._replay(episodes, strategy="diverse")

    def replay_feedback(self, limit: int = 20) -> ReplayReport:
        """
        يعيد تدريب NeuralCore على الحلقات التي لديها external_feedback،
        مع تعديل الـ target بناءً على التقييم.

        معادلة تعديل الـ target:
        ─────────────────────────
        target_original = target_used (المخزَّن في الحلقة)

        إذا rating == "up":
            target_adjusted = target_original  (نُقوّيه كما هو — لا تغيير)
            يعني: الشبكة كانت صحيحة، نُعزّز نفس الهدف

        إذا rating == "down":
            target_adjusted = 1.0 - target_original  (عكس الهدف)
            المعادلة: target_adjusted_i = 1.0 - target_original_i  لكل عنصر i
            يعني: إذا كان target=[0.3, 0.5, 0.1, 0.1]
                  يصبح:  target=[0.7, 0.5, 0.9, 0.9]
            ثم يُطبَّع ليجمع إلى 1.0:
                  target_final = target_adjusted / sum(target_adjusted)
            المنطق: "down" يعني أن الشبكة أعطت أوزاناً خاطئة،
                    فندفعها في الاتجاه المعاكس.

        إذا correction_text موجود:
            يُسجَّل في الـ metadata فقط (للمراجعة البشرية لاحقاً)،
            لا يؤثر على الـ target حالياً (مستقبلي).

        الأولوية: حلقات الـ "down" أولاً (أكثر أهمية للتصحيح)،
                  ثم حلقات الـ "up".
        ─────────────────────────
        """
        episodes_with_fb = self.store.get_with_feedback(limit=limit)
        if not episodes_with_fb:
            return ReplayReport("feedback", 0, None, None, [], [])

        # ترتيب: "down" أولاً ثم "up"
        down_eps = [ep for ep in episodes_with_fb
                    if ep.external_feedback and ep.external_feedback.get("rating") == "down"]
        up_eps = [ep for ep in episodes_with_fb
                  if ep.external_feedback and ep.external_feedback.get("rating") == "up"]
        other_eps = [ep for ep in episodes_with_fb
                     if ep not in down_eps and ep not in up_eps]
        ordered = down_eps + up_eps + other_eps

        signals = []
        used_ids = []
        for ep in ordered:
            if not ep.context_vector:
                continue

            x = np.array(ep.context_vector, dtype=np.float64)

            if ep.target_used is not None and len(ep.target_used) == 4:
                target_original = np.array(ep.target_used, dtype=np.float64)
            else:
                target_original = np.array([0.30, 0.35, 0.25, 0.10], dtype=np.float64)

            rating = ep.external_feedback.get("rating") if ep.external_feedback else None

            if rating == "down":
                target_adjusted = 1.0 - target_original
                total = target_adjusted.sum()
                target = target_adjusted / total if total > 0 else target_original
            else:
                # "up" أو None: نُعزّز الهدف الأصلي كما هو
                target = target_original

            signals.append((x, target, ep))
            used_ids.append(ep.episode_id)

        if not signals:
            return ReplayReport("feedback", 0, None, None, [], [])

        losses_before = []
        for x, target, _ in signals:
            out = self.core.forward(x)
            from ai.neural_core import mse_loss
            l, _ = mse_loss(out, target)
            losses_before.append(l)

        losses_after = []
        for x, target, ep in signals:
            correction = ep.external_feedback.get("correction_text") if ep.external_feedback else None
            # correction_text مسجَّل للمراجعة البشرية فقط — لا يؤثر على التدريب حالياً
            if correction:
                logger.info(f"replay_feedback: episode={ep.episode_id} has correction_text (logged only): {correction[:80]}")
            l = self.core.train_step(x, target)
            losses_after.append(l)
            self.core.evolve_if_plateau()

        self.store.mark_replayed(used_ids)

        return ReplayReport(
            strategy="feedback",
            episodes_used=len(signals),
            avg_loss_before=round(float(np.mean(losses_before)), 8),
            avg_loss_after=round(float(np.mean(losses_after)), 8),
            losses=[round(float(l), 8) for l in losses_after],
            episode_ids=used_ids,
        )

    # ── دورة تدريب دورية كاملة (Requirement #3/#7) ─────────────────────

    def run_training_cycle(
        self,
        top_limit: int = 10,
        recent_limit: int = 10,
        diverse_limit: int = 10,
        save: bool = True,
        save_path: str = "models/neural_core",
        seed: Optional[int] = None,
        benchmark: Optional[Any] = None,  # ← BenchmarkSuite أو None (lazy-typed لتجنب circular import)
        rollback_threshold: float = 0.05,  # ← هامش التراجع المسموح في MSE
        consolidate_every: int = 5,  # ← دمج الذاكرة كل N دورة (0 = تعطيل)
        promote_every: int = 10,  # ← تنوّع بنيوي + ترقية كل N دورة (0 = تعطيل)
        n_variants: int = 3,  # ← عدد الـ variants في select_and_promote
        improvement_threshold: float = 0.02,  # ← الحد الأدنى للتحسّن المطلوب للترقية
    ) -> Dict[str, Any]:
        """
        دورة replay كاملة: top + recent + diverse، بالترتيب.
        تُحفَظ النواة بعد الدورة إن save=True.

        إذا مُرّر benchmark:
          1. يُحسب score_before = benchmark.evaluate(core) قبل التدريب.
          2. تُحفظ نسخة احتياطية مؤقتة من NeuralCore.
          3. تُنفَّذ دورة التدريب كالمعتاد.
          4. يُحسب score_after = benchmark.evaluate(core) بعد التدريب.
          5. إذا score_after > score_before + rollback_threshold:
               → تُستعاد النسخة الاحتياطية (rollback)، rolled_back=True.
             وإلا: تُحذف النسخة الاحتياطية، rolled_back=False.
          6. تُسجَّل نتائج Benchmark في التقرير النهائي.

        إن لم يُمرَّر benchmark (None)، تعمل الدالة بالضبط كما كانت
        (backward compatible) دون أي تغيير في السلوك.

        Returns: تقرير يحتوي نتائج الثلاث استراتيجيات + إحصاءات المخزن
                 (+ معلومات benchmark إن وُجدت).
        """
        n_episodes = self.store.count()
        if n_episodes == 0:
            return {
                "status": "no_episodes",
                "message": "لا توجد حلقات مخزَّنة بعد — لا يمكن تشغيل دورة تدريب.",
            }

        # ── سجل التاريخ: hash الحالة الأب قبل أي تعديل ──
        parent_hash = self._history.get_last_state_hash()

        # 1. قبل التدريب: benchmark + نسخة احتياطية
        backup_path = None
        score_before = None
        if benchmark is not None:
            score_before = benchmark.evaluate(self.core)["score"]
            backup_path = save_path + "_rollback_backup"
            try:
                self.core.save(backup_path)
            except Exception as e:
                logger.warning(f"Benchmark backup failed: {e}")
                backup_path = None

        # 2. تنفيذ التدريب (دون تغيير)
        report_top = self.replay_top(limit=top_limit)
        report_recent = self.replay_recent(limit=recent_limit)
        report_diverse = self.replay_diverse(limit=diverse_limit, seed=seed)

        # 3. بعد التدريب: benchmark + قرار rollback
        score_after = None
        rolled_back = False
        if benchmark is not None and backup_path is not None:
            score_after = benchmark.evaluate(self.core)["score"]
            if score_after > score_before + rollback_threshold:
                # تراجع الأداء بأكثر من الهامش المسموح → rollback
                try:
                    from ai.neural_core import NeuralCore
                    self.core = NeuralCore.load(backup_path)
                    rolled_back = True
                    logger.warning(
                        f"run_training_cycle: ROLLBACK triggered — "
                        f"score_before={score_before:.6f}, score_after={score_after:.6f}"
                    )
                except Exception as e:
                    logger.error(f"Rollback failed: {e}")
            # تنظيف النسخة الاحتياطية
            try:
                import os
                import shutil
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path, ignore_errors=True)
            except Exception:
                pass

        # ── سجل التاريخ: حدث rollback إن حدث ──
        if rolled_back:
            self._history.log_event(
                core=self.core,
                event_type=EVENT_ROLLBACK,
                benchmark_score=score_before,  # score ما قبل التدريب (الأفضل)
                parent_hash=parent_hash,
                extra={
                    "score_before": score_before,
                    "score_after": score_after,
                    "rollback_threshold": rollback_threshold,
                },
            )

        # 4. حفظ إن save=True (بعد rollback أو بعد تدريب ناجح)
        if save:
            try:
                self.core.save(save_path)
            except Exception as e:
                logger.warning(f"NeuralCore save failed after training cycle: {e}")

        # 4.5. دمج الذاكرة الدوري (consolidation)
        self._cycle_count += 1
        consolidation_result = None
        if consolidate_every > 0 and self._cycle_count % consolidate_every == 0:
            try:
                consolidation_result = self.core.memory.consolidate()
                logger.info(f"Memory consolidation at cycle {self._cycle_count}: {consolidation_result}")
            except Exception as e:
                logger.warning(f"Memory consolidation failed: {e}")

        # 4.6. تنوّع بنيوي + ترقية دورية (structural variation & promotion)
        promotion_result = None
        if (benchmark is not None
                and promote_every > 0
                and self._cycle_count % promote_every == 0):
            try:
                from ai.neural_core import NeuralCore
                new_core, promo_report = NeuralCore.select_and_promote(
                    core=self.core,
                    benchmark=benchmark,
                    n_variants=n_variants,
                    improvement_threshold=improvement_threshold,
                )
                if promo_report["promoted"]:
                    self.core = new_core   # استبدال النواة الحالية بالأفضل
                promotion_result = promo_report
            except Exception as e:
                logger.warning(f"select_and_promote failed: {e}")

        # ── سجل التاريخ: حدث promotion إن حدث ──
        if promotion_result is not None and promotion_result.get("promoted"):
            self._history.log_event(
                core=self.core,
                event_type=EVENT_PROMOTION,
                benchmark_score=promotion_result.get("best_variant_score"),
                parent_hash=parent_hash,
                extra=promotion_result,
            )

        # 5. التقرير النهائي
        result = {
            "status": "ok",
            "store_stats": self.store.stats(),
            "top": report_top.to_dict(),
            "recent": report_recent.to_dict(),
            "diverse": report_diverse.to_dict(),
        }
        if benchmark is not None:
            result["benchmark"] = {
                "score_before": round(score_before, 8) if score_before is not None else None,
                "score_after": round(score_after, 8) if score_after is not None else None,
                "rolled_back": rolled_back,
                "rollback_threshold": rollback_threshold,
                "n_samples": benchmark.n_samples,
            }
        if consolidation_result is not None:
            result["memory_consolidation"] = consolidation_result
        if promotion_result is not None:
            result["structural_variation"] = promotion_result

        # ── سجل التاريخ: حدث training_cycle في النهاية ──
        self._history.log_event(
            core=self.core,
            event_type=EVENT_TRAINING_CYCLE,
            benchmark_score=score_after if score_after is not None else None,
            parent_hash=parent_hash,
            extra={
                "cycle": self._cycle_count,
                "top_trained": report_top.to_dict().get("episodes_used"),
                "recent_trained": report_recent.to_dict().get("episodes_used"),
            },
        )

        return result
