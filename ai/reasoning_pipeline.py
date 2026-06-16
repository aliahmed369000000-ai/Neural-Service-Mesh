"""
Reasoning Pipeline — Question → CKG → Neural Core → Decision → Answer
========================================================================
يربط NeuralCore (ai/neural_core.py) بالـ Cognitive Knowledge Graph الفعلي
(knowledge/cognitive_graph.json عبر CKGManager في ai/knowledge_trainer.py).

التدفق
------
1. Question (نص عربي)
     → ArabicNLPEngine.analyse(text) → متجه 7 أبعاد أولي (نحوي/صرفي/دلالي)
       + استخراج المفاهيم المرشّحة من النص (مطابقة مباشرة مع أسماء CKG)

2. CKG
     → لكل مفهوم مطابق: vec = VectorEncoder.encode(...، importance=strength)
     → دمج متجهات المفاهيم المطابقة (متوسط مرجّح بـ strength) مع متجه
       ArabicNLP → متجه سياق نهائي (7,) — "context_vector"

3. Neural Core
     → NeuralCore.forward(context_vector) → 4 أوزان توجيه
       (W_SEMANTIC, W_SCORE, W_MEMORY, W_TOPOLOGY) [Decision]
     → NeuralCore.recall(context_vector) → ذكريات سابقة ذات صلة

4. Decision
     → استخدام أوزان التوجيه لترتيب:
         - المفاهيم المطابقة مباشرة (وزنها W_SEMANTIC * strength)
         - المفاهيم المرتبطة عبر CKG.relations (وزنها W_TOPOLOGY * relation.weight)
         - الذكريات المسترجَعة من NeuralCore.memory (وزنها W_MEMORY * similarity)
       ثم دمج وترتيب القائمة الكلية بالنتيجة المركّبة (وزنها W_SCORE كعامل عام)

5. Answer
     → ملخص نصي + قائمة مفاهيم مرتبة + البيانات الخام (للاستخدام البرمجي)

التدريب (target)
-----------------
الهدف (target) المستخدم في train_and_remember لكل سؤال **مبني فعلياً من
strength/weight المفاهيم المسترجعة من CKG** — لا قيم ثابتة ولا قياسات
منفصلة:

  target[0] = avg(strength) للمفاهيم المطابقة مباشرة      (مكوّن دلالي)
  target[1] = avg(relation.weight) للعلاقات المُتبعة       (مكوّن تسجيل/ترتيب)
  target[2] = نسبة الذكريات المسترجَعة لها تشابه >= 0.5    (مكوّن ذاكرة)
  target[3] = تنوع الـ clusters المطابقة / إجمالي المطابقات (مكوّن طوبولوجي)

ثم target = target / sum(target) (تطبيع لتجمع=1)، وإن كانت كل القيم صفراً
يُستخدم target = [0.30, 0.35, 0.25, 0.10] (التوزيع الافتراضي الموجود
أصلاً في النظام).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ai.neural_core import NeuralCore, get_default_core
from ai.knowledge_trainer import CKGManager, VectorEncoder, DOMAIN_CODES
from ai.experience_store import Episode, EpisodeStore
from ai.experience_trainer import score_episode

try:
    from ai.arabic_nlp import ArabicNLPEngine
except Exception:  # pragma: no cover - الوحدة قد لا تكون متاحة في كل البيئات
    ArabicNLPEngine = None  # type: ignore


logger = logging.getLogger("ReasoningPipeline")

_TASHKEEL_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")


def _normalize_arabic(text: str) -> str:
    clean = _TASHKEEL_RE.sub("", text)
    clean = re.sub(r"[أإآٱ]", "ا", clean)
    return clean.strip()


# ════════════════════════════════════════════════════════════════════════
# نتائج التدفق (Dataclasses)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class MatchedConcept:
    name: str
    cluster: str
    strength: float
    frequency: int
    score: float = 0.0  # تُحسب في خطوة Decision


@dataclass
class RelatedConcept:
    name: str
    cluster: str
    relation_type: str
    relation_weight: float
    via: str  # المفهوم المصدر الذي قاد إليه
    score: float = 0.0


@dataclass
class MemoryHit:
    similarity: float
    metadata: dict
    score: float = 0.0


@dataclass
class PipelineResult:
    question: str
    context_vector: List[float]
    decision_weights: Dict[str, float]   # W_SEMANTIC, W_SCORE, W_MEMORY, W_TOPOLOGY
    matched_concepts: List[MatchedConcept]
    related_concepts: List[RelatedConcept]
    memory_hits: List[MemoryHit]
    ranked_concepts: List[Dict[str, Any]]  # القائمة النهائية المرتبة (للإجابة)
    answer_text: str
    target_used: Optional[List[float]] = None
    train_loss: Optional[float] = None
    memory_index: Optional[int] = None
    episode_id: Optional[str] = None
    quality: Optional[Dict[str, float]] = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "context_vector": self.context_vector,
            "decision_weights": self.decision_weights,
            "matched_concepts": [vars(m) for m in self.matched_concepts],
            "related_concepts": [vars(r) for r in self.related_concepts],
            "memory_hits": [vars(h) for h in self.memory_hits],
            "ranked_concepts": self.ranked_concepts,
            "answer_text": self.answer_text,
            "target_used": self.target_used,
            "train_loss": self.train_loss,
            "memory_index": self.memory_index,
            "episode_id": self.episode_id,
            "quality": self.quality,
        }


# ════════════════════════════════════════════════════════════════════════
# خط الأنابيب الرئيسي
# ════════════════════════════════════════════════════════════════════════

class ReasoningPipeline:
    """
    Question → CKG → Neural Core → Decision → Answer

    Parameters
    ----------
    core : NeuralCore أو None (يُستخدم get_default_core() إن لم يُحدَّد)
    ckg  : CKGManager أو None (يُحمَّل من knowledge/cognitive_graph.json تلقائياً)
    max_matched_concepts : أقصى عدد مفاهيم مطابقة مباشرة من النص
    max_related_per_concept : أقصى عدد علاقات تُتبَع لكل مفهوم مطابق
    top_k_memory : عدد الذكريات المسترجَعة من NeuralCore.memory
    train_on_query : إن كان True، كل استدعاء answer() يُدرِّب NeuralCore
        ويخزّن النتيجة في الذاكرة (train_and_remember). إن False،
        فقط forward() + recall() (بدون تعلّم).
    """

    def __init__(
        self,
        core: Optional[NeuralCore] = None,
        ckg: Optional[CKGManager] = None,
        max_matched_concepts: int = 8,
        max_related_per_concept: int = 5,
        top_k_memory: int = 5,
        train_on_query: bool = True,
        domain: str = "general",
        core_save_path: Optional[str] = "models/neural_core",
        autosave_every: int = 1,
        episode_store: Optional[EpisodeStore] = None,
        record_episodes: bool = True,
    ):
        self.core = core if core is not None else get_default_core(core_save_path or "models/neural_core")
        self.ckg = ckg if ckg is not None else CKGManager()
        self.encoder = VectorEncoder()

        self.arabic_engine = None
        if ArabicNLPEngine is not None:
            try:
                self.arabic_engine = ArabicNLPEngine(ckg=self.ckg)
            except Exception as e:
                logger.warning(f"ArabicNLPEngine init failed: {e}")

        self.max_matched_concepts = max_matched_concepts
        self.max_related_per_concept = max_related_per_concept
        self.top_k_memory = top_k_memory
        self.train_on_query = train_on_query
        self.domain = domain
        self.core_save_path = core_save_path
        self.autosave_every = max(0, autosave_every)
        self._queries_since_save = 0

        self.record_episodes = record_episodes
        self.episode_store = episode_store if episode_store is not None else (
            EpisodeStore() if record_episodes else None
        )

        logger.info(
            f"ReasoningPipeline ready — core={self.core!r}  "
            f"ckg_concepts={self.ckg.concept_count()}  "
            f"arabic_nlp={'on' if self.arabic_engine else 'off'}  "
            f"episodes={'on' if self.episode_store else 'off'}"
        )

    def save_core(self) -> Optional[str]:
        """يحفظ NeuralCore (الشبكة + الذاكرة + حالة التطور) إلى core_save_path."""
        if self.core_save_path is None:
            return None
        return self.core.save(self.core_save_path)

    # ────────────────────────────────────────────────────────────────
    # الخطوة 1+2: Question → CKG → context_vector
    # ────────────────────────────────────────────────────────────────

    def _match_concepts(self, text: str) -> List[MatchedConcept]:
        """يطابق أسماء مفاهيم CKG الموجودة كنص فرعي داخل السؤال."""
        clean_q = _normalize_arabic(text)
        concepts = self.ckg._data.get("concepts", {})

        matches: List[MatchedConcept] = []
        for name, c in concepts.items():
            name_clean = _normalize_arabic(name)
            if name_clean and name_clean in clean_q:
                matches.append(MatchedConcept(
                    name=name,
                    cluster=c.get("cluster", "عام"),
                    strength=float(c.get("strength", 0.1)),
                    frequency=int(c.get("frequency", 1)),
                ))

        # الأقوى أولاً (strength)، ثم الأكثر تكراراً
        matches.sort(key=lambda m: (m.strength, m.frequency), reverse=True)
        return matches[: self.max_matched_concepts]

    def _related_concepts(self, matched: List[MatchedConcept]) -> List[RelatedConcept]:
        """يتبع العلاقات (relations) من كل مفهوم مطابق إلى مفاهيم مجاورة."""
        relations = self.ckg._data.get("relations", {})
        concepts = self.ckg._data.get("concepts", {})
        matched_names = {m.name for m in matched}

        related: List[RelatedConcept] = []
        for m in matched:
            found_for_this = 0
            for key, r in relations.items():
                src, tgt = r.get("source"), r.get("target")
                if src != m.name:
                    continue
                if tgt in matched_names:
                    continue  # تجاهل ما هو مطابق مباشرة بالفعل
                target_info = concepts.get(tgt, {})
                related.append(RelatedConcept(
                    name=tgt,
                    cluster=target_info.get("cluster", "عام"),
                    relation_type=r.get("relation_type", "related"),
                    relation_weight=float(r.get("weight", 0.5)),
                    via=m.name,
                ))
                found_for_this += 1
                if found_for_this >= self.max_related_per_concept:
                    break
        # الأعلى وزن علاقة أولاً
        related.sort(key=lambda r: r.relation_weight, reverse=True)
        return related

    def _build_context_vector(
        self,
        text: str,
        matched: List[MatchedConcept],
    ) -> np.ndarray:
        """
        يبني متجه السياق (7,) بدمج:
          - متجه ArabicNLP (نحوي/صرفي/دلالي) إن كان متاحاً
          - متوسط متجهات VectorEncoder للمفاهيم المطابقة (مرجّح بـ strength)
        """
        vectors: List[np.ndarray] = []
        weights: List[float] = []

        # 1) متجه التحليل اللغوي العربي (إن وُجد)
        if self.arabic_engine is not None:
            try:
                result = self.arabic_engine.analyse(text)
                arabic_vec = np.array(result.feature_vector.to_list(), dtype=np.float64)
                vectors.append(arabic_vec)
                weights.append(1.0)
            except Exception as e:
                logger.warning(f"ArabicNLP analyse failed: {e}")

        # 2) متجهات المفاهيم المطابقة (مرجّحة بـ strength)
        known_concepts = self.ckg.known_names()
        for m in matched:
            vec = self.encoder.encode(
                text=m.name,
                domain=self.domain,
                importance=m.strength,
                certainty=0.8,
                abstraction=0.5,
                known_concepts=known_concepts,
                related_count=m.frequency,
            )
            vectors.append(vec)
            weights.append(max(m.strength, 0.05))

        if not vectors:
            # fallback: متجه عام عبر VectorEncoder للنص الخام
            vec = self.encoder.encode(
                text=text, domain=self.domain,
                known_concepts=known_concepts, related_count=len(matched),
            )
            return vec

        weights_arr = np.array(weights, dtype=np.float64)
        weights_arr = weights_arr / weights_arr.sum()
        stacked = np.vstack(vectors)
        context_vector = (weights_arr[:, None] * stacked).sum(axis=0)
        return context_vector

    # ────────────────────────────────────────────────────────────────
    # الخطوة 4: Decision → ترتيب المفاهيم
    # ────────────────────────────────────────────────────────────────

    def _decide(
        self,
        weights: Dict[str, float],
        matched: List[MatchedConcept],
        related: List[RelatedConcept],
        memory_hits: List[MemoryHit],
    ) -> List[Dict[str, Any]]:
        """
        يحسب نتيجة مركّبة لكل عنصر (مفهوم مطابق/مرتبط/ذكرى) باستخدام
        أوزان القرار الأربعة الصادرة من NeuralCore.forward():

          matched.score  = W_SEMANTIC * strength       * W_SCORE
          related.score  = W_TOPOLOGY * relation_weight * W_SCORE
          memory.score   = W_MEMORY   * similarity      * W_SCORE
        """
        w_sem = weights["W_SEMANTIC"]
        w_score = weights["W_SCORE"]
        w_mem = weights["W_MEMORY"]
        w_topo = weights["W_TOPOLOGY"]

        ranked: List[Dict[str, Any]] = []

        for m in matched:
            m.score = round(w_sem * m.strength * (1.0 + w_score), 6)
            ranked.append({
                "type": "matched_concept",
                "name": m.name,
                "cluster": m.cluster,
                "strength": m.strength,
                "score": m.score,
            })

        for r in related:
            r.score = round(w_topo * r.relation_weight * (1.0 + w_score), 6)
            ranked.append({
                "type": "related_concept",
                "name": r.name,
                "cluster": r.cluster,
                "via": r.via,
                "relation_type": r.relation_type,
                "relation_weight": r.relation_weight,
                "score": r.score,
            })

        for h in memory_hits:
            h.score = round(w_mem * h.similarity * (1.0 + w_score), 6)
            ranked.append({
                "type": "memory",
                "similarity": h.similarity,
                "metadata": {k: v for k, v in h.metadata.items() if k != "raw_vector"},
                "score": h.score,
            })

        ranked.sort(key=lambda d: d["score"], reverse=True)
        return ranked

    # ────────────────────────────────────────────────────────────────
    # هدف التدريب (target) من strength/weight الفعلية
    # ────────────────────────────────────────────────────────────────

    def _build_target(
        self,
        matched: List[MatchedConcept],
        related: List[RelatedConcept],
        memory_hits: List[MemoryHit],
    ) -> np.ndarray:
        """
        target[0] = متوسط strength للمفاهيم المطابقة مباشرة
        target[1] = متوسط relation_weight للعلاقات المتبوعة
        target[2] = نسبة الذكريات بتشابه >= 0.5
        target[3] = تنوع clusters المطابقة (عدد clusters مختلفة / عدد المطابقات)
        """
        # target[0]
        if matched:
            t0 = float(np.mean([m.strength for m in matched]))
        else:
            t0 = 0.0

        # target[1]
        if related:
            t1 = float(np.mean([r.relation_weight for r in related]))
        else:
            t1 = 0.0

        # target[2]
        if memory_hits:
            t2 = sum(1 for h in memory_hits if h.similarity >= 0.5) / len(memory_hits)
        else:
            t2 = 0.0

        # target[3]
        if matched:
            n_clusters = len({m.cluster for m in matched})
            t3 = n_clusters / len(matched)
        else:
            t3 = 0.0

        target = np.array([t0, t1, t2, t3], dtype=np.float64)
        total = target.sum()
        if total <= 0.0:
            # الهدف الافتراضي الموجود أصلاً في النظام (W_SEMANTIC..W_TOPOLOGY)
            return np.array([0.30, 0.35, 0.25, 0.10], dtype=np.float64)
        return target / total

    # ────────────────────────────────────────────────────────────────
    # الخطوة 5: Answer
    # ────────────────────────────────────────────────────────────────

    def _build_answer_text(
        self,
        question: str,
        ranked: List[Dict[str, Any]],
        weights: Dict[str, float],
    ) -> str:
        if not ranked:
            return f"لم يتم العثور على مفاهيم مرتبطة بالسؤال: «{question}»."

        top = ranked[:5]
        lines = [f"بناءً على السؤال «{question}»، أهم المفاهيم المرتبطة:"]
        for item in top:
            if item["type"] == "matched_concept":
                lines.append(
                    f"- {item['name']} (تصنيف: {item['cluster']}, "
                    f"قوة={item['strength']:.3f}, نتيجة={item['score']:.4f})"
                )
            elif item["type"] == "related_concept":
                lines.append(
                    f"- {item['name']} ← مرتبط بـ {item['via']} "
                    f"({item['relation_type']}, وزن={item['relation_weight']:.3f}, "
                    f"نتيجة={item['score']:.4f})"
                )
            else:  # memory
                meta = item["metadata"]
                lines.append(
                    f"- (من الذاكرة) تشابه={item['similarity']:.3f}, "
                    f"نتيجة={item['score']:.4f}, بيانات={meta}"
                )

        lines.append(
            "\nأوزان القرار (Decision): "
            f"دلالي={weights['W_SEMANTIC']:.3f}, "
            f"تسجيل={weights['W_SCORE']:.3f}, "
            f"ذاكرة={weights['W_MEMORY']:.3f}, "
            f"طوبولوجي={weights['W_TOPOLOGY']:.3f}"
        )
        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────
    # الواجهة الرئيسية
    # ────────────────────────────────────────────────────────────────

    def answer(self, question: str) -> PipelineResult:
        """
        ينفّذ التدفق الكامل: Question → CKG → Neural Core → Decision → Answer.

        إن كان `train_on_query=True`، يُدرِّب NeuralCore على هذا السؤال
        (target مبني من strength/weight الفعلية) ويخزّن النتيجة في الذاكرة.
        """
        # 1+2: Question → CKG → context_vector
        matched = self._match_concepts(question)
        related = self._related_concepts(matched)
        context_vector = self._build_context_vector(question, matched)

        # 3: Neural Core → Decision weights + memory recall
        if self.train_on_query:
            target = self._build_target(matched, related, [])
            train_result = self.core.train_and_remember(
                context_vector, target,
                metadata={
                    "question": question,
                    "matched": [m.name for m in matched],
                    "domain": self.domain,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            decision_vec = np.array(train_result["output"], dtype=np.float64)
            train_loss = train_result["loss"]
            memory_index = train_result["memory_index"]
        else:
            decision_vec = self.core.forward(context_vector)
            train_loss = None
            memory_index = None
            target = None

        memory_raw = self.core.recall(context_vector, top_k=self.top_k_memory)
        memory_hits = [MemoryHit(similarity=r["similarity"], metadata=r["metadata"])
                        for r in memory_raw]

        weights = {
            "W_SEMANTIC": round(float(decision_vec[0]), 6),
            "W_SCORE":    round(float(decision_vec[1]), 6),
            "W_MEMORY":   round(float(decision_vec[2]), 6),
            "W_TOPOLOGY": round(float(decision_vec[3]), 6),
        }

        # إن كان target قد بُني بناءً على memory_hits قبل recall، أعد بناءه
        # هنا بشكل صحيح لإظهار target_used الفعلي (لا يُعاد التدريب).
        if target is not None:
            target = self._build_target(matched, related, memory_hits)

        # 4: Decision
        ranked = self._decide(weights, matched, related, memory_hits)

        # 5: Answer
        answer_text = self._build_answer_text(question, ranked, weights)

        # ── Experience Learning: بناء وتخزين Episode (Requirements #1,#2,#4,#5) ──
        episode_id: Optional[str] = None
        quality: Optional[Dict[str, float]] = None
        memory_hits_dicts = [
            {"similarity": h.similarity, "metadata": {k: v for k, v in h.metadata.items() if k != "raw_vector"}}
            for h in memory_hits
        ]

        if self.episode_store is not None:
            quality = score_episode(
                matched_concepts=[vars(m) for m in matched],
                related_concepts=[vars(r) for r in related],
                memory_hits=memory_hits_dicts,
                decision_weights=weights,
            )
            confidence = quality["answer_confidence"]

            episode = Episode(
                question=question,
                matched_concepts=[vars(m) for m in matched],
                related_concepts=[vars(r) for r in related],
                decision_weights=weights,
                confidence=confidence,
                answer=answer_text,
                context_vector=context_vector.tolist(),
                target_used=target.tolist() if target is not None else None,
                train_loss=train_loss,
                memory_hits=memory_hits_dicts,
                quality=quality,
            )
            try:
                episode_id = self.episode_store.add(episode)
            except Exception as e:
                logger.warning(f"EpisodeStore.add failed: {e}")
                episode_id = None

        # حفظ تلقائي اختياري (بعد التدريب فقط)
        if self.train_on_query and self.autosave_every > 0:
            self._queries_since_save += 1
            if self._queries_since_save >= self.autosave_every:
                try:
                    self.save_core()
                except Exception as e:
                    logger.warning(f"NeuralCore autosave failed: {e}")
                self._queries_since_save = 0

        return PipelineResult(
            question=question,
            context_vector=context_vector.tolist(),
            decision_weights=weights,
            matched_concepts=matched,
            related_concepts=related,
            memory_hits=memory_hits,
            ranked_concepts=ranked,
            answer_text=answer_text,
            target_used=target.tolist() if target is not None else None,
            train_loss=train_loss,
            memory_index=memory_index,
            episode_id=episode_id,
            quality=quality,
        )

    def submit_feedback(
        self,
        episode_id: str,
        rating: Optional[str] = None,
        correction_text: Optional[str] = None,
    ) -> dict:
        """
        واجهة خارجية لتقديم تغذية رجعية على حلقة سابقة.

        Parameters
        ----------
        episode_id : str
            معرّف الحلقة (episode_id) من نتيجة answer() السابقة.
        rating : "up" / "down" / None
            "up"   = الإجابة كانت صحيحة / مفيدة
            "down" = الإجابة كانت خاطئة / مضللة
            None   = تعليق فقط بدون تقييم
        correction_text : str أو None
            نص تصحيحي اختياري من المستخدم.

        Returns
        -------
        dict : {"status": "ok"/"not_found"/"error", "episode_id": ..., "rating": ...}
        """
        if rating not in ("up", "down", None):
            return {"status": "error", "message": "rating يجب أن يكون 'up' أو 'down' أو None"}

        if self.episode_store is None:
            return {"status": "error", "message": "episode_store غير مُهيَّأ"}

        success = self.episode_store.update_feedback(episode_id, rating, correction_text)
        if not success:
            return {"status": "not_found", "episode_id": episode_id}

        return {
            "status": "ok",
            "episode_id": episode_id,
            "rating": rating,
            "has_correction": correction_text is not None,
        }
