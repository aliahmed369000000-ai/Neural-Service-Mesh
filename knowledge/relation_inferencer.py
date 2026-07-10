"""
Relation Inferencer — الأولوية 3 (الذكاء)
==========================================
يكتشف العلاقات بين المفاهيم تلقائياً عبر:

  1. Co-occurrence  — مفهومان ظهرا في نفس الآية
  2. Cluster Affinity — مفاهيم في نفس الـ cluster ترتبط دلالياً
  3. Chain Inference — إذا A→B و B→C بشكل قوي، نستنتج A→C
  4. Strength Decay  — علاقات لم تُرَ منذ فترة تضعف تدريجياً

يُشغَّل في background بعد كل batch ingestion.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RelationInferencer:
    """
    محرك استنتاج العلاقات بين المفاهيم.

    Usage:
        inferencer = RelationInferencer(ckg)
        report = inferencer.run(verbose=True)
    """

    def __init__(
        self,
        ckg,                                   # CognitiveKnowledgeGraph
        chain_threshold:   float = 0.5,        # الحد الأدنى لوزن علاقة لتُشكّل سلسلة
        chain_weight:      float = 0.3,        # وزن العلاقة المُستنتجة
        affinity_weight:   float = 0.2,        # وزن علاقات cluster
        decay_factor:      float = 0.02,       # معدل الضعف اليومي للعلاقات غير المُرؤية
        decay_days:        float = 30.0,       # عدد أيام عدم الرؤية قبل بدء الضعف
    ):
        self._ckg              = ckg
        self._chain_threshold  = chain_threshold
        self._chain_weight     = chain_weight
        self._affinity_weight  = affinity_weight
        self._decay_factor     = decay_factor
        self._decay_days       = decay_days

    def run(self, verbose: bool = False) -> Dict:
        """
        شغّل كل آليات الاستنتاج وارجع تقريراً.
        """
        report = {
            "cluster_affinity":  self._infer_cluster_affinity(verbose),
            "chain_inference":   self._infer_chains(verbose),
            "strength_decay":    self._apply_decay(verbose),
        }
        if verbose:
            print(f"[RelationInferencer] نتائج الاستنتاج:")
            for k, v in report.items():
                print(f"  {k}: {v}")
        return report

    # ── 1. Cluster Affinity ───────────────────────────────────────────────

    def _infer_cluster_affinity(self, verbose: bool) -> Dict:
        """
        المفاهيم في نفس الـ cluster لها علاقة دلالية ضمنية.
        مثال: "صبر" و"تقوى" كلاهما في cluster "أخلاق" → علاقة دلالية بوزن 0.2
        """
        ckg = self._ckg
        added = 0

        with ckg._lock:
            # تجميع المفاهيم حسب cluster
            by_cluster: Dict[str, List[str]] = {}
            for name, concept in ckg._concepts.items():
                cl = concept.cluster
                if cl and cl != "هيكل" and cl != "غير مصنّف":
                    by_cluster.setdefault(cl, []).append(name)

            for cluster, names in by_cluster.items():
                if len(names) < 2:
                    continue
                # ربط كل المفاهيم في نفس الـ cluster ببعضها (semantic)
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        a, b = names[i], names[j]
                        key_ab = f"{a}→{b}"
                        key_ba = f"{b}→{a}"

                        # أضف فقط إذا لم تكن علاقة co_occurrence أقوى موجودة
                        existing_ab = ckg._relations.get(key_ab)
                        if not existing_ab or existing_ab.relation_type != "co_occurrence":
                            ckg.add_relation(
                                a, b,
                                evidence=f"cluster:{cluster}",
                                relation_type="semantic",
                                weight_boost=self._affinity_weight,
                            )
                            added += 1

                        existing_ba = ckg._relations.get(key_ba)
                        if not existing_ba or existing_ba.relation_type != "co_occurrence":
                            ckg.add_relation(
                                b, a,
                                evidence=f"cluster:{cluster}",
                                relation_type="semantic",
                                weight_boost=self._affinity_weight,
                            )
                            added += 1

        return {"relations_inferred": added}

    # ── 2. Chain Inference ────────────────────────────────────────────────

    def _infer_chains(self, verbose: bool) -> Dict:
        """
        إذا A→B (weight ≥ threshold) و B→C (weight ≥ threshold)
        وليست A→C موجودة بعد، استنتج A→C بوزن أقل.

        صبر → ابتلاء → آخرة  ⟹  صبر → آخرة
        """
        ckg = self._ckg
        inferred = 0
        threshold = self._chain_threshold

        with ckg._lock:
            # جمع العلاقات القوية فقط
            strong: List[Tuple[str, str, float]] = [
                (r.source, r.target, r.weight)
                for r in ckg._relations.values()
                if r.weight >= threshold
            ]

        # بناء فهرس سريع: source → [(target, weight)]
        out_edges: Dict[str, List[Tuple[str, float]]] = {}
        for src, tgt, w in strong:
            out_edges.setdefault(src, []).append((tgt, w))

        # استنتاج السلاسل
        to_add: List[Tuple[str, str, str]] = []   # (a, c, evidence)
        for a, a_edges in out_edges.items():
            for b, w_ab in a_edges:
                for c, w_bc in out_edges.get(b, []):
                    if c == a:
                        continue
                    key_ac = f"{a}→{c}"
                    with ckg._lock:
                        if key_ac not in ckg._relations:
                            to_add.append((a, c, f"chain:{a}→{b}→{c}"))

        for a, c, evidence in to_add:
            ckg.add_relation(
                a, c,
                evidence=evidence,
                relation_type="causal",
                weight_boost=self._chain_weight,
            )
            inferred += 1

        return {"chains_inferred": inferred}

    # ── 3. Strength Decay ─────────────────────────────────────────────────

    def _apply_decay(self, verbose: bool) -> Dict:
        """
        علاقات لم تُرَ منذ decay_days أيام تضعف بمعدل decay_factor.
        يمنع تراكم علاقات قديمة ضعيفة الصلة.
        """
        now = datetime.now(timezone.utc)
        decayed = 0
        removed = 0

        with self._ckg._lock:
            to_remove = []
            for key, r in self._ckg._relations.items():
                try:
                    last = datetime.fromisoformat(r.last_seen)
                    # جعل last aware إذا كان naive
                    if last.tzinfo is None:
                        last = last.replace(tzinfo=timezone.utc)
                    days_old = (now - last).days
                except Exception:
                    continue

                if days_old >= self._decay_days:
                    r.weight = round(
                        max(0.0, r.weight - self._decay_factor * (days_old - self._decay_days + 1)),
                        4,
                    )
                    decayed += 1
                    if r.weight <= 0.0:
                        to_remove.append(key)

            for key in to_remove:
                r = self._ckg._relations.pop(key)
                self._ckg._adj[r.source].discard(r.target)
                self._ckg._radj[r.target].discard(r.source)
                removed += 1

        return {"relations_decayed": decayed, "relations_removed": removed}
