"""
Cognitive Knowledge Graph (CKG) — الأولوية 2 (القلب)
======================================================
الطبقة التي ستحوّل النظام من "مخزن بيانات" إلى "جهاز معرفي".

البنية:
  ┌─────────────────────────────────────────────────────┐
  │  CONCEPT (عقدة)                                     │
  │    name        : str          "صبر"                  │
  │    cluster     : str          "أخلاق"               │
  │    sources     : List[str]    ["quran:2:45", ...]    │
  │    frequency   : int          عدد مرات الظهور       │
  │    strength    : float        0-1 قوة المفهوم       │
  │    first_seen  : ISO str                             │
  │    last_seen   : ISO str                             │
  ├─────────────────────────────────────────────────────┤
  │  RELATION (حافة موزونة)                             │
  │    source      : str          "صبر"                  │
  │    target      : str          "ابتلاء"              │
  │    weight      : float        0-1 قوة العلاقة       │
  │    relation_type: str         "co_occurrence" / ...  │
  │    evidence    : List[str]    الآيات الداعمة        │
  │    count       : int          عدد مرات المشاهدة     │
  └─────────────────────────────────────────────────────┘

الواجهة الكاملة:
  ckg.add_concept(name, cluster, source)
  ckg.add_relation(source, target, evidence, relation_type)
  ckg.query_related(concept, top_k)       → [(concept, weight)]
  ckg.get_concept_strength(concept)        → float
  ckg.get_strongest_concepts(cluster, n)   → list
  ckg.find_path(a, b)                      → [concept, ...]
  ckg.cross_source_concepts()              → concepts in 2+ sources
  ckg.stats()                              → dict
  ckg.save() / ckg.load()

التخزين:
  knowledge/cognitive_graph.json
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_GRAPH_FILE = Path("./knowledge/cognitive_graph.json")
_NOW = lambda: datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════════
# Concept Node
# ═══════════════════════════════════════════════════════════════════════════

class Concept:
    """عقدة في الـ CKG تمثل مفهوماً معرفياً واحداً."""

    __slots__ = (
        "name", "cluster", "sources", "frequency",
        "strength", "first_seen", "last_seen",
    )

    def __init__(
        self,
        name:       str,
        cluster:    str = "غير مصنّف",
        sources:    Optional[List[str]] = None,
        frequency:  int = 1,
        strength:   float = 0.0,
        first_seen: Optional[str] = None,
        last_seen:  Optional[str] = None,
    ):
        self.name       = name
        self.cluster    = cluster
        self.sources    = sources or []
        self.frequency  = frequency
        self.strength   = strength
        self.first_seen = first_seen or _NOW()
        self.last_seen  = last_seen  or _NOW()

    def touch(self, source: str) -> None:
        """سجّل ظهوراً جديداً."""
        self.frequency += 1
        self.last_seen  = _NOW()
        if source and source not in self.sources:
            self.sources.append(source)

    def compute_strength(self, max_freq: int) -> None:
        """
        strength = log-normalized frequency.
        نفضّل log حتى لا تهيمن المفاهيم الشائعة جداً.
        """
        if max_freq <= 0:
            self.strength = 0.0
            return
        self.strength = round(
            math.log(self.frequency + 1) / math.log(max_freq + 1), 4
        )

    def to_dict(self) -> dict:
        return {
            "name":       self.name,
            "cluster":    self.cluster,
            "sources":    self.sources,
            "frequency":  self.frequency,
            "strength":   self.strength,
            "first_seen": self.first_seen,
            "last_seen":  self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Concept":
        return cls(
            name       = d["name"],
            cluster    = d.get("cluster", "غير مصنّف"),
            sources    = d.get("sources", []),
            frequency  = d.get("frequency", 1),
            strength   = d.get("strength", 0.0),
            first_seen = d.get("first_seen"),
            last_seen  = d.get("last_seen"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Relation Edge
# ═══════════════════════════════════════════════════════════════════════════

class Relation:
    """حافة موزونة بين مفهومين."""

    __slots__ = (
        "source", "target", "weight",
        "relation_type", "evidence", "count",
        "first_seen", "last_seen",
    )

    TYPES = {
        "co_occurrence": "تزامن",   # ظهرا في نفس الآية
        "semantic":      "دلالي",    # علاقة معنوية مُستنتجة
        "causal":        "سببي",     # أ يؤدي إلى ب
        "antonym":       "تضاد",
        "synonym":       "مرادف",
    }

    def __init__(
        self,
        source:        str,
        target:        str,
        weight:        float = 0.1,
        relation_type: str   = "co_occurrence",
        evidence:      Optional[List[str]] = None,
        count:         int = 1,
        first_seen:    Optional[str] = None,
        last_seen:     Optional[str] = None,
    ):
        self.source        = source
        self.target        = target
        self.weight        = round(weight, 4)
        self.relation_type = relation_type
        self.evidence      = evidence or []
        self.count         = count
        self.first_seen    = first_seen or _NOW()
        self.last_seen     = last_seen  or _NOW()

    @property
    def key(self) -> str:
        return f"{self.source}→{self.target}"

    def strengthen(self, evidence_ref: str, delta: float = 0.05) -> None:
        """تقوية العلاقة بعد كل مشاهدة جديدة."""
        self.count     += 1
        self.last_seen  = _NOW()
        self.weight     = round(min(1.0, self.weight + delta * (1 - self.weight)), 4)
        if evidence_ref and evidence_ref not in self.evidence:
            self.evidence.append(evidence_ref)
            self.evidence = self.evidence[-20:]   # max 20 references

    def to_dict(self) -> dict:
        return {
            "source":        self.source,
            "target":        self.target,
            "weight":        self.weight,
            "relation_type": self.relation_type,
            "evidence":      self.evidence,
            "count":         self.count,
            "first_seen":    self.first_seen,
            "last_seen":     self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Relation":
        return cls(
            source        = d["source"],
            target        = d["target"],
            weight        = d.get("weight", 0.1),
            relation_type = d.get("relation_type", "co_occurrence"),
            evidence      = d.get("evidence", []),
            count         = d.get("count", 1),
            first_seen    = d.get("first_seen"),
            last_seen     = d.get("last_seen"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Knowledge Graph
# ═══════════════════════════════════════════════════════════════════════════

class CognitiveKnowledgeGraph:
    """
    الجراف المعرفي المركزي للنظام.

    الخصائص:
      - Thread-safe (RLock)
      - Atomic JSON persistence
      - كل عملية add_concept / add_relation تُحدّث الـ graph فوراً
      - save() صريح أو يمكن جعله تلقائياً بعد batch
    """

    def __init__(self, graph_file: Path = _GRAPH_FILE):
        self._file  = Path(graph_file)
        self._lock  = threading.RLock()

        # البيانات الأساسية
        self._concepts:  Dict[str, Concept]  = {}    # name → Concept
        self._relations: Dict[str, Relation] = {}    # "A→B" → Relation
        self._adj:       Dict[str, Set[str]] = defaultdict(set)   # adjacency list
        self._radj:      Dict[str, Set[str]] = defaultdict(set)   # reverse adjacency

        # إحصائيات سريعة
        self._max_freq: int = 1

        self._file.parent.mkdir(parents=True, exist_ok=True)
        if self._file.exists():
            self.load()
            logger.info(
                f"[CKG] loaded: {len(self._concepts)} concepts, "
                f"{len(self._relations)} relations"
            )
        else:
            logger.info("[CKG] new graph — starting empty")

    # ══════════════════════════════════════════════════════════════════════
    # Public API — Concepts
    # ══════════════════════════════════════════════════════════════════════

    def add_concept(
        self,
        name:    str,
        cluster: str = "غير مصنّف",
        source:  str = "",
    ) -> Concept:
        """
        أضف مفهوماً أو سجّل ظهوراً جديداً إذا كان موجوداً.
        Returns: Concept
        """
        if not name or not name.strip():
            raise ValueError("concept name cannot be empty")
        name = name.strip()

        with self._lock:
            if name in self._concepts:
                c = self._concepts[name]
                c.touch(source)
            else:
                c = Concept(name=name, cluster=cluster, sources=[source] if source else [])
                self._concepts[name]   = c
                self._adj[name]        # ensure key exists
                self._radj[name]       # ensure key exists

            self._max_freq = max(self._max_freq, c.frequency)
            self._recompute_strengths()
            return c

    def get_concept(self, name: str) -> Optional[Concept]:
        with self._lock:
            return self._concepts.get(name)

    def get_concept_strength(self, name: str) -> float:
        """ارجع قوة المفهوم (0–1). صفر إذا غير موجود."""
        with self._lock:
            c = self._concepts.get(name)
            return c.strength if c else 0.0

    def get_strongest_concepts(
        self,
        cluster: Optional[str] = None,
        n: int = 10,
    ) -> List[Dict[str, Any]]:
        """أقوى N مفهوم (اختياري: مُصفًّى بـ cluster)."""
        with self._lock:
            concepts = list(self._concepts.values())
            if cluster:
                concepts = [c for c in concepts if c.cluster == cluster]
            concepts.sort(key=lambda c: c.strength, reverse=True)
            return [c.to_dict() for c in concepts[:n]]

    def all_concepts(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [c.to_dict() for c in self._concepts.values()]

    def concept_count(self) -> int:
        with self._lock:
            return len(self._concepts)

    # ══════════════════════════════════════════════════════════════════════
    # Public API — Relations
    # ══════════════════════════════════════════════════════════════════════

    def add_relation(
        self,
        source:        str,
        target:        str,
        evidence:      str = "",
        relation_type: str = "co_occurrence",
        weight_boost:  float = 0.05,
    ) -> Optional[Relation]:
        """
        أضف علاقة أو قوّي علاقة قائمة.
        إذا لم يكن المفهومان موجودَين، يُنشئهما تلقائياً.
        """
        if not source or not target or source == target:
            return None
        source, target = source.strip(), target.strip()

        with self._lock:
            # تأكد أن المفهومين موجودان
            for name in (source, target):
                if name not in self._concepts:
                    self._concepts[name] = Concept(name=name)
                    self._adj[name]
                    self._radj[name]

            key = f"{source}→{target}"
            if key in self._relations:
                self._relations[key].strengthen(evidence, delta=weight_boost)
            else:
                r = Relation(
                    source        = source,
                    target        = target,
                    weight        = weight_boost,
                    relation_type = relation_type,
                    evidence      = [evidence] if evidence else [],
                )
                self._relations[key] = r
                self._adj[source].add(target)
                self._radj[target].add(source)

            return self._relations[key]

    def get_relation(self, source: str, target: str) -> Optional[Relation]:
        with self._lock:
            return self._relations.get(f"{source}→{target}")

    def get_relation_weight(self, source: str, target: str) -> float:
        with self._lock:
            r = self._relations.get(f"{source}→{target}")
            return r.weight if r else 0.0

    def all_relations(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in self._relations.values()]

    def relation_count(self) -> int:
        with self._lock:
            return len(self._relations)

    # ══════════════════════════════════════════════════════════════════════
    # Public API — Queries
    # ══════════════════════════════════════════════════════════════════════

    def query_related(
        self,
        concept: str,
        top_k:   int = 10,
        direction: str = "both",   # "out" | "in" | "both"
    ) -> List[Tuple[str, float]]:
        """
        أرجع أقوى top_k مفاهيم مرتبطة بـ concept.
        Returns: [(related_name, weight), ...]  مرتبة تنازلياً
        """
        with self._lock:
            if concept not in self._concepts:
                return []

            scores: Dict[str, float] = {}

            if direction in ("out", "both"):
                for target in self._adj.get(concept, set()):
                    r = self._relations.get(f"{concept}→{target}")
                    if r:
                        scores[target] = max(scores.get(target, 0), r.weight)

            if direction in ("in", "both"):
                for source in self._radj.get(concept, set()):
                    r = self._relations.get(f"{source}→{concept}")
                    if r:
                        scores[source] = max(scores.get(source, 0), r.weight)

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]

    def find_path(
        self,
        start: str,
        end:   str,
        max_depth: int = 5,
    ) -> Optional[List[str]]:
        """
        BFS للعثور على أقصر مسار بين مفهومين.
        Returns path or None if unreachable.
        """
        with self._lock:
            if start not in self._concepts or end not in self._concepts:
                return None
            if start == end:
                return [start]

            visited = {start}
            queue: deque = deque([[start]])

            while queue:
                path = queue.popleft()
                if len(path) > max_depth:
                    break
                current = path[-1]
                for neighbor in self._adj.get(current, set()):
                    if neighbor == end:
                        return path + [neighbor]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(path + [neighbor])
            return None

    def cross_source_concepts(self, min_sources: int = 2) -> List[Dict[str, Any]]:
        """
        مفاهيم تظهر في أكثر من مصدر (مفاهيم مركزية عابرة للمصادر).
        مفيد لـ dashboard: "هل مفهوم الصبر من القرآن يرتبط بمصادر أخرى؟"
        """
        with self._lock:
            result = [
                c.to_dict()
                for c in self._concepts.values()
                if len(set(c.sources)) >= min_sources
            ]
            result.sort(key=lambda c: len(c["sources"]), reverse=True)
            return result

    def cluster_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        ملخص لكل cluster: عدد المفاهيم، متوسط القوة، أقوى مفهوم.
        """
        with self._lock:
            summary: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
                "count": 0, "total_strength": 0.0, "top_concept": None, "top_strength": 0.0
            })
            for c in self._concepts.values():
                s = summary[c.cluster]
                s["count"] += 1
                s["total_strength"] += c.strength
                if c.strength > s["top_strength"]:
                    s["top_strength"] = c.strength
                    s["top_concept"]  = c.name
            # حساب المتوسط
            for cl, s in summary.items():
                s["avg_strength"] = round(s["total_strength"] / max(s["count"], 1), 4)
                del s["total_strength"]
            return dict(summary)

    def relation_density(self) -> float:
        """
        نسبة العلاقات الموجودة إلى الممكنة (0–1).
        مقياس تعقيد الجراف.
        """
        with self._lock:
            n = len(self._concepts)
            if n < 2:
                return 0.0
            possible = n * (n - 1)   # directed
            return round(len(self._relations) / possible, 6)

    def concept_growth_rate(self, since_iso: str) -> int:
        """
        عدد المفاهيم التي أُضيفت منذ since_iso.
        لـ dashboard: Concept Growth Rate.
        """
        with self._lock:
            return sum(
                1 for c in self._concepts.values()
                if c.first_seen >= since_iso
            )

    # ══════════════════════════════════════════════════════════════════════
    # Batch Ingestion (يُستدعى بعد كل batch من QuranFeeder)
    # ══════════════════════════════════════════════════════════════════════

    def ingest_from_concept_matches(
        self,
        concept_matches: List[Any],   # List[ConceptMatch] من concept_extractor
        ayah_ref:        str = "",
        auto_relate:     bool = True,
    ) -> int:
        """
        أضف كل المفاهيم المستخرجة من آية واحدة، وأنشئ علاقات co-occurrence بينها.
        Returns: عدد المفاهيم المُضافة/المُحدَّثة.
        """
        semantic_concepts = [m for m in concept_matches if m.cluster != "هيكل"]
        if not semantic_concepts:
            return 0

        with self._lock:
            added = 0
            names_in_ayah = []

            for match in semantic_concepts:
                self.add_concept(
                    name    = match.concept,
                    cluster = match.cluster,
                    source  = ayah_ref,
                )
                names_in_ayah.append(match.concept)
                added += 1

            # علاقات co_occurrence بين كل أزواج المفاهيم في نفس الآية
            if auto_relate and len(names_in_ayah) > 1:
                for i in range(len(names_in_ayah)):
                    for j in range(i + 1, len(names_in_ayah)):
                        a, b = names_in_ayah[i], names_in_ayah[j]
                        self.add_relation(a, b, evidence=ayah_ref, weight_boost=0.05)
                        self.add_relation(b, a, evidence=ayah_ref, weight_boost=0.05)

        return added

    def ingest_batch(
        self,
        all_matches:  List[List[Any]],   # نتيجة extract_batch()
        references:   List[str],
        auto_save:    bool = True,
    ) -> Dict[str, int]:
        """
        أضف نتائج batch كاملة من الـ Concept Extractor.
        Returns: {"concepts_added": N, "relations_added": M}
        """
        before_c = self.concept_count()
        before_r = self.relation_count()

        for matches, ref in zip(all_matches, references):
            self.ingest_from_concept_matches(matches, ayah_ref=ref)

        if auto_save:
            self.save()

        return {
            "concepts_added":  self.concept_count()  - before_c,
            "relations_added": self.relation_count() - before_r,
            "total_concepts":  self.concept_count(),
            "total_relations": self.relation_count(),
        }

    # ══════════════════════════════════════════════════════════════════════
    # Persistence
    # ══════════════════════════════════════════════════════════════════════

    def save(self) -> None:
        """حفظ الجراف atomically إلى JSON."""
        with self._lock:
            data = {
                "_meta": {
                    "schema_version": "1.0.0",
                    "saved_at":       _NOW(),
                    "total_concepts": len(self._concepts),
                    "total_relations": len(self._relations),
                    "description":    "Cognitive Knowledge Graph — Neural Service Mesh",
                },
                "concepts":  {k: v.to_dict() for k, v in self._concepts.items()},
                "relations": {k: v.to_dict() for k, v in self._relations.items()},
            }
        tmp = self._file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self._file)
            logger.info(
                f"[CKG] saved: {len(self._concepts)} concepts, "
                f"{len(self._relations)} relations → {self._file}"
            )
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            logger.error(f"[CKG] save failed: {exc}")
            raise

    def load(self) -> None:
        """تحميل الجراف من JSON."""
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"[CKG] load failed: {exc}")
            return

        with self._lock:
            self._concepts.clear()
            self._relations.clear()
            self._adj.clear()
            self._radj.clear()

            for name, d in data.get("concepts", {}).items():
                self._concepts[name] = Concept.from_dict(d)
                self._adj[name]    # init
                self._radj[name]   # init

            for key, d in data.get("relations", {}).items():
                r = Relation.from_dict(d)
                self._relations[key] = r
                self._adj[r.source].add(r.target)
                self._radj[r.target].add(r.source)

            if self._concepts:
                self._max_freq = max(c.frequency for c in self._concepts.values())
                self._recompute_strengths()

    # ══════════════════════════════════════════════════════════════════════
    # Internal
    # ══════════════════════════════════════════════════════════════════════

    def _recompute_strengths(self) -> None:
        """إعادة حساب strength لكل المفاهيم بعد كل تغيير."""
        for c in self._concepts.values():
            c.compute_strength(self._max_freq)

    # ══════════════════════════════════════════════════════════════════════
    # Stats & Repr
    # ══════════════════════════════════════════════════════════════════════

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            cluster_dist: Dict[str, int] = defaultdict(int)
            for c in self._concepts.values():
                cluster_dist[c.cluster] += 1

            rel_type_dist: Dict[str, int] = defaultdict(int)
            for r in self._relations.values():
                rel_type_dist[r.relation_type] += 1

            top5 = sorted(
                self._concepts.values(), key=lambda c: c.strength, reverse=True
            )[:5]

            return {
                "total_concepts":    len(self._concepts),
                "total_relations":   len(self._relations),
                "relation_density":  self.relation_density(),
                "cluster_distribution": dict(cluster_dist),
                "relation_type_distribution": dict(rel_type_dist),
                "top_concepts": [
                    {"name": c.name, "cluster": c.cluster,
                     "strength": c.strength, "frequency": c.frequency}
                    for c in top5
                ],
                "graph_file": str(self._file),
            }

    def __repr__(self) -> str:
        return (
            f"<CognitiveKnowledgeGraph "
            f"concepts={len(self._concepts)} "
            f"relations={len(self._relations)} "
            f"density={self.relation_density():.4f}>"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════

_ckg_instance: Optional[CognitiveKnowledgeGraph] = None


def get_ckg(graph_file: Path = _GRAPH_FILE) -> CognitiveKnowledgeGraph:
    """ارجع النسخة الوحيدة (singleton) من CKG."""
    global _ckg_instance
    if _ckg_instance is None:
        _ckg_instance = CognitiveKnowledgeGraph(graph_file=graph_file)
    return _ckg_instance


def get_cognitive_graph(graph_file: Path = _GRAPH_FILE) -> CognitiveKnowledgeGraph:
    """Alias for get_ckg() — used by arabic_concept_discovery and other modules."""
    return get_ckg(graph_file)


# ═══════════════════════════════════════════════════════════════════════════
# Integration Helper — يُشغَّل بعد كل batch ingestion من القرآن
# ═══════════════════════════════════════════════════════════════════════════

def build_ckg_from_quran(
    quran_items:    List[Any],                       # List[KnowledgeItem]
    concept_extractor,                               # ConceptExtractor (fitted)
    ckg:            Optional[CognitiveKnowledgeGraph] = None,
    batch_size:     int = 500,
    verbose:        bool = True,
) -> CognitiveKnowledgeGraph:
    """
    بناء CKG كامل من KnowledgeItems القرآنية.

    الخطوات:
      1. استخرج المفاهيم من كل آية عبر concept_extractor
      2. أضف كل مفهوم للـ CKG مع المصدر (reference)
      3. أنشئ علاقات co-occurrence بين المفاهيم في نفس الآية
      4. احفظ

    Example:
        from knowledge_sources.concept_extractor import ConceptExtractor, fit_extractor_on_quran
        from knowledge.cognitive_graph import build_ckg_from_quran

        extractor = fit_extractor_on_quran(all_texts)
        ckg = build_ckg_from_quran(quran_items, extractor)
    """
    if ckg is None:
        ckg = get_ckg()

    total = len(quran_items)
    if verbose:
        print(f"🧠 بناء CKG من {total} آية …")

    processed = 0
    for start in range(0, total, batch_size):
        batch = quran_items[start: start + batch_size]

        texts      = [item.raw_content  for item in batch]
        references = [item.raw_reference for item in batch]
        snames     = []
        for item in batch:
            sname = ""
            for tag in item.derived_tags:
                if tag.startswith("سورة") or len(tag) > 2:
                    sname = tag
                    break
            snames.append(sname)

        # استخراج batch
        all_matches = concept_extractor.extract_batch(
            texts, references=references, surah_names=snames
        )

        # إضافة للـ CKG
        result = ckg.ingest_batch(all_matches, references, auto_save=False)
        processed += len(batch)

        if verbose:
            pct = round(processed / total * 100, 1)
            print(
                f"  ✔ {pct}%  آيات={processed}/{total} "
                f"| مفاهيم={result['total_concepts']} "
                f"| علاقات={result['total_relations']}"
            )

    ckg.save()

    if verbose:
        stats = ckg.stats()
        print(f"\n✅ CKG مكتمل:")
        print(f"   مفاهيم  : {stats['total_concepts']}")
        print(f"   علاقات  : {stats['total_relations']}")
        print(f"   كثافة   : {stats['relation_density']:.4f}")
        print(f"   clusters: {stats['cluster_distribution']}")
        print(f"   أقوى 5  : {[c['name'] for c in stats['top_concepts']]}")

    return ckg
