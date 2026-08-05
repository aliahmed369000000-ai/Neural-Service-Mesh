"""
Yemeni Knowledge Graph (YKG) — هيكل معرفي منفصل تماماً عن CKG
================================================================
⚠️ هذا الملف **لا يمسّ ولا يستورد من** knowledge/cognitive_graph.py.
الـ CKG يبقى حصرياً لبيانات القرآن كما هو مقرَّر في تصميم NSM — أي
معرفة يمنية (لهجة/ثقافة/مفردات) تُبنى هنا في جراف مستقل موازٍ له،
بنفس فكرة العقد/العلاقات لكن بتخزين وحقول خاصة به.

البنية:
  ┌─────────────────────────────────────────────────────────┐
  │  YemeniConcept (عقدة)                                   │
  │    name           : str   الكلمة/المفهوم اللهجي         │
  │    dialect_region : str   "صنعاني"|"تعزي"|"عدني"|        │
  │                            "حضرمي"|"تهامي"|"عام"          │
  │    cluster        : str   تصنيف عام (مفردات/عادات/أمثال) │
  │    msa_equivalent : str   المقابل بالفصحى (اختياري)      │
  │    gloss_en        : str  شرح إنجليزي مختصر (اختياري)    │
  │    sources        : List[str]                            │
  │    frequency       : int                                 │
  │    strength        : float                                │
  ├─────────────────────────────────────────────────────────┤
  │  YemeniRelation (حافة موزونة)                            │
  │    source/target/weight/relation_type/evidence/count      │
  └─────────────────────────────────────────────────────────┘

الواجهة:
  ykg.add_concept(name, dialect_region, cluster, source, msa_equivalent, gloss_en)
  ykg.add_relation(source, target, evidence, relation_type)
  ykg.query_related(concept, top_k)
  ykg.get_strongest_concepts(dialect_region=None, cluster=None, n=10)
  ykg.stats()
  ykg.save() / ykg.load()

التخزين:
  knowledge/yemeni_graph.json   (ملف منفصل تماماً عن cognitive_graph.json)
"""
from __future__ import annotations

import json
import logging
import math
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_GRAPH_FILE = Path("./knowledge/yemeni_graph.json")
_NOW = lambda: datetime.now(timezone.utc).isoformat()

VALID_DIALECT_REGIONS = {"صنعاني", "تعزي", "عدني", "حضرمي", "تهامي", "عام"}


# ═══════════════════════════════════════════════════════════════════════════
# Concept Node
# ═══════════════════════════════════════════════════════════════════════════
class YemeniConcept:
    """عقدة في الـ YKG تمثل كلمة/مفهوماً لهجياً أو ثقافياً يمنياً واحداً."""

    __slots__ = (
        "name", "dialect_region", "cluster", "msa_equivalent", "gloss_en",
        "sources", "frequency", "strength", "first_seen", "last_seen",
    )

    def __init__(
        self,
        name: str,
        dialect_region: str = "عام",
        cluster: str = "غير مصنّف",
        msa_equivalent: str = "",
        gloss_en: str = "",
        sources: Optional[List[str]] = None,
        frequency: int = 1,
        strength: float = 0.0,
        first_seen: Optional[str] = None,
        last_seen: Optional[str] = None,
    ):
        self.name = name
        self.dialect_region = dialect_region if dialect_region in VALID_DIALECT_REGIONS else "عام"
        self.cluster = cluster
        self.msa_equivalent = msa_equivalent
        self.gloss_en = gloss_en
        self.sources = sources or []
        self.frequency = frequency
        self.strength = strength
        self.first_seen = first_seen or _NOW()
        self.last_seen = last_seen or _NOW()

    def touch(self, source: str, msa_equivalent: str = "", gloss_en: str = "") -> None:
        self.frequency += 1
        self.last_seen = _NOW()
        if source and source not in self.sources:
            self.sources.append(source)
        # نحدّث المقابل الفصيح/الشرح فقط لو كان فارغاً سابقاً — لا نكتب فوق
        # قيمة موجودة بصمت بمصدر جديد قد يكون أقل دقة.
        if msa_equivalent and not self.msa_equivalent:
            self.msa_equivalent = msa_equivalent
        if gloss_en and not self.gloss_en:
            self.gloss_en = gloss_en

    def compute_strength(self, max_freq: int) -> None:
        if max_freq <= 0:
            self.strength = 0.0
            return
        self.strength = round(math.log(self.frequency + 1) / math.log(max_freq + 1), 4)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dialect_region": self.dialect_region,
            "cluster": self.cluster,
            "msa_equivalent": self.msa_equivalent,
            "gloss_en": self.gloss_en,
            "sources": self.sources,
            "frequency": self.frequency,
            "strength": self.strength,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "YemeniConcept":
        return cls(
            name=d["name"],
            dialect_region=d.get("dialect_region", "عام"),
            cluster=d.get("cluster", "غير مصنّف"),
            msa_equivalent=d.get("msa_equivalent", ""),
            gloss_en=d.get("gloss_en", ""),
            sources=d.get("sources", []),
            frequency=d.get("frequency", 1),
            strength=d.get("strength", 0.0),
            first_seen=d.get("first_seen"),
            last_seen=d.get("last_seen"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Relation Edge
# ═══════════════════════════════════════════════════════════════════════════
class YemeniRelation:
    """حافة موزونة بين مفهومين يمنيين (مثلاً: كلمة لهجية ↔ مقابلها الفصيح)."""

    __slots__ = ("source", "target", "weight", "relation_type", "evidence", "count")

    def __init__(
        self,
        source: str,
        target: str,
        weight: float = 0.1,
        relation_type: str = "co_occurrence",
        evidence: Optional[List[str]] = None,
        count: int = 1,
    ):
        self.source = source
        self.target = target
        self.weight = weight
        self.relation_type = relation_type
        self.evidence = evidence or []
        self.count = count

    def reinforce(self, evidence: str = "", weight_boost: float = 0.05) -> None:
        self.count += 1
        self.weight = min(1.0, round(self.weight + weight_boost, 4))
        if evidence and evidence not in self.evidence:
            self.evidence.append(evidence)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "relation_type": self.relation_type,
            "evidence": self.evidence,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "YemeniRelation":
        return cls(
            source=d["source"],
            target=d["target"],
            weight=d.get("weight", 0.1),
            relation_type=d.get("relation_type", "co_occurrence"),
            evidence=d.get("evidence", []),
            count=d.get("count", 1),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Yemeni Knowledge Graph
# ═══════════════════════════════════════════════════════════════════════════
class YemeniKnowledgeGraph:
    """
    الجراف المعرفي اليمني — منفصل تماماً عن CognitiveKnowledgeGraph (CKG).

    الخصائص:
      - Thread-safe (RLock)
      - Atomic JSON persistence
      - لا استيراد ولا اعتماد على knowledge/cognitive_graph.py إطلاقاً
    """

    def __init__(self, graph_file: Path = _GRAPH_FILE):
        self._file = Path(graph_file)
        self._lock = threading.RLock()

        self._concepts: Dict[str, YemeniConcept] = {}
        self._relations: Dict[str, YemeniRelation] = {}
        self._adj: Dict[str, Set[str]] = defaultdict(set)
        self._radj: Dict[str, Set[str]] = defaultdict(set)
        self._max_freq: int = 1

        self._file.parent.mkdir(parents=True, exist_ok=True)
        if self._file.exists():
            self.load()
            logger.info(
                f"[YKG] loaded: {len(self._concepts)} concepts, "
                f"{len(self._relations)} relations"
            )
        else:
            logger.info("[YKG] new graph — starting empty")

    # ─────────────────────────────────────────────────────────────────
    # Concepts
    # ─────────────────────────────────────────────────────────────────
    def add_concept(
        self,
        name: str,
        dialect_region: str = "عام",
        cluster: str = "غير مصنّف",
        source: str = "",
        msa_equivalent: str = "",
        gloss_en: str = "",
    ) -> YemeniConcept:
        if not name or not name.strip():
            raise ValueError("concept name cannot be empty")
        name = name.strip()

        with self._lock:
            if name in self._concepts:
                c = self._concepts[name]
                c.touch(source, msa_equivalent, gloss_en)
            else:
                c = YemeniConcept(
                    name=name,
                    dialect_region=dialect_region,
                    cluster=cluster,
                    msa_equivalent=msa_equivalent,
                    gloss_en=gloss_en,
                    sources=[source] if source else [],
                )
                self._concepts[name] = c
                self._adj[name]
                self._radj[name]

            self._max_freq = max(self._max_freq, c.frequency)
            self._recompute_strengths()
            return c

    def get_concept(self, name: str) -> Optional[YemeniConcept]:
        with self._lock:
            return self._concepts.get(name)

    def concept_count(self) -> int:
        with self._lock:
            return len(self._concepts)

    def all_concepts(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [c.to_dict() for c in self._concepts.values()]

    def get_strongest_concepts(
        self,
        dialect_region: Optional[str] = None,
        cluster: Optional[str] = None,
        n: int = 10,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            concepts = list(self._concepts.values())
            if dialect_region:
                concepts = [c for c in concepts if c.dialect_region == dialect_region]
            if cluster:
                concepts = [c for c in concepts if c.cluster == cluster]
            concepts.sort(key=lambda c: c.strength, reverse=True)
            return [c.to_dict() for c in concepts[:n]]

    def _recompute_strengths(self) -> None:
        for c in self._concepts.values():
            c.compute_strength(self._max_freq)

    # ─────────────────────────────────────────────────────────────────
    # Relations
    # ─────────────────────────────────────────────────────────────────
    def add_relation(
        self,
        source: str,
        target: str,
        evidence: str = "",
        relation_type: str = "co_occurrence",
        weight_boost: float = 0.05,
    ) -> Optional[YemeniRelation]:
        if not source or not target or source == target:
            return None
        source, target = source.strip(), target.strip()

        with self._lock:
            if source not in self._concepts:
                self.add_concept(source)
            if target not in self._concepts:
                self.add_concept(target)

            key = f"{source}→{target}"
            if key in self._relations:
                r = self._relations[key]
                r.reinforce(evidence, weight_boost)
            else:
                r = YemeniRelation(
                    source=source, target=target,
                    weight=weight_boost, relation_type=relation_type,
                    evidence=[evidence] if evidence else [],
                )
                self._relations[key] = r
                self._adj[source].add(target)
                self._radj[target].add(source)
            return r

    def query_related(self, concept: str, top_k: int = 5) -> List[Tuple[str, float]]:
        with self._lock:
            neighbors = self._adj.get(concept, set()) | self._radj.get(concept, set())
            scored = []
            for n in neighbors:
                key_fwd = f"{concept}→{n}"
                key_bwd = f"{n}→{concept}"
                r = self._relations.get(key_fwd) or self._relations.get(key_bwd)
                if r:
                    scored.append((n, r.weight))
            scored.sort(key=lambda t: t[1], reverse=True)
            return scored[:top_k]

    def relation_count(self) -> int:
        with self._lock:
            return len(self._relations)

    # ─────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            by_region: Dict[str, int] = defaultdict(int)
            for c in self._concepts.values():
                by_region[c.dialect_region] += 1
            return {
                "total_concepts": len(self._concepts),
                "total_relations": len(self._relations),
                "by_dialect_region": dict(by_region),
            }

    # ─────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────
    def save(self) -> None:
        with self._lock:
            data = {
                "_meta": {
                    "schema_version": "1.0.0",
                    "saved_at": _NOW(),
                    "total_concepts": len(self._concepts),
                    "total_relations": len(self._relations),
                    "description": "Yemeni Knowledge Graph (YKG) — منفصل تماماً عن CKG (حصري للقرآن)",
                },
                "concepts": {k: v.to_dict() for k, v in self._concepts.items()},
                "relations": {k: v.to_dict() for k, v in self._relations.items()},
            }
        tmp = self._file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self._file)
            logger.info(
                f"[YKG] saved: {len(self._concepts)} concepts, "
                f"{len(self._relations)} relations → {self._file}"
            )
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            logger.error(f"[YKG] save failed: {exc}")
            raise

    def load(self) -> None:
        try:
            raw_text = self._file.read_text(encoding="utf-8")
            if raw_text.startswith("version https://git-lfs.github.com"):
                logger.warning(f"[YKG] {self._file} is a Git LFS pointer — starting empty.")
                return
            data = json.loads(raw_text)
        except Exception as exc:
            logger.error(f"[YKG] load failed: {exc}")
            return

        with self._lock:
            self._concepts.clear()
            self._relations.clear()
            self._adj.clear()
            self._radj.clear()

            for name, d in data.get("concepts", {}).items():
                self._concepts[name] = YemeniConcept.from_dict(d)
                self._adj[name]
                self._radj[name]

            for key, d in data.get("relations", {}).items():
                r = YemeniRelation.from_dict(d)
                self._relations[key] = r
                self._adj[r.source].add(r.target)
                self._radj[r.target].add(r.source)

            self._max_freq = max((c.frequency for c in self._concepts.values()), default=1)
            self._recompute_strengths()


# ═══════════════════════════════════════════════════════════════════════════
# Singleton accessor (نفس نمط get_ckg في cognitive_graph.py)
# ═══════════════════════════════════════════════════════════════════════════
_ykg_instance: Optional[YemeniKnowledgeGraph] = None
_ykg_lock = threading.Lock()


def get_ykg(graph_file: Path = _GRAPH_FILE) -> YemeniKnowledgeGraph:
    global _ykg_instance
    with _ykg_lock:
        if _ykg_instance is None:
            _ykg_instance = YemeniKnowledgeGraph(graph_file)
        return _ykg_instance
