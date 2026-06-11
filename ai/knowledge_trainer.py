"""
Knowledge Trainer — محرك التدريب المعرفي المدمج
================================================
يحوّل كل معلومة من أي مصدر إلى:
  1. متجه 7 أبعاد ثابتة → يُغذَّى مباشرة في DynamicWeightLayer
  2. مفاهيم + علاقات → cognitive_graph.json (ملف واحد فقط)
  3. سجل التدريب → data/mesh.db (SQLite واحد فقط)

الأبعاد السبعة الثابتة:
  [0] IMPORTANCE  : أهمية المعلومة (0-1)
  [1] CERTAINTY   : درجة اليقين - حقيقة=1.0، نظرية=0.7، فرضية=0.3
  [2] ABSTRACTION : تجريد المفهوم - ملموس=0، مجرد=1
  [3] DOMAIN      : رمز المجال المعياري (physics=0.14, math=0.28, ...)
  [4] CONNECTIVITY: كثافة العلاقات مع مفاهيم أخرى (0-1)
  [5] TEMPORALITY : الزمنية - قديم=0، حديث=1
  [6] NOVELTY     : جِدَّة المعلومة للنظام - موجودة=0، جديدة=1
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("KnowledgeTrainer")

# ── ثوابت الأبعاد ──────────────────────────────────────────────────────────
VECTOR_DIM = 7  # ثابت لا يتغير أبداً

DOMAIN_CODES: Dict[str, float] = {
    "physics":        0.14,
    "math":           0.28,
    "history":        0.42,
    "biology":        0.56,
    "civilizations":  0.70,
    "github":         0.84,
    "wikipedia":      0.98,
    "general":        0.50,
}

_DB_PATH  = Path("./data/mesh.db")
_CKG_PATH = Path("./knowledge/cognitive_graph.json")
_LOCK     = threading.Lock()
_NOW      = lambda: datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════════
# 1. مشفّر المتجهات — يحوّل أي حقيقة إلى 7 أبعاد
# ═══════════════════════════════════════════════════════════════════════════

class VectorEncoder:
    """يُشفِّر مفهوماً أو حقيقة إلى متجه 7 أبعاد ثابت."""

    @staticmethod
    def encode(
        text: str,
        domain: str,
        importance: float = 0.5,
        certainty: float = 0.8,
        abstraction: float = 0.5,
        year: Optional[int] = None,
        known_concepts: Optional[set] = None,
        related_count: int = 0,
    ) -> np.ndarray:
        vec = np.zeros(VECTOR_DIM, dtype=np.float64)

        # [0] IMPORTANCE
        vec[0] = float(np.clip(importance, 0.0, 1.0))

        # [1] CERTAINTY
        vec[1] = float(np.clip(certainty, 0.0, 1.0))

        # [2] ABSTRACTION — نستنتجه من طول النص ومحتواه
        words = text.split()
        has_numbers = any(ch.isdigit() for ch in text)
        abst = abstraction
        if has_numbers:
            abst = max(0.0, abst - 0.2)  # أرقام = أكثر ملموسية
        if len(words) > 20:
            abst = min(1.0, abst + 0.1)  # نص طويل = أكثر تجريداً
        vec[2] = float(np.clip(abst, 0.0, 1.0))

        # [3] DOMAIN
        vec[3] = DOMAIN_CODES.get(domain, DOMAIN_CODES["general"])

        # [4] CONNECTIVITY
        vec[4] = float(np.clip(related_count / 20.0, 0.0, 1.0))

        # [5] TEMPORALITY
        if year is not None:
            vec[5] = float(np.clip((year - 1000) / (2026 - 1000), 0.0, 1.0))
        else:
            vec[5] = 0.5

        # [6] NOVELTY
        if known_concepts is not None:
            concept_key = text[:40].strip().lower()
            vec[6] = 0.0 if concept_key in known_concepts else 1.0
        else:
            vec[6] = 0.8

        return vec


# ═══════════════════════════════════════════════════════════════════════════
# 2. مدير قاعدة البيانات — سجل تدريب في SQLite واحد
# ═══════════════════════════════════════════════════════════════════════════

class TrainingDB:
    """يُضيف جدول training_log إلى data/mesh.db الموجود."""

    def __init__(self, db_path: Path = _DB_PATH):
        self.db_path = db_path
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge_training (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain      TEXT    NOT NULL,
                    concept     TEXT    NOT NULL,
                    text        TEXT    NOT NULL,
                    vector_json TEXT    NOT NULL,
                    loss        REAL,
                    train_step  INTEGER,
                    trained_at  TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_kt_domain
                    ON knowledge_training(domain);
                CREATE INDEX IF NOT EXISTS idx_kt_concept
                    ON knowledge_training(concept);

                CREATE TABLE IF NOT EXISTS training_sessions (
                    session_id  TEXT PRIMARY KEY,
                    domain      TEXT NOT NULL,
                    total_items INTEGER DEFAULT 0,
                    total_steps INTEGER DEFAULT 0,
                    avg_loss    REAL    DEFAULT 0.0,
                    started_at  TEXT    NOT NULL,
                    finished_at TEXT
                );
            """)

    def log_item(
        self,
        domain: str,
        concept: str,
        text: str,
        vector: np.ndarray,
        loss: float,
        train_step: int,
    ):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO knowledge_training
                   (domain, concept, text, vector_json, loss, train_step, trained_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (domain, concept, text[:500],
                 json.dumps(vector.tolist()), loss, train_step, _NOW()),
            )

    def start_session(self, session_id: str, domain: str):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO training_sessions
                   (session_id, domain, started_at)
                   VALUES (?,?,?)""",
                (session_id, domain, _NOW()),
            )

    def finish_session(
        self, session_id: str, total_items: int,
        total_steps: int, avg_loss: float,
    ):
        with self._conn() as conn:
            conn.execute(
                """UPDATE training_sessions
                   SET total_items=?, total_steps=?, avg_loss=?, finished_at=?
                   WHERE session_id=?""",
                (total_items, total_steps, avg_loss, _NOW(), session_id),
            )

    def stats(self) -> Dict[str, Any]:
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as c FROM knowledge_training").fetchone()["c"]
            by_domain = conn.execute(
                "SELECT domain, COUNT(*) as c FROM knowledge_training GROUP BY domain"
            ).fetchall()
            sessions = conn.execute(
                "SELECT COUNT(*) as c FROM training_sessions WHERE finished_at IS NOT NULL"
            ).fetchone()["c"]
            last = conn.execute(
                "SELECT AVG(loss) as l FROM knowledge_training ORDER BY id DESC LIMIT 100"
            ).fetchone()["l"]
        return {
            "total_items_trained": total,
            "completed_sessions": sessions,
            "by_domain": {r["domain"]: r["c"] for r in by_domain},
            "recent_avg_loss": round(last or 0.0, 6),
        }

    def known_concepts(self) -> set:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT concept FROM knowledge_training").fetchall()
        return {r["concept"].lower() for r in rows}


# ═══════════════════════════════════════════════════════════════════════════
# 3. مدير الـ CKG — يُضيف المفاهيم والعلاقات إلى cognitive_graph.json
# ═══════════════════════════════════════════════════════════════════════════

class CKGManager:
    """يُضيف المعرفة مباشرة إلى cognitive_graph.json الموجود."""

    def __init__(self, path: Path = _CKG_PATH):
        self.path = path
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    raw = f.read()
                # فحص Git LFS pointer
                if raw.startswith("version https://git-lfs.github.com"):
                    logger.warning(
                        f"CKGManager: {self.path} is a Git LFS pointer — "
                        "initialising empty CKG and writing valid JSON."
                    )
                    empty = {
                        "_meta": {
                            "schema_version": "1.0.0",
                            "saved_at": _NOW(),
                            "total_concepts": 0,
                            "total_relations": 0,
                            "description": "Cognitive Knowledge Graph — Neural Service Mesh",
                        },
                        "concepts": {},
                        "relations": {},
                    }
                    tmp = self.path.with_suffix(".tmp")
                    with open(tmp, "w", encoding="utf-8") as fw:
                        json.dump(empty, fw, ensure_ascii=False, indent=2)
                    tmp.replace(self.path)
                    return empty
                return json.loads(raw)
            except Exception:
                pass
        return {
            "_meta": {
                "schema_version": "1.0.0",
                "saved_at": _NOW(),
                "total_concepts": 0,
                "total_relations": 0,
                "description": "Cognitive Knowledge Graph — Neural Service Mesh",
            },
            "concepts": {},
            "relations": {},
        }

    def _save(self):
        meta = self._data.setdefault("_meta", {})
        meta["saved_at"]       = _NOW()
        meta["total_concepts"] = len(self._data.get("concepts", {}))
        meta["total_relations"]= len(self._data.get("relations", {}))
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    def add_concept(
        self,
        name: str,
        cluster: str,
        source: str,
        vector: Optional[np.ndarray] = None,
    ):
        concepts = self._data.setdefault("concepts", {})
        if name not in concepts:
            entry: Dict[str, Any] = {
                "name":       name,
                "cluster":    cluster,
                "sources":    [source],
                "frequency":  1,
                "strength":   0.1,
                "first_seen": _NOW(),
                "last_seen":  _NOW(),
            }
            if vector is not None:
                entry["vector"] = vector.tolist()
            concepts[name] = entry
        else:
            c = concepts[name]
            if source not in c.get("sources", []):
                c.setdefault("sources", []).append(source)
            c["frequency"] = c.get("frequency", 0) + 1
            c["strength"]  = round(
                min(1.0, 0.1 * math.log1p(c["frequency"])), 4)
            c["last_seen"] = _NOW()
            if vector is not None:
                c["vector"] = vector.tolist()

    def add_relation(
        self,
        src: str,
        tgt: str,
        relation_type: str,
        evidence: str,
        weight: float = 0.5,
    ):
        if src == tgt:
            return
        key = f"{src}|||{tgt}"
        relations = self._data.setdefault("relations", {})
        if key not in relations:
            relations[key] = {
                "source":        src,
                "target":        tgt,
                "weight":        round(weight, 4),
                "relation_type": relation_type,
                "evidence":      [evidence],
                "count":         1,
            }
        else:
            r = relations[key]
            r["count"]  = r.get("count", 0) + 1
            r["weight"] = round(min(1.0, r["weight"] + 0.02), 4)
            if evidence not in r.get("evidence", []):
                r["evidence"].append(evidence)

    def ingest_batch(
        self,
        items: List[Dict[str, Any]],
        save_every: int = 100,
    ):
        """
        أدخل دفعة من المفاهيم دفعة واحدة مع حفظ دوري.
        كل عنصر: {name, cluster, source, vector, relations:[{target,type,weight}]}
        """
        for i, item in enumerate(items):
            name = item["name"]
            self.add_concept(
                name,
                item.get("cluster", "general"),
                item.get("source", "unknown"),
                item.get("vector"),
            )
            for rel in item.get("relations", []):
                self.add_relation(
                    name,
                    rel["target"],
                    rel.get("type", "related"),
                    item.get("source", ""),
                    rel.get("weight", 0.5),
                )
            if (i + 1) % save_every == 0:
                self._save()
        self._save()

    def concept_count(self) -> int:
        return len(self._data.get("concepts", {}))

    def known_names(self) -> set:
        return set(self._data.get("concepts", {}).keys())


# ═══════════════════════════════════════════════════════════════════════════
# 4. محرك التدريب الرئيسي
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeTrainer:
    """
    المحرك المدمج الوحيد للتدريب المعرفي.

    الاستخدام:
        trainer = KnowledgeTrainer(mesh)
        result  = trainer.train_domain("physics", items)
    """

    def __init__(self, mesh=None):
        self.mesh   = mesh
        self.db     = TrainingDB()
        self.ckg    = CKGManager()
        self.encoder= VectorEncoder()
        self._layer = self._get_layer()
        logger.info(
            f"KnowledgeTrainer ready — "
            f"layer={self._layer.__class__.__name__ if self._layer else 'None'}  "
            f"ckg_concepts={self.ckg.concept_count()}"
        )

    def _get_layer(self):
        """يحصل على أفضل طبقة أوزان متاحة (Deep > Dynamic > Neural)."""
        if self.mesh is None:
            return None
        # أولاً: DeepRoutingNetwork — الأولوية القصوى
        layer = getattr(self.mesh, "deep_network", None)
        if layer is not None:
            return layer
        # ثانياً: DynamicWeightLayer
        layer = getattr(self.mesh, "dynamic_layer", None)
        if layer is not None:
            return layer
        # ثالثاً: NeuralWeightLayer كـ fallback
        layer = getattr(self.mesh, "neural_layer", None)
        return layer

    def train_domain(
        self,
        domain: str,
        items: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        batch_size: int = 50,
    ) -> Dict[str, Any]:
        """
        دِرِّب النظام على مجال معرفي كامل.

        كل عنصر في items:
          {
            "concept":     str,           # اسم المفهوم
            "text":        str,           # وصف/نص
            "cluster":     str,           # التصنيف
            "importance":  float 0-1,
            "certainty":   float 0-1,
            "abstraction": float 0-1,
            "year":        int (optional),
            "relations":   [{target, type, weight}],
          }
        """
        sid = session_id or f"{domain}_{int(time.time())}"
        self.db.start_session(sid, domain)
        known = self.db.known_concepts()
        known_names = self.ckg.known_names()

        total_loss = 0.0
        total_steps = 0
        ckg_batch: List[Dict[str, Any]] = []

        t0 = time.time()
        for i, item in enumerate(items):
            concept = item.get("concept", item.get("text", "")[:40])
            text    = item.get("text", concept)
            source  = f"{domain}:{concept[:20]}"

            # — تشفير المتجه
            vec = self.encoder.encode(
                text       = text,
                domain     = domain,
                importance = item.get("importance", 0.5),
                certainty  = item.get("certainty",  0.8),
                abstraction= item.get("abstraction",0.5),
                year       = item.get("year"),
                known_concepts = known,
                related_count  = len(item.get("relations", [])),
            )

            # — تحديث مصفوفة الأوزان مباشرة
            loss = 0.0
            if self._layer is not None:
                try:
                    target = float(np.mean(vec))
                    loss   = self._layer.train_step(vec.tolist(), target)
                    total_steps += 1
                    total_loss  += loss
                except Exception as e:
                    logger.warning(f"train_step failed for '{concept}': {e}")

            # — تسجيل في SQLite
            self.db.log_item(domain, concept, text, vec, loss, total_steps)
            known.add(concept.lower())

            # — تجميع لـ CKG
            ckg_batch.append({
                "name":      concept,
                "cluster":   item.get("cluster", domain),
                "source":    source,
                "vector":    vec,
                "relations": item.get("relations", []),
            })

            # — إدخال دفعي كل batch_size عنصر
            if len(ckg_batch) >= batch_size:
                self.ckg.ingest_batch(ckg_batch, save_every=batch_size)
                ckg_batch.clear()

        if ckg_batch:
            self.ckg.ingest_batch(ckg_batch, save_every=batch_size)

        avg_loss = total_loss / max(1, total_steps)
        self.db.finish_session(sid, len(items), total_steps, avg_loss)

        elapsed = round(time.time() - t0, 2)
        layer_shape = (
            list(self._layer.weights.shape)
            if self._layer is not None else "N/A"
        )

        logger.info(
            f"[KnowledgeTrainer] domain='{domain}'  "
            f"items={len(items)}  steps={total_steps}  "
            f"avg_loss={avg_loss:.6f}  elapsed={elapsed}s  "
            f"matrix={layer_shape}"
        )

        return {
            "session_id":    sid,
            "domain":        domain,
            "items_trained": len(items),
            "train_steps":   total_steps,
            "avg_loss":      round(avg_loss, 6),
            "elapsed_s":     elapsed,
            "matrix_shape":  layer_shape,
            "ckg_total":     self.ckg.concept_count(),
        }

    def train_all(
        self,
        domain_items: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """دِرِّب على جميع المجالات دفعة واحدة."""
        results = {}
        for domain, items in domain_items.items():
            results[domain] = self.train_domain(domain, items)
        return {
            "total_domains": len(results),
            "total_items":   sum(r["items_trained"] for r in results.values()),
            "total_steps":   sum(r["train_steps"]   for r in results.values()),
            "overall_avg_loss": round(
                sum(r["avg_loss"] for r in results.values()) / max(1, len(results)), 6
            ),
            "by_domain": results,
            "db_stats":  self.db.stats(),
        }

    def stats(self) -> Dict[str, Any]:
        layer_info = {}
        if self._layer is not None:
            layer_info = {
                "shape":       list(self._layer.weights.shape),
                "train_steps": getattr(self._layer, "_train_steps", 0),
                "last_loss":   getattr(self._layer, "_last_loss", None),
            }
        return {
            "layer":        layer_info,
            "ckg_concepts": self.ckg.concept_count(),
            "db":           self.db.stats(),
        }
