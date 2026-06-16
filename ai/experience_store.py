"""
Experience Learning — الجزء 1: تخزين الحلقات (Episodes)
==========================================================
يضيف جدول `neural_episodes` مستقل في SQLite (لا يلمس قاعدة
episodic_memory.py الحالية ولا جدول qa_episodes).

كل تفاعل عبر ReasoningPipeline.answer() يُحوَّل إلى Episode يحتوي:
  - question, matched_concepts, related_concepts
  - decision_weights (W_SEMANTIC..W_TOPOLOGY)
  - confidence
  - answer
  - timestamp
  - context_vector (للـ replay اللاحق)
  - target_used (الهدف الذي تدرّب عليه NeuralCore وقتها)
  - quality scores (concept_coverage, relation_coverage,
    memory_recall_quality, answer_confidence, overall_quality)
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_NOW = lambda: datetime.now(timezone.utc).isoformat()

DEFAULT_DB_PATH = Path("memory/experience.db")


# ════════════════════════════════════════════════════════════════════════
# Episode dataclass
# ════════════════════════════════════════════════════════════════════════

@dataclass
class Episode:
    """حلقة تجربة واحدة — تطابق Requirement #2."""

    question: str
    matched_concepts: List[Dict[str, Any]]
    related_concepts: List[Dict[str, Any]]
    decision_weights: Dict[str, float]
    confidence: float
    answer: str
    timestamp: str = field(default_factory=_NOW)

    # بيانات إضافية لازمة لإعادة التدريب (replay)
    context_vector: List[float] = field(default_factory=list)
    target_used: Optional[List[float]] = None
    train_loss: Optional[float] = None
    memory_hits: List[Dict[str, Any]] = field(default_factory=list)

    # نتائج جودة التجربة (Requirement #4/#5) — تُحسب وتُملأ بعد البناء
    quality: Dict[str, float] = field(default_factory=dict)

    episode_id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:16]}")

    # تغذية رجعية خارجية من المستخدم
    external_feedback: Optional[Dict[str, Any]] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "id": self.episode_id,
            "question": self.question,
            "matched_concepts": json.dumps(self.matched_concepts, ensure_ascii=False),
            "related_concepts": json.dumps(self.related_concepts, ensure_ascii=False),
            "decision_weights": json.dumps(self.decision_weights, ensure_ascii=False),
            "confidence": float(self.confidence),
            "answer": self.answer,
            "timestamp": self.timestamp,
            "context_vector": json.dumps(self.context_vector),
            "target_used": json.dumps(self.target_used) if self.target_used is not None else None,
            "train_loss": self.train_loss,
            "memory_hits": json.dumps(self.memory_hits, ensure_ascii=False),
            "quality": json.dumps(self.quality, ensure_ascii=False),
            "external_feedback": json.dumps(self.external_feedback) if self.external_feedback is not None else None,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Episode":
        return cls(
            episode_id=row["id"],
            question=row["question"],
            matched_concepts=json.loads(row["matched_concepts"]),
            related_concepts=json.loads(row["related_concepts"]),
            decision_weights=json.loads(row["decision_weights"]),
            confidence=row["confidence"],
            answer=row["answer"],
            timestamp=row["timestamp"],
            context_vector=json.loads(row["context_vector"]),
            target_used=json.loads(row["target_used"]) if row["target_used"] else None,
            train_loss=row["train_loss"],
            memory_hits=json.loads(row["memory_hits"]),
            quality=json.loads(row["quality"]) if row["quality"] else {},
            external_feedback=json.loads(row["external_feedback"]) if row["external_feedback"] else None,
        )


# ════════════════════════════════════════════════════════════════════════
# EpisodeStore — تخزين/استرجاع SQLite
# ════════════════════════════════════════════════════════════════════════

class EpisodeStore:
    """تخزين دائم للحلقات (جدول neural_episodes مستقل)."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        # مسار مطلق فوراً (انظر نفس الملاحظة في ai/core_history.py) — يحمي
        # من سلوك مختلف إن تغيّر CWD بعد إنشاء الكائن.
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS neural_episodes (
                id                TEXT PRIMARY KEY,
                question          TEXT NOT NULL,
                matched_concepts  TEXT NOT NULL,
                related_concepts  TEXT NOT NULL,
                decision_weights  TEXT NOT NULL,
                confidence        REAL NOT NULL,
                answer            TEXT NOT NULL,
                timestamp         TEXT NOT NULL,
                context_vector    TEXT NOT NULL,
                target_used       TEXT,
                train_loss        REAL,
                memory_hits       TEXT NOT NULL,
                quality           TEXT NOT NULL,
                replayed_count    INTEGER NOT NULL DEFAULT 0,
                last_replayed_at  TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_neural_episodes_ts ON neural_episodes(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_neural_episodes_quality "
                      "ON neural_episodes(json_extract(quality, '$.overall_quality'))")
        conn.commit()
        # هجرة قواعد البيانات القديمة — أضف العمود إن لم يكن موجوداً
        try:
            conn.execute("ALTER TABLE neural_episodes ADD COLUMN external_feedback TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # العمود موجود مسبقاً
        conn.close()

    # ── إضافة ────────────────────────────────────────────────────────

    def add(self, episode: Episode) -> str:
        row = episode.to_row()
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT INTO neural_episodes
               (id, question, matched_concepts, related_concepts, decision_weights,
                confidence, answer, timestamp, context_vector, target_used,
                train_loss, memory_hits, quality)
               VALUES (:id, :question, :matched_concepts, :related_concepts,
                       :decision_weights, :confidence, :answer, :timestamp,
                       :context_vector, :target_used, :train_loss, :memory_hits, :quality)""",
            row,
        )
        conn.commit()
        conn.close()
        return episode.episode_id

    # ── استرجاع ──────────────────────────────────────────────────────

    def count(self) -> int:
        conn = sqlite3.connect(str(self.db_path))
        n = conn.execute("SELECT COUNT(*) FROM neural_episodes").fetchone()[0]
        conn.close()
        return int(n)

    def get_recent(self, limit: int = 20) -> List[Episode]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM neural_episodes ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [Episode.from_row(r) for r in rows]

    def get_top_by_quality(self, limit: int = 20) -> List[Episode]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM neural_episodes
               ORDER BY json_extract(quality, '$.overall_quality') DESC
               LIMIT ?""", (limit,)
        ).fetchall()
        conn.close()
        return [Episode.from_row(r) for r in rows]

    def get_diverse_sample(self, limit: int = 20, seed: Optional[int] = None) -> List[Episode]:
        """
        عينة متنوعة: تجمع الحلقات حسب أول مفهوم مطابق (proxy لـ cluster)
        وتأخذ حداً أقصى من كل مجموعة لتجنّب التحيّز نحو موضوع واحد.
        """
        import random
        rng = random.Random(seed)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM neural_episodes").fetchall()
        conn.close()
        episodes = [Episode.from_row(r) for r in rows]
        if not episodes:
            return []

        groups: Dict[str, List[Episode]] = {}
        for ep in episodes:
            key = "_none_"
            if ep.matched_concepts:
                key = ep.matched_concepts[0].get("cluster", "_none_")
            groups.setdefault(key, []).append(ep)

        group_keys = list(groups.keys())
        rng.shuffle(group_keys)

        result: List[Episode] = []
        idx = 0
        while len(result) < limit and any(groups[k] for k in group_keys):
            key = group_keys[idx % len(group_keys)]
            if groups[key]:
                pick = rng.randrange(len(groups[key]))
                result.append(groups[key].pop(pick))
            idx += 1
            if idx > limit * len(group_keys) + 10:
                break
        return result[:limit]

    def mark_replayed(self, episode_ids: List[str]) -> None:
        if not episode_ids:
            return
        conn = sqlite3.connect(str(self.db_path))
        now = _NOW()
        conn.executemany(
            "UPDATE neural_episodes SET replayed_count = replayed_count + 1, "
            "last_replayed_at = ? WHERE id = ?",
            [(now, eid) for eid in episode_ids],
        )
        conn.commit()
        conn.close()

    def update_feedback(
        self,
        episode_id: str,
        rating: Optional[str],
        correction_text: Optional[str] = None,
    ) -> bool:
        """
        يحدّث حقل external_feedback لحلقة موجودة.
        يرجع True إن وُجد الـ episode_id، False إن لم يوجد.
        """
        from datetime import datetime, timezone
        feedback = {
            "rating": rating,
            "correction_text": correction_text,
            "feedback_at": datetime.now(timezone.utc).isoformat(),
        }
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(
            "UPDATE neural_episodes SET external_feedback = ? WHERE id = ?",
            (json.dumps(feedback), episode_id),
        )
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0

    def get_with_feedback(self, limit: int = 50) -> List[Episode]:
        """يرجع الحلقات التي لديها external_feedback مرتبة بالأحدث أولاً."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM neural_episodes
               WHERE external_feedback IS NOT NULL
               ORDER BY timestamp DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        return [Episode.from_row(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        conn = sqlite3.connect(str(self.db_path))
        total = conn.execute("SELECT COUNT(*) FROM neural_episodes").fetchone()[0]
        avg_q = conn.execute(
            "SELECT AVG(json_extract(quality, '$.overall_quality')) FROM neural_episodes"
        ).fetchone()[0]
        avg_conf = conn.execute("SELECT AVG(confidence) FROM neural_episodes").fetchone()[0]
        total_replays = conn.execute("SELECT SUM(replayed_count) FROM neural_episodes").fetchone()[0]
        conn.close()
        return {
            "total_episodes": int(total),
            "avg_overall_quality": round(avg_q, 6) if avg_q is not None else None,
            "avg_confidence": round(avg_conf, 6) if avg_conf is not None else None,
            "total_replays": int(total_replays) if total_replays else 0,
        }
