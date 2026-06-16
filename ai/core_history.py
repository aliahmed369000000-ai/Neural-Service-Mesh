"""
ai/core_history.py — Self-History / Identity Tracking

يسجّل تاريخ تطوّر NeuralCore عبر الزمن:
  - كل دورة تدريب (training cycle)
  - كل نمو (growth/evolve)
  - كل fork (structural variation)
  - كل rollback

يُخزَّن في: memory/core_history.db (SQLite)
"""

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("memory/core_history.db")

# أنواع الأحداث المدعومة
EVENT_TRAINING_CYCLE = "training_cycle"
EVENT_GROWTH         = "growth"
EVENT_FORK           = "fork"
EVENT_ROLLBACK       = "rollback"
EVENT_PROMOTION      = "promotion"
EVENT_CONSOLIDATION  = "consolidation"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_state_hash(layer_dims: list, params_count: int, train_steps: int) -> str:
    """
    يحسب hash مختصر لتمييز حالة النواة.
    يعتمد على: البنية + عدد الأوزان + عدد خطوات التدريب.
    """
    raw = f"{layer_dims}|{params_count}|{train_steps}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class CoreHistory:
    """
    يسجّل ويسترجع تاريخ تطوّر NeuralCore.

    الاستخدام:
        history = CoreHistory()
        history.log_event(core, event_type="training_cycle", extra={...})
        lineage = history.get_lineage()
    """

    def __init__(self, db_path: Path = DB_PATH):
        # تحويل إلى مسار مطلق فوراً — لو ظل نسبياً، أي os.chdir() لاحق في
        # نفس العملية يجعل sqlite3.connect() يفتح/ينشئ ملفاً مختلفاً تماماً
        # بصمت (بدون الجدول)، لأن sqlite3 يُعيد حلّ المسارات النسبية عند كل
        # connect() لا عند إنشاء الكائن.
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """ينشئ جدول core_history إن لم يكن موجوداً."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS core_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    event_type      TEXT    NOT NULL,
                    core_name       TEXT    NOT NULL,
                    architecture    TEXT    NOT NULL,
                    params_count    INTEGER NOT NULL,
                    train_steps     INTEGER NOT NULL,
                    benchmark_score REAL,
                    state_hash      TEXT    NOT NULL,
                    parent_hash     TEXT,
                    extra           TEXT
                )
            """)
            conn.commit()

    def log_event(
        self,
        core,
        event_type: str,
        benchmark_score: Optional[float] = None,
        parent_hash: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        يسجّل حدثاً في تاريخ NeuralCore.

        Parameters
        ----------
        core : NeuralCore
        event_type : str
            أحد: training_cycle, growth, fork, rollback, promotion, consolidation
        benchmark_score : float أو None
            درجة الأداء (MSE) على benchmark إن توفّرت
        parent_hash : str أو None
            hash الحالة الأب (قبل fork أو rollback)
        extra : dict أو None
            بيانات إضافية (مثل rollback details, promotion details, إلخ)

        Returns
        -------
        int: معرّف السجل (rowid)
        """
        # استخرج معلومات البنية من core
        try:
            layer_dims = list(core.net.layer_dims)
            architecture = json.dumps(layer_dims, ensure_ascii=False)
            # حساب عدد الأوزان: مجموع (out×in + out) لكل طبقة
            params_count = sum(
                layer.W.size + layer.b.size
                for layer in core.net.layers
            )
            train_steps = int(core.net._train_steps)
        except Exception as e:
            logger.warning(f"CoreHistory.log_event: فشل استخراج بنية النواة: {e}")
            layer_dims = []
            architecture = "[]"
            params_count = 0
            train_steps = 0

        state_hash = _compute_state_hash(layer_dims, params_count, train_steps)

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                INSERT INTO core_history
                    (timestamp, event_type, core_name, architecture,
                     params_count, train_steps, benchmark_score,
                     state_hash, parent_hash, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                _now(),
                event_type,
                str(core.name),
                architecture,
                params_count,
                train_steps,
                round(float(benchmark_score), 8) if benchmark_score is not None else None,
                state_hash,
                parent_hash,
                json.dumps(extra, ensure_ascii=False) if extra else None,
            ))
            conn.commit()
            rowid = cursor.lastrowid

        logger.debug(
            f"CoreHistory: [{event_type}] '{core.name}' "
            f"arch={layer_dims} params={params_count} "
            f"hash={state_hash}"
        )
        return rowid

    def get_lineage(self, limit: int = 100) -> List[dict]:
        """
        يرجع تاريخ تطوّر NeuralCore كقائمة زمنية (الأحدث أولاً).

        Returns
        -------
        List[dict]: كل عنصر يحتوي:
            id, timestamp, event_type, core_name, architecture,
            params_count, train_steps, benchmark_score,
            state_hash, parent_hash, extra
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM core_history
                ORDER BY id DESC
                LIMIT ?
            """, (limit,)).fetchall()

        result = []
        for row in rows:
            entry = dict(row)
            # فك تشفير architecture وextra
            try:
                entry["architecture"] = json.loads(entry["architecture"])
            except Exception:
                pass
            if entry.get("extra"):
                try:
                    entry["extra"] = json.loads(entry["extra"])
                except Exception:
                    pass
            result.append(entry)

        return result

    def get_last_state_hash(self) -> Optional[str]:
        """يرجع hash آخر حالة مسجَّلة (للاستخدام كـ parent_hash)."""
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute("""
                SELECT state_hash FROM core_history
                ORDER BY id DESC LIMIT 1
            """).fetchone()
        return row[0] if row else None

    def summary(self) -> dict:
        """ملخص إحصائي لتاريخ النواة."""
        with sqlite3.connect(str(self.db_path)) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM core_history"
            ).fetchone()[0]
            by_type = conn.execute("""
                SELECT event_type, COUNT(*) as cnt
                FROM core_history
                GROUP BY event_type
            """).fetchall()
            last = conn.execute("""
                SELECT timestamp, event_type, architecture, benchmark_score
                FROM core_history ORDER BY id DESC LIMIT 1
            """).fetchone()

        return {
            "total_events": total,
            "by_type": {row[0]: row[1] for row in by_type},
            "last_event": {
                "timestamp": last[0],
                "event_type": last[1],
                "architecture": last[2],
                "benchmark_score": last[3],
            } if last else None,
            "db_path": str(self.db_path),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_default_history: Optional[CoreHistory] = None


def get_default_history(db_path: Path = DB_PATH) -> CoreHistory:
    global _default_history
    if _default_history is None:
        _default_history = CoreHistory(db_path)
    return _default_history
