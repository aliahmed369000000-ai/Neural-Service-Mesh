"""سجل تاريخي لدورات تشخيص عُقد الشبكة (penalized / recovered).

يُخزَّن في SQLite بجانب بيانات الـmesh حتى يبقى بعد إعادة التشغيل
ويمكن رسم اتجاه عدد المعاقَبين والمعافين عبر الزمن.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("memory/nodes_diagnose_history.db")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NodesDiagnoseStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DEFAULT_DB).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnose_cycles (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts              TEXT    NOT NULL,
                    scanned         INTEGER NOT NULL DEFAULT 0,
                    errors          INTEGER NOT NULL DEFAULT 0,
                    avg_reputation  REAL,
                    effective_thr   REAL,
                    low_rep_count   INTEGER NOT NULL DEFAULT 0,
                    high_unread_count INTEGER NOT NULL DEFAULT 0,
                    recovered_count INTEGER NOT NULL DEFAULT 0,
                    full_summary    TEXT    NOT NULL,
                    logged_at       TEXT    NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_diagnose_cycles_ts "
                "ON diagnose_cycles(ts)"
            )
            conn.commit()

    def log_cycle(self, summary: Dict[str, Any]) -> int:
        """يسجّل ملخص دورة تشخيص واحدة. لا يرفع استثناء."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cur = conn.execute(
                    """
                    INSERT INTO diagnose_cycles (
                        ts, scanned, errors, avg_reputation, effective_thr,
                        low_rep_count, high_unread_count, recovered_count,
                        full_summary, logged_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(summary.get("ts") or _now()),
                        int(summary.get("scanned") or 0),
                        int(summary.get("errors") or 0),
                        float(summary.get("avg_reputation") or 0.0),
                        float(
                            summary.get("effective_low_rep_threshold")
                            or summary.get("effective_thr")
                            or 0.0
                        ),
                        len(summary.get("low_reputation") or []),
                        len(summary.get("high_unread") or []),
                        len(summary.get("recovered") or []),
                        json.dumps(summary, ensure_ascii=False),
                        _now(),
                    ),
                )
                conn.commit()
                return int(cur.lastrowid or -1)
        except Exception as e:
            logger.warning("NodesDiagnoseStore.log_cycle failed: %s", e)
            return -1

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """آخر الدورات، الأحدث أولاً — للرسوم البيانية والواجهة."""
        limit = max(1, min(int(limit), 500))
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT id, ts, scanned, errors, avg_reputation, effective_thr,
                           low_rep_count, high_unread_count, recovered_count, logged_at
                    FROM diagnose_cycles
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning("NodesDiagnoseStore.get_recent failed: %s", e)
            return []

    def trend(self, limit: int = 30) -> Dict[str, Any]:
        """سلسلة زمنية مبسطة لعدد المعاقَبين والمعافين."""
        rows = list(reversed(self.get_recent(limit=limit)))  # أقدم → أحدث
        return {
            "points": len(rows),
            "ts": [r.get("ts") for r in rows],
            "low_rep_count": [int(r.get("low_rep_count") or 0) for r in rows],
            "recovered_count": [int(r.get("recovered_count") or 0) for r in rows],
            "avg_reputation": [float(r.get("avg_reputation") or 0) for r in rows],
            "effective_thr": [float(r.get("effective_thr") or 0) for r in rows],
        }

    def export_csv(self, limit: int = 200) -> str:
        """تصدير آخر الدورات كـ CSV (نص)."""
        rows = list(reversed(self.get_recent(limit=limit)))
        headers = [
            "id", "ts", "scanned", "errors", "avg_reputation", "effective_thr",
            "low_rep_count", "high_unread_count", "recovered_count", "logged_at",
        ]
        lines = [",".join(headers)]
        for r in rows:
            lines.append(",".join(
                str(r.get(h, "")).replace(",", ";") for h in headers
            ))
        return "\n".join(lines) + "\n"

    def spike_vs_average(self, current_low: int, lookback: int = 20) -> Dict[str, Any]:
        """هل low_rep_count الحالي أعلى بوضوح من المتوسط التاريخي؟"""
        recent = self.get_recent(limit=lookback)
        # استبعد أحدث نقطة إن طابقت current (نحسب على السابق)
        vals = [int(r.get("low_rep_count") or 0) for r in recent]
        if len(vals) < 3:
            return {
                "spike": False,
                "reason": "insufficient_history",
                "current": current_low,
                "avg": None,
            }
        # المتوسط على كل النقاط السابقة في النافذة (بدون فرضية ترتيب معقد)
        avg = sum(vals) / len(vals)
        # عتبة: أعلى من المتوسط بـ +1 على الأقل وبنسبة 50%
        threshold = max(avg * 1.5, avg + 1.0)
        spike = current_low > threshold and current_low > avg
        return {
            "spike": bool(spike),
            "current": int(current_low),
            "avg": round(avg, 3),
            "threshold": round(threshold, 3),
            "lookback": len(vals),
        }

    def summary(self) -> Dict[str, Any]:
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS n,
                           AVG(low_rep_count) AS avg_low,
                           AVG(recovered_count) AS avg_rec,
                           MAX(ts) AS last_ts
                    FROM diagnose_cycles
                    """
                ).fetchone()
            if not row or not row[0]:
                return {"cycles": 0}
            return {
                "cycles": int(row[0]),
                "avg_low_rep": round(float(row[1] or 0), 3),
                "avg_recovered": round(float(row[2] or 0), 3),
                "last_ts": row[3],
            }
        except Exception as e:
            return {"cycles": 0, "error": str(e)}



    def prune_old(self, keep_last: int = 500) -> int:
        """يحذف الدورات الأقدم من keep_last. يُرجع عدد الصفوف المحذوفة."""
        keep_last = max(10, int(keep_last))
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                row = conn.execute("SELECT COUNT(*) FROM diagnose_cycles").fetchone()
                total = int(row[0] or 0) if row else 0
                if total <= keep_last:
                    return 0
                cur = conn.execute(
                    """
                    DELETE FROM diagnose_cycles
                    WHERE id NOT IN (
                        SELECT id FROM diagnose_cycles
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    """,
                    (keep_last,),
                )
                conn.commit()
                deleted = int(cur.rowcount or 0)
                if deleted:
                    logger.info(
                        "NodesDiagnoseStore: pruned %d old cycles (keep=%d)",
                        deleted, keep_last,
                    )
                return deleted
        except Exception as e:
            logger.warning("NodesDiagnoseStore.prune_old failed: %s", e)
            return 0

    def weekly_report(self, days: int = 7) -> Dict[str, Any]:
        """ملخص الفترة الأخيرة (افتراضياً 7 أيام) للتنبيه الدوري."""
        days = max(1, int(days))
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT ts, low_rep_count, recovered_count, avg_reputation,
                           scanned, errors, logged_at
                    FROM diagnose_cycles
                    ORDER BY id DESC
                    LIMIT 500
                    """
                ).fetchall()
            data = list(reversed([dict(r) for r in rows]))
            if not data:
                return {"period_days": days, "cycles": 0}
            # صفِّ حسب logged_at تقريباً — إن فشل نستخدم الكل
            from datetime import datetime, timezone, timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            filtered = []
            for r in data:
                ts = r.get("logged_at") or r.get("ts") or ""
                try:
                    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt >= cutoff:
                        filtered.append(r)
                except Exception:
                    filtered.append(r)
            if not filtered:
                filtered = data[-min(50, len(data)):]
            lows = [int(r.get("low_rep_count") or 0) for r in filtered]
            recs = [int(r.get("recovered_count") or 0) for r in filtered]
            spikes = 0
            for i in range(1, len(lows)):
                if lows[i] > lows[i - 1] * 1.5 and lows[i] > lows[i - 1] + 1:
                    spikes += 1
            return {
                "period_days": days,
                "cycles": len(filtered),
                "avg_low_rep": round(sum(lows) / len(lows), 3) if lows else 0,
                "max_low_rep": max(lows) if lows else 0,
                "min_low_rep": min(lows) if lows else 0,
                "total_recovered": sum(recs),
                "avg_recovered": round(sum(recs) / len(recs), 3) if recs else 0,
                "approx_spike_events": spikes,
                "first_ts": filtered[0].get("ts") if filtered else None,
                "last_ts": filtered[-1].get("ts") if filtered else None,
            }
        except Exception as e:
            return {"period_days": days, "cycles": 0, "error": str(e)}



_STORE: Optional[NodesDiagnoseStore] = None


def get_nodes_diagnose_store(db_path: Optional[Path] = None) -> NodesDiagnoseStore:
    global _STORE
    if db_path is not None:
        return NodesDiagnoseStore(db_path=db_path)
    if _STORE is None:
        _STORE = NodesDiagnoseStore()
    return _STORE
