"""
ai/route_log_store.py
======================
ذاكرة تراكمية لسجل التوجيه (nsm_route_log) — تحفظ القرارات في SQLite
بدل الاكتفاء بـ st.session_state التي تُفقد عند انتهاء الجلسة.

الاستخدام:
    from ai.route_log_store import append_entry, get_recent, clear_all

    append_entry(entry_dict)          # بعد كل قرار توجيه
    rows = get_recent(limit=100)      # لعرض اللوحة الحية عبر كل الجلسات
    clear_all()                       # زر "مسح سجل التوجيه"
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List

DB_PATH = Path("memory/route_log.db")
_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS route_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT,
    query       TEXT,
    category    TEXT,
    cat_icon    TEXT,
    confidence  REAL,
    node        TEXT,
    latency_ms  INTEGER,
    success     INTEGER,
    attempt     INTEGER,
    failover    INTEGER,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_route_log_created ON route_log(created_at);
"""

# سقف الاحتفاظ — يمنع نمو الملف بلا حدود
MAX_ROWS = 5000


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.executescript(_SCHEMA)
    return conn


def append_entry(entry: Dict[str, Any]) -> None:
    """يحفظ قرار توجيه واحد بشكل دائم."""
    with _LOCK:
        try:
            conn = _connect()
            with conn:
                conn.execute(
                    """INSERT INTO route_log
                       (ts, query, category, cat_icon, confidence, node,
                        latency_ms, success, attempt, failover)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry.get("ts", ""),
                        entry.get("query", ""),
                        entry.get("category", "general"),
                        entry.get("cat_icon", "💬"),
                        float(entry.get("confidence", 0.0)),
                        entry.get("node", "?"),
                        int(entry.get("latency_ms", 0)),
                        1 if entry.get("success") else 0,
                        int(entry.get("attempt", 1)),
                        1 if entry.get("failover") else 0,
                    ),
                )
                # تقليم السجل إذا تجاوز السقف
                conn.execute(
                    """DELETE FROM route_log WHERE id NOT IN (
                           SELECT id FROM route_log ORDER BY id DESC LIMIT ?
                       )""",
                    (MAX_ROWS,),
                )
            conn.close()
        except Exception:
            # لا نكسر تدفق الاستجابة الرئيسي بسبب فشل تسجيل
            pass


def get_recent(limit: int = 100) -> List[Dict[str, Any]]:
    """يعيد آخر N قرار توجيه (الأقدم أولاً) بنفس شكل قاموس session_state القديم."""
    with _LOCK:
        try:
            conn = _connect()
            cur = conn.execute(
                """SELECT ts, query, category, cat_icon, confidence, node,
                          latency_ms, success, attempt, failover
                   FROM route_log ORDER BY id DESC LIMIT ?""",
                (limit,),
            )
            rows = cur.fetchall()
            conn.close()
        except Exception:
            return []

    rows.reverse()  # الأقدم أولاً لمطابقة سلوك القائمة السابقة
    result = []
    for r in rows:
        result.append({
            "ts": r[0],
            "query": r[1],
            "category": r[2],
            "cat_icon": r[3],
            "confidence": r[4],
            "node": r[5],
            "latency_ms": r[6],
            "success": bool(r[7]),
            "attempt": r[8],
            "failover": bool(r[9]),
        })
    return result


def clear_all() -> None:
    """يمسح كل السجل التراكمي (زر مسح سجل التوجيه)."""
    with _LOCK:
        try:
            conn = _connect()
            with conn:
                conn.execute("DELETE FROM route_log")
            conn.close()
        except Exception:
            pass


def count_total() -> int:
    with _LOCK:
        try:
            conn = _connect()
            n = conn.execute("SELECT COUNT(*) FROM route_log").fetchone()[0]
            conn.close()
            return int(n)
        except Exception:
            return 0
