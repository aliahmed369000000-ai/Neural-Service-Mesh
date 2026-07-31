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
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

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
    quality_score REAL,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_route_log_created ON route_log(created_at);
"""

# سقف الاحتفاظ — يمنع نمو الملف بلا حدود
MAX_ROWS = 5000

# ⚡ أداء: إنشاء الجدول + فحص الترحيل (migration) كانا يُعادان تنفيذهما في
# *كل* اتصال (أي في كل رسالة محادثة، لأن append_entry تُستدعى بعد كل رد).
# هذه العملية idempotent لكنها تفرض استعلامات كتالوج إضافية على القرص في
# المسار الحرج لكل رسالة. الآن تُنفَّذ مرة واحدة فقط لكل عملية Python
# (محمية بنفس _LOCK الذي يحمي كل استدعاءات _connect أدناه، فهي آمنة خيطياً).
_SCHEMA_READY = False

# ⚡ أداء: عملية التقليم (DELETE + ORDER BY + LIMIT) مكلفة نسبياً (فرز على
# القرص) وكانت تُنفَّذ بعد *كل* رسالة. الآن تُنفَّذ كل _TRIM_EVERY إدخال
# فقط، فيبقى الجدول محدوداً عملياً (لن يتجاوز MAX_ROWS + _TRIM_EVERY تقريباً)
# مع خفض تكرار العملية المكلفة بمقدار _TRIM_EVERY مرة.
_TRIM_EVERY = 25
_inserts_since_trim = 0


def _connect() -> sqlite3.Connection:
    global _SCHEMA_READY
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    if not _SCHEMA_READY:
        conn.executescript(_SCHEMA)
        # ترحيل آمن: قواعد منشورة سابقاً قد لا تملك عمود quality_score
        try:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(route_log)")]
            if "quality_score" not in cols:
                conn.execute("ALTER TABLE route_log ADD COLUMN quality_score REAL")
        except Exception as e:
            logger.warning(
                f"route_log_store: فشل ترحيل عمود quality_score "
                f"({type(e).__name__}: {e}) — القراءة/الكتابة لهذا العمود قد "
                "تفشل لاحقاً بصمت إن لم يكن موجوداً فعلياً."
            )
        _SCHEMA_READY = True
    return conn


def append_entry(entry: Dict[str, Any]) -> None:
    """يحفظ قرار توجيه واحد بشكل دائم."""
    global _inserts_since_trim
    with _LOCK:
        try:
            conn = _connect()
            with conn:
                conn.execute(
                    """INSERT INTO route_log
                       (ts, query, category, cat_icon, confidence, node,
                        latency_ms, success, attempt, failover, quality_score)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                        entry.get("quality_score"),
                    ),
                )
                # تقليم السجل كل _TRIM_EVERY إدخال بدل كل رسالة — يمنع نمو
                # الملف بلا حدود مع تفادي كلفة DELETE+ORDER BY على كل رسالة
                _inserts_since_trim += 1
                if _inserts_since_trim >= _TRIM_EVERY:
                    conn.execute(
                        """DELETE FROM route_log WHERE id NOT IN (
                               SELECT id FROM route_log ORDER BY id DESC LIMIT ?
                           )""",
                        (MAX_ROWS,),
                    )
                    _inserts_since_trim = 0
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
                          latency_ms, success, attempt, failover, quality_score
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
            "quality_score": r[10],
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
        except Exception as e:
            logger.warning(
                f"route_log_store.clear_all() فشل ({type(e).__name__}: {e}) — "
                "المستخدم سيظن أن السجل مُسح (زر الواجهة) بينما هو لا يزال موجوداً."
            )


def count_total() -> int:
    with _LOCK:
        try:
            conn = _connect()
            n = conn.execute("SELECT COUNT(*) FROM route_log").fetchone()[0]
            conn.close()
            return int(n)
        except Exception:
            return 0
