"""
ai/agent_audit.py — Agent Interaction Audit Log
================================================================
يسجّل كل تفاعل مع وكلاء "🤖 وكلاء AI" (CategoryAgentChat) لأغراض
الرقابة والتدقيق (Observability):
  - أي فئة وكيل استُدعيت
  - من أين استُدعيت (hub: تبويب الوكيل المباشر، orchestrator: منسّق الوكلاء)
  - هل استُخدم بحث ويب حقيقي قبل الرد
  - مزوّد LLM الذي أجاب فعلياً

نفس نمط ai/core_history.py (SQLite + دوال log/get/summary)، لكن مخصّص
لتفاعلات الوكلاء بدل تطوّر النواة العصبية. لا علاقة له بـ CKG (القرآن) —
هذا سجل تشغيلي بحت، منفصل تماماً عن قاعدة المعرفة.

يُخزَّن في: memory/agent_audit.db (SQLite)
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("memory/agent_audit.db")

# مصادر الاستدعاء المدعومة
SOURCE_HUB          = "hub"           # تبويب "🤖 وكلاء AI" مباشرة
SOURCE_ORCHESTRATOR = "orchestrator"  # تبويب "🤝 منسّق الوكلاء"

_MAX_PREVIEW_CHARS = 300


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _preview(text: str, limit: int = _MAX_PREVIEW_CHARS) -> str:
    """يقتصر النص المخزَّن على معاينة قصيرة — هذا سجل تدقيق تشغيلي
    (متى/من أين/بأي مزوّد)، وليس أرشيفاً كاملاً للمحادثات."""
    text = text or ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


class AgentAuditLog:
    """
    يسجّل ويسترجع تفاعلات وكلاء AI.

    الاستخدام:
        audit = get_default_audit_log()
        audit.log_event(
            category_key="research", category_title="وكيل البحث",
            source="hub", question="...", response="...",
            provider="anthropic", web_used=True,
        )
        recent = audit.get_recent(20)
    """

    def __init__(self, db_path: Path = DB_PATH):
        # مسار مطلق فوراً، لنفس السبب الموثّق في core_history.py: أي
        # os.chdir() لاحق في نفس العملية لا يجب أن يغيّر مكان قاعدة البيانات.
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_audit (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    category_key    TEXT    NOT NULL,
                    category_title  TEXT    NOT NULL,
                    source          TEXT    NOT NULL,
                    question_preview  TEXT,
                    response_preview  TEXT,
                    provider        TEXT,
                    web_used        INTEGER NOT NULL DEFAULT 0,
                    extra           TEXT
                )
            """)
            conn.commit()

    def log_event(
        self,
        category_key: str,
        category_title: str,
        source: str,
        question: str = "",
        response: str = "",
        provider: Optional[str] = None,
        web_used: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """يسجّل تفاعلاً واحداً مع وكيل. لا يرفع استثناء عند فشل الكتابة
        (يُسجَّل تحذير فقط) — التدقيق لا يجب أن يُعطّل المحادثة أبداً."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    INSERT INTO agent_audit
                        (timestamp, category_key, category_title, source,
                         question_preview, response_preview, provider,
                         web_used, extra)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    _now(),
                    category_key,
                    category_title,
                    source,
                    _preview(question),
                    _preview(response),
                    provider,
                    1 if web_used else 0,
                    json.dumps(extra, ensure_ascii=False) if extra else None,
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.warning(f"AgentAuditLog.log_event: فشل تسجيل الحدث: {e}")
            return -1

    def get_recent(self, limit: int = 20) -> List[dict]:
        """يرجع آخر تفاعلات الوكلاء (الأحدث أولاً)."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT * FROM agent_audit
                    ORDER BY id DESC LIMIT ?
                """, (limit,)).fetchall()
        except Exception as e:
            logger.warning(f"AgentAuditLog.get_recent: {e}")
            return []

        result = []
        for row in rows:
            entry = dict(row)
            if entry.get("extra"):
                try:
                    entry["extra"] = json.loads(entry["extra"])
                except Exception:
                    pass
            result.append(entry)
        return result

    def summary(self) -> dict:
        """ملخص إحصائي: إجمالي التفاعلات، حسب الفئة، حسب المصدر،
        نسبة استخدام البحث في الويب."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM agent_audit"
                ).fetchone()[0]
                by_category = conn.execute("""
                    SELECT category_title, COUNT(*) as cnt
                    FROM agent_audit GROUP BY category_title
                    ORDER BY cnt DESC
                """).fetchall()
                by_source = conn.execute("""
                    SELECT source, COUNT(*) as cnt
                    FROM agent_audit GROUP BY source
                """).fetchall()
                web_count = conn.execute(
                    "SELECT COUNT(*) FROM agent_audit WHERE web_used = 1"
                ).fetchone()[0]
        except Exception as e:
            logger.warning(f"AgentAuditLog.summary: {e}")
            return {
                "total_events": 0, "by_category": {}, "by_source": {},
                "web_used_count": 0, "db_path": str(self.db_path),
            }

        return {
            "total_events": total,
            "by_category": {row[0]: row[1] for row in by_category},
            "by_source": {row[0]: row[1] for row in by_source},
            "web_used_count": web_count,
            "db_path": str(self.db_path),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_default_audit_log: Optional[AgentAuditLog] = None


def get_default_audit_log(db_path: Path = DB_PATH) -> AgentAuditLog:
    global _default_audit_log
    if _default_audit_log is None:
        _default_audit_log = AgentAuditLog(db_path)
    return _default_audit_log
