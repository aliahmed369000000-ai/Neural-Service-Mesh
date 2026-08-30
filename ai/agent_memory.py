"""
ai/agent_memory.py
==================
🆕 نظام الذاكرة طويلة المدى للوكلاء (Long-Term Memory).

يُمكّن الوكلاء من:
  • تذكر السياق عبر الجلسات (persistent memory)
  • تخزين الخبرات والدروس المستفادة
  • استرجاع معلومات ذات صلة بالمهمة الحالية
  • نسيان المعلومات القديمة (TTL)

التخزين: SQLite في artifacts/agent_memory/

الاستخدام:
    from ai.agent_memory import AgentMemory
    memory = AgentMemory(agent_id="research_agent")
    memory.store("تمكّن Groq من أداء أفضل من OpenRouter لـ Arabic")
    relevant = memory.search("Groq أداء عربي")
"""
from __future__ import annotations
import json
import logging
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger("nsm.agent_memory")

ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = ROOT / "artifacts" / "agent_memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
_db_lock = threading.Lock()


class AgentMemory:
    """ذاكرة طويلة المدى للوكلاء — تخزين واسترجاع السياق."""

    def __init__(self, agent_id: str = "default", db_path: Optional[Path] = None):
        self.agent_id = agent_id
        self._db_path = db_path or (MEMORY_DIR / f"{agent_id}.db")
        self._init_db()

    def _init_db(self):
        with _db_lock:
            conn = sqlite3.connect(str(self._db_path), timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    importance INTEGER DEFAULT 5,
                    created_at TEXT NOT NULL,
                    accessed_at TEXT,
                    access_count INTEGER DEFAULT 0,
                    ttl_seconds INTEGER DEFAULT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_agent ON memories(agent_id);
                CREATE INDEX IF NOT EXISTS idx_category ON memories(category);
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                    USING fts5(content, content=memories, content_rowid=id);
            """)
            conn.commit()
            conn.close()

    def store(
        self,
        content: str,
        category: str = "general",
        importance: int = 5,
        ttl_seconds: Optional[int] = None,
    ) -> int:
        """تخزين معلومة في الذاكرة بعد تنقيح الأسرار والتحقق من الحدود."""
        content = self._sanitize(str(content or "")).strip()
        if not content:
            raise ValueError("لا يمكن تخزين ذاكرة فارغة")
        importance = max(1, min(10, int(importance)))
        if ttl_seconds is not None:
            ttl_seconds = max(60, int(ttl_seconds))
        now = datetime.now(timezone.utc).isoformat()
        with _db_lock:
            conn = sqlite3.connect(str(self._db_path), timeout=10)
            cursor = conn.execute(
                """INSERT INTO memories (agent_id, content, category, importance,
                   created_at, ttl_seconds) VALUES (?, ?, ?, ?, ?, ?)""",
                (self.agent_id, content, category, importance, now, ttl_seconds),
            )
            mem_id = cursor.lastrowid
            # FTS index
            conn.execute(
                "INSERT INTO memories_fts(rowid, content) VALUES (?, ?)",
                (mem_id, content),
            )
            conn.commit()
            conn.close()
        return mem_id

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """البحث في الذاكرة (full-text search + relevance ranking)."""
        results = []
        try:
            with _db_lock:
                conn = sqlite3.connect(str(self._db_path), timeout=10)
                conn.execute("DELETE FROM memories WHERE ttl_seconds IS NOT NULL AND created_at < datetime('now', '-' || ttl_seconds || ' seconds')")
                # FTS search
                rows = conn.execute(
                    """SELECT m.id, m.content, m.category, m.importance,
                              m.access_count, m.created_at,
                              rank FROM memories m
                       JOIN memories_fts f ON m.id = f.rowid
                       WHERE memories_fts MATCH ? AND m.agent_id = ?
                       ORDER BY rank LIMIT ?""",
                    (query, self.agent_id, limit),
                ).fetchall()
                conn.close()

            for row in rows:
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "category": row[2],
                    "importance": row[3],
                    "access_count": row[4],
                    "created_at": row[5],
                    "relevance_score": abs(row[6]) if row[6] else 0,
                })
                # تحديث accessed_at
                self._mark_accessed(row[0])

        except Exception as e:
            logger.warning(f"Memory search error: {e}")

        return results

    def get_recent(self, limit: int = 20, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """جلب أحدث الذكريات."""
        query = "SELECT id, content, category, importance, created_at FROM memories WHERE agent_id = ?"
        params: List[Any] = [self.agent_id]
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with _db_lock:
            conn = sqlite3.connect(str(self._db_path), timeout=10)
            rows = conn.execute(query, params).fetchall()
            conn.close()

        return [{
            "id": r[0], "content": r[1], "category": r[2],
            "importance": r[3], "created_at": r[4],
        } for r in rows]

    def get_context_for_task(self, task_description: str, limit: int = 5) -> str:
        """جلب سياق ذاكري ذو صلة بالمهمة الحالية (لتمريره إلى LLM)."""
        memories = self.search(task_description, limit=limit)
        if not memories:
            memories = self.get_recent(limit=limit)

        if not memories:
            return "لا توجد ذكريات سابقة."

        lines = ["📚 سياق من الذاكرة طويلة المدى:"]
        for m in memories:
            lines.append(f"  [{m['category']}] {m['content']}")
        return "\n".join(lines)

    def forget_old(self, max_age_days: int = 30) -> int:
        """حذف الذكريات القديمة."""
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_days * 86400)
        cutoff_str = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
        with _db_lock:
            conn = sqlite3.connect(str(self._db_path), timeout=10)
            cursor = conn.execute(
                "DELETE FROM memories WHERE created_at < ? AND agent_id = ? AND importance < 5",
                (cutoff_str, self.agent_id),
            )
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
        return deleted

    def clear(self) -> int:
        """مسح كل الذكريات."""
        with _db_lock:
            conn = sqlite3.connect(str(self._db_path), timeout=10)
            conn.execute("DELETE FROM memories WHERE agent_id = ?", (self.agent_id,))
            conn.execute("DELETE FROM memories_fts WHERE rowid NOT IN (SELECT id FROM memories)")
            deleted = conn.execute("SELECT changes()").fetchone()[0]
            conn.commit()
            conn.close()
        return deleted

    def stats(self) -> Dict[str, Any]:
        """إحصائيات الذاكرة."""
        with _db_lock:
            conn = sqlite3.connect(str(self._db_path), timeout=10)
            total = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE agent_id = ?", (self.agent_id,)
            ).fetchone()[0]
            categories = conn.execute(
                "SELECT category, COUNT(*) FROM memories WHERE agent_id = ? GROUP BY category",
                (self.agent_id,),
            ).fetchall()
            conn.close()
        return {
            "total_memories": total,
            "categories": {c[0]: c[1] for c in categories},
            "db_path": str(self._db_path),
        }

    @staticmethod
    def _sanitize(content: str) -> str:
        """إخفاء مفاتيح شائعة قبل حفظها في الذاكرة الدائمة."""
        patterns = (
            r"(?i)(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*[^\s,;]+",
            r"gh[pousr]_[A-Za-z0-9_]+",
            r"sk-[A-Za-z0-9_-]+",
        )
        for pattern in patterns:
            content = re.sub(pattern, lambda m: m.group(0).split(':', 1)[0].split('=', 1)[0] + ": [REDACTED]", content)
        return content

    def _mark_accessed(self, mem_id: int):
        """تحديث accessed_at و access_count."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with _db_lock:
                conn = sqlite3.connect(str(self._db_path), timeout=10)
                conn.execute(
                    "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
                    (now, mem_id),
                )
                conn.commit()
                conn.close()
        except Exception:
            pass
