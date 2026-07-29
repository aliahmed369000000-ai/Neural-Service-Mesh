"""
Translation History — سجل تاريخ الترجمات لتبويب 🌐 ترجمة
===========================================================================
كانت كل ترجمة تُفقد بمجرد تحديث الصفحة أو تبديل التبويب (تُخزَّن في
st.session_state.tr_result فقط، بلا أي قاعدة بيانات) — نفس النمط الذي
كان في سيناريوهات 🎤 وثائقي/🎬 Shorts قبل ربطها بـ shorts_history.

وحدة معزولة تماماً (ملف جديد + قاعدة بيانات SQLite مستقلة خاصة بها في
memory/translations.db) — لا تلمس أي جدول أو ملف موجود، فلا خطر تعارض.

الاستخدام:
    from ai.translation_history import get_history
    hist = get_history()
    hist.save(src_lang="auto", tgt_lang="الإنجليزية",
              source_text="...", translated_text="...", provider="groq")
    recent = hist.list_recent(limit=20)
    hist.delete(record_id)
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = "memory/translations.db"
_history_singleton: Optional["TranslationHistory"] = None


class TranslationHistory:
    """يحفظ ويسترجع سجل الترجمات — لا يرفع استثناءً أبداً من save()/
    list_recent()/delete() (best-effort، فشل الحفظ لا يجوز أن يُفشل عرض
    الترجمة نفسها للمستخدم)."""

    def __init__(self, db_path: str = _DB_PATH) -> None:
        self._db_path = db_path
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        try:
            with self._conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS translations (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        src_lang        TEXT NOT NULL,
                        tgt_lang        TEXT NOT NULL,
                        source_text     TEXT NOT NULL,
                        translated_text TEXT NOT NULL,
                        provider        TEXT DEFAULT '',
                        created_at      REAL NOT NULL
                    )
                """)
                conn.commit()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"translation_history: فشل تهيئة القاعدة: {e}")

    def save(
        self, src_lang: str, tgt_lang: str, source_text: str,
        translated_text: str, provider: str = "",
    ) -> Optional[int]:
        if not (source_text or "").strip() or not (translated_text or "").strip():
            return None
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    "INSERT INTO translations "
                    "(src_lang, tgt_lang, source_text, translated_text, provider, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (src_lang, tgt_lang, source_text[:2000], translated_text[:2000],
                     provider, time.time()),
                )
                conn.commit()
                return cur.lastrowid
        except Exception as e:  # noqa: BLE001
            logger.debug(f"translation_history.save: {e}")
            return None

    def list_recent(self, limit: int = 20) -> List[sqlite3.Row]:
        try:
            with self._conn() as conn:
                return conn.execute(
                    "SELECT * FROM translations ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"translation_history.list_recent: {e}")
            return []

    def delete(self, record_id: int) -> bool:
        try:
            with self._conn() as conn:
                conn.execute("DELETE FROM translations WHERE id = ?", (record_id,))
                conn.commit()
                return True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"translation_history.delete: {e}")
            return False


def get_history() -> TranslationHistory:
    """singleton واحد لعملية Streamlit كاملة."""
    global _history_singleton
    if _history_singleton is None:
        _history_singleton = TranslationHistory()
    return _history_singleton
