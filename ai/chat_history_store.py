"""
ai/chat_history_store.py
=========================
تخزين دائم (SQLite) لسجل محادثات تبويب 💬 المحادثة — يحل مشكلة أن
st.session_state.nsm_messages يُفقَد بالكامل عند إغلاق التبويب/انتهاء
الجلسة، فلا طريقة للرجوع لمحادثة سابقة أو معرفة "من ردّ أولاً" بعد
انتهاء الجلسة الحيّة.

قاعدة البيانات: memory/chat_history.db — نفس مسار ونمط
memory/accounts.db في ai/accounts.py (SQLite، WAL mode، دالة _db()
تنشئ الجدول لو غير موجود).

⚠️ نفس القيد المعماري الموجود أصلاً في accounts.py: قرص Streamlit
Community Cloud غير دائم عبر إعادة نشر (redeploy) — البيانات تبقى
طالما التطبيق لم يُعَد نشره/تشغيله من الصفر. هذا يكفي تماماً لهدف هذه
الوحدة (استرجاع محادثة ضمن نفس فترة تشغيل التطبيق، حتى بعد انتهاء
الجلسة أو إعادة تحميل الصفحة).

المعرّف: session_id عشوائي (uuid4) يُولَّد مرة واحدة فقط لكل جلسة
متصفح ويُخزَّن في st.session_state.nsm_chat_session_id — لا يوجد نظام
دخول مربوط بالمحادثة حالياً (ai/accounts.py منفصل تماماً وغير مفعَّل
بهذا التبويب)، فهذا أفضل معرّف متاح بلا تغيير جذري لواجهة
عدم-تسجيل-الدخول الحالية.

تدهور آمن كامل: أي فشل بالكتابة/القراءة (قرص ممتلئ، قفل قاعدة بيانات،
...) يُبتلَع صامتاً مع تحذير بالسجلّات فقط — هذا التخزين اختياري تماماً
ولا يجوز أن يكسر تجربة المحادثة الحيّة نفسها (نفس مبدأ
_record_chat_episode في app_core.py).

الاستخدام النموذجي:
    from ai.chat_history_store import save_message, get_first_message

    save_message(session_id, "user", "ما حكم الصبر؟")
    save_message(session_id, "nsm", "...", source_badge="⚡ كاش متعلَّم")

    first = get_first_message(session_id)   # من ردّ/سأل أولاً في الجلسة
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("NSMChatHistoryStore")

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "memory" / "chat_history.db"


def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT NOT NULL,
            role         TEXT NOT NULL,
            content      TEXT NOT NULL,
            source_badge TEXT,
            created_at   TEXT NOT NULL
        )""")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id, id)"
    )
    conn.commit()
    return conn


def save_message(session_id: str, role: str, content: str, source_badge: str = "") -> None:
    """يخزّن رسالة واحدة. لا يرفع استثناءً أبداً — أي فشل يُبتلَع صامتاً
    (بتحذير بالسجلّات فقط) عشان لا يكسر تجربة المحادثة الحيّة."""
    if not session_id or not (content or "").strip():
        return
    try:
        now = datetime.now(timezone.utc).isoformat()
        with _db() as c:
            c.execute(
                "INSERT INTO chat_messages (session_id, role, content, source_badge, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, source_badge, now),
            )
            c.commit()
    except Exception as e:
        logger.warning(f"[chat_history_store] فشل حفظ رسالة: {e}")


def get_session_messages(session_id: str, limit: int = 500) -> List[Dict]:
    """يعيد كل رسائل جلسة معيّنة مرتّبة زمنياً (الأقدم أولاً). قائمة
    فارغة عند أي فشل أو عدم وجود رسائل — بلا استثناء أبداً."""
    if not session_id:
        return []
    try:
        with _db() as c:
            rows = c.execute(
                "SELECT role, content, source_badge, created_at FROM chat_messages "
                "WHERE session_id = ? ORDER BY id ASC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [
            {"role": r[0], "content": r[1], "source_badge": r[2], "created_at": r[3]}
            for r in rows
        ]
    except Exception as e:
        logger.warning(f"[chat_history_store] فشل قراءة رسائل الجلسة: {e}")
        return []


def get_first_message(session_id: str) -> Optional[Dict]:
    """يعيد أول رسالة فعلياً في جلسة معيّنة (من ردّ/سأل أولاً) — None لو
    لا توجد أي رسالة مسجَّلة لهذه الجلسة أو عند أي فشل."""
    if not session_id:
        return None
    try:
        with _db() as c:
            row = c.execute(
                "SELECT role, content, source_badge, created_at FROM chat_messages "
                "WHERE session_id = ? ORDER BY id ASC LIMIT 1",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return {"role": row[0], "content": row[1], "source_badge": row[2], "created_at": row[3]}
    except Exception as e:
        logger.warning(f"[chat_history_store] فشل قراءة أول رسالة: {e}")
        return None


def list_sessions(limit: int = 50) -> List[Dict]:
    """يعيد قائمة الجلسات الأخيرة (session_id، عدد الرسائل، أول/آخر
    توقيت) — مفيد لعرض سجل محادثات سابقة لاحقاً بأي واجهة إدارية.
    قائمة فارغة عند أي فشل، بلا استثناء أبداً."""
    try:
        with _db() as c:
            rows = c.execute(
                """SELECT session_id, COUNT(*), MIN(created_at), MAX(created_at)
                   FROM chat_messages GROUP BY session_id
                   ORDER BY MAX(created_at) DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {"session_id": r[0], "message_count": r[1], "started_at": r[2], "last_at": r[3]}
            for r in rows
        ]
    except Exception as e:
        logger.warning(f"[chat_history_store] فشل قراءة قائمة الجلسات: {e}")
        return []
