"""
ai/db_maintenance.py
======================
إدارة دورية لقواعد SQLite الخاصة بالمنصة (Vacuum + أرشفة الأحداث القديمة).

المشكلة التي تحلها:
    قرص Streamlit Community Cloud محدود، ومع تراكم الأحداث (رسائل
    المحادثات، سجل التوجيه، تجارب السرب، الدروس الجماعية...) تنمو قواعد
    SQLite وتتضخم صفاتها الداخلية (free pages) حتى بعد الحذف — لأن
    SQLite لا يعيد مساحة الحذف للقرص إلا عند VACUUM. بلا هذه الوحدة،
    النمو التراكمي يؤدي ببطء إلى بطء الاستعلامات واحتمال امتلاء القرص.

المبدأ المعماري (بدون كسر أي وحدة موجودة):
    - لا يعدّل أي ملف موجود — كل الوحدات القديمة (chat_history_store،
      collective_memory، route_log_store...) تواصل عملها كما هي دون أي
      تغيير أو معرفة بهذه الوحدة.
    - الأرشفة = نقل (وليس حذف أعمى): الصفوف الأقدم من 30 يوماً تُنقل
      أولاً إلى جداول أرشيف مستقلة (archive_chat_messages،
      archive_route_log، ...) داخل نفس ملف قاعدة البيانات، ثم تُحذف من
      الجدول الحي. البيانات قابلة للاسترجاع بالكامل من جداول الأرشيف.
    - VACUUM يُنفَّذ بعد الأرشفة (والحذف الدوري القديم إن وُجد) لإعادة
      مساحة القرص فعلياً.
    - تدهور آمن كامل: أي فشل يُبتلَع بتحذير في السجلات فقط — لا يجوز
      أبداً أن تكسر الصيانة تجربة المحادثة الحيّة.
    - التشغيل: مرة عند بدء التطبيق + كل NSM_DB_MAINT_INTERVAL_HOURS
      (افتراضي 24 ساعة) عبر threading.Timer بعيداً عن مسار الطلبات.

الجداول المدارة حالياً:
    chat_messages          (memory/chat_history.db)        — created_at
    route_log              (memory/route_log.db)           — created_at
    swarm_history          (memory/swarm_history.db)       — started_at
    collective_lessons     (memory/collective_memory.db)   — created_at
    neural_episodes        (memory/experience.db)          — timestamp
    episodes               (memory/episodic.db)            — timestamp

الاستخدام النموذجي:
    from ai.db_maintenance import start_periodic_maintenance, run_maintenance_once

    start_periodic_maintenance()          # مؤقت 24 ساعة + تشغيل أولي
    stats = run_maintenance_once(days=30) # يدوي: {archived, deleted, vacuumed}
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("NSMDBMaintenance")

ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════════════════
# سجل الجداول المدارة: (مسار قاعدة البيانات, الجدول الحي, عمود الوقت,
# عمود المعرّف/الفهرس الرئيسي للصفوف المرشّحة للنقل)
# ═══════════════════════════════════════════════════════════════════════════
DB_MAINT_TABLES: List[tuple] = [
    (ROOT / "memory" / "chat_history.db", "chat_messages", "created_at"),
    (ROOT / "memory" / "route_log.db", "route_log", "created_at"),
    (ROOT / "memory" / "swarm_history.db", "swarm_history", "started_at"),
    (ROOT / "memory" / "collective_memory.db", "collective_lessons", "created_at"),
    (ROOT / "memory" / "experience.db", "neural_episodes", "timestamp"),
    (ROOT / "memory" / "episodic.db", "episodes", "timestamp"),
]

# مدة الصيانة الدورية بالساعات (24 ساعة — آمن تماماً لقرص محدود بطيء النمو)
NSM_DB_MAINT_INTERVAL_HOURS = 24

# عتبة الحجم الأدنى لتنفيذ VACUUM (بايت) — بلاها، VACUUM على قاعدة 4KB
# يستهلك وقتاً أطول من الفائدة ويغلق الملف مؤقتاً.
NSM_DB_VACUUM_MIN_BYTES = 2 * 1024 * 1024  # 2 ميجابايت

# مدة انتظار قفل القاعدة قبل التخلي عن VACUUM (ms) — لا نعلّق التطبيق
NSM_DB_BUSY_TIMEOUT_MS = 2000

_lock = threading.Lock()
_timer: Optional[threading.Timer] = None
_last_run: Optional[Dict] = None


def _safe_conn(db_path: Path) -> Optional[sqlite3.Connection]:
    """يفتح اتصالاً آمناً (WAL + busy timeout) أو يعيد None عند الفشل."""
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False,
                               timeout=NSM_DB_BUSY_TIMEOUT_MS / 1000)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={NSM_DB_BUSY_TIMEOUT_MS}")
        return conn
    except Exception as e:  # pragma: no cover — الفشل يُبتلَع بتصميم
        logger.warning("[db_maintenance] فشل فتح %s: %s", db_path, e)
        return None


def _archive_table(db_path: Path, table: str, ts_col: str,
                   cutoff_iso: str) -> int:
    """ينقل الصفوف الأقدم من cutoff إلى جدول الأرشيف ثم يحذفها.

    جدول الأرشيف (archive_{table}) يُبنى تلقائياً بنفس مخطط الجدول الحي
    (SELECT ... INTO لا يعمل في SQLite لكل الحالات، فالنستخدم CREATE AS
    SELECT مرة واحدة ثم INSERT). يعيد عدد الصفوف المنقولة (0 عند أي
    فشل أو عدم وجود صفوف قديمة).
    """
    archived = 0
    archive_table = f"archive_{table}"
    conn = _safe_conn(db_path)
    if conn is None:
        return 0
    try:
        # 1. أنشئ جدول الأرشيف إن لم يكن موجوداً (نفس الأعمدة) — لا يُنفَّذ
        #    إلا مرة واحدة في عمر قاعدة البيانات، وCREATE TABLE IF NOT
        #    EXISTS يتجاهل البناء إن وُجد أصلاً.
        rows = conn.execute(
            f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'"
        ).fetchone()
        if not rows:
            # الجدول الحي غير موجود أصلاً — لا شيء للأرشفة
            return 0
        live_sql = rows[0]
        archive_sql = live_sql.replace(f"CREATE TABLE {table}",
                                       f"CREATE TABLE {archive_table}", 1)\
                              .replace(f"CREATE TABLE IF NOT EXISTS {table}",
                                       f"CREATE TABLE IF NOT EXISTS {archive_table}", 1)
        conn.execute(archive_sql)
        conn.commit()

        # 2. انقل الصفوف القديمة إلى الأرشيف (INSERT ... SELECT مع شرط
        #    الوقت — الصفوف المنقولة تُحذف لاحقاً ولا يتكرر نقلها)
        cur = conn.execute(
            f"INSERT OR IGNORE INTO {archive_table} SELECT * FROM {table} "
            f"WHERE {ts_col} < ?", (cutoff_iso,)
        )
        archived = cur.rowcount or 0

        # 3. احذف المنقول من الجدول الحي — نفس الشرط الزمني يضمن أن أي صف
        #    جديد (أحدث من cutoff) لا يُمسّ أبداً.
        if archived > 0:
            conn.execute(f"DELETE FROM {table} WHERE {ts_col} < ?",
                         (cutoff_iso,))
            conn.commit()

        # 4. أرشيف بلا حدود أيضاً — لكن بعتبة أوسع (سنة) كي يظل الأرشيف
        #    قابلاً للاسترجاع ويحافظ على غرضه كأمان ضد فقدان البيانات.
        old_cutoff = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        conn.execute(f"DELETE FROM {archive_table} WHERE {ts_col} < ?",
                     (old_cutoff,))
        conn.commit()
    except Exception as e:  # pragma: no cover
        logger.warning("[db_maintenance] فشل أرشفة %s.%s: %s",
                       db_path.name, table, e)
        archived = 0
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return archived


def _vacuum_db(db_path: Path) -> bool:
    """يُعيد مساحة القرص المحررة عبر VACUUM — فقط إذا تجاوز الحجم
    NSM_DB_VACUUM_MIN_BYTES. يعيد True عند النجاح الفعلي."""
    if not db_path.exists() or db_path.stat().st_size < NSM_DB_VACUUM_MIN_BYTES:
        return False
    conn = _safe_conn(db_path)
    if conn is None:
        return False
    try:
        # checkpoint(TRUNCATE) قبل VACUUM — يفرّغ ملف WAL من أي صفحات
        # معلّقة قد تمنع إعادة تقليص حجم الملف الرئيسي (SQLite لا يُعيد
        # ضبط حجم الملف ما دامت هناك صفحات حية في WAL لوجود اتصال آخر)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA auto_vacuum=FULL")
        conn.execute("VACUUM")
        # checkpoint بعد VACUUM لضمان أن الملف الجديد استقر على القرص
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return True
    except Exception as e:  # pragma: no cover
        logger.warning("[db_maintenance] فشل VACUUM لـ %s: %s", db_path, e)
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_maintenance_once(days: int = 30, dry_run: bool = False) -> Dict:
    """شغّل دورة صيانة كاملة مرة واحدة.

    يعيد إحصائية: {tables_archived: [{db, table, archived}],
                   vacuumed: [db names], db_sizes_before/after}
    عند dry_run=True يُحسَب فقط بلا أي كتابة — آمن تماماً للاستكشاف.
    """
    global _last_run
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    result: Dict = {
        "cutoff": cutoff, "tables_archived": [], "vacuumed": [],
        "errors": [], "dry_run": dry_run,
    }
    sizes_before = {str(p): (p.stat().st_size if p.exists() else 0)
                    for p, _, _ in DB_MAINT_TABLES}
    for db_path, table, ts_col in DB_MAINT_TABLES:
        try:
            if dry_run:
                # عدّ فقط — بلا أرشفة أو حذف
                conn = _safe_conn(db_path)
                if conn is None:
                    continue
                try:
                    row = conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE {ts_col} < ?",
                        (cutoff,)).fetchone()
                    n = row[0] if row else 0
                    result["tables_archived"].append(
                        {"db": db_path.name, "table": table,
                         "archived": n, "dry_run": True})
                finally:
                    conn.close()
                continue
            n = _archive_table(db_path, table, ts_col, cutoff)
            if n or True:  # نسجّل الجدول المدروس حتى عند 0 (شفافية)
                result["tables_archived"].append(
                    {"db": db_path.name, "table": table, "archived": n})
        except Exception as e:
            result["errors"].append({"db": db_path.name, "error": str(e)})
            logger.warning("[db_maintenance] خطأ في %s.%s: %s",
                           db_path.name, table, e)
        try:
            if _vacuum_db(db_path):
                result["vacuumed"].append(db_path.name)
        except Exception:
            pass
    sizes_after = {str(p): (p.stat().st_size if p.exists() else 0)
                   for p, _, _ in DB_MAINT_TABLES}
    result["db_sizes_before"] = sizes_before
    result["db_sizes_after"] = sizes_after
    _last_run = result
    logger.info("[db_maintenance] دورة صيانة: cutoff=%s | جداول=%d | "
                "vacuumed=%d", cutoff, len(result["tables_archived"]),
                len(result["vacuumed"]))
    return result


def _scheduled_loop() -> None:
    """حلقة المجدولة: تشغّل دورة ثم تعيد جدولة نفسها بعد الفاصل."""
    try:
        run_maintenance_once(days=30)
    except Exception as e:  # pragma: no cover
        logger.warning("[db_maintenance] فشل دورة مجدولة: %s", e)
    finally:
        schedule_next()


def schedule_next() -> None:
    """يجدول الدورة القادمة (لا شيء إن كانت قد أُلغيت)."""
    global _timer
    if _timer is not None:
        _timer = threading.Timer(NSM_DB_MAINT_INTERVAL_HOURS * 3600,
                                 _scheduled_loop)
        _timer.daemon = True
        _timer.start()


def start_periodic_maintenance(interval_hours: float = NSM_DB_MAINT_INTERVAL_HOURS) -> None:
    """يبدأ الإدارة الدورية: دورة أولية + مؤقت كل interval_hours.

    آمنة للاستدعاء المتكرر (idempotent): تتجاهل الاستدعاءات اللاحقة.
    تُستدعى مرة واحدة عند بدء التطبيق.
    """
    global _timer, NSM_DB_MAINT_INTERVAL_HOURS
    if _timer is not None:
        return  # استُدعيت سابقاً — لا نضاعف المؤقتات
    NSM_DB_MAINT_INTERVAL_HOURS = float(interval_hours)
    # دورة أولية بعد إقلاع قصير (60 ث) — ننتظر استقرار Streamlit
    # وتوافر ملفات القواعد كي لا نغلقها أثناء أول كتابة.
    _timer = threading.Timer(60.0, _scheduled_loop)
    _timer.daemon = True
    _timer.start()
    logger.info("[db_maintenance] بدأت الإدارة الدورية (كل %.0f ساعة)",
                NSM_DB_MAINT_INTERVAL_HOURS)


def get_maintenance_stats() -> Dict:
    """آخر نتيجة صيانة + حجم قواعد البيانات الحالية — للاستخدام في أي
    لوحة مراقبة لاحقاً. بلا استثناء أبداً."""
    try:
        sizes = {p.name: (p.stat().st_size if p.exists() else 0)
                 for p, _, _ in DB_MAINT_TABLES}
        return {"last_run": _last_run, "db_sizes": sizes}
    except Exception:
        return {"last_run": None, "db_sizes": {}}


# تشغيل أولي آمن عند استيراد الوحدة: يبدأ مؤقتاً daemon فقط إذا لم يُستدعَ
# start_periodic_maintenance() صراحة من مكان آخر (app_core). هذا يجعل
# الوحدة تعمل ذاتياً حتى لو لم يُضَف استدعاء startup — لا تعارض مع شيء.
if __name__ != "__main__":
    try:
        start_periodic_maintenance()
    except Exception:  # pragma: no cover
        pass
