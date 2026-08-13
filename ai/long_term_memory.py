"""long_term_memory.py — الذاكرة الذاتية المستمرة (Long-Term Memory)
===================================================================

طبقة ذاكرة طويلة المدى قابلة للاستدعاء السياقي: تتعلّم تلقائيًا من كل
محادثة ناجحة (أسئلة وأجوبتها)، وتربط هذه التجارب بالدروس الجماعية
(collective_memory) بحيث يستحضر الوكيل ذكرياته عند الإجابة عن أسئلة
متشابهة — خطوة أولى نحو التعلم الذاتي المستمر.

المبادئ (تدهور آمن كامل):
- قاعدة بيانات مستقلة تمامًا: memory/long_term_memory.db — لا تلمس
  mesh.db ولا collective_memory.db ولا chat_history_store.
- كل فشل يُبتلَع بتحذير مسجّل فقط؛ لا يوجد أي مسار يرمي استثناء إلى
  واجهة chat — النظام يعمل بالكامل بلا هذه الطبقة لو فشلت.
- التعلم تلقائي وضمن الحدود: حد أقصى للذكريات + تآكل تلقائي للذكريات
  ضعيفة الجودة (quality < 0 غير محتمل: نُزيل القديمة غير المُنسوبة
  بعد 180 يومًا) — لا تراكم بلا حدود.
- الاستدعاء لا يستهلك توكنز: الذكريات تُعرض كسياق مساعد في واجهة
  المحادثة وتُستدعى من openrouter context_info عند توفر NSMChatPlus
  (حقن خفيف فقط).

جداول long_term_memory.db:
- long_term_memories(id, question_hint, topic, insight, memory_type,
  domain, quality, access_count, last_accessed, created_at)
- memory_access_log(id, memory_id, query_hint, accessed_at) — لأجل
  get_maintenance_stats وقياس الاستخدام

أنواع الذاكرة (memory_type):
- question   — سؤال متكرر الشائع مع أجوبته النموذجية
- correction — تصويب تعلمته المنصة من تفاعل سابق (جودة عالية)
- preference — تفضيل مستخدم مسجل (نبرة، مستوى التفصيل...)
- lesson     — درس مشتق من collective_memory ذات صلة قوية
- fact       — حقيقة معرفية استُخرجت من محادثة

"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nsm.long_term_memory")

ROOT = Path(__file__).resolve().parent.parent
MEM_DIR = ROOT / "memory"
LTM_DB_PATH = MEM_DIR / "long_term_memory.db"

MAX_MEMORIES = 2000           # سقف الذكريات الكلية
RECALL_TOP_K = 4              # ذكريات تُستحضر لكل سؤال
QUALITY_BOOST_ACCESS = 0.01   # تعزيز الجودة مع كل استخدام ناجح
QUALITY_DECAY_DAYS = 180      # بعد 180 يومًا دون وصول: تُحذف
ACCESS_DECAY_DAYS = 30        # ذاكرة بلا وصول 30 يومًا تُخفّض جودتها

# ── تطبيع عربي متوافق مع normalize_arabic في app_core ─────────────────
_NORMALIZE_RE_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670\u0640]')
_NORMALIZE_RE_HAMZA = re.compile(r'[أإآٱ]')
_NORMALIZE_RE_BOM = re.compile(r'\ufeff')
_NORMALIZE_RE_WS = re.compile(r'\s+')


def normalize_ltm(text: str) -> str:
    text = _NORMALIZE_RE_DIACRITICS.sub('', text)
    text = _NORMALIZE_RE_HAMZA.sub('ا', text)
    text = _NORMALIZE_RE_BOM.sub('', text)
    text = _NORMALIZE_RE_WS.sub(' ', text)
    # تطابق normalize_arabic في app_core حرفيًا (وُثّق أن مخرجها الفعلي
    # لـ"أُتْقِنَ" هو "اتقن" — إزالة التشكيل مرة واحدة تكفي لأن re.sub
    # يزيل كل الرموز المركّبة في تمريرة واحدة)، مع lower للتطابق الدلالي
    # في الاستحضار.
    return text.strip().lower()


# كلمات وقوف عربية شائعة لا تحمل دلالة للربط الدلالي
_STOP_WORDS = {
    'ما', 'في', 'من', 'على', 'إلى', 'الى', 'عن', 'مع', 'هو', 'هي', 'هم',
    'هذا', 'هذه', 'ذلك', 'تلك', 'ال', 'و', 'أو', 'لا', 'أن', 'ان', 'إن',
    'كان', 'كانت', 'يكون', 'هناك', 'كيف', 'لماذا', 'ماهو', 'ماهو',
    'شرح', 'اشرح', 'عرف', 'عرفني', 'اعرف', 'ماهي', 'عني',
}


def _keyword_tokens(text: str) -> List[str]:
    return [t for t in normalize_ltm(text).split() if t and t not in _STOP_WORDS and len(t) >= 2]


class LongTermMemory:
    """ذاكرة طويلة المدى قابلة للاستدعاء: تعلّم تلقائي + استحضار دلالي."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else LTM_DB_PATH
        self._lock = threading.Lock()
        self._ensure_db()

    # ───────────────────────────── init ──────────────────────────────────
    def _ensure_db(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS long_term_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question_hint TEXT NOT NULL,
                        topic TEXT NOT NULL,
                        insight TEXT NOT NULL,
                        memory_type TEXT NOT NULL DEFAULT 'question',
                        domain TEXT NOT NULL DEFAULT 'عام',
                        quality REAL NOT NULL DEFAULT 0.5,
                        access_count INTEGER NOT NULL DEFAULT 0,
                        last_accessed TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ltm_domain ON
                        long_term_memories(domain)
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_access_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        memory_id INTEGER NOT NULL,
                        query_hint TEXT,
                        accessed_at TEXT NOT NULL
                    )
                """)
                conn.commit()
        except Exception as e:  # pragma: no cover
            logger.warning("[long_term_memory] تعذّر تهيئة قاعدة الذكريات: %s", e)

    # ───────────────────────────── تعلّم ──────────────────────────────────
    def learn(self, question: str, answer_or_outcome: str = "",
              memory_type: str = "question", domain: Optional[str] = None,
              quality: float = 0.5) -> Optional[int]:
        """تسجيل تجربة جديدة في الذاكرة الطويلة.

        عند وجود ذكرى مشابهة جدًا (موضوعها ذاته) نحدّثها بدل التكرار:
        نرفع جودة القديمة ونوسّع hint. النوع 'correction' يتجاوز جودة
        القديمة دائمًا (تصويب أهم من سؤال عادي).
        """
        if not question or not question.strip():
            return None
        tokens = _keyword_tokens(question)
        if not tokens:
            return None
        memory_type = memory_type if memory_type in {
            "question", "correction", "preference", "lesson", "fact"
        } else "question"
        domain = normalize_ltm(domain or "عام")
        quality = max(0.0, min(1.0, float(quality)))

        try:
            with self._lock, sqlite3.connect(str(self.db_path)) as conn:
                # ── 1. البحث عن ذكرى مشابهة (موضوعها يطابق كلمتين على الأقل) ──
                existing = conn.execute(
                    "SELECT id, topic, quality, memory_type FROM long_term_memories "
                    "WHERE domain = ? OR domain = 'عام'", (domain,)
                ).fetchall()
                best = None
                best_overlap = 0
                existing_tokens_cache = []
                for eid, etopic, eq, etype in existing:
                    et = _keyword_tokens(etopic)
                    existing_tokens_cache.append((eid, et, eq, etype))
                    overlap = len(set(tokens) & set(et))
                    if overlap > best_overlap:
                        best_overlap, best = overlap, (eid, eq, etype)

                if best and best_overlap >= 2:
                    eid, eq, etype = best
                    # تصويب يرفع الجودة، والسؤال العادي يعززها قليلًا
                    boost = 0.08 if memory_type == "correction" else 0.03
                    new_q = min(1.0, eq + boost)
                    conn.execute(
                        "UPDATE long_term_memories SET quality = ?, "
                        "question_hint = question_hint || ' | ' || ? "
                        "WHERE id = ?",
                        (round(new_q, 3), normalize_ltm(question)[:80], eid)
                    )
                    conn.commit()
                    return eid

                # ── 2. لا ذكرى مشابهة: أدرج جديدة (مع تطبيق السقف) ──
                conn.execute("DELETE FROM long_term_memories WHERE id NOT IN "
                             "(SELECT id FROM long_term_memories ORDER BY "
                             "quality DESC, access_count DESC LIMIT ?)",
                             (MAX_MEMORIES,))
                conn.execute(
                    "INSERT INTO long_term_memories "
                    "(question_hint, topic, insight, memory_type, domain, "
                    "quality, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (normalize_ltm(question)[:200], normalize_ltm(question)[:120],
                     normalize_ltm(answer_or_outcome)[:500], memory_type,
                     domain, quality, datetime.now(timezone.utc).isoformat())
                )
                mid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                conn.commit()
                return mid
        except Exception as e:  # pragma: no cover
            logger.warning("[long_term_memory] فشل التعلّم: %s", e)
            return None

    # ───────────────────────────── استدعاء ─────────────────────────────────
    def recall(self, query: str, domain: Optional[str] = None,
               top_k: int = RECALL_TOP_K) -> List[Dict[str, Any]]:
        """استحضار أفضل الذكريات ذات الصلة بالسؤال.

        المعيار: (1) مطابقة موضوعية عبر كلمات مشتركة (كلمتان+)،
        (2) فلتر جودة: quality + 0.05·access_count ≥ 0.3،
        (3) ترتيب: quality DESC ثم access_count DESC.
        """
        tokens = _keyword_tokens(query)
        if not tokens:
            return []
        domain = normalize_ltm(domain or "")

        try:
            with self._lock, sqlite3.connect(str(self.db_path)) as conn:
                rows = conn.execute(
                    "SELECT id, question_hint, topic, insight, memory_type, "
                    "domain, quality, access_count, last_accessed "
                    "FROM long_term_memories "
                    "WHERE domain = ? OR domain = 'عام' "
                    "ORDER BY quality DESC, access_count DESC",
                    (domain or "عام",)
                ).fetchall()
        except Exception as e:  # pragma: no cover
            logger.warning("[long_term_memory] فشل الاستحضار: %s", e)
            return []

        result = []
        now = datetime.now(timezone.utc).isoformat()
        for row in rows:
            if len(result) >= top_k:
                break
            (mid, hint, topic, insight, mtype, d, q, acc, last_acc) = row
            topic_tokens = _keyword_tokens(topic)
            overlap = len(set(tokens) & set(topic_tokens))
            if overlap < 2:
                continue
            score = float(q) + 0.05 * int(acc)
            if score < 0.3:
                continue
            # سجل الوصول ورفّع الجودة قليلًا
            try:
                with sqlite3.connect(str(self.db_path)) as conn2:
                    conn2.execute(
                        "UPDATE long_term_memories SET access_count = access_count + 1, "
                        "last_accessed = ?, quality = quality + ? WHERE id = ?",
                        (now, QUALITY_BOOST_ACCESS, mid))
                    conn2.execute(
                        "INSERT INTO memory_access_log (memory_id, query_hint, accessed_at) "
                        "VALUES (?, ?, ?)", (mid, normalize_ltm(query)[:80], now))
                    conn2.commit()
            except Exception:
                pass
            result.append({
                "memory_id": mid, "question_hint": hint, "topic": topic,
                "insight": insight, "memory_type": mtype, "domain": d,
                "quality": round(float(q) + QUALITY_BOOST_ACCESS, 3),
                "access_count": int(acc) + 1,
            })
        return result[:top_k]

    # ──────────────────────── تعلّم من الدروس الجماعية ─────────────────────
    def ingest_collective_lessons(self, collective_db_path: Optional[Path] = None) -> int:
        """مزامنة أفضل الدروس الجماعية الجيدة (quality >= 0.7) كذكريات
        من نوع lesson — بلا تكرار (بالحقل topic)."""
        cm_path = Path(collective_db_path) if collective_db_path else (MEM_DIR / "collective_memory.db")
        if not cm_path.exists():
            return 0
        try:
            imported = 0
            with sqlite3.connect(str(cm_path)) as cm, \
                 sqlite3.connect(str(self.db_path)) as lc:
                lessons = cm.execute(
                    "SELECT question_hint, lesson, domain, quality "
                    "FROM collective_lessons WHERE quality >= 0.7 "
                    "ORDER BY quality DESC, task_hits DESC LIMIT 50"
                ).fetchall()
                for hint, lesson, d, q in lessons:
                    topic = normalize_ltm(hint or "")[:120]
                    exists = lc.execute(
                        "SELECT 1 FROM long_term_memories WHERE topic = ? "
                        "AND memory_type = 'lesson'", (topic,)
                    ).fetchone()
                    if exists:
                        continue
                    lc.execute(
                        "INSERT INTO long_term_memories "
                        "(question_hint, topic, insight, memory_type, domain, "
                        "quality, created_at) VALUES (?, ?, ?, 'lesson', ?, ?, ?)",
                        (normalize_ltm(hint or "")[:200], topic,
                         normalize_ltm(lesson or "")[:500],
                         normalize_ltm(d or "عام"), min(1.0, float(q or 0.7)),
                         datetime.now(timezone.utc).isoformat()))
                    imported += 1
                lc.commit()
            return imported
        except Exception as e:  # pragma: no cover
            logger.warning("[long_term_memory] فشل مزامنة الدروس: %s", e)
            return 0

    # ───────────────────────────── صيانة ───────────────────────────────────
    def decay(self) -> int:
        """إزالة الذكريات المهجورة القديمة (تآكل تلقائي)."""
        try:
            cutoff = (datetime.now(timezone.utc) -
                      timedelta(days=ACCESS_DECAY_DAYS)).isoformat()
            with self._lock, sqlite3.connect(str(self.db_path)) as conn:
                res = conn.execute(
                    "DELETE FROM long_term_memories WHERE "
                    "(last_accessed IS NULL AND created_at < ?) "
                    "OR (last_accessed < ? AND quality < 0.45)",
                    (cutoff, cutoff))
                conn.commit()
                return res.rowcount
        except Exception as e:  # pragma: no cover
            logger.warning("[long_term_memory] فشل التآكل: %s", e)
            return 0

    def stats(self) -> Dict[str, Any]:
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM long_term_memories").fetchone()[0]
                by_type = {
                    r[0]: r[1] for r in conn.execute(
                        "SELECT memory_type, COUNT(*) FROM long_term_memories "
                        "GROUP BY memory_type").fetchall()}
                last_access = conn.execute(
                    "SELECT accessed_at FROM memory_access_log "
                    "ORDER BY id DESC LIMIT 1").fetchone()
                return {
                    "total_memories": int(total or 0),
                    "by_type": by_type,
                    "last_memory_access": last_access[0] if last_access else None,
                    "db_path": str(self.db_path),
                }
        except Exception as e:  # pragma: no cover
            return {"total_memories": 0, "by_type": {}, "error": str(e)}


# ── singleton ──────────────────────────────────────────────────────────────
_LTM_INSTANCE: Optional[LongTermMemory] = None
_LTM_LOCK = threading.Lock()


def get_ltm(db_path: Optional[Path] = None) -> LongTermMemory:
    global _LTM_INSTANCE
    with _LTM_LOCK:
        if _LTM_INSTANCE is None:
            try:
                _LTM_INSTANCE = LongTermMemory(db_path or LTM_DB_PATH)
            except Exception:  # pragma: no cover
                logger.warning("[long_term_memory] فشل إنشاء الذاكرة الطويلة")
                raise
    return _LTM_INSTANCE


def reset_ltm_cache() -> None:  # للاختبارات فقط
    global _LTM_INSTANCE
    with _LTM_LOCK:
        _LTM_INSTANCE = None
