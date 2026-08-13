"""
ai/team_experience.py — سجل الخبرات والقرارات الجماعية المتراكم (TEM)
======================================================================
ذاكرة ذاتية جماعية مستمرة لفريق الوكلاء: كل مهمة تعاونية أو طويلة الأمد
تُسجَّل فيها **خبرة** عن قرار جرى اتخاذه ونتيجته الفعلية (success / partial /
failure)، ثم تُستحضر هذه الخبرات **قبل التخطيط** للمهام المماثلة الجديدة
فيوجه الوكلاء بعيدًا عن المسارات الفاشلة سابقًا ونحو الأنجح منها.

الفرق عن الوحدات المجاورة:
- ai/collective_memory.py: دروس نصية (ماذا تعلّمنا) مع جودة تصويتية،
  تُحقن في برومبت المدير الموحّد للمهام الفردية فقط.
- ai/shared_knowledge.py (SKB): نتائج بحث/جلب **لحظية** داخل المهمة الواحدة.
- هذه الوحدة: **قرارات وخبرات متراكمة عبر المهام** (ماذا فعلنا/ماذا حصل)
  تُستحضر في مرحلتي **التخطيط** (قبل بناء الخطة/الأدوار) و**الإتمام**.

التخزين: SQLite محلي دائم في data/team_experience.db (فصل كامل عن
data/mesh.db وdata/shared_knowledge.db وmemory/collective_memory.db).

لا يفشل أي استدعاء ولا يرفع استثناء خارجيًا — كل شيء محمي بـ try/except
مع تسجيل تحذير. التدهور: فشل كامل → وحدة معطلة بصمت (block _TEM_OK).

الجداول:
  tem_decisions: خبرة واحدة (قرار/إجراء ونتيجته) مع تكرارها (hits) وترتيبها
    تلقائيًا بالثقة × التكرار. البحث حرفي عربي مرن (OR كلمات) بدون أي
    اعتماد على Qdrant أو نموذج التضمين.

التكامل: استيراد اختياري (_TEM_OK) في app_core — أي فشل يعيد السلوك الأصلي.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── الثوابت ────────────────────────────────────────────────────────────────

_LOCAL_DB = "data/team_experience.db"
_MAX_CATALOG = 300                 # سقف الخبرات المحفوظة (LRU بالأقدم)
_MAX_EXPERIENCE_CHARS = 2500       # سقف طول سياق/قرار الواحد
_MIN_CONFIDENCE = 0.3              # أدنى ثقة للاستمعال في الاستحضار
_STEP_MAX = 25                     # أقصى خبرات خطوات تُضاف لمهمة واحدة

_CATEGORIES = (
    "plan_strategy",   # استراتيجية الخطة (عدد الخطوات/الترتيب)
    "role_assign",     # تعيين الأدوار وتوزيعها
    "search_method",   # أسلوب البحث/الجلب
    "verification",    # التحقق والتصحيح
    "failure_avoid",   # تحذير من فشل سابق
    "general",         # خبرة عامة
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tem_decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    category    TEXT    NOT NULL DEFAULT 'general',
    context     TEXT    NOT NULL DEFAULT '',
    decision    TEXT    NOT NULL,
    searchable  TEXT    NOT NULL DEFAULT '',
    outcome     TEXT    NOT NULL CHECK(outcome IN
                ('success', 'partial', 'failure')) DEFAULT 'success',
    confidence  REAL    NOT NULL DEFAULT 0.5,
    hits        INTEGER NOT NULL DEFAULT 1,
    task_id     TEXT    NOT NULL DEFAULT '',
    agents      TEXT    NOT NULL DEFAULT '',
    created_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tem_search ON tem_decisions(searchable);
CREATE INDEX IF NOT EXISTS idx_tem_category ON tem_decisions(category);
"""

# ── دوال مساعدة داخلية ──────────────────────────────────────────────────────


def _now_ts() -> float:
    return time.time()


def _normalize(text: str) -> str:
    """تنظيف نص عربي للبحث الحرفي: إزالة التشكيل والهمزات والمسافات المكررة."""
    if not text:
        return ""
    t = text.lower()
    # إزالة التشكيل
    t = "".join(ch for ch in t if ch not in (
        "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652"))
    # توحيد الألف والهمزات والياء
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ة", "ه").replace("ى", "ي")
    t = t.replace("ؤ", "و").replace("ئ", "ي")
    t = t.replace("\u200f", "").replace("\u200e", "")
    parts = " ".join(t.split())
    return parts


def _extract_words(text: str) -> List[str]:
    return [w for w in _normalize(text).split() if len(w) >= 2]


# ── السجل الجماعي ────────────────────────────────────────────────────────────


class SharedExperienceLog:
    """سجل الخبرات والقرارات الجماعية المتراكم لفريق الوكلاء."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db = os.path.abspath(db_path or _LOCAL_DB)
        self._lock = threading.Lock()
        try:
            os.makedirs(os.path.dirname(self._db) or ".", exist_ok=True)
            with sqlite3.connect(self._db, timeout=30) as conn:
                conn.executescript(_SCHEMA)
        except Exception as exc:
            logger.warning("TEM init failed: %s", exc)

    # ── الكتابة ────────────────────────────────────────────────────────────

    def record(self, context: str, decision: str, outcome: str,
               category: str = "general", confidence: float = 0.5,
               task_id: str = "", agents: str = "") -> bool:
        """تسجيل خبرة واحدة (قرار + نتيجته الفعلية).

        outcome في ('success', 'partial', 'failure').
        الثقة تُعدَّل حسب النتيجة: فشل → تخفض، نجاح → ترفع حتى 0.95."""
        if outcome not in ("success", "partial", "failure"):
            outcome = "partial"
        if category not in _CATEGORIES:
            category = "general"
        context = (context or "")[:_MAX_EXPERIENCE_CHARS]
        decision = (decision or "").strip()[:_MAX_EXPERIENCE_CHARS]
        if not decision:
            return False
        confidence = float(min(0.95, max(0.0, confidence)))
        if outcome == "success":
            confidence = min(0.95, confidence + 0.1)
        elif outcome == "failure":
            confidence = max(0.0, confidence - 0.25)
        searchable = " ".join(_extract_words(context)) + " " + \
            " ".join(_extract_words(decision))
        try:
            with self._lock, sqlite3.connect(self._db, timeout=30) as conn:
                conn.execute("""
                    INSERT INTO tem_decisions
                        (category, context, decision, searchable, outcome,
                         confidence, hits, task_id, agents, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (category, context, decision, searchable, outcome,
                      confidence, 1, (task_id or "")[:60],
                      (agents or "")[:200], _now_ts()))
                conn.commit()
            self._prune()
            return True
        except Exception as exc:
            logger.warning("TEM record failed: %s", exc)
            return False

    def promote_on_outcome(self, context: str, outcome: str) -> bool:
        """مطابقة خبرة موجودة بالسياق وتعزيز outcome جديد لها (إعادة استخدام)."""
        if not context or outcome not in ("success", "failure"):
            return False
        words = _extract_words(context)[:5]
        if not words:
            return False
        try:
            with self._lock, sqlite3.connect(self._db, timeout=30) as conn:
                row = conn.execute("""
                    SELECT id, hits, outcome, confidence FROM tem_decisions
                    WHERE (%s)
                    ORDER BY confidence * (hits + 1) DESC LIMIT 1
                """ % " OR ".join("searchable LIKE ?" for _ in words),
                                   ["%" + w + "%" for w in words]
                                   ).fetchone()
                if row is None:
                    return False
                rid, hits, prev_outcome, q = row
                new_hits = hits + 1
                if outcome == "success":
                    new_q = min(0.95, q + 0.08)
                else:
                    new_q = max(0.0, q - 0.15)
                conn.execute("""
                    UPDATE tem_decisions
                    SET hits=?, confidence=?, outcome=?, task_id=?, agents=?
                    WHERE id=?
                """, (new_hits, new_q, outcome, "", "", rid))
                conn.commit()
                return True
        except Exception as exc:
            logger.warning("TEM promote failed: %s", exc)
            return False

    # ── الاستحضار ──────────────────────────────────────────────────────────

    def recall(self, query: str, category: Optional[str] = None,
               top_k: int = 8, min_confidence: float = _MIN_CONFIDENCE
               ) -> List[Dict[str, Any]]:
        """استحضار الخبرات الأنسب للتخطيط: بحث حرفي OR + ترتيب بالثقة×التكرار."""
        words = _extract_words(query)
        if not words:
            return []
        conds = ["searchable LIKE ?"] + [
            "searchable LIKE ?" for _ in words]
        params = ["%" + _normalize(query)[:200].replace(" ", "%") + "%"] + [
            "%" + w + "%" for w in words]
        if category and category in _CATEGORIES:
            conds.append("category = ?")
            params.append(category)
        try:
            with self._lock, sqlite3.connect(self._db, timeout=30) as conn:
                rows = conn.execute(
                    "SELECT id, category, context, decision, outcome, "
                    "confidence, hits, task_id, agents, created_at "
                    "FROM tem_decisions WHERE (" + " OR ".join(conds) + ") "
                    "ORDER BY confidence * (hits + 1) DESC, "
                    "created_at DESC LIMIT ?",
                    params + [top_k * 3]).fetchall()
        except Exception:
            return []
        result = []
        for row in rows:
            if len(result) >= top_k:
                break
            rid, cat, ctx, dec, out, q, hits, tid, ag, ts = row
            if q < min_confidence:
                continue
            result.append({
                "id": rid, "category": cat, "context": ctx, "decision": dec,
                "outcome": out, "confidence": round(q, 3), "hits": hits,
                "task_id": tid, "agents": ag, "ts": ts,
            })
        return result[:top_k]

    def latest(self, k: int = 10) -> List[Dict[str, Any]]:
        """أحدث الخبرات دون استعلام (للوحة المراقبة)."""
        try:
            with self._lock, sqlite3.connect(self._db, timeout=30) as conn:
                rows = conn.execute(
                    "SELECT id, category, context, decision, outcome, "
                    "confidence, hits, task_id, agents, created_at "
                    "FROM tem_decisions ORDER BY created_at DESC LIMIT ?",
                    (k,)).fetchall()
        except Exception:
            return []
        return [{
            "id": r[0], "category": r[1], "context": r[2], "decision": r[3],
            "outcome": r[4], "confidence": round(r[5], 3), "hits": r[6],
            "task_id": r[7], "agents": r[8], "ts": r[9],
        } for r in rows]

    # ── الإحصاء والصيانة ──────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        try:
            with sqlite3.connect(self._db, timeout=30) as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM tem_decisions").fetchone()[0]
                by_outcome = {
                    row[0]: row[1] for row in conn.execute(
                        "SELECT outcome, COUNT(*) FROM tem_decisions "
                        "GROUP BY outcome").fetchall()}
                by_cat = {
                    row[0]: row[1] for row in conn.execute(
                        "SELECT category, COUNT(*) FROM tem_decisions "
                        "GROUP BY category ORDER BY COUNT(*) DESC").fetchall()}
                success_rate = (
                    by_outcome.get("success", 0) / total if total else 0.0)
                return {
                    "total_experiences": total,
                    "by_outcome": by_outcome,
                    "by_category": by_cat,
                    "success_rate": round(success_rate, 3),
                }
        except Exception as exc:
            logger.warning("TEM stats failed: %s", exc)
            return {"total_experiences": 0, "by_outcome": {},
                    "by_category": {}, "success_rate": 0.0}

    def _prune(self) -> None:
        """حذف أقدم الخبرات عند تجاوز السقف."""
        try:
            with self._lock, sqlite3.connect(self._db, timeout=30) as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM tem_decisions").fetchone()[0]
                if total > _MAX_CATALOG:
                    conn.execute("""
                        DELETE FROM tem_decisions
                        WHERE created_at IN (
                            SELECT created_at FROM tem_decisions
                            ORDER BY created_at ASC
                            LIMIT ?)
                    """, (total - _MAX_CATALOG,))
                    conn.commit()
        except Exception as exc:
            logger.warning("TEM prune failed: %s", exc)


# ── Singleton ────────────────────────────────────────────────────────────────

_default: Optional[SharedExperienceLog] = None


def get_experience_log(db_path: Optional[str] = None
                       ) -> SharedExperienceLog:
    global _default
    if _default is None:
        _default = SharedExperienceLog(db_path)
    return _default


def reset_experience_log() -> None:
    """للاختبار فقط: إعادة بناء الـsingleton."""
    global _default
    _default = None


# ── دوال مستوى المهمة (واجهة الدمج في التعاون وLHT) ─────────────────────────

def record_task_experience(task_id: str, decision: str, outcome: str,
                           category: str = "general",
                           confidence: float = 0.5,
                           agents: str = "") -> bool:
    """تسجيل خبرة مهمة واحدة (تُستخدم بعد اكتمال المهمة)."""
    try:
        return get_experience_log().record(
            context=task_id, decision=decision, outcome=outcome,
            category=category, confidence=confidence,
            task_id=task_id, agents=agents)
    except Exception:
        return False


def recall_task_experiences(query: str, category: Optional[str] = None,
                            top_k: int = 8) -> List[Dict[str, Any]]:
    """استحضار الخبرات ذات الصلة قبل التخطيط (تُستخدم في التخطيط)."""
    try:
        return get_experience_log().recall(
            query, category=category, top_k=top_k)
    except Exception:
        return []


def tem_stats() -> Dict[str, Any]:
    try:
        return get_experience_log().stats()
    except Exception:
        return {}


def tem_latest(k: int = 10) -> List[Dict[str, Any]]:
    try:
        return get_experience_log().latest(k)
    except Exception:
        return []
