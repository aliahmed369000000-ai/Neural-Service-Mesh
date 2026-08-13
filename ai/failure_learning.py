# -*- coding: utf-8 -*-
"""
ai/failure_learning.py — تعلّم الأخطاء الجماعي (Failure Learning)
==================================================================
عندما يفشل وكيل في تنفيذ مهمته، لا يضيع هذا الفشل سدى: هذه الوحدة
تُصنِّف الخطأ نمطيًا (دون الاعتماد على مزوّد LLM)، ثم تحوّله إلى
«درس تحذيري» يُخزَّن في الذاكرة الجماعية الدائمة (collective_memory.db)،
وبذلك يستفيد منه بقية الوكلاء في المهام اللاحقة: عند استرجاع الدروس
لمهمة جديدة تُحقَن الدروس التحذيرية الخاصة بمجالها كتحذيرات صريحة
داخل برومبت التوليف، فيتجنبها الجميع حتى لو لم يختبرها أحد منهم.

لماذا تصنيف نمطي بدلًا من LLM؟
- يعمل بدون مفاتيح API وبلا تكلفة — الفشل قد يحدث أصلًا بسبب
  تعذّر الاتصال بمزوّد النماذج، ولا نريد تعلّمًا يعتمد على نفس المزود.
- أنماط أخطاء مكوّنات النظام معروفة ومحدودة (انتهاء مهلة، حدود استخدام،
  أخطاء برمجية...) لذا يكفي قاموس regex قوي.

الدورة:
  1) chat() في agent_categories يلتقط حدث agent_error → يستدعي
     sync_failure_lessons(events, memory) بعد اكتمال المهمة.
  2) كل خطأ يُصنَّف → إن تكرر نمطه يُخفَّض quality أكثر (درس أقوى).
  3) failure_warnings_for_task(task, memory) يعيد تحذيرات المجال
     ذات الصلة للمهمة الجديدة.
  4) _synthesize يدمج التحذيرات في برومبت التوليف المرجّح.
  5) events: failure_lesson_recorded / failure_lesson_recalled.

لا يفشل أي استدعاء من هذه الوحدة — كل شيء محمي بـ try/except.
"""
from __future__ import annotations

import logging
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── أحداث النظام ─────────────────────────────────────────────────────────────

FAILURE_LEARNING_EVENTS = ("failure_lesson_recorded", "failure_lesson_recalled")

FAILURE_CATEGORY_KEYS = ("network", "rate_limit", "quota", "timeout",
                         "coding", "auth", "not_found", "other")

# ── قاموس التصنيف النمطي ────────────────────────────────────────────────────
# (pattern_regex, category, lesson_arabic) — يُفحص بالترتيب
# الترتيب مهم: الأنماط الأعمق (limits/timeout) قبل العامة

ERROR_PATTERNS: List[tuple] = [
    # انتهاء المهلة وشبكة
    (r"(timeout|timed ?out| timed out)",
     "timeout", "انتهت مهلة الاستجابة — جرّب تبسيط الطلب أو تقصير الإدخال قبل إعادة المحاولة."),
    (r"(?i)(network error|DNS|unable to reach|unreachable|connection (refused|reset)|EOF)",
     "network", "تعذّر الاتصال بالشبكة/المزود — تحقق من الوصول للإنترنت قبل إعادة المحاولة."),
    # حدود الاستخدام والمفاتيح
    (r"(?i)(rate ?limit|too many requests|429)",
     "rate_limit", "وصلنا إلى حد الاستخدام (rate limit) — انتظر فترة قصيرة أو بسّط الطلب قبل إعادة المحاولة."),
    (r"(?i)(quota exceeded|insufficient (funds|credits|balance)|billing)",
     "quota", "تجاوزت الحصة/المفاتيح المسموحة (quota) — تحقق من رصيد المزود أو استخدم مزودًا بديلًا."),
    (r"(?i)(401|unauthorized|invalid (api|auth) ?key|invalid credentials|forbidden)",
     "auth", "مفتاح API غير صالح أو غير مصرَّح به — حدّث المفتاح أو المسار قبل إعادة المحاولة."),
    # أخطاء برمجية
    (r"(?i)(syntax ?error|indentation ?error|unexpected token)",
     "coding", "خطأ في بناء الجملة (syntax) — راجع النص المولَّد قبل إعادة المحاولة."),
    (r"(?i)(name ?error|undefined|not defined)",
     "coding", "مرجع غير معرّف (name error) — تحقق من الأسماء والمتغيرات الممرَّرة."),
    (r"(?i)(type ?error|no attribute|NoneType)",
     "coding", "خطأ نوع بيانات (type error) — تحقق من بنى المدخلات قبل إعادة المحاولة."),
    (r"(?i)(import ?error|no module named|cannot import|ModuleNotFoundError)",
     "coding", "وحدة/مكتبة مفقودة (import error) — ثبّت المكتبة المطلوبة قبل إعادة المحاولة."),
    (r"(?i)(key ?error|index ?error|list index out of range)",
     "coding", "خطأ وصول للبيانات (key/index error) — تحقق من وجود الحقول المطلوبة في المخرجات."),
    # موارد مفقودة
    (r"(?i)(404|not found|does not exist)",
     "not_found", "المورد المطلوب غير موجود (404) — تحقق من صحة المعرف أو الرابط قبل إعادة المحاولة."),
    # تعثر LLM عام
    (r"(?i)(status code: ?5\d\d|internal server error|server error)",
     "network", "خطأ داخلي في خادم المزود — أعد المحاولة بعد فترة قصيرة."),
    (r"(?i)(content filter|unsafe|safety|moderation)",
     "other", "فلتر المحتوى رفض الطلب — أعد صياغة الطلب بصيغة أنسب أو أسئلة أكثر مباشرة."),
]

MAX_LESSON_CHARS = 2000
MIN_ERROR_DETAIL_CHARS = 2            # لا نصوّف نص تفصيل أقصر من هذا
REPEAT_QUALITY_DELTA = -0.15          # خصم إضافي لكل تكرار (تسقيف عند -0.9)
SEEN_WINDOW_HOURS = 72                # تكرار داخل هذه النافذة يُعدّ «متكررًا»

# ── دوال نقية (سهلة الاختبار، بلا اعتماديات) ──────────────────────────────────


def classify_error(detail: str) -> Dict[str, Any]:
    """
    يصنّف نص خطأ نمطيًا. يعيد:
      {"category": str, "lesson": str, "matched": bool}
    دائمًا يعيد قاموسًا صالحًا حتى لو لم يُطابق أي نمط.
    """
    detail = str(detail or "")
    for pattern, category, lesson in ERROR_PATTERNS:
        if re.search(pattern, detail):
            return {"category": category, "lesson": lesson, "matched": True}
    return {"category": "other",
            "lesson": "فشل غير مصنف: «%s» — راجع سجل الأخطاء قبل إعادة المحاولة." % detail[:160],
            "matched": bool(detail.strip())}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_emit(event_type: str, agent_id: str, title: str, status: str,
               detail: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """إطلاق حدث على ناقل الأحداث دون أي اعتماديات قاسية."""
    try:
        from ai.agent_event_bus import emit_event
        emit_event(event_type, agent_id=agent_id, title=title,
                   status=status, detail=detail, metadata=metadata)
    except Exception as exc:  # pragma: no cover - حماية فقط
        logger.warning("failure_learning: emit_event failed: %s", exc)


def _existing_fingerprint(memory: Any, fingerprint: str) -> Optional[int]:
    """هل الدرس (بصمة) موجود في الذاكرة؟ يعيد lesson_id أو None."""
    try:
        import sqlite3
        db_path = getattr(memory, "db_path", None)
        if not db_path:
            return None
        with memory._lock, sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT lesson_id FROM collective_lessons "
                "WHERE question_hint=? LIMIT 1", (fingerprint,)).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _extract_domain(task: str) -> str:
    """اشتقاق المجال من نص المهمة — متسق مع تصنيفات الذاكرة الجماعية."""
    keywords = {
        "برمجة": ["كود", "برمجة", "python", "api", "تطبيق", "دالة", "خطأ برمجي", "برامج"],
        "محتوى": ["محتوى", "مقال", "تقرير", "نص", "كتابة", "ملخص", "ترجمة"],
        "بحث": ["بحث", "معلومات", "تحليل", "دراسة", "اكتشاف", "استكشاف"],
        "تصميم": ["تصميم", "واجهة", "ui", "css", "أشكال", "ألوان", "شعار"],
        "بيانات": ["بيانات", "إحصاء", "جدول", "excel"],
        "مساعدة": ["مساعدة", "سؤال", "استشارة", "اقتراح", "نصيحة"],
    }
    task_lower = str(task or "").lower()
    for domain, markers in keywords.items():
        if any(m.lower() in task_lower for m in markers):
            return domain
    return "عام"


def _lesson_fingerprint(category: str, source_agent: str,
                        detail_hint: str) -> str:
    """بصمة درس فريدة: نفس النمط من نفس الوكيل = نفس الدرس."""
    return f"fl:{category}:{source_agent or 'unknown'}:{detail_hint[:80]}"


# ── الاستخراج من أحداث الجلسة ────────────────────────────────────────────────


def failure_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ترشيح أحداث agent_error فقط من قائمة أحداث الجلسة."""
    return [ev for ev in (events or [])
            if (ev.get("event_type") or "").strip() == "agent_error"]


def extract_failure_insights(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    من أحداث فشل الجلسة يستخرج قائمة بصائر:
      [{"category", "lesson", "detail", "source_agent", "seen_at", "is_repeat"}]
    يُحدّد التكرار: نفس بصمة الدرس (النمط + الوكيل) يتكرر أكثر من مرة
    في الدفعة المعطاة (فشل متكرر داخل المهمة الحالية).
    لا يعتمد على ناقل ولا DB — نقية تمامًا.
    """
    now = datetime.now(timezone.utc).timestamp()
    counts: Dict[str, int] = {}
    insights: List[Dict[str, Any]] = []
    for ev in failure_events(events):
        detail = ev.get("detail") or ""
        if len(str(detail)) < MIN_ERROR_DETAIL_CHARS:
            continue
        agent_id = ev.get("agent_id") or "unknown"
        meta = ev.get("metadata") or {}
        # إن كان الاستدعاء المطلق أرفق task في metadata — نستخدمه للمجال
        task_hint = meta.get("task") or ev.get("title") or detail
        cls = classify_error(detail)
        fp = _lesson_fingerprint(cls["category"], agent_id, detail)
        counts[fp] = counts.get(fp, 0) + 1
        insights.append({
            "category": cls["category"],
            "lesson": cls["lesson"],
            "detail": str(detail)[:400],
            "source_agent": agent_id,
            "task_hint": task_hint,
            "seen_at": _now_iso(),
            "ts": now,
            "seen_window_hours": SEEN_WINDOW_HOURS,
            "occurrence": counts[fp],
            "is_repeat": counts[fp] > 1,
        })
    return insights


def quality_for_insight(insight: Dict[str, Any]) -> float:
    """جودة ابتدائية للدرس: أساس الفشل ثم خصم تكرار مسقّف."""
    q = -0.5
    if insight.get("is_repeat"):
        repeat_level = min(int(insight.get("occurrence") or 1) - 1, 3)
        q = max(-0.95, q + repeat_level * REPEAT_QUALITY_DELTA)
    return q


# ── المزامنة مع الذاكرة الجماعية ─────────────────────────────────────────────

_fl_lock = threading.Lock()


def sync_failure_lessons(events: List[Dict[str, Any]],
                         memory: Optional[Any] = None) -> List[int]:
    """
    يحوّل بصائر فشل الجلسة إلى دروس في الذاكرة الجماعية (دون تكرار).
    يعيد قائمة lesson_id المسجلة (فارغة عند التعذر أو عدم وجود فشل).
    """
    registered: List[int] = []
    if not memory:
        try:
            from ai.collective_memory import CollectiveMemory
            memory = CollectiveMemory()
        except Exception as exc:
            logger.warning("failure_learning: no memory backend: %s", exc)
            return []
    insights = extract_failure_insights(events)
    with _fl_lock:
        for ins in insights:
            try:
                task = ins.get("task_hint") or ""
                domain = _extract_domain(task)
                fingerprint = _lesson_fingerprint(
                    ins["category"], ins["source_agent"], ins["detail"])
                _seen = _existing_fingerprint(memory, fingerprint)
                if _seen:
                    # الدرس موجود أصلًا في الذاكرة — نزوّل جودته أكثر بسبب
                    # التكرار الجديد بدل إنشاء مكرر (نفس البصمة)
                    try:
                        memory.vote(_seen, -1)
                    except Exception:
                        pass
                    continue
            except Exception:
                pass
            try:
                lid = memory.record_lesson(
                    domain=domain,
                    question_hint=fingerprint,
                    lesson=ins["lesson"],
                    evidence=ins["detail"],
                    source_agent=ins["source_agent"],
                    source_run_id="failure_learning",
                    quality=quality_for_insight(ins),
                )
                if lid:
                    registered.append(lid)
                    _safe_emit(
                        "failure_lesson_recorded",
                        agent_id=ins["source_agent"],
                        title="تعلّم الأخطاء",
                        status="fail",
                        detail=ins["lesson"][:160],
                        metadata={"category": ins["category"],
                                  "lesson_id": lid,
                                  "is_repeat": bool(ins.get("is_repeat"))},
                    )
            except Exception as exc:  # never break the sync
                logger.warning("failure_learning: sync one lesson failed: %s", exc)
    return registered


# ── إفادة بقية الوكلاء: تحذيرات للمهمة الجديدة ────────────────────────────────


def _recalled_rows(memory: Any, domain: str,
                   top_k: int) -> List[Dict[str, Any]]:
    """استرجاع مباشر من SQLite عبر الذاكرة الجماعية — يشمل الدروس السلبية
    (تحذيرات الفشل) التي يستبعدها recall() الافتراضي."""
    try:
        from ai.collective_memory import MAX_RECALL
    except Exception:
        MAX_RECALL = 5  # pragma: no cover
    top_k = top_k or MAX_RECALL
    import sqlite3
    db_path = getattr(memory, "db_path", None)
    if db_path is None:
        try:
            from ai.collective_memory import DB_PATH
            db_path = DB_PATH
        except Exception:
            db_path = None
    if db_path is None or not hasattr(memory, "_lock"):
        return []
    try:
        with memory._lock, sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute("""
                SELECT lesson_id, domain, question_hint, lesson,
                       source_agent, quality, task_hits, task_fails,
                       created_at, seen_at
                FROM collective_lessons
                WHERE (domain = ? OR domain = 'عام')
                  AND quality < 0.0
                ORDER BY quality ASC, task_fails DESC, seen_at DESC
                LIMIT ?
            """, (domain, top_k * 2)).fetchall()
    except Exception as exc:
        logger.warning("failure_learning: recalled_rows failed: %s", exc)
        return []
    from ai.collective_memory import MAX_RECALL as _mr
    out, seen_ids = [], set()
    for row in rows:
        if len(out) >= top_k:
            break
        lid = row[0]
        if lid in seen_ids:
            continue
        seen_ids.add(lid)
        out.append(dict(zip(
            ["lesson_id", "domain", "question_hint", "lesson",
             "source_agent", "quality", "task_hits", "task_fails",
             "created_at", "seen_at"], row)))
    return out[:_mr]


def failure_warnings_for_task(task: str, memory: Optional[Any] = None,
                              top_k: int = 3) -> List[Dict[str, Any]]:
    """تحذيرات فشل ذات صلة بمجال المهمة — تُحقَن في برومبت الوكلاء."""
    warnings: List[Dict[str, Any]] = []
    if not memory:
        try:
            from ai.collective_memory import CollectiveMemory
            memory = CollectiveMemory()
        except Exception as exc:
            logger.warning("failure_learning: no memory backend: %s", exc)
            return []
    domain = _extract_domain(task)
    try:
        warnings = _recalled_rows(memory, domain, top_k)
        _safe_emit(
            "failure_lesson_recalled",
            agent_id="failure_learning",
            title="تعلّم الأخطاء",
            status="done",
            detail=f"استرجاع {len(warnings)} تحذيرات فشل لمجال {domain}",
            metadata={"domain": domain, "count": len(warnings)},
        )
    except Exception as exc:
        logger.warning("failure_learning: recall failed: %s", exc)
    return warnings


def failure_warnings_prompt_text(warnings: List[Dict[str, Any]]) -> str:
    """تحويل التحذيرات إلى فقرة نصية عربية تُحقَن في برومبت التوليف."""
    if not warnings:
        return ""
    lines = ["⚠️ تحذيرات من أخطاء سابقة لوكلاء آخرين في هذا المجال:"]
    for i, w in enumerate(warnings, 1):
        src = w.get("source_agent") or "غير معروف"
        lines.append(f"{i}) من وكيل «{src}»: {w.get('lesson') or ''}")
    return "\n".join(lines)


def sync_and_warn(events: List[Dict[str, Any]], task: str,
                  top_k: int = 3) -> str:
    """مسار مختصر: يسجّل دروس فشل الجلسة ثم يعيد نص تحذيرات المهمة الجديدة."""
    try:
        from ai.collective_memory import CollectiveMemory
        memory = CollectiveMemory()
    except Exception:
        return ""
    try:
        sync_failure_lessons(events, memory=memory)
    except Exception:
        pass
    warnings = failure_warnings_for_task(task, memory=memory, top_k=top_k)
    return failure_warnings_prompt_text(warnings)
