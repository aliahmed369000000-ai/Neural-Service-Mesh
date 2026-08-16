"""
ذاكرة الأخطاء التكيفية (Reflection Memory) — تكملة لنظام التقييم الذاتي.

نظام التقييم الذاتي (agent_reflection.py) يُشخّص فشلًا واحدًا ويصنّفه في
اللحظة، لكنه لا يتذكّر شيئًا بعد انتهاء المهمة: إذا تكرر نفس الخطأ في
مهمة لاحقة (مثال: انقطاع مزوّد Groq عند كلمة "تحليل معمّق") سيُعيد نفس
دورة التشخيص من الصفر في كل مرة.

هذا الملف يضيف طبقة ذاكرة دائمة:
1) بعد انتهاء reflecting_call (نجاحًا أو استسلامًا) يُسجَّل "درس" صغير:
   توقيع الخطأ (أول 120 حرفًا من الرسالة) + السبب المصنّف + الاستراتيجية
   التي نجحت في النهاية.
2) في بداية كل محاولة جديدة يُقارَن توقيع الخطأ الحالي بالدروس المحفوظة؛
   إن وُجد درس سابق يطابقه يُطبَّق حلّه فورًا دون استهلاك دورة تشخيص،
   ويُرفع العدّاد (كم مرة نجح هذا الدرس) حتى تتقدم الدروس المجربة
   تاريخيًا على التخمينات الجديدة.

التخزين: JSON بسيط في `memory/reflection_memory.json` (إنشاء تلقائي،
تجاهل أي فشل كتابي — الذاكرة خيار تحسين لا يجوز أن تكسر المسار الأصلي).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory")
_MEMORY_FILE = os.path.join(_MEMORY_DIR, "reflection_memory.json")

# حدود الذاكرة حتى لا تتضخم وتبطّئ المطابقة في كل محاولة
_MAX_LESSONS = 200
_MAX_PER_SIGNATURE = 10
_SHORT_THRESHOLD = 120
_SIGNATURE_THRESHOLD = 0.85  # حد التشابه الأدنى للنص المختصر

# أنماط لا يُفيد حفظها لأنها عامة جدًا أو متغيّرة دائمًا
_NOISE_PATTERNS = ("empty", "فارغ", "تعذّر", "فشل", "خطأ")


class ReflectionMemory:
    """ذاكرة دروس الأخطاء مع مطابقة تواقيع وتقدّم تجريبي."""

    def __init__(self, path: str = _MEMORY_FILE) -> None:
        self.path = path
        self.lessons: List[Dict[str, Any]] = []
        self.load()

    # ── تخزين ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                data = raw.get("lessons", raw) if isinstance(raw, dict) else raw
                self.lessons = [l for l in data if isinstance(l, dict)]
        except Exception as exc:  # pragma: no cover - ذاكرة اختيارية
            logger.warning("reflection_memory: تعذّر التحميل — ذاكرة فارغة: %s", exc)
            self.lessons = []

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"lessons": self.lessons[-_MAX_LESSONS:]}, f, ensure_ascii=False, indent=1)
        except Exception as exc:  # pragma: no cover
            logger.warning("reflection_memory: تعذّر الحفظ: %s", exc)

    # ── توقيعات ─────────────────────────────────────────────────────────────

    @staticmethod
    def signature(error_message: str) -> str:
        """توقيع ثابت للخطأ: نص قصير بالأحرف الصغيرة."""
        text = (error_message or "").lower().strip()
        return text[:_SHORT_THRESHOLD]

    @classmethod
    def _signature_score(cls, a: str, b: str) -> float:
        """تشابه جاكارد تقريبي على كلمات التوقيعين."""
        wa, wb = set(a.split()), set(b.split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    # ── واجهة الاستخدام ─────────────────────────────────────────────────────

    def lookup(self, error_message: str) -> Optional[Dict[str, Any]]:
        """
        يبحث عن درس سابق يطابق الخطأ الحالي.
        يعيد أفضل درس مطابق (الأعلى نجاحًا تجريبيًا) أو None.
        """
        sig = self.signature(error_message)
        if not sig or sig in _NOISE_PATTERNS:
            return None
        best: Optional[Dict[str, Any]] = None
        best_score = _SIGNATURE_THRESHOLD
        for lesson in self.lessons:
            if sig == lesson.get("signature"):
                lesson["score"] = lesson.get("successes", 0)
                if best is None or lesson["score"] > best["score"]:
                    best = lesson
                    best_score = 2.0  # تطابق حرفي يتفوق دائمًا
                continue
            score = self._signature_score(sig, lesson.get("signature", ""))
            if score >= best_score and score > 0:
                lesson["score"] = lesson.get("successes", 0)
                best = lesson
                best_score = score
        if best is None:
            return None
        return {
            "lesson": best,
            "hint": (
                f"💡 درس محفوظ مسبقًا لهذا الخطأ (نجح {best.get('successes', 0)} مرة): "
                f"السبب «{best.get('reason', 'غير مصنف')}» — استخدم استراتيجية "
                f"«{best.get('strategy', 'إعادة المحاولة')}» مباشرة"
            ),
        }

    def record(self, error_message: str, reason: str, strategy: str,
               success: bool) -> None:
        """يحدّث الذاكرة بعد نتيجة محاولة: نجاح الدرس يتقدم بالعدّاد."""
        sig = self.signature(error_message)
        if not sig or sig in _NOISE_PATTERNS:
            return
        try:
            for lesson in self.lessons:
                if lesson.get("signature") == sig:
                    if success:
                        lesson["successes"] = int(lesson.get("successes", 0)) + 1
                        lesson["strategy"] = strategy or lesson.get("strategy", "")
                        lesson["reason"] = reason or lesson.get("reason", "")
                    else:
                        lesson["failures"] = int(lesson.get("failures", 0)) + 1
                    lesson["last_seen"] = _now_iso()
                    self.save()
                    return
            # درس جديد — نحفظه فقط إن كان الخطأ جديدًا فعلًا
            if len([l for l in self.lessons if l.get("signature") == sig]) >= _MAX_PER_SIGNATURE:
                return
            self.lessons.append({
                "signature": sig,
                "reason": reason or "unknown",
                "strategy": strategy or "retry_with_backoff",
                "successes": 1 if success else 0,
                "failures": 0 if success else 1,
                "first_seen": _now_iso(),
                "last_seen": _now_iso(),
            })
            self.save()
        except Exception as exc:  # pragma: no cover
            logger.warning("reflection_memory: تعذّر التسجيل: %s", exc)

    def summary(self) -> Dict[str, Any]:
        total = len(self.lessons)
        proven = [l for l in self.lessons if int(l.get("successes", 0)) >= 3]
        return {
            "total_lessons": total,
            "proven_lessons": len(proven),
            "top_lessons": sorted(self.lessons, key=lambda l: int(l.get("successes", 0)), reverse=True)[:5],
        }


def _now_iso() -> str:
    try:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
    except Exception:  # pragma: no cover
        return ""


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[ReflectionMemory] = None


def get_reflection_memory() -> ReflectionMemory:
    global _instance
    if _instance is None:
        _instance = ReflectionMemory()
    return _instance
