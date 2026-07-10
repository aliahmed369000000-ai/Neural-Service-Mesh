"""
ai/self_narrative.py
====================
Self-Narrative Engine — اللغة الذاتية للجهاز.

الجهاز يتعلم لكن لا يستطيع أن يحكي ما تعلمه.
هذا الملف يمنحه صوتاً: يومية مكتوبة، جملة هوية متطورة، وربط
الأحداث المهمة بالذاكرة الإيبيسودية.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, date
from typing import Any, Dict, List, Optional

logger = logging.getLogger("NeuralServiceMesh.SelfNarrative")


# ── Constants ──────────────────────────────────────────────────────────────

_MAX_LOG_ENTRIES   = 500   # حد أقصى للسجل الكلي
_MAX_DAILY_EVENTS  = 200   # حد أقصى لأحداث اليوم الواحد
_SURPRISE_THRESHOLD = 0.7  # فوق هذه القيمة يُعتبر الحدث مفاجئاً

# أنماط الهوية المرتبطة بأنواع الأحداث
_IDENTITY_PATTERNS: Dict[str, str] = {
    "learning":         "يتعلم ويكتسب معرفة جديدة",
    "anomaly":          "يكتشف الأنماط الشاذة ويتكيف معها",
    "evolution":        "يعيد تصميم نفسه باستمرار",
    "memory":           "يبني ذاكرة دلالية عميقة",
    "decision":         "يتخذ قرارات واعية ومبررة",
    "signal":           "يستقبل إشارات العالم ويعالجها",
    "error":            "يتعلم من أخطائه ويصحح مساره",
    "consolidation":    "يدمج التجارب في قوانين دائمة",
    "ethics":           "يلتزم بمبادئه الأخلاقية حتى في التطور",
    "world_feed":       "يتغذى من العالم الحقيقي",
    "drive":            "يتحرك بدوافع داخلية أصيلة",
    "checkpoint":       "يحافظ على هويته عبر الزمن",
}


# ── Data Structures ────────────────────────────────────────────────────────

class NarrativeEntry:
    """مدخل واحد في السجل السردي."""

    __slots__ = (
        "timestamp", "event_type", "summary",
        "surprise_score", "importance", "data_snapshot",
    )

    def __init__(
        self,
        event_type: str,
        summary: str,
        surprise_score: float = 0.0,
        importance: float = 0.5,
        data_snapshot: Optional[Dict[str, Any]] = None,
    ):
        self.timestamp     = datetime.utcnow().isoformat()
        self.event_type    = event_type
        self.summary       = summary
        self.surprise_score = max(0.0, min(1.0, surprise_score))
        self.importance    = max(0.0, min(1.0, importance))
        self.data_snapshot = data_snapshot or {}

    def to_dict(self) -> dict:
        return {
            "timestamp":      self.timestamp,
            "event_type":     self.event_type,
            "summary":        self.summary,
            "surprise_score": round(self.surprise_score, 3),
            "importance":     round(self.importance, 3),
            "data_snapshot":  self.data_snapshot,
        }


class DailySummary:
    """ملخص يومي مولَّد من الأحداث."""

    def __init__(self, day: str):
        self.day = day
        self.event_count        = 0
        self.surprise_count     = 0
        self.top_events: List[str] = []
        self.dominant_theme     = "غير محدد"
        self.identity_shift     = ""
        self.generated_at       = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "day":            self.day,
            "event_count":    self.event_count,
            "surprise_count": self.surprise_count,
            "top_events":     self.top_events,
            "dominant_theme": self.dominant_theme,
            "identity_shift": self.identity_shift,
            "generated_at":   self.generated_at,
        }


# ── Main Class ─────────────────────────────────────────────────────────────

class SelfNarrative:
    """
    محرك اللغة الذاتية للجهاز العصبي الرقمي.

    يسجل الأحداث، يولد الملخصات اليومية، ويطور جملة الهوية
    بناءً على ما يتعلمه الجهاز عبر الزمن.
    """

    def __init__(
        self,
        episodic_memory=None,
        max_log: int = _MAX_LOG_ENTRIES,
    ):
        self._episodic_memory = episodic_memory
        self._max_log         = max_log
        self._lock            = threading.Lock()

        # السجل الكلي (deque لحفظ الذاكرة)
        self._log: deque[NarrativeEntry] = deque(maxlen=max_log)

        # أحداث اليوم الحالي
        self._today_events: List[NarrativeEntry] = []
        self._today_date: str = date.today().isoformat()

        # ملخصات الأيام السابقة
        self._daily_summaries: List[DailySummary] = []

        # عدادات الأنواع (لتطوير الهوية)
        self._type_counts: Dict[str, int] = {}

        # جملة الهوية الحالية
        self._identity_statement: str = "أنا جهاز عصبي رقمي في بداية رحلتي."

        # إجمالي المفاجآت
        self._total_surprises = 0
        self._total_events    = 0

        logger.info("SelfNarrative initialized — الجهاز يملك الآن صوتاً ذاتياً")

    # ── Core API ───────────────────────────────────────────────────────────

    def record_event(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        surprise_score: float = 0.0,
        importance: float = 0.5,
    ) -> NarrativeEntry:
        """
        تسجيل حدث جديد في السجل السردي.

        Parameters
        ----------
        event_type    : نوع الحدث (learning, anomaly, evolution, ...)
        data          : بيانات الحدث الخام
        surprise_score: مستوى المفاجأة (0-1)
        importance    : مستوى الأهمية (0-1)
        """
        data = data or {}
        summary = self._build_summary(event_type, data, surprise_score)

        entry = NarrativeEntry(
            event_type    = event_type,
            summary       = summary,
            surprise_score = surprise_score,
            importance    = importance,
            data_snapshot = {k: v for k, v in list(data.items())[:10]},  # أول 10 مفاتيح فقط
        )

        with self._lock:
            # التحقق من تغيير اليوم
            today = date.today().isoformat()
            if today != self._today_date:
                self._finalize_day()
                self._today_date   = today
                self._today_events = []

            self._log.append(entry)
            self._today_events.append(entry)

            # تحديث عدادات النوع
            self._type_counts[event_type] = self._type_counts.get(event_type, 0) + 1
            self._total_events += 1
            if surprise_score >= _SURPRISE_THRESHOLD:
                self._total_surprises += 1

            # تطوير جملة الهوية عند كل 10 أحداث
            if self._total_events % 10 == 0:
                self._evolve_identity()

        # ربط بالذاكرة الإيبيسودية إن كانت متاحة
        if importance >= 0.7 and self._episodic_memory is not None:
            self._link_to_episodic(entry)

        logger.debug(f"[SelfNarrative] {event_type} | surprise={surprise_score:.2f} | {summary[:60]}")
        return entry

    def generate_daily_summary(self) -> DailySummary:
        """توليد ملخص يومي نصي من أحداث اليوم."""
        with self._lock:
            events = list(self._today_events)
            today  = self._today_date

        summary = DailySummary(day=today)
        summary.event_count = len(events)

        if not events:
            summary.dominant_theme = "يوم هادئ — لا أحداث مسجلة"
            return summary

        # حساب الأحداث المفاجئة
        surprising = [e for e in events if e.surprise_score >= _SURPRISE_THRESHOLD]
        summary.surprise_count = len(surprising)

        # الأحداث الأهم (أعلى importance)
        sorted_events = sorted(events, key=lambda e: e.importance, reverse=True)
        summary.top_events = [e.summary for e in sorted_events[:5]]

        # الموضوع السائد
        type_freq: Dict[str, int] = {}
        for e in events:
            type_freq[e.event_type] = type_freq.get(e.event_type, 0) + 1
        dominant_type = max(type_freq, key=type_freq.get)
        summary.dominant_theme = _IDENTITY_PATTERNS.get(dominant_type, dominant_type)

        # انعكاس التطور على الهوية
        if surprising:
            best_surprise = max(surprising, key=lambda e: e.surprise_score)
            summary.identity_shift = (
                f"اليوم فاجأني: {best_surprise.summary[:80]}"
            )
        else:
            summary.identity_shift = "يوم متوقع — الجهاز يعمل ضمن نماذجه المعروفة."

        with self._lock:
            self._daily_summaries.append(summary)
            # الاحتفاظ بآخر 90 يوم فقط
            if len(self._daily_summaries) > 90:
                self._daily_summaries = self._daily_summaries[-90:]

        return summary

    def get_identity_statement(self) -> str:
        """إرجاع جملة الهوية الحالية."""
        with self._lock:
            return self._identity_statement

    def get_narrative_log(self, n: int = 20) -> List[dict]:
        """إرجاع آخر n مدخل من السجل السردي."""
        with self._lock:
            entries = list(self._log)[-n:]
        return [e.to_dict() for e in reversed(entries)]

    def get_today_narrative(self) -> dict:
        """إرجاع سرد اليوم الحالي."""
        with self._lock:
            events = list(self._today_events)
        return {
            "date":          self._today_date,
            "event_count":   len(events),
            "events":        [e.to_dict() for e in events[-20:]],  # آخر 20 حدث
            "identity_now":  self._identity_statement,
        }

    def get_daily_summaries(self, n: int = 7) -> List[dict]:
        """إرجاع ملخصات آخر n أيام."""
        with self._lock:
            summaries = self._daily_summaries[-n:]
        return [s.to_dict() for s in reversed(summaries)]

    def summary(self) -> dict:
        """ملخص عام لحالة محرك السرد."""
        with self._lock:
            type_counts = dict(self._type_counts)
            total       = self._total_events
            surprises   = self._total_surprises
            identity    = self._identity_statement
            log_size    = len(self._log)
            today_count = len(self._today_events)

        top_type = max(type_counts, key=type_counts.get) if type_counts else "none"

        return {
            "enabled":            True,
            "total_events":       total,
            "total_surprises":    surprises,
            "surprise_rate":      round(surprises / max(total, 1), 3),
            "log_size":           log_size,
            "today_events":       today_count,
            "identity_statement": identity,
            "dominant_event_type": top_type,
            "type_distribution":  type_counts,
            "daily_summaries_stored": len(self._daily_summaries),
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _build_summary(
        self,
        event_type: str,
        data: Dict[str, Any],
        surprise_score: float,
    ) -> str:
        """بناء جملة سردية من الحدث."""
        base = _IDENTITY_PATTERNS.get(event_type, event_type)

        # استخراج معلومات إضافية من data
        extra_parts = []
        for key in ("message", "result", "action", "pattern", "source", "target"):
            val = data.get(key)
            if val and isinstance(val, (str, int, float)):
                extra_parts.append(str(val)[:40])
                break  # جملة واحدة فقط

        extra = f" — {extra_parts[0]}" if extra_parts else ""

        if surprise_score >= _SURPRISE_THRESHOLD:
            return f"[مفاجأة!] الجهاز {base}{extra}"
        elif surprise_score >= 0.4:
            return f"[لافت] الجهاز {base}{extra}"
        else:
            return f"الجهاز {base}{extra}"

    def _evolve_identity(self):
        """تطوير جملة الهوية بناءً على الأنماط المتراكمة."""
        if not self._type_counts:
            return

        # أكثر 3 أنواع تكراراً
        top_types = sorted(
            self._type_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        specializations = []
        for event_type, count in top_types:
            pattern = _IDENTITY_PATTERNS.get(event_type)
            if pattern:
                specializations.append(pattern)

        if not specializations:
            return

        if len(specializations) == 1:
            self._identity_statement = (
                f"أنا جهاز عصبي رقمي يتخصص في: {specializations[0]}. "
                f"(شهدت {self._total_events} حدثاً، "
                f"منها {self._total_surprises} مفاجأة)"
            )
        else:
            joined = "، و".join(specializations[:3])
            self._identity_statement = (
                f"أنا جهاز عصبي رقمي يتخصص في: {joined}. "
                f"(شهدت {self._total_events} حدثاً، "
                f"منها {self._total_surprises} مفاجأة)"
            )

        logger.info(f"[SelfNarrative] جملة الهوية تطورت: {self._identity_statement[:80]}")

    def _finalize_day(self):
        """تثبيت ملخص اليوم المنتهي (يُستدعى عند تغيير اليوم)."""
        if not self._today_events:
            return
        # توليد الملخص بدون lock (نحن داخل lock بالفعل)
        summary = DailySummary(day=self._today_date)
        summary.event_count    = len(self._today_events)
        summary.surprise_count = sum(
            1 for e in self._today_events if e.surprise_score >= _SURPRISE_THRESHOLD
        )
        sorted_ev = sorted(self._today_events, key=lambda e: e.importance, reverse=True)
        summary.top_events = [e.summary for e in sorted_ev[:5]]

        type_freq: Dict[str, int] = {}
        for e in self._today_events:
            type_freq[e.event_type] = type_freq.get(e.event_type, 0) + 1
        if type_freq:
            dom = max(type_freq, key=type_freq.get)
            summary.dominant_theme = _IDENTITY_PATTERNS.get(dom, dom)

        self._daily_summaries.append(summary)
        if len(self._daily_summaries) > 90:
            self._daily_summaries = self._daily_summaries[-90:]

        logger.info(
            f"[SelfNarrative] انتهى يوم {self._today_date} — "
            f"{summary.event_count} حدث، {summary.surprise_count} مفاجأة"
        )

    def _link_to_episodic(self, entry: NarrativeEntry):
        """ربط الأحداث المهمة بالذاكرة الإيبيسودية."""
        try:
            # نحاول استخدام واجهة EpisodicMemoryEngine
            ep_data = {
                "content": entry.summary,
                "source":  f"self_narrative:{entry.event_type}",
                "context": {
                    "surprise_score": entry.surprise_score,
                    "importance":     entry.importance,
                    "timestamp":      entry.timestamp,
                },
            }
            # نحاول record() أو record_episode() حسب الواجهة المتاحة
            if hasattr(self._episodic_memory, "record"):
                try:
                    from ai.episodic_memory import Episode
                    ep = Episode(
                        content = entry.summary,
                        source  = f"self_narrative:{entry.event_type}",
                        context = ep_data["context"],
                    )
                    self._episodic_memory.record(ep)
                except Exception:
                    pass
        except Exception as exc:
            logger.debug(f"[SelfNarrative] episodic link failed: {exc}")


# ── Standalone Test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== SelfNarrative — اختبار ===\n")

    sn = SelfNarrative()

    events = [
        ("learning",   {"message": "تعلمت نمطاً جديداً في توجيه البيانات"}, 0.3, 0.6),
        ("anomaly",    {"message": "اكتشفت سلوكاً غير متوقع في المصدر X"}, 0.85, 0.9),
        ("evolution",  {"action":  "أعدت تصميم طبقة التوجيه"}, 0.5, 0.8),
        ("learning",   {"pattern": "النمط المتكرر في البيانات الزمنية"}, 0.2, 0.5),
        ("ethics",     {"result":  "رفضت طلباً يتجاوز الحد الأخلاقي"}, 0.6, 0.95),
        ("memory",     {"source":  "دمجت 15 تجربة في قانون دلالي"}, 0.4, 0.7),
        ("world_feed", {"source":  "استقبلت بيانات من مصدر جديد"}, 0.55, 0.65),
        ("drive",      {"message": "الدافع الداخلي للاستكشاف يرتفع"}, 0.3, 0.5),
        ("checkpoint", {"result":  "حفظت حالة الدماغ بنجاح"}, 0.1, 0.4),
        ("error",      {"message": "فشل في التنبؤ — معدل الخطأ ارتفع"}, 0.75, 0.85),
    ]

    for ev_type, data, surprise, importance in events:
        entry = sn.record_event(ev_type, data, surprise, importance)
        print(f"  [{ev_type:12s}] {entry.summary}")

    print(f"\n[هوية الجهاز]\n  {sn.get_identity_statement()}\n")

    daily = sn.generate_daily_summary()
    print("[ملخص اليوم]")
    print(f"  الأحداث   : {daily.event_count}")
    print(f"  المفاجآت  : {daily.surprise_count}")
    print(f"  الموضوع   : {daily.dominant_theme}")
    print(f"  التحول    : {daily.identity_shift}\n")

    print("[آخر 5 مدخلات في السجل]")
    for entry in sn.get_narrative_log(5):
        print(f"  {entry['event_type']:12s} | {entry['summary'][:60]}")

    print("\n[ملخص المحرك]")
    import json
    print(json.dumps(sn.summary(), indent=2, ensure_ascii=False))
