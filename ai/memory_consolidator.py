"""
ai/memory_consolidator.py
=========================
Memory Consolidator — الدمج الدلالي للذاكرة.

يحول "رأيت هذا النمط 50 مرة" → "هذا قانون دائم في ذاكرتي الدلالية".
يعمل في background thread ليدمج التجارب المتكررة تلقائياً.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("NeuralServiceMesh.MemoryConsolidator")


# ── Constants ──────────────────────────────────────────────────────────────

_DEFAULT_THRESHOLD      = 10    # عدد التكرارات اللازمة لتحويل نمط لقانون
_DEFAULT_INTERVAL_MIN   = 15    # كل 15 دقيقة افتراضياً
_MAX_CONSOLIDATED_LAWS  = 500   # حد أقصى للقوانين المكتسبة
_MAX_PATTERN_HISTORY    = 2000  # حد الذاكرة الإيبيسودية المُفحوصة


# ── Data Structures ────────────────────────────────────────────────────────

class ConsolidatedLaw:
    """قانون مكتسب — نمط تكرر حتى أصبح قاعدة راسخة."""

    __slots__ = (
        "law_id", "pattern_key", "description",
        "occurrence_count", "confidence",
        "first_seen", "last_seen", "consolidated_at",
        "source_episodes_freed",
    )

    def __init__(
        self,
        pattern_key: str,
        description: str,
        occurrence_count: int,
        confidence: float,
        first_seen: str,
        last_seen: str,
        source_episodes_freed: int = 0,
    ):
        self.law_id               = hashlib.md5(pattern_key.encode()).hexdigest()[:12]
        self.pattern_key          = pattern_key
        self.description          = description
        self.occurrence_count     = occurrence_count
        self.confidence           = round(min(1.0, confidence), 4)
        self.first_seen           = first_seen
        self.last_seen            = last_seen
        self.consolidated_at      = datetime.utcnow().isoformat()
        self.source_episodes_freed = source_episodes_freed

    def to_dict(self) -> dict:
        return {
            "law_id":                self.law_id,
            "pattern_key":           self.pattern_key,
            "description":           self.description,
            "occurrence_count":      self.occurrence_count,
            "confidence":            self.confidence,
            "first_seen":            self.first_seen,
            "last_seen":             self.last_seen,
            "consolidated_at":       self.consolidated_at,
            "source_episodes_freed": self.source_episodes_freed,
        }


class ConsolidationReport:
    """تقرير دورة دمج واحدة."""

    def __init__(self):
        self.timestamp          = datetime.utcnow().isoformat()
        self.episodes_scanned   = 0
        self.patterns_found     = 0
        self.new_laws           = 0
        self.updated_laws       = 0
        self.episodes_freed     = 0
        self.duration_ms        = 0.0
        self.new_law_summaries: List[str] = []

    def to_dict(self) -> dict:
        return {
            "timestamp":          self.timestamp,
            "episodes_scanned":   self.episodes_scanned,
            "patterns_found":     self.patterns_found,
            "new_laws":           self.new_laws,
            "updated_laws":       self.updated_laws,
            "episodes_freed":     self.episodes_freed,
            "duration_ms":        round(self.duration_ms, 2),
            "new_law_summaries":  self.new_law_summaries[:10],
        }


# ── Main Class ─────────────────────────────────────────────────────────────

class MemoryConsolidator:
    """
    محرك الدمج الدلالي — يحول التجارب المتكررة إلى قوانين.

    يعمل بشكل مستقل في background thread، يفحص الذاكرة
    الإيبيسودية دورياً ويدمج الأنماط المتكررة في الذاكرة الدلالية.
    """

    def __init__(
        self,
        episodic_memory=None,
        pattern_threshold: int   = _DEFAULT_THRESHOLD,
        max_laws: int            = _MAX_CONSOLIDATED_LAWS,
    ):
        self._episodic_memory   = episodic_memory
        self._threshold         = pattern_threshold
        self._max_laws          = max_laws

        self._lock              = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running           = False
        self._stop_event        = threading.Event()

        # القوانين المكتسبة: pattern_key → ConsolidatedLaw
        self._laws: Dict[str, ConsolidatedLaw] = {}

        # تاريخ دورات الدمج
        self._consolidation_history: List[ConsolidationReport] = []

        # ذاكرة الأنماط المحلية (للعمل بدون episodic_memory)
        self._local_pattern_counts: Counter = Counter()
        self._local_pattern_meta: Dict[str, Dict] = {}

        # إحصاءات
        self._total_consolidations = 0
        self._total_laws_created   = 0
        self._total_episodes_freed = 0

        logger.info(
            f"MemoryConsolidator initialized — "
            f"threshold={pattern_threshold} | max_laws={max_laws}"
        )

    # ── Core API ───────────────────────────────────────────────────────────

    def consolidate(self) -> dict:
        """
        تشغيل دورة دمج واحدة (synchronous).

        Returns
        -------
        dict: تقرير الدورة
        """
        t0     = time.time()
        report = ConsolidationReport()

        episodes = self._fetch_episodes()
        report.episodes_scanned = len(episodes)

        if not episodes and not self._local_pattern_counts:
            report.duration_ms = (time.time() - t0) * 1000
            return report.to_dict()

        # استخراج الأنماط من الحلقات
        patterns = self._extract_patterns(episodes)
        report.patterns_found = len(patterns)

        # ضم الأنماط المحلية
        for pk, meta in self._local_pattern_meta.items():
            count = self._local_pattern_counts[pk]
            if pk not in patterns:
                patterns[pk] = {"count": count, **meta}
            else:
                patterns[pk]["count"] = max(patterns[pk].get("count", 0), count)

        # تحديد الأنماط التي تجاوزت العتبة
        qualifying = {
            pk: meta for pk, meta in patterns.items()
            if meta.get("count", 0) >= self._threshold
        }

        with self._lock:
            for pattern_key, meta in qualifying.items():
                count       = meta.get("count", self._threshold)
                description = meta.get("description", pattern_key)
                first_seen  = meta.get("first_seen", datetime.utcnow().isoformat())
                last_seen   = meta.get("last_seen",  datetime.utcnow().isoformat())

                if pattern_key in self._laws:
                    # تحديث قانون موجود
                    law = self._laws[pattern_key]
                    law.occurrence_count = max(law.occurrence_count, count)
                    law.last_seen        = last_seen
                    law.confidence       = min(1.0, count / (count + 10))
                    report.updated_laws += 1
                else:
                    # إنشاء قانون جديد
                    if len(self._laws) >= self._max_laws:
                        self._evict_weakest_law()

                    confidence = min(1.0, count / (count + 10))
                    freed = self._free_episodic_episodes(pattern_key, episodes)
                    report.episodes_freed += freed

                    law = ConsolidatedLaw(
                        pattern_key           = pattern_key,
                        description           = description,
                        occurrence_count      = count,
                        confidence            = confidence,
                        first_seen            = first_seen,
                        last_seen             = last_seen,
                        source_episodes_freed = freed,
                    )
                    self._laws[pattern_key] = law
                    report.new_laws += 1
                    report.new_law_summaries.append(
                        f"[{law.law_id}] {description[:60]} (×{count})"
                    )
                    self._total_laws_created  += 1
                    self._total_episodes_freed += freed

                    # نقل القانون للذاكرة الدلالية إن كانت متاحة
                    self._push_to_semantic(law)

            self._total_consolidations += 1

        # تنظيف الأنماط المحلية التي أصبحت قوانين
        for pk in qualifying:
            if pk in self._local_pattern_counts:
                del self._local_pattern_counts[pk]
                self._local_pattern_meta.pop(pk, None)

        report.duration_ms = (time.time() - t0) * 1000

        with self._lock:
            self._consolidation_history.append(report)
            if len(self._consolidation_history) > 100:
                self._consolidation_history = self._consolidation_history[-100:]

        logger.info(
            f"[Consolidator] دورة اكتملت — "
            f"نمط={report.patterns_found} | "
            f"قوانين_جديدة={report.new_laws} | "
            f"حلقات_محررة={report.episodes_freed} | "
            f"{report.duration_ms:.1f}ms"
        )
        return report.to_dict()

    def observe_pattern(
        self,
        pattern_key: str,
        description: str = "",
        metadata: Optional[dict] = None,
    ) -> None:
        """
        تسجيل ملاحظة نمط (للاستخدام بدون episodic_memory).

        Parameters
        ----------
        pattern_key  : مفتاح النمط الفريد
        description  : وصف النمط بالعربية أو الإنجليزية
        metadata     : بيانات إضافية
        """
        metadata = metadata or {}
        now = datetime.utcnow().isoformat()

        with self._lock:
            self._local_pattern_counts[pattern_key] += 1
            count = self._local_pattern_counts[pattern_key]

            if pattern_key not in self._local_pattern_meta:
                self._local_pattern_meta[pattern_key] = {
                    "description": description or pattern_key,
                    "first_seen":  now,
                    "last_seen":   now,
                    "count":       count,
                    **metadata,
                }
            else:
                self._local_pattern_meta[pattern_key]["last_seen"] = now
                self._local_pattern_meta[pattern_key]["count"]     = count

    def start(self, interval_minutes: float = _DEFAULT_INTERVAL_MIN) -> None:
        """تشغيل background thread للدمج الدوري."""
        with self._lock:
            if self._running:
                logger.warning("[Consolidator] يعمل بالفعل.")
                return
            self._running = True
            self._stop_event.clear()

        self._thread = threading.Thread(
            target   = self._loop,
            args     = (interval_minutes,),
            daemon   = True,
            name     = "MemoryConsolidatorThread",
        )
        self._thread.start()
        logger.info(f"[Consolidator] بدأ — كل {interval_minutes} دقيقة.")

    def stop(self) -> None:
        """إيقاف background thread."""
        with self._lock:
            if not self._running:
                return
            self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("[Consolidator] توقف.")

    def get_consolidated_laws(self, min_confidence: float = 0.0) -> List[dict]:
        """إرجاع القوانين المكتسبة مرتبة بالثقة تنازلياً."""
        with self._lock:
            laws = list(self._laws.values())
        filtered = [
            law.to_dict()
            for law in laws
            if law.confidence >= min_confidence
        ]
        return sorted(filtered, key=lambda l: l["confidence"], reverse=True)

    def get_law(self, pattern_key: str) -> Optional[dict]:
        """إرجاع قانون محدد."""
        with self._lock:
            law = self._laws.get(pattern_key)
        return law.to_dict() if law else None

    def get_recent_reports(self, n: int = 5) -> List[dict]:
        """إرجاع آخر n تقارير دمج."""
        with self._lock:
            reports = self._consolidation_history[-n:]
        return [r.to_dict() for r in reversed(reports)]

    def summary(self) -> dict:
        """ملخص عام لحالة المدمج."""
        with self._lock:
            total_laws   = len(self._laws)
            total_cons   = self._total_consolidations
            total_created = self._total_laws_created
            total_freed  = self._total_episodes_freed
            running      = self._running
            local_patterns = len(self._local_pattern_counts)

        # أعلى قانون ثقة
        laws = self.get_consolidated_laws()
        top_law = laws[0]["description"][:60] if laws else "لا قوانين بعد"

        return {
            "enabled":             True,
            "running":             running,
            "total_laws":          total_laws,
            "total_consolidations": total_cons,
            "total_laws_created":  total_created,
            "total_episodes_freed": total_freed,
            "local_patterns_tracked": local_patterns,
            "top_law":             top_law,
            "threshold":           self._threshold,
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _loop(self, interval_minutes: float):
        """حلقة الدمج الدوري في background thread."""
        while not self._stop_event.wait(timeout=interval_minutes * 60):
            if not self._running:
                break
            try:
                self.consolidate()
            except Exception as exc:
                logger.error(f"[Consolidator] خطأ في دورة الدمج: {exc}")

    def _fetch_episodes(self) -> List[dict]:
        """جلب الحلقات من الذاكرة الإيبيسودية."""
        if self._episodic_memory is None:
            return []
        try:
            # نحاول get_strongest_memories أو طريقة مشابهة
            if hasattr(self._episodic_memory, "get_strongest_memories"):
                memories = self._episodic_memory.get_strongest_memories(_MAX_PATTERN_HISTORY)
                if isinstance(memories, list):
                    return memories
            if hasattr(self._episodic_memory, "_working_mem"):
                return [e.to_dict() if hasattr(e, "to_dict") else e
                        for e in list(self._episodic_memory._working_mem)[:_MAX_PATTERN_HISTORY]]
        except Exception as exc:
            logger.debug(f"[Consolidator] لا يمكن جلب الحلقات: {exc}")
        return []

    def _extract_patterns(self, episodes: List[dict]) -> Dict[str, dict]:
        """استخراج الأنماط المتكررة من الحلقات."""
        pattern_data: Dict[str, Dict] = defaultdict(lambda: {
            "count":       0,
            "description": "",
            "first_seen":  "",
            "last_seen":   "",
        })

        for ep in episodes:
            # استخراج مفاتيح النمط من بنية الحلقة
            keys = self._episode_to_pattern_keys(ep)
            ts   = ep.get("timestamp", ep.get("recorded_at", datetime.utcnow().isoformat()))

            for pk, description in keys:
                pd = pattern_data[pk]
                pd["count"] += 1
                pd["description"] = description
                if not pd["first_seen"] or ts < pd["first_seen"]:
                    pd["first_seen"] = ts
                if not pd["last_seen"] or ts > pd["last_seen"]:
                    pd["last_seen"] = ts

        return dict(pattern_data)

    def _episode_to_pattern_keys(self, ep: dict) -> List[Tuple[str, str]]:
        """
        تحويل حلقة واحدة إلى قائمة (pattern_key, description).
        يستخلص المعنى من بنيات متعددة.
        """
        keys = []

        # نمط المصدر
        source = ep.get("source", "")
        if source and source not in ("real", "synthetic", ""):
            pk = f"source:{source}"
            keys.append((pk, f"بيانات متكررة من المصدر '{source}'"))

        # نمط المحتوى/النص
        content = str(ep.get("content", ep.get("message", ep.get("text", ""))))
        if content and len(content) > 5:
            # استخدام أول 30 حرف كمفتاح
            snippet = content[:30].strip().lower().replace(" ", "_")
            pk = f"content:{hashlib.md5(snippet.encode()).hexdigest()[:8]}"
            keys.append((pk, f"نمط نصي متكرر: '{content[:40]}'"))

        # نمط السياق
        ctx = ep.get("context", {})
        if isinstance(ctx, dict):
            for k in ("event_type", "action", "result", "pattern"):
                val = ctx.get(k)
                if val and isinstance(val, str) and len(val) > 2:
                    pk = f"ctx_{k}:{val[:20]}"
                    keys.append((pk, f"سياق متكرر [{k}='{val[:30]}']"))

        # نمط الهدف/الوجهة
        target = ep.get("target", ep.get("goal", ""))
        if target and isinstance(target, (int, float)) and not isinstance(target, bool):
            # نمط في نطاق الهدف
            bucket = round(float(target), 1)
            pk = f"target_bucket:{bucket}"
            keys.append((pk, f"هدف متكرر في النطاق {bucket}"))

        return keys[:4]  # أقصاه 4 أنماط لكل حلقة

    def _free_episodic_episodes(self, pattern_key: str, episodes: List[dict]) -> int:
        """
        تحرير مساحة من الذاكرة الإيبيسودية للحلقات الممثَّلة
        بقانون جديد (نبقي 20% فقط كعيّنات).
        """
        if self._episodic_memory is None:
            return 0

        try:
            matched = [
                ep for ep in episodes
                if any(pk == pattern_key for pk, _ in self._episode_to_pattern_keys(ep))
            ]
            to_free = matched[: max(0, len(matched) - max(2, len(matched) // 5))]

            freed = 0
            for ep in to_free:
                ep_id = ep.get("episode_id", ep.get("id"))
                if ep_id and hasattr(self._episodic_memory, "remove_episode"):
                    try:
                        self._episodic_memory.remove_episode(ep_id)
                        freed += 1
                    except Exception:
                        pass
            return freed
        except Exception as exc:
            logger.debug(f"[Consolidator] خطأ في تحرير الذاكرة: {exc}")
            return 0

    def _push_to_semantic(self, law: ConsolidatedLaw):
        """دفع القانون المكتسب للذاكرة الدلالية إن كانت متاحة."""
        if self._episodic_memory is None:
            return
        try:
            # نحاول add_semantic_rule() أو semantic_rules مباشرة
            if hasattr(self._episodic_memory, "add_semantic_rule"):
                self._episodic_memory.add_semantic_rule(
                    pattern = law.pattern_key,
                    law     = law.description,
                    confidence = law.confidence,
                )
            elif hasattr(self._episodic_memory, "semantic_rules"):
                self._episodic_memory.semantic_rules[law.pattern_key] = {
                    "description": law.description,
                    "confidence":  law.confidence,
                    "count":       law.occurrence_count,
                    "law_id":      law.law_id,
                }
        except Exception as exc:
            logger.debug(f"[Consolidator] لا يمكن الدفع للذاكرة الدلالية: {exc}")

    def _evict_weakest_law(self):
        """إزالة أضعف قانون لإفساح المجال (يُستدعى داخل lock)."""
        if not self._laws:
            return
        weakest_key = min(self._laws, key=lambda k: self._laws[k].confidence)
        removed = self._laws.pop(weakest_key)
        logger.debug(f"[Consolidator] أُزيل القانون الأضعف: {removed.law_id}")


# ── Standalone Test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== MemoryConsolidator — اختبار ===\n")

    mc = MemoryConsolidator(pattern_threshold=5)

    # محاكاة أنماط متكررة
    patterns = [
        ("routing:high_latency",       "زمن الاستجابة مرتفع في مسار التوجيه"),
        ("source:world_feed",          "بيانات متكررة من WorldFeed"),
        ("ctx_action:learn",           "سياق التعلم يتكرر باستمرار"),
        ("routing:high_latency",       "زمن الاستجابة مرتفع في مسار التوجيه"),
        ("source:world_feed",          "بيانات متكررة من WorldFeed"),
        ("routing:high_latency",       "زمن الاستجابة مرتفع في مسار التوجيه"),
        ("ctx_action:learn",           "سياق التعلم يتكرر باستمرار"),
        ("source:world_feed",          "بيانات متكررة من WorldFeed"),
        ("routing:high_latency",       "زمن الاستجابة مرتفع في مسار التوجيه"),
        ("ctx_action:learn",           "سياق التعلم يتكرر باستمرار"),
        ("routing:high_latency",       "زمن الاستجابة مرتفع في مسار التوجيه"),
        ("source:world_feed",          "بيانات متكررة من WorldFeed"),
        ("ctx_action:learn",           "سياق التعلم يتكرر باستمرار"),
        ("routing:high_latency",       "زمن الاستجابة مرتفع في مسار التوجيه"),
        ("source:world_feed",          "بيانات متكررة من WorldFeed"),
        ("rare_pattern",               "نمط نادر لا يتكرر كثيراً"),
    ]

    for pk, desc in patterns:
        mc.observe_pattern(pk, desc)

    print("[تشغيل دورة دمج]\n")
    report = mc.consolidate()

    print(f"  الحلقات المفحوصة  : {report['episodes_scanned']}")
    print(f"  الأنماط المكتشفة  : {report['patterns_found']}")
    print(f"  القوانين الجديدة  : {report['new_laws']}")
    print(f"  الحلقات المحررة   : {report['episodes_freed']}")
    print(f"  الزمن             : {report['duration_ms']:.2f} ms\n")

    print("[القوانين المكتسبة]")
    laws = mc.get_consolidated_laws()
    for law in laws:
        print(
            f"  [{law['law_id']}] {law['description'][:50]:50s} "
            f"× {law['occurrence_count']:3d}  ثقة={law['confidence']:.3f}"
        )

    print(f"\n[نمط نادر — لم يتحول لقانون: 'rare_pattern' = "
          f"{mc._local_pattern_counts.get('rare_pattern', 0)} مرة < {mc._threshold}]\n")

    print("[ملخص المحرك]")
    import json
    print(json.dumps(mc.summary(), indent=2, ensure_ascii=False))
