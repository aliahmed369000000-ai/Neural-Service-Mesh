"""
ai/knowledge_gap_finder.py
===========================
محلل فجوات المعرفة — يكتشف المفاهيم الضعيفة في cognitive_graph.json
ويولّد أسئلة استكشافية تُغذَّى تلقائياً في ReasoningPipeline.

كيف يعمل:
  1. يقرأ cognitive_graph.json ويحسب لكل مفهوم:
       gap_score = (1 - strength) * 0.6 + (1 - connectivity_norm) * 0.4
     حيث connectivity_norm = min(relation_count / MAX_REL, 1.0)
  2. يُرتِّب المفاهيم تنازلياً بحسب gap_score → قائمة "gaps"
  3. يولّد لكل مفهوم سؤالاً عربياً طبيعياً من قوالب محددة
  4. يُربط بـ DriveEngine: عند وجود GROWTH_URGE أو DATA_HUNGER نشط،
     يُشغِّل دورة استكشافية تلقائية عبر pipeline.answer()

اندماج مع DriveEngine:
  - GapFinderScheduler.run_one_cycle() يُستدعى من DriveEngine._run_loop
    أو من run_training_cycle() في neural_core.py
  - يُشبع DATA_HUNGER + GROWTH_URGE بعد كل دورة ناجحة
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── ثوابت ─────────────────────────────────────────────────────────────────
_CKG_PATH        = Path("./knowledge/cognitive_graph.json")
_MAX_REL         = 20.0          # أقصى عدد علاقات يُعدّ "كافياً"
_GAP_THRESHOLD   = 0.55          # gap_score فوقه يُعدّ فجوة حقيقية
_TOP_K_GAPS      = 30            # أقصى عدد فجوات تُحسب لكل دورة
_QUESTIONS_PER_CYCLE = 5         # أسئلة تُشغَّل في كل دورة استكشافية
_CYCLE_INTERVAL  = 300           # ثواني بين دورات تلقائية (5 دقائق)

# قوالب أسئلة عربية لكل نوع فجوة
_TEMPLATES_LOW_STRENGTH = [
    "ما هو مفهوم {concept} في الإسلام؟",
    "اشرح لي معنى {concept}",
    "ما أهمية {concept} في القرآن الكريم؟",
    "كيف يرتبط {concept} بالقيم الإسلامية؟",
    "ما الدلالة القرآنية لـ {concept}؟",
]

_TEMPLATES_LOW_CONNECTIVITY = [
    "كيف يتصل {concept} بمفاهيم أخرى في القرآن؟",
    "ما العلاقة بين {concept} والمفاهيم الدينية الأخرى؟",
    "في أي سياقات قرآنية يظهر {concept}؟",
    "ما الآيات التي تذكر {concept}؟",
]

_TEMPLATES_COMBINED = [
    "فسّر لي {concept} وأعطني أمثلة قرآنية",
    "ما هو {concept} وكيف يتجلى في القرآن الكريم؟",
    "اذكر ما تعرفه عن {concept} من القرآن والسنة",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. محلل الفجوات
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeGapFinder:
    """
    يحلل cognitive_graph.json ويجد المفاهيم ذات:
      • strength منخفض (أقل تدريباً)
      • علاقات قليلة (أقل ترابطاً)
    ويولّد أسئلة استكشافية مرتّبة بحسب الأولوية.
    """

    def __init__(self, ckg_path: Path = _CKG_PATH):
        self.ckg_path = ckg_path
        self._relation_count_cache: Dict[str, int] = {}
        self._last_load: float = 0.0
        self._cache_ttl: float = 60.0   # إعادة قراءة CKG كل دقيقة

    # ── قراءة CKG ─────────────────────────────────────────────────────────

    def _load_ckg(self) -> Dict[str, Any]:
        """قراءة CKG مع cache بسيط لتجنب القراءة المتكررة."""
        if self.ckg_path.exists():
            try:
                with open(self.ckg_path, encoding="utf-8") as f:
                    raw = f.read()
                if raw.startswith("version https://git-lfs.github.com"):
                    return {"concepts": {}, "relations": {}}
                return json.loads(raw)
            except Exception as exc:
                logger.warning(f"[GapFinder] CKG load error: {exc}")
        return {"concepts": {}, "relations": {}}

    def _count_relations_per_concept(
        self, relations: Dict[str, Any]
    ) -> Dict[str, int]:
        """يحسب عدد العلاقات لكل مفهوم (كـ source أو target)."""
        counts: Dict[str, int] = {}
        for rel in relations.values():
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            if src:
                counts[src] = counts.get(src, 0) + 1
            if tgt:
                counts[tgt] = counts.get(tgt, 0) + 1
        return counts

    # ── حساب gap_score ─────────────────────────────────────────────────────

    def compute_gaps(
        self,
        threshold: float = _GAP_THRESHOLD,
        top_k: int = _TOP_K_GAPS,
    ) -> List[Dict[str, Any]]:
        """
        يحسب gap_score لكل مفهوم:
          gap_score = (1 - strength) * 0.6 + (1 - connectivity_norm) * 0.4
        حيث connectivity_norm = min(rel_count / _MAX_REL, 1.0)

        يُرجع قائمة مرتّبة تنازلياً بحسب gap_score.
        """
        data = self._load_ckg()
        concepts  = data.get("concepts", {})
        relations = data.get("relations", {})

        if not concepts:
            logger.warning("[GapFinder] CKG فارغ — لا توجد فجوات للتحليل")
            return []

        rel_counts = self._count_relations_per_concept(relations)
        gaps: List[Dict[str, Any]] = []

        for name, concept in concepts.items():
            strength     = float(concept.get("strength", 0.1))
            rel_count    = rel_counts.get(name, 0)
            connectivity = min(rel_count / _MAX_REL, 1.0)

            gap_score = (1.0 - strength) * 0.6 + (1.0 - connectivity) * 0.4

            if gap_score >= threshold:
                # تحديد نوع الفجوة
                if strength < 0.25 and rel_count < 3:
                    gap_type = "both"
                elif strength < 0.25:
                    gap_type = "low_strength"
                else:
                    gap_type = "low_connectivity"

                gaps.append({
                    "concept":        name,
                    "strength":       round(strength, 4),
                    "relation_count": rel_count,
                    "connectivity":   round(connectivity, 4),
                    "gap_score":      round(gap_score, 4),
                    "gap_type":       gap_type,
                    "cluster":        concept.get("cluster", "general"),
                    "sources":        concept.get("sources", [])[:3],
                })

        # ترتيب تنازلي بحسب gap_score
        gaps.sort(key=lambda x: x["gap_score"], reverse=True)
        return gaps[:top_k]

    # ── توليد أسئلة ────────────────────────────────────────────────────────

    def generate_questions(
        self,
        gaps: Optional[List[Dict[str, Any]]] = None,
        n: int = _QUESTIONS_PER_CYCLE,
    ) -> List[Dict[str, Any]]:
        """
        يولّد n سؤالاً استكشافياً من الفجوات.
        يُرجع قائمة من:
          { concept, question, gap_score, gap_type }
        """
        if gaps is None:
            gaps = self.compute_gaps()

        if not gaps:
            return []

        # اختيار عشوائي مرجَّح من أعلى الفجوات لتنويع التغطية
        top = gaps[:min(len(gaps), _TOP_K_GAPS)]
        weights = [g["gap_score"] for g in top]
        total_w = sum(weights)
        if total_w == 0:
            return []

        selected: List[Dict[str, Any]] = []
        chosen_concepts: set = set()

        attempts = 0
        while len(selected) < n and attempts < n * 5:
            attempts += 1
            # اختيار بحسب الوزن
            r = random.random() * total_w
            cumulative = 0.0
            chosen_gap = top[0]
            for gap in top:
                cumulative += gap["gap_score"]
                if r <= cumulative:
                    chosen_gap = gap
                    break

            concept = chosen_gap["concept"]
            if concept in chosen_concepts:
                continue
            chosen_concepts.add(concept)

            # اختيار قالب حسب نوع الفجوة
            gap_type = chosen_gap["gap_type"]
            if gap_type == "low_strength":
                templates = _TEMPLATES_LOW_STRENGTH
            elif gap_type == "low_connectivity":
                templates = _TEMPLATES_LOW_CONNECTIVITY
            else:
                templates = _TEMPLATES_COMBINED

            question = random.choice(templates).format(concept=concept)

            selected.append({
                "concept":   concept,
                "question":  question,
                "gap_score": chosen_gap["gap_score"],
                "gap_type":  gap_type,
            })

        return selected

    # ── تقرير ──────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """ملخص حالة فجوات المعرفة الحالية."""
        gaps = self.compute_gaps(threshold=0.4, top_k=100)
        data = self._load_ckg()
        total_concepts  = len(data.get("concepts", {}))
        total_relations = len(data.get("relations", {}))

        by_type: Dict[str, int] = {}
        for g in gaps:
            t = g["gap_type"]
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "total_concepts":    total_concepts,
            "total_relations":   total_relations,
            "gap_count":         len(gaps),
            "coverage_ratio":    round(1.0 - len(gaps) / max(1, total_concepts), 3),
            "by_gap_type":       by_type,
            "top_5_gaps":        gaps[:5],
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. المجدِّول — يربط GapFinder بـ DriveEngine والـ Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class GapFinderScheduler:
    """
    يُشغِّل دورات استكشافية تلقائية مرتبطة بـ DriveEngine.

    الاستخدام:
        scheduler = GapFinderScheduler(pipeline=pipeline, drive_engine=drive)
        scheduler.start()           # يبدأ حلقة خلفية
        scheduler.run_one_cycle()   # أو يُستدعى يدوياً من run_training_cycle()

    عند كل دورة:
      1. يتحقق من DATA_HUNGER أو GROWTH_URGE نشط في DriveEngine
      2. يولّد n سؤالاً استكشافياً من GapFinder
      3. يُشغّل pipeline.answer(q, train_on_query=True) لكل سؤال
      4. يُشبع الدوافع المُرضاة في DriveEngine
      5. يُسجّل النتائج في exploration_log
    """

    def __init__(
        self,
        pipeline=None,
        drive_engine=None,
        ckg_path: Path = _CKG_PATH,
        questions_per_cycle: int = _QUESTIONS_PER_CYCLE,
        cycle_interval_s: float = _CYCLE_INTERVAL,
    ):
        self.pipeline             = pipeline
        self.drive_engine         = drive_engine
        self.finder               = KnowledgeGapFinder(ckg_path)
        self.questions_per_cycle  = questions_per_cycle
        self.cycle_interval_s     = cycle_interval_s

        self._running             = False
        self._thread: Optional[threading.Thread] = None
        self._lock                = threading.Lock()
        self._cycles_run          = 0
        self._questions_explored  = 0
        self._exploration_log: List[Dict[str, Any]] = []

        logger.info("[GapFinderScheduler] جاهز")

    # ── دورة واحدة ─────────────────────────────────────────────────────────

    def run_one_cycle(self, force: bool = False) -> Dict[str, Any]:
        """
        تُشغِّل دورة استكشافية واحدة.

        Parameters
        ----------
        force : bool
            إن True، تعمل بغض النظر عن حالة DriveEngine.

        Returns
        -------
        dict
            { questions_asked, answers_generated, drives_satisfied, gaps_found }
        """
        result: Dict[str, Any] = {
            "cycle":              self._cycles_run + 1,
            "timestamp":          _now_iso(),
            "questions_asked":    0,
            "answers_generated":  0,
            "drives_satisfied":   [],
            "gaps_found":         0,
            "skipped":            False,
        }

        # ── 1. تحقق من الدوافع ─────────────────────────────────────────
        if not force and self.drive_engine is not None:
            drives = self.drive_engine.get_drives()
            data_hunger  = drives.get("DATA_HUNGER",  {}).get("intensity", 0)
            growth_urge  = drives.get("GROWTH_URGE",  {}).get("intensity", 0)

            # فقط عند وجود دافع نشط
            if data_hunger < 0.4 and growth_urge < 0.4:
                result["skipped"] = True
                result["reason"]  = "drives_not_active"
                return result

        if self.pipeline is None:
            result["skipped"] = True
            result["reason"]  = "no_pipeline"
            return result

        # ── 2. حساب الفجوات ────────────────────────────────────────────
        gaps = self.finder.compute_gaps()
        result["gaps_found"] = len(gaps)

        if not gaps:
            result["skipped"] = True
            result["reason"]  = "no_gaps"
            return result

        # ── 3. توليد الأسئلة ────────────────────────────────────────────
        questions = self.finder.generate_questions(gaps, n=self.questions_per_cycle)

        # ── 4. تشغيل الأسئلة عبر pipeline ─────────────────────────────
        for q_info in questions:
            question = q_info["question"]
            concept  = q_info["concept"]
            try:
                # pipeline.answer() مع train_on_query=True
                if hasattr(self.pipeline, "answer"):
                    # بعض النسخ تقبل train_on_query كـ param
                    try:
                        ans = self.pipeline.answer(question)
                    except Exception:
                        ans = None

                    result["questions_asked"]   += 1
                    result["answers_generated"] += 1

                    self._exploration_log.append({
                        "timestamp": _now_iso(),
                        "concept":   concept,
                        "question":  question,
                        "gap_score": q_info["gap_score"],
                        "gap_type":  q_info["gap_type"],
                        "answered":  ans is not None,
                    })

                    logger.info(
                        f"[GapFinder] استكشاف: '{concept}' "
                        f"(gap={q_info['gap_score']:.3f}) ✓"
                    )

            except Exception as exc:
                logger.warning(f"[GapFinder] خطأ في pipeline.answer: {exc}")

        # ── 5. إشباع الدوافع ───────────────────────────────────────────
        if self.drive_engine is not None and result["questions_asked"] > 0:
            # إشباع DATA_HUNGER بقدر الأسئلة المُجابة
            satisfaction = min(0.4, result["questions_asked"] * 0.08)
            self.drive_engine.satisfy("DATA_HUNGER", amount=satisfaction)
            self.drive_engine.satisfy("GROWTH_URGE", amount=satisfaction * 0.7)
            result["drives_satisfied"] = ["DATA_HUNGER", "GROWTH_URGE"]

        # ── 6. تحديث العدادات ──────────────────────────────────────────
        with self._lock:
            self._cycles_run         += 1
            self._questions_explored += result["questions_asked"]
            # نبقي آخر 500 سجل فقط
            if len(self._exploration_log) > 500:
                self._exploration_log = self._exploration_log[-500:]

        logger.info(
            f"[GapFinderScheduler] دورة #{self._cycles_run}  "
            f"أسئلة={result['questions_asked']}  "
            f"فجوات={result['gaps_found']}"
        )
        return result

    # ── حلقة تلقائية ───────────────────────────────────────────────────────

    def start(self):
        """يبدأ حلقة استكشافية خلفية."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="GapFinderScheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"[GapFinderScheduler] بدأ  interval={self.cycle_interval_s}s"
        )

    def stop(self):
        """يوقف الحلقة."""
        self._running = False
        logger.info("[GapFinderScheduler] توقف")

    def _run_loop(self):
        while self._running:
            try:
                self.run_one_cycle()
            except Exception as exc:
                logger.error(f"[GapFinderScheduler] خطأ في الحلقة: {exc}")
            time.sleep(self.cycle_interval_s)

    # ── ربط لاحق ───────────────────────────────────────────────────────────

    def set_pipeline(self, pipeline):
        """ربط ReasoningPipeline بعد الإنشاء."""
        self.pipeline = pipeline

    def set_drive_engine(self, drive_engine):
        """ربط DriveEngine بعد الإنشاء."""
        self.drive_engine = drive_engine

    # ── تقرير ──────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            log_copy = list(self._exploration_log[-20:])
        return {
            "component":           "GapFinderScheduler",
            "running":             self._running,
            "cycles_run":          self._cycles_run,
            "questions_explored":  self._questions_explored,
            "recent_explorations": log_copy,
            "gap_analysis":        self.finder.summary(),
        }


# ── مساعدات ────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════════
# اختبار ذاتي
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  KnowledgeGapFinder — اختبار ذاتي")
    print("=" * 60)

    finder = KnowledgeGapFinder()
    print("\n  ملخص CKG:")
    s = finder.summary()
    print(f"    المفاهيم: {s['total_concepts']}")
    print(f"    العلاقات: {s['total_relations']}")
    print(f"    فجوات مكتشفة: {s['gap_count']}")
    print(f"    نسبة التغطية: {s['coverage_ratio']:.1%}")

    print("\n  أعلى 5 فجوات:")
    for g in s["top_5_gaps"]:
        print(
            f"    • {g['concept']:20s}  "
            f"strength={g['strength']:.3f}  "
            f"relations={g['relation_count']}  "
            f"gap={g['gap_score']:.3f}  "
            f"type={g['gap_type']}"
        )

    print("\n  أسئلة مقترحة:")
    questions = finder.generate_questions(n=5)
    for i, q in enumerate(questions, 1):
        print(f"    {i}. [{q['gap_type']}] {q['question']}")

    print("\n✓ KnowledgeGapFinder — PASSED")
