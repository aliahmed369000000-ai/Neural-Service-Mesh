# -*- coding: utf-8 -*-
"""
ai/pre_action_reasoning.py — التفكير ما قبل الفعل (Pre-Action Reasoning)
═══════════════════════════════════════════════════════════════════════
قبل أن ينفّذ أي وكيل "فعلًا" (خطوة مهمة طويلة الأمد أو دور في مهمة
تعاونية) يفكّر أولًا:
  1. يستخرج الخطوات المتوقعة من الهدف/الخطة (تحليل أفعال محلي بلا LLM).
  2. يقدّر مخاطر كل خطوة (استدعاء خارجي = متوسط، استدعاءات خارجية
     مركّزة أو زمن طويل = عالي، داخلي بحت = منخفض).
  3. يقترح بدائل أكثر أمانًا للخطوات عالية الخطورة.
  4. يحسب ثقة جماعية ويصدر حكمًا: proceed (نفّذ) أو revise (أضف خطوة
     فحص أولية / استبدل أداة).
  5. يسجّل كل جلسة تفكير في قاعدة SQLite محلية دائمة لتراكم الخبرة.

الحلقة الاستدراكية (Learning from Actual Outcomes):
  بعد اكتمال أي مهمة، تُصنَّف النتيجة الفعلية (success/failure) وتُربط
  بجلسة التفكير السابقة لنفس task_id عبر learn(): يُسجَّل outcome في
  سجل الجلسة، وتُقارن النتيجة بقرارها (proceed/revise) لقياس دقة
  التوقعات وتغذية الذاكرة الحسية بسوابق معلومة النتائج (recalled
  memories ذات outcome حقيقي تزن أكثر).

لا يستدعي أي API خارجي — التحليل نمطي محلي بالكامل (مثل agent_reflection).
التدهور: فشل كامل → كتلة _PAR_OK في app_core تعطل الوحدة بصمت.

الجداول:
  par_records: id, task_id, role, goal, steps_json, risks_json, verdict,
               confidence, revised_n, outcome, created_at

الذاكرة الحسية للأدوار (Role Sensory Memory):
  قبل جلسة التفكير، يستحضر المحرك جلسات التفكير السابقة لنفس الدور
  أو لأهداف متشابهة لفظيًا، ويستخدمها لرفع الثقة عند وجود سوابق
  ناجحة (proceed) أو خفضها عند تكرار المراجعات (revise)، وترجع
  الذكريات المستحضرة مع نتيجة الجلسة في حقل recalled_memories.

استحضار الدقة التاريخية (Historical Accuracy Calibration):
  قبل كل جلسة تفكير يستحضر المحرك دقة التوقعات السابقة لنفس الدور
  (سجل outcome المقيس)، ويستخدمها لمعايرة ثقة الجلسة: دورٌ كانت
  توقعاته صحيحة تاريخًا (ثلثان أو أكثر) تُرفع ثقته بحد +0.05، ودورٌ
  كثير الأخطاء (أقل من نصف توقعاته صحيحة) تُخفَّض ثقته بحد -0.075 —
  أي دور بين ذلك تبقى ثقته بلا معايرة. المعايرة لا تحدث إلا بعد 3
  جلسات مقاسة للدور على الأقل (حتى لا تتأثر الثقة بعينة صغيرة),
  ومحصورة دائما بحد آمن صارم ±0.15 مهما تعددت المعايرات. تُعرض دقة
  كل دور في لوحة المراقبة لتُفهم ثقة الفريق لحظة بلحظة.

التوقع المتعدد المسارات (Multi-Path Forecasting):
  قبل أي فعل يمكن عرض عدة خطط بديلة للهدف نفسه (candidate paths)؛
  فيجري المحرك جلسة تفكير مستقلة لكل مسار (تُحفظ جميعها في السجل),
  ثم يصنّف كل مسار مقابل السوابق المقاسة تاريخيا للدور: كل خطوة
  تتشابه مع جلسة صحيحة ترفع المسار (+0.05), ومع جلسة خاطئة تخفضه
  (−0.08), والأثر محصور بحد صارم ±0.15. يختار المحرك المسار الأعلى
  ثقة بعد كل المعايرات ويعيد MultiPathPlan بكل المسارات ومؤشر
  المسار المختار.

حُكم الفريق على المسارات (Multi-Role Path Voting):
  يوزّع المحرك الخطط البديلة على أدوار الفريق؛ كل دور يجري جلسة
  تفكير مستقلة لكل مسار (مسلك حفظ: task_id::role::pathN)، ثم تجمع
  الأحكام بوزن الدقة التاريخية للدور (≥2/3 → 1.5، <1/2 → 0.5) مع
  مكافأة توافق +0.03 وتأثير تاريخي نمطي على المسار، ويختار المسار
  الأعلى توافقًا وثقةً — حد صارم ±0.15 على كل أثر، وتساقط آمن
  كامل عند غياب الأدوار أو تعطل دور أو فشل الكل.

التوقع الجماعي المتطور (Collective Resolution):
  بعد جمع أحكام الأدوار، يحاكي المحرك سيناريو تعارضٍ افتراضي بين
  الأحكام (كل حكم ينحرف عن متوسط مسار مساره بمقدار > 0.10 يعد
  تعارضًا محسوبًا رياضيًا — بلا أي نموذج لغوي)، ويولّد خطة واحدة
  موسّعة تدمج أفضل خطوات كل مسار: المسار الأعلى تصويتًا أولاً ثم
  خطوات المسارات الأخرى غير المكررة، وكل خطوة تحمل مصدرها ووزنها
  وثقتها. تحفظ الخطة المدمجة جلسةً واحدة في السجل
  (task_id::collective) بثقة محصورة [0,1] ومكافأة دمج ≤ +0.15.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOCAL_DB = "data/pre_action_reasoning.db"
_MAX_RECORDS = 400
_MAX_GOAL_CHARS = 400

# درجة مخاطر خطوة حسب نوع الأداة/النشاط المتوقع
_HIGH_RISK_HINTS = (
    "api_call", "llm_call", "api", "llm", "web_fetch",
    "external", "payment", "write_file", "deploy",
)
_MEDIUM_RISK_HINTS = (
    "web_search", "fetch", "search", "download", "scrape",
    "send_email", "notify", "broadcast",
)

# أنماط أفعال عربية وإنجليزية تشير لخطوة متوقعة في التنفيذ
_ACTION_PATTERNS = (
    ("ابحث", "بحث"), ("اجمع", "تجميع"), ("حلل", "تحليل"),
    ("اكتب", "كتابة"), ("اقرأ", "قراءة"), ("احسب", "حساب"),
    ("قارن", "مقارنة"), ("تحقق", "تدقيق"), ("صمم", "تصميم"),
    ("نفذ", "تنفيذ"), ("ركّب", "تركيب"), ("نشر", "نشر"),
    ("لخص", "تلخيص"), ("ارسم", "رسم"), ("قدّر", "تقدير"),
    ("search", "search"), ("analyze", "analysis"), ("compute", "compute"),
    ("write", "write"), ("verify", "verification"), ("deploy", "deploy"),
    ("fetch", "fetch"), ("collect", "collect"), ("compare", "comparison"),
)
# أدوات/كلمات تلمّح لنشاط خارجي
_EXTERNAL_HINTS = ("web_search", "web_fetch", "api_call", "fetch",
                   "llm_call", "search", "download", "scrape",
                   "send_email", "notify", "deploy")
_INTERNAL_HINTS = ("compute", "calc", "aggregate", "summarize", "format",
                   "internal", "finalize")

_PAR_MIN_CONFIDENCE = 0.55   # أقل ثقة تسمح بـ proceed بلا تعديل
_CONFIDENCE_LOW = 0.45       # الثقة تحت هذا الحد → revise حتمًا


class ReasoningPlan:
    """نتيجة جلسة تفكير ما قبل الفعل."""

    def __init__(self, goal: str, role: str = "") -> None:
        self.goal = goal
        self.role = role
        self.expected_steps: List[Dict[str, Any]] = []
        self.risks: List[Dict[str, Any]] = []
        self.confidence: float = 0.0
        self.base_confidence: float = 0.0
        self.recalled_memories: List[Dict[str, Any]] = []
        self.memory_effect: float = 0.0  # أثر الذاكرة على الثقة (+/-)
        self.calibration_effect: float = 0.0  # معايرة الدقة التاريخية (+/-)
        self.historical_accuracy: float = 0.0  # دقة الدور التاريخية المستحضرة
        self.historical_n: int = 0  # عدد السجلات المقاسة للدور
        self.verdict: str = "proceed"  # proceed | revise
        self.revisions: List[str] = []
        self.created_at: float = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal, "role": self.role,
            "expected_steps": self.expected_steps,
            "risks": self.risks,
            "confidence": round(self.confidence, 3),
            "base_confidence": round(self.base_confidence, 3),
            "memory_effect": round(self.memory_effect, 3),
            "calibration_effect": round(self.calibration_effect, 3),
            "historical_accuracy": round(self.historical_accuracy, 3),
            "historical_n": self.historical_n,
            "recalled_memories": self.recalled_memories,
            "verdict": self.verdict,
            "revisions": self.revisions,
            "created_at": self.created_at,
        }


class MultiPathPlan:
    """مقارنة عدة خطط بديلة للهدف نفسه قبل الفعل (Multi-Path Forecasting).

    يُحسب لكل مسار جلسة تفكير مستقلة (تحفظ في السجل)، ثم تُصنَّف خطوات
    المسار مقابل السوابق المقاسة تاريخًا للدور: خطوات متشابهة لفظًا مع
    جلسات صحيحة ترفع المسار، ومع جلسات خاطئة تخفضه، بحد صارم ±0.15.
    يختار المحرك المسار الأعلى ثقةً تاريخًا ويعيد كل الجلسات داخل
    MultiPathPlan مع المسار المختار مؤشَّرًا في recalled_memories.
    """

    def __init__(self, task_id: str, goal: str, role: str = "") -> None:
        self.task_id = task_id
        self.goal = goal
        self.role = role
        self.plans: List[ReasoningPlan] = []  # جلسة تفكير لكل مسار
        self.history_scores: List[float] = []  # أثر التشابه التاريخي لكل مسار
        self.chosen_index: int = 0  # مسار المسار المختار
        self.created_at: float = time.time()

    @property
    def chosen_plan(self) -> Optional[ReasoningPlan]:
        return self.plans[self.chosen_index] if self.plans else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id, "goal": self.goal, "role": self.role,
            "n_paths": len(self.plans),
            "history_scores": [round(s, 3) for s in self.history_scores],
            "chosen_index": self.chosen_index,
            "chosen_confidence": (round(self.chosen_plan.confidence, 3)
                                  if self.chosen_plan else 0.0),
            "paths": [p.to_dict() for p in self.plans],
            "created_at": self.created_at,
        }


class MultiRolePlan:
    """التوقع المتعدد المسارات عبر أدوار الفريق (Multi-Role Path Voting).

    يوزّع الخطط البديلة على أدوار متخصصة: لكل دور تجري جلسة تفكير
    مستقلة لكل مسار (كل جلسة تحفظ في السجل باسمها المسلكي
    task_id::role::path{i} مع ذاكرتها الحسية ومعايرتها التاريخية
    وأثرها النمطي)، ثم يجمع المحرك الأحكام عبر الأدوار:
    لكل مسار متوسط مرجّح بالثقة التاريخية للدور + مكافأة توافق
    (أدوار متقاربة الآراء) + أثر تاريخي على المسار نفسه — بحد صارم
    ±0.15 على كل معايرة — ويختار المسار الأعلى توافقًا وثقةً.
    """

    def __init__(self, task_id: str, goal: str,
                 roles: Optional[List[str]] = None) -> None:
        self.task_id = task_id
        self.goal = goal
        self.roles: List[str] = roles or []
        self.multi_path: Optional[MultiPathPlan] = None
        # {role: {path_idx: ReasoningPlan}}
        self.role_judgments: Dict[str, Dict[int, ReasoningPlan]] = {}
        # {role: {n, correct, accuracy}}
        self.role_accuracies: Dict[str, Dict[str, Any]] = {}
        self.consensus_scores: List[float] = []  # degree agreement/مسار
        self.vote_scores: List[float] = []  # score نهائي/مسار
        self.chosen_index: int = 0
        self.created_at: float = time.time()
        self.resolved: Optional[CollectivePlan] = None  # الخطة المدمجة

    @property
    def chosen_role_note(self) -> str:
        if not self.roles:
            return ""
        judges = {}
        for r, judg in self.role_judgments.items():
            if self.chosen_index in judg:
                judges[r] = judg[self.chosen_index].confidence
        if not judges:
            return ""
        top = max(judges, key=judges.get)
        return f"أعلى حكم من دور {top} بثقة {round(judges[top], 3)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id, "goal": self.goal,
            "roles": self.roles,
            "n_roles": len(self.role_judgments),
            "chosen_index": self.chosen_index,
            "chosen_role_note": self.chosen_role_note,
            "consensus_scores": [round(s, 3) for s in self.consensus_scores],
            "vote_scores": [round(s, 3) for s in self.vote_scores],
            "role_judgments": {
                r: {str(i): {
                    "confidence": round(p.confidence, 3),
                    "verdict": p.verdict,
                    "memory_effect": round(p.memory_effect, 3),
                    "calibration_effect": round(p.calibration_effect, 3),
                } for i, p in judg.items()}
                for r, judg in self.role_judgments.items()
            },
            "role_accuracies": {
                r: {"learned": a.get("learned", 0),
                    "correct": a.get("correct", 0),
                    "accuracy": round(float(a.get("accuracy", 0) or 0), 3)}
                for r, a in self.role_accuracies.items()
            },
            "multi_path": (self.multi_path.to_dict()
                           if self.multi_path else None),
            "resolved": (self.resolved.to_dict()
                         if self.resolved else None),
            "created_at": self.created_at,
        }


class CollectivePlan:
    """التوقع الجماعي المتطور (Collective Resolution).

    خطة واحدة موسّعة تولّدها المحاكاة الجماعية بعد حُكم الفريق على
    المسارات: تدمج أفضل خطوات كل مسار (المسار الأعلى تصويتًا أولًا
    ثم الخطوات غير المكررة)، وتحمل محاكاة التعارض الافتراضي بين
    الأحكام (conflict_sim ∈ [0,1]) وثقة الخطة المدمجة المحصورة
    [0,1] بمكافأة دمج لا تتجاوز +0.15.
    """

    def __init__(self, task_id: str, goal: str,
                 roles: List[str],
                 candidate_paths: Optional[List[List[str]]] = None
                 ) -> None:
        self.task_id = task_id
        self.goal = goal
        self.roles: List[str] = list(roles or [])
        # Dict[n, action, source_path, source_role,
        # source_confidence, step_weight, risk_level]
        self.merged_steps: List[Dict[str, Any]] = []
        self.conflict_sim: float = 0.0   # نسبة التعارض الافتراضي
        self.resolution_note: str = ""
        self.confidence: float = 0.0     # [0,1]
        self.created_at: float = time.time()

    @property
    def n_merged(self) -> int:
        return len(self.merged_steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id, "goal": self.goal,
            "roles": self.roles,
            "n_merged": self.n_merged,
            "merged_steps": self.merged_steps,
            "conflict_sim": round(self.conflict_sim, 3),
            "resolution_note": self.resolution_note,
            "confidence": round(self.confidence, 3),
            "created_at": self.created_at,
        }


def _classify_risk(text: str) -> tuple:
    """تصنيف مخاطر خطوة من نصها (level, why)."""
    t = f" {text} ".lower()
    if any(h in t for h in _HIGH_RISK_HINTS):
        return "high", "خطوة تستدعي نشاطًا خارجيًا مؤثرًا أو غير قابل للتراجع"
    if any(h in t for h in _MEDIUM_RISK_HINTS):
        return "medium", "خطوة تعتمد على مصدر خارجي قد يتأخر أو يفشل"
    return "low", "خطوة داخلية منخفضة المخاطر"


def _extract_plan_steps(goal: str,
                        plan_steps: Optional[List[str]] = None) -> List[str]:
    """استخراج خطوات متوقعة (بلا LLM): نص الخطة أولًا، وإلا من أفعال الهدف."""
    if plan_steps:
        return [s[:200] for s in plan_steps if s and s.strip()][:10]
    if not goal or not goal.strip():
        return []
    steps: List[str] = []
    for verb, noun in _ACTION_PATTERNS:
        if verb in goal:
            idx = goal.index(verb)
            snippet = goal[idx:idx + 40].strip()
            if snippet and snippet not in steps:
                steps.append(snippet)
        if len(steps) >= 4:
            break
    if not steps and len(goal) >= 8:
        steps.append(goal[:200])
    return steps


class PreActionReasoner:
    """محرك التفكير ما قبل الفعل (thread-safe، تدهور آمن، بدون أي API)."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db = db_path or _LOCAL_DB
        self._lock = threading.RLock()
        with self._lock, _connect(self._db):
            pass

    # ── جوهر التفكير ────────────────────────────────────────────────────

    def reason(self, task_id: str, goal: str,
               plan_steps: Optional[List[str]] = None,
               role: str = "") -> ReasoningPlan:
        """يجري جلسة تفكير كاملة ويحفظها في السجل."""
        plan = ReasoningPlan(goal, role)
        try:
            # ═══ الذاكرة الحسية: استحضار السوابق قبل التفكير ═══
            try:
                plan.recalled_memories = self.recall(goal, role, top_k=5)
            except Exception as exc:
                logger.warning("PAR sensory recall failed: %s", exc)
            # أثر الذاكرة على الثقة:
            # سوابق ناجحة (proceed) → ترفع الثقة، مراجعات متكررة (revise)
            # → تخفض الثقة. الحد الإجمالي ±0.2 حتى لا تطغى الذاكرة.
            if plan.recalled_memories:
                # ترجيح الأثر حسب التشابه: سوابق الدور نفسه (sim=1.0)
                # تؤثر بالأثر الكامل، وسوابق الأهداف المشتركة فقط
                # (sim<1.0، أدوار أخرى) تؤثر بنصف الأثر — حتى لا تُرفع
                # ثقة دورٍ ضعيفٍ بسوابق نجاح دورٍ آخر.
                # سوابق معلومة النتيجة الفاشلة (outcome=failure) لا
                # ترفع الثقة أبدًا حتى لو كان قرارها proceed — الدقة
                # المقاسة أثقل من القرار الظاهري.
                def _mweight(m):
                    return m.get("similarity", 0.0) or 0.0
                proc = sum(_mweight(m) for m in plan.recalled_memories
                           if m["verdict"] == "proceed"
                           and m.get("outcome") != "failure")
                rev = sum(_mweight(m) for m in plan.recalled_memories
                          if m["verdict"] == "revise"
                          or m.get("outcome") == "failure")
                plan.memory_effect = max(-0.2, min(
                    0.2, 0.05 * proc - 0.08 * rev))
            # ═══ استحضار الدقة التاريخية: معايرة الثقة حسب دقة الدور ═══
            # قبل الجلسة يستحضر المحرك دقة التوقعات السابقة لنفس الدور:
            # دورٌ دقيق تاريخًا (≥2/3) → رفع الثقة بحد +0.05، ودورٌ
            # كثير الأخطاء (<1/2) → خفضها بحد -0.075. المعايرة لا
            # تحدث قبل 3 جلسات مقاسة للدور، ومحكومة بحد صارم ±0.15.
            try:
                _cal = self.calibrate_historical(role)
                plan.calibration_effect = _cal.get(
                    "calibration_effect", 0.0)
                plan.historical_accuracy = _cal.get("accuracy", 0.0)
                plan.historical_n = _cal.get("n", 0)
                if _cal.get("reason"):
                    plan.recalled_memories.insert(0, {
                        "task_id": "",
                        "role": role[:60],
                        "goal": "استحضار الدقة التاريخية: "
                                + _cal["reason"],
                        "verdict": "proceed",
                        "confidence": round(
                            float(_cal.get("accuracy", 0) or 0), 3),
                        "created_at": time.time(),
                        "similarity": 1.0,
                        "outcome": "",
                        "historical_accuracy": round(
                            float(_cal.get("accuracy", 0) or 0), 3),
                        "historical_n": _cal.get("n", 0),
                    })
            except Exception as exc:
                logger.warning(
                    "PAR historical calibration failed: %s", exc)
            steps = _extract_plan_steps(goal, plan_steps)
            for i, step in enumerate(steps, 1):
                level, why = _classify_risk(step)
                plan.expected_steps.append({
                    "n": i, "action": step[:200],
                    "risk_level": level, "risk_reason": why,
                })
                plan.risks.append({"step": i, "level": level, "why": why})

            high = sum(1 for r in plan.risks if r["level"] == "high")
            medium = sum(1 for r in plan.risks if r["level"] == "medium")
            total = max(1, len(plan.risks))
            # الثقة تنخفض مع الخطورة والتعقيد
            plan.confidence = max(0.0, min(1.0,
                1.0 - (0.25 * high) - (0.08 * medium)
                - 0.03 * min(6, max(0, total - 4))))
            if len(plan.risks) == 0:
                plan.confidence = 0.3  # لا خطوات قابلة للتحليل → ثقة محدودة

            # تعديلات (revise): بديل داخلي للخطوات العالية، أو خطوة فحص أولى
            plan.base_confidence = plan.confidence
            plan.confidence = max(0.0, min(
                1.0, plan.confidence
                + plan.memory_effect
                + plan.calibration_effect))
            if plan.confidence < _PAR_MIN_CONFIDENCE:
                plan.verdict = "revise"
                for r in plan.risks:
                    if r["level"] == "high":
                        plan.revisions.append(
                            f"الخطوة {r['step']}: استبدل النشاط الخارجي "
                            f"ببديل داخلي أو أضف خطوة تحقق أولية")
                if not plan.revisions and plan.confidence < 0.4:
                    plan.revisions.append(
                        "أضف خطوة فحص أولية منخفضة التكلفة قبل التنفيذ")
            else:
                plan.verdict = "proceed"
            # حفظ في السجل
            with self._lock, _connect(self._db) as conn:
                conn.execute("""
                    INSERT INTO par_records
                      (task_id, role, goal, steps_json, risks_json,
                       verdict, confidence, revised_n, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (task_id, role[:60], goal[:_MAX_GOAL_CHARS],
                      json.dumps(plan.expected_steps, ensure_ascii=False),
                      json.dumps(plan.risks, ensure_ascii=False),
                      plan.verdict, plan.confidence,
                      len(plan.revisions), time.time()))
                conn.commit()
                self._prune()
        except Exception as exc:
            logger.warning("PAR reason failed: %s", exc)
        return plan

    def recall(self, goal: str = "", role: str = "",
               top_k: int = 5) -> List[Dict[str, Any]]:
        """استحضار جلسات التفكير السابقة المماثلة (الذاكرة الحسية).
        يطابق الدور حرفيًا أولًا، ثم يشترط تطابق كلمة واحدة على الأقل
        من كلمات الهدف مع كلمات الأهداف المسجلة (بلا LLM، بلا متجهات).
        كل ذكرى ترجع مع درجة تشابه بسيطة (0.0 - 1.0) وتاريخها."""
        out: List[Dict[str, Any]] = []
        try:
            with self._lock, _connect(self._db) as conn:
                role = (role or "").strip()
                words = [w for w in (goal or "").split()
                         if len(w) >= 2]
                rows = conn.execute("""
                    SELECT task_id, role, goal, verdict, confidence,
                           created_at, outcome FROM par_records
                    ORDER BY created_at DESC LIMIT 300
                """).fetchall()
                for r in rows:
                    sim = 0.0
                    if role and r[1] and r[1] == role:
                        sim = 1.0
                    elif words and r[2]:
                        rwords = [w for w in r[2].split() if len(w) >= 2]
                        if rwords:
                            common = sum(1 for w in words if w in rwords)
                            sim = round(common / len(words), 2)
                    if sim > 0.0:
                        out.append({
                            "task_id": r[0], "role": r[1], "goal": r[2],
                            "verdict": r[3], "confidence": r[4],
                            "created_at": r[5],
                            "similarity": sim,
                            "outcome": r[6] or "",
                        })
                    if len(out) >= top_k:
                        break
        except Exception as exc:
            logger.warning("PAR recall failed: %s", exc)
        return out

    def learn(self, task_id: str, outcome: str) -> Optional[Dict[str, Any]]:
        """الحلقة الاستدراكية: ربط النتيجة الفعلية (success/failure)
        بأحدث جلسة تفكير لنفس المهمة — يسجّل outcome ويقيس دقة التوقع.
        outcome=None يمسح النتيجة فقط (لإعادة التقييم)."""
        if outcome not in (None, "success", "failure"):
            return None
        try:
            with self._lock, _connect(self._db) as conn:
                row = conn.execute("""
                    SELECT id, verdict, outcome FROM par_records
                    WHERE task_id = ?
                    ORDER BY created_at DESC LIMIT 1
                """, (task_id,)).fetchone()
                if not row:
                    return None
                rid, verdict, prev = row
                conn.execute(
                    "UPDATE par_records SET outcome=? WHERE id=?",
                    (outcome, rid))
                conn.commit()
                # قياس الدقة: verdict متوقع مقابل النتيجة الفعلية
                # proceed + success → صحيح، revise + failure → صحيح
                correct = ((verdict == "proceed" and outcome == "success")
                           or (verdict == "revise" and outcome == "failure"))
                return {"task_id": task_id, "verdict": verdict,
                        "outcome": outcome, "was_correct": correct,
                        "had_outcome_before": prev is not None
                        and prev != ""}
        except Exception as exc:
            logger.warning("PAR learn failed: %s", exc)
        return None

    def calibrate_historical(self, role: str = "") -> Dict[str, Any]:
        """استحضار الدقة التاريخية للدور ومعايرة الثقة حسبها.

        يحسب دقة توقعات الدور من السجلات ذات outcome معلوم:
          - n >= 3 جلسات مقاسة فقط (عينة ذات معنى)، وإلا لا معايرة.
          - accuracy >= 2/3 → رفع +0.05 (دور موثوق تاريخًا)
          - accuracy <  1/2 → خفض -0.075 (دور كثير الأخطاء)
          - بينهما → لا معايرة (0.0)
        ومعايرة أي دور محصورة بحد صارم ±0.15 لا يتجاوزها أبدًا.
        ترجع {role, n, correct, accuracy, calibration_effect, reason}."""
        role = (role or "").strip()
        result: Dict[str, Any] = {
            "role": role, "n": 0, "correct": 0, "accuracy": 0.0,
            "calibration_effect": 0.0, "reason": "",
        }
        try:
            with self._lock, _connect(self._db) as conn:
                if role:
                    row = conn.execute("""
                        SELECT COUNT(*) FROM par_records
                        WHERE role = ? AND outcome IN ('success', 'failure')
                    """, (role,)).fetchone()
                    n = row[0] if row else 0
                else:
                    n = 0
                if n < 3:
                    result["n"] = n
                    result["reason"] = (
                        f"لا معايرة: {n} جلسة مقاسة فقط للدور"
                        " (الحد الأدنى 3)"
                        if n else "لا معايرة: لا توجد سجلات مقاسة للدور")
                    return result
                if role:
                    c = conn.execute("""
                        SELECT COUNT(*) FROM par_records
                        WHERE role = ? AND outcome IN ('success', 'failure')
                          AND (verdict = 'proceed' AND outcome = 'success'
                               OR verdict = 'revise' AND outcome = 'failure')
                    """, (role,)).fetchone()
                else:
                    c = (0,)
                correct = c[0] if c else 0
                accuracy = correct / max(1, n)
                effect = 0.0
                if accuracy >= 2 / 3:
                    effect = 0.05
                elif accuracy < 0.5:
                    effect = -0.075
                effect = max(-0.15, min(0.15, effect))  # حد صارم ±0.15
                if effect > 0:
                    reason = (
                        f"دور {role!r} دقيق تاريخًا: "
                        f"{correct}/{n} ({round(accuracy, 2)})"
                        " → رفع الثقة")
                elif effect < 0:
                    reason = (
                        f"دور {role!r} كثير الأخطاء تاريخًا: "
                        f"{correct}/{n} ({round(accuracy, 2)})"
                        " → خفض الثقة")
                else:
                    reason = (
                        f"دور {role!r}: دقة {round(accuracy, 2)} "
                        f"({correct}/{n}) — بين الحدَّين، بلا معايرة")
                result.update({
                    "n": n, "correct": correct, "accuracy": accuracy,
                    "calibration_effect": effect, "reason": reason,
                })
                return result
        except Exception as exc:
            logger.warning("PAR calibrate_historical failed: %s", exc)
        return result

    # ═══ التوقع المتعدد المسارات (Multi-Path Forecasting) ═══

    def history_score_for_path(self, path_steps: List[str],
                               role: str = "") -> Dict[str, Any]:
        """تصنيف مسار تنفيذي مقابل السوابق المقاسة تاريخًا للدور.

        يطابق كلمات كل خطوة مع خطوات الجلسات المقاسة (outcome معلوم):
        كل خطوة تتشابه مع جلسة صحيحة (verdict+outcome متطابقان) تضيف
        +0.05، ومع جلسة خاطئة تخصم −0.08 — أي تأثير محصور بحد صارم
        ±0.15 مهما تعددت المسارات أو السجلات.
        ترجع {path_key, matching_steps, correct_n, wrong_n,
        history_effect, n_measured, verdict_summary}.
        """
        result: Dict[str, Any] = {
            "matching_steps": 0, "correct_n": 0, "wrong_n": 0,
            "history_effect": 0.0, "n_measured": 0, "verdict_summary": "",
        }
        try:
            with self._lock, _connect(self._db) as conn:
                rows = conn.execute("""
                    SELECT goal, verdict, outcome FROM par_records
                    WHERE role = ? AND outcome IN ('success', 'failure')
                """, ((role or "").strip(),)).fetchall()
                if not rows:
                    result["verdict_summary"] = (
                        "لا سوابق مقاسة للدور — بلا أثر تاريخي")
                    return result
                correct, wrong = 0, 0
                matched = 0
                for _step in (path_steps or []):
                    t = f" {_step} ".lower()
                    if not t.strip():
                        continue
                    s_correct = any(self._path_matches_record(
                        _step, r[0]) for r in rows
                        if self._record_is_correct(r[1], r[2]))
                    s_wrong = any(self._path_matches_record(
                        _step, r[0]) for r in rows
                        if not self._record_is_correct(r[1], r[2]))
                    if s_correct:
                        correct += 1
                        matched += 1
                    elif s_wrong:
                        wrong += 1
                        matched += 1
                effect = max(-0.15, min(0.15,
                                        0.05 * correct - 0.08 * wrong))
                n_meas = len(rows)
                summary = ""
                if effect > 0:
                    summary = (f"مسار يتشابه مع {correct} سوابق صحيحة "
                               f"({n_meas} جلسة مقاسة) → رفع الثقة")
                elif effect < 0:
                    summary = (f"مسار يتشابه مع {wrong} سوابق خاطئة "
                               f"({n_meas} جلسة مقاسة) → خفض الثقة")
                else:
                    summary = (f"مسار بلا تشابه حاسم مع السوابق "
                               f"({n_meas} جلسة مقاسة) — بلا أثر")
                result.update({"matching_steps": matched,
                               "correct_n": correct, "wrong_n": wrong,
                               "history_effect": effect,
                               "n_measured": n_meas,
                               "verdict_summary": summary})
        except Exception as exc:
            logger.warning("PAR history_score_for_path failed: %s", exc)
        return result

    @staticmethod
    def _record_is_correct(verdict: str, outcome: str) -> bool:
        """سجل صحيح: proceed+success أو revise+failure."""
        return ((verdict == "proceed" and outcome == "success")
                or (verdict == "revise" and outcome == "failure"))

    @staticmethod
    def _path_matches_record(step: str, record_goal: str) -> bool:
        """تطابق نمطي بسيط: كلمة مشتركة بطول ≥ 3 أحرف بين الخطوة والهدف."""
        t = f" {step} ".lower()
        rwords = [w for w in (record_goal or "").split() if len(w) >= 3]
        return any(w in t for w in rwords)

    def reason_multi(self, task_id: str, goal: str,
                     candidate_paths: List[List[str]],
                     role: str = "") -> MultiPathPlan:
        """التوقع المتعدد المسارات: مقارنة عدة خطط بديلة واختيار
        الأعلى ثقةً تاريخًا للدور.

        لكل مسار: جلسة تفكير مستقلة (تُحفظ في السجل) مع نفس استحضار
        الذاكرة الحسية والمعايرة التاريخية، ثم يضاف أثر التشابه
        التاريخي مع السوابق المقاسة للدور (history_score). المسار
        الأعلى ثقةً بعد كل التعديلات هو المختار — لكن الثقة تبقى
        محصورة [0,1] والمعايرات محكومة بحد ±0.15.

        الحد الأدنى 2 مسارات وإلا تسقط إلى reason() لمسار واحد. المسار
        المختار يوسم في recalled_memories، وكل مسار يحفظ تاريخه
        (n_measured, correct_n, wrong_n, history_effect) في to_dict().
        """
        mp = MultiPathPlan(task_id, goal, role)
        if not candidate_paths or len(candidate_paths) < 2:
            # تساقط آمن: جلسة تفكير واحدة كالمعتاد
            plan = self.reason(task_id, goal, role=role)
            mp.plans.append(plan)
            mp.history_scores.append(0.0)
            mp.chosen_index = 0
            return mp
        # جلسة تفكير لكل مسار (task_id::path{i})
        for i, path in enumerate(candidate_paths):
            plan = self.reason(f"{task_id}::path{i}", goal,
                               plan_steps=path, role=role)
            mp.plans.append(plan)
        # أثر التشابه التاريخي لكل مسار (من سجلات الدور المقاسة)
        mp._path_histories: List[Dict[str, Any]] = []  # type: ignore[attr-defined]
        for plan in mp.plans:
            steps = [s.get("action", "") for s in plan.expected_steps]
            hs = self.history_score_for_path(steps, role)
            plan.history_score = hs  # type: ignore[attr-defined]
            mp._path_histories.append(hs)  # type: ignore[attr-defined]
            mp.history_scores.append(hs["history_effect"])
        # الثقة النهائية لكل مسار = clamp(... + history_effect)
        # (history_effect محصور بـ ±0.15 أصلًا)
        for plan in mp.plans:
            if hasattr(plan, "history_score"):
                plan.confidence = max(0.0, min(1.0, plan.confidence
                    + plan.history_score["history_effect"]))
        # اختيار المسار الأعلى ثقةً
        if mp.plans:
            mp.chosen_index = max(range(len(mp.plans)),
                                  key=lambda i: mp.plans[i].confidence)
        # وسم المسار المختار في ذكريات كل جلسة
        for i, plan in enumerate(mp.plans):
            hs = mp._path_histories[i]  # type: ignore[attr-defined]
            summary = hs.get("verdict_summary", "")
            plan.recalled_memories.insert(0, {
                "task_id": "",
                "role": role[:60],
                "goal": ("توقع متعدد المسارات: مسار "
                         f"{i + 1}/{len(mp.plans)}"
                         + (" (المختار)" if i == mp.chosen_index else "")
                         + " — " + summary),
                "verdict": plan.verdict,
                "confidence": round(plan.confidence, 3),
                "created_at": time.time(),
                "similarity": 1.0,
                "outcome": "",
                "path_index": i,
                "is_chosen": i == mp.chosen_index,
                "history_effect": round(mp.history_scores[i], 3),
            })
        return mp

    # ═══ التوقع المتعدد المسارات عبر أدوار الفريق ═══

    def reason_multi_role(self, task_id: str, goal: str,
                          candidate_paths: List[List[str]],
                          roles: Optional[List[str]] = None
                          ) -> MultiRolePlan:
        """التوقع المتعدد المسارات عبر أدوار الفريق (Multi-Role Voting).

        توزع الخطط البديلة على أدوار متخصصة: لكل دور تجري جلسة تفكير
        مستقلة لكل مسار (بذاكرته الحسية ومعايرته التاريخية وأثره
        النمطي، وتُحفظ جميعًا في السجل باسمها المسلكي
        task_id::role::path{i}). ثم يجمع المحرك الأحكام عبر الأدوار:

          vote[i] = weighted_mean(conf_judge_ij)
                    + consensus_bonus_i + history_effect_i

        حيث الوزن التاريخي للدور: دقة ≥2/3 → 1.5، <1/2 → 0.5،
        بينها → 1.0 (بعد 3 جلسات مقاسة للدور، وإلا 1.0)؛
        consensus_bonus: +0.03 إذا كان انحراف الثقات بين الأدوار
        ≤ 0.04 (أدوار متقاربة الآراء على المسار)؛ وhistory_effect
        هو أثر التشابه النمطي لمسار الدور المختص (محصور بـ ±0.15).
        كل معايرة إضافية محصورة بحد صارم ±0.15 والمتوسط يبقى في
        [0,1] لأن المدخلات كذلك. في حالة تعادل يفوز المسار ذو
        التوافق الأعلى.

        تساقط آمن: بلا أدوار أو دور واحد → reason_multi بوحدة
        «الفريق»؛ وأي دور يفشل يسقط من التصويت دون إفساد الجلسة؛
        إن فشلت كل الأدوار تسقط الجلسة إلى reason_multi عادي."""
        mr = MultiRolePlan(task_id, goal, roles)
        active_roles: List[str] = []
        for r in (roles or []):
            if r and r.strip():
                active_roles.append(r.strip())
        if len(active_roles) < 2:
            mp = self.reason_multi(task_id, goal, candidate_paths,
                                   role="الفريق")
            mr.multi_path = mp
            mr.role_judgments["الفريق"] = {
                i: p for i, p in enumerate(mp.plans)}
            mr.consensus_scores = [0.0] * len(mp.plans)
            mr.vote_scores = list(mp.history_scores)
            mr.chosen_index = mp.chosen_index
            return mr
        # 1. كل دور يقيم كل مسار
        for role in active_roles:
            judg: Dict[int, ReasoningPlan] = {}
            for i, path in enumerate(candidate_paths):
                try:
                    judg[i] = self.reason(
                        f"{task_id}::{role}::path{i}", goal,
                        plan_steps=path, role=role)
                except Exception as exc:
                    logger.warning("Multi-role judge %s path %d "
                                   "failed: %s", role, i, exc)
            if judg:
                mr.role_judgments[role] = judg
            try:
                mr.role_accuracies[role] = self.calibrate_historical(
                    role)
            except Exception:
                mr.role_accuracies[role] = {
                    "n": 0, "correct": 0, "accuracy": 0.0}
        if not mr.role_judgments:
            try:
                mp = self.reason_multi(task_id, goal, candidate_paths,
                                       role="الفريق")
            except Exception as exc:
                logger.warning("Multi-role full fallback "
                               "reason_multi failed: %s", exc)
                mp = None
            if mp is not None:
                mr.multi_path = mp
                mr.role_judgments["الفريق"] = {
                    i: p for i, p in enumerate(mp.plans)}
                mr.consensus_scores = [0.0] * len(mp.plans)
                mr.vote_scores = list(mp.history_scores)
                mr.chosen_index = mp.chosen_index
                return mr
            # لا يوجد حتى تساقط آمن: خطة صفريّة بأحكم جزئية
            mr.vote_scores = [0.5] * len(candidate_paths)
            mr.consensus_scores = [0.0] * len(candidate_paths)
            if candidate_paths:
                mr.chosen_index = 0
            return mr
        n_paths = len(candidate_paths)
        # 2. جمع الأحكام عبر الأدوار لكل مسار
        for i in range(n_paths):
            confs, weighted, wsum = [], [], 0.0
            for role, judg in mr.role_judgments.items():
                if i not in judg:
                    continue
                p = judg[i]
                confs.append(p.confidence)
                a = float(mr.role_accuracies[role]
                          .get("accuracy") or 0.0)
                n = int(mr.role_accuracies[role].get("n") or 0)
                weight = 1.0
                if n >= 3:
                    weight = (1.5 if a >= 2 / 3
                              else (0.5 if a < 0.5 else 1.0))
                weighted.append(p.confidence * weight)
                wsum += weight
            if not confs:
                mr.consensus_scores.append(0.0)
                mr.vote_scores.append(0.0)
                continue
            # متوسط موزون (confs ∈ [0,1] والوزون > 0 → الوسط ∈ [0,1])
            std = 0.0
            if len(confs) >= 2:
                mu = sum(confs) / len(confs)
                std = (sum((c - mu) ** 2 for c in confs)
                       / len(confs)) ** 0.5
            consensus = 1.0 - min(1.0, std / 0.2)
            consensus_bonus = 0.03 if std <= 0.04 else 0.0
            # أثر تاريخي نمطي على المسار عبر الأدوار
            hist = 0.0
            for role, judg in mr.role_judgments.items():
                if i not in judg:
                    continue
                steps = [s.get("action", "")
                         for s in judg[i].expected_steps]
                try:
                    hs = self.history_score_for_path(steps, role)
                    hist = max(-0.15, min(0.15, hist
                                          + hs["history_effect"]))
                except Exception:
                    pass
            vote = max(0.0, min(1.0, sum(weighted) / max(1e-9, wsum)
                       + consensus_bonus + hist))
            mr.consensus_scores.append(round(consensus, 3))
            mr.vote_scores.append(vote)
        # 3. اختيار المسار الأعلى توافقًا وثقةً (التعادل → توافق أعلى)
        if mr.vote_scores:
            mr.chosen_index = max(
                range(len(mr.vote_scores)),
                key=lambda i: (mr.vote_scores[i],
                               mr.consensus_scores[i]))
            # التوقع الجماعي المتطور: محاكاة تعارض افتراضي
            # وتوليد خطة مدمجة إذا تعددت المسارات والأدوار
            if n_paths >= 2 and len(mr.role_judgments) >= 2:
                try:
                    mr.resolved = self.resolve_collective(
                        mr, candidate_paths)
                except Exception as exc:
                    logger.warning("Collective resolution failed: %s",
                                   exc)
            # وسم المسار المختار في ذكريات أحكام كل دور
            for role, judg in mr.role_judgments.items():
                if mr.chosen_index in judg:
                    p = judg[mr.chosen_index]
                    p.recalled_memories.insert(0, {
                        "task_id": "",
                        "role": role[:60],
                        "goal": ("حُكم فريق عبر الأدوار: "
                                 f"مسار {mr.chosen_index + 1}"
                                 f"/{n_paths} (المختار)"),
                        "verdict": p.verdict,
                        "confidence": round(p.confidence, 3),
                        "created_at": time.time(),
                        "similarity": 1.0,
                        "outcome": "",
                        "path_index": mr.chosen_index,
                        "is_chosen": True,
                        "vote_score": round(
                            mr.vote_scores[mr.chosen_index], 3),
                        "consensus": round(
                            mr.consensus_scores[mr.chosen_index], 3),
                    })
        return mr

    def role_accuracy_stats(self) -> Dict[str, Dict[str, Any]]:
        """دقة التوقعات التاريخية لكل دور على حدة (للوحة والمعايرة).
        ترجع {role: {role, learned, correct, accuracy}} — الأدوار ذات
        outcome معلوم فقط."""
        out: Dict[str, Dict[str, Any]] = {}
        try:
            with self._lock, _connect(self._db) as conn:
                rows = conn.execute("""
                    SELECT role, COUNT(*),
                           SUM(CASE WHEN
                             (verdict = 'proceed' AND outcome = 'success')
                             OR (verdict = 'revise' AND outcome = 'failure')
                             THEN 1 ELSE 0 END)
                    FROM par_records
                    WHERE role IS NOT NULL AND role != ''
                      AND outcome IN ('success', 'failure')
                    GROUP BY role
                """).fetchall()
                for role, total, correct in rows:
                    out[role] = {
                        "role": role, "learned": total,
                        "correct": correct,
                        "accuracy": (round(correct / total, 3)
                                     if total else 0.0),
                    }
        except Exception as exc:
            logger.warning("PAR role_accuracy_stats failed: %s", exc)
        return out

    def resolve_collective(self, mr: MultiRolePlan,
                           candidate_paths: List[List[str]]
                           ) -> "CollectivePlan":
        """التوقع الجماعي المتطور: محاكاة تعارضٍ افتراضي بين أحكام
        الأدوار وتوليد خطة واحدة موسّعة تدمج أفضل خطوات كل مسار.

        التعارض يُحسب رياضيًا بلا أي نموذج لغوي: يقيس تباين الأحكام
        المجمعة النهائية بين المسارات (vote_scores — متوسط مرجح
        بأوزان دقة الأدوار)؛ كل مسار ينحرف عن المتوسط بأكثر من
        0.10 يعد مسارًا متعارضًا، ومحاكاة التعارض conflict_sim هي
        نسبة المسارات المتعارضة مع إضافة معيار استمراري (الانحراف
        المعياري النسبي للأحكام المجمعة، محصور [0,1]) حتى لا يهبط
        التعارض إلى صفر كلما تقاربت الأحكام جزئيًا. الخطة المدمجة تُرتّب
        المسار الأعلى تصويتًا أولًا ثم تضيف خطوات المسارات الأخرى
        غير المكررة (تطابق نمطي بكلمة مشتركة ≥ 3 أحرف)، وكل خطوة
        تحمل مصدرها ووزنها. تحفظ الخطة جلسةً واحدة في السجل
        (task_id::collective) بثقة محصورة [0,1]."""
        cp = CollectivePlan(mr.task_id, mr.goal, mr.roles,
                            candidate_paths)
        if not mr.vote_scores or not mr.role_judgments:
            cp.resolution_note = "لا أحكام كافية — بلا خطة مدمجة"
            self._save_collective_record(cp, candidate_paths)
            return cp
        n_paths = len(mr.vote_scores)
        # 1. محاكاة التعارض الافتراضي: تباين الأحكام المجمعة بين
        # المسارات — كل مسار ينحرف عن متوسط الأحكام المجمعة بأكثر
        # من 0.10 يعد مسارًا متعارضًا (نسبة المسارات المتعارضة)،
        # مضافًا إليه معيار استمراري هو الانحراف المعياري النسبي
        # للأحكام المجمعة محصورًا [0,1]؛ والنسبتان تُرجع أعلى
        # قيمة بينهما (قياس بأكمله رياضي بلا أي نموذج لغوي).
        mu = sum(mr.vote_scores) / max(1, n_paths)
        diverged = sum(1 for v in mr.vote_scores
                       if abs(v - mu) > 0.10)
        ratio = diverged / max(1, n_paths)
        if mu > 0 and n_paths > 1:
            var = sum((v - mu) ** 2 for v in mr.vote_scores) / n_paths
            norm_std = (var ** 0.5) / mu
            cont = min(1.0, norm_std)
        else:
            cont = 0.0
        cp.conflict_sim = round(max(ratio, cont), 3)
        # 2. دمج أفضل الخطوات: المسار الأعلى تصويتًا أولًا
        order = sorted(range(n_paths),
                       key=lambda i: (mr.vote_scores[i],
                                      mr.consensus_scores[i]),
                       reverse=True)
        seen_words = set()
        for idx in order:
            best = None
            best_weight = -1.0
            for r, judg in mr.role_judgments.items():
                if idx not in judg:
                    continue
                p = judg[idx]
                a = float(mr.role_accuracies.get(r, {}).get("n") or 0)
                acc = float(mr.role_accuracies.get(r, {})
                            .get("accuracy") or 0.0)
                w = (1.5 if a >= 3 and acc >= 2 / 3
                     else (0.5 if a >= 3 and acc < 0.5 else 1.0))
                if p.confidence * w > best_weight:
                    best, best_weight = p, p.confidence * w
            if best is None:
                continue
                # لا سوابق لهذا المسار — تخطى
            for s in best.expected_steps:
                action = (s.get("action") or "").strip()
                if not action:
                    continue
                words = {w for w in action.split() if len(w) >= 3}
                if words and words <= seen_words:
                    # كل كلماتها موجودة في خطوات مدمجة سابقًا
                    continue
                if words:
                    seen_words.update(words)
                cp.merged_steps.append({
                    "n": len(cp.merged_steps) + 1,
                    "action": action,
                    "source_path": idx,
                    "source_role": best.role or "",
                    "source_confidence": round(best.confidence, 3),
                    "step_weight": round(best_weight, 3),
                    "risk_level": s.get("risk_level", "low"),
                })
        # 3. الثقة: متوسط موزون للجلسات المندمجة + مكافأة دمج محصورة
        if cp.merged_steps:
            src_confs = [s["source_confidence"] for s in cp.merged_steps]
            base = sum(src_confs) / len(src_confs)
            merge_bonus = min(0.15, 0.02 * max(
                0, len(mr.role_judgments) - 1))
            cp.confidence = max(0.0, min(1.0, base + merge_bonus))
            n_votes = sum(len(j) for j in mr.role_judgments.values())
            cp.resolution_note = (
                f"دمج {cp.n_merged} خطوات من {n_paths} مسارات "
                f"بأحكام {len(mr.role_judgments)} أدوار "
                f"(تعارض افتراضي {round(cp.conflict_sim * 100)}% "
                f"من {n_votes} حكم) — مكافأة دمج +"
                f"{round(merge_bonus, 3)}")
        else:
            cp.resolution_note = "لا خطوات قابلة للدمج — بلا أثر"
        self._save_collective_record(cp, candidate_paths)
        return cp

    def _save_collective_record(self, cp: "CollectivePlan",
                                candidate_paths: List[List[str]]
                                ) -> None:
        """حفظ جلسة الخطة المدمجة في السجل باسم task_id::collective."""
        try:
            steps = json.dumps(
                [{"action": s["action"],
                  "source_path": s["source_path"],
                  "source_role": s["source_role"]}
                 for s in cp.merged_steps],
                ensure_ascii=False)
            risks = json.dumps([], ensure_ascii=False)
            with self._lock, _connect(self._db) as conn:
                conn.execute("""
                    INSERT INTO par_records
                      (task_id, role, goal, steps_json, risks_json,
                       verdict, confidence, revised_n, outcome,
                       created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (f"{cp.task_id}::collective", "collective",
                      f"{cp.goal} — خطة مدمجة ({cp.n_merged} خطوة)",
                      steps, risks,
                      "proceed" if cp.confidence >= _PAR_MIN_CONFIDENCE
                      else "revise",
                      round(cp.confidence, 3), 0, None, time.time()))
                self._prune()
        except Exception as exc:
            logger.warning("Collective record save failed: %s", exc)

    def learned_stats(self) -> Dict[str, Any]:
        """دقة التوقعات المقاسة من السجلات ذات النتائج الفعلية."""
        try:
            with self._lock, _connect(self._db) as conn:
                total = conn.execute("""
                    SELECT COUNT(*) FROM par_records
                    WHERE outcome IN ('success', 'failure')
                """).fetchone()[0]
                correct = conn.execute("""
                    SELECT COUNT(*) FROM par_records
                    WHERE outcome IN ('success', 'failure')
                      AND (verdict = 'proceed' AND outcome = 'success'
                           OR verdict = 'revise' AND outcome = 'failure')
                """).fetchone()[0]
                return {"learned": total, "correct": correct,
                        "accuracy": (round(correct / total, 3)
                                     if total else 0.0)}
        except Exception as exc:
            logger.warning("PAR learned_stats failed: %s", exc)
        return {"learned": 0, "correct": 0, "accuracy": 0.0}

    def latest(self, task_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._lock, _connect(self._db) as conn:
                row = conn.execute("""
                    SELECT goal, role, steps_json, risks_json, verdict,
                           confidence, revised_n, created_at, outcome
                    FROM par_records WHERE task_id = ?
                    ORDER BY created_at DESC LIMIT 1
                """, (task_id,)).fetchone()
                if not row:
                    return None
                return {"task_id": task_id, "goal": row[0], "role": row[1],
                        "expected_steps": json.loads(row[2] or "[]"),
                        "risks": json.loads(row[3] or "[]"),
                        "verdict": row[4], "confidence": row[5],
                        "revised_n": row[6], "created_at": row[7],
                        "outcome": row[8] or ""}
        except Exception as exc:
            logger.warning("PAR latest failed: %s", exc)
        return None

    def stats(self) -> Dict[str, Any]:
        try:
            with self._lock, _connect(self._db) as conn:
                total = conn.execute("SELECT COUNT(*) FROM par_records"
                                     ).fetchone()[0]
                proc = conn.execute(
                    "SELECT COUNT(*) FROM par_records WHERE verdict='proceed'"
                ).fetchone()[0]
                rev = conn.execute(
                    "SELECT COUNT(*) FROM par_records WHERE verdict='revise'"
                ).fetchone()[0]
                avg = conn.execute(
                    "SELECT COALESCE(AVG(confidence), 0) FROM par_records"
                ).fetchone()[0]
                return {"reasoned": total, "proceeded": proc, "revised": rev,
                        "avg_confidence": round(avg, 3)}
        except Exception as exc:
            logger.warning("PAR stats failed: %s", exc)
        return {"reasoned": 0, "proceeded": 0, "revised": 0,
                "avg_confidence": 0.0}

    def verdict_counts(self) -> Dict[str, int]:
        try:
            with self._lock, _connect(self._db) as conn:
                rows = conn.execute("""
                    SELECT verdict, COUNT(*) FROM par_records
                    GROUP BY verdict
                """)
                return {v: n for v, n in rows}
        except Exception:
            return {}

    def _prune(self) -> None:
        try:
            with self._lock, _connect(self._db) as conn:
                n = conn.execute("SELECT COUNT(*) FROM par_records"
                                 ).fetchone()[0]
                if n > _MAX_RECORDS:
                    conn.execute("""
                        DELETE FROM par_records WHERE id IN (
                            SELECT id FROM par_records
                            ORDER BY created_at ASC LIMIT ?
                        )
                    """, (n - _MAX_RECORDS,))
                    conn.commit()
        except Exception as exc:
            logger.warning("PAR prune failed: %s", exc)

    def reset(self) -> None:
        try:
            with self._lock, _connect(self._db) as conn:
                conn.execute("DELETE FROM par_records")
                conn.commit()
        except Exception as exc:
            logger.warning("PAR reset failed: %s", exc)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS par_records (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT NOT NULL,
    role        TEXT,
    goal        TEXT,
    steps_json  TEXT,
    risks_json  TEXT,
    verdict     TEXT NOT NULL DEFAULT 'proceed',
    confidence  REAL NOT NULL DEFAULT 0.0,
    revised_n   INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL DEFAULT 0.0
);
"""
_OUTCOME_COL = "ALTER TABLE par_records ADD COLUMN outcome TEXT"


def _migrate_outcome(conn: sqlite3.Connection) -> None:
    """إضافة عمود outcome للجداول القديمة (migration آمن للصمت)."""
    try:
        conn.execute(_OUTCOME_COL)
        conn.commit()
    except Exception:
        pass


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    _migrate_outcome(conn)
    return conn


# ── singleton + helpers ──────────────────────────────────────────────────

_par_instance = None


def get_pre_action_reasoner(db_path: Optional[str] = None) -> PreActionReasoner:
    global _par_instance
    if _par_instance is None:
        _par_instance = PreActionReasoner(db_path)
    return _par_instance


def reset_reasoning() -> None:
    global _par_instance
    _par_instance = None


def reason_task(task_id: str, goal: str,
                plan_steps: Optional[List[str]] = None,
                role: str = "") -> Optional[Dict[str, Any]]:
    """مساعدة للدمج المباشر: يفكّر قبل الفعل ويعيد النتيجة كـ dict."""
    try:
        return get_pre_action_reasoner().reason(
            task_id, goal, plan_steps, role).to_dict()
    except Exception:
        return None


def par_stats() -> Dict[str, Any]:
    try:
        return get_pre_action_reasoner().stats()
    except Exception:
        return {"reasoned": 0, "proceeded": 0, "revised": 0,
                "avg_confidence": 0.0}


def par_latest(task_id: str) -> Optional[Dict[str, Any]]:
    try:
        return get_pre_action_reasoner().latest(task_id)
    except Exception:
        return None


def par_recall(goal: str = "", role: str = "",
               top_k: int = 5) -> List[Dict[str, Any]]:
    """مساعدة للدمج: استحضار ذكريات حسية لسوابق مماثلة."""
    try:
        return get_pre_action_reasoner().recall(goal, role, top_k)
    except Exception:
        return []


def par_learn(task_id: str, outcome: str) -> Optional[Dict[str, Any]]:
    """مساعدة للدمج: الحلقة الاستدراكية — ربط نتيجة فعلية بجلسة التفكير."""
    try:
        return get_pre_action_reasoner().learn(task_id, outcome)
    except Exception:
        return None


def par_learned_stats() -> Dict[str, Any]:
    """مساعدة للدمج: دقة التوقعات المقاسة."""
    try:
        return get_pre_action_reasoner().learned_stats()
    except Exception:
        return {"learned": 0, "correct": 0, "accuracy": 0.0}


def par_calibration(role: str = "") -> Dict[str, Any]:
    """مساعدة للدمج: معايرة الدقة التاريخية للدور (±0.15 كحد صارم)."""
    try:
        return get_pre_action_reasoner().calibrate_historical(role)
    except Exception:
        return {"role": role, "n": 0, "correct": 0, "accuracy": 0.0,
                "calibration_effect": 0.0, "reason": ""}


def par_role_accuracy() -> Dict[str, Dict[str, Any]]:
    """مساعدة للدمج: دقة التوقعات التاريخية لكل دور على حدة."""
    try:
        return get_pre_action_reasoner().role_accuracy_stats()
    except Exception:
        return {}


def reason_multi_task(task_id: str, goal: str,
                      candidate_paths: Optional[
                          List[List[str]]] = None,
                      role: str = "") -> Optional[Dict[str, Any]]:
    """مساعدة للدمج: التوقع المتعدد المسارات — مقارنة خطط بديلة
    واختيار الأعلى ثقةً تاريخا للدور (بلا API، حد ±0.15 صارم)."""
    try:
        return get_pre_action_reasoner().reason_multi(
            task_id, goal, candidate_paths or [], role).to_dict()
    except Exception:
        return None


def reason_multi_role_task(task_id: str, goal: str,
                           candidate_paths: Optional[
                               List[List[str]]] = None,
                           roles: Optional[List[str]] = None
                           ) -> Optional[Dict[str, Any]]:
    """مساعدة للدمج: التوقع المتعدد المسارات عبر أدوار الفريق —
    يوزع الخطط البديلة على أدوار متخصصة ويجمع أحكامها في المسار
    الأعلى توافقًا وثقةً (بلا API، حد ±0.15 صارم)."""
    try:
        return get_pre_action_reasoner().reason_multi_role(
            task_id, goal, candidate_paths or [], roles).to_dict()
    except Exception:
        return None


def resolve_collective_task(task_id: str, goal: str,
                            candidate_paths: Optional[
                                List[List[str]]] = None,
                            roles: Optional[List[str]] = None
                            ) -> Optional[Dict[str, Any]]:
    """مساعدة للدمج: التوقع الجماعي المتطور — يحاكي تعارضًا
    افتراضيًا بين أحكام أدوار الفريق ويولّد خطة موسّعة تدمج
    أفضل خطوات كل مسار (بلا API، مكافأة دمج ≤ +0.15)."""
    try:
        reasoner = get_pre_action_reasoner()
        mr = reasoner.reason_multi_role(
            task_id, goal, candidate_paths or [], roles)
        if mr is not None and mr.resolved is not None:
            return mr.resolved.to_dict()
        return None
    except Exception:
        return None


def reset_learned_outcomes() -> None:
    """مساعدة للدمج: مسح النتائج الفعلية (لإعادة التقييم)."""
    try:
        with __import__("sqlite3").connect(_LOCAL_DB, timeout=30) as conn:
            conn.execute("""
                UPDATE par_records SET outcome = NULL
                WHERE outcome IN ('success', 'failure')
            """)
            conn.commit()
    except Exception as exc:
        logger.warning("PAR reset learned failed: %s", exc)
