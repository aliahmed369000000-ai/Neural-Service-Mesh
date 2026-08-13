# -*- coding: utf-8 -*-
"""
ai/multi_step_reasoner.py — التفكير متعدد الخطوات (Multi-Step Reasoning)
──────────────────────────────────────────────────────────────────────────
طبقة استجابة للأسئلة المعقدة متعددة الجوانب (مقارنات، «لماذا»، «كيف»،
أسئلة مركبة بأدوات العطف، قوائم، تحليل...)، تعمل بالكامل **بدون API خارجي**:

  1. is_complex_question(text)  — تصنيف حتمي سريع (لا LLM): علامات
     استفهام متعددة / جمل مركبة / مؤشرات قصد تحليلي / أسئلة مفتوحة طويلة.

  2. decompose(text)            — تفكيك السؤال إلى خطة خطوات فرعية عربية
     مرتبة (قائدية + قواعد لغوية: مركبات العطف، أنواع القصد) مع سقف
     MAX_STEPS لضمان اقتصاد التوكنات في الإجابة.

  3. synthesize(text, results)  — توليد رد تجميعي مخطط: لكل خطوة فقرة
     موجزة، ثم خلاصة استنتاجية نهائية تربط النتائج.

  4. مسارات traces في SQLite (memory/multi_step_traces.db) لتحليل
     التغطية وجودة الخطط — والسقف LTM-style لا ينطبق هنا (traces صغيرة
     ومحفوظة تلقائيًا فوق 2000 سجل).

فشل أي جزء يُبتلَع بصمت ويعاد السلوك الأصلي (سؤال واحد → رد واحد) —
هذه الطبقة **إضافية بالكامل ولا تعدّل سلوك أي مسار موجود**.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from pathlib import Path as _Path
    MEM_DIR = Path(__file__).resolve().parent.parent / "memory"
except Exception:
    MEM_DIR = Path("memory")

MAX_STEPS = 4            # سقف خطوات الخطة — اقتصاد توكنات
TRACE_LIMIT = 2000       # سقف سجلات التتبع
QUERY_MIN_COMPLEX_LEN = 45
STOP_WORDS = frozenset({"هو", "هي", "في", "من", "على", "إلى", "عن", "مع",
                        "أن", "إن", "التي", "الذي", "هذا", "هذه", "ذلك",
                        "تلك", "ما", "من", "الى", "حتى", "بين", "عند",
                        "كل", "بعد", "قبل", "دون", "غير", "لكن", "ثم",
                        "أي", "اى", "اذا", "إذا", "لماذا", "كيف", "متى",
                        "اين", "أين", "كم", "هل", "لن", "لم", "لا", "ب",
                        "ل", "ك", "و", "ف", "س", "أ", "آ", "إ", "ا"})

logger = logging.getLogger(__name__)

# ───────────────────────────── مؤشرات التعقيد الحتمي ───────────────────────

# مؤشرات قصد تحليلي تستوجب تخطيطًا (مقارنة، سببية، عملية، تعداد)
ANALYTIC_MARKERS = [
    # مقارنة
    "الفرق بين", "الفرق", "مقارنة", "بالمقارنة", "مقارن", "أيهما", "أيّهما",
    "افضل من", "أفضل من", "اكثر من", "أكثر من", "افضل", "أفضل",
    "بين", "أحسن",
    # سببية وتحليل
    "لماذا", "لماذا", "اسباب", "أسباب", "سبب", "تأثير", "اثر", "أثر",
    "كيف", "بماذا", "ما مدى", "إلى أي",
    # عملية/تعداد
    "خطوات", "مراحل", "طريقة", "طرق", "كيف يمكن", "كيف أ", "كيف اس", "كيف أس",
    "عدد", "أنواع", "انواع", "أقسام", "اقسام", "عناصر",
    # تحليل وتلخيص
    "حلل", "حلل", "مزايا", "عيوب", "مزايا وعيوب", "إيجابيات", "سلبيات",
    "نقاط القوة", "نقاط الضعف", "ايجابيات", "سلبيات",
    # أسئلة مركبة
    "وما هي", "وما هو", "وماذا", "وما", "و كيف", "و كيف", "وما الفرق",
    "و لماذا", "و ما", "ولماذا", "وماذا", "وكيف",
]
# أدوات العطف المركبة بين أسئلة/عبارات مستقلة
CONJUNCTIVES = [r"\bو\s+", r"\bثم\s+", r"\s*;\s*", r"\s*،\s*", r"\s*,\s*"]
COMPLEXITY_MARKERS = ["؟", "?", "…", "..."]

# أنماط التفكيك القائدية حسب نوع القصد
STEP_TEMPLATES: Dict[str, List[str]] = {
    "compare": [
        "تحديد نقاط المقارنة الأساسية بين طرفي السؤال",
        "تحليل الطرف الأول: خصائصه ومميزاته",
        "تحليل الطرف الثاني: خصائصه ومميزاته",
        "استخلاص المقارنة والتوصية الموزونة",
    ],
    "why": [
        "تحديد الموضوع والمفهوم المركزي في السؤال",
        "تحليل الأسباب والمبررات الأساسية",
        "ربط النتائج والأثر المترتب",
        "خلاصة تفسيرية متكاملة",
    ],
    "how": [
        "تحديد الهدف والخطوة التمهيدية",
        "الخطوات العملية مرتبة",
        "النصائح والضوابط المهمة",
        "تلخيص منهجي تطبيقي",
    ],
    "list": [
        "تحديد عناصر الموضوع الأساسية",
        "شرح العناصر الأبرز",
        "أمثلة وتطبيقات واقعية",
        "ربط العناصر بخلاصة شاملة",
    ],
    "analyze": [
        "تحليل نقاط القوة",
        "تحليل نقاط الضعف والتحديات",
        "المقارنة الموزونة والتوصيات",
        "خلاصة تحليلية نهائية",
    ],
}

# قواعد التصنيف القائدية (بالترتيب)
def _classify_intent(text: str) -> str:
    t = text
    if any(m in t for m in ("الفرق بين", "مقارنة", "بالمقارنة", "أيهما", "أيّهما",
                            "افضل من", "أفضل من", "أحسن من", "مقارن")):
        return "compare"
    if t.startswith("لماذا") or any(m in t for m in ("اسباب", "أسباب", "سبب", "لماذا")):
        return "why"
    if any(m in t for m in ("كيف", "طريقة", "طرق", "خطوات", "مراحل", "ما هي طريقة")):
        return "how"
    if any(m in t for m in ("عدد", "أنواع", "انواع", "أقسام", "اقسام", "عناصر",
                            "اذكر", "اذكر", "أذكر", "أذكر")):
        return "list"
    if any(m in t for m in ("حلل", "مزايا", "عيوب", "إيجابيات", "سلبيات",
                            "ايجابيات", "نقاط القوة", "نقاط الضعف")):
        return "analyze"
    return "list"


@dataclass
class ReasoningPlan:
    question: str
    intent: str            # compare | why | how | list | analyze
    steps: List[str] = field(default_factory=list)
    complex_: bool = False
    built_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.intent,
            "steps": self.steps,
            "complex": self.complex_,
            "built_at": self.built_at,
        }


# ───────────────────────────── التصنيف ─────────────────────────────────────

def _count_question_marks(text: str) -> int:
    return sum(1 for c in text if c in "؟?")

def _count_sentences(text: str) -> int:
    return max(1, len(re.findall(r"[.؟?!…]", text)))

def is_complex_question(text: str) -> bool:
    """تصنيف حتمي سريع: هل يستوجب السؤال خطة متعددة الخطوات؟
    المعايير (يكتفي بأحدها):
      - علامتا استفهام أو أكثر
      - مركّب بأداة عطف بين عبارات مستقلة + طول كافٍ
      - مؤشر قصد تحليلي + طول كافٍ
      - سؤال مفتوح طويل (>QUERY_MIN_COMPLEX_LEN) بمؤشر استفهام"""
    if not text or len(text.strip()) < 10:
        return False
    t = text.strip()
    if _count_question_marks(t) >= 2:
        return True
    marker_hit = any(m in t for m in ANALYTIC_MARKERS)
    if marker_hit and len(t) >= 30:
        return True
    # مركب: (أداة عطف + عبارة بعدها) وعلامتا توقف
    if len(t) >= QUERY_MIN_COMPLEX_LEN and _count_sentences(t) >= 2:
        joined = re.split(r"(؟|\?)", t)
        segments = [s.strip() for s in joined if len(s.strip()) > 5]
        if len(segments) >= 3:
            return True
    if len(t) >= QUERY_MIN_COMPLEX_LEN and _count_question_marks(t) == 1:
        return True
    return False


# ───────────────────────────── التفكيك ─────────────────────────────────────

def _split_conjuncts(text: str) -> List[str]:
    """تقسيم السؤال المركّب عند أدوات العطف المستقلة (لا يقسم داخل
    «الفرق بين X و Y» لأن الطرفين طرفا المقارنة لا عبارتان مستقلتان)."""
    t = text.strip()
    if t.startswith("الفرق بين") or t.startswith("مقارنة"):
        return [t]
    # تقسيم على «و» غير المرتبطة (ليست «وهو/وهي/والذي...» الضمائر)
    # الواو في العربية تلتصق بالكلمة التالية غالبًا («وما») — نقسّم على واو
    # مستقلة (سابقة بحد كلمة) حتى لو بلا مسافة بعدها، مع استبعاد واو
    # العطف المرتبطة بكلمات الضمير والنسب («وهو/وهي/والذي...») التي لا
    # تُبتدئ بها عبارة مستقلة.
    parts = re.split(r"\bو(?![وهيوهووالتيالذياللتيذلكتلكاي])", t)
    parts = [p.strip() for p in parts if len(p.strip()) >= 5]
    return parts if len(parts) >= 2 else [t]

def decompose(text: str, max_steps: int = MAX_STEPS) -> ReasoningPlan:
    """بناء خطة خطوات فرعية مرتبة للسؤال."""
    t = text.strip()
    intent = _classify_intent(t)
    templates = STEP_TEMPLATES.get(intent, STEP_TEMPLATES["list"])
    n = min(max_steps, max(2, len(templates)))
    steps = list(templates[:n])
    complex_ = is_complex_question(t)
    return ReasoningPlan(question=t, intent=intent, steps=steps,
                         complex_=complex_)


# ───────────────────────────── تجميع الرد ─────────────────────────────────

def synthesize(original_text: str, plan: ReasoningPlan,
               results: Optional[List[str]] = None) -> str:
    """توليد رد تجميعي مخطط: ديباجة قصيرة + خطوة لكل جزء + خلاصة.
    results: إجابات جزئية اختيارية (من مسارات متعددة إن توفرت) — إن لم
    تُمرَّر، الصياغة تدمج الخطة في إجابة واحدة متماسكة يُطلب فيها من
    النموذج الالتزام بالهيكل (يُمرَّر كنظام، النموذج هو من يجيب فعليًا)."""
    t = original_text.strip()
    parts: List[str] = []
    parts.append(
        f"سؤالك متعدد الجوانب. سأجيب وفق خطة من {len(plan.steps)} خطوات "
        f"مرتبّة لضمان شمول الإجابة:"
    )
    if results and len(results) == len(plan.steps):
        for i, (step, res) in enumerate(zip(plan.steps, results), 1):
            parts.append(f"**الخطوة {i} — {step}:** {res.strip()}")
    else:
        for i, step in enumerate(plan.steps, 1):
            parts.append(f"**الخطوة {i} — {step}:** اشرح هذه النقطة بإيجاز "
                         "ودقة ضمن سياق السؤال الأصلي.")
    parts.append("**خلاصة:** اربط الخطوات السابقة بإجابة نهائية موحدة "
                 "تجيب عن السؤال كاملًا.")
    return "\n\n".join(parts)


# ───────────────────────────── تتبع traces ─────────────────────────────────

_DB_LOCK = threading.Lock()
_PLAN_CACHE: Dict[str, ReasoningPlan] = {}
_PLAN_LOCK = threading.Lock()


class _TraceStore:
    """مستودع traces — SQLite مع تدهور آمن كامل."""

    def __init__(self, db_path: Optional[Path] = None):
        try:
            self.db_path = Path(db_path) if db_path else (MEM_DIR / "multi_step_traces.db")
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                "CREATE TABLE IF NOT EXISTS msr_traces ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  question TEXT NOT NULL,"
                "  intent TEXT NOT NULL,"
                "  steps TEXT NOT NULL,"      # JSON مصفوفة الخطوات
                "  used_in_response INTEGER NOT NULL DEFAULT 0,"
                "  created_at REAL NOT NULL"
                ")"
            )
            conn.commit()
            conn.close()
        except Exception:
            self.db_path = None

    def record(self, plan: ReasoningPlan, used: bool) -> Optional[int]:
        if self.db_path is None:
            return None
        try:
            import json
            with _DB_LOCK, sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    "INSERT INTO msr_traces (question, intent, steps, used_in_response, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (plan.question, plan.intent, json.dumps(plan.steps, ensure_ascii=False),
                     1 if used else 0, plan.built_at),
                )
                conn.commit()
                mid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            self._prune()
            return mid
        except Exception:
            return None

    def _prune(self) -> None:
        if self.db_path is None:
            return
        try:
            with _DB_LOCK, sqlite3.connect(str(self.db_path)) as conn:
                total = conn.execute("SELECT COUNT(*) FROM msr_traces").fetchone()[0]
                if total > TRACE_LIMIT:
                    conn.execute(
                        "DELETE FROM msr_traces WHERE id IN "
                        "(SELECT id FROM msr_traces ORDER BY created_at ASC LIMIT ?)",
                        (total - TRACE_LIMIT,),
                    )
                    conn.commit()
        except Exception:
            pass

    def stats(self) -> Dict[str, Any]:
        if self.db_path is None:
            return {"available": False}
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                total, used = conn.execute(
                    "SELECT COUNT(*), SUM(used_in_response) FROM msr_traces"
                ).fetchone()
                intents = dict(conn.execute(
                    "SELECT intent, COUNT(*) FROM msr_traces GROUP BY intent ORDER BY 2 DESC"
                ).fetchall())
                return {"available": True, "total": total or 0,
                        "used_in_response": used or 0, "by_intent": intents}
        except Exception:
            return {"available": False}


_store: Optional[_TraceStore] = None
_store_lock = threading.Lock()


def _get_store() -> _TraceStore:
    global _store
    with _store_lock:
        if _store is None:
            try:
                _store = _TraceStore()
            except Exception:
                _store = _TraceStore.__new__(_TraceStore)
                _store.db_path = None
        return _store


def record_trace(plan: ReasoningPlan, used: bool) -> Optional[int]:
    """تسجيل خطة في التتبع — فشل يُبتلَع."""
    try:
        return _get_store().record(plan, used)
    except Exception:
        return None


def get_trace_stats() -> Dict[str, Any]:
    try:
        return _get_store().stats()
    except Exception:
        return {"available": False}


# ───────────────────────────── الواجهة الرئيسية ───────────────────────────

def build_plan(text: str, max_steps: int = MAX_STEPS) -> Optional[ReasoningPlan]:
    """بناء خطة لسؤال معقد — تعيد None للأسئلة البسيطة أو عند أي فشل."""
    try:
        t = text.strip()
        if not is_complex_question(t):
            return None
        with _PLAN_LOCK:
            cached = _PLAN_CACHE.get(t)
            if cached is not None:
                return cached
            plan = decompose(t, max_steps=max_steps)
            if len(_PLAN_CACHE) >= 2000:
                _PLAN_CACHE.clear()
            _PLAN_CACHE[t] = plan
        return plan
    except Exception:
        return None


def plan_system_prompt(text: str) -> Optional[str]:
    """رسالة نظام تتضمن الخطة — تُلحق قبل النافذة الأخيرة للنموذج.
    تعيد None إذا كان السؤال بسيطًا أو حدث أي خطأ (السلوك الأصلي)."""
    try:
        plan = build_plan(text)
        if plan is None:
            return None
        prompt = synthesize(text, plan)
        record_trace(plan, used=True)
        return prompt
    except Exception:
        return None


def reset_cache() -> None:
    """إعادة تعيين الكاش — للاختبار فقط."""
    with _PLAN_LOCK:
        _PLAN_CACHE.clear()
