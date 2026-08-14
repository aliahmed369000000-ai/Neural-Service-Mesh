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

لا يستدعي أي API خارجي — التحليل نمطي محلي بالكامل (مثل agent_reflection).
التدهور: فشل كامل → كتلة _PAR_OK في app_core تعطل الوحدة بصمت.

الجداول:
  par_records: id, task_id, role, goal, steps_json, risks_json, verdict,
               confidence, revised_n, memory_recall_json, created_at

الذاكرة الحسية للأدوار (Role Sensory Memory):
  قبل جلسة التفكير، يستحضر المحرك جلسات التفكير السابقة لنفس الدور
  أو لأهداف متشابهة لفظيًا، ويستخدمها لرفع الثقة عند وجود سوابق
  ناجحة (proceed) أو خفضها عند تكرار المراجعات (revise)، وترجع
  الذكريات المستحضرة مع نتيجة الجلسة في حقل recalled_memories.
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
            "recalled_memories": self.recalled_memories,
            "verdict": self.verdict,
            "revisions": self.revisions,
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
                proc = sum(1 for m in plan.recalled_memories
                           if m["verdict"] == "proceed")
                rev = sum(1 for m in plan.recalled_memories
                          if m["verdict"] == "revise")
                plan.memory_effect = max(-0.2, min(
                    0.2, 0.05 * proc - 0.08 * rev))
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
                1.0, plan.confidence + plan.memory_effect))
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
                           created_at FROM par_records
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
                        })
                    if len(out) >= top_k:
                        break
        except Exception as exc:
            logger.warning("PAR recall failed: %s", exc)
        return out

    def latest(self, task_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._lock, _connect(self._db) as conn:
                row = conn.execute("""
                    SELECT goal, role, steps_json, risks_json, verdict,
                           confidence, revised_n, created_at
                    FROM par_records WHERE task_id = ?
                    ORDER BY created_at DESC LIMIT 1
                """, (task_id,)).fetchone()
                if not row:
                    return None
                return {"task_id": task_id, "goal": row[0], "role": row[1],
                        "expected_steps": json.loads(row[2] or "[]"),
                        "risks": json.loads(row[3] or "[]"),
                        "verdict": row[4], "confidence": row[5],
                        "revised_n": row[6], "created_at": row[7]}
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


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
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
