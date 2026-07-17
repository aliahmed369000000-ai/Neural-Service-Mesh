"""
NSM Router Bridge — وصل RoutingEngine/ScoringEngine/MemoryEngine/LearningValidator
بمسارات التوليد الفعلية في streamlit_app.py.

العقد المعرّفة (مسارات التوليد الحقيقية):
  NSM_INPUT_NODE   = "nsm:input"        ← نقطة البداية (الإدخال)
  NODE_OPENROUTER  = "nsm:openrouter"   ← _or_stream() عبر OpenRouter
  NODE_AGENT       = "nsm:agent"        ← NSMAgent.run_stream()
  NODE_FREE_ROUTER = "nsm:free_router"  ← bot.chat() / free_router fallback

آلية التوجيه:
  عند كل طلب يُحدَّد أولاً مجموعة العقد *المتاحة فعلاً* (openrouter إذا
  كان المفتاح موجوداً، agent إذا كانت الخدمة تعمل، free_router دائماً).
  ثم يختار select_node() أفضلها بناءً على ConnectionScore التاريخي من
  ScoringEngine. بعد كل رد يُسجَّل record_result() في ScoringEngine +
  MemoryEngine. LearningValidator يقرأ هذه البيانات لإثبات التحسّن.
"""
from __future__ import annotations

import logging
import time
from typing import List

logger = logging.getLogger(__name__)

# ── أسماء العقد الثابتة ───────────────────────────────────────────────────
NSM_INPUT_NODE   = "nsm:input"
NODE_OPENROUTER  = "nsm:openrouter"
NODE_AGENT       = "nsm:agent"
NODE_FREE_ROUTER = "nsm:free_router"

ALL_NODES: List[str] = [NODE_OPENROUTER, NODE_AGENT, NODE_FREE_ROUTER]

# ── مثيلات مفردة (singleton) ─────────────────────────────────────────────
_scoring    = None
_memory     = None
_routing    = None
_validator  = None
_initialized = False


def _init() -> None:
    """تهيئة المحركات مرة واحدة (lazy singleton)."""
    global _scoring, _memory, _routing, _validator, _initialized
    if _initialized:
        return
    try:
        from ai.scoring_engine     import ScoringEngine
        from ai.memory_engine      import MemoryEngine
        from ai.routing_engine     import RoutingEngine
        from ai.learning_validator import LearningValidator

        _scoring   = ScoringEngine(db_path="./data/mesh.db")
        _memory    = MemoryEngine(db_path="./data/mesh.db")
        _routing   = RoutingEngine(scoring_engine=_scoring, memory_engine=_memory)
        _validator = LearningValidator(
            memory_engine=_memory,
            scoring_engine=_scoring,
        )
        _initialized = True
        logger.info(
            "NSM Router Bridge: تهيئة ناجحة — "
            "ScoringEngine + MemoryEngine + RoutingEngine + LearningValidator"
        )
    except Exception as exc:
        logger.error(f"NSM Router Bridge: فشل التهيئة — {exc}")


def is_ready() -> bool:
    """هل تمّت التهيئة بنجاح؟"""
    _init()
    return _initialized


# ── واجهة التوجيه ──────────────────────────────────────────────────────────

def select_node(available_nodes: List[str]) -> str:
    """
    اختر أفضل عقدة من القائمة المتاحة بناءً على السجل التاريخي.

    المنطق:
      - لكل عقدة يُسأَل ScoringEngine عن connection_score للحافة
        (NSM_INPUT_NODE → node).
      - العقدة ذات أعلى درجة تُختار.
      - في غياب بيانات (أول تشغيل) الدرجة المحايدة = 50 لجميعها → أول
        عقدة في القائمة تُختار (الأعلى أولوية بحسب ترتيب المُعطى).
    """
    _init()
    if not available_nodes:
        return NODE_FREE_ROUTER
    if not _scoring:
        return available_nodes[0]

    best_node  = available_nodes[0]
    best_score = -1.0
    for node in available_nodes:
        cs    = _scoring.get_score(NSM_INPUT_NODE, node)
        score = cs.connection_score
        logger.debug(f"NSM Router Bridge: العقدة={node}  درجة={score:.2f}")
        if score > best_score:
            best_score = score
            best_node  = node

    logger.info(
        f"NSM Router Bridge: اختار «{best_node}» "
        f"(درجة={best_score:.2f} من {len(available_nodes)} عقد متاحة)"
    )
    return best_node


def record_result(
    node_id: str,
    success: bool,
    latency_ms: float,
) -> None:
    """
    سجّل نتيجة استجابة حقيقية في ScoringEngine + MemoryEngine.
    يُستدعى مباشرةً بعد كل رد فعلي.
    """
    _init()
    run_result = {
        "run_id":            f"{node_id}:{int(time.time() * 1000)}",
        "status":            "success" if success else "failure",
        "path":              [NSM_INPUT_NODE, node_id],
        "total_duration_ms": latency_ms,
        "steps": [{
            "node_id":     node_id,
            "node_name":   node_id.replace("nsm:", ""),
            "status":      "success" if success else "failure",
            "duration_ms": latency_ms,
        }],
    }

    if _scoring:
        try:
            _scoring.record_run(run_result)
        except Exception as exc:
            logger.warning(f"NSM Router Bridge ScoringEngine.record_run: {exc}")

    if _memory:
        try:
            _memory.learn_from_run(run_result)
        except Exception as exc:
            logger.warning(f"NSM Router Bridge MemoryEngine.learn_from_run: {exc}")


# ── واجهة البيانات لعرض الواجهة ───────────────────────────────────────────

def get_scores_summary() -> List[dict]:
    """قائمة درجات الاتصال لكل حافة عقدة."""
    _init()
    if not _scoring:
        return []
    try:
        return _scoring.list_scores()
    except Exception as exc:
        logger.warning(f"NSM Router Bridge get_scores_summary: {exc}")
        return []


def get_learning_report() -> dict:
    """
    تقرير التعلم الكامل:
      proof      ← LearningValidator.prove_learning()
      curve      ← LearningValidator.get_learning_curve()
      reputation ← LearningValidator.get_node_reputation()
    """
    _init()
    if not _validator:
        return {"error": "LearningValidator غير مهيأ"}
    try:
        proof      = _validator.prove_learning()
        curve      = _validator.get_learning_curve()
        reputation = _validator.get_node_reputation()
        return {
            "proof":      proof,
            "curve":      curve,
            "reputation": reputation,
        }
    except Exception as exc:
        logger.error(f"NSM Router Bridge get_learning_report: {exc}")
        return {"error": str(exc)}


def get_node_scores_for_display() -> List[dict]:
    """
    درجات العقد الثلاث بصيغة مناسبة للعرض في الواجهة.
    """
    _init()
    result = []
    labels = {
        NODE_OPENROUTER:  ("🌐 OpenRouter",   "openrouter"),
        NODE_AGENT:       ("🧠 NSM Agent",    "agent"),
        NODE_FREE_ROUTER: ("⚡ Free Router",  "free_router"),
    }
    for node_id in ALL_NODES:
        label, key = labels[node_id]
        if _scoring:
            cs = _scoring.get_score(NSM_INPUT_NODE, node_id)
            result.append({
                "node_id":        node_id,
                "label":          label,
                "connection_score": cs.connection_score,
                "total_runs":     cs.total_runs,
                "success_rate":   round(cs.success_rate * 100, 1),
                "avg_latency_ms": round(cs.avg_latency_ms, 0),
            })
        else:
            result.append({
                "node_id":        node_id,
                "label":          label,
                "connection_score": 50.0,
                "total_runs":     0,
                "success_rate":   50.0,
                "avg_latency_ms": 0.0,
            })
    return result
