"""
Agent ↔ Project Bridge — نقطة دمج موحّدة للوكيل في المشروع
==========================================================
أي واجهة (وكلاء AI، الوكيل الموحّد، المحادثة، MCP) تستدعي:

    from ai.agent_project_bridge import dispatch_agent_message
    reply = dispatch_agent_message(text)

الترتيب:
  1) أوامر model_training_agent (تدريب، CKG، سرب، RL، اقتصاد، حضارة…)
  2) أوامر الصيانة إن وُجدت
  3) None → يكمل المسار بـ LLM عادي
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("AgentProjectBridge")


def dispatch_agent_message(user_input: str) -> Optional[str]:
    text = (user_input or "").strip()
    if not text:
        return None

    try:
        from ai.model_training_agent import handle_training_command
        r = handle_training_command(text)
        if r is not None:
            return r
    except Exception as e:
        logger.warning("training dispatch: %s", e)

    try:
        from ai.code_agent import handle_maintenance_command
        r = handle_maintenance_command(text)
        if r is not None:
            return r
    except Exception:
        try:
            from ai.maintenance_agent import handle_maintenance_command as _hm
            r = _hm(text)
            if r is not None:
                return r
        except Exception:
            pass

    return None


def dispatch_with_meta(user_input: str) -> Tuple[Optional[str], str]:
    """يعيد (الرد, شارة المصدر)."""
    text = (user_input or "").strip()
    if not text:
        return None, ""
    try:
        from ai.model_training_agent import handle_training_command
        r = handle_training_command(text)
        if r is not None:
            return r, "🧬 Project Agent"
    except Exception as e:
        logger.warning("training dispatch: %s", e)
    try:
        from ai.code_agent import handle_maintenance_command
        r = handle_maintenance_command(text)
        if r is not None:
            return r, "🛠️ Maintenance"
    except Exception:
        try:
            from ai.maintenance_agent import handle_maintenance_command as _hm
            r = _hm(text)
            if r is not None:
                return r, "🛠️ Maintenance"
        except Exception:
            pass
    return None, ""


def agent_integration_status() -> Dict[str, Any]:
    st: Dict[str, Any] = {"bridge": True, "components": {}}
    for name, path in (
        ("model_training_agent", "ai.model_training_agent"),
        ("reasoning_pipeline", "ai.reasoning_pipeline"),
        ("social_swarm", "ai.social_swarm"),
        ("reinforcement_learning", "ai.reinforcement_learning"),
        ("ckg_quality", "ai.ckg_quality_tool"),
        ("sovereignty_loop", "ai.sovereignty_loop"),
        ("kaggle_provider", "ai.kaggle_provider"),
    ):
        try:
            __import__(path)
            st["components"][name] = True
        except Exception as e:
            st["components"][name] = str(e)
    return st
