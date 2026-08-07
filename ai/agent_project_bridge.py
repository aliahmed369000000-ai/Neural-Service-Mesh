"""
Agent ↔ Project Bridge — نقطة دمج موحّدة للوكيل في المشروع
==========================================================
أي واجهة (وكلاء AI، الوكيل الموحّد، MCP، سكربتات) تستدعي:

    from ai.agent_project_bridge import dispatch_agent_message
    reply = dispatch_agent_message(text)

تمرّ على:
  1) أوامر model_training_agent (كل الطبقات المدمجة فيه)
  2) صيانة إن وُجدت
  3) وإلا None ليُكمل المسار LLM العادي
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger("AgentProjectBridge")


def dispatch_agent_message(user_input: str) -> Optional[str]:
    text = (user_input or "").strip()
    if not text:
        return None

    # 1) وكيل التدريب + الطبقات المربوطة به (RL, CKG quality, social swarm, …)
    try:
        from ai.model_training_agent import handle_training_command
        r = handle_training_command(text)
        if r is not None:
            return r
    except Exception as e:
        logger.warning("training dispatch: %s", e)

    # 2) وكيل الصيانة
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
    """يعيد (الرد, مصدر الشارة)."""
    text = (user_input or "").strip()
    if not text:
        return None, ""
    try:
        from ai.model_training_agent import handle_training_command
        r = handle_training_command(text)
        if r is not None:
            return r, "🧬 Project Agent"
    except Exception:
        pass
    try:
        from ai.code_agent import handle_maintenance_command
        r = handle_maintenance_command(text)
        if r is not None:
            return r, "🛠️ Maintenance"
    except Exception:
        pass
    return None, ""


def agent_integration_status() -> dict:
    st = {"bridge": True, "components": {}}
    for name, path in (
        ("model_training_agent", "ai.model_training_agent"),
        ("reasoning_pipeline", "ai.reasoning_pipeline"),
        ("social_swarm", "ai.social_swarm"),
        ("reinforcement_learning", "ai.reinforcement_learning"),
        ("ckg_quality", "ai.ckg_quality_tool"),
        ("sovereignty_loop", "ai.sovereignty_loop"),
    ):
        try:
            __import__(path)
            st["components"][name] = True
        except Exception as e:
            st["components"][name] = str(e)
    return st
