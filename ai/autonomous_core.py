"""
ai/autonomous_core.py
======================
منسّق طبقة الأمان الذاتي (Safety/Governance Orchestrator).

هذا الملف يربط الوحدات التالية التي كانت مكتوبة بالكامل لكنها لم تكن
مستوردة أو مُفعّلة في أي مكان بالتطبيق:

    ai/governor.py          -> AIGovernanceLayer   (يمنع الحلقات وتوسّع غير محكوم)
    ai/immune_system.py     -> ImmuneSystem        (يفحص كل بيانات واردة)
    ai/evolution_ethics.py  -> EvolutionEthics      (يمنع تجاوز حدود التطور الذاتي)

كل الوحدات اختيارية (best-effort): إذا فشل استيراد أي وحدة، يستمر
البقية بالعمل بدون كسر التطبيق الرئيسي.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("NeuralServiceMesh.AutonomousCore")

_instance: Optional["AutonomousCore"] = None


class AutonomousCore:
    """يجمع طبقات الحوكمة/المناعة/الأخلاقيات في كائن واحد قابل للاستخدام من NSMAgent."""

    def __init__(self) -> None:
        self.governor = self._safe_init_governor()
        self.immune = self._safe_init_immune()
        self.ethics = self._safe_init_ethics()

    @staticmethod
    def _safe_init_governor():
        try:
            from ai.governor import AIGovernanceLayer
            return AIGovernanceLayer()
        except Exception as e:
            logger.warning(f"تعذّر تفعيل AIGovernanceLayer: {e}")
            return None

    @staticmethod
    def _safe_init_immune():
        try:
            from ai.immune_system import ImmuneSystem
            return ImmuneSystem()
        except Exception as e:
            logger.warning(f"تعذّر تفعيل ImmuneSystem: {e}")
            return None

    def _safe_init_ethics(self):
        try:
            from ai.evolution_ethics import EvolutionEthics
            # نربط الأخلاقيات بنظام المناعة إن توفّر (immune_rejection_threshold logic)
            return EvolutionEthics(immune_system=self.immune)
        except Exception as e:
            logger.warning(f"تعذّر تفعيل EvolutionEthics: {e}")
            return None

    # ── واجهة موحّدة يستخدمها NSMAgent ─────────────────────────────────

    def inspect_input(self, source: str, content: str) -> Dict[str, Any]:
        """يفحص مُدخل وارد عبر نظام المناعة قبل قبوله. آمن حتى لو المناعة غير مفعّلة."""
        if self.immune is None:
            return {"allowed": True, "action": "pass", "flags": ["immune_disabled"]}
        try:
            return self.immune.inspect({"source": source, "content": content})
        except Exception as e:
            logger.warning(f"immune.inspect فشل: {e}")
            return {"allowed": True, "action": "pass", "flags": ["immune_error"]}

    def check_evolution_action(self, action_type: str, params: Optional[dict] = None) -> Dict[str, Any]:
        """يفحص فعل تطوري (حذف ذاكرة، تغيير معمارية...) عبر الأخلاقيات التطورية."""
        if self.ethics is None:
            return {"allowed": True, "reason": "ethics_disabled"}
        try:
            return self.ethics.check(action_type, params or {})
        except Exception as e:
            logger.warning(f"ethics.check فشل: {e}")
            return {"allowed": True, "reason": f"ethics_error: {e}"}

    def evaluate_governance(self, action: str, context: Optional[dict] = None) -> Dict[str, Any]:
        """يقيّم فعلاً حراً (routing/توسّع) عبر طبقة الحوكمة Phase 5."""
        if self.governor is None:
            return {"allowed": True, "reason": "governance_disabled"}
        try:
            decision = self.governor.evaluate(action, context or {})
            return decision.to_dict()
        except Exception as e:
            logger.warning(f"governor.evaluate فشل: {e}")
            return {"allowed": True, "reason": f"governance_error: {e}"}

    def get_status(self) -> Dict[str, Any]:
        """حالة موجزة لكل الطبقات — تُستخدم في رد الوكيل على استعلام 'حالة الحوكمة'."""
        return {
            "governance_active": self.governor is not None,
            "immune_active": self.immune is not None,
            "ethics_active": self.ethics is not None,
        }


def get_autonomous_core() -> AutonomousCore:
    """Singleton — يُنشأ مرة واحدة فقط عند أول استدعاء (lazy)."""
    global _instance
    if _instance is None:
        _instance = AutonomousCore()
    return _instance
