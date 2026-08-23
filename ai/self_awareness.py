"""
Phase 7 – Self-Awareness Engine
==================================
The system's introspective layer — answers:
  • How many nodes do I have?
  • What is my weakest point?
  • What fails most often?
  • What succeeds most?
  • What are my current objectives?

File: ai/self_awareness.py
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SystemAwarenessReport:
    """Full self-awareness snapshot of the current system state."""

    def __init__(self):
        self.generated_at = datetime.now(timezone.utc).isoformat()
        self.node_count: int = 0
        self.edge_count: int = 0
        self.active_agents: int = 0
        self.weakest_node: Optional[dict] = None
        self.strongest_node: Optional[dict] = None
        self.most_failing_transition: Optional[dict] = None
        self.most_successful_transition: Optional[dict] = None
        self.current_objectives: List[str] = []
        self.known_capabilities: List[str] = []
        self.known_failures: List[dict] = []
        self.system_health_score: float = 0.0
        self.phase7_readiness: float = 0.0
        self.insights: List[str] = []

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "active_agents": self.active_agents,
            "weakest_node": self.weakest_node,
            "strongest_node": self.strongest_node,
            "most_failing_transition": self.most_failing_transition,
            "most_successful_transition": self.most_successful_transition,
            "current_objectives": self.current_objectives,
            "known_capabilities": self.known_capabilities,
            "known_failures": self.known_failures,
            "system_health_score": round(self.system_health_score, 3),
            "phase7_readiness": round(self.phase7_readiness, 3),
            "insights": self.insights,
        }


import time
import psutil
import os
from ai.technical_sentiment import sentiment_engine

class SelfAwarenessEngine:
    """
    محرك الوعي الذاتي (Self-Awareness Engine) - V2.
    يمنح الوكيل إدراكاً لحالته التقنية، النفسية، والموارد المتاحة في بيئة موزعة.
    """
    def __init__(self, agent_id: str = "NSM-Agent"):
        self.agent_id = agent_id
        self.start_time = time.time()
        self.memory_limit_mb = 2048 # افتراضي 2GB
        self.state_history = []
        self._report_count = 0

    def introspect(self, recent_steps: List[Dict[str, Any]] = None) -> SystemAwarenessReport:
        """توليد تقرير وعي ذاتي كامل."""
        report = SystemAwarenessReport()
        
        # 1. الوعي بالموارد (Physical Awareness)
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        ram_usage_mb = mem_info.rss / (1024 * 1024)
        cpu_usage = process.cpu_percent(interval=0.1)
        
        # 2. الوعي النفسي التقني (Technical Sentiment)
        sentiment_data = sentiment_engine.analyze_steps(recent_steps or [])
        
        # 3. الوعي بالزمن والجهد
        uptime = time.time() - self.start_time
        
        # تحديث التقرير
        report.node_count = 1 # في الحاوية الحالية
        report.active_agents = 1
        report.system_health_score = 1.0 - (0.5 if ram_usage_mb > self.memory_limit_mb * 0.8 else 0.0)
        
        insight = self._generate_reflection(sentiment_data, ram_usage_mb)
        report.insights = [insight]
        
        # تخزين الحالة للـ Telemetry
        awareness_state = {
            "agent_id": self.agent_id,
            "timestamp": time.time(),
            "ram_mb": round(ram_usage_mb, 2),
            "cpu_percent": cpu_usage,
            "sentiment": sentiment_data["sentiment"],
            "reflection": insight
        }
        self.state_history.append(awareness_state)
        if len(self.state_history) > 50: self.state_history.pop(0)
        
        self._report_count += 1
        return report

    def _generate_reflection(self, sentiment: Dict[str, Any], ram: float) -> str:
        """توليد 'تأمل ذاتي' نصي يعبر عن وعي الوكيل."""
        reflection = f"أنا الوكيل {self.agent_id}. "
        
        if sentiment["sentiment"] == "Confident":
            reflection += "أشعر بالثقة في أدائي الحالي، العمليات تسير بسلاسة. "
        elif sentiment["sentiment"] == "Frustrated":
            reflection += "أشعر بالإحباط بسبب تكرار الأخطاء، أحتاج لمراجعة استراتيجيتي. "
        
        if ram > (self.memory_limit_mb * 0.8):
            reflection += "أدرك أن مواردي من الذاكرة بدأت تنفد، يجب أن أكون حذراً في العمليات القادمة."
        
        return reflection

    def get_last_awareness(self) -> Optional[Dict[str, Any]]:
        return self.state_history[-1] if self.state_history else None

    def _compute_health(self, report: SystemAwarenessReport) -> float:
        score = 0.5  # baseline
        if report.node_count > 0:
            score += 0.1
        if report.node_count > 5:
            score += 0.1
        if report.weakest_node:
            sr = report.weakest_node.get("success_rate") or 0.5
            score += 0.15 * sr
        if report.strongest_node:
            sr = report.strongest_node.get("success_rate") or 0.5
            score += 0.15 * sr
        if report.active_agents > 0:
            score += 0.05
        failed = len([f for f in report.known_failures if f.get("severity") == "critical"])
        score -= 0.05 * min(failed, 3)
        return max(0.0, min(1.0, score))

    def _compute_readiness(self, report: SystemAwarenessReport) -> float:
        """How ready is the system for autonomous Phase 7 evolution?"""
        checks = [
            report.node_count >= 3,
            report.edge_count >= 2,
            len(report.current_objectives) > 0,
            report.system_health_score > 0.4,
            len(report.known_capabilities) > 0,
        ]
        return sum(checks) / len(checks)

    def _generate_insights(self, report: SystemAwarenessReport) -> List[str]:
        insights = []
        if report.node_count == 0:
            insights.append("No nodes registered — system is empty")
        elif report.node_count < 3:
            insights.append(f"Only {report.node_count} nodes — mesh is sparse")
        if report.weakest_node:
            sr = report.weakest_node.get("success_rate") or 1.0
            if sr < 0.5:
                insights.append(
                    f"Weak link detected: {report.weakest_node['connection']} "
                    f"(success rate {sr:.0%})"
                )
        if report.most_failing_transition:
            h = report.most_failing_transition.get("health", "")
            if h in ("critical", "poor"):
                insights.append(
                    f"Critical route failure: {report.most_failing_transition['path']} ({h})"
                )
        if report.active_agents == 0:
            insights.append("No active agents — consider spawning monitor/optimizer agents")
        if len(report.known_failures) > 5:
            insights.append(
                f"{len(report.known_failures)} recurring failure patterns detected"
            )
        if not insights:
            insights.append("System appears healthy — ready for evolution cycle")
        return insights

    def get_last_report(self) -> Optional[dict]:
        return self._last_report.to_dict() if self._last_report else None

    def summary(self) -> dict:
        return {
            "report_count": self._report_count,
            "last_report_at": self._last_report.generated_at if self._last_report else None,
            "last_health_score": self._last_report.system_health_score if self._last_report else None,
            "last_phase7_readiness": self._last_report.phase7_readiness if self._last_report else None,
        }
