"""
Phase 6 – Meta Reasoner
=========================
A high-level thinking layer that reflects on past decisions,
explains them, and proposes improvements.

Asks:
  • Why was this decision made?
  • Was there a better option?
  • What did we learn from it?

Usage:
  reasoner = MetaReasoner(memory_engine, scoring_engine, knowledge_store)
  analysis = reasoner.reflect(decision_context)
  explanation = reasoner.explain_route(path)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class DecisionExplanation:
    """Explanation of a single past or proposed decision."""

    def __init__(
        self,
        decision_id: str,
        decision_type: str,
        subject: str,
        chosen_option: str,
        reasoning: List[str],
        alternatives_considered: List[dict],
        confidence: float,
        lessons_learned: List[str],
    ):
        self.decision_id = decision_id
        self.decision_type = decision_type
        self.subject = subject
        self.chosen_option = chosen_option
        self.reasoning = reasoning
        self.alternatives_considered = alternatives_considered
        self.confidence = confidence
        self.lessons_learned = lessons_learned
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type,
            "subject": self.subject,
            "chosen_option": self.chosen_option,
            "reasoning": self.reasoning,
            "alternatives_considered": self.alternatives_considered,
            "confidence": round(self.confidence, 4),
            "lessons_learned": self.lessons_learned,
            "created_at": self.created_at,
        }


class MetaReasonerInsight:
    """A high-level insight produced by the meta reasoner."""

    def __init__(
        self,
        insight_type: str,   # pattern / warning / opportunity / lesson
        title: str,
        body: str,
        evidence: List[str],
        recommended_action: Optional[str] = None,
        priority: int = 5,
    ):
        self.insight_id = f"insight_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self.insight_type = insight_type
        self.title = title
        self.body = body
        self.evidence = evidence
        self.recommended_action = recommended_action
        self.priority = priority
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "insight_id": self.insight_id,
            "insight_type": self.insight_type,
            "priority": self.priority,
            "title": self.title,
            "body": self.body,
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
            "created_at": self.created_at,
        }


class MetaReasoner:
    """
    Phase 6: Meta-Reasoning Layer.

    Provides explanation, reflection, and learning capabilities on top
    of the existing AI decision infrastructure.
    """

    def __init__(
        self,
        memory_engine=None,
        scoring_engine=None,
        knowledge_store=None,
        evolution_engine=None,
        governance_layer=None,
    ):
        self._memory = memory_engine
        self._scoring = scoring_engine
        self._knowledge = knowledge_store
        self._evolution = evolution_engine
        self._governance = governance_layer
        self._explanations: List[DecisionExplanation] = []
        self._insights: List[MetaReasonerInsight] = []
        logger.info("MetaReasoner initialised (Phase 6)")

    # ── Route explanation ─────────────────────────────────────────────────

    def explain_route(
        self,
        path: List[str],
        node_names: Optional[List[str]] = None,
        goal: Optional[str] = None,
    ) -> DecisionExplanation:
        """Explain why a specific routing path was chosen."""
        names = node_names or path
        path_str = " → ".join(n[:8] if len(n) > 8 else n for n in names)

        reasoning: List[str] = []
        alternatives: List[dict] = []
        confidence = 0.80

        # Pull scoring evidence
        if self._scoring:
            try:
                scores = self._scoring.list_scores()
                path_scores = [
                    s for s in scores
                    if s.get("source_id") in path or s.get("target_id") in path
                ]
                if path_scores:
                    avg_sr = sum(s.get("success_rate", 0) for s in path_scores) / len(path_scores)
                    reasoning.append(
                        f"Historical success rate along this path: {avg_sr:.1%}"
                    )
                    confidence = min(0.98, confidence + avg_sr * 0.1)
            except Exception:
                pass

        # Pull memory evidence
        if self._memory:
            try:
                routes = self._memory.all_routes()
                path_key = "→".join(path)
                match = next((r for r in routes if r.get("path_key") == path_key), None)
                if match:
                    reasoning.append(
                        f"Route memory: {match.get('runs', 0)} prior runs, "
                        f"health={match.get('health', 'unknown')}"
                    )
                    sr = match.get("success_rate", 0.0)
                    confidence = min(0.98, confidence + sr * 0.05)
                    if match.get("runs", 0) == 0:
                        reasoning.append("No prior runs — confidence based on semantic matching only")
            except Exception:
                pass

        if not reasoning:
            reasoning = [
                "Selected via semantic matching against goal",
                "No historical data available — using prior probability",
            ]

        lessons: List[str] = self._extract_lessons_for_path(path)

        explanation = DecisionExplanation(
            decision_id=f"route_exp_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            decision_type="route_selection",
            subject=goal or "unknown goal",
            chosen_option=path_str,
            reasoning=reasoning,
            alternatives_considered=alternatives,
            confidence=confidence,
            lessons_learned=lessons,
        )
        self._explanations.append(explanation)
        return explanation

    # ── Reflection ────────────────────────────────────────────────────────

    def reflect(self, lookback_runs: int = 50) -> List[MetaReasonerInsight]:
        """
        Reflect on recent execution history and produce actionable insights.
        """
        insights: List[MetaReasonerInsight] = []

        insights.extend(self._detect_failure_patterns(lookback_runs))
        insights.extend(self._detect_latency_trends())
        insights.extend(self._detect_underused_capabilities())
        insights.extend(self._detect_evolution_opportunities())

        insights.sort(key=lambda i: i.priority)
        self._insights.extend(insights)
        return insights

    # ── Internal analysis ─────────────────────────────────────────────────

    def _detect_failure_patterns(self, lookback: int) -> List[MetaReasonerInsight]:
        insights: List[MetaReasonerInsight] = []
        if not self._memory:
            return insights
        try:
            routes = self._memory.all_routes()
            for route in routes:
                sr = route.get("success_rate", 1.0)
                runs = route.get("runs", 0)
                if runs >= 5 and sr < 0.60:
                    insights.append(MetaReasonerInsight(
                        insight_type="warning",
                        priority=1,
                        title=f"Chronic failure on route {route.get('path_key', '')[:30]}",
                        body=(
                            f"Route '{route.get('path_key', '')}' has failed "
                            f"{(1 - sr):.0%} of {runs} executions. "
                            "Consider rerouting or replacing a node."
                        ),
                        evidence=[f"success_rate={sr:.2%}", f"runs={runs}"],
                        recommended_action="Trigger SelfOptimizer or EvolutionEngine",
                    ))
        except Exception as exc:
            logger.debug(f"Failure pattern detection error: {exc}")
        return insights

    def _detect_latency_trends(self) -> List[MetaReasonerInsight]:
        insights: List[MetaReasonerInsight] = []
        if not self._scoring:
            return insights
        try:
            scores = self._scoring.list_scores()
            slow = [s for s in scores if s.get("avg_latency_ms", 0) > 800]
            if slow:
                worst = max(slow, key=lambda s: s.get("avg_latency_ms", 0))
                insights.append(MetaReasonerInsight(
                    insight_type="opportunity",
                    priority=2,
                    title="High-latency connection detected",
                    body=(
                        f"Connection {worst.get('source_id','?')[:8]} → "
                        f"{worst.get('target_id','?')[:8]} "
                        f"averages {worst.get('avg_latency_ms',0):.0f} ms. "
                        "Parallelisation or caching could help."
                    ),
                    evidence=[
                        f"avg_latency_ms={worst.get('avg_latency_ms',0):.0f}",
                        f"runs={worst.get('total_runs',0)}",
                    ],
                    recommended_action="Consider swarm parallelisation via SwarmCoordinator",
                ))
        except Exception as exc:
            logger.debug(f"Latency trend error: {exc}")
        return insights

    def _detect_underused_capabilities(self) -> List[MetaReasonerInsight]:
        insights: List[MetaReasonerInsight] = []
        if not self._knowledge:
            return insights
        try:
            marketplace = self._knowledge.load("capability_marketplace") or {}
            ads = marketplace.get("advertisements", [])
            for ad in ads:
                if ad.get("usage_count", 0) == 0:
                    insights.append(MetaReasonerInsight(
                        insight_type="pattern",
                        priority=4,
                        title=f"Unused capability: {ad.get('capability', '?')}",
                        body=(
                            f"Capability '{ad.get('capability')}' is registered "
                            "but has never been used. Consider retiring or promoting it."
                        ),
                        evidence=["usage_count=0"],
                        recommended_action="Advertise capability in MarketplaceEngine or retire",
                    ))
        except Exception:
            pass
        return insights

    def _detect_evolution_opportunities(self) -> List[MetaReasonerInsight]:
        insights: List[MetaReasonerInsight] = []
        if not self._evolution:
            return insights
        try:
            history = self._evolution.history(limit=5)
            for cycle in history:
                if cycle.get("summary", {}).get("gaps_found", 0) > 3:
                    insights.append(MetaReasonerInsight(
                        insight_type="opportunity",
                        priority=3,
                        title=f"Multiple gaps found in evolution cycle {cycle.get('cycle_number')}",
                        body=(
                            f"{cycle['summary']['gaps_found']} gaps were detected. "
                            f"Only {cycle['summary']['services_approved']} services were approved. "
                            "Consider lowering governance thresholds or pre-registering templates."
                        ),
                        evidence=[str(cycle.get("summary", {}))],
                        recommended_action="Review AIGovernanceLayer thresholds",
                    ))
        except Exception:
            pass
        return insights

    def _extract_lessons_for_path(self, path: List[str]) -> List[str]:
        lessons: List[str] = []
        if not self._memory:
            return lessons
        try:
            routes = self._memory.all_routes()
            path_key = "→".join(path)
            match = next((r for r in routes if r.get("path_key") == path_key), None)
            if match:
                if match.get("success_rate", 1.0) > 0.9:
                    lessons.append("This route has a proven track record — high confidence.")
                elif match.get("success_rate", 1.0) < 0.5:
                    lessons.append("Route has failed frequently — monitor closely.")
        except Exception:
            pass
        return lessons

    # ── Public query ─────────────────────────────────────────────────────

    def ask(self, question: str) -> dict:
        """
        Answer a meta-reasoning question about the system.
        Supports:
          "why did you choose X?"   → latest explanation for X
          "what did you learn?"     → recent insights
          "is there a better path?" → placeholder for future planning
        """
        q = question.lower()
        if "why" in q or "explain" in q or "reason" in q:
            latest = self._explanations[-1].to_dict() if self._explanations else None
            return {
                "question": question,
                "answer_type": "explanation",
                "explanation": latest or "No decisions recorded yet",
            }
        if "learn" in q or "insight" in q or "pattern" in q:
            return {
                "question": question,
                "answer_type": "insights",
                "insights": [i.to_dict() for i in self._insights[-5:]],
            }
        return {
            "question": question,
            "answer_type": "unknown",
            "message": "Question not understood. Try asking 'why', 'explain', or 'what did you learn'.",
        }

    def recent_explanations(self, limit: int = 10) -> List[dict]:
        return [e.to_dict() for e in self._explanations[-limit:]]

    def recent_insights(self, limit: int = 10) -> List[dict]:
        return [i.to_dict() for i in self._insights[-limit:]]

    def summary(self) -> dict:
        return {
            "total_explanations": len(self._explanations),
            "total_insights": len(self._insights),
            "insight_types": {
                t: sum(1 for i in self._insights if i.insight_type == t)
                for t in ("pattern", "warning", "opportunity", "lesson")
            },
        }
