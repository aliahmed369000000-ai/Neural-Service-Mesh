"""
Phase 6 – Agent Factory
========================
Dynamically creates, registers, and manages autonomous AI agents.

Each agent is a specialized worker with:
  - A defined role (ResearchAgent, TranslationAgent, ReviewAgent, etc.)
  - Capability profile used in the marketplace
  - Lifecycle management (spawn / retire / replace)

Usage:
  factory = AgentFactory(registry, knowledge_store)
  agent = factory.spawn("ResearchAgent")
  factory.retire(agent.agent_id)
"""
from __future__ import annotations

import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ── Agent roles catalogue ────────────────────────────────────────────────

AGENT_CATALOGUE: Dict[str, dict] = {
    "ResearchAgent": {
        "description": "Searches and aggregates knowledge from multiple sources",
        "capabilities": ["search", "summarize", "fact_check"],
        "input_schema": {"query": "str", "depth": "int"},
        "output_schema": {"findings": "list", "confidence": "float"},
        "tags": ["research", "knowledge", "search"],
    },
    "TranslationAgent": {
        "description": "Translates content between formats, languages, or schemas",
        "capabilities": ["translate", "transform", "normalize"],
        "input_schema": {"content": "any", "source_format": "str", "target_format": "str"},
        "output_schema": {"translated": "any", "mapping": "dict"},
        "tags": ["translation", "transform", "format"],
    },
    "ReviewAgent": {
        "description": "Validates, audits, and scores outputs from other agents",
        "capabilities": ["validate", "score", "audit"],
        "input_schema": {"artifact": "any", "criteria": "list"},
        "output_schema": {"passed": "bool", "score": "float", "feedback": "list"},
        "tags": ["review", "quality", "audit"],
    },
    "PlanningAgent": {
        "description": "Decomposes high-level goals into executable sub-tasks",
        "capabilities": ["plan", "decompose", "schedule"],
        "input_schema": {"goal": "str", "constraints": "dict"},
        "output_schema": {"tasks": "list", "timeline": "dict"},
        "tags": ["planning", "orchestration", "goals"],
    },
    "MonitorAgent": {
        "description": "Continuously monitors node health and performance metrics",
        "capabilities": ["monitor", "alert", "report"],
        "input_schema": {"node_ids": "list", "interval_s": "int"},
        "output_schema": {"health_report": "dict", "alerts": "list"},
        "tags": ["monitoring", "health", "metrics"],
    },
    "OptimizationAgent": {
        "description": "Identifies and applies performance improvements autonomously",
        "capabilities": ["optimize", "benchmark", "tune"],
        "input_schema": {"target": "str", "metric": "str"},
        "output_schema": {"improvements": "list", "before": "dict", "after": "dict"},
        "tags": ["optimization", "performance", "tuning"],
    },
}


class AgentInstance:
    """Represents a live agent spawned by the factory."""

    def __init__(
        self,
        role: str,
        spec: dict,
        config: Optional[dict] = None,
    ):
        self.agent_id = f"agent_{role.lower()}_{str(uuid.uuid4())[:8]}"
        self.role = role
        self.description = spec["description"]
        self.capabilities = spec["capabilities"]
        self.input_schema = spec["input_schema"]
        self.output_schema = spec["output_schema"]
        self.tags = spec["tags"]
        self.config = config or {}
        self.status = "active"        # active / idle / retired / error
        self.spawned_at = datetime.now(timezone.utc).isoformat()
        self.retired_at: Optional[str] = None
        self.task_count = 0
        self.success_count = 0
        self.error_count = 0
        self.last_task_at: Optional[str] = None
        self.performance_score: float = 1.0

    @property
    def success_rate(self) -> float:
        if self.task_count == 0:
            return 1.0
        return self.success_count / self.task_count

    def record_task(self, success: bool):
        self.task_count += 1
        self.last_task_at = datetime.now(timezone.utc).isoformat()
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        # Rolling performance score (EMA)
        outcome = 1.0 if success else 0.0
        self.performance_score = 0.9 * self.performance_score + 0.1 * outcome

    def retire(self):
        self.status = "retired"
        self.retired_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "description": self.description,
            "capabilities": self.capabilities,
            "tags": self.tags,
            "status": self.status,
            "spawned_at": self.spawned_at,
            "retired_at": self.retired_at,
            "task_count": self.task_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": round(self.success_rate, 4),
            "performance_score": round(self.performance_score, 4),
            "last_task_at": self.last_task_at,
        }


class AgentFactory:
    """
    Phase 6: Autonomous Agent Factory.

    Spawns specialised agents on demand, tracks their lifecycle,
    and retires underperforming instances.
    """

    def __init__(self, knowledge_store=None):
        self._agents: Dict[str, AgentInstance] = {}
        self._knowledge = knowledge_store
        self._spawn_history: List[dict] = []
        logger.info("AgentFactory initialised (Phase 6)")

    # ── Spawn ─────────────────────────────────────────────────────────────

    def spawn(self, role: str, config: Optional[dict] = None) -> AgentInstance:
        """Create a new agent of the given role."""
        if role not in AGENT_CATALOGUE:
            raise ValueError(
                f"Unknown agent role '{role}'. "
                f"Available: {list(AGENT_CATALOGUE.keys())}"
            )
        spec = AGENT_CATALOGUE[role]
        agent = AgentInstance(role, spec, config)
        self._agents[agent.agent_id] = agent

        record = {
            "event": "spawn",
            "agent_id": agent.agent_id,
            "role": role,
            "timestamp": agent.spawned_at,
        }
        self._spawn_history.append(record)
        logger.info(f"Spawned {role} → {agent.agent_id}")
        return agent

    def spawn_multiple(self, roles: List[str]) -> List[AgentInstance]:
        """Spawn a list of agents at once."""
        return [self.spawn(r) for r in roles]

    # ── Retire ────────────────────────────────────────────────────────────

    def retire(self, agent_id: str) -> bool:
        agent = self._agents.get(agent_id)
        if not agent or agent.status == "retired":
            return False
        agent.retire()
        self._spawn_history.append({
            "event": "retire",
            "agent_id": agent_id,
            "role": agent.role,
            "timestamp": agent.retired_at,
            "final_performance": agent.performance_score,
        })
        logger.info(f"Retired {agent.role} → {agent_id}")
        return True

    def replace(self, agent_id: str) -> Optional[AgentInstance]:
        """Retire a specific agent and spawn a fresh replacement."""
        old = self._agents.get(agent_id)
        if not old:
            return None
        role = old.role
        self.retire(agent_id)
        return self.spawn(role)

    # ── Query ─────────────────────────────────────────────────────────────

    def get(self, agent_id: str) -> Optional[AgentInstance]:
        return self._agents.get(agent_id)

    def list_active(self) -> List[AgentInstance]:
        return [a for a in self._agents.values() if a.status == "active"]

    def list_by_role(self, role: str) -> List[AgentInstance]:
        return [a for a in self._agents.values() if a.role == role]

    def list_by_capability(self, capability: str) -> List[AgentInstance]:
        return [
            a for a in self.list_active()
            if capability in a.capabilities
        ]

    def best_agent_for(self, capability: str) -> Optional[AgentInstance]:
        """Return the highest-performing active agent with a given capability."""
        candidates = self.list_by_capability(capability)
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.performance_score)

    # ── Auto-retirement ───────────────────────────────────────────────────

    def prune_underperformers(
        self, min_tasks: int = 5, min_score: float = 0.5
    ) -> List[str]:
        """
        Retire agents whose performance score is below threshold
        (only if they have enough tasks to judge).
        """
        retired_ids: List[str] = []
        for agent in list(self.list_active()):
            if agent.task_count >= min_tasks and agent.performance_score < min_score:
                self.retire(agent.agent_id)
                retired_ids.append(agent.agent_id)
                logger.warning(
                    f"Auto-retired {agent.role} {agent.agent_id} "
                    f"(score={agent.performance_score:.2f})"
                )
        return retired_ids

    # ── Catalogue ─────────────────────────────────────────────────────────

    @staticmethod
    def available_roles() -> List[str]:
        return list(AGENT_CATALOGUE.keys())

    @staticmethod
    def role_spec(role: str) -> Optional[dict]:
        return AGENT_CATALOGUE.get(role)

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> dict:
        active = self.list_active()
        role_counts: Dict[str, int] = {}
        for a in active:
            role_counts[a.role] = role_counts.get(a.role, 0) + 1
        return {
            "total_agents": len(self._agents),
            "active_agents": len(active),
            "retired_agents": len(self._agents) - len(active),
            "role_distribution": role_counts,
            "total_spawned": sum(
                1 for e in self._spawn_history if e["event"] == "spawn"
            ),
            "available_roles": self.available_roles(),
        }

    def all_agents(self) -> List[dict]:
        return [a.to_dict() for a in self._agents.values()]

    def spawn_history(self) -> List[dict]:
        return list(self._spawn_history)
