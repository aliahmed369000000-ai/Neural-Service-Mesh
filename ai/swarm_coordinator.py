"""
Phase 6 – Swarm Coordinator
=============================
Orchestrates a swarm of autonomous agents to execute tasks in parallel.

Workflow:
  1. Receive a high-level task
  2. Decompose into sub-tasks
  3. Assign each sub-task to the best available agent
  4. Monitor execution
  5. Merge and validate results

Example:
  coordinator = SwarmCoordinator(factory, max_agents=20)
  result = coordinator.execute("translate and review document", data={...})
"""
from __future__ import annotations

import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable

from ai.agent_factory import AgentFactory, AgentInstance

logger = logging.getLogger(__name__)


class SwarmTask:
    """A single unit of work assigned to one agent."""

    def __init__(
        self,
        task_id: str,
        sub_goal: str,
        required_capability: str,
        data: dict,
        priority: int = 5,
    ):
        self.task_id = task_id
        self.sub_goal = sub_goal
        self.required_capability = required_capability
        self.data = data
        self.priority = priority          # 1 (highest) … 10 (lowest)
        self.assigned_agent_id: Optional[str] = None
        self.status = "pending"           # pending / running / done / failed
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.duration_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "sub_goal": self.sub_goal,
            "required_capability": self.required_capability,
            "priority": self.priority,
            "assigned_agent_id": self.assigned_agent_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
        }


class SwarmResult:
    """Aggregated result from a swarm execution run."""

    def __init__(self, swarm_id: str, goal: str):
        self.swarm_id = swarm_id
        self.goal = goal
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.finished_at: Optional[str] = None
        self.tasks: List[SwarmTask] = []
        self.merged_output: Optional[dict] = None
        self.status = "running"

    @property
    def success_count(self):
        return sum(1 for t in self.tasks if t.status == "done")

    @property
    def failed_count(self):
        return sum(1 for t in self.tasks if t.status == "failed")

    def complete(self, merged: dict):
        self.merged_output = merged
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.status = "done" if self.failed_count == 0 else "partial"

    def to_dict(self) -> dict:
        return {
            "swarm_id": self.swarm_id,
            "goal": self.goal,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_tasks": len(self.tasks),
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "merged_output": self.merged_output,
            "tasks": [t.to_dict() for t in self.tasks],
        }


class SwarmCoordinator:
    """
    Phase 6: Multi-Agent Swarm Coordinator.

    Decomposes goals into parallel tasks, assigns agents, and merges results.
    Supports up to `max_agents` concurrent workers.
    """

    # Default task decomposition rules: goal keyword → list of (sub_goal, capability)
    DECOMPOSITION_RULES: Dict[str, List[tuple]] = {
        "translate": [
            ("translate content", "translate"),
            ("validate translation", "validate"),
        ],
        "research": [
            ("search for information", "search"),
            ("summarize findings", "summarize"),
            ("fact-check results", "fact_check"),
        ],
        "review": [
            ("audit artifact", "audit"),
            ("score quality", "score"),
        ],
        "plan": [
            ("decompose goal into tasks", "plan"),
            ("schedule task execution", "schedule"),
        ],
        "optimize": [
            ("benchmark current performance", "benchmark"),
            ("identify improvements", "optimize"),
            ("apply tuning", "tune"),
        ],
        "monitor": [
            ("collect health metrics", "monitor"),
            ("generate alerts if needed", "alert"),
        ],
    }

    def __init__(
        self,
        factory: AgentFactory,
        max_agents: int = 20,
        knowledge_store=None,
    ):
        self._factory = factory
        self._max_agents = max_agents
        self._knowledge = knowledge_store
        self._history: List[SwarmResult] = []
        self._lock = threading.Lock()
        logger.info(f"SwarmCoordinator initialised (max_agents={max_agents})")

    # ── Main execution ────────────────────────────────────────────────────

    def execute(
        self,
        goal: str,
        data: dict,
        custom_tasks: Optional[List[dict]] = None,
    ) -> SwarmResult:
        """
        Execute a goal using the swarm.

        Args:
            goal:         High-level goal string.
            data:         Input data passed to each sub-task.
            custom_tasks: Optional manual task list
                          [{"sub_goal": ..., "capability": ..., "priority": ...}]
        """
        swarm_id = f"swarm_{str(uuid.uuid4())[:8]}"
        result = SwarmResult(swarm_id, goal)

        # 1. Decompose goal into tasks
        tasks = self._decompose(goal, data, custom_tasks)
        result.tasks = tasks

        if not tasks:
            result.merged_output = {"error": "Could not decompose goal into tasks"}
            result.status = "failed"
            result.finished_at = datetime.now(timezone.utc).isoformat()
            self._history.append(result)
            return result

        # 2. Sort by priority
        tasks.sort(key=lambda t: t.priority)

        # 3. Execute in parallel (bounded by max_agents)
        workers = min(len(tasks), self._max_agents)
        task_outputs: Dict[str, dict] = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_task: Dict[Future, SwarmTask] = {}
            for task in tasks:
                agent = self._factory.best_agent_for(task.required_capability)
                if agent is None:
                    # Auto-spawn an agent for this capability
                    agent = self._auto_spawn_for_capability(task.required_capability)
                if agent:
                    task.assigned_agent_id = agent.agent_id
                    fut = pool.submit(self._run_task, task, agent)
                    future_to_task[fut] = task
                else:
                    task.status = "failed"
                    task.error = f"No agent available for capability '{task.required_capability}'"

            for fut in as_completed(future_to_task):
                task = future_to_task[fut]
                try:
                    output = fut.result()
                    task.result = output
                    task.status = "done"
                    task_outputs[task.task_id] = output
                except Exception as exc:
                    task.status = "failed"
                    task.error = str(exc)
                    logger.error(f"Task {task.task_id} failed: {exc}")

        # 4. Merge results
        merged = self._merge_results(goal, tasks, task_outputs)
        result.complete(merged)

        with self._lock:
            self._history.append(result)

        logger.info(
            f"Swarm {swarm_id} finished: "
            f"{result.success_count}/{len(tasks)} tasks succeeded"
        )
        return result

    # ── Internals ─────────────────────────────────────────────────────────

    def _decompose(
        self,
        goal: str,
        data: dict,
        custom_tasks: Optional[List[dict]],
    ) -> List[SwarmTask]:
        if custom_tasks:
            return [
                SwarmTask(
                    task_id=f"task_{i}_{str(uuid.uuid4())[:6]}",
                    sub_goal=t["sub_goal"],
                    required_capability=t.get("capability", "search"),
                    data=data,
                    priority=t.get("priority", 5),
                )
                for i, t in enumerate(custom_tasks)
            ]

        goal_lower = goal.lower()
        for keyword, sub_specs in self.DECOMPOSITION_RULES.items():
            if keyword in goal_lower:
                return [
                    SwarmTask(
                        task_id=f"task_{i}_{str(uuid.uuid4())[:6]}",
                        sub_goal=sub_goal,
                        required_capability=cap,
                        data=data,
                        priority=i + 1,
                    )
                    for i, (sub_goal, cap) in enumerate(sub_specs)
                ]

        # Fallback: single generic task
        return [
            SwarmTask(
                task_id=f"task_0_{str(uuid.uuid4())[:6]}",
                sub_goal=goal,
                required_capability="search",
                data=data,
                priority=5,
            )
        ]

    def _run_task(self, task: SwarmTask, agent: AgentInstance) -> dict:
        import time
        task.status = "running"
        task.started_at = datetime.now(timezone.utc).isoformat()
        t0 = time.time()

        try:
            # Simulate agent execution (in production, call actual agent logic)
            output = {
                "task_id": task.task_id,
                "sub_goal": task.sub_goal,
                "agent_id": agent.agent_id,
                "agent_role": agent.role,
                "capability_used": task.required_capability,
                "status": "completed",
                "data_keys_processed": list(task.data.keys()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            agent.record_task(success=True)
            return output
        except Exception as exc:
            agent.record_task(success=False)
            raise exc
        finally:
            task.finished_at = datetime.now(timezone.utc).isoformat()
            task.duration_ms = round((time.time() - t0) * 1000, 2)

    def _auto_spawn_for_capability(self, capability: str) -> Optional[AgentInstance]:
        """Find a role that has the required capability and spawn an agent."""
        from ai.agent_factory import AGENT_CATALOGUE
        for role, spec in AGENT_CATALOGUE.items():
            if capability in spec.get("capabilities", []):
                try:
                    agent = self._factory.spawn(role)
                    logger.info(f"Auto-spawned {role} for capability '{capability}'")
                    return agent
                except Exception:
                    pass
        return None

    def _merge_results(
        self,
        goal: str,
        tasks: List[SwarmTask],
        outputs: Dict[str, dict],
    ) -> dict:
        return {
            "goal": goal,
            "tasks_completed": len(outputs),
            "tasks_failed": sum(1 for t in tasks if t.status == "failed"),
            "outputs": list(outputs.values()),
            "merged_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Public helpers ────────────────────────────────────────────────────

    def history(self, limit: int = 20) -> List[dict]:
        return [r.to_dict() for r in self._history[-limit:]]

    def summary(self) -> dict:
        total = len(self._history)
        done = sum(1 for r in self._history if r.status == "done")
        partial = sum(1 for r in self._history if r.status == "partial")
        failed = sum(1 for r in self._history if r.status == "failed")
        return {
            "total_swarms": total,
            "done": done,
            "partial": partial,
            "failed": failed,
            "max_agents": self._max_agents,
            "active_agents": len(self._factory.list_active()),
        }
