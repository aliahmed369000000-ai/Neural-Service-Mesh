
from __future__ import annotations
import uuid
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.node import BaseNode
from core.registry import NodeRegistry
from core.graph import ServiceGraph
from connectors.data_transformer import DataTransformer
from storage.file_storage import FileStorage

logger = logging.getLogger(__name__)
LOGS_FILE = "logs.json"


class ExecutionStep:
    def __init__(self, index: int, node_id: str, node_name: str):
        self.index = index
        self.node_id = node_id
        self.node_name = node_name
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.duration_ms: Optional[float] = None
        self.input_data: Optional[dict] = None
        self.output_data: Optional[dict] = None
        self.status: str = "pending"
        self.error: Optional[str] = None

    def to_dict(self):
        return {
            "index": self.index,
            "node_id": self.node_id,
            "node_name": self.node_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "status": self.status,
            "error": self.error,
        }


class ExecutionResult:
    def __init__(self, run_id: str, path: List[str]):
        self.run_id = run_id
        self.path = path
        self.steps: List[ExecutionStep] = []
        self.started_at = datetime.utcnow().isoformat()
        self.finished_at: Optional[str] = None
        self.total_duration_ms: Optional[float] = None
        self.final_output: Optional[dict] = None
        self.status: str = "running"

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "path": self.path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_duration_ms": self.total_duration_ms,
            "status": self.status,
            "final_output": self.final_output,
            "steps": [s.to_dict() for s in self.steps],
        }


class ExecutionEngine:
    def __init__(self, registry: NodeRegistry, graph: ServiceGraph,
                 storage: FileStorage, transformer: DataTransformer = None):
        self._registry = registry
        self._graph = graph
        self._storage = storage
        self._transformer = transformer or DataTransformer()
        self._history: List[ExecutionResult] = []
        logger.info("ExecutionEngine initialized")

    def run_path(self, path: List[str], initial_data: Dict[str, Any]) -> ExecutionResult:
        run_id = str(uuid.uuid4())
        result = ExecutionResult(run_id, path)
        t_start = time.time()
        logger.info(f"[Run {run_id[:8]}] path={[p[:8] for p in path]}")

        current = dict(initial_data)
        for idx, node_id in enumerate(path):
            step = ExecutionStep(idx, node_id, "unknown")
            result.steps.append(step)
            node: Optional[BaseNode] = self._registry.get(node_id)
            if not node:
                step.status = "error"
                step.error = f"Node '{node_id}' not found"
                result.status = "failed"
                result.finished_at = datetime.utcnow().isoformat()
                self._persist(result)
                return result

            step.node_name = node.name
            step.started_at = datetime.utcnow().isoformat()
            step.input_data = dict(current)
            step.status = "running"
            t0 = time.time()

            try:
                transformed = self._transformer.transform(current, node.input_schema)
                output = node.execute(transformed)
                step.output_data = dict(output)
                step.status = "success"
                step.duration_ms = round((time.time() - t0) * 1000, 2)
                step.finished_at = datetime.utcnow().isoformat()
                current = output
                logger.info(f"[Run {run_id[:8]}] step {idx} '{node.name}' OK ({step.duration_ms}ms)")
            except Exception as e:
                step.status = "error"
                step.error = str(e)
                step.duration_ms = round((time.time() - t0) * 1000, 2)
                step.finished_at = datetime.utcnow().isoformat()
                result.status = "failed"
                result.finished_at = datetime.utcnow().isoformat()
                result.total_duration_ms = round((time.time() - t_start) * 1000, 2)
                self._persist(result)
                logger.error(f"[Run {run_id[:8]}] step {idx} FAILED: {e}")
                return result

        result.final_output = current
        result.status = "success"
        result.finished_at = datetime.utcnow().isoformat()
        result.total_duration_ms = round((time.time() - t_start) * 1000, 2)
        self._persist(result)
        logger.info(f"[Run {run_id[:8]}] DONE in {result.total_duration_ms}ms")
        return result

    def run_between(self, start_id: str, end_id: str, data: Dict[str, Any]) -> ExecutionResult:
        path = self._graph.find_path_bfs(start_id, end_id)
        if not path:
            r = ExecutionResult(str(uuid.uuid4()), [])
            r.status = "failed"
            r.finished_at = datetime.utcnow().isoformat()
            logger.error(f"No path from {start_id[:8]} to {end_id[:8]}")
            self._persist(r)
            return r
        return self.run_path(path, data)

    def run_full_graph(self, data: Dict[str, Any]) -> ExecutionResult:
        order = self._graph.topological_sort()
        if not order:
            r = ExecutionResult(str(uuid.uuid4()), [])
            r.status = "failed"
            r.finished_at = datetime.utcnow().isoformat()
            self._persist(r)
            return r
        return self.run_path(order, data)

    def get_history(self, limit: int = 50) -> List[dict]:
        return [r.to_dict() for r in self._history[-limit:]]

    def get_run(self, run_id: str) -> Optional[dict]:
        for r in self._history:
            if r.run_id == run_id:
                return r.to_dict()
        data = self._storage.load(LOGS_FILE) or {}
        return next((r for r in data.get("runs", []) if r["run_id"] == run_id), None)

    def _persist(self, result: ExecutionResult):
        self._history.append(result)
        data = self._storage.load(LOGS_FILE) or {"runs": []}
        data["runs"].append(result.to_dict())
        self._storage.save(LOGS_FILE, data)

    def __repr__(self):
        return f"<ExecutionEngine runs={len(self._history)}>"
