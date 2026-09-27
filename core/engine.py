from __future__ import annotations
import uuid
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.node import BaseNode, NodeState
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
        self.is_fallback: bool = False

    def to_dict(self):
        return {
            "index": self.index, "node_id": self.node_id, "node_name": self.node_name,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "duration_ms": self.duration_ms, "input_data": self.input_data,
            "output_data": self.output_data, "status": self.status,
            "error": self.error, "is_fallback": self.is_fallback,
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
        self.ai_suggested: bool = False

    def to_dict(self):
        return {
            "run_id": self.run_id, "path": self.path,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "total_duration_ms": self.total_duration_ms, "status": self.status,
            "final_output": self.final_output, "ai_suggested": self.ai_suggested,
            "steps": [s.to_dict() for s in self.steps],
        }


class ExecutionEngine:
    def __init__(self, registry, graph, storage, transformer=None, db=None, ai=None):
        self._registry = registry
        self._graph = graph
        self._storage = storage
        self._transformer = transformer or DataTransformer()
        self._db = db
        self._ai = ai
        self._history: List[ExecutionResult] = []
        logger.info("ExecutionEngine initialized (Phase 2)")

    def _finalize(self, result: ExecutionResult, status: str, t_start: float) -> ExecutionResult:
        """يُنهي أي مسار خروج (نجاح أو فشل) بنفس الخطوات دائماً: يضبط
        الحالة/الزمن، يحفظ، ثم يُغذّي self._ai.learn_from_run بالنتيجة —
        بما فيها نتائج الفشل. قبل هذا التعديل كانت 3 من أصل 4 مخارج فشل
        في run_path (عقدة غير موجودة بلا بديل، عقدة محجورة بلا بديل، فشل
        تنفيذ حقيقي بلا بديل) تُرجِع النتيجة مباشرة دون استدعاء
        learn_from_run إطلاقاً — يستدعيها فقط مسار النجاح النهائي. الأثر:
        AIDecisionLayer._path_stats لم يكن يُحتسِب أي فشل قط، فـ
        success_rate في get_insights() كانت تظهر 100% دائماً لأي مسار جرّب
        الفشل ولو مرات كثيرة، لأن العدّاد runs نفسه لا يزيد إلا عند
        النجاح — يُفسِد كامل الغرض من health='critical' عند تدهور الأداء.
        اثنان من مخارج الفشل الثلاثة أيضاً كانا لا يضبطان
        result.total_duration_ms إطلاقاً (يبقى None)."""
        result.status = status
        result.finished_at = datetime.utcnow().isoformat()
        result.total_duration_ms = round((time.time() - t_start) * 1000, 2)
        self._persist(result)
        if self._ai:
            self._ai.learn_from_run(result.to_dict())
        return result

    def run_path(self, path: List[str], initial_data: Dict[str, Any], use_fallback: bool = True) -> ExecutionResult:
        run_id = str(uuid.uuid4())
        result = ExecutionResult(run_id, path)
        t_start = time.time()
        current = dict(initial_data)

        for idx, node_id in enumerate(path):
            step = ExecutionStep(idx, node_id, "unknown")
            result.steps.append(step)
            node = self._registry.get(node_id)

            if not node:
                step.status = "error"
                step.error = f"Node '{node_id}' not found"
                if use_fallback and self._ai:
                    fb_id = self._ai.should_fallback(node_id, step.error)
                    if fb_id:
                        node = self._registry.get(fb_id)
                        if node:
                            step.is_fallback = True
                            step.node_id = fb_id
                            step.node_name = node.name
                if not node:
                    return self._finalize(result, "failed", t_start)

            # عقدة موجودة لكنها محجورة (NodeState.PAUSED — عادة عبر
            # MeshBundle._apply_reputation_feedback بسبب سمعة منخفضة).
            # node.execute() كانت سترفع RuntimeError مباشرة، فيلتقطها
            # except العام أدناه كخطأ تنفيذ عادي ويُسقِط المسار بالكامل
            # فوراً — رغم أن نفس هذا المحرك يملك بالفعل مسار fallback
            # حقيقياً (self._ai.should_fallback) لعقدة غير موجودة تماماً
            # في السطور أعلاه؛ عقدة محجورة مؤقتاً أولى بمحاولة بديل من
            # الفشل الصامت الكامل. لا نعيد تشغيل should_fallback على
            # البديل نفسه (مستوى واحد فقط) تفادياً لأي تسلسل غير منتهٍ.
            if node.state == NodeState.PAUSED:
                step.status = "error"
                step.error = (
                    f"Node '{node.name}' [{node.node_id[:8]}] is paused "
                    f"(quarantined) and cannot execute"
                )
                fb_node = None
                if use_fallback and self._ai:
                    fb_id = self._ai.should_fallback(node.node_id, step.error)
                    if fb_id:
                        candidate = self._registry.get(fb_id)
                        if candidate and candidate.state != NodeState.PAUSED:
                            fb_node = candidate
                if fb_node:
                    step.is_fallback = True
                    step.node_id = fb_node.node_id
                    node = fb_node
                else:
                    return self._finalize(result, "failed", t_start)

            # ── تنفيذ فعلي — بمحاولة واحدة إضافية عبر بديل عند فشل حقيقي ──
            # كان فشل حقيقي أثناء process() (استثناء من منطق العقدة نفسها،
            # أو رفض NodeSchema.validate() لبيانات غير مطابقة) يُسقِط
            # المسار بالكامل مباشرة، دون أي محاولة بديل — رغم أن should_fallback
            # أعلاه مصمَّمة أصلاً لهذه الحالة بالضبط (اسمها الوسيطة الأولى
            # "failed_node_id"، ومعناها العام "عقدة فشلت"، وليس فقط "غير
            # موجودة" أو "محجورة"). مستوى واحد فقط من الاستبدال (كما في
            # الحالتين أعلاه): لا نحاول بديلاً لبديل فشل هو الآخر.
            already_fell_back = step.is_fallback
            attempt_node = node
            for attempt in range(2):
                step.node_name = attempt_node.name
                step.node_id = attempt_node.node_id
                step.started_at = datetime.utcnow().isoformat()
                step.input_data = dict(current)
                step.status = "running"
                t0 = time.time()
                try:
                    transformed = self._transformer.transform(current, attempt_node.input_schema)
                    output = attempt_node.execute(transformed)
                    step.output_data = dict(output)
                    step.status = "success"
                    step.duration_ms = round((time.time() - t0) * 1000, 2)
                    step.finished_at = datetime.utcnow().isoformat()
                    current = output
                    self._registry.refresh_meta(attempt_node.node_id)
                    break
                except Exception as e:
                    step.status = "error"
                    step.error = str(e)
                    step.duration_ms = round((time.time() - t0) * 1000, 2)
                    step.finished_at = datetime.utcnow().isoformat()
                    self._registry.refresh_meta(attempt_node.node_id)

                    fb_node = None
                    if attempt == 0 and not already_fell_back and use_fallback and self._ai:
                        fb_id = self._ai.should_fallback(attempt_node.node_id, step.error)
                        if fb_id:
                            candidate = self._registry.get(fb_id)
                            if candidate and candidate.state != NodeState.PAUSED:
                                fb_node = candidate
                    if fb_node:
                        step.is_fallback = True
                        attempt_node = fb_node
                        continue  # محاولة ثانية وأخيرة عبر البديل

                    return self._finalize(result, "failed", t_start)

        result.final_output = current
        return self._finalize(result, "success", t_start)

    def run_between(self, start_id: str, end_id: str, data: Dict[str, Any], use_ai: bool = True) -> ExecutionResult:
        path = None
        if use_ai and self._ai:
            path = self._ai.choose_path(start_id, end_id)
        if not path:
            path = self._graph.find_path_bfs(start_id, end_id)
        if not path:
            r = ExecutionResult(str(uuid.uuid4()), [])
            r.status = "failed"
            r.finished_at = datetime.utcnow().isoformat()
            self._persist(r)
            return r
        result = self.run_path(path, data)
        if use_ai and self._ai:
            result.ai_suggested = True
        return result

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
        if self._db:
            return self._db.list_runs(limit)
        return [r.to_dict() for r in self._history[-limit:]]

    def get_run(self, run_id: str) -> Optional[dict]:
        for r in self._history:
            if r.run_id == run_id:
                return r.to_dict()
        if self._db:
            return self._db.get_run(run_id)
        data = self._storage.load(LOGS_FILE) or {}
        return next((r for r in data.get("runs", []) if r["run_id"] == run_id), None)

    def _persist(self, result: ExecutionResult):
        self._history.append(result)
        if self._db:
            self._db.save_run(result.to_dict())
        data = self._storage.load(LOGS_FILE) or {"runs": []}
        data["runs"].append(result.to_dict())
        self._storage.save(LOGS_FILE, data)

    def __repr__(self):
        return f"<ExecutionEngine runs={len(self._history)}>"
