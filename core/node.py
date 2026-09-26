
from __future__ import annotations
import uuid
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class NodeState:
    """حالات دورة حياة العقدة (BaseNode)."""
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    FAILED = "failed"

    ALL = (CREATED, ACTIVE, PAUSED, FAILED)


@dataclass
class NodeSchema:
    fields: Dict[str, str]
    required: list = field(default_factory=list)
    description: str = ""

    def validate(self, data: Dict[str, Any]) -> bool:
        for f in self.required:
            if f not in data:
                raise ValueError(f"Missing required field: '{f}'")
        return True


@dataclass
class NodeMetadata:
    node_id: str
    name: str
    node_type: str
    created_at: str
    version: str = "1.0.0"
    tags: list = field(default_factory=list)
    description: str = ""

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type,
            "created_at": self.created_at,
            "version": self.version,
            "tags": self.tags,
            "description": self.description,
        }


class BaseNode(ABC):
    def __init__(self, name: str, description: str = "", tags: list = None,
                 node_id: Optional[str] = None):
        # node_id قابل للتمرير عمداً (بدل uuid4 عشوائي دائماً) حتى تقدر
        # الطبقات الأعلى (مثل MeshBundle._register_roles) تعيد بناء نفس
        # العقدة بنفس الهوية بعد إعادة تشغيل العملية، ثم تستدعي
        # restore_state() لاسترجاع تاريخها بدل ما تبدأ من الصفر بمعرّف جديد.
        self.node_id = node_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.tags = tags or []
        self.created_at = datetime.utcnow().isoformat()
        self._execution_count = 0
        self._last_executed: Optional[str] = None
        self.state: str = NodeState.CREATED
        self._last_error: Optional[str] = None

    @property
    @abstractmethod
    def input_schema(self) -> NodeSchema: ...

    @property
    @abstractmethod
    def output_schema(self) -> NodeSchema: ...

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]: ...

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.state == NodeState.PAUSED:
            raise RuntimeError(
                f"Node '{self.name}' [{self.node_id[:8]}] is paused and cannot execute"
            )
        try:
            self.input_schema.validate(data)
            result = self.process(data)
            self.output_schema.validate(result)
        except Exception as e:
            self.mark_failed(str(e))
            raise
        self._execution_count += 1
        self._last_executed = datetime.utcnow().isoformat()
        self.state = NodeState.ACTIVE
        self._last_error = None
        logger.info(f"[{self.name}] executed #{self._execution_count}")
        return result

    def pause(self) -> None:
        """إيقاف العقدة مؤقتاً؛ يمنع execute() حتى يتم استئنافها."""
        self.state = NodeState.PAUSED
        logger.info(f"[{self.name}] paused")

    def resume(self) -> None:
        """استئناف عقدة متوقفة مؤقتاً أو فاشلة وإعادتها لحالة نشطة."""
        self.state = NodeState.ACTIVE
        self._last_error = None
        logger.info(f"[{self.name}] resumed")

    def mark_failed(self, error: str) -> None:
        """تسجيل فشل العقدة يدوياً أو تلقائياً عند استثناء أثناء execute()."""
        self.state = NodeState.FAILED
        self._last_error = error
        logger.warning(f"[{self.name}] failed: {error}")

    def record_execution(self, success: bool, error: Optional[str] = None) -> None:
        """تسجيل نتيجة تنفيذ حقيقي جرى خارج process()/execute() — مثل عقدة
        دور وكيل (AgentRoleNode) ينفّذ عملها الفعلي عبر AgentFactory.run_task
        وليس عبر process() نفسها. execute() لا يناسب هذه الحالة (سيعيد
        استدعاء process() فارغة)، فهذه الدالة تحدّث دورة حياة العقدة
        (execution_count/state/last_error) مباشرة من نتيجة معروفة سلفاً."""
        if success:
            self._execution_count += 1
            self._last_executed = datetime.utcnow().isoformat()
            self.state = NodeState.ACTIVE
            self._last_error = None
            logger.info(f"[{self.name}] execution recorded #{self._execution_count} (external)")
        else:
            self.mark_failed(error or "فشل تنفيذ خارجي بدون تفاصيل")

    def restore_state(self, snapshot: Dict[str, Any]) -> None:
        """استرجاع تاريخ العقدة (state/execution_count/last_executed/last_error)
        من snapshot محفوظ سابقاً (عادة من NodeRegistry.get_meta_by_name بعد
        إعادة تشغيل العملية). يُفترض أن snapshot جاء من to_dict() لنفس
        node_id — إن اختلف يُسجَّل تحذير لكن الاسترجاع يكمل، لأن الأولوية
        عدم إسقاط أي استثناء أثناء إقلاع الشبكة."""
        snap_id = snapshot.get("node_id")
        if snap_id and snap_id != self.node_id:
            logger.warning(
                f"[{self.name}] restore_state: node_id mismatch "
                f"(self={self.node_id[:8]}, snapshot={snap_id[:8]})"
            )
        self._execution_count = snapshot.get("execution_count", 0) or 0
        self._last_executed = snapshot.get("last_executed")
        self.state = snapshot.get("state") or NodeState.CREATED
        self._last_error = snapshot.get("last_error")
        logger.info(
            f"[{self.name}] state restored: {self.state} "
            f"(execution_count={self._execution_count})"
        )

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            self.node_id, self.name,
            self.__class__.__name__,
            self.created_at,
            tags=self.tags,
            description=self.description,
        )

    def to_dict(self):
        return {
            **self.metadata.to_dict(),
            "input_schema": {
                "fields": self.input_schema.fields,
                "required": self.input_schema.required,
            },
            "output_schema": {
                "fields": self.output_schema.fields,
                "required": self.output_schema.required,
            },
            "execution_count": self._execution_count,
            "last_executed": self._last_executed,
            "state": self.state,
            "last_error": self._last_error,
        }

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} name='{self.name}' "
            f"id='{self.node_id[:8]}' state='{self.state}'>"
        )
