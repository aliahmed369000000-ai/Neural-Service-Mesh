
from __future__ import annotations
import uuid
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


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
    def __init__(self, name: str, description: str = "", tags: list = None):
        self.node_id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.tags = tags or []
        self.created_at = datetime.utcnow().isoformat()
        self._execution_count = 0
        self._last_executed: Optional[str] = None

    @property
    @abstractmethod
    def input_schema(self) -> NodeSchema: ...

    @property
    @abstractmethod
    def output_schema(self) -> NodeSchema: ...

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]: ...

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.input_schema.validate(data)
        result = self.process(data)
        self.output_schema.validate(result)
        self._execution_count += 1
        self._last_executed = datetime.utcnow().isoformat()
        logger.info(f"[{self.name}] executed #{self._execution_count}")
        return result

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
        }

    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self.name}' id='{self.node_id[:8]}'>"
