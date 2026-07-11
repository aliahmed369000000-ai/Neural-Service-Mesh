from __future__ import annotations
from typing import Any, Dict
from core.node import BaseNode, NodeSchema


class PassThroughNode(BaseNode):
    """
    Phase 2: A generic node that can be created dynamically via API.
    Passes data through unchanged — useful for testing topology or as a placeholder.
    """

    def __init__(self, name: str, description: str = "", tags: list = None):
        super().__init__(name=name, description=description or "API-created pass-through node",
                         tags=tags or ["dynamic", "passthrough"])

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={"data": "Any"}, required=[])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={"data": "Any"}, required=[])

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return dict(data)
