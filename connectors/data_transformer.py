from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional

from connectors.base_connector import BaseConnector
from core.node import NodeSchema

logger = logging.getLogger(__name__)


class DataTransformer(BaseConnector):
    def __init__(self):
        super().__init__("DataTransformer")

    def connect(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.normalize(data)

    def normalize(self, data: Any) -> Dict[str, Any]:
        if isinstance(data, dict):
            return {k: v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
                    for k, v in data.items()}
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            return {"text": data}
        if isinstance(data, list):
            return {"items": data}
        if data is None:
            return {}
        return {"value": data}

    def transform(self, data: Dict[str, Any], target_schema: Optional[NodeSchema] = None) -> Dict[str, Any]:
        normalized = self.normalize(data)
        if not target_schema:
            return normalized
        result = {}
        for fname in target_schema.fields:
            if fname in normalized:
                result[fname] = normalized[fname]
            elif fname not in target_schema.required:
                result[fname] = None
        for k, v in normalized.items():
            if k not in result:
                result[k] = v
        return result

    def map_fields(self, data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        return {mapping.get(k, k): v for k, v in data.items()}

    def filter_fields(self, data: Dict[str, Any], allowed: List[str]) -> Dict[str, Any]:
        return {k: v for k, v in data.items() if k in allowed}
