"""
Phase 7 – Code Generation Engine
===================================
When a gap is detected (e.g. "Need: CSV → JSON converter"),
this engine generates complete, runnable Python module code.

Workflow:
  1. Receives a gap description + context
  2. Generates a full Python module as a string
  3. Writes it to services/<module_name>.py (proposed)
  4. Returns the spec for Sandbox Testing

File: ai/code_generator.py
"""
from __future__ import annotations

import ast
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Built-in code templates ────────────────────────────────────────────────

_TEMPLATES: Dict[str, str] = {
    "data_transformer": '''"""
Auto-generated Phase 7 data transformer: {name}
Generated: {timestamp}
Gap context: {gap_context}
"""
from __future__ import annotations
from typing import Any, Dict
from core.node import BaseNode, NodeSchema


class {class_name}(BaseNode):
    """Phase 7 auto-generated transformer: {description}"""

    def __init__(self):
        super().__init__(
            name="{name}",
            description="{description}",
            tags={tags},
        )

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={input_fields}, required={required_inputs})

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={output_fields}, required=[])

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data: {description}"""
        result = dict(data)
        {transform_body}
        return result
''',
    "validator": '''"""
Auto-generated Phase 7 validator: {name}
Generated: {timestamp}
"""
from __future__ import annotations
from typing import Any, Dict
from core.node import BaseNode, NodeSchema


class {class_name}(BaseNode):
    """Phase 7 auto-generated validator: {description}"""

    def __init__(self):
        super().__init__(
            name="{name}",
            description="{description}",
            tags={tags},
        )

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={input_fields}, required={required_inputs})

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={output_fields}, required=[])

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(data)
        errors = []
        {validation_body}
        result["validation_passed"] = len(errors) == 0
        result["validation_errors"] = errors
        return result
''',
}


class GeneratedModule:
    """A generated Python module with metadata."""

    def __init__(self, name: str, code: str, module_type: str, gap_context: Optional[dict] = None):
        self.module_id = str(uuid.uuid4())[:12]
        self.name = name
        self.class_name = self._to_class_name(name)
        self.code = code
        self.module_type = module_type
        self.gap_context = gap_context or {}
        self.generated_at = datetime.now(timezone.utc).isoformat()
        self.status = "proposed"     # proposed / tested / approved / rejected / deployed
        self.test_result: Optional[dict] = None
        self.file_path: Optional[str] = None
        self.syntax_valid: bool = False
        self._validate_syntax()

    def _to_class_name(self, name: str) -> str:
        parts = re.sub(r"[^a-zA-Z0-9 ]", " ", name).split()
        return "".join(p.capitalize() for p in parts)

    def _validate_syntax(self):
        try:
            ast.parse(self.code)
            self.syntax_valid = True
        except SyntaxError as exc:
            self.syntax_valid = False
            logger.warning(f"[CodeGen] syntax error in {self.name}: {exc}")

    def to_dict(self) -> dict:
        return {
            "module_id": self.module_id,
            "name": self.name,
            "class_name": self.class_name,
            "module_type": self.module_type,
            "gap_context": self.gap_context,
            "generated_at": self.generated_at,
            "status": self.status,
            "syntax_valid": self.syntax_valid,
            "test_result": self.test_result,
            "file_path": self.file_path,
            "code_lines": len(self.code.splitlines()),
        }


class CodeGenerationEngine:
    """
    Phase 7: Autonomous Code Generation Engine.

    Generates Python service node code from gap descriptions.
    Supports template-based generation with gap-aware body synthesis.
    """

    def __init__(
        self,
        output_dir: str = "./services",
        knowledge_store=None,
        governance=None,
    ):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._knowledge = knowledge_store
        self._governance = governance
        self._generated: Dict[str, GeneratedModule] = {}
        self._generate_count = 0

    # ── Main generation API ────────────────────────────────────────────────

    def generate_from_gap(self, gap: dict) -> GeneratedModule:
        """
        Generate a Python module to fill a detected gap.

        gap: {
            "missing_service": str,
            "gap_type": str,
            "source_node": dict,
            "target_node": dict,
            "confidence": float,
        }
        """
        name = gap.get("missing_service", "GeneratedService")
        gap_type = gap.get("gap_type", "semantic")
        source = gap.get("source_node", {})
        target = gap.get("target_node", {})

        # Determine module type
        module_type = self._infer_module_type(name, gap_type, source, target)

        # Build code from template
        code = self._build_code(name, module_type, source, target, gap)
        module = GeneratedModule(name=name, code=code, module_type=module_type, gap_context=gap)

        self._generated[module.module_id] = module
        self._generate_count += 1
        logger.info(f"[CodeGen] generated '{name}' ({module_type}) — syntax_valid={module.syntax_valid}")
        return module

    def generate_custom(
        self,
        name: str,
        description: str,
        input_fields: Optional[Dict[str, str]] = None,
        output_fields: Optional[Dict[str, str]] = None,
        required_inputs: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        transform_logic: Optional[str] = None,
    ) -> GeneratedModule:
        """Generate a custom module with explicit spec."""
        gap_context = {
            "missing_service": name,
            "description": description,
            "custom": True,
        }
        code = self._render_template(
            module_type="data_transformer",
            name=name,
            description=description,
            input_fields=input_fields or {"data": "Any"},
            output_fields=output_fields or {"data": "Any"},
            required_inputs=required_inputs or [],
            tags=tags or ["generated", "custom"],
            transform_body=transform_logic or "# TODO: implement transform logic",
            gap_context=str(gap_context),
        )
        module = GeneratedModule(name=name, code=code, module_type="data_transformer",
                                 gap_context=gap_context)
        self._generated[module.module_id] = module
        self._generate_count += 1
        return module

    # ── File writing ───────────────────────────────────────────────────────

    def write_to_file(self, module: GeneratedModule, subdir: str = "") -> str:
        """Write generated code to the services directory."""
        filename = re.sub(r"[^a-z0-9_]", "_", module.name.lower()) + ".py"
        if subdir:
            target_dir = self._output_dir / subdir
            target_dir.mkdir(parents=True, exist_ok=True)
            path = target_dir / filename
        else:
            path = self._output_dir / filename

        try:
            with open(path, "w") as f:
                f.write(module.code)
            module.file_path = str(path)
            logger.info(f"[CodeGen] wrote {path}")
            return str(path)
        except Exception as exc:
            logger.error(f"[CodeGen] write error: {exc}")
            return ""

    # ── Template helpers ───────────────────────────────────────────────────

    def _infer_module_type(self, name: str, gap_type: str, source: dict, target: dict) -> str:
        name_lower = name.lower()
        if any(w in name_lower for w in ("validator", "checker", "verifier")):
            return "validator"
        return "data_transformer"

    def _build_code(self, name: str, module_type: str, source: dict, target: dict,
                    gap: dict) -> str:
        src_name = source.get("name", "Source")
        tgt_name = target.get("name", "Target")
        description = f"Bridges gap between {src_name} and {tgt_name}"
        tags = ["generated", "phase7", gap.get("gap_type", "semantic")]

        # Infer field types from node names/tags
        input_fields = self._infer_input_fields(source)
        output_fields = self._infer_output_fields(target)
        required = list(input_fields.keys())[:2]

        transform_body = self._generate_transform_body(name, source, target)

        return self._render_template(
            module_type=module_type,
            name=name,
            description=description,
            input_fields=input_fields,
            output_fields=output_fields,
            required_inputs=required,
            tags=tags,
            transform_body=transform_body,
            gap_context=str({k: v for k, v in gap.items() if k != "source_node" and k != "target_node"}),
        )

    def _render_template(self, module_type: str, name: str, description: str,
                         input_fields: dict, output_fields: dict, required_inputs: list,
                         tags: list, transform_body: str, gap_context: str) -> str:
        class_name = "".join(p.capitalize() for p in re.sub(r"[^a-zA-Z0-9 ]", " ", name).split())
        tpl = _TEMPLATES.get(module_type, _TEMPLATES["data_transformer"])
        return tpl.format(
            name=name,
            class_name=class_name,
            description=description,
            input_fields=repr(input_fields),
            output_fields=repr(output_fields),
            required_inputs=repr(required_inputs),
            tags=repr(tags),
            transform_body=transform_body,
            validation_body="# Auto-generated validation",
            gap_context=gap_context,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _infer_input_fields(self, source: dict) -> Dict[str, str]:
        tags = source.get("tags", [])
        name = source.get("name", "").lower()
        if "text" in name or "text" in tags:
            return {"text": "str", "source": "str"}
        if "csv" in name or "csv" in tags:
            return {"csv_data": "str", "delimiter": "str"}
        if "json" in name or "json" in tags:
            return {"json_data": "str"}
        if "image" in name:
            return {"image_path": "str", "format": "str"}
        return {"data": "Any", "metadata": "dict"}

    def _infer_output_fields(self, target: dict) -> Dict[str, str]:
        tags = target.get("tags", [])
        name = target.get("name", "").lower()
        if "json" in name or "json" in tags:
            return {"json_data": "str", "record_count": "int"}
        if "text" in name or "text" in tags:
            return {"text": "str", "processed": "bool"}
        if "summary" in name or "summary" in tags:
            return {"summary": "str", "word_count": "int"}
        return {"result": "Any", "processed": "bool"}

    def _generate_transform_body(self, name: str, source: dict, target: dict) -> str:
        name_lower = name.lower()
        if "csv" in name_lower and "json" in name_lower:
            return (
                "import csv, io\n"
                "        csv_data = data.get('csv_data', '')\n"
                "        reader = csv.DictReader(io.StringIO(csv_data))\n"
                "        records = list(reader)\n"
                "        import json\n"
                "        result['json_data'] = json.dumps(records)\n"
                "        result['record_count'] = len(records)"
            )
        if "json" in name_lower and "csv" in name_lower:
            return (
                "import json, csv, io\n"
                "        records = json.loads(data.get('json_data', '[]'))\n"
                "        if records:\n"
                "            buf = io.StringIO()\n"
                "            writer = csv.DictWriter(buf, fieldnames=records[0].keys())\n"
                "            writer.writeheader()\n"
                "            writer.writerows(records)\n"
                "            result['csv_data'] = buf.getvalue()"
            )
        if "filter" in name_lower or "clean" in name_lower:
            return (
                "text = str(data.get('text') or data.get('data', ''))\n"
                "        result['text'] = text.strip()\n"
                "        result['processed'] = True"
            )
        if "enrich" in name_lower or "augment" in name_lower:
            return (
                "result['enriched'] = True\n"
                "        result['enriched_at'] = __import__('datetime').datetime.utcnow().isoformat()"
            )
        # Generic pass-through with marker
        return (
            "result['_transformed_by'] = self.name\n"
            "        result['_transform_ts'] = __import__('datetime').datetime.utcnow().isoformat()"
        )

    # ── Listing / summary ──────────────────────────────────────────────────

    def list_generated(self, status_filter: Optional[str] = None) -> List[dict]:
        modules = list(self._generated.values())
        if status_filter:
            modules = [m for m in modules if m.status == status_filter]
        return [m.to_dict() for m in modules]

    def get_module(self, module_id: str) -> Optional[GeneratedModule]:
        return self._generated.get(module_id)

    def summary(self) -> dict:
        statuses: Dict[str, int] = {}
        for m in self._generated.values():
            statuses[m.status] = statuses.get(m.status, 0) + 1
        return {
            "total_generated": self._generate_count,
            "modules_in_store": len(self._generated),
            "by_status": statuses,
        }
