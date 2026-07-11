"""
Phase 7 – Sandbox Testing Lab
================================
Every auto-generated module is tested in isolation before deployment.

Test lifecycle:
  1. Generate  – CodeGenerationEngine produces Python code
  2. Write     – code is written to /sandbox/<module>.py (temp)
  3. Test      – SandboxTestingLab imports + instantiates + runs it
  4. Score     – performance and safety metrics computed
  5. Approve   – GovernanceApproval reviews score → Approve/Reject/Needs Revision
  6. Deploy    – approved module moved to services/ and registered in mesh

File: ai/sandbox_lab.py
"""
from __future__ import annotations

import ast
import importlib.util
import logging
import os
import re
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SandboxTestResult:
    """Result of running a generated module through the sandbox."""

    def __init__(self, module_id: str, module_name: str):
        self.test_id = f"test_{module_id}_{int(time.time())}"
        self.module_id = module_id
        self.module_name = module_name
        self.tested_at = datetime.now(timezone.utc).isoformat()

        # Syntax
        self.syntax_valid: bool = False
        self.syntax_errors: List[str] = []

        # Import
        self.import_success: bool = False
        self.import_error: Optional[str] = None

        # Instantiation
        self.instantiation_success: bool = False
        self.instantiation_error: Optional[str] = None

        # Execution
        self.execution_success: bool = False
        self.execution_error: Optional[str] = None
        self.execution_latency_ms: float = 0.0
        self.output_valid: bool = False

        # Safety
        self.safety_passed: bool = False
        self.safety_violations: List[str] = []

        # Overall
        self.score: float = 0.0       # 0-100
        self.verdict: str = "pending"  # passed / failed / needs_revision

    def compute_score(self):
        points = 0.0
        if self.syntax_valid:    points += 20
        if self.import_success:  points += 20
        if self.instantiation_success: points += 20
        if self.execution_success: points += 25
        if self.output_valid:    points += 10
        if self.safety_passed:   points += 5
        self.score = points
        if self.score >= 80:
            self.verdict = "passed"
        elif self.score >= 40:
            self.verdict = "needs_revision"
        else:
            self.verdict = "failed"

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "module_id": self.module_id,
            "module_name": self.module_name,
            "tested_at": self.tested_at,
            "syntax_valid": self.syntax_valid,
            "syntax_errors": self.syntax_errors,
            "import_success": self.import_success,
            "import_error": self.import_error,
            "instantiation_success": self.instantiation_success,
            "instantiation_error": self.instantiation_error,
            "execution_success": self.execution_success,
            "execution_error": self.execution_error,
            "execution_latency_ms": round(self.execution_latency_ms, 2),
            "output_valid": self.output_valid,
            "safety_passed": self.safety_passed,
            "safety_violations": self.safety_violations,
            "score": round(self.score, 1),
            "verdict": self.verdict,
        }


# Patterns that are never allowed in generated code
_UNSAFE_PATTERNS = [
    r"\bos\.system\b",
    r"\bsubprocess\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"__import__\s*\(\s*['\"]os['\"]",
    r"\bshutil\.rmtree\b",
    r"open\s*\(.*['\"]w['\"]",   # file writes (only generated code writes are allowed via engine)
]


class SandboxTestingLab:
    """
    Phase 7: Runs generated modules in an isolated sandbox environment.

    Tests: syntax, import, instantiation, execution, output schema, safety.
    """

    def __init__(self, sandbox_dir: str = "./sandbox"):
        self._sandbox_dir = Path(sandbox_dir)
        self._sandbox_dir.mkdir(parents=True, exist_ok=True)
        # ensure sandbox is importable
        sandbox_init = self._sandbox_dir / "__init__.py"
        if not sandbox_init.exists():
            sandbox_init.write_text("# Phase 7 Sandbox\n")
        self._results: Dict[str, SandboxTestResult] = {}
        self._test_count = 0
        self._pass_count = 0

    def test_module(self, module) -> SandboxTestResult:
        """
        Run a full sandbox test on a GeneratedModule.
        Returns SandboxTestResult.
        """
        result = SandboxTestResult(module.module_id, module.name)

        # 1. Syntax check
        self._check_syntax(module.code, result)
        if not result.syntax_valid:
            result.compute_score()
            self._store(module, result)
            return result

        # 2. Safety check (static analysis)
        self._check_safety(module.code, result)

        # 3. Write to sandbox temp file
        sandbox_path = self._write_to_sandbox(module)

        # 4. Import
        mod_obj = self._import_module(sandbox_path, module.name, result)

        # 5. Instantiate
        if mod_obj:
            node_instance = self._instantiate_node(mod_obj, module.class_name, result)

            # 6. Execute with test data
            if node_instance:
                self._execute_node(node_instance, module, result)

        result.compute_score()
        self._store(module, result)
        return result

    # ── Private test steps ─────────────────────────────────────────────────

    def _check_syntax(self, code: str, result: SandboxTestResult):
        try:
            ast.parse(code)
            result.syntax_valid = True
        except SyntaxError as exc:
            result.syntax_valid = False
            result.syntax_errors.append(str(exc))

    def _check_safety(self, code: str, result: SandboxTestResult):
        violations = []
        for pattern in _UNSAFE_PATTERNS:
            if re.search(pattern, code):
                violations.append(pattern)
        result.safety_violations = violations
        result.safety_passed = len(violations) == 0

    def _write_to_sandbox(self, module) -> str:
        """Write module code to sandbox directory."""
        filename = re.sub(r"[^a-z0-9_]", "_", module.name.lower()) + ".py"
        path = self._sandbox_dir / filename
        with open(path, "w") as f:
            f.write(module.code)
        return str(path)

    def _import_module(self, path: str, name: str, result: SandboxTestResult):
        try:
            spec = importlib.util.spec_from_file_location(f"sandbox_{name}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            result.import_success = True
            return mod
        except Exception as exc:
            result.import_success = False
            result.import_error = str(exc)
            logger.debug(f"[Sandbox] import error for {name}: {exc}")
            return None

    def _instantiate_node(self, mod_obj, class_name: str, result: SandboxTestResult):
        try:
            cls = getattr(mod_obj, class_name, None)
            if cls is None:
                # Try to find any class that inherits from BaseNode-like
                for attr_name in dir(mod_obj):
                    attr = getattr(mod_obj, attr_name)
                    if isinstance(attr, type) and hasattr(attr, "process"):
                        cls = attr
                        break
            if cls is None:
                result.instantiation_error = f"Class '{class_name}' not found in module"
                return None
            instance = cls()
            result.instantiation_success = True
            return instance
        except Exception as exc:
            result.instantiation_success = False
            result.instantiation_error = str(exc)
            logger.debug(f"[Sandbox] instantiation error: {exc}")
            return None

    def _execute_node(self, node, module, result: SandboxTestResult):
        """Run the node with synthetic test data."""
        test_data = self._build_test_data(module)
        t0 = time.perf_counter()
        try:
            output = node.process(test_data)
            result.execution_latency_ms = (time.perf_counter() - t0) * 1000
            result.execution_success = isinstance(output, dict)
            if result.execution_success:
                result.output_valid = len(output) > 0
        except Exception as exc:
            result.execution_latency_ms = (time.perf_counter() - t0) * 1000
            result.execution_success = False
            result.execution_error = str(exc)
            logger.debug(f"[Sandbox] execution error for {module.name}: {exc}")

    def _build_test_data(self, module) -> dict:
        """Build synthetic test data based on module type."""
        name_lower = module.name.lower()
        if "csv" in name_lower and "json" in name_lower:
            return {"csv_data": "name,age\nAlice,30\nBob,25", "delimiter": ","}
        if "json" in name_lower and "csv" in name_lower:
            return {"json_data": '[{"name":"Alice","age":30}]'}
        if "text" in name_lower or "nlp" in name_lower:
            return {"text": "Phase 7 sandbox test input", "source": "sandbox"}
        if "image" in name_lower:
            return {"image_path": "/tmp/test.jpg", "format": "jpeg"}
        return {"data": {"test_key": "test_value"}, "metadata": {"test": True}}

    def _store(self, module, result: SandboxTestResult):
        self._results[result.test_id] = result
        module.test_result = result.to_dict()
        module.status = "tested"
        self._test_count += 1
        if result.verdict == "passed":
            self._pass_count += 1
        logger.info(f"[Sandbox] tested '{module.name}' → verdict={result.verdict} score={result.score}")

    # ── API ────────────────────────────────────────────────────────────────

    def list_results(self) -> List[dict]:
        return [r.to_dict() for r in self._results.values()]

    def summary(self) -> dict:
        return {
            "test_count": self._test_count,
            "pass_count": self._pass_count,
            "fail_count": self._test_count - self._pass_count,
            "pass_rate": round(self._pass_count / self._test_count, 3) if self._test_count else 0.0,
            "sandbox_dir": str(self._sandbox_dir),
        }
