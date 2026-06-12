"""
Phase 7 – Autonomous Evolution Pipeline
==========================================
The top-level orchestrator for Phase 7 self-evolution.

Command: python main.py --mode evolve7

Full cycle (Observe → Detect Gap → Generate Module → Test → Score → Approve → Deploy → Update Knowledge):

  1. Observe      – SensorHub polls all sensors; events fed to EnvironmentModel
  2. Detect Gap   – GapDetectionEngine + SelfAwarenessEngine identify what's missing
  3. Generate     – CodeGenerationEngine writes Python code for the gap
  4. Test         – SandboxTestingLab runs the module in isolation
  5. Score        – sandbox produces a 0-100 score
  6. Approve      – P7GovernanceLayer reviews; verdict = Approve / Reject / Needs Revision
  7. Deploy       – Approved module copied to services/ and registered in mesh
  8. Update       – KnowledgeStore + EnvironmentModel + ObjectivesEngine updated

File: ai/evolution_pipeline.py
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PipelineStepResult:
    def __init__(self, step: str, success: bool, details: dict):
        self.step = step
        self.success = success
        self.details = details
        self.ts = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {"step": self.step, "success": self.success, "ts": self.ts, **self.details}


class EvolutionCycleP7:
    """Full Phase 7 evolution cycle result."""

    def __init__(self, cycle_number: int):
        self.cycle_id = f"p7_cycle_{cycle_number}_{int(time.time())}"
        self.cycle_number = cycle_number
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: Optional[str] = None
        self.steps: List[PipelineStepResult] = []
        self.sensor_events_count: int = 0
        self.gaps_detected: int = 0
        self.modules_generated: int = 0
        self.modules_tested: int = 0
        self.modules_approved: int = 0
        self.modules_deployed: int = 0
        self.objectives_measured: dict = {}
        self.recommendations: List[dict] = []
        self.errors: List[str] = []

    def add_step(self, step: str, success: bool, **details):
        self.steps.append(PipelineStepResult(step, success, details))

    def complete(self):
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "cycle_number": self.cycle_number,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "summary": {
                "sensor_events": self.sensor_events_count,
                "gaps_detected": self.gaps_detected,
                "modules_generated": self.modules_generated,
                "modules_tested": self.modules_tested,
                "modules_approved": self.modules_approved,
                "modules_deployed": self.modules_deployed,
            },
            "objectives": self.objectives_measured,
            "recommendations": self.recommendations,
            "steps": [s.to_dict() for s in self.steps],
            "errors": self.errors,
        }


class EvolutionPipeline:
    """
    Phase 7: Autonomous Evolution Pipeline.

    Coordinates all Phase 7 components through a complete Observe → Deploy cycle.
    """

    def __init__(
        self,
        mesh=None,
        sensor_hub=None,
        environment_model=None,
        self_awareness=None,
        code_generator=None,
        sandbox_lab=None,
        governance_p7=None,
        objectives_engine=None,
        gap_detector=None,
        knowledge_store=None,
        services_dir: str = "./services",
    ):
        self._mesh = mesh
        self._sensor_hub = sensor_hub
        self._env_model = environment_model
        self._awareness = self_awareness
        self._codegen = code_generator
        self._sandbox = sandbox_lab
        self._governance = governance_p7
        self._objectives = objectives_engine
        self._gap_detector = gap_detector
        self._knowledge = knowledge_store
        self._services_dir = services_dir
        self._cycle_count = 0
        self._cycle_history: List[dict] = []
        self._deployed_modules: List[dict] = []

    def run_cycle(self, verbose: bool = True) -> EvolutionCycleP7:
        """Execute one full Phase 7 evolution cycle."""
        self._cycle_count += 1
        cycle = EvolutionCycleP7(self._cycle_count)

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Phase 7 Evolution Cycle #{self._cycle_count}")
            print(f"{'='*60}")

        # ── Step 1: Observe (sensor poll + world model snapshot) ───────────
        try:
            if self._sensor_hub:
                events = self._sensor_hub.poll_now()
                cycle.sensor_events_count = len(events)
                if verbose:
                    print(f"  [1/8] Observe: {len(events)} sensor events")
                cycle.add_step("observe", True, sensor_events=len(events))
            if self._env_model and self._mesh:
                self._env_model.snapshot_from_mesh(self._mesh)
                cycle.add_step("world_model_snapshot", True)
        except Exception as exc:
            cycle.errors.append(f"observe: {exc}")
            cycle.add_step("observe", False, error=str(exc))

        # ── Step 2: Self-awareness introspection ───────────────────────────
        try:
            if self._awareness:
                report = self._awareness.introspect()
                if verbose:
                    print(f"  [2/8] Self-Awareness: health={report.system_health_score:.2f}, "
                          f"nodes={report.node_count}, agents={report.active_agents}")
                cycle.add_step("self_awareness", True,
                               health=report.system_health_score,
                               node_count=report.node_count,
                               insights=report.insights)
        except Exception as exc:
            cycle.errors.append(f"self_awareness: {exc}")
            cycle.add_step("self_awareness", False, error=str(exc))

        # ── Step 3: Measure objectives ────────────────────────────────────
        try:
            if self._objectives and self._mesh:
                measurements = self._objectives.measure_from_mesh(self._mesh)
                cycle.objectives_measured = {k: round(v, 4) for k, v in measurements.items()}
                recs = self._objectives.get_recommendations()
                cycle.recommendations = recs
                if verbose:
                    print(f"  [3/8] Objectives: measured {len(measurements)} metrics, "
                          f"{len(recs)} recommendations")
                cycle.add_step("objectives", True, measurements=len(measurements),
                               recommendations=len(recs))
        except Exception as exc:
            cycle.errors.append(f"objectives: {exc}")
            cycle.add_step("objectives", False, error=str(exc))

        # ── Step 4: Gap detection ─────────────────────────────────────────
        gaps = []
        try:
            if self._gap_detector:
                detected = self._gap_detector.scan()
                gaps = [g.to_dict() for g in detected]
                cycle.gaps_detected = len(gaps)
                if verbose:
                    print(f"  [4/8] Gap Detection: {len(gaps)} gaps found")
                cycle.add_step("gap_detection", True, gaps_found=len(gaps),
                               gap_types=[g.get("gap_type") for g in gaps])
        except Exception as exc:
            cycle.errors.append(f"gap_detection: {exc}")
            cycle.add_step("gap_detection", False, error=str(exc))

        # ── Steps 5-8: For each gap: Generate → Test → Approve → Deploy ───
        for gap in gaps[:3]:   # max 3 per cycle to stay safe
            module_name = gap.get("missing_service", "UnknownModule")
            if verbose:
                print(f"\n  --- Processing gap: {module_name} ---")

            # 5. Generate
            module = None
            try:
                if self._codegen:
                    module = self._codegen.generate_from_gap(gap)
                    cycle.modules_generated += 1
                    if verbose:
                        print(f"  [5/8] Generated '{module_name}' "
                              f"(syntax_valid={module.syntax_valid})")
                    cycle.add_step(f"generate:{module_name}", module.syntax_valid,
                                   syntax_valid=module.syntax_valid)
            except Exception as exc:
                cycle.errors.append(f"generate:{module_name}: {exc}")
                continue

            if not module or not module.syntax_valid:
                continue

            # 6. Test in sandbox
            test_result_dict = {}
            try:
                if self._sandbox:
                    test_result = self._sandbox.test_module(module)
                    test_result_dict = test_result.to_dict()
                    cycle.modules_tested += 1
                    if verbose:
                        print(f"  [6/8] Sandbox Test: score={test_result.score:.1f} "
                              f"verdict={test_result.verdict}")
                    cycle.add_step(f"test:{module_name}", test_result.score >= 40,
                                   score=test_result.score, verdict=test_result.verdict)
            except Exception as exc:
                cycle.errors.append(f"test:{module_name}: {exc}")
                continue

            # 7. Governance approval
            decision = None
            try:
                if self._governance and test_result_dict:
                    decision = self._governance.review(module, test_result_dict)
                    if verbose:
                        print(f"  [7/8] Governance: {decision.verdict} — {decision.reason}")
                    cycle.add_step(f"approve:{module_name}", decision.allowed,
                                   verdict=decision.verdict, reason=decision.reason)
            except Exception as exc:
                cycle.errors.append(f"approve:{module_name}: {exc}")
                continue

            if not decision or not decision.allowed:
                continue

            cycle.modules_approved += 1

            # 8. Deploy
            try:
                deployed_path = self._deploy_module(module)
                if deployed_path:
                    cycle.modules_deployed += 1
                    self._deployed_modules.append({
                        "module_id": module.module_id,
                        "name": module.name,
                        "file_path": deployed_path,
                        "deployed_at": datetime.now(timezone.utc).isoformat(),
                        "cycle": self._cycle_count,
                    })
                    if verbose:
                        print(f"  [8/8] Deployed → {deployed_path}")
                    cycle.add_step(f"deploy:{module_name}", True, file_path=deployed_path)

                    # Update knowledge
                    if self._knowledge:
                        try:
                            self._knowledge.write_graph_metrics({
                                "phase7_deployed": self._deployed_modules[-20:]
                            })
                        except Exception:
                            pass
            except Exception as exc:
                cycle.errors.append(f"deploy:{module_name}: {exc}")
                cycle.add_step(f"deploy:{module_name}", False, error=str(exc))

        cycle.complete()
        self._cycle_history.append(cycle.to_dict())
        self._cycle_history = self._cycle_history[-50:]

        if verbose:
            print(f"\n  Cycle #{self._cycle_count} complete:")
            print(f"    Generated: {cycle.modules_generated}  "
                  f"Tested: {cycle.modules_tested}  "
                  f"Approved: {cycle.modules_approved}  "
                  f"Deployed: {cycle.modules_deployed}")
            if cycle.errors:
                print(f"    Errors: {len(cycle.errors)}")

        return cycle

    def run_cycles(self, n: int = 3, verbose: bool = True) -> List[dict]:
        results = []
        for _ in range(n):
            cycle = self.run_cycle(verbose=verbose)
            results.append(cycle.to_dict())
        return results

    def _deploy_module(self, module) -> str:
        """Move approved module to the services directory."""
        if not module.file_path or not os.path.exists(module.file_path):
            # Write to sandbox first if not there
            if self._codegen:
                self._codegen.write_to_file(module, subdir="generated")
                if not module.file_path:
                    return ""

        # Copy from sandbox → services/generated/
        import re
        filename = re.sub(r"[^a-z0-9_]", "_", module.name.lower()) + ".py"
        dest_dir = os.path.join(self._services_dir, "generated")
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        try:
            shutil.copy2(module.file_path, dest_path)
            module.status = "deployed"
            return dest_path
        except Exception as exc:
            logger.error(f"[Pipeline] deploy copy error: {exc}")
            return ""

    def summary(self) -> dict:
        return {
            "total_cycles": self._cycle_count,
            "total_deployed": len(self._deployed_modules),
            "recent_cycle": self._cycle_history[-1] if self._cycle_history else None,
            "deployed_modules": self._deployed_modules[-10:],
        }

    def get_history(self, limit: int = 10) -> List[dict]:
        return self._cycle_history[-limit:]
