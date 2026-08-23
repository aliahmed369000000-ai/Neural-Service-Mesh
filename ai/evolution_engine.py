"""
Phase 5 – Evolution Engine
============================
Orchestrates full autonomous evolution cycles.

Command: python main.py --mode evolve

Each cycle:
  1. GapDetector scans the graph for missing capabilities
  2. ServiceGenerator proposes new nodes for detected gaps
  3. AIGovernanceLayer validates each proposal
  4. Approved services are instantiated and registered
  5. KnowledgeGraph is updated with the new topology
  6. EvolutionHistory records what changed

The system transitions from:
  "System that learns" → "System that creates and manages its own ecosystem"
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class EvolutionCycleResult:
    """Full report from a single evolution cycle."""

    def __init__(self, cycle_number: int):
        self.cycle_number = cycle_number
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: Optional[str] = None
        self.gaps_found: List[dict] = []
        self.services_proposed: List[dict] = []
        self.services_approved: List[dict] = []
        self.services_rejected: List[dict] = []
        self.nodes_added: List[str] = []
        self.connections_added: List[dict] = []
        self.knowledge_updates: List[str] = []
        self.governance_decisions: List[dict] = []
        self.errors: List[str] = []

    def complete(self):
        self.completed_at = datetime.now(timezone.utc).isoformat()

    @property
    def summary(self) -> dict:
        return {
            "cycle": self.cycle_number,
            "gaps_found": len(self.gaps_found),
            "services_proposed": len(self.services_proposed),
            "services_approved": len(self.services_approved),
            "services_rejected": len(self.services_rejected),
            "nodes_added": len(self.nodes_added),
            "connections_added": len(self.connections_added),
        }

    def to_dict(self) -> dict:
        return {
            "cycle_number": self.cycle_number,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "summary": self.summary,
            "gaps_found": self.gaps_found,
            "services_proposed": self.services_proposed,
            "services_approved": self.services_approved,
            "services_rejected": self.services_rejected,
            "nodes_added": self.nodes_added,
            "connections_added": self.connections_added,
            "knowledge_updates": self.knowledge_updates,
            "governance_decisions": self.governance_decisions,
            "errors": self.errors,
        }


class EvolutionEngine:
    """
    Phase 5: Autonomous Evolution Engine.

    Coordinates the full lifecycle of autonomous system growth:
      GapDetector → ServiceGenerator → Governance → Registration → KnowledgeUpdate

    Runs as a continuous loop or single-shot cycle.
    """

    def __init__(
        self,
        mesh=None,
        gap_detector=None,
        service_generator=None,
        governance=None,
        capability_marketplace=None,
        multi_goal_planner=None,
        knowledge_store=None,
    ):
        self._mesh = mesh
        self._gap_detector = gap_detector
        self._generator = service_generator
        self._governance = governance
        self._marketplace = capability_marketplace
        self._planner = multi_goal_planner
        self._knowledge = knowledge_store
        self._cycle_count = 0
        self._history: List[EvolutionCycleResult] = []
        logger.info("EvolutionEngine initialised (Phase 5)")

    def set_mesh(self, mesh):
        self._mesh = mesh

    # ── Single evolution cycle ─────────────────────────────────────────────

    def run_cycle(self, auto_register: bool = True, verbose: bool = True) -> EvolutionCycleResult:
        """
        Execute one full evolution cycle.

        Steps:
          1. Reset governance cycle counters
          2. Scan for gaps
          3. Generate service specs for each gap
          4. Validate each spec via governance
          5. Instantiate and register approved services
          6. Update knowledge graph
          7. Record history
        """
        self._cycle_count += 1
        cycle = EvolutionCycleResult(self._cycle_count)

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"  Evolution Cycle #{self._cycle_count}")
            logger.info(f"{'='*60}")

        # Step 1: Reset governance counters
        if self._governance:
            self._governance.reset_cycle()

        # Step 2: Scan for gaps
        try:
            new_gaps = self._gap_detector.scan() if self._gap_detector else []
            cycle.gaps_found = [g.to_dict() for g in new_gaps]
            if verbose:
                logger.info(f"  Gaps found: {len(new_gaps)}")
        except Exception as e:
            cycle.errors.append(f"Gap scan error: {e}")
            new_gaps = []

        # Step 3: Generate service specs for each gap
        for gap in new_gaps:
            try:
                spec = self._generator.generate_for_gap(gap.to_dict() if hasattr(gap, 'to_dict') else gap)
                if spec:
                    cycle.services_proposed.append(spec.to_dict())
            except Exception as e:
                cycle.errors.append(f"Generation error: {e}")

        # Step 4: Governance validation
        approved_specs = []
        total_generated = self._generator.summary()["total_generated"] if self._generator else 0

        for spec_dict in cycle.services_proposed:
            if self._governance:
                decision = self._governance.evaluate_generation(spec_dict, total_generated)
                cycle.governance_decisions.append(decision.to_dict())

                if decision.allowed:
                    approved_specs.append(spec_dict)
                    cycle.services_approved.append(spec_dict)
                    if verbose:
                        logger.info(f"  ✓ Approved: '{spec_dict['name']}'")
                else:
                    cycle.services_rejected.append(spec_dict)
                    if verbose:
                        logger.info(f"  ✗ Rejected: '{spec_dict['name']}' — {decision.reason}")
            else:
                # No governance = all approved
                approved_specs.append(spec_dict)
                cycle.services_approved.append(spec_dict)

        # Step 5: Instantiate and register approved services
        if auto_register and self._mesh and approved_specs:
            for spec_dict in approved_specs:
                try:
                    spec_id = spec_dict.get("spec_id")
                    spec = self._generator.get_spec(spec_id) if self._generator else None
                    if not spec:
                        continue

                    # Create node from spec
                    node = self._generator.instantiate_spec(spec)
                    # Find best connection points
                    gap_ctx = spec_dict.get("gap_context", {})
                    connect_to = self._find_connection_target(gap_ctx)

                    node_id = self._mesh.register_node(node, connect_to=connect_to)
                    cycle.nodes_added.append(node_id)

                    # Register in marketplace
                    if self._marketplace:
                        self._marketplace.advertise_from_node(node)

                    if connect_to:
                        cycle.connections_added.append({
                            "from": connect_to,
                            "to": node_id,
                            "node_name": spec_dict["name"],
                        })

                    if verbose:
                        logger.info(f"  + Registered node '{spec_dict['name']}' [{node_id[:8]}]")

                except Exception as e:
                    cycle.errors.append(f"Registration error for '{spec_dict.get('name', '?')}': {e}")

        # Step 6: Update knowledge graph
        cycle.knowledge_updates.extend(self._update_knowledge(cycle))

        # Step 7: Record and complete
        cycle.complete()
        self._history.append(cycle)

        if verbose:
            s = cycle.summary
            logger.info(f"\n  Cycle #{self._cycle_count} complete:")
            logger.info(f"    Gaps: {s['gaps_found']} | Proposed: {s['services_proposed']} | "
                       f"Approved: {s['services_approved']} | Added: {s['nodes_added']}")

        return cycle

    def _find_connection_target(self, gap_context: dict) -> Optional[str]:
        """Find the source node ID from gap context to connect the new service."""
        if not self._mesh:
            return None
        source_name = gap_context.get("source_name", "")
        if not source_name:
            return None
        # Look up by name in registry
        try:
            nodes = self._mesh.registry.list_metadata()
            for node in nodes:
                if node.get("name", "") == source_name:
                    return node.get("node_id")
        except Exception:
            pass
        return None

    def _update_knowledge(self, cycle: EvolutionCycleResult) -> List[str]:
        """Update knowledge store with evolution results."""
        updates = []
        if not self._knowledge:
            return updates

        try:
            # Store evolution history
            history_entry = {
                "cycle": cycle.cycle_number,
                "timestamp": cycle.completed_at,
                "summary": cycle.summary,
            }
            existing = {}
            try:
                existing = self._knowledge.read_custom("evolution_history") or {}
            except Exception:
                pass
            existing[str(cycle.cycle_number)] = history_entry
            self._knowledge.write_custom("evolution_history", existing)
            updates.append("evolution_history")
        except Exception as e:
            logger.warning(f"Knowledge update error: {e}")

        # Update node rankings if mesh is available
        if self._mesh:
            try:
                self._knowledge.update_node_rankings(self._mesh.memory)
                updates.append("node_rankings")
            except Exception:
                pass

        return updates

    # ── Multi-cycle evolution loop ─────────────────────────────────────────

    def evolve(
        self,
        cycles: int = 3,
        auto_register: bool = True,
        verbose: bool = True,
    ) -> dict:
        """
        Run multiple evolution cycles.
        Returns a summary report.
        """
        if verbose:
            logger.info(f"\n🧬 Starting evolution: {cycles} cycle(s)")

        results = []
        for i in range(cycles):
            try:
                result = self.run_cycle(auto_register=auto_register, verbose=verbose)
                results.append(result.to_dict())
            except Exception as e:
                logger.error(f"Cycle {i+1} error: {e}")

        total_gaps = sum(r["summary"]["gaps_found"] for r in results)
        total_added = sum(r["summary"]["nodes_added"] for r in results)
        total_approved = sum(r["summary"]["services_approved"] for r in results)

        report = {
            "cycles_run": len(results),
            "total_gaps_found": total_gaps,
            "total_services_approved": total_approved,
            "total_nodes_added": total_added,
            "cycles": results,
        }

        if verbose:
            logger.info(f"\n🧬 Evolution complete:")
            logger.info(f"   Cycles: {len(results)} | Gaps: {total_gaps} | "
                       f"Approved: {total_approved} | Added: {total_added}")

        return report

    # ── Status ─────────────────────────────────────────────────────────────

    def history(self, limit: int = 10) -> List[dict]:
        return [c.to_dict() for c in self._history[-limit:]]

    def summary(self) -> dict:
        total_nodes = sum(
            len(c.nodes_added) for c in self._history
        )
        total_gaps = sum(
            len(c.gaps_found) for c in self._history
        )
        return {
            "cycles_run": self._cycle_count,
            "total_gaps_processed": total_gaps,
            "total_nodes_generated": total_nodes,
            "gap_detector_connected": self._gap_detector is not None,
            "generator_connected": self._generator is not None,
            "governance_connected": self._governance is not None,
            "marketplace_connected": self._marketplace is not None,
        }

    def _log_evolution(self, entry: dict):
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        log_file = root / "memory" / "evolution_log.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def propose_innovation(self, agent_id: str, innovation_type: str, description: str, proposed_code_changes: Dict[str, str]) -> Dict[str, Any]:
        """اقتراح ابتكار أو تعديل في الكود المصدري مع التحقق من السلامة."""
        import time
        innovation_id = f"inn_{int(time.time())}"
        
        # 1. التحقق الأولي من التغييرات المقترحة
        verification_results = self._verify_proposed_changes(proposed_code_changes)
        
        proposal = {
            "id": innovation_id,
            "agent_id": agent_id,
            "type": innovation_type,
            "desc": description,
            "changes": proposed_code_changes,
            "verification": verification_results,
            "status": "pending_review" if verification_results["safe"] else "rejected_by_safety",
            "ts": datetime.now(timezone.utc).isoformat()
        }
        self._log_evolution(proposal)
        return proposal

    def _verify_proposed_changes(self, changes: Dict[str, str]) -> Dict[str, Any]:
        """فحص أمني وتقني للتعديلات المقترحة قبل قبولها كمسودة."""
        import ast
        import time
        from pathlib import Path
        results = {"safe": True, "errors": []}
        
        for file_path, code in changes.items():
            # فحص السنتكس
            try:
                ast.parse(code)
            except SyntaxError as e:
                results["safe"] = False
                results["errors"].append(f"خطأ سنتكس في {file_path}: {e}")
            
            # فحص الأنماط المحظورة (استخدام Sandbox Lab logic)
            from ai.sandbox_lab import SandboxTestingLab
            lab = SandboxTestingLab()
            
            class MockMod:
                def __init__(self, name, c):
                    self.module_id = f"mod_{int(time.time())}"
                    self.name = name
                    self.code = c
                    self.class_name = "InnovationNode"
                    self.status = "pending"
                    self.test_result = None
            
            # ضمان وجود كلاس متوافق مع الساندبوكس للاختبار
            test_code = code
            if "class " not in code:
                test_code = f"class InnovationNode:\n    def process(self, data): return True\n\n" + code

            mock = MockMod(Path(file_path).stem, test_code)
            res = lab.test_module(mock)
            if not res.safety_passed:
                results["safe"] = False
                results["errors"].extend(res.safety_violations)
                
        return results

    def apply_innovation(self, innovation_id: str) -> Dict[str, Any]:
        """تطبيق ابتكار معتمد فعلياً على الكود المصدري مع آلية تراجع (Rollback)."""
        import json
        import shutil
        import os
        from pathlib import Path
        
        root = Path(__file__).resolve().parent.parent
        log_file = root / "memory" / "evolution_log.jsonl"
        if not log_file.exists():
            return {"ok": False, "error": f"سجل الابتكارات غير موجود في {log_file}."}
            
        proposal = None
        with log_file.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data["id"] == innovation_id:
                    proposal = data
                    break
        
        if not proposal:
            return {"ok": False, "error": "الابتكار غير موجود في السجل."}
            
        if proposal["status"] == "rejected_by_safety":
            return {"ok": False, "error": "لا يمكن تطبيق ابتكار مرفوض أمنياً."}

        applied_files = []
        try:
            for file_path, new_code in proposal["changes"].items():
                p = Path(file_path)
                if not p.is_absolute():
                    p = root / file_path
                
                # 1. إنشاء نسخة احتياطية
                if p.exists():
                    shutil.copy2(p, str(p) + ".bak")
                
                # 2. تطبيق التعديل
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(new_code, encoding="utf-8")
                applied_files.append(file_path)
            
            return {"ok": True, "applied_files": applied_files, "msg": f"تم تطبيق الابتكار {innovation_id} بنجاح."}
        except Exception as e:
            # محاولة التراجع التلقائي في حال الفشل
            for f in applied_files:
                bak = Path(f + ".bak")
                if bak.exists():
                    shutil.move(bak, f)
            return {"ok": False, "error": f"فشل التطبيق: {e}"}

    def rollback_innovation(self, innovation_id: str) -> Dict[str, Any]:
        """التراجع عن ابتكار مطبق واستعادة النسخ الاحتياطية."""
        import json
        import os
        from pathlib import Path
        import shutil
        
        root = Path(__file__).resolve().parent.parent
        log_file = root / "memory" / "evolution_log.jsonl"
        proposal = None
        if not log_file.exists():
             return {"ok": False, "error": "سجل الابتكارات غير موجود."}

        with log_file.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data["id"] == innovation_id:
                    proposal = data
                    break
        
        if not proposal:
            return {"ok": False, "error": "الابتكار غير موجود."}

        restored = []
        for file_path_str in proposal["changes"].keys():
            p = Path(file_path_str)
            if not p.is_absolute():
                p = root / file_path_str
            
            bak = Path(str(p) + ".bak")
            if bak.exists():
                if p.exists(): os.remove(p)
                shutil.move(bak, p)
                restored.append(file_path_str)
        
        return {"ok": True, "restored_files": restored}

def evolution_engine(params: Dict[str, Any]) -> str:
    """أداة الوكيل لإدارة تطوره الذاتي (السيادة التطورية V2)."""
    import json
    action = params.get("action", "status")
    engine = EvolutionEngine()
    
    if action == "propose":
        res = engine.propose_innovation(
            params.get("agent_id", os.environ.get("AGENT_ID", "sovereign_agent")),
            params.get("type", "feature"),
            params.get("description", ""),
            params.get("changes", {})
        )
        return json.dumps({"ok": True, "proposal": res}, ensure_ascii=False)
    
    if action == "apply":
        innovation_id = params.get("innovation_id")
        if not innovation_id: return "❌ evolution_engine: يجب توفير innovation_id."
        res = engine.apply_innovation(innovation_id)
        return json.dumps(res, ensure_ascii=False)
        
    if action == "rollback":
        innovation_id = params.get("innovation_id")
        if not innovation_id: return "❌ evolution_engine: يجب توفير innovation_id."
        res = engine.rollback_innovation(innovation_id)
        return json.dumps(res, ensure_ascii=False)
    
    return json.dumps({"ok": True, "status": "Evolution Engine Active (Phase 5 - V2)"}, ensure_ascii=False)
