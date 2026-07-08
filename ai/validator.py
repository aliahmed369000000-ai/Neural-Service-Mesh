"""
Phase 6 Validator — Pre-Phase 7 Readiness Report
==================================================
Analyses the entire codebase and live system to produce a full
validation report before transitioning to Phase 7 (External World Integration).

Report includes:
  • File count & breakdown by layer
  • Node count (registered + generated)
  • Route count (active routes in memory)
  • Agent count (live + historical)
  • Module usage analysis (is each module actually imported/called?)
  • Phase coverage 1-6 (which phases are truly active)
  • Dead code detection (files never imported by main.py or app.py)
  • Phase 7 readiness score

Usage:
  from ai.phase6_validator import Phase6Validator
  report = Phase6Validator(mesh).generate()
"""
from __future__ import annotations

import ast
import os
import sys
import importlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

logger = logging.getLogger(__name__)

# ── Module→Phase mapping ───────────────────────────────────────────────────
_MODULE_PHASE_MAP: Dict[str, int] = {
    # Phase 1 — Core
    "core.engine": 1,
    "core.graph": 1,
    "core.node": 1,
    "core.registry": 1,
    "storage.db": 1,
    "storage.file_storage": 1,
    "connectors.data_transformer": 1,
    "connectors.base_connector": 1,
    "services.input_service": 1,
    "services.processor_service": 1,
    "services.output_service": 1,
    "services.dynamic_node": 1,
    "logs.mesh_logger": 1,
    # Phase 2
    "ai.decision": 2,
    # Phase 3
    "knowledge.knowledge_store": 3,
    "ai.semantic_matcher": 3,
    "ai.scoring_engine": 3,
    "ai.memory_engine": 3,
    "ai.discovery_engine": 3,
    "ai.routing_engine": 3,
    "ai.goal_planner": 3,
    "ai.optimization_engine": 3,
    # Phase 4
    "ai.learning_validator": 4,
    "ai.reputation_engine": 4,
    "ai.simulation_engine": 4,
    # Phase 5
    "ai.service_generator": 5,
    "ai.gap_detector": 5,
    "ai.capability_marketplace": 5,
    "ai.multi_goal_planner": 5,
    "ai.governor": 5,
    "ai.evolution_engine": 5,
    # Phase 6
    "ai.agent_factory": 6,
    "ai.swarm_coordinator": 6,
    "ai.self_optimizer": 6,
    "ai.simulation_lab": 6,
    "ai.meta_reasoner": 6,
    "ai.economic_engine": 6,
    "ai.system_dna": 6,
}

# Phases that should be active in v6
_EXPECTED_PHASES = {1, 2, 3, 4, 5, 6}


class Phase6Validator:
    """
    Validates the Neural Service Mesh before Phase 7.
    Works on the live mesh object AND the codebase on disk.
    """

    VERSION = "1.0.0"

    def __init__(self, mesh, project_root: Optional[str] = None):
        self.mesh = mesh
        self.project_root = Path(project_root or self._detect_root()).resolve()
        self._import_map: Dict[str, Set[str]] = {}   # file → set of modules it imports
        self._all_py_files: List[Path] = []
        self._entry_points: List[str] = ["main.py", "api/app.py"]

    # ── Root detection ─────────────────────────────────────────────────────

    def _detect_root(self) -> str:
        """Try to find project root by walking up from this file."""
        here = Path(__file__).resolve().parent
        for p in [here.parent, here.parent.parent]:
            if (p / "main.py").exists():
                return str(p)
        return str(here.parent)

    # ── File analysis ──────────────────────────────────────────────────────

    def _collect_python_files(self) -> List[Path]:
        """Collect all .py files under project root (excluding tests, venv)."""
        skip_dirs = {"venv", ".venv", "__pycache__", ".git", "node_modules", "models"}
        files = []
        for path in self.project_root.rglob("*.py"):
            if any(part in skip_dirs for part in path.parts):
                continue
            if path.stat().st_size == 0:
                continue
            files.append(path)
        return sorted(files)

    def _extract_imports(self, filepath: Path) -> Set[str]:
        """Parse a .py file and extract all imported module names."""
        try:
            source = filepath.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=str(filepath))
        except SyntaxError:
            return set()

        imports: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Record full dotted path
                    imports.add(node.module)
                    # Also record top-level package
                    imports.add(node.module.split(".")[0])
        return imports

    def _build_import_map(self) -> Dict[str, Set[str]]:
        """Build {relative_path: set_of_imported_modules} for all py files."""
        result: Dict[str, Set[str]] = {}
        for f in self._all_py_files:
            rel = str(f.relative_to(self.project_root))
            result[rel] = self._extract_imports(f)
        return result

    # ── Reachability (dead code) ───────────────────────────────────────────

    def _reachable_modules(self) -> Set[str]:
        """
        BFS from entry points to find all modules reachable via imports.
        Returns set of relative paths (e.g. 'ai/scoring_engine.py').
        """
        # Build module → file mapping
        mod_to_file: Dict[str, str] = {}
        for f in self._all_py_files:
            rel = str(f.relative_to(self.project_root))
            # Convert path to dotted module name
            mod = rel.replace(os.sep, ".").replace("/", ".").removesuffix(".py")
            mod_to_file[mod] = rel
            # also by package.module shorthand
            parts = mod.split(".")
            if len(parts) >= 2:
                mod_to_file[f"{parts[-2]}.{parts[-1]}"] = rel

        visited: Set[str] = set()
        queue: List[str] = list(self._entry_points)

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            imports = self._import_map.get(current, set())
            for imp in imports:
                # Try to resolve import to a file
                for key, fpath in mod_to_file.items():
                    if key == imp or key.endswith("." + imp) or imp in key:
                        if fpath not in visited:
                            queue.append(fpath)

        return visited

    # ── Live mesh stats ────────────────────────────────────────────────────

    def _get_live_stats(self) -> Dict[str, Any]:
        """Pull stats from the live mesh object."""
        stats: Dict[str, Any] = {}
        try:
            stats["node_count"] = self.mesh.registry.count()
        except Exception:
            stats["node_count"] = 0

        try:
            routes = self.mesh.memory.all_routes()
            stats["route_count"] = len(routes)
            stats["healthy_routes"] = sum(1 for r in routes if r.get("health") == "healthy")
        except Exception:
            stats["route_count"] = 0
            stats["healthy_routes"] = 0

        try:
            agent_summary = self.mesh.agent_factory.summary()
            stats["agent_count_live"] = agent_summary.get("active_agents", 0)
            stats["agent_count_total"] = agent_summary.get("total_spawned", 0)
        except Exception:
            stats["agent_count_live"] = 0
            stats["agent_count_total"] = 0

        try:
            scores = self.mesh.scoring.list_scores()
            stats["connection_scores_count"] = len(scores)
            if scores:
                avg = sum(s.get("connection_score", 0) for s in scores) / len(scores)
                stats["avg_connection_score"] = round(avg, 4)
            else:
                stats["avg_connection_score"] = 0.0
        except Exception:
            stats["connection_scores_count"] = 0
            stats["avg_connection_score"] = 0.0

        try:
            dna = self.mesh.system_dna.summary()
            stats["dna_snapshots"] = dna.get("total_snapshots", 0)
        except Exception:
            stats["dna_snapshots"] = 0

        try:
            swarm = self.mesh.swarm.summary()
            stats["swarm_tasks_run"] = swarm.get("total_executions", 0)
        except Exception:
            stats["swarm_tasks_run"] = 0

        try:
            rep = self.mesh.reputation.summary()
            stats["reputation_nodes_tracked"] = rep.get("nodes_tracked", 0)
        except Exception:
            stats["reputation_nodes_tracked"] = 0

        return stats

    # ── Module usage analysis ──────────────────────────────────────────────

    def _module_usage_analysis(self, reachable: Set[str]) -> Dict[str, Any]:
        """
        For each known module, determine:
          - Is it physically present?
          - Is it reachable from entry points?
          - Which phase does it belong to?
        """
        results: List[Dict[str, Any]] = []
        unreachable: List[str] = []

        for mod_dotted, phase in _MODULE_PHASE_MAP.items():
            # Convert dotted module to relative path
            file_rel = mod_dotted.replace(".", "/") + ".py"
            file_path = self.project_root / file_rel

            present = file_path.exists()
            reachable_flag = file_rel in reachable

            entry = {
                "module": mod_dotted,
                "phase": phase,
                "file": file_rel,
                "present": present,
                "reachable_from_entrypoints": reachable_flag,
                "status": "✅ active" if (present and reachable_flag)
                          else "⚠️ present_not_imported" if (present and not reachable_flag)
                          else "❌ missing",
            }
            results.append(entry)
            if present and not reachable_flag:
                unreachable.append(mod_dotted)

        # Sort by phase then module name
        results.sort(key=lambda x: (x["phase"], x["module"]))
        return {
            "modules": results,
            "unreachable_modules": unreachable,
            "total_tracked": len(results),
            "active_count": sum(1 for r in results if r["present"] and r["reachable_from_entrypoints"]),
            "present_not_imported": len(unreachable),
            "missing_count": sum(1 for r in results if not r["present"]),
        }

    # ── Phase coverage ─────────────────────────────────────────────────────

    def _phase_coverage(self, module_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine coverage per phase (1-6)."""
        from collections import defaultdict
        phase_status: Dict[int, Dict[str, Any]] = {}

        per_phase: Dict[int, List[dict]] = defaultdict(list)
        for m in module_analysis["modules"]:
            per_phase[m["phase"]].append(m)

        for phase_num in sorted(_EXPECTED_PHASES):
            mods = per_phase.get(phase_num, [])
            total = len(mods)
            active = sum(1 for m in mods if m["present"] and m["reachable_from_entrypoints"])
            present = sum(1 for m in mods if m["present"])
            coverage_pct = round((active / total * 100) if total else 0, 1)

            phase_status[phase_num] = {
                "phase": phase_num,
                "total_modules": total,
                "present": present,
                "active": active,
                "coverage_pct": coverage_pct,
                "status": "✅ complete" if active == total and total > 0
                          else "⚠️ partial" if active > 0
                          else "❌ inactive",
            }

        overall_active = sum(p["active"] for p in phase_status.values())
        overall_total = sum(p["total_modules"] for p in phase_status.values())
        overall_pct = round((overall_active / overall_total * 100) if overall_total else 0, 1)

        return {
            "phases": phase_status,
            "overall_coverage_pct": overall_pct,
            "phases_complete": sum(1 for p in phase_status.values() if p["status"] == "✅ complete"),
            "phases_partial": sum(1 for p in phase_status.values() if p["status"] == "⚠️ partial"),
            "phases_inactive": sum(1 for p in phase_status.values() if p["status"] == "❌ inactive"),
        }

    # ── Dead code detection ────────────────────────────────────────────────

    def _dead_code_detection(self, reachable: Set[str]) -> Dict[str, Any]:
        """Find .py files that are never imported from entry points."""
        all_files = {str(f.relative_to(self.project_root)) for f in self._all_py_files}
        dead = sorted(all_files - reachable - set(self._entry_points))

        # Categorise dead files
        dead_by_layer: Dict[str, List[str]] = {}
        for f in dead:
            layer = f.split("/")[0] if "/" in f else "root"
            dead_by_layer.setdefault(layer, []).append(f)

        return {
            "dead_files": dead,
            "dead_count": len(dead),
            "total_files": len(all_files),
            "reachable_count": len(reachable),
            "dead_by_layer": dead_by_layer,
            "dead_pct": round(len(dead) / len(all_files) * 100 if all_files else 0, 1),
        }

    # ── File breakdown ─────────────────────────────────────────────────────

    def _file_breakdown(self) -> Dict[str, Any]:
        """Count files per directory layer."""
        breakdown: Dict[str, int] = {}
        total_lines = 0
        for f in self._all_py_files:
            rel = f.relative_to(self.project_root)
            layer = rel.parts[0] if len(rel.parts) > 1 else "root"
            breakdown[layer] = breakdown.get(layer, 0) + 1
            try:
                total_lines += len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
            except Exception:
                pass
        return {
            "total_py_files": len(self._all_py_files),
            "total_lines_of_code": total_lines,
            "by_layer": breakdown,
        }

    # ── Phase 7 readiness score ────────────────────────────────────────────

    def _phase7_readiness(
        self,
        live: Dict[str, Any],
        coverage: Dict[str, Any],
        dead: Dict[str, Any],
        module_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute a 0-100 readiness score based on:
          - Phase 6 coverage (40 pts)
          - Live system health (30 pts)
          - Low dead-code ratio (15 pts)
          - Agents & swarm operational (15 pts)
        """
        score = 0.0
        notes: List[str] = []

        # 1. Phase 6 coverage (40 pts)
        p6 = coverage["phases"].get(6, {})
        p6_pct = p6.get("coverage_pct", 0)
        phase_score = p6_pct * 0.40
        score += phase_score
        notes.append(f"Phase 6 module coverage: {p6_pct}% → +{phase_score:.1f} pts")

        # 2. Live system health (30 pts)
        nodes = live.get("node_count", 0)
        routes = live.get("route_count", 0)
        conn_score = live.get("avg_connection_score", 0.0)
        live_pts = min(30.0, (
            (min(nodes, 5) / 5 * 10) +    # up to 10 pts for 5+ nodes
            (min(routes, 5) / 5 * 10) +   # up to 10 pts for 5+ routes
            (conn_score * 10)              # up to 10 pts for avg score = 1.0
        ))
        score += live_pts
        notes.append(f"Live system (nodes={nodes}, routes={routes}, avg_score={conn_score:.2f}) → +{live_pts:.1f} pts")

        # 3. Dead code penalty (15 pts)
        dead_pct = dead.get("dead_pct", 0)
        dead_pts = max(0, 15 - dead_pct * 0.15)
        score += dead_pts
        notes.append(f"Dead code: {dead_pct:.1f}% of files → +{dead_pts:.1f} pts")

        # 4. Agents & swarm (15 pts)
        agents_total = live.get("agent_count_total", 0)
        swarm_tasks = live.get("swarm_tasks_run", 0)
        agent_pts = min(15.0, (
            min(agents_total, 5) / 5 * 8 +
            min(swarm_tasks, 5) / 5 * 7
        ))
        score += agent_pts
        notes.append(f"Agents/Swarm (spawned={agents_total}, tasks={swarm_tasks}) → +{agent_pts:.1f} pts")

        overall = round(min(score, 100), 1)
        verdict = (
            "🚀 Ready for Phase 7" if overall >= 75
            else "⚠️ Needs improvement before Phase 7" if overall >= 50
            else "🔴 Not ready — significant gaps"
        )

        return {
            "score": overall,
            "max_score": 100,
            "verdict": verdict,
            "breakdown_notes": notes,
            "recommendations": self._phase7_recommendations(live, coverage, dead),
        }

    def _phase7_recommendations(
        self,
        live: Dict[str, Any],
        coverage: Dict[str, Any],
        dead: Dict[str, Any],
    ) -> List[str]:
        recs: List[str] = []
        p6_pct = coverage["phases"].get(6, {}).get("coverage_pct", 0)
        if p6_pct < 100:
            recs.append("Ensure all Phase 6 modules are imported and reachable from main.py/app.py")
        if live.get("node_count", 0) < 3:
            recs.append("Register at least 3 nodes to prove routing is functional")
        if live.get("route_count", 0) < 1:
            recs.append("Run at least one pipeline to populate route memory before Phase 7")
        if live.get("agent_count_total", 0) < 1:
            recs.append("Spawn at least one agent to validate AgentFactory is operational")
        if live.get("dna_snapshots", 0) < 1:
            recs.append("Take a DNA snapshot before Phase 7 to enable rollback capability")
        if dead.get("dead_pct", 0) > 30:
            recs.append(f"High dead-code ratio ({dead['dead_pct']:.1f}%). Consider importing or removing unused modules.")
        if not recs:
            recs.append("System looks good — proceed to Phase 7 (External World Integration)")
        return recs

    # ── Main entry point ───────────────────────────────────────────────────

    def generate(self) -> Dict[str, Any]:
        """Generate the full Phase 6 Validation Report."""
        print("\n" + "=" * 65)
        print("  Neural Service Mesh — Phase 6 Validation Report")
        print("=" * 65 + "\n")

        print("  [1/6] Collecting Python files...")
        self._all_py_files = self._collect_python_files()

        print("  [2/6] Building import map...")
        self._import_map = self._build_import_map()

        print("  [3/6] Tracing reachability from entry points...")
        reachable = self._reachable_modules()

        print("  [4/6] Analysing module usage & phase coverage...")
        file_stats = self._file_breakdown()
        module_analysis = self._module_usage_analysis(reachable)
        phase_coverage = self._phase_coverage(module_analysis)
        dead_code = self._dead_code_detection(reachable)

        print("  [5/6] Collecting live system stats...")
        live_stats = self._get_live_stats()

        print("  [6/6] Computing Phase 7 readiness score...")
        readiness = self._phase7_readiness(live_stats, phase_coverage, dead_code, module_analysis)

        report = {
            "report_type": "Phase6ValidationReport",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "validator_version": self.VERSION,
            "project_root": str(self.project_root),
            # ── File stats
            "files": file_stats,
            # ── Live mesh stats
            "live_system": live_stats,
            # ── Module usage
            "module_analysis": module_analysis,
            # ── Phase coverage 1-6
            "phase_coverage": phase_coverage,
            # ── Dead code
            "dead_code": dead_code,
            # ── Readiness
            "phase7_readiness": readiness,
        }

        self._print_report(report)
        return report

    # ── Pretty-print ───────────────────────────────────────────────────────

    def _print_report(self, r: Dict[str, Any]):
        f = r["files"]
        l = r["live_system"]
        pc = r["phase_coverage"]
        dc = r["dead_code"]
        ma = r["module_analysis"]
        rd = r["phase7_readiness"]

        print("\n── FILES ─────────────────────────────────────────────────────")
        print(f"  Total Python files : {f['total_py_files']}")
        print(f"  Total lines of code: {f['total_lines_of_code']:,}")
        print(f"  By layer:")
        for layer, count in sorted(f["by_layer"].items()):
            print(f"    {layer:<20} {count} files")

        print("\n── LIVE SYSTEM ───────────────────────────────────────────────")
        print(f"  Registered nodes     : {l['node_count']}")
        print(f"  Routes in memory     : {l['route_count']}  (healthy: {l['healthy_routes']})")
        print(f"  Connection scores    : {l['connection_scores_count']}  (avg: {l['avg_connection_score']})")
        print(f"  Agents (total/live)  : {l['agent_count_total']} / {l['agent_count_live']}")
        print(f"  Swarm tasks run      : {l['swarm_tasks_run']}")
        print(f"  DNA snapshots        : {l['dna_snapshots']}")
        print(f"  Reputation nodes     : {l['reputation_nodes_tracked']}")

        print("\n── MODULE USAGE ──────────────────────────────────────────────")
        print(f"  Tracked modules      : {ma['total_tracked']}")
        print(f"  Active (reachable)   : {ma['active_count']}")
        print(f"  Present not imported : {ma['present_not_imported']}")
        print(f"  Missing              : {ma['missing_count']}")
        if ma["unreachable_modules"]:
            print(f"  ⚠️  Not imported from entry points:")
            for m in ma["unreachable_modules"]:
                print(f"      • {m}")

        print("\n── PHASE COVERAGE 1–6 ────────────────────────────────────────")
        for ph_num, ph_data in sorted(pc["phases"].items()):
            bar = "█" * int(ph_data["coverage_pct"] / 10)
            bar = bar.ljust(10)
            print(f"  Phase {ph_num}  [{bar}] {ph_data['coverage_pct']:5.1f}%  "
                  f"{ph_data['status']}  "
                  f"({ph_data['active']}/{ph_data['total_modules']} modules)")
        print(f"\n  Overall coverage : {pc['overall_coverage_pct']}%  "
              f"({pc['phases_complete']} complete, "
              f"{pc['phases_partial']} partial, "
              f"{pc['phases_inactive']} inactive)")

        print("\n── DEAD CODE DETECTION ───────────────────────────────────────")
        print(f"  Total files     : {dc['total_files']}")
        print(f"  Reachable       : {dc['reachable_count']}")
        print(f"  Dead (unreached): {dc['dead_count']}  ({dc['dead_pct']}%)")
        if dc["dead_files"]:
            print(f"  Dead files by layer:")
            for layer, files in sorted(dc["dead_by_layer"].items()):
                print(f"    [{layer}]")
                for df in files:
                    print(f"      • {df}")

        print("\n── PHASE 7 READINESS ─────────────────────────────────────────")
        print(f"  Score   : {rd['score']} / {rd['max_score']}")
        print(f"  Verdict : {rd['verdict']}")
        print(f"\n  Score breakdown:")
        for note in rd["breakdown_notes"]:
            print(f"    • {note}")
        print(f"\n  Recommendations:")
        for rec in rd["recommendations"]:
            print(f"    ▶ {rec}")

        print("\n" + "=" * 65)
        print(f"  Report generated at {r['generated_at']}")
        print("=" * 65 + "\n")
