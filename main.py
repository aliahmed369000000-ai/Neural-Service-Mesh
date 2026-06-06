from __future__ import annotations
import sys
from typing import Optional

# ── Logging first ──────────────────────────────────────────────────────────
from logs.mesh_logger import MeshLogger
_mesh_logger = MeshLogger(log_dir="./logs", level="INFO")

import logging
logger = logging.getLogger("NeuralServiceMesh.v3")

# ── Core imports ───────────────────────────────────────────────────────────
from storage.file_storage import FileStorage
from storage.db import SQLiteStorage
from core.registry import NodeRegistry
from core.graph import ServiceGraph
from core.engine import ExecutionEngine
from connectors.data_transformer import DataTransformer

# ── Phase 2 AI ─────────────────────────────────────────────────────────────
from ai.decision import AIDecisionLayer

# ── Phase 3 AI ─────────────────────────────────────────────────────────────
from knowledge.knowledge_store import KnowledgeStore
from ai.semantic_matcher import SemanticMatcher
from ai.scoring_engine import ScoringEngine
from ai.memory_engine import MemoryEngine
from ai.discovery_engine import DiscoveryEngine
from ai.routing_engine import RoutingEngine
from ai.goal_planner import GoalPlanner
from ai.optimization_engine import OptimizationEngine

# ── Phase 4 AI ─────────────────────────────────────────────────────────────
from ai.learning_validator import LearningValidator
from ai.reputation_engine import NodeReputationEngine

# ── Phase 5 AI ─────────────────────────────────────────────────────────────
from ai.service_generator import ServiceGeneratorEngine
from ai.gap_detector import GapDetectionEngine
from ai.capability_marketplace import CapabilityMarketplace
from ai.multi_goal_planner import MultiGoalPlanner
from ai.governor import AIGovernanceLayer
from ai.evolution_engine import EvolutionEngine

# ── Services ───────────────────────────────────────────────────────────────
from services.input_service import InputNode
from services.processor_service import ProcessorNode
from services.output_service import OutputNode


class NeuralServiceMesh:
    """
    Phase 3: Autonomous Neural Service Mesh.

    Extends Phase 2 with:
      - SemanticMatcher:    understands node input/output semantics
      - ScoringEngine:      scores every connection from execution history
      - MemoryEngine:       persists successful/failed routes to SQLite
      - DiscoveryEngine:    auto-discovery via node self-announcement
      - RoutingEngine:      multi-factor intelligent route selection
      - GoalPlanner:        goal-driven execution (no explicit path needed)
      - OptimizationEngine: self-improving graph topology

    All Phase 2 APIs are preserved unchanged.
    """

    VERSION = "5.0.0"

    def __init__(self, storage_dir: str = "./data", db_path: str = "./data/mesh.db"):
        # ── Storage ────────────────────────────────────────────────────────
        self.storage = FileStorage(storage_dir)
        self.db = SQLiteStorage(db_path)

        # ── Core ───────────────────────────────────────────────────────────
        self.registry = NodeRegistry(self.storage)
        self.graph = ServiceGraph()
        self.transformer = DataTransformer()

        # ── Phase 2 AI ────────────────────────────────────────────────────
        self.ai = AIDecisionLayer()

        # ── Phase 3 Knowledge Layer (JSON persistent knowledge files) ────
        self.knowledge = KnowledgeStore(knowledge_dir="./knowledge")

        # ── Phase 3 AI modules ────────────────────────────────────────────
        self.semantic = SemanticMatcher()
        self.scoring = ScoringEngine(db_path=db_path)
        self.memory = MemoryEngine(db_path=db_path)
        self.discovery = DiscoveryEngine(db_path=db_path, semantic_matcher=self.semantic)
        self.routing = RoutingEngine(
            graph=self.graph,
            semantic_matcher=self.semantic,
            scoring_engine=self.scoring,
            memory_engine=self.memory,
        )
        self.planner = GoalPlanner(
            semantic_matcher=self.semantic,
            routing_engine=self.routing,
            memory_engine=self.memory,
            registry=self.registry,
        )
        self.optimizer = OptimizationEngine(
            graph=self.graph,
            scoring_engine=self.scoring,
            memory_engine=self.memory,
            semantic_matcher=self.semantic,
        )

        # ── Engine (Phase 2 + Phase 3 hooks) ──────────────────────────────
        self.engine = ExecutionEngine(
            registry=self.registry,
            graph=self.graph,
            storage=self.storage,
            transformer=self.transformer,
            db=self.db,
            ai=self.ai,
        )

        # ── Wire Phase 2 AI to graph
        self.ai.set_graph(self.graph)
        self.ai.set_db(self.db)

        # ── Wire KnowledgeStore into Phase 3 AI modules ───────────────
        self.memory.set_knowledge_store(self.knowledge)
        self.discovery.set_knowledge_store(self.knowledge)
        self.routing.set_knowledge_store(self.knowledge)
        self.optimizer.set_knowledge_store(self.knowledge)

        # ── Phase 4: Learning Validation & Reputation ──────────────────
        self.validator = LearningValidator(
            memory_engine=self.memory,
            scoring_engine=self.scoring,
            knowledge_store=self.knowledge,
        )
        self.reputation = NodeReputationEngine(
            knowledge_store=self.knowledge,
            memory_engine=self.memory,
        )

        # ── Phase 5: Autonomous Service Creation & Evolution ───────────
        self.governance = AIGovernanceLayer(
            graph=self.graph,
            reputation_engine=self.reputation,
            knowledge_store=self.knowledge,
        )
        self.marketplace = CapabilityMarketplace(
            knowledge_store=self.knowledge,
        )
        self.gap_detector = GapDetectionEngine(
            graph=self.graph,
            memory_engine=self.memory,
            semantic_matcher=self.semantic,
            knowledge_store=self.knowledge,
            scoring_engine=self.scoring,
        )
        self.service_generator = ServiceGeneratorEngine(
            knowledge_store=self.knowledge,
            semantic_matcher=self.semantic,
            governance=self.governance,
        )
        self.multi_planner = MultiGoalPlanner(
            capability_marketplace=self.marketplace,
            goal_planner=self.planner,
            memory_engine=self.memory,
            knowledge_store=self.knowledge,
        )
        self.evolution = EvolutionEngine(
            mesh=self,
            gap_detector=self.gap_detector,
            service_generator=self.service_generator,
            governance=self.governance,
            capability_marketplace=self.marketplace,
            multi_goal_planner=self.multi_planner,
            knowledge_store=self.knowledge,
        )

        # ── Install Phase 3 post-run hook ──────────────────────────────────
        self._install_phase3_hook()

        logger.info(f"NeuralServiceMesh v{self.VERSION} ready (Phase 5 — Autonomous Service Creation & Evolution)")

    # ── Phase 3 hook into ExecutionEngine ─────────────────────────────────

    def _install_phase3_hook(self):
        """Monkey-patch ExecutionEngine._persist to also call Phase 3+4 learners."""
        original_persist = self.engine._persist

        def _phase4_persist(result):
            original_persist(result)                    # Phase 2 behaviour
            result_dict = result.to_dict()
            self.scoring.record_run(result_dict)        # Update connection scores
            self.memory.learn_from_run(result_dict)     # Update route memory

            # Phase 4: Update node reputation from execution result
            try:
                success = result_dict.get("status") == "success"
                latency = result_dict.get("total_duration_ms", 0.0)
                for step in result_dict.get("steps", []):
                    nid = step.get("node_id", "")
                    name = step.get("node_name", "")
                    step_ok = step.get("status") == "success"
                    self.reputation.record_execution(nid, name, step_ok, latency)
            except Exception:
                pass

        self.engine._persist = _phase4_persist

    # ── Node Management ────────────────────────────────────────────────────

    def register_node(self, node, connect_to: Optional[str] = None) -> str:
        """Phase 2-compatible register + Phase 3 announcement + Phase 5 marketplace."""
        node_id = self.registry.register(node)
        self.graph.add_node(node_id, node.metadata.to_dict())
        self.db.upsert_node(node.to_dict())

        # Phase 3: announce to discovery layer (populates semantic matcher)
        self.discovery.announce(node)

        # Phase 5: advertise capabilities in marketplace
        try:
            self.marketplace.advertise_from_node(node)
        except Exception:
            pass

        if connect_to:
            self.graph.add_edge(connect_to, node_id)
            self.db.upsert_connection(connect_to, node_id)

        logger.info(f"Registered node '{node.name}' [{node_id[:8]}]")
        return node_id

    # ── Execution (Phase 2 API preserved) ─────────────────────────────────

    def run(self, start_id: str, end_id: str, data: dict, use_ai: bool = True) -> dict:
        """Phase 2: run between two known nodes."""
        return self.engine.run_between(start_id, end_id, data, use_ai=use_ai).to_dict()

    # ── Phase 3: Goal-driven execution ────────────────────────────────────

    def run_goal(self, goal: str, data: dict,
                 preferred_start: Optional[str] = None,
                 preferred_end: Optional[str] = None,
                 max_hops: int = 10) -> dict:
        """
        Phase 3: Execute based on a high-level goal.
        The system discovers the best path automatically.
        """
        plan = self.planner.plan(
            goal,
            preferred_start=preferred_start,
            preferred_end=preferred_end,
            max_hops=max_hops,
        )
        if not plan:
            return {
                "status": "failed",
                "error": f"GoalPlanner could not find a path for goal: '{goal}'",
                "goal": goal,
            }

        plan.status = "running"
        result = self.engine.run_path(plan.path, data)
        result_dict = result.to_dict()

        plan.status = result.status
        plan.result = result_dict
        result_dict["goal_plan"] = plan.to_dict()
        logger.info(f"run_goal '{goal}' → {result.status}")
        return result_dict

    # ── Phase 3: Optimization ──────────────────────────────────────────────

    def optimize(self, auto_apply: bool = False) -> dict:
        """
        Run optimization analysis.
        If auto_apply=True, applies safe actions (new edges, weight updates).
        Destructive actions (prune, remove) are never auto-applied.
        """
        report = self.optimizer.analyze()
        if auto_apply:
            from ai.optimization_engine import OptimizationAction
            safe_types = {
                OptimizationAction.SUGGEST_EDGE,
                OptimizationAction.UPDATE_WEIGHT,
                OptimizationAction.PROMOTE_EDGE,
            }
            safe_report_actions = [a for a in report.actions if a.action_type in safe_types]
            # Temporarily filter
            all_actions = report.actions
            report.actions = safe_report_actions
            applied = self.optimizer.apply_report(report, self)
            report.actions = all_actions
            logger.info(f"Auto-applied {applied} safe optimization actions")
        return report.to_dict()

    # ── Phase 3: Discovery ─────────────────────────────────────────────────

    def discover_connections(self, threshold: float = 0.15) -> list:
        """Return semantically suggested new connections."""
        existing = [
            (src, e.target_id)
            for src, edges in self.graph._adjacency.items()
            for e in edges
        ]
        return self.semantic.suggest_new_connections(existing, threshold)

    def find_nodes_for_goal(self, goal: str, top_k: int = 5) -> list:
        """Find the most relevant nodes for a goal."""
        return self.discovery.find_nodes_for_goal(goal, top_k)

    # ── Status ─────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "version": self.VERSION,
            "phase": 4,
            "nodes": self.registry.count(),
            "graph": self.graph.stats(),
            "storage": self.storage.stats(),
            "db": self.db.db_stats(),
            # Phase 2 AI
            "ai_phase2": {
                "enabled": True,
                "mode": "rules+heuristics",
                "paths_tracked": len(self.ai._path_stats),
            },
            # Phase 3 AI
            "ai_phase3": {
                "semantic_profiles": self.semantic.profile_count(),
                "scoring": self.scoring.summary(),
                "memory": self.memory.summary(),
                "discovery": self.discovery.summary(),
                "optimization_runs": self.optimizer._run_count,
            },
            # Phase 3 Knowledge Layer
            "knowledge_layer": self.knowledge.summary(),
            # Phase 4 Learning & Reputation
            "ai_phase4": {
                "learning_metrics": self.validator.compute_metrics().to_dict(),
                "reputation": self.reputation.summary(),
            },
            # Phase 5 Autonomous Evolution
            "ai_phase5": {
                "governance": self.governance.summary(),
                "marketplace": self.marketplace.summary(),
                "gap_detector": self.gap_detector.summary(),
                "service_generator": self.service_generator.summary(),
                "evolution": self.evolution.summary(),
            },
        }

    # ── Phase 4: Public API methods ────────────────────────────────────────

    def get_ai_status(self) -> dict:
        """GET /ai/status — Full AI system status with learning proof."""
        proof = self.validator.prove_learning()
        return {
            "version": self.VERSION,
            "phase": 4,
            "system_status": self.status(),
            "learning_proof": proof,
        }

    def get_ai_routes(self) -> dict:
        """GET /ai/routes — All known routes ranked by memory score."""
        routes = self.memory.all_routes()
        ranked = sorted(routes, key=lambda r: r.get("memory_score", 0), reverse=True)
        return {
            "routes": ranked,
            "count": len(ranked),
            "summary": self.memory.summary(),
        }

    def get_ai_reputation(self) -> dict:
        """GET /ai/reputation — All node reputation scores."""
        self.reputation.update_from_memory()
        return {
            "nodes": self.reputation.all_reputations(),
            "summary": self.reputation.summary(),
        }

    def get_ai_knowledge(self) -> dict:
        """GET /ai/knowledge — Full knowledge layer snapshot."""
        return {
            "route_memory": self.knowledge.read_route_memory(),
            "graph_metrics": self.knowledge.read_graph_metrics(),
            "node_profiles_summary": {
                "total": len(self.knowledge.list_active_node_profiles()),
                "active_nodes": [
                    {k: v for k, v in p.items() if k in ("node_id", "name", "discovery_score")}
                    for p in self.knowledge.list_active_node_profiles()
                ],
            },
            "learning_curve": self.validator.get_learning_curve(),
        }

    # ── Phase 5: Public API methods ────────────────────────────────────────

    def run_multi_goal(self, goal: str, data: dict) -> dict:
        """Phase 5: Execute a complex multi-goal plan."""
        plan = self.multi_planner.plan(goal)
        if plan.status == "failed":
            return {
                "status": "failed",
                "error": f"MultiGoalPlanner could not resolve goal: '{goal}'",
                "plan": plan.to_dict(),
            }
        result = self.multi_planner.execute_plan(plan, self.engine, data)
        return result

    def evolve(self, cycles: int = 1, auto_register: bool = True) -> dict:
        """Phase 5: Run evolution cycle(s) — discover gaps and generate services."""
        return self.evolution.evolve(cycles=cycles, auto_register=auto_register, verbose=True)

    def scan_gaps(self) -> dict:
        """Phase 5: Scan for capability gaps without triggering generation."""
        gaps = self.gap_detector.scan()
        return {
            "gaps": [g.to_dict() for g in gaps],
            "count": len(gaps),
            "summary": self.gap_detector.summary(),
        }

    def get_marketplace(self) -> dict:
        """Phase 5: Get capability marketplace snapshot."""
        return {
            "summary": self.marketplace.summary(),
            "capabilities": self.marketplace.list_capabilities(),
            "advertisements": self.marketplace.all_advertisements(),
        }

    def get_governance(self) -> dict:
        """Phase 5: Get governance layer status and audit log."""
        return {
            "summary": self.governance.summary(),
            "recent_audit": self.governance.audit_log(limit=20),
        }

    def get_generated_services(self, status: Optional[str] = None) -> dict:
        """Phase 5: List all AI-generated service specs."""
        return {
            "services": self.service_generator.list_generated(status_filter=status),
            "summary": self.service_generator.summary(),
        }


# ── Demo ───────────────────────────────────────────────────────────────────

def demo():
    import json
    print("\n" + "="*65)
    print("  Neural Service Mesh  —  Phase 3 Demo (Autonomous Intelligence)")
    print("="*65 + "\n")

    mesh = NeuralServiceMesh()

    # Register pipeline nodes
    inp  = mesh.register_node(InputNode("TextInput"))
    proc = mesh.register_node(ProcessorNode("TextProcessor"), connect_to=inp)
    out  = mesh.register_node(OutputNode("TextOutput", output_format="summary"), connect_to=proc)

    sample_text = (
        "Neural networks are inspired by the human brain. "
        "They consist of interconnected nodes that process information. "
        "Deep learning has achieved remarkable results in vision and language tasks."
    )

    # ── Run 1: Phase 2 API (backward compat) ──────────────────────────────
    print("[ Run 1 — Phase 2 API (backward compatible) ]\n")
    r1 = mesh.run(inp, out, {"text": sample_text, "source": "demo"}, use_ai=True)
    print(f"  Status       : {r1['status']}")
    print(f"  Duration     : {r1['total_duration_ms']} ms")
    print(f"  AI Suggested : {r1['ai_suggested']}\n")

    # ── Run 2: Phase 3 goal-based execution ───────────────────────────────
    print("[ Run 2 — Phase 3 Goal-Based Execution ]\n")
    r2 = mesh.run_goal(
        goal="Process and summarize text content",
        data={"text": sample_text, "source": "phase3_demo"},
    )
    print(f"  Status    : {r2['status']}")
    print(f"  Duration  : {r2.get('total_duration_ms')} ms")
    plan = r2.get("goal_plan", {})
    print(f"  Goal      : {plan.get('goal')}")
    print(f"  Confidence: {plan.get('confidence')}")
    print(f"  Path      : {' → '.join(n[:8] for n in plan.get('path', []))}")
    print(f"  Reasoning : {plan.get('reasoning', [])[-1]}\n")

    # ── Discover new connections ───────────────────────────────────────────
    print("[ Phase 3 — Connection Discovery ]\n")
    suggestions = mesh.discover_connections(threshold=0.10)
    if suggestions:
        for s in suggestions[:3]:
            print(f"  Suggested: {s['source_id'][:8]} → {s['target_id'][:8]}  score={s['semantic_score']:.3f}")
    else:
        print("  No new connections suggested (nodes may already be connected)")

    # ── Node discovery for goal ────────────────────────────────────────────
    print("\n[ Phase 3 — Nodes for Goal ]\n")
    matches = mesh.find_nodes_for_goal("analyze and process text", top_k=3)
    for m in matches:
        print(f"  {m['name']} — score={m['capability_score']:.3f}")

    # ── Optimization ──────────────────────────────────────────────────────
    print("\n[ Phase 3 — Optimization Analysis ]\n")
    opt = mesh.optimize(auto_apply=False)
    print(f"  Total recommendations: {opt['total_actions']}")
    print(f"  Action types: {opt['summary'].get('action_counts', {})}")

    # ── Scoring snapshot ───────────────────────────────────────────────────
    print("\n[ Phase 3 — Connection Scores ]\n")
    scores = mesh.scoring.list_scores()
    for s in scores[:5]:
        print(f"  {s['source_id'][:8]} → {s['target_id'][:8]}  "
              f"score={s['connection_score']}  sr={s['success_rate']:.2%}  "
              f"runs={s['total_runs']}")

    # ── Memory ────────────────────────────────────────────────────────────
    print("\n[ Phase 3 — Route Memory ]\n")
    routes = mesh.memory.all_routes()
    for rm in routes[:3]:
        print(f"  {rm['path_key']}  health={rm['health']}  "
              f"sr={rm['success_rate']:.2%}  runs={rm['runs']}")

    # ── Full status ────────────────────────────────────────────────────────
    print("\n[ System Status ]\n")
    print(json.dumps(mesh.status(), indent=2))


# ── Entry point ────────────────────────────────────────────────────────────

def simulate(rounds: int = 20, delay: float = 0.1):
    """Phase 4 simulation mode — proves the AI learns from experience."""
    from ai.simulation_engine import SimulationEngine
    mesh = NeuralServiceMesh()
    sim = SimulationEngine(mesh, validator=mesh.validator)
    results = sim.run_simulation(
        rounds=rounds,
        executions_per_round=5,
        delay_between_rounds=delay,
        verbose=True,
    )
    # Save simulation report
    import json, os
    os.makedirs("./data", exist_ok=True)
    report_path = "./data/simulation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Simulation report saved to: {report_path}")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Neural Service Mesh v5")
    p.add_argument("--mode", choices=["demo", "api", "simulate", "evolve"], default="demo",
                   help="demo: example pipeline | api: Flask server | simulate: Phase 4 learning sim | evolve: Phase 5 evolution")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--rounds", type=int, default=20,
                   help="Number of simulation rounds (simulate mode)")
    p.add_argument("--delay", type=float, default=0.1,
                   help="Delay between rounds in seconds (simulate mode)")
    p.add_argument("--cycles", type=int, default=3,
                   help="Number of evolution cycles (evolve mode)")
    args = p.parse_args()

    if args.mode == "demo":
        demo()
    elif args.mode == "simulate":
        simulate(rounds=args.rounds, delay=args.delay)
    elif args.mode == "evolve":
        import json
        print("\n" + "="*65)
        print("  Neural Service Mesh  —  Phase 5 (Autonomous Evolution)")
        print("="*65 + "\n")
        mesh = NeuralServiceMesh()
        # Register some nodes first
        from services.input_service import InputNode
        from services.processor_service import ProcessorNode
        from services.output_service import OutputNode
        inp = mesh.register_node(InputNode("TextInput"))
        proc = mesh.register_node(ProcessorNode("TextProcessor"), connect_to=inp)
        out = mesh.register_node(OutputNode("TextOutput", output_format="summary"), connect_to=proc)
        # Run evolution
        result = mesh.evolve(cycles=args.cycles, auto_register=True)
        print(json.dumps(result, indent=2))
    else:
        from api.app import run_api
        mesh = NeuralServiceMesh()
        run_api(mesh, host=args.host, port=args.port, debug=args.debug)
