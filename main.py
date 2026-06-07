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

# ── Phase 6 AI ─────────────────────────────────────────────────────────────
from ai.agent_factory import AgentFactory
from ai.swarm_coordinator import SwarmCoordinator
from ai.self_optimizer import SelfOptimizer
from ai.simulation_lab import SimulationLab
from ai.meta_reasoner import MetaReasoner
from ai.economic_engine import EconomicEngine
from ai.system_dna import SystemDNA
from ai.phase6_validator import Phase6Validator

# ── Phase 7 AI ─────────────────────────────────────────────────────────────
from sensors.sensor_hub import SensorHub
from sensors.api_sensor import APISensor
from sensors.filesystem_sensor import FilesystemSensor
from sensors.log_sensor import LogSensor
from sensors.webhook_sensor import WebhookSensor
from world_model.environment_model import EnvironmentModel
from ai.self_awareness import SelfAwarenessEngine
from ai.code_generator import CodeGenerationEngine
from ai.sandbox_lab import SandboxTestingLab
from ai.governance_p7 import P7GovernanceLayer
from ai.objectives import ObjectivesEngine
from ai.evolution_pipeline import EvolutionPipeline

# ── Phase 8 AI — Real Neural Weights ───────────────────────────────────────
from ai.neural_weights import NeuralWeightLayer, extract_routing_weights

# ── Phase 9 AI — Rich Data + Dynamic Growth + Deep Network ─────────────────
try:
    from ai.rich_data_collector import RichDataCollector
    from ai.dynamic_weight_layer import DynamicWeightLayer, extract_routing_weights_dynamic
    from ai.deep_routing_network import DeepRoutingNetwork, extract_deep_routing_weights
    _PHASE9_AVAILABLE = True
except ImportError as _p9_err:
    _PHASE9_AVAILABLE = False
    import logging as _log
    _log.getLogger(__name__).warning(f"Phase 9 modules not fully available: {_p9_err}")

# ── Phase 10 — Continuous Signal Stream ────────────────────────────────────
try:
    from ai.signal_stream import SignalBus, ReplayBuffer, Experience
    _PHASE10_AVAILABLE = True
except ImportError as _p10_err:
    _PHASE10_AVAILABLE = False
    import logging as _log
    _log.getLogger(__name__).warning(f"Phase 10 not available: {_p10_err}")

# ── Phase 11 — Episodic Memory ─────────────────────────────────────────────
try:
    from ai.episodic_memory import EpisodicMemoryEngine, Episode
    _PHASE11_AVAILABLE = True
except ImportError as _p11_err:
    _PHASE11_AVAILABLE = False
    import logging as _log
    _log.getLogger(__name__).warning(f"Phase 11 not available: {_p11_err}")

# ── Phase 12 — Deep Self-Awareness ─────────────────────────────────────────
try:
    from ai.self_awareness_deep import DeepSelfAwareness
    _PHASE12_AVAILABLE = True
    # ── Phase 13 — Structural Self-Redesign
try:
    from ai.structural_redesign import StructuralEvolutionEngine
    _PHASE13_AVAILABLE = True
except ImportError as _p13_err:
    _PHASE13_AVAILABLE = False

# ── Phase 14 — Complete Digital Being
try:
    from ai.digital_being import DigitalBeingCore
    _PHASE14_AVAILABLE = True
except ImportError as _p14_err:
    _PHASE14_AVAILABLE = False
except ImportError as _p12_err:
    _PHASE12_AVAILABLE = False
    import logging as _log
    _log.getLogger(__name__).warning(f"Phase 12 not available: {_p12_err}")

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

    VERSION = "12.0.0"

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

        # ── Phase 6: Multi-Agent Self-Improving Platform ────────────────
        self.agent_factory = AgentFactory(knowledge_store=self.knowledge)
        self.swarm = SwarmCoordinator(
            factory=self.agent_factory,
            max_agents=20,
            knowledge_store=self.knowledge,
        )
        self.self_optimizer = SelfOptimizer(
            registry=self.registry,
            graph=self.graph,
            scoring_engine=self.scoring,
            memory_engine=self.memory,
            knowledge_store=self.knowledge,
            agent_factory=self.agent_factory,
        )
        self.simulation_lab = SimulationLab(
            registry=self.registry,
            graph=self.graph,
            routing_engine=self.routing,
            scoring_engine=self.scoring,
            memory_engine=self.memory,
            knowledge_store=self.knowledge,
        )
        self.meta_reasoner = MetaReasoner(
            memory_engine=self.memory,
            scoring_engine=self.scoring,
            knowledge_store=self.knowledge,
            evolution_engine=self.evolution,
            governance_layer=self.governance,
        )
        self.economic_engine = EconomicEngine(
            scoring_engine=self.scoring,
            reputation_engine=self.reputation,
            knowledge_store=self.knowledge,
            registry=self.registry,
        )
        self.system_dna = SystemDNA(knowledge_store=self.knowledge)

        # ── Phase 7: Autonomous Evolution Platform ─────────────────────────
        # 1. Sensory Layer
        self.sensor_hub = SensorHub(interval_s=30.0)
        self.api_sensor = APISensor(name="MeshAPISensor")
        self.filesystem_sensor = FilesystemSensor(
            name="CodeSensor",
            config={"watch_paths": ["./ai", "./services", "./sensors"], "extensions": [".py"]},
        )
        self.log_sensor = LogSensor(name="MeshLogSensor", config={"log_paths": ["./logs"]})
        self.webhook_sensor = WebhookSensor(name="MeshWebhookSensor")
        self.sensor_hub.register(self.api_sensor)
        self.sensor_hub.register(self.filesystem_sensor)
        self.sensor_hub.register(self.log_sensor)
        self.sensor_hub.register(self.webhook_sensor)

        # 2. Environment / World Model
        self.env_model = EnvironmentModel(model_dir="./world_model")
        self.sensor_hub.on_event(lambda e: self.env_model.ingest_sensor_event(e.to_dict()))

        # 3. Self-Awareness Engine
        self.self_awareness = SelfAwarenessEngine(
            registry=self.registry,
            graph=self.graph,
            scoring_engine=self.scoring,
            memory_engine=self.memory,
            agent_factory=self.agent_factory,
            knowledge_store=self.knowledge,
            environment_model=self.env_model,
        )

        # 4. Code Generation Engine
        self.code_generator = CodeGenerationEngine(
            output_dir="./services",
            knowledge_store=self.knowledge,
            governance=self.governance,
        )

        # 5. Sandbox Testing Lab
        self.sandbox_lab_p7 = SandboxTestingLab(sandbox_dir="./sandbox")

        # 6. Phase 7 Governance
        self.governance_p7 = P7GovernanceLayer(
            min_sandbox_score=75.0,
            knowledge_store=self.knowledge,
            base_governance=self.governance,
        )

        # 7. Strategic Objectives Engine
        self.objectives = ObjectivesEngine(knowledge_store=self.knowledge)

        # Wire self-awareness objectives
        self.self_awareness._objectives = self.objectives

        # 8. Evolution Pipeline (Phase 7)
        self.evolution_pipeline = EvolutionPipeline(
            mesh=self,
            sensor_hub=self.sensor_hub,
            environment_model=self.env_model,
            self_awareness=self.self_awareness,
            code_generator=self.code_generator,
            sandbox_lab=self.sandbox_lab_p7,
            governance_p7=self.governance_p7,
            objectives_engine=self.objectives,
            gap_detector=self.gap_detector,
            knowledge_store=self.knowledge,
            services_dir="./services",
        )

        logger.info(f"NeuralServiceMesh v{self.VERSION} ready (Phase 7 — Autonomous Evolution Platform)")

        # ── Phase 8: Real Neural Weights ───────────────────────────────────
        # The RoutingEngine already bootstraps its NeuralWeightLayer internally.
        # We expose a top-level reference here for API access and status reporting.
        self.neural_layer: Optional[NeuralWeightLayer] = self.routing.get_neural_layer()
        if self.neural_layer is not None:
            logger.info(
                f"NeuralServiceMesh Phase 8: neural weight layer active  "
                f"shape={self.neural_layer.weights.shape}  "
                f"steps={self.neural_layer._train_steps}"
            )
        else:
            logger.warning(
                "NeuralServiceMesh Phase 8: NeuralWeightLayer unavailable — "
                "install numpy to enable real neural weights."
            )

        logger.info(f"NeuralServiceMesh v{self.VERSION} fully ready (Phase 8 — Real Neural Weights)")

        # ── Phase 9: Rich Data + Dynamic Growth + Deep Network ─────────────
        # Axis-1: RichDataCollector
        self.rich_data_collector = getattr(self.routing, '_rich_data', None)
        if self.rich_data_collector is not None and hasattr(self.routing, '_rich_data'):
            # Wire the environment model so external signals flow in
            try:
                self.rich_data_collector.set_env_model(self.env_model)
            except Exception:
                pass

        # Axis-2: DynamicWeightLayer
        self.dynamic_layer = getattr(self.routing, '_dynamic_layer', None)

        # Axis-3: DeepRoutingNetwork
        self.deep_network = getattr(self.routing, '_deep_network', None)

        _p9_components = sum([
            self.rich_data_collector is not None,
            self.dynamic_layer is not None,
            self.deep_network is not None,
        ])
        if _p9_components > 0:
            logger.info(
                f"NeuralServiceMesh Phase 9: {_p9_components}/3 axes active — "
                f"RichData={'✓' if self.rich_data_collector else '✗'}  "
                f"DynamicLayer={'✓' if self.dynamic_layer else '✗'}  "
                f"DeepNetwork={'✓' if self.deep_network else '✗'}"
            )

        # ── Phase 10: Continuous Signal Stream ─────────────────────────────
        self.signal_bus: Optional["SignalBus"] = None
        if _PHASE10_AVAILABLE:
            self.signal_bus = SignalBus(
                deep_network=self.deep_network,
                dynamic_layer=self.dynamic_layer,
                neural_layer=getattr(self.routing, '_neural_layer', None),
                rich_data_collector=self.rich_data_collector,
            )
            logger.info("NeuralServiceMesh Phase 10: SignalBus ready — "
                        "SelfStimulator + CuriosityEngine + DreamConsolidator active")

        # ── Phase 11: Episodic Memory ───────────────────────────────────────
        self.episodic_memory: Optional["EpisodicMemoryEngine"] = None
        if _PHASE11_AVAILABLE:
            self.episodic_memory = EpisodicMemoryEngine()
            self.episodic_memory.start()
            logger.info("NeuralServiceMesh Phase 11: EpisodicMemoryEngine active — "
                        "3-tier memory online (working + episodic + semantic)")

        # ── Phase 12: Deep Self-Awareness ───────────────────────────────────
        self.self_awareness: Optional["DeepSelfAwareness"] = None
        if _PHASE12_AVAILABLE:
            self.self_awareness = DeepSelfAwareness()
            logger.info("NeuralServiceMesh Phase 12: DeepSelfAwareness active — "
                        "ConfidenceEstimator + WeaknessDetector + MetaCognition online")

        logger.info(
            f"NeuralServiceMesh v{self.VERSION} fully ready — "
            f"Phases 1-12 active | "
            f"SignalBus={'✓' if self.signal_bus else '✗'}  "
            f"Memory={'✓' if self.episodic_memory else '✗'}  "
            f"SelfAware={'✓' if self.self_awareness else '✗'}"
        )

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
            # Phase 6 Multi-Agent Self-Improving Platform
            "ai_phase6": {
                "agent_factory": self.agent_factory.summary(),
                "swarm": self.swarm.summary(),
                "self_optimizer": self.self_optimizer.summary(),
                "simulation_lab": self.simulation_lab.summary(),
                "meta_reasoner": self.meta_reasoner.summary(),
                "economic_engine": self.economic_engine.summary(),
                "system_dna": self.system_dna.summary(),
            },
            # Phase 7 Autonomous Evolution Platform
            "ai_phase7": {
                "sensor_hub": self.sensor_hub.summary(),
                "environment_model": self.env_model.summary(),
                "self_awareness": self.self_awareness.summary(),
                "code_generator": self.code_generator.summary(),
                "sandbox_lab": self.sandbox_lab_p7.summary(),
                "governance_p7": self.governance_p7.summary(),
                "objectives": self.objectives.summary(),
                "evolution_pipeline": self.evolution_pipeline.summary(),
            },
            # Phase 8 Real Neural Weights
            "ai_phase8": self.routing.neural_weights_summary(),
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

    # ── Phase 6: Public API methods ────────────────────────────────────────

    def spawn_agent(self, role: str, config: Optional[dict] = None) -> dict:
        """Phase 6: Spawn a new autonomous agent."""
        agent = self.agent_factory.spawn(role, config)
        return agent.to_dict()

    def swarm_execute(self, goal: str, data: dict, custom_tasks: Optional[list] = None) -> dict:
        """Phase 6: Execute a goal using the agent swarm."""
        result = self.swarm.execute(goal, data, custom_tasks)
        return result.to_dict()

    def self_optimize(self) -> dict:
        """Phase 6: Run one self-optimization cycle."""
        report = self.self_optimizer.run_cycle()
        return report.to_dict()

    def simulate_plans(self, goal: str, data: dict, n_plans: int = 100) -> dict:
        """Phase 6: Run simulation lab to find the best execution plan."""
        report = self.simulation_lab.run(goal=goal, data=data, n_plans=n_plans)
        return report.to_dict()

    def meta_reflect(self) -> dict:
        """Phase 6: Run meta-reasoning reflection over recent history."""
        insights = self.meta_reasoner.reflect()
        return {
            "insights": [i.to_dict() for i in insights],
            "summary": self.meta_reasoner.summary(),
        }

    def meta_ask(self, question: str) -> dict:
        """Phase 6: Ask the meta-reasoner a question about system decisions."""
        return self.meta_reasoner.ask(question)

    def economic_leaderboard(self, top_n: int = 10) -> dict:
        """Phase 6: Get node economic leaderboard."""
        return {
            "leaderboard": self.economic_engine.leaderboard(top_n),
            "summary": self.economic_engine.summary(),
        }

    def dna_snapshot(self, notes: str = "") -> dict:
        """Phase 6: Capture a DNA snapshot of the current system state."""
        snap = self.system_dna.snapshot(
            registry=self.registry,
            scoring_engine=self.scoring,
            memory_engine=self.memory,
            notes=notes,
        )
        return snap.to_dict()

    def dna_diff(self, snapshot_id_a: str, snapshot_id_b: str) -> dict:
        """Phase 6: Diff two DNA snapshots."""
        return self.system_dna.diff(snapshot_id_a, snapshot_id_b)

    def dna_rollback(self, snapshot_id: str) -> dict:
        """Phase 6: Activate (roll back to) a prior DNA snapshot."""
        success = self.system_dna.apply(snapshot_id)
        return {"success": success, "snapshot_id": snapshot_id}

    def get_agent_factory(self) -> dict:
        """Phase 6: Get agent factory summary and active agents."""
        return {
            "summary": self.agent_factory.summary(),
            "agents": self.agent_factory.all_agents(),
        }

    def get_swarm_history(self, limit: int = 10) -> dict:
        """Phase 6: Get recent swarm execution history."""
        return {
            "history": self.swarm.history(limit),
            "summary": self.swarm.summary(),
        }

    # ── Phase 7: Public API methods ───────────────────────────────────────

    def evolve7(self, cycles: int = 1, verbose: bool = True) -> dict:
        """Phase 7: Run evolution pipeline cycle(s) — full Observe→Deploy loop."""
        results = self.evolution_pipeline.run_cycles(n=cycles, verbose=verbose)
        return {"cycles": results, "total": len(results)}

    def evolve7_once(self, verbose: bool = True) -> dict:
        """Phase 7: Run a single evolution pipeline cycle."""
        cycle = self.evolution_pipeline.run_cycle(verbose=verbose)
        return cycle.to_dict()

    def introspect(self) -> dict:
        """Phase 7: Run self-awareness introspection."""
        report = self.self_awareness.introspect()
        return report.to_dict()

    def sensor_status(self) -> dict:
        """Phase 7: Get sensor hub summary and recent events."""
        return {
            "summary": self.sensor_hub.summary(),
            "recent_events": self.sensor_hub.recent_events(limit=20),
        }

    def world_model(self) -> dict:
        """Phase 7: Get the current environment/world model state."""
        return self.env_model.get_state()

    def push_sensor_event(self, event_type: str, payload: Optional[dict] = None,
                          severity: str = "info"):
        """Phase 7: Push a manual event to the webhook sensor."""
        self.webhook_sensor.push_event(event_type, payload=payload, severity=severity)

    def get_objectives(self) -> dict:
        """Phase 7: Get all strategic objectives and their current progress."""
        return {
            "objectives": self.objectives.get_all_objectives(),
            "summary": self.objectives.summary(),
            "recommendations": self.objectives.get_recommendations(),
        }

    def measure_objectives(self) -> dict:
        """Phase 7: Measure current metrics against strategic objectives."""
        measurements = self.objectives.measure_from_mesh(self)
        return {
            "measurements": measurements,
            "objectives": self.objectives.get_all_objectives(),
            "recommendations": self.objectives.get_recommendations(),
        }

    def generate_module(self, gap_description: str, source_name: str = "",
                        target_name: str = "") -> dict:
        """Phase 7: Manually trigger code generation for a described gap."""
        gap = {
            "missing_service": gap_description,
            "gap_type": "manual",
            "source_node": {"name": source_name or "Source"},
            "target_node": {"name": target_name or "Target"},
            "confidence": 0.9,
        }
        module = self.code_generator.generate_from_gap(gap)
        # Write to sandbox
        self.code_generator.write_to_file(module, subdir="generated")
        # Test
        test_result = self.sandbox_lab_p7.test_module(module)
        # Approve
        decision = self.governance_p7.review(module, test_result.to_dict())
        return {
            "module": module.to_dict(),
            "test_result": test_result.to_dict(),
            "decision": decision.to_dict(),
        }

    def list_generated_modules(self, status: Optional[str] = None) -> dict:
        """Phase 7: List all auto-generated modules."""
        return {
            "modules": self.code_generator.list_generated(status_filter=status),
            "summary": self.code_generator.summary(),
        }

    def start_sensors(self, interval_s: float = 30.0):
        """Phase 7: Start the background sensor polling loop."""
        self.sensor_hub.start(interval_s=interval_s)
        return {"status": "started", "interval_s": interval_s}

    def stop_sensors(self):
        """Phase 7: Stop the background sensor polling loop."""
        self.sensor_hub.stop()
        return {"status": "stopped"}

    def get_evolution_history(self, limit: int = 10) -> dict:
        """Phase 7: Get evolution pipeline cycle history."""
        return {
            "history": self.evolution_pipeline.get_history(limit),
            "summary": self.evolution_pipeline.summary(),
        }

    # ── Phase 8: Real Neural Weights ──────────────────────────────────────

    def get_neural_weights(self) -> dict:
        """
        Phase 8: Return the full neural weight layer summary.

        Includes the 10×7 weight matrix, training statistics, and the
        current routing scalars derived from the first matrix row.
        """
        summary = self.routing.neural_weights_summary()
        if self.neural_layer is not None:
            summary["weights_matrix"] = self.neural_layer.get_weights_list()
        return summary

    def train_neural_weights(
        self,
        input_vector: list,
        target: float,
    ) -> dict:
        """
        Phase 8: Manually submit one training step to the NeuralWeightLayer.

        Parameters
        ----------
        input_vector : list[float]  length 7
            Feature vector (semantic, score, memory, topology, avg,
            sem×score, mem×topo — all normalised to [0, 1]).
        target : float
            Desired output scalar in [0, 1].

        Returns
        -------
        dict with keys: loss, train_steps, routing_scalars
        """
        if self.neural_layer is None:
            return {"error": "NeuralWeightLayer not available — install numpy"}
        loss = self.neural_layer.train_step(input_vector, target)
        self.routing._sync_weights_from_layer()
        return {
            "loss":            round(loss, 8),
            "train_steps":     self.neural_layer._train_steps,
            "routing_scalars": {
                "W_SEMANTIC": self.routing.W_SEMANTIC,
                "W_SCORE":    self.routing.W_SCORE,
                "W_MEMORY":   self.routing.W_MEMORY,
                "W_TOPOLOGY": self.routing.W_TOPOLOGY,
            },
        }

    def save_neural_weights(self, path: Optional[str] = None) -> dict:
        """
        Phase 8: Persist the current neural weight matrix to disk.

        Parameters
        ----------
        path : str, optional
            Destination .npy path.  Defaults to
            ``models/classifiers/routing_weights.npy``.
        """
        if self.neural_layer is None:
            return {"error": "NeuralWeightLayer not available"}
        dest = path or self.routing._WEIGHTS_PATH
        saved = self.neural_layer.save(dest)
        return {"saved": True, "path": saved}

    def load_neural_weights(self, path: str) -> dict:
        """
        Phase 8: Load a neural weight matrix from a .npy file and sync
        the routing scalars.
        """
        if self.neural_layer is None:
            return {"error": "NeuralWeightLayer not available"}
        self.neural_layer.load(path)
        self.routing._sync_weights_from_layer()
        return {
            "loaded": True,
            "path":   path,
            "routing_scalars": {
                "W_SEMANTIC": self.routing.W_SEMANTIC,
                "W_SCORE":    self.routing.W_SCORE,
                "W_MEMORY":   self.routing.W_MEMORY,
                "W_TOPOLOGY": self.routing.W_TOPOLOGY,
            },
        }

    # ── Phase 9: Rich Data + Dynamic Growth + Deep Network ────────────────

    def get_phase9_status(self) -> dict:
        """
        Phase 9: Return comprehensive status of all three Phase 9 axes.

        Returns
        -------
        dict with keys:
          axis1_rich_data    — RichDataCollector summary
          axis2_dynamic_layer — DynamicWeightLayer summary (shape, growth events)
          axis3_deep_network  — DeepRoutingNetwork summary (layers, loss)
          routing_scalars    — current W_SEMANTIC / W_SCORE / W_MEMORY / W_TOPOLOGY
        """
        return self.routing.neural_weights_summary()

    def get_deep_network_summary(self) -> dict:
        """
        Phase 9 Axis-3: Return deep neural network architecture and training state.
        """
        if self.deep_network is None:
            return {"error": "DeepRoutingNetwork not available"}
        return self.deep_network.summary()

    def get_dynamic_layer_summary(self) -> dict:
        """
        Phase 9 Axis-2: Return DynamicWeightLayer state including growth history.
        """
        if self.dynamic_layer is None:
            return {"error": "DynamicWeightLayer not available"}
        return self.dynamic_layer.summary()

    def get_rich_data_summary(self) -> dict:
        """
        Phase 9 Axis-1: Return RichDataCollector summary (7 data sources).
        """
        if self.rich_data_collector is None:
            return {"error": "RichDataCollector not available"}
        return self.rich_data_collector.summary()

    def train_deep_network(self, input_vector: list, target: float) -> dict:
        """
        Phase 9 Axis-3: Manually submit one training step to the DeepRoutingNetwork.

        Parameters
        ----------
        input_vector : list[float]  length 7
        target : float  in [0, 1]

        Returns
        -------
        dict with keys: loss, train_steps, routing_scalars
        """
        if self.deep_network is None:
            return {"error": "DeepRoutingNetwork not available"}
        loss = self.deep_network.train_step(input_vector, target)
        self.routing._sync_weights_from_deep_network()
        return {
            "loss": round(loss, 8),
            "train_steps": self.deep_network._train_steps,
            "architecture": self.deep_network.architecture_str(),
            "routing_scalars": {
                "W_SEMANTIC": self.routing.W_SEMANTIC,
                "W_SCORE":    self.routing.W_SCORE,
                "W_MEMORY":   self.routing.W_MEMORY,
                "W_TOPOLOGY": self.routing.W_TOPOLOGY,
            },
        }

    def save_phase9_models(self) -> dict:
        """
        Phase 9: Persist all three Phase 9 models to disk.
        """
        results = {}
        if self.dynamic_layer is not None:
            try:
                path = self.dynamic_layer.save(self.routing._DYNAMIC_WEIGHTS_PATH)
                results["dynamic_layer"] = {"saved": True, "path": path}
            except Exception as e:
                results["dynamic_layer"] = {"saved": False, "error": str(e)}
        else:
            results["dynamic_layer"] = {"saved": False, "error": "not available"}

        if self.deep_network is not None:
            try:
                path = self.deep_network.save(self.routing._DEEP_NETWORK_DIR)
                results["deep_network"] = {"saved": True, "path": path}
            except Exception as e:
                results["deep_network"] = {"saved": False, "error": str(e)}
        else:
            results["deep_network"] = {"saved": False, "error": "not available"}

        return results

    # ── Phase 10: Signal Stream API ────────────────────────────────────────

    def start_signal_stream(self, interval_s: float = 2.0) -> dict:
        """
        Phase 10: Start the continuous signal stream.
        The system begins generating signals autonomously — 24/7.
        """
        if self.signal_bus is None:
            return {"error": "SignalBus (Phase 10) not available"}
        if self.signal_bus.is_running:
            return {"status": "already_running", "stats": self.signal_bus.get_stats()}
        self.signal_bus.start(interval_s=interval_s)
        return {
            "status": "started",
            "interval_s": interval_s,
            "message": "Signal stream active — SelfStimulator + Curiosity + Dream mode online",
        }

    def stop_signal_stream(self) -> dict:
        """Phase 10: Stop the signal stream."""
        if self.signal_bus is None:
            return {"error": "SignalBus not available"}
        self.signal_bus.stop()
        return {"status": "stopped", "final_stats": self.signal_bus.get_stats()}

    def get_signal_stream_status(self) -> dict:
        """Phase 10: Get full signal stream statistics."""
        if self.signal_bus is None:
            return {"enabled": False}
        return {"enabled": True, **self.signal_bus.get_stats()}

    def push_real_signal(
        self,
        feature_vec: list,
        target: float,
        reward: float = 0.0,
        context: Optional[dict] = None,
    ) -> dict:
        """
        Phase 10: Inject a real experience into the signal stream.
        Called automatically by RoutingEngine; can also be called manually.
        """
        if self.signal_bus is None:
            return {"error": "SignalBus not available"}
        self.signal_bus.push_real(feature_vec, target, reward, context)

        # Also record in episodic memory
        if self.episodic_memory is not None:
            self.episodic_memory.record(
                feature_vec=feature_vec,
                target=target,
                source="real",
                reward=reward,
                context=context,
            )
        return {"status": "recorded", "buffer_size": self.signal_bus.replay_buffer.size}

    # ── Phase 11: Episodic Memory API ─────────────────────────────────────

    def get_memory_status(self) -> dict:
        """Phase 11: Full episodic memory status (3 tiers)."""
        if self.episodic_memory is None:
            return {"enabled": False}
        return {"enabled": True, **self.episodic_memory.summary()}

    def recall_similar(self, feature_vec: list, top_k: int = 5) -> dict:
        """
        Phase 11: Recall similar past experiences from memory.
        The system searches its episodic store for related events.
        """
        if self.episodic_memory is None:
            return {"error": "EpisodicMemoryEngine not available"}
        episodes = self.episodic_memory.recall_similar(feature_vec, top_k=top_k)
        return {
            "query_vec":   feature_vec,
            "recalled":    [e.to_dict() for e in episodes],
            "count":       len(episodes),
        }

    def predict_from_memory(self, feature_vec: list) -> dict:
        """
        Phase 11: Ask the memory system to predict an outcome.
        Uses semantic rules + episodic similarity.
        """
        if self.episodic_memory is None:
            return {"error": "EpisodicMemoryEngine not available"}
        prediction = self.episodic_memory.predict_from_memory(feature_vec)
        return {
            "feature_vec": feature_vec,
            "prediction":  round(prediction, 4) if prediction is not None else None,
            "has_prediction": prediction is not None,
            "semantic_rules": len(self.episodic_memory.semantic_rules),
        }

    def get_strongest_memories(self, n: int = 10) -> dict:
        """Phase 11: Return the n strongest memories in episodic store."""
        if self.episodic_memory is None:
            return {"error": "EpisodicMemoryEngine not available"}
        return {
            "memories": self.episodic_memory.get_strongest_memories(n),
            "total_episodes": self.episodic_memory.episode_count(),
        }

    # ── Phase 12: Self-Awareness API ───────────────────────────────────────

    def self_assess(self) -> dict:
        """
        Phase 12: Full self-assessment report.
        The system reflects on its own learning, weaknesses, and goals.
        """
        if self.self_awareness is None:
            return {"error": "DeepSelfAwareness (Phase 12) not available"}
        signal_stats = self.signal_bus.get_stats() if self.signal_bus else None
        mem_summary  = self.episodic_memory.summary() if self.episodic_memory else None
        return self.self_awareness.self_assess(signal_stats, mem_summary)

    def get_self_awareness_status(self) -> dict:
        """Phase 12: Get confidence, weakness, and metacognition summaries."""
        if self.self_awareness is None:
            return {"enabled": False}
        return {"enabled": True, **self.self_awareness.summary()}

    def get_active_goals(self) -> dict:
        """Phase 12: Return the system's currently active self-improvement goals."""
        if self.self_awareness is None:
            return {"error": "DeepSelfAwareness not available"}
        return {
            "active_goals":    self.self_awareness.metacog._active_goals,
            "completed_goals": len(self.self_awareness.metacog._completed_goals),
            "explore_rate":    round(self.self_awareness.metacog._explore_rate, 4),
            "is_stagnating":   self.self_awareness.metacog._is_stagnating(),
        }

    def get_weak_regions(self) -> dict:
        """
        Phase 12: Identify feature-space regions where the network performs poorly.
        Returns top weaknesses with severity scores and remedial signal count.
        """
        if self.self_awareness is None:
            return {"error": "DeepSelfAwareness not available"}
        weaknesses = self.self_awareness.weakness.scan()
        remedial   = self.self_awareness.get_remedial_signals(n=len(weaknesses) * 2)
        return {
            "weak_regions":          weaknesses,
            "count":                 len(weaknesses),
            "remedial_signals_ready": len(remedial),
            "coverage_pct":          self.self_awareness.weakness.coverage_pct(),
        }

    def get_full_system_status(self) -> dict:
        """
        Master status endpoint — all phases in one call.
        The complete picture of the digital nervous system.
        """
        return {
            "version":       self.VERSION,
            "timestamp":     datetime.utcnow().isoformat(),
            "phase9":        self.get_phase9_status(),
            "phase10_signal_stream": self.get_signal_stream_status(),
            "phase11_memory":        self.get_memory_status(),
            "phase12_self_awareness": self.get_self_awareness_status(),
        }

    # ── Pre-Phase 7: Validation ────────────────────────────────────────────

    def validate_phase6(self, project_root: Optional[str] = None, save_report: bool = True) -> dict:
        """
        Generate a full Phase 6 Validation Report before transitioning to Phase 7.

        The report covers:
          • File count & line-of-code breakdown
          • Live system: nodes, routes, agents, swarm tasks, DNA snapshots
          • Module usage analysis (which modules are actually imported)
          • Phase coverage 1-6 (% of modules active per phase)
          • Dead code detection (files never reached from main.py/app.py)
          • Phase 7 readiness score (0-100)

        Args:
            project_root: Override the project root directory (auto-detected if None)
            save_report:  If True, saves report JSON to ./data/phase6_validation_report.json

        Returns:
            Full validation report as a dict.
        """
        import json, os
        validator = Phase6Validator(self, project_root=project_root)
        report = validator.generate()

        if save_report:
            os.makedirs("./data", exist_ok=True)
            report_path = "./data/phase6_validation_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"  Report saved → {report_path}\n")

        return report


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
    p.add_argument("--mode", choices=["demo", "api", "simulate", "evolve", "phase6", "validate", "evolve7"], default="demo",
                   help="demo: example pipeline | api: Flask server | simulate: Phase 4 learning sim | evolve: Phase 5 evolution | phase6: Phase 6 multi-agent demo | validate: Pre-Phase 7 validation report | evolve7: Phase 7 autonomous evolution")
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
    elif args.mode == "phase6":
        import json
        print("\n" + "="*65)
        print("  Neural Service Mesh  —  Phase 6 (Multi-Agent Self-Improving Platform)")
        print("="*65 + "\n")
        mesh = NeuralServiceMesh()
        from services.input_service import InputNode
        from services.processor_service import ProcessorNode
        from services.output_service import OutputNode
        inp  = mesh.register_node(InputNode("TextInput"))
        proc = mesh.register_node(ProcessorNode("TextProcessor"), connect_to=inp)
        out  = mesh.register_node(OutputNode("TextOutput", output_format="summary"), connect_to=proc)

        sample_data = {"text": "The neural mesh is self-improving.", "source": "phase6_demo"}

        print("[ Phase 6.1 — Agent Factory ]\n")
        agent = mesh.spawn_agent("ResearchAgent")
        print(f"  Spawned: {agent['role']} → {agent['agent_id']}\n")

        print("[ Phase 6.2 — Swarm Execution ]\n")
        swarm_result = mesh.swarm_execute("translate and review document", sample_data)
        print(f"  Swarm status : {swarm_result['status']}")
        print(f"  Tasks run    : {swarm_result['total_tasks']}")
        print(f"  Tasks OK     : {swarm_result['success_count']}\n")

        print("[ Phase 6.3 — Simulation Lab (100 plans) ]\n")
        sim = mesh.simulate_plans("translate document", sample_data, n_plans=100)
        best = sim.get("best_plan") or {}
        print(f"  Plans simulated : {sim.get('n_plans_simulated')}")
        print(f"  Best plan score : {best.get('composite_score')}")
        print(f"  Best path       : {' → '.join(best.get('node_names', []))}\n")

        print("[ Phase 6.4 — Self Optimizer ]\n")
        opt = mesh.self_optimize()
        print(f"  Cycle          : {opt['cycle']}")
        print(f"  Nodes checked  : {opt['nodes_checked']}")
        print(f"  Nodes flagged  : {opt['nodes_flagged']}\n")

        print("[ Phase 6.5 — Meta Reasoner ]\n")
        insights = mesh.meta_reflect()
        print(f"  Insights produced: {len(insights['insights'])}")
        for i in insights["insights"][:3]:
            print(f"    [{i['insight_type'].upper()}] {i['title']}")

        print("\n[ Phase 6.6 — Economic Leaderboard ]\n")
        board = mesh.economic_leaderboard()
        for entry in board["leaderboard"][:3]:
            print(f"  #{entry['rank']}  {entry['node_name']:20s}  "
                  f"cap={entry['capability_score']:.2f}  "
                  f"trust={entry['trust_score']:.2f}  "
                  f"composite={entry['composite_score']:.2f}")

        print("\n[ Phase 6.7 — System DNA Snapshot ]\n")
        dna = mesh.dna_snapshot(notes="Phase 6 demo snapshot")
        print(f"  Snapshot   : {dna['snapshot_id']}")
        print(f"  Version    : {dna['version']}")
        print(f"  Health     : {dna['composite_health']}")
        print(f"  Nodes      : {dna['node_count']}")
        print(f"  Routes     : {dna['route_count']}")

        print("\n[ Full System Status — Phase 6 ]\n")
        status = mesh.status()
        print(json.dumps(status.get("ai_phase6", {}), indent=2))
    elif args.mode == "validate":
        import json
        print("\n" + "="*65)
        print("  Neural Service Mesh  —  Pre-Phase 7 Validation")
        print("="*65 + "\n")
        # Boot the mesh and run a minimal pipeline so live stats are populated
        mesh = NeuralServiceMesh()
        from services.input_service import InputNode
        from services.processor_service import ProcessorNode
        from services.output_service import OutputNode
        inp  = mesh.register_node(InputNode("TextInput"))
        proc = mesh.register_node(ProcessorNode("TextProcessor"), connect_to=inp)
        out  = mesh.register_node(OutputNode("TextOutput", output_format="summary"), connect_to=proc)

        # Run a quick pipeline so route memory is populated
        sample = {"text": "Validation pre-flight check.", "source": "validate_mode"}
        mesh.run(inp, out, sample, use_ai=True)

        # Spawn one agent so factory stats are non-zero
        mesh.spawn_agent("MonitorAgent")

        # Take a DNA snapshot so dna_snapshots > 0
        mesh.dna_snapshot(notes="Pre-Phase 7 validation snapshot")

        # Generate the full report
        report = mesh.validate_phase6(save_report=True)

        # Print readiness summary
        rd = report["phase7_readiness"]
        print(f"\n  ╔══════════════════════════════════════════╗")
        print(f"  ║  PHASE 7 READINESS SCORE: {rd['score']:5.1f} / 100      ║")
        print(f"  ╚══════════════════════════════════════════╝")
        print(f"\n  {rd['verdict']}\n")
    elif args.mode == "evolve7":
        import json
        print("\n" + "="*65)
        print("  Neural Service Mesh  —  Phase 7 (Autonomous Evolution Platform)")
        print("="*65 + "\n")
        mesh = NeuralServiceMesh()

        from services.input_service import InputNode
        from services.processor_service import ProcessorNode
        from services.output_service import OutputNode
        inp  = mesh.register_node(InputNode("TextInput"))
        proc = mesh.register_node(ProcessorNode("TextProcessor"), connect_to=inp)
        out  = mesh.register_node(OutputNode("TextOutput", output_format="summary"), connect_to=proc)

        # Seed some execution history
        sample = {"text": "Phase 7 autonomous evolution test.", "source": "evolve7_demo"}
        mesh.run(inp, out, sample, use_ai=True)

        print("[ Phase 7.1 — Self-Awareness Introspection ]\n")
        awareness = mesh.introspect()
        print(f"  Nodes         : {awareness['node_count']}")
        print(f"  Edges         : {awareness['edge_count']}")
        print(f"  Health Score  : {awareness['system_health_score']:.2f}")
        print(f"  Readiness     : {awareness['phase7_readiness']:.0%}")
        for insight in awareness["insights"]:
            print(f"  Insight       : {insight}")

        print("\n[ Phase 7.2 — Strategic Objectives ]\n")
        objectives = mesh.get_objectives()
        for obj in objectives["objectives"]:
            val = obj.get("current_value")
            val_str = f"{val:.4f}" if val is not None else "not yet measured"
            print(f"  [{obj['priority']}] {obj['name']:35s} → {val_str}")

        print("\n[ Phase 7.3 — Measure Objectives ]\n")
        measurements = mesh.measure_objectives()
        for k, v in measurements["measurements"].items():
            print(f"  {k:30s}: {v:.4f}")
        if measurements["recommendations"]:
            print("\n  Recommendations:")
            for rec in measurements["recommendations"][:3]:
                print(f"    [{rec['priority']}] {rec['objective']}: {rec['reason']}")

        print("\n[ Phase 7.4 — Manual Module Generation ]\n")
        result = mesh.generate_module(
            gap_description="CsvToJsonConverter",
            source_name="CSVInput",
            target_name="JSONProcessor",
        )
        mod = result["module"]
        test = result["test_result"]
        decision = result["decision"]
        print(f"  Module      : {mod['name']} ({mod['class_name']})")
        print(f"  Syntax valid: {mod['syntax_valid']}")
        print(f"  Code lines  : {mod['code_lines']}")
        print(f"  Test score  : {test['score']}")
        print(f"  Verdict     : {test['verdict']}")
        print(f"  Decision    : {decision['verdict']} — {decision['reason'][:60]}")

        print("\n[ Phase 7.5 — Sensor Hub Status ]\n")
        sensors = mesh.sensor_status()
        hub = sensors["summary"]
        print(f"  Sensors registered : {hub['sensors_registered']}")
        print(f"  Total events       : {hub['total_events']}")
        for s in hub["sensors"]:
            print(f"  [{s['sensor_type']:12s}] {s['name']:25s} events={s['event_count']}")

        print("\n[ Phase 7.6 — World Model ]\n")
        wm = mesh.env_model.summary()
        print(f"  Total services     : {wm['total_services']}")
        print(f"  Healthy            : {wm['healthy_services']}")
        print(f"  Total capabilities : {wm['total_capabilities']}")
        print(f"  Failure patterns   : {wm['active_failure_patterns']}")

        print(f"\n[ Phase 7.7 — Full Evolution Pipeline ({args.cycles} cycle(s)) ]\n")
        evolution = mesh.evolve7(cycles=args.cycles, verbose=True)
        for i, cycle in enumerate(evolution["cycles"]):
            s = cycle["summary"]
            print(f"\n  Cycle {i+1} summary:")
            print(f"    Sensor events : {s['sensor_events']}")
            print(f"    Gaps detected : {s['gaps_detected']}")
            print(f"    Generated     : {s['modules_generated']}")
            print(f"    Tested        : {s['modules_tested']}")
            print(f"    Approved      : {s['modules_approved']}")
            print(f"    Deployed      : {s['modules_deployed']}")

        print("\n[ Phase 7 Complete — System Status ]\n")
        status = mesh.status()
        print(json.dumps(status.get("ai_phase7", {}), indent=2))
    elif args.mode == "api":
        from api.app import run_api
        mesh = NeuralServiceMesh()
        run_api(mesh, host=args.host, port=args.port, debug=args.debug)
