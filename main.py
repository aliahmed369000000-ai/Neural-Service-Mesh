from __future__ import annotations
import sys
from typing import Optional

# ── Logging ────────────────────────────────────────────────────────────────
from logs.mesh_logger import MeshLogger
_mesh_logger = MeshLogger(log_dir="./logs", level="INFO")

import logging
logger = logging.getLogger("NeuralServiceMesh.v3")

# ── Core ───────────────────────────────────────────────────────────────────
from storage.file_storage import FileStorage
from storage.db import SQLiteStorage
from core.registry import NodeRegistry
from core.graph import ServiceGraph
from core.engine import ExecutionEngine
from connectors.data_transformer import DataTransformer

# ── AI: Decision Layer ─────────────────────────────────────────────────────
from ai.decision import AIDecisionLayer

# ── AI: Knowledge & Semantic Layer ────────────────────────────────────────
from knowledge.knowledge_store import KnowledgeStore
from ai.semantic_matcher import SemanticMatcher
from ai.scoring_engine import ScoringEngine
from ai.memory_engine import MemoryEngine
from ai.discovery_engine import DiscoveryEngine
from ai.routing_engine import RoutingEngine
from ai.goal_planner import GoalPlanner
from ai.optimization_engine import OptimizationEngine

# ── AI: Learning & Reputation ─────────────────────────────────────────────
from ai.learning_validator import LearningValidator
from ai.reputation_engine import NodeReputationEngine

# ── AI: Autonomous Evolution ──────────────────────────────────────────────
from ai.service_generator import ServiceGeneratorEngine
from ai.gap_detector import GapDetectionEngine
from ai.capability_marketplace import CapabilityMarketplace
from ai.multi_goal_planner import MultiGoalPlanner
from ai.governor import AIGovernanceLayer
from ai.evolution_engine import EvolutionEngine

# ── AI: Multi-Agent Platform ──────────────────────────────────────────────
from ai.agent_factory import AgentFactory
from ai.swarm_coordinator import SwarmCoordinator
from ai.self_optimizer import SelfOptimizer
from ai.simulation_lab import SimulationLab
from ai.meta_reasoner import MetaReasoner
from ai.economic_engine import EconomicEngine
from ai.system_dna import SystemDNA
from ai.phase6_validator import Phase6Validator

# ── AI: Sensors & World Model ─────────────────────────────────────────────
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

# ── AI: Neural Weights ────────────────────────────────────────────────────
from ai.neural_weights import NeuralWeightLayer, extract_routing_weights

# ── AI: Rich Data + Dynamic Growth + Deep Network ─────────────────────────
try:
    from ai.rich_data_collector import RichDataCollector
    from ai.dynamic_weight_layer import DynamicWeightLayer, extract_routing_weights_dynamic
    from ai.deep_routing_network import DeepRoutingNetwork, extract_deep_routing_weights
    _RICH_DATA_AVAILABLE = True
except ImportError as _err:
    _RICH_DATA_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Rich data / deep network modules not available: {_err}")

# ── AI: Continuous Signal Stream ──────────────────────────────────────────
try:
    from ai.signal_stream import SignalBus, ReplayBuffer, Experience
    _SIGNAL_STREAM_AVAILABLE = True
except ImportError as _err:
    _SIGNAL_STREAM_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Signal stream not available: {_err}")

# ── AI: Episodic Memory ───────────────────────────────────────────────────
try:
    from ai.episodic_memory import EpisodicMemoryEngine, Episode
    _EPISODIC_MEMORY_AVAILABLE = True
except ImportError as _err:
    _EPISODIC_MEMORY_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Episodic memory not available: {_err}")

# ── AI: Deep Self-Awareness ───────────────────────────────────────────────
try:
    from ai.self_awareness_deep import DeepSelfAwareness
    _DEEP_SELF_AWARENESS_AVAILABLE = True
except ImportError as _err:
    _DEEP_SELF_AWARENESS_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Deep self-awareness not available: {_err}")

# ── AI: Structural Self-Redesign ──────────────────────────────────────────
try:
    from ai.structural_redesign import StructuralEvolutionEngine
    _STRUCTURAL_EVOLUTION_AVAILABLE = True
except ImportError as _err:
    _STRUCTURAL_EVOLUTION_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Structural evolution not available: {_err}")

# ── AI: Digital Being ─────────────────────────────────────────────────────
try:
    from ai.digital_being import DigitalBeingCore
    _DIGITAL_BEING_AVAILABLE = True
except ImportError as _err:
    _DIGITAL_BEING_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Digital being not available: {_err}")

# ── AI: Phase 15 — Quality, Immune, Replication ───────────────────────────
try:
    from ai.quality_engine import QualityEngine
    _QUALITY_ENGINE_AVAILABLE = True
except ImportError as _err:
    _QUALITY_ENGINE_AVAILABLE = False
    logging.getLogger(__name__).warning(f"QualityEngine not available: {_err}")

try:
    from ai.immune_system import ImmuneSystem
    _IMMUNE_SYSTEM_AVAILABLE = True
except ImportError as _err:
    _IMMUNE_SYSTEM_AVAILABLE = False
    logging.getLogger(__name__).warning(f"ImmuneSystem not available: {_err}")

try:
    from ai.self_replication import SelfReplicationEngine
    _SELF_REPLICATION_AVAILABLE = True
except ImportError as _err:
    _SELF_REPLICATION_AVAILABLE = False
    logging.getLogger(__name__).warning(f"SelfReplicationEngine not available: {_err}")

# ── AI: Phase 15 — Brain Checkpoint, World Feed, Drive Engine ─────────────
try:
    from ai.brain_checkpoint import BrainCheckpoint
    _BRAIN_CHECKPOINT_AVAILABLE = True
except ImportError as _err:
    _BRAIN_CHECKPOINT_AVAILABLE = False
    logging.getLogger(__name__).warning(f"BrainCheckpoint not available: {_err}")

try:
    from ai.world_feed import WorldFeed
    _WORLD_FEED_AVAILABLE = True
except ImportError as _err:
    _WORLD_FEED_AVAILABLE = False
    logging.getLogger(__name__).warning(f"WorldFeed not available: {_err}")

try:
    from ai.drive_engine import DriveEngine
    _DRIVE_ENGINE_AVAILABLE = True
except ImportError as _err:
    _DRIVE_ENGINE_AVAILABLE = False
    logging.getLogger(__name__).warning(f"DriveEngine not available: {_err}")

# ── AI: Phase 16 — Self-Narrative, Evolution Ethics, Memory Consolidator ───
try:
    from ai.self_narrative import SelfNarrative
    _SELF_NARRATIVE_AVAILABLE = True
except ImportError as _err:
    _SELF_NARRATIVE_AVAILABLE = False
    logging.getLogger(__name__).warning(f"SelfNarrative not available: {_err}")

try:
    from ai.evolution_ethics import EvolutionEthics
    _EVOLUTION_ETHICS_AVAILABLE = True
except ImportError as _err:
    _EVOLUTION_ETHICS_AVAILABLE = False
    logging.getLogger(__name__).warning(f"EvolutionEthics not available: {_err}")

try:
    from ai.memory_consolidator import MemoryConsolidator
    _MEMORY_CONSOLIDATOR_AVAILABLE = True
except ImportError as _err:
    _MEMORY_CONSOLIDATOR_AVAILABLE = False
    logging.getLogger(__name__).warning(f"MemoryConsolidator not available: {_err}")

# ── Services ───────────────────────────────────────────────────────────────
from services.input_service import InputNode
from services.processor_service import ProcessorNode
from services.output_service import OutputNode


class NeuralServiceMesh:
    """
    Neural Service Mesh — Autonomous Digital Nervous System.

    A self-evolving, self-aware neural service mesh that grows and learns
    continuously. Built across 14 progressive capability layers:

      Layers  1-2 : Core execution engine + AI decision layer
      Layers  3-4 : Semantic routing, memory, learning & reputation
      Layer   5   : Autonomous service creation & gap detection
      Layer   6   : Multi-agent swarm + meta-reasoning + economic engine
      Layer   7   : Sensors, world model, strategic objectives, code generation
      Layer   8   : Real neural weight matrix (gradient descent)
      Layer   9   : Rich data collection, dynamic growth, deep routing network
      Layer  10   : Continuous signal stream (24/7 self-stimulation)
      Layer  11   : Episodic memory (3-tier: working + episodic + semantic)
      Layer  12   : Deep self-awareness (confidence + weakness + metacognition)
      Layer  13   : Structural self-redesign (autonomous architecture evolution)
      Layer  14   : Complete digital being (unified consciousness core)
      Layer  15   : Quality engine + immune system + self-replication + identity + world feed + drives
      Layer  16   : Self-narrative (voice) + evolution ethics (conscience) + memory consolidation (laws)

    All public APIs are backward-compatible across all layers.
    """

    VERSION = "18.0.0"

    def __init__(self, storage_dir: str = "./data", db_path: str = "./data/mesh.db"):
        # ── Storage ────────────────────────────────────────────────────────
        self.storage = FileStorage(storage_dir)
        self.db = SQLiteStorage(db_path)

        # ── Core ───────────────────────────────────────────────────────────
        self.registry = NodeRegistry(self.storage)
        self.graph = ServiceGraph()
        self.transformer = DataTransformer()

        # ── Decision Layer ────────────────────────────────────────────────
        self.ai = AIDecisionLayer()

        # ── Knowledge Layer ───────────────────────────────────────────────
        self.knowledge = KnowledgeStore(knowledge_dir="./knowledge")

        # ── Semantic & Routing Intelligence ──────────────────────────────
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

        # ── Execution Engine ───────────────────────────────────────────────
        self.engine = ExecutionEngine(
            registry=self.registry,
            graph=self.graph,
            storage=self.storage,
            transformer=self.transformer,
            db=self.db,
            ai=self.ai,
        )

        # Wire decision layer
        self.ai.set_graph(self.graph)
        self.ai.set_db(self.db)

        # Wire knowledge store into routing modules
        self.memory.set_knowledge_store(self.knowledge)
        self.discovery.set_knowledge_store(self.knowledge)
        self.routing.set_knowledge_store(self.knowledge)
        self.optimizer.set_knowledge_store(self.knowledge)

        # ── Learning & Reputation ──────────────────────────────────────────
        self.validator = LearningValidator(
            memory_engine=self.memory,
            scoring_engine=self.scoring,
            knowledge_store=self.knowledge,
        )
        self.reputation = NodeReputationEngine(
            knowledge_store=self.knowledge,
            memory_engine=self.memory,
        )

        # ── Autonomous Service Creation & Evolution ────────────────────────
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

        # Install execution hook
        self._install_phase3_hook()

        # ── Multi-Agent Self-Improving Platform ────────────────────────────
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

        # ── Sensors & World Model ──────────────────────────────────────────
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

        self.env_model = EnvironmentModel(model_dir="./world_model")
        self.sensor_hub.on_event(lambda e: self.env_model.ingest_sensor_event(e.to_dict()))

        # NOTE: self.self_awareness is first set here (SelfAwarenessEngine from layer 7),
        # then overwritten below with DeepSelfAwareness (layer 12) if available.
        # We keep a reference to the basic engine for wiring.
        _basic_self_awareness = SelfAwarenessEngine(
            registry=self.registry,
            graph=self.graph,
            scoring_engine=self.scoring,
            memory_engine=self.memory,
            agent_factory=self.agent_factory,
            knowledge_store=self.knowledge,
            environment_model=self.env_model,
        )

        self.code_generator = CodeGenerationEngine(
            output_dir="./services",
            knowledge_store=self.knowledge,
            governance=self.governance,
        )
        self.sandbox_lab_p7 = SandboxTestingLab(sandbox_dir="./sandbox")
        self.governance_p7 = P7GovernanceLayer(
            min_sandbox_score=75.0,
            knowledge_store=self.knowledge,
            base_governance=self.governance,
        )
        self.objectives = ObjectivesEngine(knowledge_store=self.knowledge)
        _basic_self_awareness._objectives = self.objectives

        self.evolution_pipeline = EvolutionPipeline(
            mesh=self,
            sensor_hub=self.sensor_hub,
            environment_model=self.env_model,
            self_awareness=_basic_self_awareness,
            code_generator=self.code_generator,
            sandbox_lab=self.sandbox_lab_p7,
            governance_p7=self.governance_p7,
            objectives_engine=self.objectives,
            gap_detector=self.gap_detector,
            knowledge_store=self.knowledge,
            services_dir="./services",
        )

        # ── Neural Weight Layer ────────────────────────────────────────────
        self.neural_layer: Optional[NeuralWeightLayer] = self.routing.get_neural_layer()
        if self.neural_layer is not None:
            logger.info(
                f"Neural weight layer active  "
                f"shape={self.neural_layer.weights.shape}  "
                f"steps={self.neural_layer._train_steps}"
            )
        else:
            logger.warning("NeuralWeightLayer unavailable — install numpy to enable real neural weights.")

        # ── Rich Data + Dynamic Growth + Deep Network ──────────────────────
        self.rich_data_collector = getattr(self.routing, '_rich_data', None)
        if self.rich_data_collector is not None:
            try:
                self.rich_data_collector.set_env_model(self.env_model)
            except Exception:
                pass

        self.dynamic_layer = getattr(self.routing, '_dynamic_layer', None)
        self.deep_network = getattr(self.routing, '_deep_network', None)

        _active_axes = sum([
            self.rich_data_collector is not None,
            self.dynamic_layer is not None,
            self.deep_network is not None,
        ])
        if _active_axes > 0:
            logger.info(
                f"Rich data axes active: {_active_axes}/3 — "
                f"RichData={'✓' if self.rich_data_collector else '✗'}  "
                f"DynamicLayer={'✓' if self.dynamic_layer else '✗'}  "
                f"DeepNetwork={'✓' if self.deep_network else '✗'}"
            )

        # ── Continuous Signal Stream ───────────────────────────────────────
        self.signal_bus: Optional["SignalBus"] = None
        if _SIGNAL_STREAM_AVAILABLE:
            self.signal_bus = SignalBus(
                deep_network=self.deep_network,
                dynamic_layer=self.dynamic_layer,
                neural_layer=getattr(self.routing, '_neural_layer', None),
                rich_data_collector=self.rich_data_collector,
            )
            logger.info("SignalBus ready — SelfStimulator + CuriosityEngine + DreamConsolidator active")

        # ── Episodic Memory ────────────────────────────────────────────────
        self.episodic_memory: Optional["EpisodicMemoryEngine"] = None
        if _EPISODIC_MEMORY_AVAILABLE:
            self.episodic_memory = EpisodicMemoryEngine()
            self.episodic_memory.start()
            logger.info("EpisodicMemoryEngine active — 3-tier memory online (working + episodic + semantic)")

        # ── Deep Self-Awareness ────────────────────────────────────────────
        self.self_awareness: Optional["DeepSelfAwareness"] = None
        if _DEEP_SELF_AWARENESS_AVAILABLE:
            self.self_awareness = DeepSelfAwareness()
            logger.info("DeepSelfAwareness active — ConfidenceEstimator + WeaknessDetector + MetaCognition online")
        else:
            # Fall back to the basic awareness engine so the attribute is never None
            self.self_awareness = _basic_self_awareness

        # ── Structural Self-Redesign ───────────────────────────────────────
        self.structural_evolution = None
        if _STRUCTURAL_EVOLUTION_AVAILABLE:
            self.structural_evolution = StructuralEvolutionEngine(
                mesh=self,
                deep_network=self.deep_network,
                episodic_memory=self.episodic_memory,
                signal_bus=self.signal_bus,
                self_awareness=self.self_awareness,
            )
            logger.info("StructuralEvolutionEngine active — autonomous architecture evolution online")

        # ── Digital Being ──────────────────────────────────────────────────
        self.digital_being = None
        if _DIGITAL_BEING_AVAILABLE:
            self.digital_being = DigitalBeingCore(
                mesh=self,
                signal_bus=self.signal_bus,
                episodic_memory=self.episodic_memory,
                self_awareness=self.self_awareness,
                structural_evolution=self.structural_evolution,
                evolution_pipeline=getattr(self, "evolution_pipeline", None),
                evolution_ethics=getattr(self, "evolution_ethics", None),
                self_narrative=getattr(self, "self_narrative", None),
            )
            logger.info("DigitalBeingCore active — unified digital consciousness online")

        logger.info(
            f"NeuralServiceMesh v{self.VERSION} fully ready — "
            f"SignalBus={'✓' if self.signal_bus else '✗'}  "
            f"Memory={'✓' if self.episodic_memory else '✗'}  "
            f"SelfAware={'✓' if self.self_awareness else '✗'}  "
            f"StructuralEvo={'✓' if self.structural_evolution else '✗'}  "
            f"DigitalBeing={'✓' if self.digital_being else '✗'}"
        )

        # ── Phase 15: Quality Engine ───────────────────────────────────────
        self.quality_engine = None
        if _QUALITY_ENGINE_AVAILABLE:
            self.quality_engine = QualityEngine(knowledge_store=self.knowledge)
            logger.info("QualityEngine active — data quality scoring 0-100 online")

        # ── Phase 15: Immune System ────────────────────────────────────────
        self.immune_system = None
        if _IMMUNE_SYSTEM_AVAILABLE:
            self.immune_system = ImmuneSystem(knowledge_store=self.knowledge)
            logger.info("ImmuneSystem active — digital immune layer online")

        # ── Phase 15: Self-Replication Engine ─────────────────────────────
        self.replication_engine = None
        if _SELF_REPLICATION_AVAILABLE:
            self.replication_engine = SelfReplicationEngine()
            logger.info("SelfReplicationEngine active — self-improvement loop online")

        # ── Phase 15: Brain Checkpoint ────────────────────────────────────
        self.brain_checkpoint = None
        if _BRAIN_CHECKPOINT_AVAILABLE:
            self.brain_checkpoint = BrainCheckpoint(
                checkpoint_dir="./checkpoints",
                max_checkpoints=5,
            )
            self.brain_checkpoint.set_mesh(self)
            # Attempt to restore last checkpoint on startup
            try:
                latest = self.brain_checkpoint.list_checkpoints()
                if latest:
                    logger.info(
                        f"[BrainCheckpoint] Restoring from: {latest[0]['filename']}"
                    )
                    # (state application handled by caller or auto-wired layers)
            except Exception as _ckpt_err:
                logger.warning(f"[BrainCheckpoint] Startup restore skipped: {_ckpt_err}")
            logger.info("BrainCheckpoint active — persistent identity across restarts online")

        # ── Phase 15: World Feed ──────────────────────────────────────────
        self.world_feed = None
        if _WORLD_FEED_AVAILABLE:
            self.world_feed = WorldFeed(
                immune_system=self.immune_system,
                quality_engine=self.quality_engine,
                min_quality=60.0,
            )
            # Wire memory callback so accepted items flow into episodic memory
            if self.episodic_memory is not None:
                def _wf_memory_cb(item: dict):
                    try:
                        quality = float(item.get("quality_score", 70)) / 100.0
                        self.episodic_memory.record(
                            feature_vec = [quality, 0.5, 0.5, 0.5, 0.5],
                            target      = quality,
                            outcome     = quality,
                            source      = item.get("source", "world_feed"),
                            reward      = quality * 0.8,
                            context     = {
                                "content":       item.get("content", "")[:500],
                                "title":         item.get("title", ""),
                                "url":           item.get("url", ""),
                                "quality_score": item.get("quality_score", 70),
                                "ingested_at":   item.get("ingested_at", ""),
                            },
                        )
                    except Exception as _e:
                        logger.debug(f"[WorldFeed→Memory] record failed: {_e}")
                self.world_feed.set_memory_callback(_wf_memory_cb)
            logger.info("WorldFeed active — real-world data ingestion online")

        # ── Phase 15: Drive Engine ────────────────────────────────────────
        self.drive_engine = None
        if _DRIVE_ENGINE_AVAILABLE:
            self.drive_engine = DriveEngine(
                signal_bus=self.signal_bus,
                mesh_ref=self,
            )
            logger.info("DriveEngine active — internal motivation system online")

        logger.info(
            f"NeuralServiceMesh v{self.VERSION} Phase 15 complete — "
            f"Checkpoint={'✓' if self.brain_checkpoint else '✗'}  "
            f"WorldFeed={'✓' if self.world_feed else '✗'}  "
            f"DriveEngine={'✓' if self.drive_engine else '✗'}  "
            f"QualityEngine={'✓' if self.quality_engine else '✗'}  "
            f"ImmuneSystem={'✓' if self.immune_system else '✗'}  "
            f"Replication={'✓' if self.replication_engine else '✗'}"
        )

        # ── Phase 16: Evolution Ethics ────────────────────────────────────
        self.evolution_ethics = None
        if _EVOLUTION_ETHICS_AVAILABLE:
            try:
                self.evolution_ethics = EvolutionEthics(
                    immune_system=self.immune_system,
                )
            except TypeError:
                # نسخة قديمة لا تقبل immune_system
                self.evolution_ethics = EvolutionEthics()
                if self.immune_system is not None:
                    try:
                        self.evolution_ethics._immune_system = self.immune_system
                    except Exception:
                        pass
            logger.info("EvolutionEthics active — ethical conscience layer online")

        # ── Phase 16: Self-Narrative ──────────────────────────────────────
        self.self_narrative = None
        if _SELF_NARRATIVE_AVAILABLE:
            try:
                self.self_narrative = SelfNarrative(
                    knowledge_store=self.knowledge,
                )
            except TypeError:
                try:
                    self.self_narrative = SelfNarrative()
                except Exception:
                    self.self_narrative = None
            # تسجيل حدث البدء
            self.self_narrative.record_event(
                event_type    = "checkpoint",
                data          = {"message": f"الجهاز بدأ — الإصدار {self.VERSION}"},
                surprise_score = 0.1,
                importance    = 0.8,
            )
            logger.info("SelfNarrative active — digital voice and identity journal online")

        # ── Phase 16: Memory Consolidator ────────────────────────────────
        self.memory_consolidator = None
        if _MEMORY_CONSOLIDATOR_AVAILABLE:
            try:
                self.memory_consolidator = MemoryConsolidator(
                    episodic_memory   = self.episodic_memory,
                    pattern_threshold = 10,
                )
            except TypeError:
                try:
                    self.memory_consolidator = MemoryConsolidator()
                except Exception:
                    self.memory_consolidator = None
            logger.info("MemoryConsolidator active — semantic law engine online")

        logger.info(
            f"NeuralServiceMesh v{self.VERSION} Phase 16+17 (fixes) complete — "
            f"SelfNarrative={'✓' if self.self_narrative else '✗'}  "
            f"EvolutionEthics={'✓' if self.evolution_ethics else '✗'}  "
            f"MemoryConsolidator={'✓' if self.memory_consolidator else '✗'}"
        )

    # ── Execution Hook ─────────────────────────────────────────────────────

    def _install_phase3_hook(self):
        """Monkey-patch ExecutionEngine._persist to feed all learners."""
        original_persist = self.engine._persist

        def _enhanced_persist(result):
            original_persist(result)
            result_dict = result.to_dict()
            self.scoring.record_run(result_dict)
            self.memory.learn_from_run(result_dict)

            try:
                latency = result_dict.get("total_duration_ms", 0.0)
                for step in result_dict.get("steps", []):
                    nid = step.get("node_id", "")
                    name = step.get("node_name", "")
                    step_ok = step.get("status") == "success"
                    self.reputation.record_execution(nid, name, step_ok, latency)
            except Exception:
                pass

        self.engine._persist = _enhanced_persist

    # ── Node Management ────────────────────────────────────────────────────

    def register_node(self, node, connect_to: Optional[str] = None) -> str:
        node_id = self.registry.register(node)
        self.graph.add_node(node_id, node.metadata.to_dict())
        self.db.upsert_node(node.to_dict())
        self.discovery.announce(node)

        try:
            self.marketplace.advertise_from_node(node)
        except Exception:
            pass

        if connect_to:
            self.graph.add_edge(connect_to, node_id)
            self.db.upsert_connection(connect_to, node_id)

        logger.info(f"Registered node '{node.name}' [{node_id[:8]}]")
        return node_id

    # ── Execution ──────────────────────────────────────────────────────────

    def run(self, start_id: str, end_id: str, data: dict, use_ai: bool = True) -> dict:
        return self.engine.run_between(start_id, end_id, data, use_ai=use_ai).to_dict()

    def run_goal(self, goal: str, data: dict,
                 preferred_start: Optional[str] = None,
                 preferred_end: Optional[str] = None,
                 max_hops: int = 10) -> dict:
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

    # ── Optimization ───────────────────────────────────────────────────────

    def optimize(self, auto_apply: bool = False) -> dict:
        report = self.optimizer.analyze()
        if auto_apply:
            from ai.optimization_engine import OptimizationAction
            safe_types = {
                OptimizationAction.SUGGEST_EDGE,
                OptimizationAction.UPDATE_WEIGHT,
                OptimizationAction.PROMOTE_EDGE,
            }
            safe_actions = [a for a in report.actions if a.action_type in safe_types]
            all_actions = report.actions
            report.actions = safe_actions
            applied = self.optimizer.apply_report(report, self)
            report.actions = all_actions
            logger.info(f"Auto-applied {applied} safe optimization actions")
        return report.to_dict()

    # ── Discovery ──────────────────────────────────────────────────────────

    def discover_connections(self, threshold: float = 0.15) -> list:
        existing = [
            (src, e.target_id)
            for src, edges in self.graph._adjacency.items()
            for e in edges
        ]
        return self.semantic.suggest_new_connections(existing, threshold)

    def find_nodes_for_goal(self, goal: str, top_k: int = 5) -> list:
        return self.discovery.find_nodes_for_goal(goal, top_k)

    # ── Status ─────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "version": self.VERSION,
            "nodes": self.registry.count(),
            "graph": self.graph.stats(),
            "storage": self.storage.stats(),
            "db": self.db.db_stats(),
            "decision_layer": {
                "enabled": True,
                "mode": "rules+heuristics",
                "paths_tracked": len(self.ai._path_stats),
            },
            "semantic_routing": {
                "semantic_profiles": self.semantic.profile_count(),
                "scoring": self.scoring.summary(),
                "memory": self.memory.summary(),
                "discovery": self.discovery.summary(),
                "optimization_runs": self.optimizer._run_count,
            },
            "knowledge_layer": self.knowledge.summary(),
            "learning": {
                "learning_metrics": self.validator.compute_metrics().to_dict(),
                "reputation": self.reputation.summary(),
            },
            "evolution": {
                "governance": self.governance.summary(),
                "marketplace": self.marketplace.summary(),
                "gap_detector": self.gap_detector.summary(),
                "service_generator": self.service_generator.summary(),
                "evolution_engine": self.evolution.summary(),
            },
            "multi_agent": {
                "agent_factory": self.agent_factory.summary(),
                "swarm": self.swarm.summary(),
                "self_optimizer": self.self_optimizer.summary(),
                "simulation_lab": self.simulation_lab.summary(),
                "meta_reasoner": self.meta_reasoner.summary(),
                "economic_engine": self.economic_engine.summary(),
                "system_dna": self.system_dna.summary(),
            },
            "world_model": {
                "sensor_hub": self.sensor_hub.summary(),
                "environment_model": self.env_model.summary(),
                "self_awareness_basic": _basic_self_awareness.summary() if hasattr(self, '_basic_self_awareness') else {},
                "code_generator": self.code_generator.summary(),
                "sandbox_lab": self.sandbox_lab_p7.summary(),
                "governance_p7": self.governance_p7.summary(),
                "objectives": self.objectives.summary(),
                "evolution_pipeline": self.evolution_pipeline.summary(),
            },
            "neural_weights": self.routing.neural_weights_summary(),
            "signal_stream": self.get_signal_stream_status(),
            "episodic_memory": self.get_memory_status(),
            "deep_self_awareness": self.get_self_awareness_status(),
            "structural_evolution": {
                "enabled": self.structural_evolution is not None,
            },
            "digital_being": {
                "enabled": self.digital_being is not None,
            },
        }

    # ── Learning & Reputation API ──────────────────────────────────────────

    def get_ai_status(self) -> dict:
        proof = self.validator.prove_learning()
        return {
            "version": self.VERSION,
            "system_status": self.status(),
            "learning_proof": proof,
        }

    def get_ai_routes(self) -> dict:
        routes = self.memory.all_routes()
        ranked = sorted(routes, key=lambda r: r.get("memory_score", 0), reverse=True)
        return {"routes": ranked, "count": len(ranked), "summary": self.memory.summary()}

    def get_ai_reputation(self) -> dict:
        self.reputation.update_from_memory()
        return {"nodes": self.reputation.all_reputations(), "summary": self.reputation.summary()}

    def get_ai_knowledge(self) -> dict:
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

    # ── Autonomous Evolution API ───────────────────────────────────────────

    def run_multi_goal(self, goal: str, data: dict) -> dict:
        plan = self.multi_planner.plan(goal)
        if plan.status == "failed":
            return {"status": "failed", "error": f"MultiGoalPlanner could not resolve goal: '{goal}'", "plan": plan.to_dict()}
        return self.multi_planner.execute_plan(plan, self.engine, data)

    def evolve(self, cycles: int = 1, auto_register: bool = True) -> dict:
        return self.evolution.evolve(cycles=cycles, auto_register=auto_register, verbose=True)

    def scan_gaps(self) -> dict:
        gaps = self.gap_detector.scan()
        return {"gaps": [g.to_dict() for g in gaps], "count": len(gaps), "summary": self.gap_detector.summary()}

    def get_marketplace(self) -> dict:
        return {
            "summary": self.marketplace.summary(),
            "capabilities": self.marketplace.list_capabilities(),
            "advertisements": self.marketplace.all_advertisements(),
        }

    def get_governance(self) -> dict:
        return {"summary": self.governance.summary(), "recent_audit": self.governance.audit_log(limit=20)}

    def get_generated_services(self, status: Optional[str] = None) -> dict:
        return {"services": self.service_generator.list_generated(status_filter=status), "summary": self.service_generator.summary()}

    # ── Multi-Agent API ────────────────────────────────────────────────────

    def spawn_agent(self, role: str, config: Optional[dict] = None) -> dict:
        return self.agent_factory.spawn(role, config).to_dict()

    def swarm_execute(self, goal: str, data: dict, custom_tasks: Optional[list] = None) -> dict:
        return self.swarm.execute(goal, data, custom_tasks).to_dict()

    def self_optimize(self) -> dict:
        return self.self_optimizer.run_cycle().to_dict()

    def simulate_plans(self, goal: str, data: dict, n_plans: int = 100) -> dict:
        return self.simulation_lab.run(goal=goal, data=data, n_plans=n_plans).to_dict()

    def meta_reflect(self) -> dict:
        insights = self.meta_reasoner.reflect()
        return {"insights": [i.to_dict() for i in insights], "summary": self.meta_reasoner.summary()}

    def meta_ask(self, question: str) -> dict:
        return self.meta_reasoner.ask(question)

    def economic_leaderboard(self, top_n: int = 10) -> dict:
        return {"leaderboard": self.economic_engine.leaderboard(top_n), "summary": self.economic_engine.summary()}

    def dna_snapshot(self, notes: str = "") -> dict:
        snap = self.system_dna.snapshot(
            registry=self.registry,
            scoring_engine=self.scoring,
            memory_engine=self.memory,
            notes=notes,
        )
        return snap.to_dict()

    def dna_diff(self, snapshot_id_a: str, snapshot_id_b: str) -> dict:
        return self.system_dna.diff(snapshot_id_a, snapshot_id_b)

    def dna_rollback(self, snapshot_id: str) -> dict:
        return {"success": self.system_dna.apply(snapshot_id), "snapshot_id": snapshot_id}

    def get_agent_factory(self) -> dict:
        return {"summary": self.agent_factory.summary(), "agents": self.agent_factory.all_agents()}

    def get_swarm_history(self, limit: int = 10) -> dict:
        return {"history": self.swarm.history(limit), "summary": self.swarm.summary()}

    # ── World Model & Sensors API ──────────────────────────────────────────

    def evolve7(self, cycles: int = 1, verbose: bool = True) -> dict:
        results = self.evolution_pipeline.run_cycles(n=cycles, verbose=verbose)
        return {"cycles": results, "total": len(results)}

    def evolve7_once(self, verbose: bool = True) -> dict:
        return self.evolution_pipeline.run_cycle(verbose=verbose).to_dict()

    def introspect(self) -> dict:
        report = self.self_awareness.introspect()
        return report.to_dict()

    def sensor_status(self) -> dict:
        return {"summary": self.sensor_hub.summary(), "recent_events": self.sensor_hub.recent_events(limit=20)}

    def world_model(self) -> dict:
        return self.env_model.get_state()

    def push_sensor_event(self, event_type: str, payload: Optional[dict] = None, severity: str = "info"):
        self.webhook_sensor.push_event(event_type, payload=payload, severity=severity)

    def get_objectives(self) -> dict:
        return {
            "objectives": self.objectives.get_all_objectives(),
            "summary": self.objectives.summary(),
            "recommendations": self.objectives.get_recommendations(),
        }

    def measure_objectives(self) -> dict:
        measurements = self.objectives.measure_from_mesh(self)
        return {
            "measurements": measurements,
            "objectives": self.objectives.get_all_objectives(),
            "recommendations": self.objectives.get_recommendations(),
        }

    def generate_module(self, gap_description: str, source_name: str = "", target_name: str = "") -> dict:
        gap = {
            "missing_service": gap_description,
            "gap_type": "manual",
            "source_node": {"name": source_name or "Source"},
            "target_node": {"name": target_name or "Target"},
            "confidence": 0.9,
        }
        module = self.code_generator.generate_from_gap(gap)
        self.code_generator.write_to_file(module, subdir="generated")
        test_result = self.sandbox_lab_p7.test_module(module)
        decision = self.governance_p7.review(module, test_result.to_dict())
        return {
            "module": module.to_dict(),
            "test_result": test_result.to_dict(),
            "decision": decision.to_dict(),
        }

    def list_generated_modules(self, status: Optional[str] = None) -> dict:
        return {"modules": self.code_generator.list_generated(status_filter=status), "summary": self.code_generator.summary()}

    def start_sensors(self, interval_s: float = 30.0):
        self.sensor_hub.start(interval_s=interval_s)
        return {"status": "started", "interval_s": interval_s}

    def stop_sensors(self):
        self.sensor_hub.stop()
        return {"status": "stopped"}

    def get_evolution_history(self, limit: int = 10) -> dict:
        return {"history": self.evolution_pipeline.get_history(limit), "summary": self.evolution_pipeline.summary()}

    # ── Neural Weights API ─────────────────────────────────────────────────

    def get_neural_weights(self) -> dict:
        summary = self.routing.neural_weights_summary()
        if self.neural_layer is not None:
            summary["weights_matrix"] = self.neural_layer.get_weights_list()
        return summary

    def train_neural_weights(self, input_vector: list, target: float) -> dict:
        if self.neural_layer is None:
            return {"error": "NeuralWeightLayer not available — install numpy"}
        loss = self.neural_layer.train_step(input_vector, target)
        self.routing._sync_weights_from_layer()
        return {
            "loss": round(loss, 8),
            "train_steps": self.neural_layer._train_steps,
            "routing_scalars": {
                "W_SEMANTIC": self.routing.W_SEMANTIC,
                "W_SCORE":    self.routing.W_SCORE,
                "W_MEMORY":   self.routing.W_MEMORY,
                "W_TOPOLOGY": self.routing.W_TOPOLOGY,
            },
        }

    def save_neural_weights(self, path: Optional[str] = None) -> dict:
        if self.neural_layer is None:
            return {"error": "NeuralWeightLayer not available"}
        dest = path or self.routing._WEIGHTS_PATH
        return {"saved": True, "path": self.neural_layer.save(dest)}

    def load_neural_weights(self, path: str) -> dict:
        if self.neural_layer is None:
            return {"error": "NeuralWeightLayer not available"}
        self.neural_layer.load(path)
        self.routing._sync_weights_from_layer()
        return {
            "loaded": True,
            "path": path,
            "routing_scalars": {
                "W_SEMANTIC": self.routing.W_SEMANTIC,
                "W_SCORE":    self.routing.W_SCORE,
                "W_MEMORY":   self.routing.W_MEMORY,
                "W_TOPOLOGY": self.routing.W_TOPOLOGY,
            },
        }

    # ── Deep Network API ───────────────────────────────────────────────────

    def get_phase9_status(self) -> dict:
        return self.routing.neural_weights_summary()

    def get_deep_network_summary(self) -> dict:
        if self.deep_network is None:
            return {"error": "DeepRoutingNetwork not available"}
        return self.deep_network.summary()

    def get_dynamic_layer_summary(self) -> dict:
        if self.dynamic_layer is None:
            return {"error": "DynamicWeightLayer not available"}
        return self.dynamic_layer.summary()

    def get_rich_data_summary(self) -> dict:
        if self.rich_data_collector is None:
            return {"error": "RichDataCollector not available"}
        return self.rich_data_collector.summary()

    def train_deep_network(self, input_vector: list, target: float) -> dict:
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

    # ── Signal Stream API ──────────────────────────────────────────────────

    def start_signal_stream(self, interval_s: float = 2.0) -> dict:
        if self.signal_bus is None:
            return {"error": "SignalBus not available"}
        if self.signal_bus.is_running:
            return {"status": "already_running", "stats": self.signal_bus.get_stats()}
        self.signal_bus.start(interval_s=interval_s)
        return {
            "status": "started",
            "interval_s": interval_s,
            "message": "Signal stream active — SelfStimulator + Curiosity + Dream mode online",
        }

    def stop_signal_stream(self) -> dict:
        if self.signal_bus is None:
            return {"error": "SignalBus not available"}
        self.signal_bus.stop()
        return {"status": "stopped", "final_stats": self.signal_bus.get_stats()}

    def get_signal_stream_status(self) -> dict:
        if self.signal_bus is None:
            return {"enabled": False}
        return {"enabled": True, **self.signal_bus.get_stats()}

    def push_real_signal(self, feature_vec: list, target: float, reward: float = 0.0, context: Optional[dict] = None) -> dict:
        if self.signal_bus is None:
            return {"error": "SignalBus not available"}
        self.signal_bus.push_real(feature_vec, target, reward, context)

        if self.episodic_memory is not None:
            self.episodic_memory.record(
                feature_vec=feature_vec,
                target=target,
                source="real",
                reward=reward,
                context=context,
            )
        return {"status": "recorded", "buffer_size": self.signal_bus.replay_buffer.size}

    # ── Episodic Memory API ────────────────────────────────────────────────

    def get_memory_status(self) -> dict:
        if self.episodic_memory is None:
            return {"enabled": False}
        return {"enabled": True, **self.episodic_memory.summary()}

    def recall_similar(self, feature_vec: list, top_k: int = 5) -> dict:
        if self.episodic_memory is None:
            return {"error": "EpisodicMemoryEngine not available"}
        episodes = self.episodic_memory.recall_similar(feature_vec, top_k=top_k)
        return {"query_vec": feature_vec, "recalled": [e.to_dict() for e in episodes], "count": len(episodes)}

    def predict_from_memory(self, feature_vec: list) -> dict:
        if self.episodic_memory is None:
            return {"error": "EpisodicMemoryEngine not available"}
        prediction = self.episodic_memory.predict_from_memory(feature_vec)
        return {
            "feature_vec": feature_vec,
            "prediction": round(prediction, 4) if prediction is not None else None,
            "has_prediction": prediction is not None,
            "semantic_rules": len(self.episodic_memory.semantic_rules),
        }

    def get_strongest_memories(self, n: int = 10) -> dict:
        if self.episodic_memory is None:
            return {"error": "EpisodicMemoryEngine not available"}
        return {"memories": self.episodic_memory.get_strongest_memories(n), "total_episodes": self.episodic_memory.episode_count()}

    # ── Deep Self-Awareness API ────────────────────────────────────────────

    def self_assess(self) -> dict:
        if self.self_awareness is None:
            return {"error": "DeepSelfAwareness not available"}
        signal_stats = self.signal_bus.get_stats() if self.signal_bus else None
        mem_summary  = self.episodic_memory.summary() if self.episodic_memory else None
        return self.self_awareness.self_assess(signal_stats, mem_summary)

    def get_self_awareness_status(self) -> dict:
        if self.self_awareness is None:
            return {"enabled": False}
        return {"enabled": True, **self.self_awareness.summary()}

    def get_active_goals(self) -> dict:
        if self.self_awareness is None:
            return {"error": "DeepSelfAwareness not available"}
        return {
            "active_goals":    self.self_awareness.metacog._active_goals,
            "completed_goals": len(self.self_awareness.metacog._completed_goals),
            "explore_rate":    round(self.self_awareness.metacog._explore_rate, 4),
            "is_stagnating":   self.self_awareness.metacog._is_stagnating(),
        }

    def get_weak_regions(self) -> dict:
        if self.self_awareness is None:
            return {"error": "DeepSelfAwareness not available"}
        weaknesses = self.self_awareness.weakness.scan()
        remedial   = self.self_awareness.get_remedial_signals(n=len(weaknesses) * 2)
        return {
            "weak_regions":           weaknesses,
            "count":                  len(weaknesses),
            "remedial_signals_ready": len(remedial),
            "coverage_pct":           self.self_awareness.weakness.coverage_pct(),
        }

    # ── Structural Evolution API ───────────────────────────────────────────

    def get_structural_evolution_status(self) -> dict:
        if self.structural_evolution is None:
            return {"enabled": False}
        return {"enabled": True, **self.structural_evolution.summary()}

    # ── Digital Being API ──────────────────────────────────────────────────

    def get_digital_being_status(self) -> dict:
        if self.digital_being is None:
            return {"enabled": False}
        return {"enabled": True, **self.digital_being.summary()}

    # ── Master Status ──────────────────────────────────────────────────────

    def get_full_system_status(self) -> dict:
        from datetime import datetime
        quran_info = {}
        try:
            quran_info = self.knowledge.quran_stats()
        except Exception:
            quran_info = {"stored": False}
        return {
            "version":              self.VERSION,
            "timestamp":            datetime.utcnow().isoformat(),
            "quran":                quran_info,
            "deep_network":         self.get_phase9_status(),
            "signal_stream":        self.get_signal_stream_status(),
            "episodic_memory":      self.get_memory_status(),
            "deep_self_awareness":  self.get_self_awareness_status(),
            "structural_evolution": self.get_structural_evolution_status(),
            "digital_being":        self.get_digital_being_status(),
            # ── Phase 15 ──────────────────────────────────────────────────
            "quality_engine":       self.get_quality_engine_status(),
            "immune_system":        self.get_immune_system_status(),
            "replication_engine":   self.get_replication_engine_status(),
            "brain_checkpoint":     self.get_brain_checkpoint_status(),
            "world_feed":           self.get_world_feed_status(),
            "drive_engine":         self.get_drive_engine_status(),
            # ── Phase 16 ──────────────────────────────────────────────────
            "self_narrative":       self.get_self_narrative_status(),
            "evolution_ethics":     self.get_evolution_ethics_status(),
            "memory_consolidator":  self.get_memory_consolidator_status(),
        }

    # ── Phase 15: Quality Engine API ──────────────────────────────────────

    def get_quality_engine_status(self) -> dict:
        if self.quality_engine is None:
            return {"enabled": False}
        try:
            return {"enabled": True, **self.quality_engine.summary()}
        except Exception:
            return {"enabled": True}

    def rate_data_quality(self, item: dict, source: str = "api") -> dict:
        """Rate a data item's quality (0-100)."""
        if self.quality_engine is None:
            return {"error": "QualityEngine not available"}
        return self.quality_engine.rate(item, source=source)

    # ── Phase 15: Immune System API ───────────────────────────────────────

    def get_immune_system_status(self) -> dict:
        if self.immune_system is None:
            return {"enabled": False}
        try:
            return {"enabled": True, **self.immune_system.summary()}
        except Exception:
            return {"enabled": True}

    def inspect_data(self, item: dict, source: str = "api") -> dict:
        """Run an item through the immune system inspection."""
        if self.immune_system is None:
            return {"error": "ImmuneSystem not available"}
        result = self.immune_system.inspect(item, source=source)
        if hasattr(result, "__dict__"):
            return result.__dict__
        return result if isinstance(result, dict) else {"status": str(result)}

    def trust_source(self, source_name: str) -> dict:
        """Mark a source as trusted."""
        if self.immune_system is None:
            return {"error": "ImmuneSystem not available"}
        self.immune_system.trust(source_name)
        return {"status": "trusted", "source": source_name}

    def blacklist_source(self, source_name: str) -> dict:
        """Blacklist a source."""
        if self.immune_system is None:
            return {"error": "ImmuneSystem not available"}
        self.immune_system.blacklist(source_name)
        return {"status": "blacklisted", "source": source_name}

    # ── Phase 15: Self-Replication API ────────────────────────────────────

    def get_replication_engine_status(self) -> dict:
        if self.replication_engine is None:
            return {"enabled": False}
        try:
            return {"enabled": True, **self.replication_engine.summary()}
        except Exception:
            return {"enabled": True}

    def run_replication_cycle(self) -> dict:
        """Trigger one self-replication / self-improvement cycle."""
        if self.replication_engine is None:
            return {"error": "SelfReplicationEngine not available"}
        return self.replication_engine.replicate(self)

    # ── Phase 15: Brain Checkpoint API ────────────────────────────────────

    def get_brain_checkpoint_status(self) -> dict:
        if self.brain_checkpoint is None:
            return {"enabled": False}
        return {"enabled": True, **self.brain_checkpoint.summary()}

    def save_checkpoint(self) -> dict:
        """Manually trigger a brain state save."""
        if self.brain_checkpoint is None:
            return {"error": "BrainCheckpoint not available"}
        path = self.brain_checkpoint.save(self)
        return {"status": "saved", "path": path}

    def load_checkpoint(self, path: Optional[str] = None) -> dict:
        """Load a brain state (latest by default)."""
        if self.brain_checkpoint is None:
            return {"error": "BrainCheckpoint not available"}
        return self.brain_checkpoint.load(path)

    def list_checkpoints(self) -> list:
        """List all available brain checkpoints."""
        if self.brain_checkpoint is None:
            return []
        return self.brain_checkpoint.list_checkpoints()

    def start_auto_checkpoint(self, interval_minutes: float = 10.0) -> dict:
        """Start periodic auto-save of brain state."""
        if self.brain_checkpoint is None:
            return {"error": "BrainCheckpoint not available"}
        self.brain_checkpoint.auto_save_start(interval_minutes=interval_minutes, mesh=self)
        return {"status": "started", "interval_minutes": interval_minutes}

    def stop_auto_checkpoint(self) -> dict:
        """Stop the periodic auto-save."""
        if self.brain_checkpoint is None:
            return {"error": "BrainCheckpoint not available"}
        self.brain_checkpoint.auto_save_stop()
        return {"status": "stopped"}

    # ── Phase 15: World Feed API ──────────────────────────────────────────

    def get_world_feed_status(self) -> dict:
        if self.world_feed is None:
            return {"enabled": False}
        return {"enabled": True, **self.world_feed.summary()}

    def start_world_feed(self, interval_s: float = 300.0) -> dict:
        """Start pulling data from real-world sources."""
        if self.world_feed is None:
            return {"error": "WorldFeed not available"}
        if self.world_feed._running:
            return {"status": "already_running", "stats": self.world_feed.get_feed_stats()}
        self.world_feed.start(interval_s=interval_s)
        return {"status": "started", "interval_s": interval_s, "sources": len(self.world_feed._sources)}

    def stop_world_feed(self) -> dict:
        """Stop the world feed."""
        if self.world_feed is None:
            return {"error": "WorldFeed not available"}
        self.world_feed.stop()
        return {"status": "stopped", "final_stats": self.world_feed.get_feed_stats()}

    def add_feed_source(self, url: str, source_type: str = "rss", name: Optional[str] = None) -> dict:
        """Add a new data source to the world feed."""
        if self.world_feed is None:
            return {"error": "WorldFeed not available"}
        self.world_feed.add_source(url=url, source_type=source_type, name=name)
        return {"status": "added", "url": url, "source_type": source_type}

    def get_recent_feed_items(self, n: int = 20) -> list:
        """Get the n most recently accepted feed items."""
        if self.world_feed is None:
            return []
        return self.world_feed.get_recent(n)

    def poll_world_feed_once(self) -> dict:
        """Manually trigger one polling cycle (blocking)."""
        if self.world_feed is None:
            return {"error": "WorldFeed not available"}
        self.world_feed.poll_once()
        return {"status": "polled", "stats": self.world_feed.get_feed_stats()}

    # ── Phase 15: Drive Engine API ────────────────────────────────────────

    def get_drive_engine_status(self) -> dict:
        if self.drive_engine is None:
            return {"enabled": False}
        return {"enabled": True, **self.drive_engine.summary()}

    def start_drive_engine(self, interval_s: float = 5.0) -> dict:
        """Start the internal motivation tick loop."""
        if self.drive_engine is None:
            return {"error": "DriveEngine not available"}
        if self.drive_engine._running:
            return {"status": "already_running"}
        self.drive_engine.start(interval_s=interval_s)
        return {"status": "started", "interval_s": interval_s}

    def stop_drive_engine(self) -> dict:
        """Stop the motivation tick loop."""
        if self.drive_engine is None:
            return {"error": "DriveEngine not available"}
        self.drive_engine.stop()
        return {"status": "stopped"}

    def get_drives(self) -> dict:
        """Get current state of all internal drives."""
        if self.drive_engine is None:
            return {"error": "DriveEngine not available"}
        return self.drive_engine.get_drives()

    def satisfy_drive(self, drive_name: str, amount: float = 0.35) -> dict:
        """Manually satisfy a specific drive."""
        if self.drive_engine is None:
            return {"error": "DriveEngine not available"}
        self.drive_engine.satisfy(drive_name, amount=amount)
        return {"status": "satisfied", "drive": drive_name, "amount": amount}

    def tick_drives(self) -> dict:
        """Manually advance drives by one tick."""
        if self.drive_engine is None:
            return {"error": "DriveEngine not available"}
        active = self.drive_engine.tick()
        return {"active_drives": active, "dominant": self.drive_engine.get_dominant_drive()}

    # ── Phase 16: Self-Narrative API ─────────────────────────────────────

    def get_self_narrative_status(self) -> dict:
        if self.self_narrative is None:
            return {"enabled": False}
        try:
            return {"enabled": True, **self.self_narrative.summary()}
        except Exception:
            return {"enabled": True}

    def record_narrative_event(
        self,
        event_type: str,
        data: Optional[dict] = None,
        surprise_score: float = 0.0,
        importance: float = 0.5,
    ) -> dict:
        """تسجيل حدث في السجل السردي للجهاز."""
        if self.self_narrative is None:
            return {"error": "SelfNarrative not available"}
        entry = self.self_narrative.record_event(
            event_type    = event_type,
            data          = data or {},
            surprise_score = surprise_score,
            importance    = importance,
        )
        return entry.to_dict() if hasattr(entry, 'to_dict') else {"status": "recorded"}

    def get_identity_statement(self) -> dict:
        """إرجاع جملة الهوية الحالية للجهاز."""
        if self.self_narrative is None:
            return {"error": "SelfNarrative not available"}
        s = self.self_narrative.summary()
        top = max(s.get("by_type", {"unknown": 1}), key=s.get("by_type", {"unknown": 1}).get)
        return {
            "identity": f"أنا جهاز عصبي رقمي سجّل {s.get('total',0)} حدثاً، أكثرها: {top}",
            "version":  self.VERSION,
            "summary":  s,
        }

    def get_narrative_log(self, n: int = 20) -> dict:
        """إرجاع آخر n حدث من السجل السردي."""
        if self.self_narrative is None:
            return {"error": "SelfNarrative not available"}
        return {
            "log":   self.self_narrative.get_log(n),
            "count": n,
        }

    def generate_daily_narrative(self) -> dict:
        """توليد ملخص سردي ليومي."""
        if self.self_narrative is None:
            return {"error": "SelfNarrative not available"}
        return {"diary": self.self_narrative.get_todays_diary()}

    def get_today_narrative(self) -> dict:
        """إرجاع سرد اليوم الحالي."""
        if self.self_narrative is None:
            return {"error": "SelfNarrative not available"}
        return {
            "diary":      self.self_narrative.get_todays_diary(),
            "log":        self.self_narrative.get_log(20),
            "summary":    self.self_narrative.summary(),
        }

    # ── Phase 16: Evolution Ethics API ───────────────────────────────────

    def get_evolution_ethics_status(self) -> dict:
        if self.evolution_ethics is None:
            return {"enabled": False}
        return {"enabled": True, **self.evolution_ethics.summary()}

    def ethics_check(self, action_type: str, params: Optional[dict] = None) -> dict:
        """فحص أخلاقي لفعل تطوري قبل تنفيذه."""
        if self.evolution_ethics is None:
            return {"allowed": True, "reason": "EvolutionEthics not available — defaulting to allow"}
        return self.evolution_ethics.check(action_type, params or {})

    def ethics_record(self, action: str, verdict: bool, reason: str, params: Optional[dict] = None) -> dict:
        """تسجيل قرار تطوري يدوياً مع مبرره الأخلاقي."""
        if self.evolution_ethics is None:
            return {"error": "EvolutionEthics not available"}
        self.evolution_ethics.record_decision(action, verdict, reason, params)
        return {"status": "recorded", "action": action, "verdict": verdict}

    def get_ethics_violations(self, limit: int = 50) -> dict:
        """إرجاع سجل الانتهاكات الأخلاقية."""
        if self.evolution_ethics is None:
            return {"error": "EvolutionEthics not available"}
        return {
            "violations": self.evolution_ethics.get_violations_log(limit),
            "blocked_sources": self.evolution_ethics.get_blocked_sources(),
        }

    def report_source_rejection(self, source: str) -> dict:
        """إبلاغ الضمير الأخلاقي برفض مصدر من الجهاز المناعي."""
        if self.evolution_ethics is None:
            return {"error": "EvolutionEthics not available"}
        return self.evolution_ethics.report_source_rejection(source)

    # ── Phase 16: Memory Consolidator API ────────────────────────────────

    def get_memory_consolidator_status(self) -> dict:
        if self.memory_consolidator is None:
            return {"enabled": False}
        return {"enabled": True, **self.memory_consolidator.summary()}

    def consolidate_memory(self) -> dict:
        """تشغيل دورة دمج فورية للذاكرة الدلالية."""
        if self.memory_consolidator is None:
            return {"error": "MemoryConsolidator not available"}
        return self.memory_consolidator.consolidate()

    def start_memory_consolidation(self, interval_minutes: float = 15.0) -> dict:
        """تشغيل الدمج الدوري في background thread."""
        if self.memory_consolidator is None:
            return {"error": "MemoryConsolidator not available"}
        if self.memory_consolidator._running:
            return {"status": "already_running"}
        self.memory_consolidator.start(interval_minutes=interval_minutes)
        return {"status": "started", "interval_minutes": interval_minutes}

    def stop_memory_consolidation(self) -> dict:
        """إيقاف الدمج الدوري."""
        if self.memory_consolidator is None:
            return {"error": "MemoryConsolidator not available"}
        self.memory_consolidator.stop()
        return {"status": "stopped"}

    def get_consolidated_laws(self, min_confidence: float = 0.0) -> dict:
        """إرجاع القوانين الدلالية المكتسبة."""
        if self.memory_consolidator is None:
            return {"error": "MemoryConsolidator not available"}
        laws = self.memory_consolidator.get_consolidated_laws(min_confidence)
        return {
            "laws":  laws,
            "count": len(laws),
            "min_confidence": min_confidence,
        }

    def observe_pattern(self, pattern_key: str, description: str = "", metadata: Optional[dict] = None) -> dict:
        """تسجيل ملاحظة نمط للدمج المستقبلي."""
        if self.memory_consolidator is None:
            return {"error": "MemoryConsolidator not available"}
        self.memory_consolidator.observe_pattern(pattern_key, description, metadata)
        count = self.memory_consolidator._local_pattern_counts.get(pattern_key, 0)
        return {"status": "observed", "pattern_key": pattern_key, "count": count}


    # ── Quran & Knowledge Feed API ─────────────────────────────────────────

    def store_quran(self, ayat: list) -> dict:
        """
        Store the full Quran in KnowledgeStore.
        ayat: list of {surah, ayah, text} dicts (6236 items).
        """
        self.knowledge.store_quran(ayat)
        stats = self.knowledge.quran_stats()
        return {"status": "stored", **stats}

    def get_ayah(self, surah: int, ayah: int) -> dict:
        """Retrieve a single ayah by surah and ayah number."""
        result = self.knowledge.get_ayah(surah, ayah)
        if result is None:
            return {"error": f"Ayah {surah}:{ayah} not found — run store_quran() first"}
        return result

    def get_surah(self, surah: int) -> dict:
        """Retrieve all ayahs of a surah."""
        ayat = self.knowledge.get_surah(surah)
        return {"surah": surah, "count": len(ayat), "ayat": ayat}

    def search_quran(self, query: str, max_results: int = 10) -> dict:
        """Search Quran text. Returns matching ayahs."""
        results = self.knowledge.search_quran(query, max_results=max_results)
        return {"query": query, "count": len(results), "results": results}

    def quran_stats(self) -> dict:
        """Quick statistics about the stored Quran."""
        try:
            return self.knowledge.quran_stats()
        except RuntimeError as e:
            return {"error": str(e)}

    def feed_quran_to_memory(self, batch_size: int = 100) -> dict:
        """
        Transfer stored Quran from KnowledgeStore into EpisodicMemoryEngine.
        Each ayah becomes a weighted memory episode.
        Longer surahs and early ayahs get slightly higher reward.
        Returns a report dict.
        """
        if self.episodic_memory is None:
            return {"error": "EpisodicMemoryEngine not available"}
        try:
            idx = self.knowledge.quran_index()
        except (KeyError, RuntimeError) as e:
            return {"error": f"Quran not stored yet — run store_quran() first: {e}"}

        total_chunks = idx["total_chunks"]
        recorded = 0
        skipped  = 0

        for ci in range(total_chunks):
            chunk = self.knowledge.read_custom(f"quran_chunk_{ci:04d}")
            if not chunk:
                continue
            for ayah_dict in chunk:
                try:
                    surah_num = int(ayah_dict.get("surah", 1))
                    ayah_num  = int(ayah_dict.get("ayah",  1))
                    text      = ayah_dict.get("text", "")
                    if not text:
                        skipped += 1
                        continue

                    # Feature vector: [surah_norm, ayah_norm, text_len_norm, reward, 0.5]
                    surah_norm   = surah_num / 114.0
                    ayah_norm    = min(ayah_num / 286.0, 1.0)   # longest surah=286
                    text_len_norm = min(len(text) / 300.0, 1.0)
                    reward       = 0.9 if surah_num <= 10 else 0.75

                    self.episodic_memory.record(
                        feature_vec = [surah_norm, ayah_norm, text_len_norm, reward, 0.5],
                        target      = reward,
                        outcome     = reward,
                        source      = f"quran:{surah_num}:{ayah_num}",
                        reward      = reward,
                        context     = {
                            "surah":      surah_num,
                            "ayah":       ayah_num,
                            "text":       text[:300],
                            "source_type": "quran",
                        },
                    )
                    recorded += 1

                    # Observe pattern in consolidator
                    if self.memory_consolidator is not None:
                        self.memory_consolidator.observe_pattern(
                            pattern_key = f"quran_surah_{surah_num}",
                            description = f"آيات السورة {surah_num} في الذاكرة",
                        )
                except Exception as _ep_err:
                    logger.debug(f"[feed_quran] skip ayah: {_ep_err}")
                    skipped += 1

        # Record narrative event
        if self.self_narrative is not None:
            self.self_narrative.record_event(
                event_type    = "world_knowledge",
                data          = {"message": f"تم تغذية {recorded} آية من القرآن الكريم للذاكرة"},
                surprise_score = 0.3,
                importance    = 1.0,
            )

        return {
            "status":   "done",
            "recorded": recorded,
            "skipped":  skipped,
            "total":    recorded + skipped,
        }

    def feed_text_to_memory(
        self,
        text: str,
        source: str = "manual",
        reward: float = 0.7,
        chunk_size: int = 200,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Feed any text (book, article, document) into EpisodicMemoryEngine.
        Splits text into chunks and records each as an episode.

        Parameters
        ----------
        text       : raw text to feed
        source     : source label (e.g. "book:sahih_bukhari")
        reward     : base reward value (0-1), higher = more important
        chunk_size : characters per chunk
        metadata   : extra context dict stored with every chunk
        """
        if self.episodic_memory is None:
            return {"error": "EpisodicMemoryEngine not available"}
        if not text or not text.strip():
            return {"error": "text is empty"}

        metadata  = metadata or {}
        chunks    = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        total     = len(chunks)
        recorded  = 0
        skipped   = 0

        for idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                skipped += 1
                continue
            try:
                position_norm = idx / max(total - 1, 1)
                text_len_norm = min(len(chunk) / chunk_size, 1.0)

                self.episodic_memory.record(
                    feature_vec = [reward, position_norm, text_len_norm, 0.5, 0.5],
                    target      = reward,
                    outcome     = reward,
                    source      = source,
                    reward      = reward,
                    context     = {
                        "text":       chunk,
                        "chunk_idx":  idx,
                        "total_chunks": total,
                        "source":     source,
                        **metadata,
                    },
                )

                if self.memory_consolidator is not None:
                    self.memory_consolidator.observe_pattern(
                        pattern_key = f"text_source:{source}",
                        description = f"نص من المصدر: {source}",
                    )
                recorded += 1
            except Exception as _e:
                logger.debug(f"[feed_text] chunk {idx} failed: {_e}")
                skipped += 1

        if self.self_narrative is not None:
            self.self_narrative.record_event(
                event_type    = "world_knowledge",
                data          = {"message": f"تم تغذية نص ({source}): {recorded} قطعة للذاكرة"},
                surprise_score = 0.2,
                importance    = reward,
            )

        return {
            "status":       "done",
            "source":       source,
            "recorded":     recorded,
            "skipped":      skipped,
            "total_chunks": total,
            "chunk_size":   chunk_size,
        }

    # ── Validation ─────────────────────────────────────────────────────────

    def validate_phase6(self, project_root: Optional[str] = None, save_report: bool = True) -> dict:
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
    print("  Neural Service Mesh  —  Demo (Autonomous Digital Nervous System)")
    print("="*65 + "\n")

    mesh = NeuralServiceMesh()

    inp  = mesh.register_node(InputNode("TextInput"))
    proc = mesh.register_node(ProcessorNode("TextProcessor"), connect_to=inp)
    out  = mesh.register_node(OutputNode("TextOutput", output_format="summary"), connect_to=proc)

    sample_text = (
        "Neural networks are inspired by the human brain. "
        "They consist of interconnected nodes that process information. "
        "Deep learning has achieved remarkable results in vision and language tasks."
    )

    print("[ Run 1 — Direct execution ]\n")
    r1 = mesh.run(inp, out, {"text": sample_text, "source": "demo"}, use_ai=True)
    print(f"  Status       : {r1['status']}")
    print(f"  Duration     : {r1['total_duration_ms']} ms")
    print(f"  AI Suggested : {r1['ai_suggested']}\n")

    print("[ Run 2 — Goal-based execution ]\n")
    r2 = mesh.run_goal(
        goal="Process and summarize text content",
        data={"text": sample_text, "source": "demo_goal"},
    )
    print(f"  Status    : {r2['status']}")
    print(f"  Duration  : {r2.get('total_duration_ms')} ms")
    plan = r2.get("goal_plan", {})
    print(f"  Goal      : {plan.get('goal')}")
    print(f"  Confidence: {plan.get('confidence')}")
    print(f"  Path      : {' → '.join(n[:8] for n in plan.get('path', []))}")
    print(f"  Reasoning : {plan.get('reasoning', [])[-1]}\n")

    print("[ Full System Status ]\n")
    print(json.dumps(mesh.get_full_system_status(), indent=2))


# ── Simulation ─────────────────────────────────────────────────────────────

def simulate(rounds: int = 20, delay: float = 0.1):
    from ai.simulation_engine import SimulationEngine
    mesh = NeuralServiceMesh()
    sim = SimulationEngine(mesh, validator=mesh.validator)
    results = sim.run_simulation(
        rounds=rounds,
        executions_per_round=5,
        delay_between_rounds=delay,
        verbose=True,
    )
    import json, os
    os.makedirs("./data", exist_ok=True)
    report_path = "./data/simulation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Simulation report saved to: {report_path}")
    return results


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Neural Service Mesh — Digital Nervous System")
    p.add_argument(
        "--mode",
        choices=[
            "demo", "api", "simulate", "evolve", "phase6",
            "validate", "evolve7",
            # ── Knowledge Sources Layer ──────────────────────────────
            "sources",        # list all registered sources
            "source-sync",    # sync all active sources
            "source-status",  # show source status + last sync results
        ],
        default="demo",
        help=(
            "demo: example pipeline | "
            "api: Flask server | "
            "simulate: learning simulation | "
            "evolve: autonomous evolution | "
            "phase6: multi-agent demo | "
            "validate: validation report | "
            "evolve7: full evolution pipeline | "
            "sources: list knowledge sources | "
            "source-sync: run knowledge source sync | "
            "source-status: show source health & last sync"
        ),
    )
    p.add_argument("--source-id", default=None, help="Target a specific source by ID")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--delay", type=float, default=0.1)
    p.add_argument("--cycles", type=int, default=3)
    args = p.parse_args()

    if args.mode == "demo":
        demo()
    elif args.mode == "simulate":
        simulate(rounds=args.rounds, delay=args.delay)
    elif args.mode == "evolve":
        import json
        mesh = NeuralServiceMesh()
        inp  = mesh.register_node(InputNode("TextInput"))
        proc = mesh.register_node(ProcessorNode("TextProcessor"), connect_to=inp)
        out  = mesh.register_node(OutputNode("TextOutput", output_format="summary"), connect_to=proc)
        result = mesh.evolve(cycles=args.cycles, auto_register=True)
        print(json.dumps(result, indent=2))
    elif args.mode == "phase6":
        import json
        mesh = NeuralServiceMesh()
        inp  = mesh.register_node(InputNode("TextInput"))
        proc = mesh.register_node(ProcessorNode("TextProcessor"), connect_to=inp)
        out  = mesh.register_node(OutputNode("TextOutput", output_format="summary"), connect_to=proc)
        sample_data = {"text": "The neural mesh is self-improving.", "source": "demo"}

        print("[ Agent Factory ]\n")
        agent = mesh.spawn_agent("ResearchAgent")
        print(f"  Spawned: {agent['role']} → {agent['agent_id']}\n")

        print("[ Swarm Execution ]\n")
        swarm_result = mesh.swarm_execute("translate and review document", sample_data)
        print(f"  Swarm status : {swarm_result['status']}")
        print(f"  Tasks run    : {swarm_result['total_tasks']}")
        print(f"  Tasks OK     : {swarm_result['success_count']}\n")

        print("[ Economic Leaderboard ]\n")
        board = mesh.economic_leaderboard()
        for entry in board["leaderboard"][:3]:
            print(f"  #{entry['rank']}  {entry['node_name']:20s}  composite={entry['composite_score']:.2f}")

        print("\n[ System DNA Snapshot ]\n")
        dna = mesh.dna_snapshot(notes="demo snapshot")
        print(f"  Snapshot : {dna['snapshot_id']}")
        print(f"  Health   : {dna['composite_health']}")
    elif args.mode == "validate":
        import json
        mesh = NeuralServiceMesh()
        inp  = mesh.register_node(InputNode("TextInput"))
        proc = mesh.register_node(ProcessorNode("TextProcessor"), connect_to=inp)
        out  = mesh.register_node(OutputNode("TextOutput", output_format="summary"), connect_to=proc)
        mesh.run(inp, out, {"text": "Validation pre-flight check.", "source": "validate"}, use_ai=True)
        mesh.spawn_agent("MonitorAgent")
        mesh.dna_snapshot(notes="Validation snapshot")
        report = mesh.validate_phase6(save_report=True)
        rd = report["phase7_readiness"]
        print(f"\n  READINESS SCORE: {rd['score']:.1f} / 100")
        print(f"  {rd['verdict']}\n")
    elif args.mode == "evolve7":
        import json
        mesh = NeuralServiceMesh()
        inp  = mesh.register_node(InputNode("TextInput"))
        proc = mesh.register_node(ProcessorNode("TextProcessor"), connect_to=inp)
        out  = mesh.register_node(OutputNode("TextOutput", output_format="summary"), connect_to=proc)
        mesh.run(inp, out, {"text": "Autonomous evolution test.", "source": "evolve7"}, use_ai=True)

        print("[ Self-Awareness Introspection ]\n")
        awareness = mesh.introspect()
        print(f"  Nodes        : {awareness['node_count']}")
        print(f"  Health Score : {awareness['system_health_score']:.2f}")

        print(f"\n[ Full Evolution Pipeline ({args.cycles} cycle(s)) ]\n")
        evolution = mesh.evolve7(cycles=args.cycles, verbose=True)
        for i, cycle in enumerate(evolution["cycles"]):
            s = cycle["summary"]
            print(f"  Cycle {i+1}: gaps={s['gaps_detected']} generated={s['modules_generated']} deployed={s['modules_deployed']}")

        print("\n[ Digital Nervous System Status ]\n")
        print(json.dumps(mesh.get_full_system_status(), indent=2))
    elif args.mode == "api":
        from api.app import run_api
        mesh = NeuralServiceMesh()
        run_api(mesh, host=args.host, port=args.port, debug=args.debug)

    # ── Knowledge Sources Layer Modes ─────────────────────────────────────

    elif args.mode == "sources":
        import json
        from knowledge_sources import SourceManager
        from knowledge_sources.quran.quran_source import create_quran_source

        sm = SourceManager()
        meta, feeder = create_quran_source()
        sm.register_source(meta, feeder)

        print("\n" + "="*65)
        print("  Knowledge Sources Registry")
        print("="*65)
        sources = sm.list_sources()
        for s in sources:
            print(f"\n  [{s['status'].upper()}] {s['name']}")
            print(f"    ID          : {s['id']}")
            print(f"    Type        : {s['source_type']}")
            print(f"    Trust Score : {s['trust_score']}")
            print(f"    Access Mode : {s['access_mode']}")
            print(f"    Frequency   : {s['update_frequency']}")
            print(f"    Registered  : {s['registered_at']}")
        print(f"\n  Total Sources: {len(sources)}")
        print(json.dumps(sm.summary(), indent=2))

    elif args.mode == "source-sync":
        import json
        from knowledge_sources import SourceManager
        from knowledge_sources.quran.quran_source import create_quran_source

        mesh = NeuralServiceMesh()
        sm   = SourceManager(min_quality_threshold=30.0)
        sm.set_knowledge_store(mesh.knowledge)
        sm.set_environment_model(mesh.env_model)
        sm.set_semantic_matcher(mesh.semantic)

        meta, feeder = create_quran_source(max_items=50)
        sm.register_source(meta, feeder)

        target = args.source_id if hasattr(args, "source_id") and args.source_id else meta.id
        print(f"\n  Syncing source: {target}")
        result = sm.sync_source(target)

        print("\n" + "="*65)
        print("  Knowledge Source Sync Complete")
        print("="*65)
        print(f"  Source      : {result.source_name}")
        print(f"  Fetched     : {result.items_fetched}")
        print(f"  Validated   : {result.items_validated}")
        print(f"  Ingested    : {result.items_ingested}")
        print(f"  Rejected    : {result.items_rejected}")
        print(f"  Avg Quality : {result.avg_quality:.1f}/100")
        print(f"  Success     : {result.success}")
        if result.errors:
            print(f"  Errors      : {result.errors}")

    elif args.mode == "source-status":
        import json
        from knowledge_sources import SourceManager
        from knowledge_sources.quran.quran_source import create_quran_source

        sm = SourceManager()
        meta, feeder = create_quran_source()
        sm.register_source(meta, feeder)

        print("\n" + "="*65)
        print("  Knowledge Sources — Status Report")
        print("="*65)
        status = sm.source_status(meta.id)
        print(json.dumps(status, indent=2, ensure_ascii=False))
        print("\n  System Summary:")
        print(json.dumps(sm.summary(), indent=2))
