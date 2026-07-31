# Phase 2
from ai.decision import AIDecisionLayer

# Phase 3
from ai.semantic_matcher import SemanticMatcher, NodeSemanticProfile
from ai.scoring_engine import ScoringEngine, ConnectionScore
from ai.memory_engine import MemoryEngine, RouteMemory
from ai.discovery_engine import DiscoveryEngine, NodeAnnouncement
from ai.routing_engine import RoutingEngine, RouteCandidate
from ai.goal_planner import GoalPlanner, ExecutionPlan
from ai.optimization_engine import OptimizationEngine, OptimizationReport

# Phase 4
from ai.reputation_engine import NodeReputation, NodeReputationEngine

# Phase 5
from ai.gap_detector import DetectedGap, GapDetectionEngine
from ai.multi_goal_planner import SubGoal, MultiGoalPlan, MultiGoalPlanner
from ai.service_generator import GeneratedServiceSpec, ServiceGeneratorEngine

__all__ = [
    # Phase 2
    "AIDecisionLayer",
    # Phase 3
    "SemanticMatcher", "NodeSemanticProfile",
    "ScoringEngine", "ConnectionScore",
    "MemoryEngine", "RouteMemory",
    "DiscoveryEngine", "NodeAnnouncement",
    "RoutingEngine", "RouteCandidate",
    "GoalPlanner", "ExecutionPlan",
    "OptimizationEngine", "OptimizationReport",
    # Phase 4
    "NodeReputation", "NodeReputationEngine",
    # Phase 5
    "DetectedGap", "GapDetectionEngine",
    "SubGoal", "MultiGoalPlan", "MultiGoalPlanner",
    "GeneratedServiceSpec", "ServiceGeneratorEngine",
]


# Phase 6
from ai.agent_factory import AgentFactory, AgentInstance
from ai.swarm_coordinator import SwarmCoordinator, SwarmResult
from ai.self_optimizer import SelfOptimizer, SelfOptimizerReport
from ai.simulation_lab import SimulationLab, SimulationReport
from ai.meta_reasoner import MetaReasoner, DecisionExplanation, MetaReasonerInsight
from ai.economic_engine import EconomicEngine, NodeEconomicProfile
from ai.system_dna import SystemDNA, DNASnapshot

# Phase 6 — Validator
from ai.validator import Phase6Validator

# Phase 7 — Strategic Objectives
from ai.objectives import Objective, ObjectivesEngine

# Self-Narrative Engine
from ai.self_narrative import NarrativeEntry, DailySummary, SelfNarrative

# Phase 8 — Real Neural Weights
from ai.neural_weights import NeuralWeightLayer, extract_routing_weights, get_default_layer

__all__ += [
    # Phase 6
    "AgentFactory", "AgentInstance",
    "SwarmCoordinator", "SwarmResult",
    "SelfOptimizer", "SelfOptimizerReport",
    "SimulationLab", "SimulationReport",
    "MetaReasoner", "DecisionExplanation", "MetaReasonerInsight",
    "EconomicEngine", "NodeEconomicProfile",
    "SystemDNA", "DNASnapshot",
    "Phase6Validator",
    # Phase 7
    "Objective", "ObjectivesEngine",
    # Self-Narrative
    "NarrativeEntry", "DailySummary", "SelfNarrative",
    # Phase 8
    "NeuralWeightLayer", "extract_routing_weights", "get_default_layer",
]
