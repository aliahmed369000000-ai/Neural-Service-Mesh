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
]


# Phase 6
from ai.agent_factory import AgentFactory, AgentInstance
from ai.swarm_coordinator import SwarmCoordinator, SwarmResult
from ai.self_optimizer import SelfOptimizer, SelfOptimizerReport
from ai.simulation_lab import SimulationLab, SimulationReport
from ai.meta_reasoner import MetaReasoner, DecisionExplanation, MetaReasonerInsight
from ai.economic_engine import EconomicEngine, NodeEconomicProfile
from ai.system_dna import SystemDNA, DNASnapshot
