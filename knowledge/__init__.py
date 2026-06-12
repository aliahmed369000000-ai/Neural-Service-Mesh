"""
Knowledge layer package for Neural Service Mesh v3.
Provides persistent JSON-based knowledge storage:
  - node_profiles.json    : node capabilities + semantic metadata
  - route_memory.json     : route history, scores, execution records
  - graph_metrics.json    : graph statistics, rankings, optimization metrics
"""
from .knowledge_store import KnowledgeStore

__all__ = ["KnowledgeStore"]
