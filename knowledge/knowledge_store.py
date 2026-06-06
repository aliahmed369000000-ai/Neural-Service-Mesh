"""
KnowledgeStore — Persistent JSON Knowledge Layer (Phase 3)
===========================================================
Manages three knowledge files:
  knowledge/node_profiles.json   – node capabilities + semantic metadata
  knowledge/route_memory.json    – route history, scores, execution records
  knowledge/graph_metrics.json   – graph stats, node/route rankings, optimisation metrics

Design principles:
  - Auto-creates files with correct schema on first startup.
  - Thread-safe file I/O via a per-file lock.
  - Every write is atomic (write tmp → rename).
  - All AI modules call read_*() / write_*() helpers; they never touch the files directly.
  - Fully backward-compatible: existing SQLite-based modules continue to work unchanged;
    the KnowledgeStore is an *additional* persistence layer, not a replacement.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Sentinel defaults ──────────────────────────────────────────────────────

_DEFAULT_NODE_PROFILES: Dict[str, Any] = {
    "_meta": {
        "schema_version": "1.0.0",
        "created_at": None,       # filled on first write
        "last_updated": None,
        "total_nodes": 0,
        "description": "Node capabilities and semantic metadata registry",
    },
    "nodes": {}
    # nodes[node_id] = {
    #   "node_id", "name", "description", "capability",
    #   "input_schema", "output_schema", "tags", "version",
    #   "announced_at", "is_active",
    #   "semantic_profile": {
    #       "input_tokens", "output_tokens", "capability_tokens",
    #       "synonym_groups"
    #   },
    #   "execution_stats": {
    #       "total_executions", "successes", "failures",
    #       "avg_latency_ms", "last_executed"
    #   },
    #   "discovery_score": float   (0-100, how often this node is used in routing)
    # }
}

_DEFAULT_ROUTE_MEMORY: Dict[str, Any] = {
    "_meta": {
        "schema_version": "1.0.0",
        "created_at": None,
        "last_updated": None,
        "total_routes": 0,
        "description": "Successful and failed route history with scores and execution records",
    },
    "routes": {},
    # routes[path_key] = {
    #   "path_key", "path", "runs", "successes", "failures",
    #   "success_rate", "avg_latency_ms", "memory_score",
    #   "health", "is_promoted", "first_seen", "last_seen",
    #   "execution_history": [ {run_id, ts, success, latency_ms} ]  (last 50)
    # }
    "failed_routes": {},
    # failed_routes[path_key] = same shape, health=critical
    "statistics": {
        "total_runs": 0,
        "total_successes": 0,
        "total_failures": 0,
        "promoted_routes": 0,
        "critical_routes": 0,
        "avg_memory_score": 0.0,
    }
}

_DEFAULT_GRAPH_METRICS: Dict[str, Any] = {
    "_meta": {
        "schema_version": "1.0.0",
        "created_at": None,
        "last_updated": None,
        "description": "Graph statistics, node rankings, route rankings and optimisation metrics",
    },
    "graph_statistics": {
        "total_nodes": 0,
        "total_edges": 0,
        "avg_degree": 0.0,
        "density": 0.0,
        "connected_components": 0,
        "last_computed": None,
    },
    "node_rankings": {
        "by_success_rate": [],     # [{node_id, name, success_rate, executions}]
        "by_usage_count": [],      # [{node_id, name, executions}]
        "by_avg_latency": [],      # [{node_id, name, avg_latency_ms}]
        "bottleneck_nodes": [],    # nodes with high usage AND high failure rate
        "idle_nodes": [],          # nodes never executed
    },
    "route_rankings": {
        "top_routes": [],          # [{path_key, path, memory_score, runs}]
        "failed_routes": [],       # [{path_key, path, health, failure_rate}]
        "most_used": [],           # [{path_key, path, runs}]
    },
    "optimization_metrics": {
        "total_optimization_runs": 0,
        "last_optimization": None,
        "actions_history": [],     # last 20 optimization reports summary
        "improvements": {
            "edges_pruned": 0,
            "edges_promoted": 0,
            "edges_suggested": 0,
            "weights_updated": 0,
        },
        "health_trend": [],        # [{ts, avg_score, total_nodes, total_edges}]
    },
    "connection_scores": {
        "top_connections": [],
        "worst_connections": [],
        "last_updated": None,
    }
}


# ── Atomic file writer ─────────────────────────────────────────────────────

def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON atomically: write to .tmp then rename."""
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── KnowledgeStore ─────────────────────────────────────────────────────────

class KnowledgeStore:
    """
    Central knowledge persistence layer.

    Usage
    -----
    ks = KnowledgeStore(knowledge_dir="./knowledge")
    # Then inject into AI modules:
    memory_engine.set_knowledge_store(ks)
    discovery_engine.set_knowledge_store(ks)
    routing_engine.set_knowledge_store(ks)
    optimization_engine.set_knowledge_store(ks)
    """

    NODE_PROFILES  = "node_profiles.json"
    ROUTE_MEMORY   = "route_memory.json"
    GRAPH_METRICS  = "graph_metrics.json"

    def __init__(self, knowledge_dir: str = "./knowledge"):
        self._dir = Path(knowledge_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        # Per-file locks for thread safety
        self._locks: Dict[str, threading.Lock] = {
            self.NODE_PROFILES: threading.Lock(),
            self.ROUTE_MEMORY:  threading.Lock(),
            self.GRAPH_METRICS: threading.Lock(),
        }

        # In-memory cache {filename: dict}
        self._cache: Dict[str, dict] = {}

        # Initialise all three files
        self._init_file(self.NODE_PROFILES,  _DEFAULT_NODE_PROFILES)
        self._init_file(self.ROUTE_MEMORY,   _DEFAULT_ROUTE_MEMORY)
        self._init_file(self.GRAPH_METRICS,  _DEFAULT_GRAPH_METRICS)

        logger.info(f"KnowledgeStore initialised at '{self._dir}' (Phase 3 knowledge layer)")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _path(self, filename: str) -> Path:
        return self._dir / filename

    def _init_file(self, filename: str, default: dict) -> None:
        """Create the file with default schema if it doesn't exist."""
        p = self._path(filename)
        if not p.exists():
            data = deepcopy(default)
            data["_meta"]["created_at"] = _now_iso()
            data["_meta"]["last_updated"] = _now_iso()
            _atomic_write(p, data)
            logger.info(f"KnowledgeStore: created '{filename}'")
        # Load into cache
        self._cache[filename] = self._read_raw(filename)

    def _read_raw(self, filename: str) -> dict:
        """Read file from disk (bypasses cache)."""
        p = self._path(filename)
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"KnowledgeStore: failed to read '{filename}': {e} — resetting")
            # Reset to default
            defaults = {
                self.NODE_PROFILES: _DEFAULT_NODE_PROFILES,
                self.ROUTE_MEMORY:  _DEFAULT_ROUTE_MEMORY,
                self.GRAPH_METRICS: _DEFAULT_GRAPH_METRICS,
            }
            data = deepcopy(defaults[filename])
            data["_meta"]["created_at"] = _now_iso()
            data["_meta"]["last_updated"] = _now_iso()
            _atomic_write(p, data)
            return data

    def _read(self, filename: str) -> dict:
        """Return a deep-copy from cache (safe for mutation)."""
        with self._locks[filename]:
            return deepcopy(self._cache[filename])

    def _write(self, filename: str, data: dict) -> None:
        """Persist data to disk and update cache."""
        data["_meta"]["last_updated"] = _now_iso()
        with self._locks[filename]:
            _atomic_write(self._path(filename), data)
            self._cache[filename] = deepcopy(data)

    # ══════════════════════════════════════════════════════════════════════
    # NODE PROFILES  (node_profiles.json)
    # ══════════════════════════════════════════════════════════════════════

    def read_node_profiles(self) -> Dict[str, Any]:
        """Return full node profiles dict."""
        return self._read(self.NODE_PROFILES)

    def get_node_profile(self, node_id: str) -> Optional[dict]:
        """Return single node profile or None."""
        return self._read(self.NODE_PROFILES)["nodes"].get(node_id)

    def upsert_node_profile(self, node_id: str, profile: dict) -> None:
        """
        Insert or update a node profile.
        Called by DiscoveryEngine on every announce().
        """
        data = self._read(self.NODE_PROFILES)
        existing = data["nodes"].get(node_id, {})
        merged = {**existing, **profile}
        merged["node_id"] = node_id
        merged["last_seen"] = _now_iso()
        if "first_seen" not in merged:
            merged["first_seen"] = _now_iso()

        # Ensure sub-dicts exist
        if "execution_stats" not in merged:
            merged["execution_stats"] = {
                "total_executions": 0,
                "successes": 0,
                "failures": 0,
                "avg_latency_ms": 0.0,
                "last_executed": None,
            }
        if "semantic_profile" not in merged:
            merged["semantic_profile"] = {
                "input_tokens": [],
                "output_tokens": [],
                "capability_tokens": [],
            }
        if "discovery_score" not in merged:
            merged["discovery_score"] = 0.0

        data["nodes"][node_id] = merged
        data["_meta"]["total_nodes"] = len(data["nodes"])
        self._write(self.NODE_PROFILES, data)
        logger.debug(f"KnowledgeStore: upserted node profile [{node_id[:8]}]")

    def update_node_execution_stats(self, node_id: str, success: bool,
                                    latency_ms: float) -> None:
        """
        Update execution statistics for a node profile.
        Called by MemoryEngine after every run.
        """
        data = self._read(self.NODE_PROFILES)
        node = data["nodes"].get(node_id)
        if not node:
            return  # Node not yet announced; skip

        stats = node.setdefault("execution_stats", {
            "total_executions": 0, "successes": 0, "failures": 0,
            "avg_latency_ms": 0.0, "last_executed": None,
        })
        stats["total_executions"] = stats.get("total_executions", 0) + 1
        if success:
            stats["successes"] = stats.get("successes", 0) + 1
        else:
            stats["failures"] = stats.get("failures", 0) + 1

        n = stats["total_executions"]
        old_avg = stats.get("avg_latency_ms", 0.0)
        stats["avg_latency_ms"] = round(old_avg + (latency_ms - old_avg) / n, 3)
        stats["last_executed"] = _now_iso()
        node["execution_stats"] = stats

        # Update discovery_score: uses (success_rate * 60) + (log-usage * 40)
        import math
        sr = stats["successes"] / max(stats["total_executions"], 1)
        usage = (math.log(stats["total_executions"] + 1) / math.log(101)) * 40.0
        node["discovery_score"] = round(min(100.0, sr * 60.0 + usage), 2)

        data["nodes"][node_id] = node
        self._write(self.NODE_PROFILES, data)

    def update_node_semantic_profile(self, node_id: str,
                                     input_tokens: List[str],
                                     output_tokens: List[str],
                                     capability_tokens: List[str]) -> None:
        """
        Update the semantic token profile of a node.
        Called by DiscoveryEngine / SemanticMatcher.
        """
        data = self._read(self.NODE_PROFILES)
        node = data["nodes"].get(node_id)
        if not node:
            return
        node["semantic_profile"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "capability_tokens": capability_tokens,
            "updated_at": _now_iso(),
        }
        data["nodes"][node_id] = node
        self._write(self.NODE_PROFILES, data)

    def deactivate_node_profile(self, node_id: str) -> None:
        data = self._read(self.NODE_PROFILES)
        if node_id in data["nodes"]:
            data["nodes"][node_id]["is_active"] = False
            data["nodes"][node_id]["deactivated_at"] = _now_iso()
            self._write(self.NODE_PROFILES, data)

    def list_active_node_profiles(self) -> List[dict]:
        data = self._read(self.NODE_PROFILES)
        return [n for n in data["nodes"].values() if n.get("is_active", True)]

    # ══════════════════════════════════════════════════════════════════════
    # ROUTE MEMORY  (route_memory.json)
    # ══════════════════════════════════════════════════════════════════════

    def read_route_memory(self) -> Dict[str, Any]:
        return self._read(self.ROUTE_MEMORY)

    def get_route(self, path_key: str) -> Optional[dict]:
        data = self._read(self.ROUTE_MEMORY)
        return data["routes"].get(path_key) or data["failed_routes"].get(path_key)

    def upsert_route(self, route_data: dict) -> None:
        """
        Insert or update a route record.
        Called by MemoryEngine.learn_from_run() after every execution.
        route_data must include: path_key, path, runs, successes, failures,
          success_rate, avg_latency_ms, memory_score, health, is_promoted,
          first_seen, last_seen.
        """
        data = self._read(self.ROUTE_MEMORY)
        path_key = route_data["path_key"]

        # Determine which bucket
        health = route_data.get("health", "good")
        is_critical = health == "critical"

        # Merge execution_history
        existing = (data["routes"].get(path_key) or
                    data["failed_routes"].get(path_key) or {})
        history = existing.get("execution_history", [])

        new_record = dict(route_data)
        new_record["execution_history"] = history  # preserved

        if is_critical:
            data["failed_routes"][path_key] = new_record
            data["routes"].pop(path_key, None)  # move out of main if was there
        else:
            data["routes"][path_key] = new_record
            data["failed_routes"].pop(path_key, None)

        # Update global statistics
        all_routes = list(data["routes"].values()) + list(data["failed_routes"].values())
        stats = data["statistics"]
        stats["total_routes"] = len(all_routes)
        stats["total_runs"]      = sum(r.get("runs", 0) for r in all_routes)
        stats["total_successes"] = sum(r.get("successes", 0) for r in all_routes)
        stats["total_failures"]  = sum(r.get("failures", 0) for r in all_routes)
        stats["promoted_routes"] = sum(1 for r in all_routes if r.get("is_promoted"))
        stats["critical_routes"] = len(data["failed_routes"])
        scores = [r.get("memory_score", 0) for r in all_routes if r.get("runs", 0) > 0]
        stats["avg_memory_score"] = round(sum(scores) / len(scores), 2) if scores else 0.0
        data["statistics"] = stats
        data["_meta"]["total_routes"] = len(all_routes)

        self._write(self.ROUTE_MEMORY, data)
        logger.debug(f"KnowledgeStore: upserted route [{path_key}] health={health}")

    def append_route_execution(self, path_key: str,
                                run_id: str,
                                success: bool,
                                latency_ms: float) -> None:
        """
        Append a single execution record to a route's history (max 50 kept).
        Called by MemoryEngine.learn_from_run().
        """
        data = self._read(self.ROUTE_MEMORY)
        route = data["routes"].get(path_key) or data["failed_routes"].get(path_key)
        if not route:
            return

        record = {
            "run_id": run_id[:8] if run_id else "?",
            "ts": _now_iso(),
            "success": success,
            "latency_ms": round(latency_ms, 2),
        }
        history = route.setdefault("execution_history", [])
        history.append(record)
        route["execution_history"] = history[-50:]  # keep last 50

        bucket = "routes" if route.get("health", "good") != "critical" else "failed_routes"
        data[bucket][path_key] = route
        self._write(self.ROUTE_MEMORY, data)

    def promote_route(self, path_key: str) -> None:
        data = self._read(self.ROUTE_MEMORY)
        for bucket in ("routes", "failed_routes"):
            if path_key in data[bucket]:
                data[bucket][path_key]["is_promoted"] = True
                data[bucket][path_key]["promoted_at"] = _now_iso()
        self._write(self.ROUTE_MEMORY, data)

    def demote_route(self, path_key: str) -> None:
        data = self._read(self.ROUTE_MEMORY)
        for bucket in ("routes", "failed_routes"):
            if path_key in data[bucket]:
                data[bucket][path_key]["is_promoted"] = False
        self._write(self.ROUTE_MEMORY, data)

    def get_best_routes(self, top_k: int = 10) -> List[dict]:
        data = self._read(self.ROUTE_MEMORY)
        routes = list(data["routes"].values())
        routes.sort(key=lambda r: r.get("memory_score", 0), reverse=True)
        return routes[:top_k]

    def get_failed_routes(self) -> List[dict]:
        data = self._read(self.ROUTE_MEMORY)
        return list(data["failed_routes"].values())

    def route_statistics(self) -> dict:
        return self._read(self.ROUTE_MEMORY)["statistics"]

    # ══════════════════════════════════════════════════════════════════════
    # GRAPH METRICS  (graph_metrics.json)
    # ══════════════════════════════════════════════════════════════════════

    def read_graph_metrics(self) -> Dict[str, Any]:
        return self._read(self.GRAPH_METRICS)

    def update_graph_statistics(self, total_nodes: int, total_edges: int,
                                  avg_degree: float = 0.0,
                                  density: float = 0.0,
                                  connected_components: int = 0) -> None:
        """
        Update high-level graph topology metrics.
        Called by OptimizationEngine.analyze().
        """
        data = self._read(self.GRAPH_METRICS)
        data["graph_statistics"] = {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "avg_degree": round(avg_degree, 4),
            "density": round(density, 6),
            "connected_components": connected_components,
            "last_computed": _now_iso(),
        }
        self._write(self.GRAPH_METRICS, data)

    def update_node_rankings(self, memory_engine) -> None:
        """
        Recompute node rankings from MemoryEngine data and persist.
        Called by OptimizationEngine.analyze().
        """
        data = self._read(self.GRAPH_METRICS)

        best = memory_engine.best_nodes(n=50)
        worst = memory_engine.worst_nodes(n=50)
        all_nodes = [nm for nm in memory_engine._nodes.values()]

        by_sr = sorted(
            [{"node_id": n["node_id"], "name": n["name"],
              "success_rate": n["success_rate"],
              "executions": n["executions"]} for n in best],
            key=lambda x: x["success_rate"], reverse=True
        )
        by_usage = sorted(
            [{"node_id": n.node_id, "name": n.name,
              "executions": n.executions} for n in all_nodes],
            key=lambda x: x["executions"], reverse=True
        )
        by_latency = sorted(
            [{"node_id": n.node_id, "name": n.name,
              "avg_latency_ms": round(n.avg_latency_ms, 2)} for n in all_nodes],
            key=lambda x: x["avg_latency_ms"]
        )
        bottlenecks = [
            {"node_id": n["node_id"], "name": n["name"],
             "success_rate": n["success_rate"],
             "executions": n["executions"]}
            for n in worst
            if n["executions"] >= 3 and n["success_rate"] < 0.5
        ]
        idle = [
            {"node_id": n.node_id, "name": n.name}
            for n in all_nodes if n.executions == 0
        ]

        data["node_rankings"] = {
            "by_success_rate":  by_sr[:20],
            "by_usage_count":   by_usage[:20],
            "by_avg_latency":   by_latency[:20],
            "bottleneck_nodes": bottlenecks[:10],
            "idle_nodes":       idle[:20],
            "last_updated":     _now_iso(),
        }
        self._write(self.GRAPH_METRICS, data)

    def update_route_rankings(self, memory_engine) -> None:
        """
        Recompute route rankings from MemoryEngine data.
        Called by OptimizationEngine.analyze().
        """
        data = self._read(self.GRAPH_METRICS)
        all_routes = [rm.to_dict() for rm in memory_engine._routes.values()]

        top = sorted(all_routes, key=lambda r: r.get("memory_score", 0), reverse=True)
        failed = [r for r in all_routes if r.get("health") == "critical"]
        most_used = sorted(all_routes, key=lambda r: r.get("runs", 0), reverse=True)

        data["route_rankings"] = {
            "top_routes":    top[:10],
            "failed_routes": failed[:10],
            "most_used":     most_used[:10],
            "last_updated":  _now_iso(),
        }
        self._write(self.GRAPH_METRICS, data)

    def record_optimization_run(self, report_dict: dict) -> None:
        """
        Persist a summary of an optimization run.
        Called by OptimizationEngine.analyze().
        """
        data = self._read(self.GRAPH_METRICS)
        opt = data["optimization_metrics"]
        opt["total_optimization_runs"] = opt.get("total_optimization_runs", 0) + 1
        opt["last_optimization"] = _now_iso()

        # Aggregate improvement counters
        from ai.optimization_engine import OptimizationAction  # local import to avoid circular
        for action in report_dict.get("actions", []):
            t = action.get("action_type", "")
            impr = opt.setdefault("improvements", {
                "edges_pruned": 0, "edges_promoted": 0,
                "edges_suggested": 0, "weights_updated": 0,
            })
            if t == OptimizationAction.PRUNE_EDGE:
                impr["edges_pruned"] = impr.get("edges_pruned", 0) + 1
            elif t == OptimizationAction.PROMOTE_EDGE:
                impr["edges_promoted"] = impr.get("edges_promoted", 0) + 1
            elif t == OptimizationAction.SUGGEST_EDGE:
                impr["edges_suggested"] = impr.get("edges_suggested", 0) + 1
            elif t == OptimizationAction.UPDATE_WEIGHT:
                impr["weights_updated"] = impr.get("weights_updated", 0) + 1
            opt["improvements"] = impr

        # Append summary to history (last 20)
        summary = {
            "run": opt["total_optimization_runs"],
            "ts": _now_iso(),
            "total_actions": report_dict.get("total_actions", 0),
            "applied_count": report_dict.get("applied_count", 0),
            "action_counts": report_dict.get("summary", {}).get("action_counts", {}),
        }
        history = opt.get("actions_history", [])
        history.append(summary)
        opt["actions_history"] = history[-20:]
        data["optimization_metrics"] = opt
        self._write(self.GRAPH_METRICS, data)

    def append_health_snapshot(self, avg_connection_score: float,
                                total_nodes: int, total_edges: int) -> None:
        """
        Append a health trend snapshot.
        Called by OptimizationEngine after every analysis.
        """
        data = self._read(self.GRAPH_METRICS)
        opt = data["optimization_metrics"]
        trend = opt.get("health_trend", [])
        trend.append({
            "ts": _now_iso(),
            "avg_score": round(avg_connection_score, 2),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
        })
        opt["health_trend"] = trend[-100:]   # keep last 100 snapshots
        data["optimization_metrics"] = opt
        self._write(self.GRAPH_METRICS, data)

    def update_connection_scores(self, scoring_engine) -> None:
        """
        Snapshot top/worst connections from ScoringEngine.
        Called by OptimizationEngine.analyze().
        """
        data = self._read(self.GRAPH_METRICS)
        data["connection_scores"] = {
            "top_connections":   scoring_engine.top_connections(n=15),
            "worst_connections": scoring_engine.worst_connections(n=15),
            "last_updated":      _now_iso(),
        }
        self._write(self.GRAPH_METRICS, data)

    # ── Convenience: full status ───────────────────────────────────────────

    def summary(self) -> dict:
        np_data = self._read(self.NODE_PROFILES)
        rm_data = self._read(self.ROUTE_MEMORY)
        gm_data = self._read(self.GRAPH_METRICS)
        return {
            "node_profiles": {
                "total_nodes": np_data["_meta"]["total_nodes"],
                "active_nodes": sum(1 for n in np_data["nodes"].values()
                                    if n.get("is_active", True)),
                "last_updated": np_data["_meta"]["last_updated"],
            },
            "route_memory": rm_data["statistics"],
            "graph_metrics": {
                "graph_statistics": gm_data["graph_statistics"],
                "total_optimization_runs": (
                    gm_data["optimization_metrics"]["total_optimization_runs"]
                ),
                "last_optimization": (
                    gm_data["optimization_metrics"]["last_optimization"]
                ),
            },
        }

    # ── Phase 5: Custom key-value knowledge storage ────────────────────────

    def read_custom(self, key: str) -> Any:
        """
        Phase 5: Read arbitrary data stored under a custom key.
        Stored in knowledge/<key>.json
        """
        import json
        custom_path = self._dir / f"{key}.json"
        if not custom_path.exists():
            raise KeyError(f"Custom knowledge key not found: '{key}'")
        with custom_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def write_custom(self, key: str, data: Any) -> None:
        """
        Phase 5: Write arbitrary data under a custom key.
        Stored atomically in knowledge/<key>.json
        """
        import json, tempfile, os
        custom_path = self._dir / f"{key}.json"
        tmp_path = custom_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(str(tmp_path), str(custom_path))

    def __repr__(self):
        s = self.summary()
        return (
            f"<KnowledgeStore "
            f"nodes={s['node_profiles']['total_nodes']} "
            f"routes={s['route_memory'].get('total_routes', 0)} "
            f"opt_runs={s['graph_metrics']['total_optimization_runs']}>"
        )
