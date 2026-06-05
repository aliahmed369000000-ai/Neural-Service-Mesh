from __future__ import annotations
import sys
from typing import Optional

# ── Logging first ──────────────────────────────────────────────────────────
from logs.mesh_logger import MeshLogger
_mesh_logger = MeshLogger(log_dir="./logs", level="INFO")

import logging
logger = logging.getLogger("NeuralServiceMesh.v2")

# ── Core imports ───────────────────────────────────────────────────────────
from storage.file_storage import FileStorage
from storage.db import SQLiteStorage
from core.registry import NodeRegistry
from core.graph import ServiceGraph
from core.engine import ExecutionEngine
from connectors.data_transformer import DataTransformer
from ai.decision import AIDecisionLayer
from services.input_service import InputNode
from services.processor_service import ProcessorNode
from services.output_service import OutputNode


class NeuralServiceMesh:
    """
    Phase 2: NeuralServiceMesh with:
      - Flask REST API Layer
      - SQLite persistent storage
      - AI Decision Layer (rules + heuristics)
      - Enhanced Execution Engine (fallback, multi-path, logging)
      - Structured logging to /logs/
    """

    VERSION = "2.0.0"

    def __init__(self, storage_dir: str = "./data", db_path: str = "./data/mesh.db"):
        # Storage
        self.storage = FileStorage(storage_dir)
        self.db = SQLiteStorage(db_path)

        # Core
        self.registry = NodeRegistry(self.storage)
        self.graph = ServiceGraph()
        self.transformer = DataTransformer()

        # AI Layer
        self.ai = AIDecisionLayer()

        # Engine (Phase 2: wired with db + ai)
        self.engine = ExecutionEngine(
            registry=self.registry,
            graph=self.graph,
            storage=self.storage,
            transformer=self.transformer,
            db=self.db,
            ai=self.ai,
        )

        # Wire AI to graph
        self.ai.set_graph(self.graph)
        self.ai.set_db(self.db)

        logger.info(f"NeuralServiceMesh v{self.VERSION} ready")

    # ── Node Management ────────────────────────────────────────────────────

    def register_node(self, node, connect_to: Optional[str] = None) -> str:
        node_id = self.registry.register(node)
        self.graph.add_node(node_id, node.metadata.to_dict())
        # Phase 2: persist node to SQLite
        self.db.upsert_node(node.to_dict())
        if connect_to:
            self.graph.add_edge(connect_to, node_id)
            self.db.upsert_connection(connect_to, node_id)
        logger.info(f"Registered node '{node.name}' [{node_id[:8]}]")
        return node_id

    # ── Execution ──────────────────────────────────────────────────────────

    def run(self, start_id: str, end_id: str, data: dict, use_ai: bool = True) -> dict:
        return self.engine.run_between(start_id, end_id, data, use_ai=use_ai).to_dict()

    # ── Status ─────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "version": self.VERSION,
            "phase": 2,
            "nodes": self.registry.count(),
            "graph": self.graph.stats(),
            "storage": self.storage.stats(),
            "db": self.db.db_stats(),
            "ai": {
                "enabled": True,
                "mode": "rules+heuristics",
                "paths_tracked": len(self.ai._path_stats),
            },
        }


# ── Demo ───────────────────────────────────────────────────────────────────

def demo():
    import json
    print("\n" + "="*60)
    print("  Neural Service Mesh  —  Phase 2 Demo")
    print("="*60 + "\n")

    mesh = NeuralServiceMesh()

    # Register pipeline
    inp  = mesh.register_node(InputNode("TextInput"))
    proc = mesh.register_node(ProcessorNode("TextProcessor"), connect_to=inp)
    out  = mesh.register_node(OutputNode("TextOutput", output_format="summary"), connect_to=proc)

    sample_text = (
        "Neural networks are inspired by the human brain. "
        "They consist of interconnected nodes that process information. "
        "Deep learning has achieved remarkable results in vision and language tasks."
    )

    # Run #1: with AI path selection
    print("[ Run 1 — AI-selected path ]\n")
    r1 = mesh.run(inp, out, {"text": sample_text, "source": "demo"}, use_ai=True)
    print(f"  Status        : {r1['status']}")
    print(f"  Duration      : {r1['total_duration_ms']} ms")
    print(f"  AI Suggested  : {r1['ai_suggested']}")
    print(f"  Path length   : {len(r1['path'])} nodes")
    print(f"  Final output  :\n{json.dumps(r1['final_output'], indent=4, ensure_ascii=False)}\n")

    # Run #2: explicit path
    print("[ Run 2 — Explicit path ]\n")
    r2 = mesh.engine.run_path([inp, proc, out], {"text": "Second test run.", "source": "demo"})
    print(f"  Status   : {r2.status}")
    print(f"  Duration : {r2.total_duration_ms} ms\n")

    # AI Insights after runs
    print("[ AI Insights ]\n")
    insights = mesh.ai.get_insights()
    print(json.dumps(insights, indent=2, ensure_ascii=False))

    # DB stats
    print("\n[ Database Stats ]\n")
    print(json.dumps(mesh.db.db_stats(), indent=2))

    # Full status
    print("\n[ System Status ]\n")
    print(json.dumps(mesh.status(), indent=2))


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Neural Service Mesh v2")
    p.add_argument("--mode", choices=["demo", "api"], default="demo",
                   help="demo: run example pipeline | api: start Flask server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if args.mode == "demo":
        demo()
    else:
        from api.app import run_api
        mesh = NeuralServiceMesh()
        run_api(mesh, host=args.host, port=args.port, debug=args.debug)
