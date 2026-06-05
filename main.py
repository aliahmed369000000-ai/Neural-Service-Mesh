
from __future__ import annotations
import logging
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("NeuralServiceMesh")

from storage.file_storage import FileStorage
from core.registry import NodeRegistry
from core.graph import ServiceGraph
from core.engine import ExecutionEngine
from connectors.data_transformer import DataTransformer
from services.input_service import InputNode
from services.processor_service import ProcessorNode
from services.output_service import OutputNode


class NeuralServiceMesh:
    def __init__(self, storage_dir: str = "./data"):
        self.storage = FileStorage(storage_dir)
        self.registry = NodeRegistry(self.storage)
        self.graph = ServiceGraph()
        self.transformer = DataTransformer()
        self.engine = ExecutionEngine(self.registry, self.graph, self.storage, self.transformer)
        logger.info("NeuralServiceMesh ready")

    def register_node(self, node, connect_to: Optional[str] = None) -> str:
        node_id = self.registry.register(node)
        self.graph.add_node(node_id, node.metadata.to_dict())
        if connect_to:
            self.graph.add_edge(connect_to, node_id)
        return node_id

    def run(self, start_id: str, end_id: str, data: dict) -> dict:
        return self.engine.run_between(start_id, end_id, data).to_dict()

    def status(self) -> dict:
        return {"nodes": self.registry.count(),
                "graph": self.graph.stats(),
                "storage": self.storage.stats()}


def demo():
    import json
    mesh = NeuralServiceMesh()
    inp  = mesh.register_node(InputNode("TextInput"))
    proc = mesh.register_node(ProcessorNode("TextProcessor"), connect_to=inp)
    out  = mesh.register_node(OutputNode("TextOutput", output_format="summary"), connect_to=proc)

    result = mesh.run(inp, out, {
        "text": "Neural networks are inspired by the human brain. "
                "They consist of interconnected nodes that process information. "
                "Deep learning has achieved remarkable results.",
        "source": "demo"
    })
    print("\n=== Result ===")
    print(json.dumps(result["final_output"], indent=2, ensure_ascii=False))
    print("\n=== Status ===")
    print(json.dumps(mesh.status(), indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["demo", "api"], default="demo")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()
    if args.mode == "demo":
        demo()
    else:
        from api.app import run_api
        mesh = NeuralServiceMesh()
        run_api(mesh, port=args.port)
