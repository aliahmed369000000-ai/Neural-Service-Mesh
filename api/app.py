from __future__ import annotations
import logging
from typing import Any

try:
    from flask import Flask, request, jsonify
    FLASK_OK = True
except ImportError:
    FLASK_OK = False

logger = logging.getLogger(__name__)


def create_app(mesh) -> Any:
    if not FLASK_OK:
        raise ImportError("Run: pip install flask")

    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # ── Health ─────────────────────────────────────────────────────────────

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "version": "2.0.0", "phase": 2})

    @app.route("/status")
    def status():
        return jsonify(mesh.status())

    # ── Nodes ──────────────────────────────────────────────────────────────

    @app.route("/nodes", methods=["GET"])
    def list_nodes():
        nodes = mesh.registry.list_metadata()
        return jsonify({"nodes": nodes, "count": len(nodes)})

    @app.route("/nodes", methods=["POST"])
    def create_node():
        """
        Phase 2: Create a dynamic PassThroughNode via API.
        Body: { "name": str, "description": str, "tags": [...] }
        """
        b = request.get_json(force=True) or {}
        name = b.get("name", "").strip()
        if not name:
            return jsonify({"error": "name is required"}), 400
        try:
            from services.dynamic_node import PassThroughNode
            node = PassThroughNode(name=name,
                                   description=b.get("description", ""),
                                   tags=b.get("tags", []))
            node_id = mesh.register_node(node)
            return jsonify({"node_id": node_id, **node.to_dict()}), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/nodes/<node_id>", methods=["GET"])
    def get_node(node_id):
        n = mesh.registry.get(node_id)
        return jsonify(n.to_dict()) if n else (jsonify({"error": "Not found"}), 404)

    @app.route("/nodes/<node_id>", methods=["DELETE"])
    def delete_node(node_id):
        ok_reg = mesh.registry.unregister(node_id)
        ok_graph = mesh.graph.remove_node(node_id)
        if mesh.db:
            mesh.db.delete_node(node_id)
        if ok_reg or ok_graph:
            return jsonify({"deleted": node_id})
        return jsonify({"error": "Not found"}), 404

    # ── Graph ──────────────────────────────────────────────────────────────

    @app.route("/graph", methods=["GET"])
    def get_graph():
        return jsonify(mesh.graph.to_dict())

    @app.route("/graph/stats", methods=["GET"])
    def graph_stats():
        return jsonify(mesh.graph.stats())

    # ── Connections ────────────────────────────────────────────────────────

    @app.route("/connect", methods=["POST"])
    def connect_nodes():
        b = request.get_json(force=True) or {}
        src, tgt = b.get("source_id"), b.get("target_id")
        if not src or not tgt:
            return jsonify({"error": "source_id and target_id required"}), 400
        try:
            weight = float(b.get("weight", 1.0))
            label = b.get("label", "")
            edge = mesh.graph.add_edge(src, tgt, weight, label)
            if mesh.db:
                mesh.db.upsert_connection(src, tgt, weight, label)
            return jsonify(edge.to_dict())
        except KeyError as e:
            return jsonify({"error": str(e)}), 404

    @app.route("/connect", methods=["DELETE"])
    def disconnect_nodes():
        b = request.get_json(force=True) or {}
        src, tgt = b.get("source_id"), b.get("target_id")
        if not src or not tgt:
            return jsonify({"error": "source_id and target_id required"}), 400
        ok = mesh.graph.remove_edge(src, tgt)
        if ok and mesh.db:
            mesh.db.delete_connection(src, tgt)
        return jsonify({"removed": ok})

    @app.route("/connections", methods=["GET"])
    def list_connections():
        if mesh.db:
            conns = mesh.db.list_connections()
        else:
            conns = [e.to_dict() for edges in mesh.graph._adjacency.values() for e in edges]
        return jsonify({"connections": conns, "count": len(conns)})

    # ── Execution ──────────────────────────────────────────────────────────

    @app.route("/run", methods=["POST"])
    def run_pipeline():
        b = request.get_json(force=True) or {}
        start, end = b.get("start_id"), b.get("end_id")
        data = b.get("data", {})
        use_ai = b.get("use_ai", True)
        if not start or not end:
            return jsonify({"error": "start_id and end_id required"}), 400
        result = mesh.engine.run_between(start, end, data, use_ai=use_ai)
        return jsonify(result.to_dict())

    @app.route("/run/path", methods=["POST"])
    def run_explicit_path():
        b = request.get_json(force=True) or {}
        path = b.get("path", [])
        if not path:
            return jsonify({"error": "path list required"}), 400
        result = mesh.engine.run_path(path, b.get("data", {}))
        return jsonify(result.to_dict())

    @app.route("/run/full", methods=["POST"])
    def run_full():
        b = request.get_json(force=True) or {}
        result = mesh.engine.run_full_graph(b.get("data", {}))
        return jsonify(result.to_dict())

    # ── Logs ───────────────────────────────────────────────────────────────

    @app.route("/runs", methods=["GET"])
    def list_runs():
        limit = int(request.args.get("limit", 20))
        status_filter = request.args.get("status")
        if mesh.db:
            runs = mesh.db.list_runs(limit=limit, status=status_filter)
        else:
            runs = mesh.engine.get_history(limit)
        return jsonify({"runs": runs, "count": len(runs)})

    @app.route("/runs/<run_id>", methods=["GET"])
    def get_run(run_id):
        r = mesh.engine.get_run(run_id)
        return jsonify(r) if r else (jsonify({"error": "Not found"}), 404)

    @app.route("/logs/stats", methods=["GET"])
    def log_stats():
        if mesh.db:
            return jsonify(mesh.db.stats())
        return jsonify({"message": "SQLite not enabled"}), 503

    # ── AI ─────────────────────────────────────────────────────────────────

    @app.route("/ai/paths", methods=["POST"])
    def ai_rank_paths():
        """Ask the AI layer to rank all paths between two nodes."""
        b = request.get_json(force=True) or {}
        start, end = b.get("start_id"), b.get("end_id")
        if not start or not end:
            return jsonify({"error": "start_id and end_id required"}), 400
        if not mesh.ai:
            return jsonify({"error": "AI layer not enabled"}), 503
        ranked = mesh.ai.rank_paths(start, end)
        return jsonify({"ranked_paths": ranked, "count": len(ranked)})

    @app.route("/ai/suggest", methods=["POST"])
    def ai_suggest_next():
        """Ask AI for next node suggestion."""
        b = request.get_json(force=True) or {}
        current = b.get("current_node_id")
        if not current:
            return jsonify({"error": "current_node_id required"}), 400
        if not mesh.ai:
            return jsonify({"error": "AI layer not enabled"}), 503
        suggestion = mesh.ai.suggest_next(current, b.get("context", {}))
        return jsonify({"suggested_node_id": suggestion})

    @app.route("/ai/insights", methods=["GET"])
    def ai_insights():
        if not mesh.ai:
            return jsonify({"error": "AI layer not enabled"}), 503
        return jsonify(mesh.ai.get_insights())

    # ── Storage ────────────────────────────────────────────────────────────

    @app.route("/storage/stats", methods=["GET"])
    def storage_stats():
        stats = {"file_storage": mesh.storage.stats()}
        if mesh.db:
            stats["sqlite"] = mesh.db.db_stats()
        return jsonify(stats)

    return app


def run_api(mesh, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    app = create_app(mesh)
    logger.info(f"API v2 starting on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
