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


def _add_phase3_routes(app, mesh):
    """Phase 3 API endpoints — added only when mesh has Phase 3 components."""

    # ── Goal-Based Execution ───────────────────────────────────────────────

    @app.route("/run/goal", methods=["POST"])
    def run_goal():
        """Phase 3: Execute based on a high-level goal description."""
        b = request.get_json(force=True) or {}
        goal = b.get("goal", "").strip()
        if not goal:
            return jsonify({"error": "goal is required"}), 400
        result = mesh.run_goal(
            goal=goal,
            data=b.get("data", {}),
            preferred_start=b.get("preferred_start"),
            preferred_end=b.get("preferred_end"),
            max_hops=int(b.get("max_hops", 10)),
        )
        return jsonify(result)

    @app.route("/ai/plan", methods=["POST"])
    def plan_goal():
        """Phase 3: Build an execution plan without running it."""
        b = request.get_json(force=True) or {}
        goal = b.get("goal", "").strip()
        if not goal:
            return jsonify({"error": "goal is required"}), 400
        plan = mesh.planner.plan(
            goal,
            preferred_start=b.get("preferred_start"),
            preferred_end=b.get("preferred_end"),
            required_tags=b.get("required_tags"),
            max_hops=int(b.get("max_hops", 10)),
        )
        if not plan:
            return jsonify({"error": "Could not build a plan for this goal"}), 404
        return jsonify(plan.to_dict())

    # ── Routing ────────────────────────────────────────────────────────────

    @app.route("/ai/routes", methods=["POST"])
    def rank_routes():
        """Phase 3: Rank all routes using full AI scoring."""
        b = request.get_json(force=True) or {}
        start, end = b.get("start_id"), b.get("end_id")
        if not start or not end:
            return jsonify({"error": "start_id and end_id required"}), 400
        candidates = mesh.routing.rank_routes(start, end,
                                               max_candidates=int(b.get("max_candidates", 8)))
        return jsonify({"routes": [c.to_dict() for c in candidates], "count": len(candidates)})

    @app.route("/ai/route/explain", methods=["POST"])
    def explain_route():
        """Phase 3: Explain why a specific path was chosen."""
        b = request.get_json(force=True) or {}
        path = b.get("path", [])
        if not path:
            return jsonify({"error": "path list required"}), 400
        return jsonify(mesh.routing.explain_route(path))

    # ── Discovery ──────────────────────────────────────────────────────────

    @app.route("/ai/discover", methods=["GET"])
    def discover_connections():
        """Phase 3: Suggest new connections based on semantic compatibility."""
        threshold = float(request.args.get("threshold", 0.15))
        suggestions = mesh.discover_connections(threshold)
        return jsonify({"suggestions": suggestions, "count": len(suggestions)})

    @app.route("/ai/discover/goal", methods=["POST"])
    def discover_for_goal():
        """Phase 3: Find nodes best suited for a goal."""
        b = request.get_json(force=True) or {}
        goal = b.get("goal", "").strip()
        if not goal:
            return jsonify({"error": "goal is required"}), 400
        top_k = int(b.get("top_k", 5))
        results = mesh.find_nodes_for_goal(goal, top_k)
        return jsonify({"nodes": results, "goal": goal})

    @app.route("/ai/announcements", methods=["GET"])
    def node_announcements():
        """Phase 3: List all node self-announcements."""
        anns = [a.to_dict() for a in mesh.discovery.active_nodes()]
        return jsonify({"announcements": anns, "count": len(anns)})

    # ── Scoring ────────────────────────────────────────────────────────────

    @app.route("/ai/scores", methods=["GET"])
    def connection_scores():
        """Phase 3: Get all connection performance scores."""
        scores = mesh.scoring.list_scores()
        return jsonify({"scores": scores, "count": len(scores),
                        "summary": mesh.scoring.summary()})

    @app.route("/ai/scores/top", methods=["GET"])
    def top_scores():
        n = int(request.args.get("n", 10))
        return jsonify({
            "top": mesh.scoring.top_connections(n),
            "worst": mesh.scoring.worst_connections(n),
        })

    # ── Memory ─────────────────────────────────────────────────────────────

    @app.route("/ai/memory", methods=["GET"])
    def memory_overview():
        """Phase 3: Route memory overview."""
        return jsonify({
            "summary": mesh.memory.summary(),
            "routes": mesh.memory.all_routes(),
            "best_nodes": mesh.memory.best_nodes(10),
        })

    @app.route("/ai/memory/promote", methods=["POST"])
    def promote_route():
        b = request.get_json(force=True) or {}
        path = b.get("path", [])
        if not path:
            return jsonify({"error": "path required"}), 400
        mesh.memory.promote_route(path)
        return jsonify({"promoted": True, "path": path})

    # ── Optimization ───────────────────────────────────────────────────────

    @app.route("/ai/optimize", methods=["POST"])
    def optimize():
        """Phase 3: Run optimization analysis."""
        b = request.get_json(force=True) or {}
        auto_apply = bool(b.get("auto_apply", False))
        report = mesh.optimize(auto_apply=auto_apply)
        return jsonify(report)

    @app.route("/ai/optimize/health", methods=["GET"])
    def optimization_health():
        """Phase 3: Quick health check from optimizer."""
        return jsonify(mesh.optimizer.quick_health_check())

    # ── Semantic ───────────────────────────────────────────────────────────

    @app.route("/ai/semantic/profiles", methods=["GET"])
    def semantic_profiles():
        """Phase 3: List all semantic profiles."""
        profiles = [p.to_dict() for p in mesh.semantic.all_profiles()]
        return jsonify({"profiles": profiles, "count": len(profiles)})

    @app.route("/ai/semantic/compatibility", methods=["POST"])
    def semantic_compatibility():
        """Phase 3: Get semantic compatibility score between two nodes."""
        b = request.get_json(force=True) or {}
        src, tgt = b.get("source_id"), b.get("target_id")
        if not src or not tgt:
            return jsonify({"error": "source_id and target_id required"}), 400
        score = mesh.semantic.compatibility_score(src, tgt)
        return jsonify({"source_id": src, "target_id": tgt, "compatibility_score": score})


def _add_phase4_routes(app, mesh):
    """Phase 4 API endpoints — Learning Validation & Autonomous Evolution."""

    # ── GET /ai/status ─────────────────────────────────────────────────────

    @app.route("/ai/status", methods=["GET"])
    def ai_status():
        """Phase 4: Full AI status with learning proof."""
        return jsonify(mesh.get_ai_status())

    # ── GET /ai/routes ─────────────────────────────────────────────────────

    @app.route("/ai/routes", methods=["GET"])
    def ai_routes_get():
        """Phase 4: All known routes ranked by memory score (GET version)."""
        return jsonify(mesh.get_ai_routes())

    # ── GET /ai/reputation ─────────────────────────────────────────────────

    @app.route("/ai/reputation", methods=["GET"])
    def ai_reputation():
        """Phase 4: Node reputation scores."""
        return jsonify(mesh.get_ai_reputation())

    @app.route("/ai/reputation/<node_id>", methods=["GET"])
    def ai_node_reputation(node_id):
        """Phase 4: Single node reputation."""
        mesh.reputation.update_from_memory()
        rep = mesh.reputation.get_reputation(node_id)
        if rep is None:
            return jsonify({"error": "Node not found in reputation engine"}), 404
        return jsonify(rep.to_dict())

    @app.route("/ai/reputation/<node_id>/quarantine", methods=["POST"])
    def quarantine_node(node_id):
        """Phase 4: Quarantine a low-performing node."""
        mesh.reputation.quarantine(node_id)
        return jsonify({"quarantined": node_id})

    @app.route("/ai/reputation/<node_id>/boost", methods=["POST"])
    def boost_node(node_id):
        """Phase 4: Manually boost a node's reputation."""
        b = request.get_json(force=True) or {}
        amount = float(b.get("amount", 10.0))
        mesh.reputation.boost(node_id, amount)
        rep = mesh.reputation.get_reputation(node_id)
        return jsonify(rep.to_dict() if rep else {"boosted": node_id})

    # ── GET /ai/knowledge ──────────────────────────────────────────────────

    @app.route("/ai/knowledge", methods=["GET"])
    def ai_knowledge():
        """Phase 4: Full knowledge layer snapshot."""
        return jsonify(mesh.get_ai_knowledge())

    # ── Learning metrics ───────────────────────────────────────────────────

    @app.route("/ai/learning/metrics", methods=["GET"])
    def ai_learning_metrics():
        """Phase 4: Current learning metrics."""
        metrics = mesh.validator.compute_metrics()
        return jsonify(metrics.to_dict())

    @app.route("/ai/learning/prove", methods=["GET"])
    def ai_learning_prove():
        """Phase 4: Generate proof that the system is learning."""
        return jsonify(mesh.validator.prove_learning())

    @app.route("/ai/learning/curve", methods=["GET"])
    def ai_learning_curve():
        """Phase 4: Learning curve data points."""
        return jsonify(mesh.validator.get_learning_curve())

    # ── Graph evolution ────────────────────────────────────────────────────

    @app.route("/ai/graph/evolve", methods=["POST"])
    def ai_graph_evolve():
        """
        Phase 4: Trigger a full graph evolution cycle.
        - Discovery engine proposes new connections
        - Optimization engine flags weak connections
        """
        b = request.get_json(force=True) or {}
        auto_apply = bool(b.get("auto_apply", False))

        # Run discovery
        suggestions = mesh.discover_connections(threshold=float(b.get("threshold", 0.15)))

        # Run optimization
        opt_report = mesh.optimize(auto_apply=auto_apply)

        # Update knowledge layer
        try:
            mesh.knowledge.update_node_rankings(mesh.memory)
            mesh.knowledge.update_route_rankings(mesh.memory)
            mesh.knowledge.update_connection_scores(mesh.scoring)
        except Exception:
            pass

        return jsonify({
            "new_connection_suggestions": suggestions,
            "optimization_report": opt_report,
            "auto_apply": auto_apply,
        })

    # ── Simulation ─────────────────────────────────────────────────────────

    @app.route("/ai/simulate", methods=["POST"])
    def ai_simulate():
        """Phase 4: Run a simulation loop via API."""
        b = request.get_json(force=True) or {}
        rounds = int(b.get("rounds", 5))
        per_round = int(b.get("executions_per_round", 3))

        if rounds > 50:
            return jsonify({"error": "Maximum 50 rounds via API"}), 400

        from ai.simulation_engine import SimulationEngine
        sim = SimulationEngine(mesh, validator=mesh.validator)
        inp, proc, out = sim.setup_simulation_nodes()
        results = sim.run_simulation(
            rounds=rounds,
            executions_per_round=per_round,
            delay_between_rounds=0,
            verbose=False,
        )
        return jsonify(results)

    # ── Health v4 ──────────────────────────────────────────────────────────

    @app.route("/health/v4", methods=["GET"])
    def health_v4():
        return jsonify({"status": "ok", "version": "4.0.0", "phase": 4})


def _add_phase5_routes(app, mesh):
    """Phase 5 API endpoints — Autonomous Service Creation & Evolution."""

    # ── Evolution ──────────────────────────────────────────────────────────

    @app.route("/ai/evolve", methods=["POST"])
    def ai_evolve():
        """Phase 5: Run evolution cycle(s) — detect gaps and generate services."""
        b = request.get_json(force=True) or {}
        cycles = int(b.get("cycles", 1))
        auto_register = bool(b.get("auto_register", True))
        if cycles > 10:
            return jsonify({"error": "Maximum 10 cycles per request"}), 400
        result = mesh.evolve(cycles=cycles, auto_register=auto_register)
        return jsonify(result)

    # ── Gap Detection ──────────────────────────────────────────────────────

    @app.route("/ai/gaps", methods=["GET"])
    def ai_gaps():
        """Phase 5: Get all detected gaps."""
        include_resolved = request.args.get("include_resolved", "false").lower() == "true"
        return jsonify({
            "gaps": mesh.gap_detector.all_gaps(include_resolved=include_resolved),
            "summary": mesh.gap_detector.summary(),
        })

    @app.route("/ai/gaps/scan", methods=["POST"])
    def ai_gaps_scan():
        """Phase 5: Trigger a gap scan."""
        return jsonify(mesh.scan_gaps())

    @app.route("/ai/gaps/<gap_id>/resolve", methods=["POST"])
    def ai_gap_resolve(gap_id):
        """Phase 5: Mark a gap as resolved."""
        ok = mesh.gap_detector.mark_resolved(gap_id)
        if ok:
            return jsonify({"resolved": gap_id})
        return jsonify({"error": "Gap not found"}), 404

    # ── Service Generator ──────────────────────────────────────────────────

    @app.route("/ai/services/generated", methods=["GET"])
    def ai_generated_services():
        """Phase 5: List all AI-generated service specs."""
        status_filter = request.args.get("status")
        return jsonify(mesh.get_generated_services(status=status_filter))

    @app.route("/ai/services/create", methods=["POST"])
    def ai_services_create():
        """Phase 5: Manually trigger service generation for a gap."""
        b = request.get_json(force=True) or {}
        gap = b.get("gap")
        if not gap:
            # Build a minimal gap from request body
            gap = {
                "missing_service": b.get("missing_service", "AutoService"),
                "confidence": float(b.get("confidence", 0.7)),
                "gap_type": b.get("gap_type", "routing"),
                "source_node": b.get("source_node", {}),
                "target_node": b.get("target_node", {}),
            }
        spec = mesh.service_generator.generate_for_gap(gap)
        if not spec:
            return jsonify({"error": "Could not generate service"}), 500
        return jsonify(spec.to_dict()), 201

    @app.route("/ai/services/generated/<spec_id>/approve", methods=["POST"])
    def ai_service_approve(spec_id):
        """Phase 5: Approve a generated service spec for registration."""
        ok = mesh.service_generator.approve_spec(spec_id)
        if not ok:
            return jsonify({"error": "Spec not found"}), 404
        spec = mesh.service_generator.get_spec(spec_id)
        if spec:
            decision = mesh.governance.evaluate_generation(
                spec.to_dict(), mesh.service_generator.summary()["total_generated"]
            )
            if decision.allowed:
                node = mesh.service_generator.instantiate_spec(spec)
                node_id = mesh.register_node(node)
                return jsonify({"approved": spec_id, "node_id": node_id})
            else:
                return jsonify({"approved": False, "reason": decision.reason}), 403
        return jsonify({"approved": spec_id})

    @app.route("/ai/services/generated/<spec_id>/reject", methods=["POST"])
    def ai_service_reject(spec_id):
        """Phase 5: Reject a generated service spec."""
        ok = mesh.service_generator.reject_spec(spec_id)
        if ok:
            return jsonify({"rejected": spec_id})
        return jsonify({"error": "Spec not found"}), 404

    # ── Capability Marketplace ─────────────────────────────────────────────

    @app.route("/ai/marketplace", methods=["GET"])
    def ai_marketplace():
        """Phase 5: Get capability marketplace snapshot."""
        return jsonify(mesh.get_marketplace())

    @app.route("/ai/marketplace/find", methods=["POST"])
    def ai_marketplace_find():
        """Phase 5: Find providers for a capability."""
        b = request.get_json(force=True) or {}
        capability = b.get("capability", "")
        if not capability:
            return jsonify({"error": "capability required"}), 400
        top_k = int(b.get("top_k", 5))
        providers = mesh.marketplace.find_providers(capability, top_k=top_k)
        return jsonify({
            "capability": capability,
            "providers": [p.to_dict() for p in providers],
            "count": len(providers),
        })

    @app.route("/ai/marketplace/advertise", methods=["POST"])
    def ai_marketplace_advertise():
        """Phase 5: Manually advertise a node's capability."""
        b = request.get_json(force=True) or {}
        node_id = b.get("node_id", "")
        node_name = b.get("node_name", "")
        capability = b.get("capability", "")
        if not all([node_id, capability]):
            return jsonify({"error": "node_id and capability required"}), 400
        ad = mesh.marketplace.advertise(
            node_id=node_id,
            node_name=node_name,
            capability=capability,
            quality_score=float(b.get("quality_score", 0.8)),
        )
        return jsonify(ad.to_dict()), 201

    # ── Multi-Goal Planning ────────────────────────────────────────────────

    @app.route("/ai/run/multi-goal", methods=["POST"])
    def ai_run_multi_goal():
        """Phase 5: Execute a complex multi-goal plan."""
        b = request.get_json(force=True) or {}
        goal = b.get("goal", "").strip()
        data = b.get("data", {})
        if not goal:
            return jsonify({"error": "goal required"}), 400
        result = mesh.run_multi_goal(goal, data)
        return jsonify(result)

    @app.route("/ai/plan/multi-goal", methods=["POST"])
    def ai_plan_multi_goal():
        """Phase 5: Build a multi-goal plan without executing it."""
        b = request.get_json(force=True) or {}
        goal = b.get("goal", "").strip()
        if not goal:
            return jsonify({"error": "goal required"}), 400
        plan = mesh.multi_planner.plan(goal)
        return jsonify(plan.to_dict())

    # ── Governance ─────────────────────────────────────────────────────────

    @app.route("/ai/governance", methods=["GET"])
    def ai_governance():
        """Phase 5: Get governance status and recent audit log."""
        return jsonify(mesh.get_governance())

    @app.route("/ai/governance/evaluate", methods=["POST"])
    def ai_governance_evaluate():
        """Phase 5: Evaluate a proposed action against governance policies."""
        b = request.get_json(force=True) or {}
        action = b.get("action", "add_route")
        context = b.get("context", {})
        decision = mesh.governance.evaluate(action, context)
        return jsonify(decision.to_dict())

    # ── Health v5 ──────────────────────────────────────────────────────────

    @app.route("/health/v5", methods=["GET"])
    def health_v5():
        return jsonify({"status": "ok", "version": "5.0.0", "phase": 5})

    # ── System DNA ─────────────────────────────────────────────────────────

    @app.route("/ai/dna", methods=["GET"])
    def ai_dna():
        """Phase 5: System DNA — philosophy, evolution rules, weights."""
        system_dna = {
            "version": "5.0.0",
            "philosophy": "Autonomous Neural Service Ecosystem",
            "capabilities": [
                "Self-learning from execution history",
                "Gap detection and service generation",
                "Capability-based routing (not name-based)",
                "AI governance to prevent runaway growth",
                "Multi-goal decomposition and planning",
                "Autonomous evolution cycles",
            ],
            "evolution_rules": {
                "max_generated_services": mesh.governance.MAX_GENERATED_SERVICES,
                "max_path_length": mesh.governance.MAX_PATH_LENGTH,
                "min_node_reputation": mesh.governance.MIN_NODE_REPUTATION,
                "min_confidence_threshold": 0.4,
            },
            "weights": {
                "quality_weight": 0.7,
                "latency_weight": 0.3,
                "reputation_influence": 0.25,
            },
            "learning_strategies": [
                "Route memory reinforcement",
                "Semantic similarity matching",
                "Execution score tracking",
                "Reputation-based node ranking",
                "Capability marketplace indexing",
            ],
            "current_state": mesh.status(),
        }
        # Persist DNA
        try:
            mesh.knowledge.write_custom("system_dna", {
                k: v for k, v in system_dna.items()
                if k != "current_state"
            })
        except Exception:
            pass
        return jsonify(system_dna)


# Patch create_app to include Phase 5
_create_app_p4 = create_app

def create_app(mesh):
    app = _create_app_p4(mesh)
    if hasattr(mesh, "evolution"):
        _add_phase5_routes(app, mesh)
    return app
