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


# ── Phase 6 routes ─────────────────────────────────────────────────────────

def _add_phase6_routes(app, mesh):
    from flask import jsonify, request

    # Agents
    @app.route("/ai/agents", methods=["GET"])
    def p6_agents():
        """Phase 6: List all agents."""
        return jsonify(mesh.get_agent_factory())

    @app.route("/ai/agents/spawn", methods=["POST"])
    def p6_spawn():
        """Phase 6: Spawn a new agent."""
        body = request.get_json(force=True) or {}
        role = body.get("role")
        if not role:
            return jsonify({"error": "role is required"}), 400
        try:
            return jsonify(mesh.spawn_agent(role, body.get("config")))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/ai/agents/<agent_id>/retire", methods=["POST"])
    def p6_retire(agent_id):
        """Phase 6: Retire an agent."""
        ok = mesh.agent_factory.retire(agent_id)
        return jsonify({"retired": ok, "agent_id": agent_id})

    # Swarm
    @app.route("/ai/swarm/execute", methods=["POST"])
    def p6_swarm():
        """Phase 6: Execute a goal via the agent swarm."""
        body = request.get_json(force=True) or {}
        goal = body.get("goal", "")
        data = body.get("data", {})
        tasks = body.get("tasks")
        return jsonify(mesh.swarm_execute(goal, data, tasks))

    @app.route("/ai/swarm/history", methods=["GET"])
    def p6_swarm_history():
        """Phase 6: Get recent swarm execution history."""
        limit = int(request.args.get("limit", 10))
        return jsonify(mesh.get_swarm_history(limit))

    # Self-optimizer
    @app.route("/ai/optimize/self", methods=["POST"])
    def p6_self_optimize():
        """Phase 6: Trigger one self-optimization cycle."""
        return jsonify(mesh.self_optimize())

    @app.route("/ai/optimize/history", methods=["GET"])
    def p6_optimize_history():
        """Phase 6: Get self-optimizer history."""
        limit = int(request.args.get("limit", 10))
        return jsonify({"history": mesh.self_optimizer.history(limit)})

    # Simulation lab
    @app.route("/ai/simulate", methods=["POST"])
    def p6_simulate():
        """Phase 6: Run simulation lab over candidate plans."""
        body = request.get_json(force=True) or {}
        goal = body.get("goal", "")
        data = body.get("data", {})
        n_plans = int(body.get("n_plans", 100))
        return jsonify(mesh.simulate_plans(goal, data, n_plans))

    # Meta-reasoner
    @app.route("/ai/meta/reflect", methods=["POST"])
    def p6_reflect():
        """Phase 6: Run meta-reasoning reflection."""
        return jsonify(mesh.meta_reflect())

    @app.route("/ai/meta/ask", methods=["POST"])
    def p6_ask():
        """Phase 6: Ask the meta-reasoner a question."""
        body = request.get_json(force=True) or {}
        question = body.get("question", "")
        return jsonify(mesh.meta_ask(question))

    @app.route("/ai/meta/insights", methods=["GET"])
    def p6_insights():
        """Phase 6: Get recent meta-reasoning insights."""
        limit = int(request.args.get("limit", 10))
        return jsonify({"insights": mesh.meta_reasoner.recent_insights(limit)})

    # Economic engine
    @app.route("/ai/economy/leaderboard", methods=["GET"])
    def p6_leaderboard():
        """Phase 6: Get economic leaderboard."""
        top_n = int(request.args.get("top", 10))
        return jsonify(mesh.economic_leaderboard(top_n))

    @app.route("/ai/economy/node/<node_id>", methods=["GET"])
    def p6_node_economy(node_id):
        """Phase 6: Get economic profile for a single node."""
        profile = mesh.economic_engine.get_profile(node_id)
        if not profile:
            return jsonify({"error": "Node not found in economic engine"}), 404
        return jsonify(profile)

    # System DNA (Phase 6 — richer than Phase 5 placeholder)
    @app.route("/ai/dna/snapshot", methods=["POST"])
    def p6_dna_snapshot():
        """Phase 6: Capture a DNA snapshot."""
        body = request.get_json(force=True) or {}
        notes = body.get("notes", "")
        return jsonify(mesh.dna_snapshot(notes))

    @app.route("/ai/dna/history", methods=["GET"])
    def p6_dna_history():
        """Phase 6: Get DNA snapshot history."""
        limit = int(request.args.get("limit", 10))
        return jsonify({"history": mesh.system_dna.history(limit)})

    @app.route("/ai/dna/diff", methods=["POST"])
    def p6_dna_diff():
        """Phase 6: Diff two DNA snapshots."""
        body = request.get_json(force=True) or {}
        a = body.get("snapshot_id_a")
        b = body.get("snapshot_id_b")
        if not a or not b:
            return jsonify({"error": "snapshot_id_a and snapshot_id_b are required"}), 400
        return jsonify(mesh.dna_diff(a, b))

    @app.route("/ai/dna/rollback", methods=["POST"])
    def p6_dna_rollback():
        """Phase 6: Rollback to a prior DNA snapshot."""
        body = request.get_json(force=True) or {}
        sid = body.get("snapshot_id")
        if not sid:
            return jsonify({"error": "snapshot_id is required"}), 400
        return jsonify(mesh.dna_rollback(sid))

    @app.route("/health/v6", methods=["GET"])
    def health_v6():
        return jsonify({"status": "ok", "version": "6.0.0", "phase": 6})

    @app.route("/ai/validate/phase6", methods=["GET"])
    def p6_validate():
        """Pre-Phase 7: Generate a full Phase 6 Validation Report."""
        try:
            save = request.args.get("save", "true").lower() == "true"
            report = mesh.validate_phase6(save_report=save)
            return jsonify(report)
        except Exception as e:
            return jsonify({"error": str(e)}), 500


# Patch create_app to include Phase 6
_create_app_p5 = create_app

def create_app(mesh):
    app = _create_app_p5(mesh)
    if hasattr(mesh, "agent_factory"):
        _add_phase6_routes(app, mesh)
    return app


# ── Phase 7 Routes ─────────────────────────────────────────────────────────

def _add_phase7_routes(app, mesh):
    from flask import request, jsonify

    @app.route("/ai/phase7/introspect", methods=["GET"])
    def p7_introspect():
        """Phase 7: Self-awareness introspection report."""
        return jsonify(mesh.introspect())

    @app.route("/ai/phase7/sensors", methods=["GET"])
    def p7_sensors():
        """Phase 7: Sensor hub status and recent events."""
        return jsonify(mesh.sensor_status())

    @app.route("/ai/phase7/sensors/push", methods=["POST"])
    def p7_sensor_push():
        """Phase 7: Push a manual event to the webhook sensor."""
        body = request.get_json(force=True) or {}
        mesh.push_sensor_event(
            event_type=body.get("event_type", "manual"),
            payload=body.get("payload", {}),
            severity=body.get("severity", "info"),
        )
        return jsonify({"status": "queued"})

    @app.route("/ai/phase7/world-model", methods=["GET"])
    def p7_world_model():
        """Phase 7: Get the current world model / environment state."""
        return jsonify(mesh.world_model())

    @app.route("/ai/phase7/objectives", methods=["GET"])
    def p7_objectives():
        """Phase 7: Get all strategic objectives and progress."""
        return jsonify(mesh.get_objectives())

    @app.route("/ai/phase7/objectives/measure", methods=["POST"])
    def p7_measure_objectives():
        """Phase 7: Measure current metrics against objectives."""
        return jsonify(mesh.measure_objectives())

    @app.route("/ai/phase7/generate", methods=["POST"])
    def p7_generate_module():
        """Phase 7: Generate a module for a described gap."""
        body = request.get_json(force=True) or {}
        gap = body.get("gap_description", "GenericTransformer")
        src = body.get("source_name", "")
        tgt = body.get("target_name", "")
        return jsonify(mesh.generate_module(gap, src, tgt))

    @app.route("/ai/phase7/modules", methods=["GET"])
    def p7_list_modules():
        """Phase 7: List all auto-generated modules."""
        status = request.args.get("status")
        return jsonify(mesh.list_generated_modules(status=status))

    @app.route("/ai/phase7/evolve", methods=["POST"])
    def p7_evolve():
        """Phase 7: Run evolution pipeline cycle(s)."""
        body = request.get_json(force=True) or {}
        cycles = int(body.get("cycles", 1))
        return jsonify(mesh.evolve7(cycles=cycles, verbose=False))

    @app.route("/ai/phase7/evolution/history", methods=["GET"])
    def p7_evolution_history():
        """Phase 7: Get evolution pipeline cycle history."""
        limit = int(request.args.get("limit", 10))
        return jsonify(mesh.get_evolution_history(limit=limit))

    @app.route("/ai/phase7/sensors/start", methods=["POST"])
    def p7_sensors_start():
        """Phase 7: Start background sensor polling."""
        body = request.get_json(force=True) or {}
        interval = float(body.get("interval_s", 30.0))
        return jsonify(mesh.start_sensors(interval_s=interval))

    @app.route("/ai/phase7/sensors/stop", methods=["POST"])
    def p7_sensors_stop():
        """Phase 7: Stop background sensor polling."""
        return jsonify(mesh.stop_sensors())

    @app.route("/health/v7", methods=["GET"])
    def health_v7():
        return jsonify({"status": "ok", "version": "7.0.0", "phase": 7})


# Patch create_app to include Phase 7
_create_app_p6 = create_app

def create_app(mesh):
    app = _create_app_p6(mesh)
    if hasattr(mesh, "evolution_pipeline"):
        _add_phase7_routes(app, mesh)
    return app
def _add_phase13_14_routes(app, mesh):
    from flask import jsonify, request

    @app.route("/ai/phase13/status", methods=["GET"])
    def p13_status(): return jsonify(mesh.get_structural_status())

    @app.route("/ai/phase13/redesign", methods=["POST"])
    def p13_redesign():
        body = request.get_json(force=True) or {}
        return jsonify(mesh.redesign(cycles=int(body.get("cycles",1)), verbose=False))

    @app.route("/ai/phase13/history", methods=["GET"])
    def p13_history(): return jsonify(mesh.get_arch_history(int(request.args.get("limit",10))))

    @app.route("/ai/phase13/rollback", methods=["POST"])
    def p13_rollback():
        body = request.get_json(force=True) or {}
        sid  = body.get("snapshot_id","")
        return jsonify(mesh.arch_rollback(sid)) if sid else (jsonify({"error":"snapshot_id required"}),400)

    @app.route("/ai/phase14/status",  methods=["GET"])
    def p14_status(): return jsonify(mesh.get_being_status())

    @app.route("/ai/phase14/cycle",   methods=["POST"])
    def p14_cycle():
        body = request.get_json(force=True) or {}
        return jsonify(mesh.run_being_cycle(fast_mode=bool(body.get("fast_mode",True)), verbose=False))

    @app.route("/ai/phase14/start",   methods=["POST"])
    def p14_start():
        body = request.get_json(force=True) or {}
        return jsonify(mesh.start_being(fast_mode=bool(body.get("fast_mode",False))))

    @app.route("/ai/phase14/stop",    methods=["POST"])
    def p14_stop(): return jsonify(mesh.stop_being())

    @app.route("/ai/phase14/history", methods=["GET"])
    def p14_history(): return jsonify(mesh.get_being_history(int(request.args.get("limit",10))))

    @app.route("/ai/full-status",     methods=["GET"])
    def full_status_all(): return jsonify(mesh.get_full_system_status())

    @app.route("/health/v14",         methods=["GET"])
    def health_v14(): return jsonify({"status":"ok","version":"14.0.0","phase":14})

_create_app_prev = create_app
def create_app(mesh):
    app = _create_app_prev(mesh)
    _add_phase13_14_routes(app, mesh)
    return app


# ── v16 Routes: Dashboard + Narrative + Ethics + Consolidator + Full Status ──

def _add_v16_routes(app, mesh):
    import os
    from flask import send_file, jsonify, request

    @app.route("/", methods=["GET"])
    @app.route("/dashboard", methods=["GET"])
    def dashboard():
        dash = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard.html")
        if os.path.exists(dash):
            return send_file(dash)
        return "<h2>Dashboard file not found</h2>", 404

    @app.route("/api/status/full", methods=["GET"])
    def api_full_status():
        return jsonify(mesh.get_full_system_status())

    @app.route("/api/narrative/log", methods=["GET"])
    def api_narrative_log():
        n = int(request.args.get("n", 20))
        return jsonify(mesh.get_narrative_log(n))

    @app.route("/api/narrative/diary", methods=["GET"])
    def api_narrative_diary():
        if mesh.self_narrative is None:
            return jsonify({"error": "SelfNarrative not available"})
        return jsonify({"diary": mesh.self_narrative.get_today_narrative()})

    @app.route("/api/ethics/violations", methods=["GET"])
    def api_ethics_violations():
        limit = int(request.args.get("limit", 50))
        return jsonify(mesh.get_ethics_violations(limit))

    @app.route("/api/consolidator/laws", methods=["GET"])
    def api_consolidated_laws():
        min_conf = float(request.args.get("min_confidence", 0.0))
        return jsonify(mesh.get_consolidated_laws(min_conf))

    @app.route("/api/immune/status", methods=["GET"])
    def api_immune_status():
        return jsonify(mesh.get_immune_system_status())

    @app.route("/api/quality/status", methods=["GET"])
    def api_quality_status():
        return jsonify(mesh.get_quality_engine_status())

    @app.route("/api/drives/status", methods=["GET"])
    def api_drives_status():
        return jsonify(mesh.get_drive_engine_status())

    @app.route("/api/replication/status", methods=["GET"])
    def api_replication_status():
        return jsonify(mesh.get_replication_engine_status())

    @app.route("/api/world_feed/status", methods=["GET"])
    def api_world_feed_status():
        return jsonify(mesh.get_world_feed_status())

    @app.route("/api/checkpoint/list", methods=["GET"])
    def api_checkpoint_list():
        if mesh.brain_checkpoint is None:
            return jsonify({"error": "BrainCheckpoint not available"})
        return jsonify({"checkpoints": mesh.brain_checkpoint.list_checkpoints()})

    @app.route("/api/checkpoint/save", methods=["POST"])
    def api_checkpoint_save():
        if mesh.brain_checkpoint is None:
            return jsonify({"error": "BrainCheckpoint not available"})
        path = mesh.brain_checkpoint.save(mesh)
        return jsonify({"saved": True, "path": path})

    @app.route("/health/v16", methods=["GET"])
    def health_v16():
        return jsonify({"status": "ok", "version": "16.0.0"})


_create_app_v15 = create_app
def create_app(mesh):
    app = _create_app_v15(mesh)
    _add_v16_routes(app, mesh)
    return app


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge Sources Layer API  (Knowledge Layer — Phase KS)
# Endpoints: /sources/*
# ═══════════════════════════════════════════════════════════════════════════

def _add_knowledge_sources_routes(app, mesh):
    """
    Register all /sources/* endpoints.

    These are the public API for the Knowledge Sources Layer.
    The SourceManager is lazily created per-app and shared via app config.
    """
    from knowledge_sources import SourceManager, SourceMetadata
    from knowledge_sources import SourceType, UpdateFrequency, AccessMode
    from knowledge_sources.quran.quran_source import create_quran_source

    # ── Initialise SourceManager — wire all mesh components ──────────────
    sm = SourceManager(min_quality_threshold=30.0)
    sm.set_knowledge_store(getattr(mesh, "knowledge", None))
    sm.set_environment_model(getattr(mesh, "env_model", None))
    sm.set_memory_engine(getattr(mesh, "memory", None))
    sm.set_semantic_matcher(getattr(mesh, "semantic", None))

    # Register Quran as the first official source
    quran_meta, quran_feeder = create_quran_source()
    sm.register_source(quran_meta, quran_feeder)

    app.config["source_manager"] = sm

    # ── GET /sources/list ─────────────────────────────────────────────────
    @app.route("/sources/list", methods=["GET"])
    def sources_list():
        """List all registered knowledge sources with metadata."""
        return jsonify({
            "sources": sm.list_sources(),
            "total":   len(sm.list_sources()),
            "summary": sm.summary(),
        })

    # ── POST /sources/register ────────────────────────────────────────────
    @app.route("/sources/register", methods=["POST"])
    def sources_register():
        """
        Register a new knowledge source.

        Body (JSON):
          {
            "name":             "My Source",
            "source_type":      "encyclopedia",
            "trust_score":      0.7,
            "update_frequency": "daily",
            "description":      "...",
            "language":         "en",
            "tags":             ["tag1", "tag2"],
            "config":           {}
          }
        """
        try:
            data = request.get_json(force=True) or {}
            meta = SourceMetadata(
                name             = data.get("name", "Unnamed Source"),
                description      = data.get("description", ""),
                source_type      = SourceType(data.get("source_type", "custom")),
                access_mode      = AccessMode(data.get("access_mode", "read_only")),
                trust_score      = float(data.get("trust_score", 0.5)),
                base_trust       = float(data.get("trust_score", 0.5)),
                update_frequency = UpdateFrequency(data.get("update_frequency", "on_demand")),
                language         = data.get("language", "en"),
                tags             = data.get("tags", []),
                config           = data.get("config", {}),
                allow_raw_modification = False,  # always enforced
            )
            registered = sm.register_source(meta, feeder=None)
            return jsonify({"registered": True, "source": registered.to_dict()}), 201
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    # ── POST /sources/quran/sync — Full Quran pipeline ───────────────────
    @app.route("/sources/quran/sync", methods=["POST"])
    def sources_quran_sync():
        """
        Full Quran ingestion pipeline (synchronous):
          1. SourceManager sync  → KnowledgeStore + MemoryEngine + SemanticMatcher
          2. Force CKG rebuild   → CognitiveKnowledgeGraph with all 6236 ayahs
          3. Update graph metrics + route rankings in KnowledgeStore
          4. Update EnvironmentModel

        Returns a combined report.
        """
        import threading
        report = {}

        # Step 1: Source sync
        try:
            result = sm.sync_source(quran_meta.id)
            report["source_sync"] = {
                "items_fetched":   result.items_fetched,
                "items_validated": result.items_validated,
                "items_ingested":  result.items_ingested,
                "items_rejected":  result.items_rejected,
                "avg_quality":     round(result.avg_quality, 2),
                "success":         result.success,
                "errors":          result.errors,
            }
        except Exception as exc:
            report["source_sync"] = {"error": str(exc)}

        # Step 2: Force-rebuild CKG from full quran.json
        try:
            from knowledge.ckg_bootstrap import bootstrap_ckg, wait_for_bootstrap
            import knowledge.ckg_bootstrap as _ckgmod
            # Reset done flag so force works
            with _ckgmod._BOOTSTRAP_LOCK:
                _ckgmod._BOOTSTRAP_DONE = False
            t = bootstrap_ckg(mesh, background=True, force=True)
            if t:
                t.join(timeout=300)   # wait up to 5 min
            from knowledge.ckg_bootstrap import is_bootstrap_done
            report["ckg_rebuild"] = {"done": is_bootstrap_done()}
        except Exception as exc:
            report["ckg_rebuild"] = {"error": str(exc)}

        # Step 3: Read CKG stats
        try:
            import json as _json
            from pathlib import Path as _P
            gf = _P("knowledge/cognitive_graph.json")
            if gf.exists():
                d = _json.loads(gf.read_text(encoding="utf-8"))
                # CKG uses 'concepts' dict and 'relations' dict (not nodes/edges)
                concepts = d.get("concepts", d.get("nodes", {}))
                relations = d.get("relations", d.get("edges", {}))
                report["ckg_stats"] = {
                    "concepts": len(concepts),
                    "relations": len(relations),
                    "clusters": list({
                        v.get("cluster", "?")
                        for v in concepts.values()
                        if isinstance(v, dict)
                    }),
                    "top_concepts": sorted(
                        concepts.keys(),
                        key=lambda k: concepts[k].get("reference_count", 0),
                        reverse=True
                    )[:5],
                }
            else:
                report["ckg_stats"] = {"concepts": 0, "relations": 0}
        except Exception as exc:
            report["ckg_stats"] = {"error": str(exc)}

        # Step 4: Update KnowledgeStore graph metrics
        try:
            if mesh.knowledge and mesh.memory:
                mesh.knowledge.update_node_rankings(mesh.memory)
                mesh.knowledge.update_route_rankings(mesh.memory)
                mesh.knowledge.update_connection_scores(mesh.scoring)
                report["knowledge_store_updated"] = True
        except Exception as exc:
            report["knowledge_store_updated"] = False
            report["knowledge_store_error"] = str(exc)

        return jsonify(report)

    # ── POST /sources/sync ────────────────────────────────────────────────
    @app.route("/sources/sync", methods=["POST"])
    def sources_sync():
        """
        Trigger a sync for one or all sources.

        Body (JSON):
          { "source_id": "<id>" }   — sync one source
          {}                         — sync all active sources
        """
        try:
            data      = request.get_json(force=True) or {}
            source_id = data.get("source_id")
            async_mode = data.get("async", False)

            if source_id:
                if async_mode:
                    sm.sync_source_async(source_id)
                    return jsonify({"status": "started", "source_id": source_id})
                result = sm.sync_source(source_id)
                return jsonify({"result": result.to_dict()})
            else:
                results = sm.sync_all(async_mode=async_mode)
                if async_mode:
                    return jsonify({"status": "started_all"})
                return jsonify({
                    "results": [r.to_dict() for r in results],
                    "total": len(results),
                })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    # ── GET /sources/status ───────────────────────────────────────────────
    @app.route("/sources/status", methods=["GET"])
    def sources_status():
        """
        Status of one or all sources.

        Query params:
          ?source_id=<id>   — status of one source
          (none)            — status of all + sync history
        """
        source_id = request.args.get("source_id")
        if source_id:
            status = sm.source_status(source_id)
            if not status:
                return jsonify({"error": "Source not found"}), 404
            return jsonify(status)
        return jsonify({
            "sources":      sm.list_sources(),
            "sync_history": sm.all_sync_history(),
            "summary":      sm.summary(),
        })

    # ── GET /sources/quality ──────────────────────────────────────────────
    @app.route("/sources/quality", methods=["GET"])
    def sources_quality():
        """
        Quality scoring summary for all sources.
        Returns scorer stats and per-source avg quality scores.
        """
        scorer_summary = sm._scorer.summary()
        per_source = [
            {
                "id":          s["id"],
                "name":        s["name"],
                "avg_quality": s.get("avg_quality_score", 0.0),
                "trust_score": s["trust_score"],
                "total_ingested": s.get("total_items_ingested", 0),
                "total_rejected": s.get("total_items_rejected", 0),
            }
            for s in sm.list_sources()
        ]
        return jsonify({
            "scorer":     scorer_summary,
            "per_source": per_source,
        })

    # ── POST /sources/pause ───────────────────────────────────────────────
    @app.route("/sources/pause", methods=["POST"])
    def sources_pause():
        data = request.get_json(force=True) or {}
        sid  = data.get("source_id")
        if not sid:
            return jsonify({"error": "source_id required"}), 400
        ok = sm.pause_source(sid)
        return jsonify({"paused": ok, "source_id": sid})

    # ── POST /sources/resume ──────────────────────────────────────────────
    @app.route("/sources/resume", methods=["POST"])
    def sources_resume():
        data = request.get_json(force=True) or {}
        sid  = data.get("source_id")
        if not sid:
            return jsonify({"error": "source_id required"}), 400
        ok = sm.resume_source(sid)
        return jsonify({"resumed": ok, "source_id": sid})

    # ── GET /health/ks ────────────────────────────────────────────────────
    @app.route("/health/ks", methods=["GET"])
    def health_ks():
        return jsonify({
            "status":  "ok",
            "layer":   "knowledge_sources",
            "version": "1.0.0",
            "sources": sm.summary()["registry"]["total_sources"],
        })


_create_app_v16 = create_app
def create_app(mesh):
    app = _create_app_v16(mesh)
    _add_knowledge_sources_routes(app, mesh)
    return app


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Knowledge Graph API  (Priority 4 — CKG Endpoints)
# Endpoints: /knowledge/concepts  /knowledge/relations  /knowledge/query
# ═══════════════════════════════════════════════════════════════════════════

def _add_ckg_routes(app, mesh):
    """
    All /knowledge/* endpoints for the Cognitive Knowledge Graph.
    These power the Dashboard Concept Viewer tab.
    """

    def _get_ckg():
        """Lazy-load CKG — only if the file exists."""
        try:
            from knowledge.cognitive_graph import get_ckg
            return get_ckg()
        except Exception as exc:
            logger.warning(f"[CKG API] CKG not available: {exc}")
            return None

    # ── GET /knowledge/concepts ───────────────────────────────────────────
    @app.route("/knowledge/concepts", methods=["GET"])
    def knowledge_concepts():
        """
        كل المفاهيم في الـ CKG.
        Query params:
          ?cluster=أخلاق   — فلتر بـ cluster
          ?top=N           — أقوى N مفهوم فقط
          ?min_strength=0.3 — حد أدنى للقوة
        """
        ckg = _get_ckg()
        if not ckg:
            return jsonify({"error": "CKG not initialised", "concepts": []}), 503

        cluster      = request.args.get("cluster")
        top          = request.args.get("top", type=int)
        min_strength = request.args.get("min_strength", 0.0, type=float)

        concepts = ckg.all_concepts()

        if cluster:
            concepts = [c for c in concepts if c["cluster"] == cluster]
        if min_strength > 0:
            concepts = [c for c in concepts if c["strength"] >= min_strength]

        concepts.sort(key=lambda c: c["strength"], reverse=True)

        if top:
            concepts = concepts[:top]

        return jsonify({
            "concepts":      concepts,
            "total":         len(concepts),
            "cluster_filter": cluster,
            "stats":         ckg.stats(),
        })

    # ── GET /knowledge/relations ──────────────────────────────────────────
    @app.route("/knowledge/relations", methods=["GET"])
    def knowledge_relations():
        """
        كل العلاقات في الـ CKG.
        Query params:
          ?concept=صبر      — علاقات مفهوم محدد
          ?top=N            — أقوى N علاقة
          ?type=co_occurrence — فلتر بنوع العلاقة
          ?min_weight=0.3
        """
        ckg = _get_ckg()
        if not ckg:
            return jsonify({"error": "CKG not initialised", "relations": []}), 503

        concept    = request.args.get("concept")
        top        = request.args.get("top", 100, type=int)
        rel_type   = request.args.get("type")
        min_weight = request.args.get("min_weight", 0.0, type=float)

        if concept:
            related = ckg.query_related(concept, top_k=top)
            relations = []
            for target, weight in related:
                r = ckg.get_relation(concept, target)
                if r and weight >= min_weight:
                    relations.append(r.to_dict())
        else:
            relations = ckg.all_relations()
            if rel_type:
                relations = [r for r in relations if r["relation_type"] == rel_type]
            if min_weight > 0:
                relations = [r for r in relations if r["weight"] >= min_weight]
            relations.sort(key=lambda r: r["weight"], reverse=True)
            relations = relations[:top]

        return jsonify({
            "relations": relations,
            "total":     len(relations),
        })

    # ── GET /knowledge/query ──────────────────────────────────────────────
    @app.route("/knowledge/query", methods=["GET"])
    def knowledge_query():
        """
        استعلام ذكي: ما المفاهيم المرتبطة بـ concept؟
        Query params:
          ?concept=صبر (required)
          ?top=10
          ?direction=both|in|out
        """
        ckg = _get_ckg()
        if not ckg:
            return jsonify({"error": "CKG not initialised"}), 503

        concept   = request.args.get("concept", "").strip()
        top       = request.args.get("top", 10, type=int)
        direction = request.args.get("direction", "both")

        if not concept:
            return jsonify({"error": "concept parameter required"}), 400

        c = ckg.get_concept(concept)
        if not c:
            return jsonify({
                "concept": concept,
                "found":   False,
                "related": [],
            })

        related = ckg.query_related(concept, top_k=top, direction=direction)
        path_to = {}
        for target, _ in related[:5]:
            p = ckg.find_path(concept, target, max_depth=4)
            if p:
                path_to[target] = p

        return jsonify({
            "concept":   c.to_dict(),
            "found":     True,
            "related":   [{"concept": n, "weight": w} for n, w in related],
            "paths":     path_to,
        })

    # ── GET /knowledge/stats ──────────────────────────────────────────────
    @app.route("/knowledge/stats", methods=["GET"])
    def knowledge_stats():
        """إحصائيات شاملة للـ CKG — يقرأ مباشرة من الكائن الحي."""
        from knowledge.cognitive_graph import get_ckg
        from datetime import datetime, timezone, timedelta
        ckg = get_ckg()
        since_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        return jsonify({
            **ckg.stats(),
            "cluster_summary":        ckg.cluster_summary(),
            "cross_source_concepts":  ckg.cross_source_concepts(min_sources=2),
            "concept_growth_24h":     ckg.concept_growth_rate(since_24h),
        })

    # ── GET /knowledge/path ───────────────────────────────────────────────
    @app.route("/knowledge/path", methods=["GET"])
    def knowledge_path():
        """
        مسار بين مفهومين.
        ?from=صبر&to=آخرة
        """
        ckg = _get_ckg()
        if not ckg:
            return jsonify({"error": "CKG not initialised"}), 503

        start = request.args.get("from", "").strip()
        end   = request.args.get("to", "").strip()

        if not start or not end:
            return jsonify({"error": "from and to required"}), 400

        path = ckg.find_path(start, end)
        return jsonify({
            "from":  start,
            "to":    end,
            "path":  path,
            "found": path is not None,
            "hops":  len(path) - 1 if path else None,
        })

    # ── GET /knowledge/search ─────────────────────────────────────────────
    @app.route("/knowledge/search", methods=["GET"])
    def knowledge_search():
        """
        بحث دلالي في آيات القرآن الكريم.

        Query params:
          ?q=نص البحث         — بحث بالنص (keyword + concept matching)
          ?concept=إيمان      — بحث بمفهوم محدد من الـ CKG
          ?cluster=عبادة      — بحث بمجموعة مفاهيم (cluster)
          ?top=10             — عدد النتائج (افتراضي 10، أقصى 100)
          ?min_quality=0      — أدنى درجة جودة (0-100)
          ?include_text=true  — إظهار النص الكامل للآية
        """
        import re as _re

        q           = request.args.get("q", "").strip()
        concept_q   = request.args.get("concept", "").strip()
        cluster_q   = request.args.get("cluster", "").strip()
        top         = min(request.args.get("top", 10, type=int), 100)
        min_quality = request.args.get("min_quality", 0.0, type=float)
        include_txt = request.args.get("include_text", "true").lower() != "false"

        if not q and not concept_q and not cluster_q:
            return jsonify({"error": "أحد الحقول مطلوب: q أو concept أو cluster"}), 400

        ckg = _get_ckg()

        # ── Step 1: تحديد المفاهيم المستهدفة ────────────────────────────
        target_concepts: Dict[str, float] = {}   # concept_name → relevance_score

        if concept_q:
            # بحث مباشر بالمفهوم + جيرانه في الـ CKG
            if ckg and ckg.get_concept(concept_q):
                target_concepts[concept_q] = 1.0
                for related_name, rel_weight in ckg.query_related(concept_q, top_k=5):
                    target_concepts[related_name] = max(
                        target_concepts.get(related_name, 0.0),
                        rel_weight * 0.6,
                    )
            else:
                target_concepts[concept_q] = 1.0   # نحاول حتى لو غير موجود في CKG

        if cluster_q and ckg:
            # كل مفاهيم المجموعة المحددة
            summary = ckg.cluster_summary()
            for c_name, c_info in ckg._concepts.items():
                if c_info.cluster == cluster_q:
                    target_concepts[c_name] = max(
                        target_concepts.get(c_name, 0.0), 0.85
                    )

        if q:
            # استخراج مفاهيم من النص باستخدام ConceptExtractor
            try:
                from knowledge_sources.concept_extractor import ConceptExtractor
                ce = ConceptExtractor()
                matches = ce.extract(q)
                for m in matches:
                    target_concepts[m.concept] = max(
                        target_concepts.get(m.concept, 0.0), m.score
                    )
            except Exception as _exc:
                logger.warning(f"[/knowledge/search] ConceptExtractor error: {_exc}")

            # بحث إضافي: هل النص يطابق اسم مفهوم مباشرة؟
            if ckg:
                for c_name in ckg._concepts:
                    if q in c_name or c_name in q:
                        target_concepts[c_name] = max(
                            target_concepts.get(c_name, 0.0), 0.75
                        )

        if not target_concepts:
            return jsonify({
                "query":   {"q": q, "concept": concept_q, "cluster": cluster_q},
                "results": [],
                "total":   0,
                "message": "لم يُعثر على مفاهيم مطابقة في الـ CKG",
            })

        # ── Step 2: البحث في KnowledgeStore عن آيات مطابقة ─────────────
        try:
            ks_data  = mesh.knowledge.read_node_profiles() if mesh.knowledge else {}
        except Exception:
            ks_data  = {}

        all_nodes = ks_data.get("nodes", {})
        quran_nodes = {
            k: v for k, v in all_nodes.items()
            if k.startswith("ks:quran:")
        }

        scored: list = []
        for node_id, node in quran_nodes.items():
            quality = node.get("quality_score", 0.0)
            if quality < min_quality:
                continue

            derived = node.get("derived_concepts", [])
            tags    = node.get("tags", [])

            # حساب درجة التطابق
            concept_score = 0.0
            matched       = []
            for c_name, c_rel in target_concepts.items():
                if c_name in derived:
                    concept_score += c_rel
                    matched.append(c_name)
                # مطابقة جزئية في الـ tags
                elif any(c_name in t or t in c_name for t in tags):
                    concept_score += c_rel * 0.4
                    matched.append(f"~{c_name}")

            # بحث نصي مباشر إذا كان q موجوداً
            text_bonus = 0.0
            raw_content = node.get("raw_content", "")
            if q and raw_content and raw_content != "[protected]":
                clean_q    = _re.sub(r"[\u064B-\u065F]", "", q)        # بدون تشكيل
                clean_text = _re.sub(r"[\u064B-\u065F]", "", raw_content)
                if clean_q and clean_q in clean_text:
                    text_bonus = 0.5

            total_score = concept_score + text_bonus
            if total_score <= 0.0:
                continue

            # بناء نتيجة واحدة
            entry = {
                "reference":      node.get("raw_reference", node_id.replace("ks:", "")),
                "node_id":        node_id,
                "score":          round(total_score, 4),
                "quality":        round(quality, 1),
                "matched_concepts": matched,
                "tags":           tags[:6],
            }
            if include_txt and raw_content and raw_content != "[protected]":
                entry["text"] = raw_content[:300]

            scored.append(entry)

        # ترتيب: score تنازلياً ثم quality
        scored.sort(key=lambda x: (-x["score"], -x["quality"]))
        results = scored[:top]

        # ── Step 3: إحصائيات المفاهيم في النتائج ────────────────────────
        concept_freq: Dict[str, int] = {}
        for r in results:
            for c in r.get("matched_concepts", []):
                concept_freq[c] = concept_freq.get(c, 0) + 1

        return jsonify({
            "query": {
                "q":       q or None,
                "concept": concept_q or None,
                "cluster": cluster_q or None,
                "top":     top,
            },
            "results":          results,
            "total":            len(results),
            "total_candidates": len(scored),
            "target_concepts":  [
                {"concept": c, "relevance": round(s, 3)}
                for c, s in sorted(target_concepts.items(), key=lambda x: -x[1])
            ],
            "concept_frequency": dict(
                sorted(concept_freq.items(), key=lambda x: -x[1])[:10]
            ),
        })


_create_app_ks = create_app
def create_app(mesh):
    app = _create_app_ks(mesh)
    _add_ckg_routes(app, mesh)
    return app


# ═══════════════════════════════════════════════════════════════════════════
# CKG Auto-Bootstrap on startup + Status endpoint
# ═══════════════════════════════════════════════════════════════════════════

def _add_ckg_bootstrap_routes(app, mesh):
    """
    يشغّل بناء CKG تلقائياً عند أول طلب /knowledge/*
    ويضيف endpoint /knowledge/bootstrap للتحكم اليدوي.
    """
    try:
        from knowledge.ckg_bootstrap import bootstrap_ckg, is_bootstrap_done, wait_for_bootstrap
        _BOOTSTRAP_AVAILABLE = True
    except ImportError:
        _BOOTSTRAP_AVAILABLE = False
        logger.warning("[CKG Bootstrap] module not available")

    # ── Auto-bootstrap on first /knowledge/* request ──────────────────────
    _boot_triggered = [False]

    @app.before_request
    def maybe_bootstrap():
        from flask import request as req
        if (
            _BOOTSTRAP_AVAILABLE
            and req.path.startswith('/knowledge/')
            and not _boot_triggered[0]
            and not is_bootstrap_done()
        ):
            _boot_triggered[0] = True
            bootstrap_ckg(mesh, background=True)

    # ── GET /knowledge/bootstrap ──────────────────────────────────────────
    @app.route("/knowledge/bootstrap", methods=["GET", "POST"])
    def knowledge_bootstrap():
        """
        GET  → حالة البناء
        POST → ابدأ بناء جديد (force=true لإعادة البناء)
        """
        if not _BOOTSTRAP_AVAILABLE:
            return jsonify({"error": "ckg_bootstrap module not available"}), 503

        from flask import request as req
        from knowledge.ckg_bootstrap import bootstrap_ckg, is_bootstrap_done

        if req.method == "GET":
            return jsonify({
                "done":    is_bootstrap_done(),
                "message": "الجراف المعرفي جاهز" if is_bootstrap_done() else "جارٍ البناء أو لم يبدأ بعد",
            })

        # POST — trigger build
        data  = req.get_json(force=True) or {}
        force = data.get("force", False)
        bootstrap_ckg(mesh, background=True, force=force)
        return jsonify({
            "status":  "started",
            "message": "بدأ بناء الجراف المعرفي في الخلفية",
            "force":   force,
        })


_create_app_ckg = create_app
def create_app(mesh):
    app = _create_app_ckg(mesh)
    _add_ckg_bootstrap_routes(app, mesh)
    return app


# ═══════════════════════════════════════════════════════════════════════════
# Arabic NLP Routes — Phase 18
# ═══════════════════════════════════════════════════════════════════════════

def _add_arabic_nlp_routes(app, mesh):
    """
    Arabic language understanding endpoints.

    Architecture: 3 analysis layers → 7-column feature vector → weight matrix
      Layer 1 نحوي  — POST /arabic/syntax
      Layer 2 صرفي  — POST /arabic/morphology
      Layer 3 دلالي — POST /arabic/semantic
      Full analysis — POST /arabic/analyze
      Neural train  — POST /arabic/train
      Status        — GET  /arabic/status
    """
    from ai.arabic_nlp import get_arabic_engine

    def _get_engine():
        ckg = None
        try:
            from knowledge.cognitive_graph import CognitiveMindGraph
            ckg = getattr(mesh, "ckg", None) or getattr(mesh, "_ckg", None)
        except Exception:
            pass
        return get_arabic_engine(ckg=ckg)

    # ── GET /arabic/status ────────────────────────────────────────────────
    @app.route("/arabic/status", methods=["GET"])
    def arabic_status():
        """حالة محرك فهم اللغة العربية."""
        try:
            eng = _get_engine()
            return jsonify(eng.status())
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # ── POST /arabic/analyze ──────────────────────────────────────────────
    @app.route("/arabic/analyze", methods=["POST"])
    def arabic_analyze():
        """
        تحليل كامل (نحوي + صرفي + دلالي) وإنتاج مُتجه الميزات (7 عناصر).

        Body: { "text": "النص العربي هنا" }

        Returns full 3-layer analysis + 7-element feature vector.
        """
        b    = request.get_json(force=True) or {}
        text = (b.get("text") or b.get("نص") or "").strip()
        if not text:
            return jsonify({"error": "حقل text مطلوب"}), 400
        try:
            eng    = _get_engine()
            result = eng.analyse(text)
            return jsonify(result.to_dict())
        except Exception as exc:
            logger.error(f"[/arabic/analyze] {exc}")
            return jsonify({"error": str(exc)}), 500

    # ── POST /arabic/syntax ────────────────────────────────────────────────
    @app.route("/arabic/syntax", methods=["POST"])
    def arabic_syntax():
        """
        الطبقة 1 — التحليل النحوي (جمل، أفعال، أسماء، حروف).

        Body: { "text": "..." }
        """
        b    = request.get_json(force=True) or {}
        text = (b.get("text") or b.get("نص") or "").strip()
        if not text:
            return jsonify({"error": "حقل text مطلوب"}), 400
        try:
            eng = _get_engine()
            return jsonify(eng.syntactic_only(text))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # ── POST /arabic/morphology ───────────────────────────────────────────
    @app.route("/arabic/morphology", methods=["POST"])
    def arabic_morphology():
        """
        الطبقة 2 — التحليل الصرفي (جذور، أوزان).

        Body: { "text": "..." }
        """
        b    = request.get_json(force=True) or {}
        text = (b.get("text") or b.get("نص") or "").strip()
        if not text:
            return jsonify({"error": "حقل text مطلوب"}), 400
        try:
            eng = _get_engine()
            return jsonify(eng.morphological_only(text))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # ── POST /arabic/semantic ─────────────────────────────────────────────
    @app.route("/arabic/semantic", methods=["POST"])
    def arabic_semantic():
        """
        الطبقة 3 — التحليل الدلالي (معاني، مفاهيم، سياقات).

        Body: { "text": "..." }
        """
        b    = request.get_json(force=True) or {}
        text = (b.get("text") or b.get("نص") or "").strip()
        if not text:
            return jsonify({"error": "حقل text مطلوب"}), 400
        try:
            eng = _get_engine()
            return jsonify(eng.semantic_only(text))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # ── POST /arabic/train ────────────────────────────────────────────────
    @app.route("/arabic/train", methods=["POST"])
    def arabic_train():
        """
        تدريب مصفوفة الأوزان العصبية باستخدام نتيجة التحليل العربي.

        يُحلَّل النص → مُتجه 7 عناصر → يُغذَّى في NeuralWeightLayer/DynamicWeightLayer.

        Body: { "text": "...", "target": 0.85 }
          target: درجة الجودة المستهدفة (0.0-1.0)
        """
        b      = request.get_json(force=True) or {}
        text   = (b.get("text") or b.get("نص") or "").strip()
        target = float(b.get("target", 0.75))

        if not text:
            return jsonify({"error": "حقل text مطلوب"}), 400
        if not 0.0 <= target <= 1.0:
            return jsonify({"error": "target يجب أن يكون بين 0.0 و 1.0"}), 400

        # Get the active neural layer from the mesh
        neural_layer = None
        try:
            neural_layer = getattr(mesh, "dynamic_layer", None)
            if neural_layer is None:
                neural_layer = getattr(mesh, "neural_layer", None)
        except Exception:
            pass

        if neural_layer is None:
            return jsonify({"error": "لا توجد طبقة أوزان عصبية نشطة"}), 503

        try:
            eng    = _get_engine()
            result = eng.analyse_and_train(text, target, neural_layer)
            return jsonify(result)
        except Exception as exc:
            logger.error(f"[/arabic/train] {exc}")
            return jsonify({"error": str(exc)}), 500

    # ── POST /arabic/batch ────────────────────────────────────────────────
    @app.route("/arabic/batch", methods=["POST"])
    def arabic_batch():
        """
        تحليل دُفعة من النصوص دفعةً واحدة.

        Body: { "texts": ["نص1", "نص2", ...] }
        """
        b     = request.get_json(force=True) or {}
        texts = b.get("texts", [])
        if not texts or not isinstance(texts, list):
            return jsonify({"error": "حقل texts (قائمة) مطلوب"}), 400
        if len(texts) > 100:
            return jsonify({"error": "الحد الأقصى 100 نص في الدُّفعة الواحدة"}), 400
        try:
            eng     = _get_engine()
            results = eng.batch_analyse(texts)
            return jsonify({
                "count":   len(results),
                "results": [r.to_dict() for r in results],
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500


_create_app_arabic = create_app
def create_app(mesh):
    app = _create_app_arabic(mesh)
    _add_arabic_nlp_routes(app, mesh)
    _add_knowledge_trainer_routes(app, mesh)
    return app


# ══════════════════════════════════════════════════════════════════════════
# Knowledge Trainer API — محرك التدريب المعرفي المدمج
# ══════════════════════════════════════════════════════════════════════════

def _add_knowledge_trainer_routes(app, mesh):
    """
    نقاط API للتدريب المعرفي المدمج.
    المعرفة → متجه 7 أبعاد → مصفوفة الأوزان + cognitive_graph.json + SQLite
    """
    from ai.knowledge_trainer import KnowledgeTrainer

    _trainer_cache: dict = {}

    def _trainer() -> KnowledgeTrainer:
        if "t" not in _trainer_cache:
            _trainer_cache["t"] = KnowledgeTrainer(mesh)
        return _trainer_cache["t"]

    # ── GET /train/status ─────────────────────────────────────────────────
    @app.route("/train/status", methods=["GET"])
    def train_status():
        """حالة محرك التدريب: المصفوفة، CKG، SQLite."""
        return jsonify(_trainer().stats())

    # ── POST /train/domain ─────────────────────────────────────────────────
    @app.route("/train/domain", methods=["POST"])
    def train_domain():
        """
        دِرِّب على مجال محدد من المصادر المدمجة.
        Body: { "domain": "physics|math|history|biology|civilizations" }
              أو "all" لتدريب جميع المجالات دفعة واحدة.
        """
        try:
            from knowledge_sources.domain_sources import get_all_domain_items, \
                get_physics_items, get_math_items, get_history_items, \
                get_biology_items, get_civilizations_items
        except ImportError as e:
            return jsonify({"error": f"domain_sources غير متاح: {e}"}), 500

        b      = request.get_json(force=True) or {}
        domain = b.get("domain", "all").strip().lower()
        t      = _trainer()

        DOMAIN_MAP = {
            "physics":       get_physics_items,
            "math":          get_math_items,
            "history":       get_history_items,
            "biology":       get_biology_items,
            "civilizations": get_civilizations_items,
        }

        if domain == "all":
            result = t.train_all(get_all_domain_items())
        elif domain in DOMAIN_MAP:
            items  = DOMAIN_MAP[domain]()
            result = t.train_domain(domain, items)
        else:
            return jsonify({
                "error": f"المجال '{domain}' غير معروف",
                "valid": ["all"] + list(DOMAIN_MAP.keys()),
            }), 400

        return jsonify(result)

    # ── POST /train/wikipedia ─────────────────────────────────────────────
    @app.route("/train/wikipedia", methods=["POST"])
    def train_wikipedia():
        """
        جلب وتدريب من ويكيبيديا العربية.
        Body: { "topics": [...], "max_items": 40 }  (اختياري)
        """
        try:
            from knowledge_sources.web_fetcher import fetch_wikipedia_items
        except ImportError as e:
            return jsonify({"error": str(e)}), 500

        b        = request.get_json(force=True) or {}
        topics   = b.get("topics")       # None = القائمة الافتراضية
        max_items= int(b.get("max_items", 40))

        items  = fetch_wikipedia_items(topics=topics, max_items=max_items)
        if not items:
            return jsonify({"warning": "لم تُجلَب أي مقالات — تحقق من الاتصال",
                            "items": 0})

        result = _trainer().train_domain("wikipedia", items)
        return jsonify(result)

    # ── POST /train/github ────────────────────────────────────────────────
    @app.route("/train/github", methods=["POST"])
    def train_github():
        """
        جلب وتدريب من GitHub.
        Body: { "queries": [...], "max_total": 50 }  (اختياري)
        """
        try:
            from knowledge_sources.web_fetcher import fetch_github_items
        except ImportError as e:
            return jsonify({"error": str(e)}), 500

        b         = request.get_json(force=True) or {}
        queries   = b.get("queries")
        max_total = int(b.get("max_total", 50))

        items  = fetch_github_items(queries=queries, max_total=max_total)
        if not items:
            return jsonify({"warning": "لم تُجلَب أي مستودعات — تحقق من الاتصال",
                            "items": 0})

        result = _trainer().train_domain("github", items)
        return jsonify(result)

    # ── POST /train/all ────────────────────────────────────────────────────
    @app.route("/train/all", methods=["POST"])
    def train_all_sources():
        """
        دِرِّب على جميع المصادر: المجالات + ويكيبيديا + GitHub.
        Body: { "include_web": true, "wiki_max": 30, "github_max": 40 }
        """
        try:
            from knowledge_sources.domain_sources import get_all_domain_items
            from knowledge_sources.web_fetcher    import fetch_wikipedia_items, fetch_github_items
        except ImportError as e:
            return jsonify({"error": str(e)}), 500

        b           = request.get_json(force=True) or {}
        include_web = bool(b.get("include_web", True))
        wiki_max    = int(b.get("wiki_max",    30))
        github_max  = int(b.get("github_max",  40))

        all_items = get_all_domain_items()

        if include_web:
            wiki_items   = fetch_wikipedia_items(max_items=wiki_max)
            github_items = fetch_github_items(max_total=github_max)
            if wiki_items:
                all_items["wikipedia"] = wiki_items
            if github_items:
                all_items["github"] = github_items

        result = _trainer().train_all(all_items)
        return jsonify(result)

    # ── POST /train/custom ────────────────────────────────────────────────
    @app.route("/train/custom", methods=["POST"])
    def train_custom():
        """
        تدريب على معلومات مخصصة من المستخدم.
        Body: {
          "domain": "general",
          "items": [
            { "concept": "...", "text": "...", "cluster": "...",
              "importance": 0.8, "certainty": 0.9, "relations": [] }
          ]
        }
        """
        b     = request.get_json(force=True) or {}
        domain= b.get("domain", "general")
        items = b.get("items", [])

        if not items:
            return jsonify({"error": "items مطلوب"}), 400
        if len(items) > 500:
            return jsonify({"error": "الحد الأقصى 500 عنصر في الطلب الواحد"}), 400

        result = _trainer().train_domain(domain, items)
        return jsonify(result)

    # ── GET /train/ckg ────────────────────────────────────────────────────
    @app.route("/train/ckg", methods=["GET"])
    def train_ckg_stats():
        """إحصائيات الـ CKG الحالية."""
        import json
        from pathlib import Path
        ckg_path = Path("./knowledge/cognitive_graph.json")
        if not ckg_path.exists():
            return jsonify({"error": "cognitive_graph.json غير موجود"}), 404
        with open(ckg_path, encoding="utf-8") as f:
            data = json.load(f)
        meta     = data.get("_meta", {})
        concepts = data.get("concepts", {})
        relations= data.get("relations", {})

        clusters: dict = {}
        for c in concepts.values():
            cl = c.get("cluster", "unknown")
            clusters[cl] = clusters.get(cl, 0) + 1

        return jsonify({
            "meta":             meta,
            "total_concepts":   len(concepts),
            "total_relations":  len(relations),
            "clusters":         clusters,
            "top_concepts": sorted(
                [{"name": n, "strength": v.get("strength", 0),
                  "frequency": v.get("frequency", 0)}
                 for n, v in concepts.items()],
                key=lambda x: x["strength"], reverse=True
            )[:20],
        })

    # ── POST /train/ask ───────────────────────────────────────────────────
    @app.route("/train/ask", methods=["POST"])
    def train_ask():
        """
        استعلام معرفي عن مفهوم من الـ CKG.
        Body: { "concept": "الجاذبية" }
        Returns: confidence_score, related_concepts, sources,
                 cross_domain_connections, quran_references
        """
        import re as _re
        from pathlib import Path
        import json as _json

        b       = request.get_json(force=True) or {}
        concept = (b.get("concept") or "").strip()
        if not concept:
            return jsonify({"error": "حقل concept مطلوب"}), 400

        # ── Load CKG directly from file (always up-to-date) ──────────────
        ckg_path = Path("./knowledge/cognitive_graph.json")
        if not ckg_path.exists():
            return jsonify({"error": "CKG غير موجود — قم بتشغيل التدريب أولاً"}), 503

        try:
            with open(ckg_path, encoding="utf-8") as _f:
                ckg_data = _json.load(_f)
        except Exception as exc:
            return jsonify({"error": f"خطأ في قراءة CKG: {exc}"}), 500

        all_concepts = ckg_data.get("concepts", {})
        all_relations = ckg_data.get("relations", {})

        # ── Normalize for fuzzy matching ──────────────────────────────────
        def _norm(t: str) -> str:
            t = _re.sub(r'[\u064B-\u065F\u0670\u0640]', '', t)
            t = _re.sub(r'[أإآٱ]', 'ا', t)
            return t.strip()

        norm_concept = _norm(concept)

        # Try exact match first, then normalized match
        concept_data = all_concepts.get(concept)
        matched_key  = concept
        if concept_data is None:
            for k, v in all_concepts.items():
                if _norm(k) == norm_concept:
                    concept_data = v
                    matched_key  = k
                    break

        # ── Confidence score ──────────────────────────────────────────────
        if concept_data:
            raw_strength    = float(concept_data.get("strength", 0.0))
            raw_frequency   = int(concept_data.get("frequency", 0))
            freq_score      = min(raw_frequency / 20.0, 1.0)
            confidence_score = round(min(raw_strength * 0.6 + freq_score * 0.4, 1.0), 4)
        else:
            confidence_score = 0.0

        # ── Related concepts & cross-domain connections ───────────────────
        related_concepts: list = []
        cross_domain_connections: list = []
        my_cluster = (concept_data or {}).get("cluster", "")

        # Gather neighbours from relations dict
        neighbours: list = []
        for rel_key, rel_val in all_relations.items():
            src = rel_val.get("source", "")
            tgt = rel_val.get("target", "")
            w   = float(rel_val.get("weight", 0.0))
            if _norm(src) == norm_concept:
                neighbours.append((tgt, w))
            elif _norm(tgt) == norm_concept:
                neighbours.append((src, w))

        neighbours.sort(key=lambda x: -x[1])

        seen_related: set = set()
        for nbr_name, _ in neighbours[:15]:
            if nbr_name in seen_related or _norm(nbr_name) == norm_concept:
                continue
            seen_related.add(nbr_name)
            nbr_data    = all_concepts.get(nbr_name, {})
            nbr_cluster = nbr_data.get("cluster", "")
            if nbr_cluster and nbr_cluster != my_cluster:
                cross_domain_connections.append(nbr_cluster)
            else:
                related_concepts.append(nbr_name)

        # De-duplicate cross-domain clusters and cap lists
        cross_domain_connections = list(dict.fromkeys(cross_domain_connections))[:6]
        related_concepts         = related_concepts[:8]

        # ── Sources (domain clusters from concept data) ───────────────────
        raw_sources = (concept_data or {}).get("sources", [])

        # Quran refs look like "chapter:verse" e.g. "2:255"
        quran_re      = _re.compile(r'^\d+:\d+$')
        quran_refs    = [s for s in raw_sources if quran_re.match(str(s))]
        domain_sources = list({
            all_concepts.get(s, {}).get("cluster", "")
            for s in raw_sources
            if not quran_re.match(str(s)) and all_concepts.get(s)
        } - {""})

        # If concept itself has a cluster, add it as a source
        if my_cluster and my_cluster not in domain_sources:
            domain_sources.insert(0, my_cluster)

        return jsonify({
            "concept":                 matched_key if concept_data else concept,
            "found_in_ckg":            concept_data is not None,
            "confidence_score":        confidence_score,
            "related_concepts":        related_concepts,
            "sources":                 domain_sources[:6],
            "cross_domain_connections": cross_domain_connections,
            "quran_references":        quran_refs[:10],
        })

    # ── GET /train/audit ──────────────────────────────────────────────────
    @app.route("/train/audit", methods=["GET"])
    def train_audit():
        """
        Forensic training audit — all values read from persisted storage.
        training_steps = SELECT COUNT(*) FROM knowledge_training (mesh.db)
        concepts / relations = live CKG object
        weights_saved = file existence check
        """
        import sqlite3 as _sqlite3
        from pathlib import Path as _Path
        from knowledge.cognitive_graph import get_ckg as _get_ckg

        db_path = _Path("./data/mesh.db")
        training_steps    = 0
        training_sessions = 0
        training_by_domain: dict = {}
        recent_avg_loss   = None

        if db_path.exists():
            try:
                conn = _sqlite3.connect(str(db_path))
                cur  = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM knowledge_training")
                training_steps = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM training_sessions")
                training_sessions = cur.fetchone()[0]
                cur.execute(
                    "SELECT domain, COUNT(*) FROM knowledge_training GROUP BY domain"
                )
                training_by_domain = {r[0]: r[1] for r in cur.fetchall()}
                cur.execute(
                    "SELECT AVG(avg_loss) FROM training_sessions"
                )
                row = cur.fetchone()
                if row and row[0] is not None:
                    recent_avg_loss = round(float(row[0]), 6)
                conn.close()
            except Exception as _e:
                logger.warning(f"[/train/audit] DB read error: {_e}")

        ckg = _get_ckg()
        weights_path = _Path("./checkpoints/neural_weights.npy")

        cursor_path = _Path("./data/quran_training_cursor.json")
        cursor: dict = {}
        if cursor_path.exists():
            try:
                import json as _json
                with open(cursor_path, encoding="utf-8") as _f:
                    cursor = _json.load(_f)
            except Exception:
                pass

        return jsonify({
            "training_steps":         training_steps,
            "training_sessions":      training_sessions,
            "training_by_domain":     training_by_domain,
            "recent_avg_loss":        recent_avg_loss,
            "concepts":               ckg.concept_count(),
            "relations":              ckg.relation_count(),
            "weights_saved":          weights_path.exists(),
            "weights_path":           str(weights_path) if weights_path.exists() else None,
            "quran_training_cursor":  cursor,
        })

    # ── GET /train/quran-audit ────────────────────────────────────────────
    @app.route("/train/quran-audit", methods=["GET"])
    def train_quran_audit():
        """
        Quran ingestion audit — reads from persisted files only.
        ayah_count = quran_index.json (not runtime memory)
        """
        import json as _json
        from pathlib import Path as _Path

        qi_path = _Path("./knowledge/quran_index.json")
        ayah_count   = 0
        chunk_count  = 0
        stored_at    = None
        total_surahs = None

        if qi_path.exists():
            try:
                with open(qi_path, encoding="utf-8") as _f:
                    qi = _json.load(_f)
                ayah_count   = qi.get("total_ayat",   0)
                chunk_count  = qi.get("total_chunks",  0)
                stored_at    = qi.get("stored_at")
                total_surahs = qi.get("total_surahs")
            except Exception as _e:
                logger.warning(f"[/train/quran-audit] quran_index read error: {_e}")

        np_path = _Path("./knowledge/node_profiles.json")
        node_count = 0
        if np_path.exists():
            try:
                with open(np_path, encoding="utf-8") as _f:
                    np_data = _json.load(_f)
                node_count = len(np_data.get("nodes", {}))
            except Exception:
                pass

        cursor_path = _Path("./data/quran_training_cursor.json")
        cursor: dict = {}
        if cursor_path.exists():
            try:
                with open(cursor_path, encoding="utf-8") as _f:
                    cursor = _json.load(_f)
            except Exception:
                pass

        hashes_path = _Path("./data/ks_seen_hashes.json")

        return jsonify({
            "ayah_count":            ayah_count,
            "chunk_count":           chunk_count,
            "total_surahs":          total_surahs,
            "stored_at":             stored_at,
            "node_profiles_count":   node_count,
            "training_cursor":       cursor,
            "quran_index_path":      str(qi_path),
            "hashes_file_exists":    hashes_path.exists(),
            "chunk_files_on_disk":   len(list(_Path("./knowledge").glob("quran_chunk_*.json"))),
        })

    # ── POST /train/intensive ─────────────────────────────────────────────
    @app.route("/train/intensive", methods=["POST"])
    def train_intensive():
        """
        High-intensity direct training on DynamicWeightLayer.
        Drives train_step() with structured vectors until loss < target_loss.
        Body: { "steps": 10000, "target_loss": 0.1, "lr_boost": 3.0 }
        Requires X-Admin-Key header matching ADMIN_KEY env var when that var is set.
        """
        import numpy as np
        import time

        required_key = os.environ.get("ADMIN_KEY", "").strip()
        if required_key:
            provided = request.headers.get("X-Admin-Key", "").strip()
            if provided != required_key:
                return jsonify({"error": "Unauthorized"}), 401

        b           = request.get_json(force=True) or {}
        max_steps   = min(int(b.get("steps", 10000)), 50000)
        target_loss = float(max(0.01, min(b.get("target_loss", 0.1), 1.0)))
        lr_boost    = float(max(1.0, min(b.get("lr_boost", 3.0), 10.0)))

        layer = getattr(mesh, "dynamic_layer", None) or getattr(mesh, "neural_layer", None)
        if layer is None:
            return jsonify({"error": "No weight layer available"}), 503

        original_lr = layer.learning_rate
        layer.learning_rate = original_lr * lr_boost

        rng = np.random.default_rng(42)
        TRAINING_PAIRS = []
        for _ in range(200):
            vec = [
                rng.uniform(0.6, 1.0),
                rng.uniform(0.5, 1.0),
                rng.uniform(0.3, 0.8),
                rng.uniform(0.2, 0.9),
                rng.uniform(0.4, 1.0),
                rng.uniform(0.1, 0.7),
                rng.uniform(0.3, 0.9),
            ]
            target = rng.uniform(0.7, 1.0)
            TRAINING_PAIRS.append((vec, float(target)))

        t0          = time.time()
        steps_done  = 0
        last_loss   = None
        loss_window = []

        for step in range(max_steps):
            vec, target = TRAINING_PAIRS[step % len(TRAINING_PAIRS)]
            loss = layer.train_step(vec, target)
            last_loss = loss
            loss_window.append(loss)
            if len(loss_window) > 100:
                loss_window.pop(0)
            steps_done += 1

            if step % 500 == 0 and len(loss_window) >= 50:
                avg = sum(loss_window[-50:]) / 50
                if avg < target_loss:
                    break

        layer.learning_rate = original_lr
        elapsed = time.time() - t0
        avg_recent = sum(loss_window[-50:]) / max(len(loss_window[-50:]), 1)

        try:
            mesh.save_neural_weights()
        except Exception:
            pass

        return jsonify({
            "steps_done":   steps_done,
            "last_loss":    round(last_loss, 8) if last_loss is not None else None,
            "avg_loss_50":  round(avg_recent, 8),
            "target_loss":  target_loss,
            "target_reached": avg_recent < target_loss,
            "elapsed_s":    round(elapsed, 2),
            "lr_used":      round(original_lr * lr_boost, 6),
        })

    # ── POST /train/save-weights ──────────────────────────────────────────
    @app.route("/train/save-weights", methods=["POST"])
    def train_save_weights():
        """Save current neural weights to checkpoints/neural_weights.npy."""
        try:
            result = mesh.save_neural_weights()
            return jsonify({"saved": True, "path": str(result.get("path", ""))})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── GET /train/matrix ─────────────────────────────────────────────────
    @app.route("/train/matrix", methods=["GET"])
    def train_matrix_status():
        """حالة مصفوفة الأوزان الحالية."""
        layer = getattr(mesh, "dynamic_layer", None) or \
                getattr(mesh, "neural_layer", None)
        if layer is None:
            return jsonify({"error": "لا توجد طبقة أوزان"}), 503
        return jsonify({
            "shape":       list(layer.weights.shape),
            "train_steps": getattr(layer, "_train_steps", 0),
            "last_loss":   getattr(layer, "_last_loss", None),
            "weight_stats": {
                "min":  round(float(layer.weights.min()), 6),
                "max":  round(float(layer.weights.max()), 6),
                "mean": round(float(layer.weights.mean()), 6),
                "std":  round(float(layer.weights.std()),  6),
            },
            "dimensions": {
                "0_IMPORTANCE":  "أهمية المعلومة",
                "1_CERTAINTY":   "درجة اليقين",
                "2_ABSTRACTION": "مستوى التجريد",
                "3_DOMAIN":      "رمز المجال",
                "4_CONNECTIVITY":"كثافة العلاقات",
                "5_TEMPORALITY": "الزمنية",
                "6_NOVELTY":     "جِدَّة المعلومة",
            },
        })
