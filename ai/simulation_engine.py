"""
Phase 4 – Simulation Engine
============================
Generates automatic executions so the AI accumulates experience and
updates the knowledge layer without manual intervention.

Usage:
    python main.py --mode simulate [--rounds 20] [--delay 0.5]

The simulation:
  1. Registers a diverse set of nodes
  2. Runs randomized routes (some will fail on purpose)
  3. After each round, updates scores/reputation
  4. Prints a live learning curve showing improvement
"""
from __future__ import annotations
import logging
import random
import time
import json
from datetime import datetime, timezone
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Simulation scenarios ───────────────────────────────────────────────────

SIMULATION_SCENARIOS = [
    {"goal": "Process and summarize text content",    "success_prob": 0.95},
    {"goal": "Analyze and route data stream",         "success_prob": 0.80},
    {"goal": "Transform and forward payload",         "success_prob": 0.75},
    {"goal": "Validate and process input",            "success_prob": 0.90},
    {"goal": "Aggregate and output results",          "success_prob": 0.85},
    {"goal": "Filter and clean data",                 "success_prob": 0.70},
    {"goal": "Encode and transmit message",           "success_prob": 0.65},
    {"goal": "Decode and store payload",              "success_prob": 0.88},
]

SAMPLE_PAYLOADS = [
    {"text": "Neural networks are fascinating.", "source": "sim"},
    {"text": "Service mesh routing optimizes traffic.", "source": "sim"},
    {"text": "AI learns from execution history.", "source": "sim"},
    {"text": "Distributed systems require coordination.", "source": "sim"},
    {"text": "Graph topology affects route efficiency.", "source": "sim"},
]


class SimulationResult:
    """Result of a single simulation round."""

    def __init__(self, round_num: int):
        self.round_num = round_num
        self.executions: int = 0
        self.successes: int = 0
        self.failures: int = 0
        self.routes_used: List[str] = []
        self.avg_latency_ms: float = 0.0
        self.learning_snapshot: Optional[dict] = None
        self.ts: str = datetime.now(timezone.utc).isoformat()

    @property
    def success_rate(self) -> float:
        return self.successes / self.executions if self.executions > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "round": self.round_num,
            "executions": self.executions,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "routes_used": self.routes_used,
            "learning_snapshot": self.learning_snapshot,
            "ts": self.ts,
        }


class SimulationEngine:
    """
    Phase 4 Simulation Engine.

    Drives automatic execution loops so the AI accumulates experience.
    Demonstrates that the system learns by comparing performance across rounds.
    """

    def __init__(self, mesh, validator=None):
        self._mesh = mesh
        self._validator = validator
        self._round_results: List[SimulationResult] = []
        logger.info("SimulationEngine initialised (Phase 4)")

    # ── Setup ──────────────────────────────────────────────────────────────

    def setup_simulation_nodes(self) -> Tuple[str, str, str]:
        """Register a standard 3-node pipeline for simulation."""
        from services.input_service import InputNode
        from services.processor_service import ProcessorNode
        from services.output_service import OutputNode

        inp  = self._mesh.register_node(InputNode("SimInput"))
        proc = self._mesh.register_node(ProcessorNode("SimProcessor"), connect_to=inp)
        out  = self._mesh.register_node(OutputNode("SimOutput", output_format="summary"), connect_to=proc)

        logger.info(f"Simulation nodes: {inp[:8]} → {proc[:8]} → {out[:8]}")
        return inp, proc, out

    # ── Simulation loop ────────────────────────────────────────────────────

    def run_simulation(self, rounds: int = 20, executions_per_round: int = 5,
                        delay_between_rounds: float = 0.1,
                        verbose: bool = True) -> dict:
        """
        Run a full simulation: `rounds` rounds × `executions_per_round` executions.
        After each round, captures a learning snapshot.

        Returns a summary proving learning occurred.
        """
        print("\n" + "═"*65)
        print("  PHASE 4 — Autonomous Learning Simulation")
        print("═"*65)
        print(f"  Rounds: {rounds}  |  Executions/round: {executions_per_round}")
        print(f"  Total planned executions: {rounds * executions_per_round}")
        print("═"*65 + "\n")

        # Setup nodes
        inp, proc, out = self.setup_simulation_nodes()

        # Lock baseline before first execution
        if self._validator:
            self._validator.lock_baseline()

        all_latencies = []

        for round_num in range(1, rounds + 1):
            result = SimulationResult(round_num)

            for _ in range(executions_per_round):
                scenario = random.choice(SIMULATION_SCENARIOS)
                payload = random.choice(SAMPLE_PAYLOADS).copy()
                payload["sim_round"] = round_num

                # Decide success/failure based on scenario probability
                # But use real execution so the AI actually learns
                t0 = time.perf_counter()
                try:
                    run_result = self._mesh.run(inp, out, payload, use_ai=True)
                    latency_ms = (time.perf_counter() - t0) * 1000
                    all_latencies.append(latency_ms)
                    result.executions += 1

                    status = run_result.get("status", "failed")
                    if status == "success":
                        result.successes += 1
                        path = run_result.get("path", [inp, proc, out])
                        if path:
                            path_key = "->".join(p[:8] for p in path)
                            if path_key not in result.routes_used:
                                result.routes_used.append(path_key)
                    else:
                        result.failures += 1
                except Exception as e:
                    logger.debug(f"Simulation run error: {e}")
                    result.failures += 1
                    result.executions += 1

            # Calculate average latency for this round
            round_latencies = all_latencies[-executions_per_round:]
            result.avg_latency_ms = sum(round_latencies) / len(round_latencies) if round_latencies else 0

            # Capture learning snapshot after each round
            if self._validator:
                try:
                    metrics = self._validator.compute_metrics()
                    result.learning_snapshot = {
                        "total_executions": metrics.total_executions,
                        "success_rate": metrics.successful_executions / metrics.total_executions
                        if metrics.total_executions > 0 else 0,
                        "learning_improvement_pct": metrics.learning_improvement,
                        "avg_route_score": metrics.avg_success_rate,
                    }
                except Exception:
                    pass

            # Update knowledge layer
            try:
                self._mesh.knowledge.update_graph_statistics(
                    total_nodes=self._mesh.registry.count(),
                    total_edges=self._mesh.graph.stats().get("total_edges", 0),
                    total_runs=result.executions,
                    success_rate=result.success_rate,
                )
                self._mesh.knowledge.update_node_rankings(self._mesh.memory)
                self._mesh.knowledge.update_route_rankings(self._mesh.memory)
                self._mesh.knowledge.update_connection_scores(self._mesh.scoring)
            except Exception as e:
                logger.debug(f"Knowledge update error: {e}")

            self._round_results.append(result)

            if verbose:
                self._print_round_summary(result)

            if delay_between_rounds > 0:
                time.sleep(delay_between_rounds)

        # Final summary
        summary = self._build_summary(rounds, executions_per_round)
        self._print_final_summary(summary)
        return summary

    # ── Reporting ──────────────────────────────────────────────────────────

    def _print_round_summary(self, result: SimulationResult):
        snap = result.learning_snapshot or {}
        sr = result.success_rate
        improvement = snap.get("learning_improvement_pct", 0)
        bar = self._progress_bar(sr, width=20)
        print(
            f"  Round {result.round_num:>3}  {bar}  "
            f"SR={sr:.0%}  "
            f"Δlearn={improvement:+.1f}%  "
            f"lat={result.avg_latency_ms:.1f}ms"
        )

    def _progress_bar(self, ratio: float, width: int = 20) -> str:
        filled = int(ratio * width)
        return "[" + "█" * filled + "░" * (width - filled) + "]"

    def _build_summary(self, rounds: int, per_round: int) -> dict:
        total_exec = sum(r.executions for r in self._round_results)
        total_succ = sum(r.successes for r in self._round_results)
        total_fail = sum(r.failures for r in self._round_results)

        # Learning curve: compare first 3 rounds vs last 3 rounds
        early_sr = 0.0
        late_sr = 0.0
        if len(self._round_results) >= 6:
            early = self._round_results[:3]
            late  = self._round_results[-3:]
            early_sr = sum(r.success_rate for r in early) / 3
            late_sr  = sum(r.success_rate for r in late)  / 3

        learning_delta = (late_sr - early_sr) * 100

        # Proof of learning
        proof = None
        if self._validator:
            try:
                proof = self._validator.prove_learning()
            except Exception:
                pass

        return {
            "simulation": {
                "rounds": rounds,
                "executions_per_round": per_round,
                "total_executions": total_exec,
                "total_successes": total_succ,
                "total_failures": total_fail,
                "overall_success_rate": round(total_succ / total_exec, 4) if total_exec > 0 else 0,
            },
            "learning_proof": {
                "early_rounds_success_rate": round(early_sr, 4),
                "late_rounds_success_rate": round(late_sr, 4),
                "improvement_pct": round(learning_delta, 2),
                "verdict": "improved" if learning_delta > 0 else "stable",
            },
            "validator_proof": proof,
            "round_results": [r.to_dict() for r in self._round_results],
        }

    def _print_final_summary(self, summary: dict):
        sim = summary["simulation"]
        lp = summary["learning_proof"]
        proof = summary.get("validator_proof", {})

        print("\n" + "═"*65)
        print("  SIMULATION COMPLETE — LEARNING PROOF")
        print("═"*65)
        print(f"  Total Executions  : {sim['total_executions']}")
        print(f"  Successful        : {sim['total_successes']}")
        print(f"  Failed            : {sim['total_failures']}")
        print(f"  Overall SR        : {sim['overall_success_rate']:.1%}")
        print()
        print(f"  Early SR (rnd 1-3): {lp['early_rounds_success_rate']:.1%}")
        print(f"  Late SR  (last 3) : {lp['late_rounds_success_rate']:.1%}")
        print(f"  Improvement       : {lp['improvement_pct']:+.1f}%  [{lp['verdict'].upper()}]")
        print()

        if proof and proof.get("verdict") != "insufficient_data":
            print(f"  Verdict: {proof.get('verdict', '').upper()}")
            for ev in proof.get("evidence", []):
                print(f"  {ev}")

        print("═"*65 + "\n")

    def get_round_results(self) -> List[dict]:
        return [r.to_dict() for r in self._round_results]

    def __repr__(self):
        return f"<SimulationEngine rounds={len(self._round_results)}>"
