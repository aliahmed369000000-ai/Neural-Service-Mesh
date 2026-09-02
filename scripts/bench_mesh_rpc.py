#!/usr/bin/env python3
"""قياس بسيط: N مهام inference عبر RPC على عمال محليين + فحص رفض التكرار."""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.living_mesh import LivingMeshNode
from ai import mesh_task_protocol as mt


async def run(host: str, base_port: int, workers: int, tasks: int) -> dict:
    client = LivingMeshNode(
        node_id="bench_client",
        host="127.0.0.1",
        port=base_port + 50,
        data_dir=str(ROOT / "artifacts" / "living_mesh" / "nodes" / "bench_client"),
    )
    roster = [(f"worker_{i}", base_port + i) for i in range(1, workers + 1)]
    results = []
    t_all = time.time()
    for i in range(tasks):
        name, port = roster[i % len(roster)]
        task_id = f"bench_{i}_{uuid.uuid4().hex[:6]}"
        t0 = time.time()
        res = await client.request_from_peer(
            host, port, mt.KIND_INFERENCE,
            {"task_id": task_id, "prompt": f"bench {i}", "max_tokens": 32},
            timeout=12.0,
        )
        results.append({
            "ok": bool(res.get("ok") and res.get("result") is not None),
            "acked": bool(res.get("acked")),
            "rtt_ms": round((time.time() - t0) * 1000, 2),
            "worker": name,
            "receipt": bool(((res.get("result") or {}).get("receipt") or {}).get("signature")),
            "task_id": task_id,
            "port": port,
        })
    src = next((r for r in results if r["ok"]), None)
    dup = None
    if src:
        t0 = time.time()
        dres = await client.request_from_peer(
            host, src["port"], mt.KIND_INFERENCE,
            {"task_id": src["task_id"], "prompt": "dup", "max_tokens": 8},
            timeout=6.0,
        )
        result_err = ((dres.get("result") or {}) if isinstance(dres.get("result"), dict) else {}).get("error")
        dup = {
            "task_id": src["task_id"],
            "ok": dres.get("ok"),
            "error": dres.get("error") or result_err,
            "has_result": dres.get("result") is not None,
            "explicit_duplicate": result_err == "duplicate_rejected",
            "rtt_ms": round((time.time() - t0) * 1000, 2),
        }
    oks = [r for r in results if r["ok"]]
    rtts = [r["rtt_ms"] for r in oks]
    return {
        "tasks": tasks,
        "success": len(oks),
        "failed": tasks - len(oks),
        "success_rate_pct": round(100 * len(oks) / tasks, 1) if tasks else 0,
        "rtt_ms": {
            "avg": round(statistics.mean(rtts), 2) if rtts else None,
            "min": min(rtts) if rtts else None,
            "max": max(rtts) if rtts else None,
            "p50": round(statistics.median(rtts), 2) if rtts else None,
        },
        "acked": sum(1 for r in results if r["acked"]),
        "receipts": sum(1 for r in results if r["receipt"]),
        "wall_sec": round(time.time() - t_all, 2),
        "duplicate_probe": dup,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--base-port", type=int, default=17860, help="seed port; workers at +1..N")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--tasks", type=int, default=20)
    args = ap.parse_args()
    report = asyncio.run(run(args.host, args.base_port, args.workers, args.tasks))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
