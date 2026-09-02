#!/usr/bin/env python3
"""
بذرة + عقدة عامل على منفذين → انضمام → مهمة موثّقة على العامل (إيصال من العامل).
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SEED_PORT = int(__import__("os").environ.get("NSM_SEED_PORT", "19876"))
WORKER_PORT = int(__import__("os").environ.get("NSM_WORKER_PORT", "19901"))
WORKER_ID = "external_live_1"


def http_json(method: str, url: str, body: dict = None, timeout: float = 20.0) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def wait_url(url: str, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.15)
    return False


async def make_app(node_id: str, host: str, port: int, data_dir: Path):
    import ai.living_mesh as lm
    from ai.living_mesh import LivingMeshNode
    from ai.node_health_layer import NodeHealthLayer
    from aiohttp import web
    from ai import node_launcher as nl

    lm.LIVING_MESH_DIR = data_dir
    lm.NETWORK_STATE = data_dir / "network_state.json"
    lm.CONTENT_DIR = data_dir / "content"
    data_dir.mkdir(parents=True, exist_ok=True)
    lm.CONTENT_DIR.mkdir(exist_ok=True)

    node = LivingMeshNode(node_id=node_id, host=host, port=port)
    health = NodeHealthLayer(node)
    app = web.Application()
    app["node"] = node
    app["health"] = health
    app.router.add_get("/health", nl.handle_health)
    app.router.add_get("/v2/join-info", nl.handle_join_info)
    app.router.add_post("/v2/join", nl.handle_join)
    app.router.add_post("/v2/accept-peer-key", nl.handle_accept_peer_key)
    app.router.add_post("/v2/first-task", nl.handle_first_verified_task)
    app.router.add_post("/v2/task", nl.handle_submit_task)
    app.router.add_get("/v2/tasks", nl.handle_tasks)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner, node, health


async def main():
    print("=" * 60)
    print("🚪 انضمام حي: مهمة موثّقة على عقدة العامل")
    print("=" * 60)
    tmp = Path(tempfile.mkdtemp(prefix="nsm_dual_join_"))
    seed_runner, seed, seed_health = await make_app("seed_live", "127.0.0.1", SEED_PORT, tmp / "seed")
    worker_runner, worker, worker_health = await make_app(WORKER_ID, "127.0.0.1", WORKER_PORT, tmp / "worker")

    seed_base = f"http://127.0.0.1:{SEED_PORT}"
    worker_base = f"http://127.0.0.1:{WORKER_PORT}"

    loop = asyncio.get_event_loop()
    ok_s = await loop.run_in_executor(None, lambda: wait_url(f"{seed_base}/health"))
    ok_w = await loop.run_in_executor(None, lambda: wait_url(f"{worker_base}/health"))
    if not (ok_s and ok_w):
        print("❌ فشل إقلاع البذرة/العامل", ok_s, ok_w)
        await seed_runner.cleanup()
        await worker_runner.cleanup()
        return 1
    print(f"✅ seed  {seed_base}")
    print(f"✅ worker {worker_base}")

    def path():
        steps = []
        # 1 join-info
        info = http_json("GET", f"{seed_base}/v2/join-info")
        steps.append({"step": "join-info", "ok": bool(info.get("ok"))})
        print("[1] join-info", info.get("seed_node_id"))

        # 2 join worker → seed
        join_res = http_json("POST", f"{seed_base}/v2/join", {
            "node_id": worker.node_id,
            "host": "127.0.0.1",
            "port": WORKER_PORT,
            "public_key": worker._pub_pem(),
            "capabilities": ["text", "storage", "map_reduce"],
        })
        steps.append({"step": "join_on_seed", "ok": bool(join_res.get("ok")), "registered_as": join_res.get("registered_as")})
        print("[2] join → seed", join_res.get("ok"))

        # 3 worker stores seed key (bidirectional)
        seed_pub = join_res.get("seed_public_key") or info.get("public_key")
        acc = http_json("POST", f"{worker_base}/v2/accept-peer-key", {
            "node_id": seed.node_id,
            "public_key": seed_pub,
        })
        steps.append({"step": "worker_stores_seed_key", "ok": bool(acc.get("ok"))})
        print("[3] worker accept-peer-key", acc.get("ok"))

        # 4 seed already has worker key from join; confirm accept-peer-key reverse optional
        acc2 = http_json("POST", f"{seed_base}/v2/accept-peer-key", {
            "node_id": worker.node_id,
            "public_key": worker._pub_pem(),
        })
        steps.append({"step": "seed_stores_worker_key", "ok": bool(acc2.get("ok"))})
        print("[4] seed accept-peer-key", acc2.get("ok"))

        # 5 first verified task ON WORKER
        task = http_json("POST", f"{worker_base}/v2/first-task", {
            "lines": [
                f"task executed on {worker.node_id}",
                "external worker verified receipt",
                "nsm bidirectional join",
            ],
        })
        ver = task.get("verification") or {}
        receipt = task.get("receipt") or {}
        worker_signed = receipt.get("node_id") == worker.node_id and ver.get("ok") is True
        steps.append({
            "step": "first-task-on-worker",
            "ok": bool(task.get("ok")) and worker_signed,
            "task_id": task.get("task_id"),
            "receipt_node_id": receipt.get("node_id"),
            "verification": ver,
            "receipt_digest": receipt.get("result_digest"),
        })
        print("[5] first-task on WORKER", task.get("task_id"), "signer=", receipt.get("node_id"))

        # 6 seed verifies worker receipt offline (same process keys)
        from ai.node_health_layer import NodeHealthLayer
        # use seed_health with worker pub already stored
        seed_ver = seed_health.verify_receipt(receipt, task.get("result"))
        steps.append({
            "step": "seed_verifies_worker_receipt",
            "ok": bool(seed_ver.get("ok")),
            "verification": seed_ver,
        })
        print("[6] seed verifies worker receipt", seed_ver)

        return {
            "ok": all(s.get("ok") for s in steps),
            "seed_base": seed_base,
            "worker_base": worker_base,
            "external_node_id": worker.node_id,
            "task_id": task.get("task_id"),
            "receipt_node_id": receipt.get("node_id"),
            "receipt_digest": receipt.get("result_digest"),
            "verification": ver,
            "seed_verification_of_worker": seed_ver,
            "steps": steps,
        }

    report = await loop.run_in_executor(None, path)
    out = REPO / "artifacts" / "join_live_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n📊 الملخص")
    for s in report["steps"]:
        print(f"  {'✅' if s.get('ok') else '❌'} {s['step']}")
    print(f"📁 {out}")
    print("🏆 النجاح" if report["ok"] else "💥 فشل")
    await seed_runner.cleanup()
    await worker_runner.cleanup()
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
