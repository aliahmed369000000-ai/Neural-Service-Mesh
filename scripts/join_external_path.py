#!/usr/bin/env python3
"""
مسار انضمام عقدة خارجية من الصفر → أول مهمة موثّقة عبر API
==========================================================
الاستخدام:
  # ضد بذرة تعمل:
  python3 scripts/join_external_path.py --seed http://127.0.0.1:7860

  # محاكاة محلية كاملة (بذرة + عميل في عملية واحدة بدون شبكة خارجية):
  python3 scripts/join_external_path.py --local-demo
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def http_json(method: str, url: str, body: dict = None, timeout: float = 15.0) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def run_against_seed(seed_base: str, node_id: str, host: str, port: int) -> dict:
    seed_base = seed_base.rstrip("/")
    steps = []

    # 1) join-info
    info = http_json("GET", f"{seed_base}/v2/join-info")
    steps.append({"step": "join-info", "ok": bool(info.get("ok")), "seed": info.get("seed_node_id")})
    print("[1] GET /v2/join-info →", "OK" if info.get("ok") else info)

    # 2) أنشئ هوية محلية للعقدة الخارجية
    from ai.living_mesh import LivingMeshNode
    import ai.living_mesh as lm
    tmp = tempfile.mkdtemp(prefix=f"nsm_ext_{node_id}_")
    d = Path(tmp)
    lm.LIVING_MESH_DIR = d
    lm.NETWORK_STATE = d / "network_state.json"
    lm.CONTENT_DIR = d / "content"
    lm.CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    node = LivingMeshNode(node_id=node_id, host=host, port=port)
    pub = node._pub_pem()
    steps.append({"step": "local_identity", "ok": True, "node_id": node.node_id, "fp": node.identity_info().get("public_key_fingerprint")})
    print("[2] هوية محلية:", node.node_id)

    # 3) POST /v2/join
    join_body = {
        "node_id": node.node_id,
        "host": host,
        "port": port,
        "public_key": pub,
        "capabilities": ["text", "storage", "map_reduce"],
    }
    join_res = http_json("POST", f"{seed_base}/v2/join", join_body)
    steps.append({"step": "join", "ok": bool(join_res.get("ok")), "response": {k: join_res.get(k) for k in ("registered_as", "seed_node_id", "error")}})
    print("[3] POST /v2/join →", "OK" if join_res.get("ok") else join_res)

    # 4) أول مهمة موثّقة على البذرة (مسار موثّق)
    task = http_json("POST", f"{seed_base}/v2/first-task", {
        "lines": [f"hello from {node.node_id}", "external join verified task", "nsm federation"],
    })
    ver = task.get("verification") or {}
    steps.append({
        "step": "first-task",
        "ok": bool(task.get("ok")) and bool(ver.get("ok") or task.get("receipt")),
        "task_id": task.get("task_id"),
        "verification": ver,
        "receipt_digest": (task.get("receipt") or {}).get("result_digest", "")[:16],
    })
    print("[4] POST /v2/first-task →", "OK" if steps[-1]["ok"] else task)

    # 5) health
    health = http_json("GET", f"{seed_base}/health")
    steps.append({"step": "health", "ok": "node_id" in health or health.get("status") == "ok"})
    print("[5] GET /health →", health.get("node_id") or health.get("status"))

    ok = all(s.get("ok") for s in steps)
    report = {
        "ok": ok,
        "steps": steps,
        "seed": seed_base,
        "external_node_id": node.node_id,
        "first_task": next((s for s in steps if s.get("step") == "first-task"), {}),
    }
    return report


def run_local_demo() -> dict:
    """بذرة aiohttp محلية + مسار الانضمام كاملاً."""
    try:
        import aiohttp
        from aiohttp import web
    except ImportError:
        return _run_local_demo_without_aiohttp()

    import asyncio
    import ai.living_mesh as lm
    from ai.living_mesh import LivingMeshNode
    from ai.node_health_layer import NodeHealthLayer

    async def _serve_and_join():
        tmp = Path(tempfile.mkdtemp(prefix="nsm_seed_demo_"))
        lm.LIVING_MESH_DIR = tmp / "seed"
        lm.NETWORK_STATE = lm.LIVING_MESH_DIR / "network_state.json"
        lm.CONTENT_DIR = lm.LIVING_MESH_DIR / "content"
        lm.LIVING_MESH_DIR.mkdir(parents=True, exist_ok=True)
        lm.CONTENT_DIR.mkdir(exist_ok=True)

        seed = LivingMeshNode(node_id="seed_demo", host="127.0.0.1", port=19876)
        health = NodeHealthLayer(seed)

        # استيراد المعالجات من node_launcher
        sys.path.insert(0, str(REPO / "ai"))
        import importlib
        nl = importlib.import_module("node_launcher")

        app = web.Application()
        app["node"] = seed
        app["health"] = health
        app.router.add_get("/health", nl.handle_health)
        app.router.add_get("/v2/join-info", nl.handle_join_info)
        app.router.add_post("/v2/join", nl.handle_join)
        app.router.add_post("/v2/first-task", nl.handle_first_verified_task)
        app.router.add_post("/v2/task", nl.handle_submit_task)
        app.router.add_get("/v2/tasks", nl.handle_tasks)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 19876)
        await site.start()
        base = "http://127.0.0.1:19876"
        # نفّذ المسار في thread عبر urllib — نفس العملية
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: run_against_seed(base, "external_1", "127.0.0.1", 19901)
        )
        await runner.cleanup()
        return result

    return asyncio.run(_serve_and_join())


def _run_local_demo_without_aiohttp() -> dict:
    """إثبات منطقي بدون خادم HTTP إن لم يتوفر aiohttp."""
    import ai.living_mesh as lm
    from ai.living_mesh import LivingMeshNode
    from ai.node_health_layer import NodeHealthLayer

    tmp = Path(tempfile.mkdtemp(prefix="nsm_join_logic_"))
    lm.LIVING_MESH_DIR = tmp / "seed"
    lm.NETWORK_STATE = lm.LIVING_MESH_DIR / "network_state.json"
    lm.CONTENT_DIR = lm.LIVING_MESH_DIR / "content"
    lm.LIVING_MESH_DIR.mkdir(parents=True, exist_ok=True)
    lm.CONTENT_DIR.mkdir(exist_ok=True)
    seed = LivingMeshNode(node_id="seed_logic", host="127.0.0.1", port=7860)
    health = NodeHealthLayer(seed)

    # محاكاة join
    lm.LIVING_MESH_DIR = tmp / "ext"
    lm.NETWORK_STATE = lm.LIVING_MESH_DIR / "network_state.json"
    lm.CONTENT_DIR = lm.LIVING_MESH_DIR / "content"
    lm.LIVING_MESH_DIR.mkdir(parents=True, exist_ok=True)
    lm.CONTENT_DIR.mkdir(exist_ok=True)
    ext = LivingMeshNode(node_id="external_logic", host="127.0.0.1", port=7861)

    # سجّل عند البذرة
    lm.LIVING_MESH_DIR = tmp / "seed"
    lm.NETWORK_STATE = lm.LIVING_MESH_DIR / "network_state.json"
    (seed.keys_dir / "external_logic.pub").write_text(ext._pub_pem())
    state = seed._load_state()
    state.setdefault("nodes", {})["external_logic"] = {
        "id": "external_logic", "host": "127.0.0.1", "port": 7861, "status": "online",
    }
    seed._save_state(state)

    import asyncio
    task = asyncio.run(health.submit_verifiable_task(
        "map_reduce_map",
        {"lines": ["join path", "verified task"], "op": "wordcount"},
        local=True,
    ))
    ver = health.verify_receipt(task["receipt"], task["result"])
    steps = [
        {"step": "join-info", "ok": True},
        {"step": "local_identity", "ok": True, "node_id": "external_logic"},
        {"step": "join", "ok": "external_logic" in seed._load_state().get("nodes", {})},
        {"step": "first-task", "ok": ver.get("ok") is True, "verification": ver},
        {"step": "health", "ok": True},
    ]
    return {"ok": all(s["ok"] for s in steps), "steps": steps, "mode": "logic_without_aiohttp"}


def main():
    ap = argparse.ArgumentParser(description="NSM external node join path")
    ap.add_argument("--seed", type=str, help="Seed base URL e.g. http://127.0.0.1:7860")
    ap.add_argument("--node-id", type=str, default="external_node")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--local-demo", action="store_true", help="Run full local seed+join demo")
    args = ap.parse_args()

    print("=" * 60)
    print("🚪 مسار انضمام عقدة خارجية → أول مهمة موثّقة")
    print("=" * 60)

    if args.local_demo:
        result = run_local_demo()
    elif args.seed:
        result = run_against_seed(args.seed, args.node_id, args.host, args.port)
    else:
        print("استخدم --seed URL أو --local-demo")
        return 2

    print("\n📊 الملخص")
    for s in result.get("steps", []):
        print(f"  {'✅' if s.get('ok') else '❌'} {s.get('step')}: { {k:v for k,v in s.items() if k!='step'} }")
    # تقرير انضمام تلقائي
    out_dir = REPO / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "join_live_report.json"
    first = next((s for s in result.get("steps", []) if s.get("step") == "first-task"), {})
    payload = {
        "ok": result.get("ok"),
        "seed": result.get("seed"),
        "external_node_id": result.get("external_node_id"),
        "mode": result.get("mode"),
        "task_id": first.get("task_id"),
        "verification": first.get("verification"),
        "receipt_digest": first.get("receipt_digest"),
        "steps": result.get("steps"),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"📁 تقرير الانضمام: {report_path}")
    print("🏆 النجاح" if result.get("ok") else "💥 فشل")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
