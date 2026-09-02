#!/usr/bin/env python3
"""
اختبار حي: طبقة صحة + مسارات + مهام قابلة للتحقق على 5 عمليات OS.
"""
from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import socket
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

HOST = "127.0.0.1"
BASE_PORT = 19101
N = 5


def wait_port(port: int, timeout: float = 12.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((HOST, port), timeout=0.3):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def worker(tmp: str, node_id: str, port: int, seed: dict | None, result_path: str):
    async def main():
        import ai.living_mesh as lm
        from ai.node_health_layer import NodeHealthLayer
        from ai import mesh_task_protocol as mt
        from aiohttp import web

        d = Path(tmp) / node_id
        d.mkdir(parents=True, exist_ok=True)
        lm.LIVING_MESH_DIR = d
        lm.NETWORK_STATE = d / "network_state.json"
        lm.CONTENT_DIR = d / "content"
        lm.CONTENT_DIR.mkdir(exist_ok=True)

        node = lm.LivingMeshNode(node_id=node_id, host=HOST, port=port)
        node.join_network()
        health = NodeHealthLayer(node)

        async def handle_ws(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    await node._handle_aiohttp_ws_msg(ws, json.loads(msg.data))
            return ws

        async def handle_health(request):
            return web.json_response(health.health())

        async def handle_routes(request):
            return web.json_response(health.routes_table())

        app = web.Application()
        app.router.add_get("/ws", handle_ws)
        app.router.add_get("/health", handle_health)
        app.router.add_get("/v2/routes", handle_routes)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, HOST, port).start()

        report = {"node_id": node_id, "port": port}

        if seed:
            await node.request_peers(seed["host"], seed["port"], retries=5, retry_delay=0.4)
            await asyncio.sleep(0.5)
            ping = await node.ping_peer(seed["host"], seed["port"])
            report["ping_seed"] = ping

        # مهمة محلية قابلة للتحقق
        vt = await health.submit_verifiable_task(
            mt.KIND_MAP,
            {"lines": [f"health-test {node_id}", "verifiable task mesh"], "op": "wordcount"},
            local=True,
        )
        ver = health.verify_receipt(vt.get("receipt"), vt.get("result"))
        report["verifiable_task"] = {
            "ok": vt.get("ok"),
            "verification": ver,
            "elapsed_ms": vt.get("elapsed_ms"),
        }
        report["health"] = health.health()

        Path(result_path).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        (Path(result_path).with_suffix(".ready")).write_text("1")
        stop = Path(result_path).with_suffix(".stop")
        for _ in range(400):
            if stop.exists():
                break
            await asyncio.sleep(0.1)
        await runner.cleanup()

    try:
        asyncio.run(main())
    except Exception as e:
        Path(result_path).write_text(json.dumps({"node_id": node_id, "error": str(e)}))


def coordinator(tmp: str, ids, ports) -> dict:
    async def main():
        import ai.living_mesh as lm
        from ai.node_health_layer import NodeHealthLayer
        from ai import mesh_task_protocol as mt

        d = Path(tmp) / "coord"
        d.mkdir(parents=True, exist_ok=True)
        lm.LIVING_MESH_DIR = d
        lm.NETWORK_STATE = d / "network_state.json"
        lm.CONTENT_DIR = d / "content"
        lm.CONTENT_DIR.mkdir(exist_ok=True)

        node = lm.LivingMeshNode(node_id="coord", host=HOST, port=0)
        node.join_network()
        health = NodeHealthLayer(node)

        state = node._load_state()
        for nid, port in zip(ids, ports):
            state["nodes"][nid] = {
                "id": nid, "host": HOST, "port": port,
                "status": "online",
                "capabilities": ["storage", "GPU_HIGH", "text"],
            }
        node._save_state(state)

        await node.request_peers(HOST, ports[0], retries=5, retry_delay=0.4)

        pings = []
        for nid, port in zip(ids, ports):
            r = {"ok": False}
            for a in range(1, 8):
                r = await node.ping_peer(HOST, port, timeout=4.0)
                if r.get("ok"):
                    r["attempts"] = a
                    break
                await asyncio.sleep(0.3 * a)
            pings.append({"node": nid, **r})

        await health.probe_routes()
        routes = health.routes_table()
        best = health.best_route()

        # مهام قابلة للتحقق محلياً من المنسّق
        local_vt = await health.submit_verifiable_task(
            mt.KIND_INFERENCE,
            {"prompt": "اختبار طبقة الصحة", "model_hint": "local"},
            local=True,
        )
        ver = health.verify_receipt(local_vt["receipt"], local_vt["result"])

        return {
            "pings": pings,
            "reachable": sum(1 for p in pings if p.get("ok")),
            "routes_count": routes.get("count"),
            "best_route": best,
            "verifiable": {"ok": local_vt.get("ok"), "verification": ver},
            "health": health.health(),
        }

    return asyncio.run(main())


def main():
    print("=" * 60)
    print("🔬 اختبار حي: صحة + مسارات + مهام قابلة للتحقق (5 عمليات)")
    print("=" * 60)
    tmp = tempfile.mkdtemp(prefix="nsm_health5_")
    results = Path(tmp) / "r"
    results.mkdir()
    ids = [f"hnode_{i}" for i in range(N)]
    ports = [BASE_PORT + i for i in range(N)]

    procs = []
    p0 = mp.Process(target=worker, args=(tmp, ids[0], ports[0], None, str(results / f"{ids[0]}.json")))
    p0.start()
    procs.append(p0)
    assert wait_port(ports[0]), "seed not ready"
    print(f"✅ {ids[0]} :{ports[0]}")

    seed = {"host": HOST, "port": ports[0]}
    for i in range(1, N):
        pi = mp.Process(target=worker, args=(tmp, ids[i], ports[i], seed, str(results / f"{ids[i]}.json")))
        pi.start()
        procs.append(pi)
        wait_port(ports[i])
        print(f"✅ {ids[i]} :{ports[i]}")
        time.sleep(0.3)

    for nid in ids:
        ready = results / f"{nid}.ready"
        for _ in range(100):
            if ready.exists():
                break
            time.sleep(0.1)

    coord = coordinator(tmp, ids, ports)
    print("\n—— نتائج المنسّق ——")
    print(f"  Ping reachable: {coord['reachable']}/{N}")
    for p in coord["pings"]:
        mark = "✅" if p.get("ok") else "❌"
        print(f"  {mark} {p.get('node')} rtt={p.get('rtt_ms')} attempts={p.get('attempts')}")
    print(f"  routes_count={coord.get('routes_count')} best={coord.get('best_route')}")
    print(f"  verifiable={coord.get('verifiable')}")

    node_reports = []
    for nid in ids:
        fp = results / f"{nid}.json"
        if fp.exists():
            node_reports.append(json.loads(fp.read_text()))
        (results / f"{nid}.stop").write_text("1")

    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    print("\n—— نتائج العقد ——")
    vt_ok = 0
    for r in node_reports:
        v = (r.get("verifiable_task") or {}).get("verification") or {}
        ok = v.get("ok")
        if ok:
            vt_ok += 1
        print(f"  · {r.get('node_id')}: verifiable={ok} health_rep={(r.get('health') or {}).get('reputation')}")

    success = coord["reachable"] >= N - 1 and vt_ok >= N - 1 and coord["verifiable"]["verification"]["ok"]
    print("\n" + ("🏆 نجح اختبار الصحة والمسارات والمهام القابلة للتحقق" if success else "💥 فشل جزئي"))
    return 0 if success else 1


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
