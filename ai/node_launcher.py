#!/usr/bin/env python3
"""
NSM Distributed Node Launcher — مشغل العقد الموزعة لمشروع NSM
==========================================================
يدعم هذا المشغل تشغيل العقد كخدمات هجينة (HTTP + WebSocket) متوافقة مع Hugging Face.
"""
import argparse
import asyncio
import logging
import sys
import os
import json
from pathlib import Path
from aiohttp import web

# إضافة مسار المشروع لـ sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.living_mesh import LivingMeshNode
from ai.node_health_layer import NodeHealthLayer

# إعداد السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("node_activity.log")
    ]
)
logger = logging.getLogger("NodeLauncher")

async def handle_status(request):
    """حالة العقدة الحية — من network_state وليس من قائمة peers الفارغة."""
    node = request.app['node']
    state = node._load_state()
    nodes = state.get("nodes") or {}
    online = [
        {"id": nid, "host": info.get("host"), "port": info.get("port"), "last_seen": info.get("last_seen")}
        for nid, info in nodes.items()
        if info.get("status") == "online"
    ]
    status = {
        "node_id": node.node_id,
        "status": "Running",
        "host": node.host,
        "port": node.port,
        "data_dir": str(getattr(node, "data_dir", "")),
        "surah_awareness": getattr(node, "surah_awareness", {"status": "unknown"}),
        "peers_count": len(online),
        "online_nodes": online,
        "known_nodes": len(nodes),
        "active_connections": len(getattr(node, "active_connections", set()) or set()),
        "public_address": f"ws://{node.host}:{node.port}",
        "ws_path": "/ws",
    }
    return web.json_response(status, dumps=lambda x: json.dumps(x, indent=2, ensure_ascii=False))


async def handle_health(request):
    """Endpoint صحة الشبكة — عبر NodeHealthLayer إن وُجدت."""
    health = request.app.get('health')
    if health is not None:
        return web.json_response(health.health())
    node = request.app['node']
    snap = node.network_health_snapshot()
    snap["status"] = "healthy" if snap.get("node_id") else "degraded"
    return web.json_response(snap)

async def handle_v2_status(request):
    node = request.app['node']
    snap = node.network_health_snapshot()
    snap["reputation_ledger"] = node.get_reputation()
    state = node._load_state()
    snap["federated_rounds"] = list((state.get("federated_rounds") or {}).keys())[-5:]
    return web.json_response(snap)

async def handle_dashboard(request):
    """لوحة حالة مختصرة HTML."""
    node = request.app['node']
    snap = node.network_health_snapshot()
    rep = node.get_reputation(node.node_id)
    html = f"""<!DOCTYPE html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8"/>
<title>NSM Node 2.0 — {snap.get('node_id')}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0b1220;color:#e8eefc;margin:0;padding:24px}}
.card{{background:#151c2e;border-radius:12px;padding:16px 20px;margin-bottom:12px;border:1px solid #24304d}}
h1{{margin:0 0 8px;font-size:1.4rem}} .muted{{color:#8b9bb8;font-size:.9rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}}
.metric{{background:#0f1626;border-radius:8px;padding:12px;text-align:center}}
.metric b{{display:block;font-size:1.3rem;color:#6ea8fe}}
</style></head><body>
<h1>NSM Node 2.0 Vertical Slice</h1>
<p class="muted">{snap.get('node_id')} · {snap.get('host')}:{snap.get('port')} · {snap.get('ts')}</p>
<div class="grid">
<div class="metric"><b>{snap.get('online_peers')}</b>أقران متصلون</div>
<div class="metric"><b>{snap.get('known_nodes')}</b>عقد معروفة</div>
<div class="metric"><b>{snap.get('reputation_self')}</b>السمعة</div>
<div class="metric"><b>{snap.get('receipts')}</b>إيصالات</div>
<div class="metric"><b>{snap.get('content_objects')}</b>محتوى</div>
<div class="metric"><b>{snap.get('identity_pub_fingerprint')}</b>بصمة الهوية</div>
</div>
<div class="card"><div class="muted">Surah: {snap.get('surah_awareness')}</div></div>
<div class="card"><div class="muted">API: /health · /v2/status · /ws · /status</div></div>
</body></html>"""
    return web.Response(text=html, content_type="text/html")


async def handle_routes(request):
    health = request.app.get("health")
    if health is None:
        return web.json_response({"error": "health_layer_missing"}, status=500)
    return web.json_response(health.routes_table())

async def handle_tasks(request):
    health = request.app.get("health")
    if health is None:
        return web.json_response({"error": "health_layer_missing"}, status=500)
    return web.json_response({"tasks": health.recent_tasks()})

async def handle_submit_task(request):
    """POST JSON: {kind, payload, local?} — مهمة قابلة للتحقق."""
    health = request.app.get("health")
    if health is None:
        return web.json_response({"error": "health_layer_missing"}, status=500)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)
    kind = body.get("kind")
    payload = body.get("payload") or {}
    local = bool(body.get("local", True))
    if not kind:
        return web.json_response({"error": "kind_required"}, status=400)
    result = await health.submit_verifiable_task(kind, payload, local=local)
    if result.get("receipt") and result.get("result"):
        result["verification"] = health.verify_receipt(result["receipt"], result["result"])
    return web.json_response(result)


async def handle_submit_job(request):
    """POST JSON: {kind, payload, n_workers?, strategy?, require_capabilities?, timeout_per_task?, retry_failed?}"""
    health = request.app.get("health")
    if health is None:
        return web.json_response({"error": "health_layer_missing"}, status=500)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)
    kind = body.get("kind")
    if not kind:
        return web.json_response({"error": "kind_required"}, status=400)
    payload = body.get("payload") or {}
    report = await health.submit_job(
        kind,
        payload,
        n_workers=int(body.get("n_workers") or 3),
        strategy=body.get("strategy") or "majority",
        require_capabilities=body.get("require_capabilities"),
        timeout_per_task=float(body.get("timeout_per_task") or 12.0),
        retry_failed=int(body.get("retry_failed") or 1),
    )
    return web.json_response(report)


async def handle_list_jobs(request):
    health = request.app.get("health")
    if health is None:
        return web.json_response({"error": "health_layer_missing"}, status=500)
    orch = health.orchestrator()
    return web.json_response({"jobs": orch.list_jobs(limit=int(request.query.get("limit") or 20))})


async def handle_join_info(request):
    """معلومات عامة لانضمام عقدة خارجية — بلا أسرار."""
    node = request.app["node"]
    health = request.app.get("health")
    snap = health.health() if health else node.network_health_snapshot()
    return web.json_response({
        "ok": True,
        "protocol": "nsm-join-v1",
        "seed_node_id": node.node_id,
        "ws_url": f"ws://{node.host}:{node.port}/ws",
        "http_base": f"http://{node.host}:{node.port}",
        "public_key": node._pub_pem(),
        "identity_fp": snap.get("identity_fp") or snap.get("identity_pub_fingerprint"),
        "endpoints": {
            "join_info": "GET /v2/join-info",
            "join": "POST /v2/join",
            "accept_peer_key": "POST /v2/accept-peer-key",
            "dispatch_task": "POST /v2/dispatch-task",
            "health": "GET /health",
            "task": "POST /v2/task",
            "tasks": "GET /v2/tasks",
            "ws": "GET /ws",
        },
        "first_task_example": {
            "kind": "map_reduce_map",
            "payload": {"lines": ["hello federation", "nsm join path"], "op": "wordcount"},
            "local": True,
        },
    })


async def handle_join(request):
    """
    تسجيل عقدة خارجية لدى البذرة.
    JSON: {node_id, host, port, public_key, capabilities?}
    """
    node = request.app["node"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid_json"}, status=400)

    peer_id = body.get("node_id") or body.get("id")
    pub = body.get("public_key")
    host = body.get("host")
    port = body.get("port")
    if not peer_id or not pub:
        return web.json_response(
            {"ok": False, "error": "node_id_and_public_key_required"}, status=400
        )
    if peer_id == node.node_id:
        return web.json_response({"ok": False, "error": "cannot_join_as_self"}, status=400)

    # احفظ المفتاح العام
    try:
        key_path = node.keys_dir / f"{peer_id}.pub"
        key_path.write_text(pub if isinstance(pub, str) else pub.decode())
    except Exception as e:
        return web.json_response({"ok": False, "error": f"key_store_failed:{e}"}, status=500)

    # سجّل في حالة الشبكة
    state = node._load_state()
    info = {
        "id": peer_id,
        "host": host or "unknown",
        "port": port,
        "status": "online",
        "capabilities": body.get("capabilities") or ["text", "storage"],
        "public_key": pub if isinstance(pub, str) else None,
        "joined_via": "api_v2_join",
        "last_seen": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
    }
    state.setdefault("nodes", {})[peer_id] = info
    node._save_state(state)

    return web.json_response({
        "ok": True,
        "protocol": "nsm-join-v1",
        "registered_as": peer_id,
        "seed_node_id": node.node_id,
        "seed_public_key": node._pub_pem(),
        "ws_url": f"ws://{node.host}:{node.port}/ws",
        "next_steps": [
            "Store seed_public_key via POST /v2/accept-peer-key on your node",
            "POST /v2/first-task on YOUR node HTTP for worker-signed receipt",
            "Optional: seed verifies worker receipt using stored public_key",
            "Open WS /ws for P2P discovery and signed messaging",
        ],
        "peer_record": {k: v for k, v in info.items() if k != "public_key"},
    })



async def handle_accept_peer_key(request):
    """
    تبادل مفاتيح ثنائي: تحفظ هذه العقدة مفتاح نظير معروف.
    JSON: {node_id, public_key}
    """
    node = request.app["node"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid_json"}, status=400)
    peer_id = body.get("node_id") or body.get("id")
    pub = body.get("public_key")
    if not peer_id or not pub:
        return web.json_response({"ok": False, "error": "node_id_and_public_key_required"}, status=400)
    try:
        (node.keys_dir / f"{peer_id}.pub").write_text(pub if isinstance(pub, str) else pub.decode())
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)
    return web.json_response({
        "ok": True,
        "stored_peer": peer_id,
        "self_node_id": node.node_id,
        "self_public_key": node._pub_pem(),
    })

async def handle_first_verified_task(request):
    """
    مسار موثّق: ينفّذ أول مهمة محلية قابلة للتحقق على هذه العقدة.
    لا يحتاج جسماً — أو JSON اختياري {lines?, kind?}.
    """
    health = request.app.get("health")
    if health is None:
        return web.json_response({"ok": False, "error": "health_layer_missing"}, status=500)
    try:
        body = await request.json()
    except Exception:
        body = {}
    kind = body.get("kind") or "map_reduce_map"
    payload = body.get("payload") or {
        "lines": body.get("lines") or [
            "nsm external join",
            "first verified task",
            "federation wordcount",
        ],
        "op": "wordcount",
    }
    result = await health.submit_verifiable_task(kind, payload, local=True)
    if result.get("receipt") and result.get("result"):
        result["verification"] = health.verify_receipt(result["receipt"], result["result"])
    result["documented_path"] = "POST /v2/first-task"
    result["ok"] = bool(result.get("ok")) and bool(
        (result.get("verification") or {}).get("ok", result.get("ok"))
    )
    return web.json_response(result)


async def handle_dispatch_task(request):
    """
    البذرة تطلب · العامل ينفّذ · البذرة تتحقق.
    JSON:
      {
        "target_url": "http://worker:19901",   # أو
        "target_node_id": "external_live_1",  # يُستنتج من network_state
        "kind": "map_reduce_map",
        "payload": {...},
        "path": "/v2/first-task" | "/v2/task"
      }
    """
    import aiohttp
    node = request.app["node"]
    health = request.app.get("health")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid_json"}, status=400)

    target_url = (body.get("target_url") or "").rstrip("/")
    target_node_id = body.get("target_node_id")
    if not target_url and target_node_id:
        state = node._load_state()
        peer = (state.get("nodes") or {}).get(target_node_id) or {}
        host = peer.get("host")
        port = peer.get("port")
        if host and port:
            target_url = f"http://{host}:{port}"
    if not target_url:
        return web.json_response(
            {"ok": False, "error": "target_url_or_resolvable_target_node_id_required"},
            status=400,
        )

    path = body.get("path") or "/v2/first-task"
    if not path.startswith("/"):
        path = "/" + path
    kind = body.get("kind") or "map_reduce_map"
    payload = body.get("payload")
    if path.endswith("first-task"):
        req_body = body.get("task_body") or {
            "lines": (payload or {}).get("lines")
            or ["dispatch from seed", "worker executes", "seed verifies"],
            "kind": kind,
            "payload": payload or {},
        }
    else:
        req_body = {
            "kind": kind,
            "payload": payload or {"lines": ["dispatch", "task"], "op": "wordcount"},
            "local": True,
        }

    url = target_url + path
    try:
        timeout = aiohttp.ClientTimeout(total=float(body.get("timeout_s") or 30))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=req_body) as resp:
                raw = await resp.text()
                try:
                    worker_result = json.loads(raw) if raw else {}
                except Exception:
                    return web.json_response({
                        "ok": False,
                        "error": "worker_non_json",
                        "status": resp.status,
                        "raw": raw[:500],
                    }, status=502)
                if resp.status >= 400:
                    return web.json_response({
                        "ok": False,
                        "error": "worker_http_error",
                        "status": resp.status,
                        "worker_result": worker_result,
                    }, status=502)
    except Exception as e:
        return web.json_response({"ok": False, "error": f"dispatch_failed:{e}"}, status=502)

    receipt = worker_result.get("receipt")
    result = worker_result.get("result")
    verification = None
    if health is not None and receipt and result is not None:
        verification = health.verify_receipt(receipt, result)
    elif receipt and result is not None:
        # تحقق مبسّط عبر مفتاح العامل المخزّن
        signer = receipt.get("node_id")
        key_path = node.keys_dir / f"{signer}.pub"
        verification = {"ok": False, "error": "no_health_layer"}
        if key_path.exists() and hasattr(node, "verify_signature"):
            digest = receipt.get("result_digest")
            # الاعتماد على health إن وُجدت فقط
            verification = {"ok": False, "error": "install_health_for_full_verify", "signer": signer}

    ok = bool(worker_result.get("ok")) and bool((verification or {}).get("ok"))
    return web.json_response({
        "ok": ok,
        "protocol": "nsm-dispatch-v1",
        "dispatcher_node_id": node.node_id,
        "target_url": target_url,
        "target_node_id": target_node_id or (receipt or {}).get("node_id"),
        "path": path,
        "worker_result": {
            "ok": worker_result.get("ok"),
            "task_id": worker_result.get("task_id"),
            "receipt": receipt,
            "result": result,
            "verification_on_worker": worker_result.get("verification"),
        },
        "verification_on_seed": verification,
        "loop": "seed_requests · worker_executes · seed_verifies",
    })

async def handle_ws(request):
    """معالج WebSocket موحّد عبر مسار الرسائل الموقّعة في LivingMeshNode.
    أُزيل مسار التجميع المركزي القديم (gradient_buffer / All-Reduce المحلي).
    كل الرسائل — بما فيها gradient_push — تمر عبر _handle_aiohttp_ws_msg.
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    node = request.app['node']

    # تتبع الاتصالات النشطة (للمراقبة فقط — ليس للتجميع المركزي)
    if not hasattr(node, 'active_connections') or node.active_connections is None:
        node.active_connections = set()
    node.active_connections.add(ws)

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except Exception as e:
                    logger.warning(f"⚠️ Invalid WS JSON: {e}")
                    continue
                # مسار موحّد موقّع — يشمل peer_discovery و gradient_push و Gossip
                await node._handle_aiohttp_ws_msg(ws, data)
            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f"WS connection closed with exception {ws.exception()}")
    finally:
        node.active_connections.discard(ws)

    return ws

def _parse_seed_url(url: str) -> dict:
    """يفسّر SEED_NODE_URL بصيغ: host:port | http(s)://host:port | ws(s)://host:port/path"""
    raw = (url or "").strip()
    if not raw:
        return {}
    cleaned = raw.replace("https://", "").replace("http://", "").replace("wss://", "").replace("ws://", "")
    cleaned = cleaned.split("/")[0]  # drop path
    host, port = cleaned, None
    if ":" in cleaned:
        host, port_s = cleaned.rsplit(":", 1)
        try:
            port = int(port_s)
        except ValueError:
            host, port = cleaned, None
    if port is None:
        port = 443 if ".hf.space" in host else 7860
    return {"id": "seed_node", "host": host, "port": port}


async def _wait_port_ready(host: str, port: int, timeout: float = 15.0) -> bool:
    import socket
    deadline = asyncio.get_event_loop().time() + timeout
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    while asyncio.get_event_loop().time() < deadline:
        try:
            with socket.create_connection((probe_host, port), timeout=0.4):
                return True
        except OSError:
            await asyncio.sleep(0.15)
    return False


async def main():
    parser = argparse.ArgumentParser(description="NSM Distributed Node Launcher")
    parser.add_argument("--id", type=str, help="Node ID (default: mesh_seed or NODE_ID)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--seed-host", type=str, help="Seed node host IP")
    parser.add_argument("--seed-port", type=int, help="Seed node port")
    parser.add_argument("--data-dir", type=str, help="Isolated data directory for this node")
    args = parser.parse_args()

    node_id = args.id or os.getenv("NODE_ID", "mesh_seed")
    bind_host = args.host
    port = int(os.getenv("PORT", args.port))

    # مجلد بيانات معزول لكل عقدة (هوية + network_state مستقلان)
    root = Path(__file__).resolve().parent.parent
    default_data = root / "artifacts" / "living_mesh" / "nodes" / node_id
    data_dir = args.data_dir or os.getenv("NSM_NODE_DATA_DIR") or str(default_data)

    hf_space_id = os.getenv("SPACE_ID")
    if hf_space_id:
        space_user, space_name = hf_space_id.split("/")
        node_host = f"{space_user}-{space_name}.hf.space".replace("_", "-").lower()
    else:
        node_host = bind_host if bind_host != "0.0.0.0" else "127.0.0.1"

    logger.info(f"🔮 Initializing node id={node_id} data_dir={data_dir}")
    node = LivingMeshNode(node_id=node_id, host=node_host, port=port, data_dir=data_dir)

    seed_nodes = []
    env_seed_url = os.getenv("SEED_NODE_URL")
    if env_seed_url:
        parsed = _parse_seed_url(env_seed_url)
        if parsed:
            seed_nodes.append(parsed)
            logger.info(f"🌐 Seed via SEED_NODE_URL: {parsed['host']}:{parsed['port']}")
    elif args.seed_host:
        seed_nodes.append({
            "id": "seed_node",
            "host": args.seed_host,
            "port": int(args.seed_port or 7860),
        })

    app = web.Application()
    app["node"] = node
    app["health"] = NodeHealthLayer(node)
    app.add_routes([
        web.get("/status", handle_status),
        web.get("/health", handle_health),
        web.get("/v2/status", handle_v2_status),
        web.get("/v2/join-info", handle_join_info),
        web.post("/v2/join", handle_join),
        web.post("/v2/accept-peer-key", handle_accept_peer_key),
        web.get("/v2/routes", handle_routes),
        web.get("/v2/tasks", handle_tasks),
        web.post("/v2/task", handle_submit_task),
        web.post("/v2/job", handle_submit_job),
        web.get("/v2/jobs", handle_list_jobs),
        web.post("/v2/first-task", handle_first_verified_task),
        web.post("/v2/dispatch-task", handle_dispatch_task),
        web.get("/dashboard", handle_dashboard),
        web.get("/ws", handle_ws),
        web.get("/", handle_status),
    ])

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, bind_host, port)
    await site.start()

    ready = await _wait_port_ready(bind_host, port, timeout=15.0)
    if not ready:
        logger.error(f"❌ Port {port} not ready — node will still attempt to run")
    else:
        logger.info(f"✅ Readiness OK — port {port} accepting connections")

    node.join_network(seed_nodes=seed_nodes)
    # إن وُجدت بذرة: اطلب الأقران فعلياً عبر الشبكة
    for s in seed_nodes:
        try:
            ok = await node.request_peers(s["host"], int(s["port"]))
            logger.info(f"🔎 request_peers → {s['host']}:{s['port']} ok={ok}")
        except Exception as e:
            logger.warning(f"⚠️ request_peers failed: {e}")

    logger.info(f"🚀 NSM Node '{node.node_id}' LIVE on {bind_host}:{port} (advertise {node_host}:{port})")

    stop_event = asyncio.Event()

    async def heartbeat_loop():
        tick = 0
        while not stop_event.is_set():
            try:
                node.send_heartbeat()
                # علّم العقد المتوقفة stale/offline عبر last_seen
                dead = node.check_network_health(timeout_seconds=int(os.getenv("NSM_PEER_STALE_SEC", "45")))
                if dead:
                    logger.info(f"📴 Marked stale/offline: {dead}")
                tick += 1
                if tick % 4 == 0:
                    try:
                        health = await node.measure_peers_health(timeout=4.0)
                        reachable = sum(1 for h in health if h.get("ok"))
                        logger.info(f"📶 Peer health: {reachable}/{len(health)} reachable")
                    except Exception as e:
                        logger.warning(f"⚠️ Peer health check failed: {e}")
            except Exception as e:
                logger.error(f"❌ heartbeat error: {e}")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                pass

    hb_task = asyncio.create_task(heartbeat_loop())

    try:
        await asyncio.Future()
    finally:
        stop_event.set()
        hb_task.cancel()
        try:
            node.mark_offline()
        except Exception:
            pass
        await runner.cleanup()
        logger.info("👋 Node shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Node shutting down gracefully (KeyboardInterrupt)...")
