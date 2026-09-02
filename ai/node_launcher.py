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
    """صفحة حالة العقدة لعرض وعي Surah والأقران."""
    node = request.app['node']
    status = {
        "node_id": node.node_id,
        "status": "Running",
        "surah_awareness": getattr(node, 'surah_awareness', {"status": "unknown"}),
        "peers_count": len(node.peers),
        "active_connections": len(node.active_connections),
        "public_address": f"ws://{node.host}:{node.port}"
    }
    return web.json_response(status, dumps=lambda x: json.dumps(x, indent=2))


async def handle_health(request):
    """Endpoint صحة الشبكة — Node 2.0."""
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

async def main():
    parser = argparse.ArgumentParser(description="NSM Distributed Node Launcher")
    parser.add_argument("--id", type=str, help="Node ID")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--seed-host", type=str, help="Seed node host IP")
    parser.add_argument("--seed-port", type=int, help="Seed node port")
    
    args = parser.parse_args()
    
    # دعم متغيرات بيئة المنصات السحابية
    node_id = args.id or os.getenv("NODE_ID", "mesh_alpha_seed")
    bind_host = args.host
    port = int(os.getenv("PORT", args.port))
    
    hf_space_id = os.getenv("SPACE_ID")
    if hf_space_id:
        space_user, space_name = hf_space_id.split("/")
        node_host = f"{space_user}-{space_name}.hf.space".replace("_", "-").lower()
    else:
        node_host = bind_host if bind_host != "0.0.0.0" else "127.0.0.1"

    logger.info(f"🔮 Initializing Distributed Node with Surah Awareness: {node_id}")
    node = LivingMeshNode(node_id=node_id, host=node_host, port=port)
    
    seed_nodes = []
    # دعم اكتشاف عقدة البذور عبر متغيرات البيئة (مفيد للتوسع السريع)
    env_seed_url = os.getenv("SEED_NODE_URL")
    if env_seed_url:
        seed_host = env_seed_url.replace("https://", "").replace("http://", "").replace("wss://", "").replace("ws://", "").strip("/")
        seed_nodes.append({
            "id": "seed_node",
            "host": seed_host,
            "port": 443 if ".hf.space" in seed_host else 80
        })
        logger.info(f"🌐 Found seed node via environment: {seed_host}")
    elif args.seed_host:
        seed_nodes.append({
            "id": "seed_node",
            "host": args.seed_host,
            "port": args.seed_port or 80
        })

    # إعداد تطبيق aiohttp
    app = web.Application()
    app['node'] = node
    app.add_routes([
        web.get('/status', handle_status),
        web.get('/health', handle_health),
        web.get('/v2/status', handle_v2_status),
        web.get('/dashboard', handle_dashboard),
        web.get('/ws', handle_ws),
        web.get('/', handle_status),
    ])
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, bind_host, port)
    await site.start()
    
    # الانضمام للشبكة
    node.join_network(seed_nodes=seed_nodes)
    
    logger.info(f"🚀 NSM Hybrid Node '{node.node_id}' is LIVE on port {port}")
    
    # مهمة نبض القلب الدورية + قياس صحة الأقران دورياً
    async def heartbeat_loop():
        tick = 0
        while True:
            node.send_heartbeat()
            node.check_network_health()
            tick += 1
            # كل ~60 ثانية: قياس RTT للأقران (#2)
            if tick % 4 == 0:
                try:
                    health = await node.measure_peers_health(timeout=4.0)
                    reachable = sum(1 for h in health if h.get("ok"))
                    logger.info(f"📶 Peer health check: {reachable}/{len(health)} reachable")
                except Exception as e:
                    logger.warning(f"⚠️ Peer health check failed: {e}")
            await asyncio.sleep(15)
    
    asyncio.create_task(heartbeat_loop())
    
    # إبقاء السكربت يعمل
    await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Node shutting down gracefully...")
