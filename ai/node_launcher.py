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

async def handle_ws(request):
    """معالج WebSocket عبر aiohttp لضمان التوافق مع Hugging Face."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    node = request.app['node']
    await node._handle_aiohttp_ws(ws)
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
        web.get('/ws', handle_ws),
        web.get('/', handle_status) # Health check endpoint
    ])
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, bind_host, port)
    await site.start()
    
    # الانضمام للشبكة
    node.join_network(seed_nodes=seed_nodes)
    
    logger.info(f"🚀 NSM Hybrid Node '{node.node_id}' is LIVE on port {port}")
    
    # مهمة نبض القلب الدورية
    async def heartbeat_loop():
        while True:
            node.send_heartbeat()
            node.check_network_health()
            await asyncio.sleep(15)
    
    asyncio.create_task(heartbeat_loop())
    
    # إبقاء السكربت يعمل
    await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Node shutting down gracefully...")
