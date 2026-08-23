#!/usr/bin/env python3
"""
NSM Distributed Node Launcher — مشغل العقد الموزعة لمشروع NSM
==========================================================
يسمح هذا السكربت بتشغيل العقد ككيانات مستقلة عبر خوادم حقيقية.
"""
import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path

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

async def main():
    parser = argparse.ArgumentParser(description="NSM Distributed Node Launcher")
    parser.add_argument("--id", type=str, help="Node ID (e.g., mesh_zeta, mesh_omega)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--seed-host", type=str, help="Seed node host IP to join network")
    parser.add_argument("--seed-port", type=int, help="Seed node port to join network")
    parser.add_argument("--public-ip", type=str, help="Public IP address if behind NAT")
    
    args = parser.parse_args()
    
    # دعم متغيرات بيئة Render و Cloud Platforms
    node_id = args.id or os.getenv("NODE_ID", "mesh_node_" + os.getenv("RENDER_SERVICE_ID", "unknown")[:8])
    host = args.host
    # Render يخصص المنفذ تلقائياً عبر متغير البيئة PORT
    port = int(os.getenv("PORT", args.port))
    
    # جلب عنوان الـ IP العام أو الـ Domain الخاص بالمنصات السحابية
    render_external_url = os.getenv("RENDER_EXTERNAL_URL")
    hf_space_id = os.getenv("SPACE_ID") # Hugging Face Space ID (user/space-name)
    
    if render_external_url:
        node_host = render_external_url.replace("https://", "").replace("http://", "").strip("/")
    elif hf_space_id:
        # بناء رابط Hugging Face Space: user-space-name.hf.space
        space_user, space_name = hf_space_id.split("/")
        node_host = f"{space_user}-{space_name}.hf.space".replace("_", "-").lower()
    else:
        node_host = args.public_ip if args.public_ip else host
        
    if node_host == "0.0.0.0":
        # محاولة جلب الـ IP المحلي إذا كان Bind على 0.0.0.0
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            node_host = s.getsockname()[0]
            s.close()
        except:
            node_host = "127.0.0.1"

    logger.info(f"🔮 Initializing Distributed Node: {node_id or 'Auto-Generated'}")
    node = LivingMeshNode(node_id=node_id, host=node_host, port=port)
    
    # تحديث الـ Host الفعلي لـ Binding
    node.bind_host = host 
    
    seed_nodes = []
    if args.seed_host and args.seed_port:
        seed_nodes.append({
            "id": "seed_node",
            "host": args.seed_host,
            "port": args.seed_port
        })
        logger.info(f"🌐 Will attempt to join network via seed: {args.seed_host}:{args.seed_port}")

    # بدء خادم العقدة
    try:
        import websockets
        async with websockets.serve(node._handle_ws_connection, host, port) as server:
            node.port = server.sockets[0].getsockname()[1]
            node.server = server
            
            # الانضمام للشبكة
            node.join_network(seed_nodes=seed_nodes)
            
            logger.info(f"🚀 NSM Node '{node.node_id}' is LIVE!")
            logger.info(f"📡 Internal Address: ws://{host}:{port}")
            logger.info(f"🌍 Public/Mesh Address: ws://{node_host}:{node.port}")
            
            # مهمة نبض القلب الدورية
            async def heartbeat_loop():
                while True:
                    node.send_heartbeat()
                    node.check_network_health()
                    await asyncio.sleep(15)
            
            asyncio.create_task(heartbeat_loop())
            
            # إبقاء السكربت يعمل
            await asyncio.Future()
            
    except Exception as e:
        logger.error(f"❌ Failed to start node: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Node shutting down gracefully...")
