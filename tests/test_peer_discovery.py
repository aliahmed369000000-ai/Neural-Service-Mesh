import asyncio
import json
import logging
import sys
import os
import websockets
from pathlib import Path

# إضافة مسار المشروع للاستيراد
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.living_mesh import LivingMeshNode, LIVING_MESH_DIR

logging.basicConfig(level=logging.INFO)

async def run_test():
    print("🚀 Starting Decentralized Peer Discovery Test...")
    
    # تنظيف الحالة السابقة
    from ai.living_mesh import NETWORK_STATE
    if NETWORK_STATE.exists():
        NETWORK_STATE.unlink()
    
    keys_dir = LIVING_MESH_DIR / "keys"
    if keys_dir.exists():
        import shutil
        shutil.rmtree(keys_dir)
    
    # 1. إنشاء العقدة الأولى (Alpha) - ستكون العقدة البذرة (Seed)
    node_alpha = LivingMeshNode("node_alpha", port=8881)
    server_alpha = await websockets.serve(node_alpha._handle_ws_connection, "127.0.0.1", node_alpha.port)
    node_alpha.server = server_alpha
    # ملاحظة إصلاح: الاختبار الأصلي كان لا يستدعي join_network() لـ Alpha
    # إطلاقاً، فلم تكن تُسجَّل بالشبكة أصلاً (سبب حقيقي لفشل الاكتشاف،
    # منفصل عن مشكلة start_node_server المفقودة التي كانت تمنع وصول
    # التنفيذ لهذه النقطة أساساً).
    node_alpha.join_network()

    await asyncio.sleep(1)
    
    # 2. إنشاء العقدة الثانية (Zeta) - تنضم عبر Alpha
    node_zeta = LivingMeshNode("node_zeta", port=8882)
    server_zeta = await websockets.serve(node_zeta._handle_ws_connection, "127.0.0.1", node_zeta.port)
    node_zeta.server = server_zeta
    
    # انضمام Zeta للشبكة باستخدام Alpha كبذرة
    node_zeta.join_network(seed_nodes=[{"id": "node_alpha", "host": "127.0.0.1", "port": 8881}])
    
    await asyncio.sleep(2)
    
    # 3. إنشاء العقدة الثالثة (Rho) - تنضم عبر Zeta
    # هذا يختبر انتشار الأقران (Zeta ستخبر Rho عن Alpha)
    node_rho = LivingMeshNode("node_rho", port=8883)
    server_rho = await websockets.serve(node_rho._handle_ws_connection, "127.0.0.1", node_rho.port)
    node_rho.server = server_rho
    
    node_rho.join_network(seed_nodes=[{"id": "node_zeta", "host": "127.0.0.1", "port": 8882}])
    
    await asyncio.sleep(2)
    
    # 4. التحقق من النتائج
    state_rho = node_rho._load_state()
    discovered_nodes = state_rho.get("nodes", {})
    
    print(f"\n📊 Rho's Network Map: {list(discovered_nodes.keys())}")
    
    if "node_alpha" in discovered_nodes and "node_zeta" in discovered_nodes:
        print("✅ Success! Rho discovered Alpha indirectly via Zeta.")
    else:
        print("❌ Failure: Peer discovery propagation failed.")
        
    # إغلاق الخوادم
    server_alpha.close()
    server_zeta.close()
    server_rho.close()
    await asyncio.gather(
        server_alpha.wait_closed(), server_zeta.wait_closed(), server_rho.wait_closed(),
        return_exceptions=True,
    )
    print("\n🏁 Peer Discovery Test Completed.")

if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        pass
    os._exit(0)
