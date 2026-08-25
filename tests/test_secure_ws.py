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
    print("🚀 Starting Secure WebSocket Integration Test...")
    
    # تنظيف الحالة السابقة
    from ai.living_mesh import NETWORK_STATE
    if NETWORK_STATE.exists():
        NETWORK_STATE.unlink()
    
    keys_dir = LIVING_MESH_DIR / "keys"
    if keys_dir.exists():
        import shutil
        shutil.rmtree(keys_dir)
    
    # 1. إنشاء العقدة الأولى (Alpha)
    node_alpha = LivingMeshNode("node_alpha", port=7771)
    
    # 2. إنشاء العقدة الثانية (Zeta)
    node_zeta = LivingMeshNode("node_zeta", port=7772)
    
    # 3. تشغيل الخوادم فعلياً عبر websockets.serve()
    server_alpha = await websockets.serve(node_alpha._handle_ws_connection, "127.0.0.1", node_alpha.port)
    server_zeta = await websockets.serve(node_zeta._handle_ws_connection, "127.0.0.1", node_zeta.port)
    node_alpha.server = server_alpha
    node_zeta.server = server_zeta

    await asyncio.sleep(1)
    
    print("\n🌐 Sending message via Secure WebSocket from Alpha to Zeta...")
    # 4. إرسال خبرة من Alpha إلى Zeta عبر WS
    await node_alpha.send_to_peer("127.0.0.1", 7772, "ws_sovereign_sync", {"status": "streaming"})
    
    await asyncio.sleep(2)
    
    # 5. التحقق من وصول الخبرة لـ Zeta
    state = node_zeta._load_state()
    exps = [e for e in state.get("global_experience", []) if e.get("kind") == "ws_sovereign_sync"]
    
    if exps:
        print(f"✅ Success! Zeta received and verified WebSocket message: {exps[0]['kind']}")
        print(f"Data: {exps[0]['data']}")
    else:
        print("❌ Failure: WebSocket message was not received or verified.")
    
    # إغلاق الخوادم
    server_alpha.close()
    server_zeta.close()
    await asyncio.gather(server_alpha.wait_closed(), server_zeta.wait_closed(), return_exceptions=True)
    print("\n🏁 Secure WebSocket Test Completed.")

if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Test Error: {e}")
        sys.exit(1)
    os._exit(0) # إنهاء قسري لتجنب تعليق المهام الخلفية
