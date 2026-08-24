import asyncio
import json
import logging
import sys
import os
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
    
    # 3. تشغيل الخوادم في الخلفية
    server_alpha = asyncio.create_task(node_alpha.start_node_server())
    server_zeta = asyncio.create_task(node_zeta.start_node_server())
    
    await asyncio.sleep(2)  # انتظار بدء الخوادم
    
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
    node_alpha.server.close()
    node_zeta.server.close()
    print("\n🏁 Secure WebSocket Test Completed.")
    # ملاحظة: asyncio loop سيتوقف عند انتهاء المهام

if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Test Error: {e}")
        sys.exit(1)
    os._exit(0) # إنهاء قسري لتجنب تعليق المهام الخلفية
