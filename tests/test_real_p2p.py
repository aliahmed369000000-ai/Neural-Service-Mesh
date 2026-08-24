import asyncio
import json
import logging
import sys
import os
from pathlib import Path

# إضافة مسار المشروع للاستيراد
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.living_mesh import LivingMeshNode

logging.basicConfig(level=logging.INFO)

async def run_test():
    print("🚀 Starting Real P2P Integration Test...")
    
    # تنظيف الحالة السابقة
    from ai.living_mesh import NETWORK_STATE
    if NETWORK_STATE.exists():
        NETWORK_STATE.unlink()
    
    # 1. إنشاء العقدة الأولى (Alpha)
    node_alpha = LivingMeshNode("node_alpha", port=8888)
    node_alpha.join_network()
    
    # 2. إنشاء العقدة الثانية (Zeta)
    node_zeta = LivingMeshNode("node_zeta", port=8889)
    node_zeta.join_network()
    
    # 3. تشغيل الخوادم في الخلفية
    server_alpha = asyncio.create_task(node_alpha.start_node_server())
    server_zeta = asyncio.create_task(node_zeta.start_node_server())
    
    await asyncio.sleep(2)  # انتظار بدء الخوادم
    
    print("\n📡 Sending experience from Alpha to Zeta via Real P2P...")
    # 4. إرسال خبرة من Alpha إلى Zeta
    await node_alpha.send_to_peer("127.0.0.1", 8889, "quantum_boost", {"power": 9000})
    
    await asyncio.sleep(2)
    
    # 5. التحقق من وصول الخبرة لـ Zeta
    state = node_zeta._load_state()
    exps = [e for e in state.get("global_experience", []) if e.get("kind") == "quantum_boost"]
    
    if exps:
        print(f"✅ Success! Zeta received experience: {exps[0]['kind']}")
        print(f"Data: {exps[0]['data']}")
    else:
        print("❌ Failure: Zeta did not receive the experience.")
    
    # إغلاق الخوادم
    node_alpha.server.close()
    node_zeta.server.close()
    await asyncio.gather(server_alpha, server_zeta, return_exceptions=True)
    print("\n🏁 Test Completed.")

if __name__ == "__main__":
    asyncio.run(run_test())
