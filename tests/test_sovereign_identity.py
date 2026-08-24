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
    print("🚀 Starting Sovereign Identity & Secure P2P Test...")
    
    # تنظيف الحالة السابقة
    from ai.living_mesh import NETWORK_STATE
    if NETWORK_STATE.exists():
        NETWORK_STATE.unlink()
    
    keys_dir = LIVING_MESH_DIR / "keys"
    if keys_dir.exists():
        import shutil
        shutil.rmtree(keys_dir)
    
    # 1. إنشاء العقدة الأولى (Alpha) - ستقوم بإنشاء مفاتيحها تلقائياً
    node_alpha = LivingMeshNode("node_alpha", port=9991)
    node_alpha.join_network()
    
    # 2. إنشاء العقدة الثانية (Zeta)
    node_zeta = LivingMeshNode("node_zeta", port=9992)
    node_zeta.join_network()
    
    # 3. تشغيل الخوادم في الخلفية
    server_alpha = asyncio.create_task(node_alpha.start_node_server())
    server_zeta = asyncio.create_task(node_zeta.start_node_server())
    
    await asyncio.sleep(2)  # انتظار بدء الخوادم وتبادل المفاتيح العامة (عبر الدليل المشترك حالياً)
    
    print("\n🔒 Sending Secure message from Alpha to Zeta...")
    # 4. إرسال خبرة من Alpha إلى Zeta
    await node_alpha.send_to_peer("127.0.0.1", 9992, "sovereign_ping", {"status": "secure"})
    
    await asyncio.sleep(2)
    
    # 5. التحقق من وصول الخبرة لـ Zeta والتحقق من التوقيع
    state = node_zeta._load_state()
    exps = [e for e in state.get("global_experience", []) if e.get("kind") == "sovereign_ping"]
    
    if exps:
        print(f"✅ Success! Zeta verified Alpha's identity and received the secure message.")
        print(f"Verified Content: {exps[0]['data']}")
    else:
        print("❌ Failure: Message was rejected or not received.")
        
    # 6. محاكاة هجوم (رسالة من عقدة مجهولة أو بتوقيع خاطئ)
    print("\n🛡️ Simulating attack from Unknown Node...")
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", 9992)
        fake_msg = {
            "payload": {"from": "hacker_node", "kind": "malicious_exp", "data": {}},
            "signature": "fake_signature_base64"
        }
        writer.write(json.dumps(fake_msg).encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    except: pass
    
    await asyncio.sleep(1)
    state = node_zeta._load_state()
    malicious = [e for e in state.get("global_experience", []) if e.get("kind") == "malicious_exp"]
    if not malicious:
        print("✅ Defense Success! Zeta rejected the unverified message from hacker_node.")
    else:
        print("❌ Defense Failure: Zeta accepted an unverified message!")
    
    # إغلاق الخوادم
    node_alpha.server.close()
    node_zeta.server.close()
    await asyncio.gather(server_alpha, server_zeta, return_exceptions=True)
    print("\n🏁 Sovereign Test Completed.")

if __name__ == "__main__":
    asyncio.run(run_test())
