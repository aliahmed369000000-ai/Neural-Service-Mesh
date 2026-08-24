import asyncio
import json
# import pytest
import sys
from pathlib import Path

# إضافة مسار المشروع
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.living_mesh import LivingMeshNode

async def test_distributed_simulation():
    """محاكاة تواصل بين عقدتين بعناوين IP مختلفة (افتراضية)."""
    # عقدة البداية (Seed Node)
    alpha = LivingMeshNode(node_id="mesh_alpha_dist", host="127.0.0.1", port=9001)
    
    # عقدة ثانية تنضم للشبكة
    zeta = LivingMeshNode(node_id="mesh_zeta_dist", host="127.0.0.1", port=9002)
    
    # بدء الخوادم
    import websockets
    
    stop_alpha = asyncio.Event()
    stop_zeta = asyncio.Event()
    
    async def run_alpha():
        async with websockets.serve(alpha._handle_ws_connection, "127.0.0.1", 9001):
            alpha.join_network()
            await stop_alpha.wait()
            
    async def run_zeta():
        async with websockets.serve(zeta._handle_ws_connection, "127.0.0.1", 9002):
            # انضمام للشبكة عبر Alpha
            seed = {"id": "mesh_alpha_dist", "host": "127.0.0.1", "port": 9001}
            zeta.join_network(seed_nodes=[seed])
            await stop_zeta.wait()

    # تشغيل المهام
    task_alpha = asyncio.create_task(run_alpha())
    task_zeta = asyncio.create_task(run_zeta())
    
    # الانتظار قليلاً للاكتشاف
    await asyncio.sleep(2)
    
    # التحقق من أن Zeta اكتشفت Alpha
    state_zeta = zeta._load_state()
    discovered = any(n.get("id") == "mesh_alpha_dist" for n in state_zeta["nodes"].values())
    
    print(f"Discovered Alpha: {discovered}")
    
    # إرسال خبرة من Alpha إلى Zeta
    exp_data = {"test": "distributed_logic", "value": 1.0}
    await alpha.send_to_peer("127.0.0.1", 9002, "distributed_test", exp_data)
    
    await asyncio.sleep(1)
    
    # إغلاق المهام
    stop_alpha.set()
    stop_zeta.set()
    task_alpha.cancel()
    task_zeta.cancel()
    
    assert discovered, "Zeta should discover Alpha in the distributed simulation"

if __name__ == "__main__":
    asyncio.run(test_distributed_simulation())
