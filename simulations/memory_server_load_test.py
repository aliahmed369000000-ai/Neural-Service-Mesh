import time
import asyncio
import aiohttp
import statistics
import subprocess
import sys
from pathlib import Path

# إعدادات الاختبار
SERVER_URL = "http://localhost:8081"
API_KEY = "nsm_secret_key_2026"
NUM_AGENTS = 500  # عدد الوكلاء الوهميين
REQUESTS_PER_AGENT = 20  # عدد الطلبات لكل وكيل
TOTAL_REQUESTS = NUM_AGENTS * REQUESTS_PER_AGENT

async def simulate_agent(agent_id, session):
    latencies = []
    for i in range(REQUESTS_PER_AGENT):
        start_time = time.time()
        try:
            # محاكاة مشاركة حقيقة
            payload = {
                "agent_id": f"Agent_{agent_id}",
                "content": f"Test fact {i} from agent {agent_id}",
                "importance": 0.8
            }
            async with session.post(f"{SERVER_URL}/share", json=payload, headers={"X-NSM-Token": API_KEY}) as response:
                if response.status == 200:
                    latencies.append(time.time() - start_time)
                else:
                    print(f"❌ Agent {agent_id} failed: {response.status}")
        except Exception as e:
            print(f"❌ Agent {agent_id} error: {e}")
    return latencies

async def run_load_test():
    print(f"🚀 بدء اختبار التحميل على خادم الذاكرة...")
    print(f"👥 الوكلاء المتزامنون: {NUM_AGENTS}")
    print(f"📊 إجمالي الطلبات: {TOTAL_REQUESTS}")
    print("-" * 50)

    # تشغيل الخادم في الخلفية على منفذ 8081 لتجنب التعارض
    server_path = Path(__file__).resolve().parent.parent / "ai" / "memory_server.py"
    server_process = subprocess.Popen(
        [sys.executable, str(server_path)],
        env={"PORT": "8081"}, # سنقوم بتعديل السيرفر ليقبل PORT من البيئة
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # انتظار بدء السيرفر (نحتاج لتعديل السيرفر ليقرأ المنفذ)
    await asyncio.sleep(3)

    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [simulate_agent(i, session) for i in range(NUM_AGENTS)]
        results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    all_latencies = [l for agent_lats in results for l in agent_lats]
    
    if all_latencies:
        print("\n📈 نتائج الأداء:")
        print(f"✅ الطلبات الناجحة: {len(all_latencies)}/{TOTAL_REQUESTS}")
        print(f"⏱️ إجمالي الوقت: {total_time:.2f} ثانية")
        print(f"⚡ معدل النقل: {len(all_latencies) / total_time:.2f} طلب/ثانية")
        print(f"📉 متوسط زمن الاستجابة: {statistics.mean(all_latencies)*1000:.2f} مللي ثانية")
        print(f"🏔️ أقصى زمن استجابة: {max(all_latencies)*1000:.2f} مللي ثانية")
        print(f"🎯 المئوي 95 (P95): {statistics.quantiles(all_latencies, n=20)[18]*1000:.2f} مللي ثانية")
    else:
        print("❌ فشل الاختبار: لم تنجح أي طلبات.")

    server_process.terminate()

if __name__ == "__main__":
    asyncio.run(run_load_test())
