import sys
import time
import subprocess
import requests
from pathlib import Path

# إضافة مسار المشروع للاستيراد
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.shared_experience import SharedExperienceManager
from ai.memory_manager import MemoryManager

def run_distributed_simulation():
    print("🌐 بدء محاكاة السرب الموزع (Distributed Swarm Simulation)...")
    print("=" * 60)
    
    # 1. تشغيل خادم الذاكرة في الخلفية
    print("\n🖥️ تشغيل خادم الذاكرة الجماعية (Memory Server)...")
    server_process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve().parent.parent / "ai" / "memory_server.py")],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    # الانتظار حتى يعمل الخادم
    time.sleep(3)
    
    try:
        # 2. إعداد وكيلين يعملان بنظام العميل (Client)
        remote_url = "http://localhost:8080"
        print(f"\n🤖 إعداد الوكلاء للاتصال بـ {remote_url}...")
        
        shared_1 = SharedExperienceManager(remote_url=remote_url)
        shared_2 = SharedExperienceManager(remote_url=remote_url)
        
        mem_1 = MemoryManager(agent_id="Cloud_Agent_1")
        mem_2 = MemoryManager(agent_id="Cloud_Agent_2")
        
        # 3. الوكيل الأول يكتشف حقيقة وينشرها سحابياً
        print("\n☁️ [Cloud_Agent_1]: يكتشف حقيقة وينشرها في السحابة...")
        fact = {"content": "تم تحسين استهلاك الذاكرة في الخادم الموزع عبر FAISS", "strength": 0.9}
        shared_1.share_fact("Cloud_Agent_1", fact)
        
        # 4. الوكيل الثاني يزامن ذاكرته من السحابة
        print("\n☁️ [Cloud_Agent_2]: يزامن ذاكرته من السحابة...")
        count = shared_2.sync_agent_memory(mem_2)
        print(f"تم مزامنة {count} حقائق.")
        
        # 5. التحقق من نجاح المزامنة
        print(f"محتوى ذاكرة الوكيل 2: {list(mem_2.ltm_semantic.values())}")
        exists = any("FAISS" in f.get("content", "") for f in mem_2.ltm_semantic.values())
        if exists:
            print("✅ نجاح: الوكيل الثاني استلم الخبرة من السحابة بنجاح!")
        else:
            print("❌ فشل: لم يتم مزامنة الخبرة عبر الخادم.")

    finally:
        print("\n🛑 إيقاف خادم الذاكرة...")
        server_process.terminate()

    print("=" * 60)
    print("🏁 انتهت محاكاة السرب الموزع.")

if __name__ == "__main__":
    run_distributed_simulation()
