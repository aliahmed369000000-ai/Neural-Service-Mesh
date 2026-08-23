import time
import requests
import subprocess
import sys
import json
from pathlib import Path

def run_monitoring_test():
    print("📊 بدء اختبار نظام المراقبة (Monitoring Test)...")
    print("=" * 60)
    
    # إعدادات
    SERVER_URL = "http://localhost:8082"
    API_KEY = "nsm_secret_key_2026"
    
    # 1. تشغيل الخادم
    server_path = Path(__file__).resolve().parent.parent / "ai" / "memory_server.py"
    server_process = subprocess.Popen(
        [sys.executable, str(server_path)],
        env={"PORT": "8082"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    time.sleep(3)
    
    try:
        # 2. توليد نشاط من عدة وكلاء
        agents = ["Agent_A", "Agent_B", "Agent_C"]
        for agent in agents:
            print(f"🤖 {agent}: يرسل طلبات للخادم...")
            # مشاركة حقيقة
            requests.post(
                f"{SERVER_URL}/share", 
                json={"agent_id": agent, "content": f"Fact from {agent}", "importance": 0.8},
                headers={"X-NSM-Token": API_KEY}
            )
            # مزامنة
            requests.get(f"{SERVER_URL}/sync?agent_id={agent}", headers={"X-NSM-Token": API_KEY})
        
        # 3. جلب مقاييس الأداء
        print("\n📈 جلب مقاييس الأداء من Metrics API...")
        response = requests.get(f"{SERVER_URL}/metrics", headers={"X-NSM-Token": API_KEY})
        
        if response.status_code == 200:
            metrics = response.json()
            print("-" * 30)
            print(f"⏱️ وقت التشغيل: {metrics['uptime_seconds']:.2f} ثانية")
            print(f"🔢 إجمالي الطلبات: {metrics['total_requests']}")
            print(f"👥 الوكلاء النشطون: {metrics['active_agents_count']}")
            print(f"🧠 الحقائق المخزنة: {metrics['memory_usage']['facts_count']}")
            print("-" * 30)
            print("✅ نجاح: تم جمع مقاييس الأداء بنجاح!")
        else:
            print(f"❌ فشل: لم يتمكن من جلب المقاييس. الحالة: {response.status_code}")

    finally:
        server_process.terminate()

    print("=" * 60)
    print("🏁 انتهى اختبار المراقبة.")

if __name__ == "__main__":
    run_monitoring_test()
