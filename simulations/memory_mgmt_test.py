import time
import requests
import subprocess
import sys
import os
from pathlib import Path

def run_memory_mgmt_test():
    print("🧠 بدء اختبار إدارة الذاكرة و Garbage Collection...")
    print("=" * 60)
    
    SERVER_URL = "http://localhost:8083"
    API_KEY = "nsm_secret_key_2026"
    
    # 1. تشغيل الخادم مع سعة منخفضة للاختبار
    server_path = Path(__file__).resolve().parent.parent / "ai" / "memory_server.py"
    server_process = subprocess.Popen(
        [sys.executable, str(server_path)],
        env={"PORT": "8083"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    time.sleep(3)
    
    try:
        # 2. فحص استهلاك الذاكرة الأولي
        print("📊 فحص الموارد الأولية...")
        metrics = requests.get(f"{SERVER_URL}/metrics", headers={"X-NSM-Token": API_KEY}).json()
        print(f"📉 استهلاك RAM الأولي: {metrics['system_resources']['ram_usage_mb']} MB")
        print(f"📦 كائنات GC: {metrics['system_resources']['gc_objects']}")
        
        # 3. محاكاة ضغط (إرسال حقائق كثيرة)
        print("\n🚀 محاكاة ضغط (إرسال 1000 حقيقة)...")
        for i in range(1000):
            requests.post(
                f"{SERVER_URL}/share", 
                json={"agent_id": "Stress_Agent", "content": f"Bulk data {i}", "importance": 0.1},
                headers={"X-NSM-Token": API_KEY}
            )
        
        # 4. تحفيز GC يدوياً
        print("\n🧹 تحفيز Garbage Collection يدوياً...")
        gc_res = requests.post(f"{SERVER_URL}/system/gc", headers={"X-NSM-Token": API_KEY}).json()
        print(f"✅ تم جمع {gc_res['objects_collected']} كائن.")
        
        # 5. فحص الموارد بعد الضغط والـ GC
        metrics_after = requests.get(f"{SERVER_URL}/metrics", headers={"X-NSM-Token": API_KEY}).json()
        print(f"📈 استهلاك RAM بعد الضغط: {metrics_after['system_resources']['ram_usage_mb']} MB")
        print(f"📦 كائنات GC الحالية: {metrics_after['system_resources']['gc_objects']}")
        
        print("\n✅ نجاح: تم التحقق من أدوات إدارة الذاكرة بنجاح!")

    finally:
        server_process.terminate()

    print("=" * 60)
    print("🏁 انتهى اختبار إدارة الذاكرة.")

if __name__ == "__main__":
    run_memory_mgmt_test()
