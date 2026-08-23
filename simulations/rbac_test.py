import time
import sys
import os
import threading
from pathlib import Path

# إضافة مسار المشروع للاستيراد
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.shared_experience import SharedExperienceManager
import uvicorn
from ai.memory_server import app

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8081, log_level="error")

def run_rbac_test():
    print("🔐 بدء اختبار التحكم في الوصول القائم على الأدوار (RBAC Test)...")
    print("=" * 65)
    
    # تشغيل السيرفر في خلفية منفصلة
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2) # انتظار تشغيل السيرفر
    
    REMOTE_URL = "http://127.0.0.1:8081"
    
    # 1. اختبار وكيل بامتيازات "مشاهد" (Viewer) - لا يمكنه الكتابة
    print("👤 [Viewer_Agent]: محاولة مشاركة حقيقة...")
    viewer = SharedExperienceManager(remote_url=REMOTE_URL, api_key="nsm_viewer_token_2026")
    res = viewer.share_fact("Viewer_Agent", {"content": "SECRET_DATA", "strength": 0.9})
    
    if not res:
        print("✅ نجاح: تم رفض وصول الوكيل 'المشاهد' للكتابة.")
    else:
        print("❌ فشل: تم السماح للوكيل 'المشاهد' بالكتابة!")

    # 2. اختبار وكيل بامتيازات "عامل" (Worker) - يمكنه الكتابة والقراءة
    print("\n👤 [Worker_Agent]: محاولة مشاركة حقيقة...")
    worker = SharedExperienceManager(remote_url=REMOTE_URL, api_key="nsm_worker_token_2026")
    res = worker.share_fact("Worker_Agent", {"content": "WORKER_DATA", "strength": 0.9})
    
    if res:
        print("✅ نجاح: تم السماح للوكيل 'العامل' بالكتابة.")
    else:
        print("❌ فشل: تم رفض وصول الوكيل 'العامل' للكتابة!")

    # 3. اختبار وكيل بامتيازات "مسؤول" (Admin) - يمكنه الوصول للمقاييس
    print("\n👤 [Admin_Agent]: محاولة الوصول للمقاييس (Metrics)...")
    admin = SharedExperienceManager(remote_url=REMOTE_URL, api_key="nsm_admin_token_2026")
    metrics = admin._request("GET", "/metrics")
    
    if metrics:
        print(f"✅ نجاح: تم السماح للمسؤول بالوصول. إجمالي الطلبات: {metrics.get('total_requests')}")
    else:
        print("❌ فشل: تم رفض وصول المسؤول للمقاييس!")

    # 4. اختبار وكيل بامتيازات "عامل" - محاولة الوصول للمقاييس (ممنوع)
    print("\n👤 [Worker_Agent]: محاولة الوصول للمقاييس (ممنوع)...")
    metrics_forbidden = worker._request("GET", "/metrics")
    
    if not metrics_forbidden:
        print("✅ نجاح: تم منع الوكيل 'العامل' من الوصول لبيانات الإدارة.")
    else:
        print("❌ فشل: تم السماح للوكيل 'العامل' بالوصول لبيانات الإدارة!")

    print("=" * 65)
    print("🏁 انتهى اختبار RBAC.")

if __name__ == "__main__":
    run_rbac_test()
