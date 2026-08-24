import time
import json
import sys
import os
from pathlib import Path

# إضافة مسار المشروع للاستيراد
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.shared_experience import SharedExperienceManager
from ai.memory_manager import MemoryManager

def run_e2ee_test():
    print("🔒 بدء اختبار التشفير الشامل (E2EE)...")
    print("=" * 60)
    
    ENCRYPTION_KEY = "nsm_super_secret_key_2026"
    STORAGE_PATH = "artifacts/learning/e2ee_test_knowledge.json"
    
    # تنظيف ملف الاختبار القديم إن وجد
    if os.path.exists(STORAGE_PATH):
        os.remove(STORAGE_PATH)
        
    # 1. إنشاء مدير الخبرة مع التشفير
    manager = SharedExperienceManager(storage_path=STORAGE_PATH, encryption_key=ENCRYPTION_KEY)
    
    # 2. مشاركة حقيقة حساسة
    sensitive_content = "SECRET_CODE: 998877 - This should be encrypted in storage!"
    print(f"📝 مشاركة حقيقة حساسة: {sensitive_content}")
    
    fact = {
        "content": sensitive_content,
        "strength": 0.9,
        "semantic_hash": "hash_123"
    }
    
    print("--- Testing Encryption directly ---")
    enc = manager._encrypt(sensitive_content)
    print(f"Encrypted: {enc[:20]}...")
    dec = manager._decrypt(enc)
    print(f"Decrypted: {dec}")
    
    print("--- Calling share_fact ---")
    manager.share_fact("Agent_Alpha", fact)
    print("--- share_fact finished ---")
    
    # 3. فحص ملف التخزين مباشرة للتأكد من التشفير
    print("\n🔍 فحص ملف التخزين (القرص)...")
    with open(STORAGE_PATH, "r") as f:
        data = json.load(f)
        shared_facts = data["shared_facts"]
        for fid, fdata in shared_facts.items():
            stored_content = fdata["content"]
            print(f"📦 المحتوى المخزن على القرص: {stored_content[:50]}...")
            if sensitive_content in stored_content:
                print("❌ فشل: المحتوى الحساس ظاهر بوضوح في ملف التخزين!")
            else:
                print("✅ نجاح: المحتوى مشفر ولا يمكن قراءته مباشرة.")
    
    # 4. محاكاة وكيل آخر يستلم الحقيقة ويفك تشفيرها
    print("\n🔄 محاكاة وكيل آخر يستلم الحقيقة...")
    receiver_manager = SharedExperienceManager(storage_path=STORAGE_PATH, encryption_key=ENCRYPTION_KEY)
    
    class MockAgentMemory:
        def __init__(self):
            self.agent_id = "Agent_Beta"
            self.ltm_semantic = {}
        def add_fact(self, content, importance, semantic_hash=None):
            self.ltm_semantic[semantic_hash] = {"content": content}

    mock_memory = MockAgentMemory()
    receiver_manager.sync_agent_memory(mock_memory)
    
    received_content = list(mock_memory.ltm_semantic.values())[0]["content"]
    print(f"🔓 المحتوى بعد فك التشفير لدى الوكيل المستلم: {received_content}")
    
    if received_content == sensitive_content:
        print("✅ نجاح: تم فك التشفير واستعادة النص الأصلي بنجاح.")
    else:
        print("❌ فشل: تعذر استعادة النص الأصلي.")

    print("=" * 60)
    print("🏁 انتهى اختبار التشفير الشامل.")

if __name__ == "__main__":
    run_e2ee_test()
