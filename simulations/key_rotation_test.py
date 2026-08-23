import time
import sys
import os
from pathlib import Path

# إضافة مسار المشروع للاستيراد
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.shared_experience import SharedExperienceManager

def run_rotation_test():
    print("🔄 بدء اختبار تدوير المفاتيح تلقائياً (Key Rotation Test)...")
    print("=" * 65)
    
    MASTER_SECRET = "nsm_master_secret_2026"
    STORAGE_PATH = "artifacts/learning/rotation_test_knowledge.json"
    
    if os.path.exists(STORAGE_PATH):
        os.remove(STORAGE_PATH)
        
    manager = SharedExperienceManager(storage_path=STORAGE_PATH, encryption_key=MASTER_SECRET)
    
    # 1. تشفير معلومة بمفتاح اليوم الحالي
    print("📅 [اليوم الحالي]: تشفير معلومة...")
    fact_v1 = "DATA_ENCRYPTED_TODAY"
    encrypted_v1 = manager._encrypt(fact_v1)
    print(f"📦 المحتوى المشفر (V1): {encrypted_v1[:40]}...")
    
    # 2. محاكاة مرور 24 ساعة (تغيير المفتاح)
    print("\n⏳ [محاكاة]: مرور 24 ساعة...")
    # تعديل داخلي لمحاكاة التغيير الزمني
    future_time = time.time() + 86401 
    new_key_id = manager._get_temporal_key_id(future_time)
    print(f"🔑 المفتاح الجديد المتوقع: {new_key_id}")
    
    # 3. تشفير معلومة جديدة بالمفتاح الجديد
    print("\n📅 [اليوم التالي]: تشفير معلومة جديدة...")
    # نقوم بتحديث المفتاح يدوياً للمحاكاة
    manager.current_key_id = new_key_id
    manager.cipher = manager._init_cipher(f"{MASTER_SECRET}_{new_key_id}")
    manager.ciphers[new_key_id] = manager.cipher
    
    fact_v2 = "DATA_ENCRYPTED_TOMORROW"
    encrypted_v2 = manager._encrypt(fact_v2)
    print(f"📦 المحتوى المشفر (V2): {encrypted_v2[:40]}...")
    
    # 4. محاولة فك تشفير المعلومتين معاً
    print("\n🔍 [التحقق]: محاولة فك تشفير البيانات القديمة والجديدة...")
    
    decrypted_v1 = manager._decrypt(encrypted_v1)
    decrypted_v2 = manager._decrypt(encrypted_v2)
    
    print(f"🔓 فك تشفير V1: {decrypted_v1}")
    print(f"🔓 فك تشفير V2: {decrypted_v2}")
    
    if decrypted_v1 == fact_v1 and decrypted_v2 == fact_v2:
        print("\n✅ نجاح: تم فك تشفير البيانات من إصدارات مفاتيح مختلفة بنجاح!")
    else:
        print("\n❌ فشل: تعذر فك تشفير بعض البيانات.")

    print("=" * 65)
    print("🏁 انتهى اختبار تدوير المفاتيح.")

if __name__ == "__main__":
    run_rotation_test()
