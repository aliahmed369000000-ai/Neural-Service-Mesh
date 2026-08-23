import sys
import os
import json
from pathlib import Path

# إضافة المسار للمشروع
sys.path.append(os.getcwd())

from ai.multimodal_memory import MultimodalMemory

def test_sharding_distribution():
    print("--- اختبار توزيع البيانات عبر الأجزاء (Shards) ---")
    memory = MultimodalMemory()
    
    # 1. تخزين عدة أصول للتحقق من التوزيع
    num_assets = 10
    asset_ids = []
    
    for i in range(num_assets):
        dummy_file = f"test_shard_{i}.txt"
        with open(dummy_file, "w") as f:
            f.write(f"content {i}")
            
        metadata = {
            "description": f"وصف الأصل رقم {i}",
            "tags": ["test", "sharding"]
        }
        
        asset_id = memory.store_asset("agent_test", dummy_file, "text", metadata)
        asset_ids.append(asset_id)
        os.remove(dummy_file)
        
    print(f"تم تخزين {num_assets} أصول.")
    
    # 2. التحقق من وجود الملفات في مجلد sharding
    shards_dir = memory.storage_dir / "shards"
    found_assets_count = 0
    shards_with_data = 0
    
    for i in range(4):  # الافتراضي 4 أجزاء
        shard_path = shards_dir / f"shard_{i}.json"
        if shard_path.exists():
            with open(shard_path, "r") as f:
                data = json.load(f)
                count = len(data["assets"])
                print(f"الجزء {i} يحتوي على {count} أصول.")
                found_assets_count += count
                if count > 0:
                    shards_with_data += 1
                    
    assert found_assets_count >= num_assets, "فشل: لم يتم العثور على جميع الأصول في الأجزاء"
    assert shards_with_data > 1, "فشل: لم يتم توزيع البيانات على أكثر من جزء"
    print("✅ نجح اختبار توزيع التجزئة!")

def test_sharded_search():
    print("\n--- اختبار البحث في الأجزاء (Sharded Search) ---")
    memory = MultimodalMemory()
    
    # البحث عن أصل معين تم تخزينه في الاختبار السابق
    query = "وصف الأصل رقم 5"
    results = memory.search_assets(query, use_shards=True)
    
    assert len(results) > 0, "فشل: لم يتم العثور على الأصل عبر البحث في الأجزاء"
    assert query.lower() in results[0]["metadata"]["description"].lower()
    print(f"تم العثور على الأصل: {results[0]['id']}")
    print("✅ نجح اختبار البحث في الأجزاء!")

if __name__ == "__main__":
    try:
        test_sharding_distribution()
        test_sharded_search()
    except Exception as e:
        print(f"❌ فشل الاختبار: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
