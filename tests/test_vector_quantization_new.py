import sys
import os
import numpy as np
import json
from pathlib import Path

# إضافة المسار للمشروع
sys.path.append(os.getcwd())

from ai.quantization_engine import VectorQuantizer
from ai.multimodal_memory import MultimodalMemory

def test_quantization_efficiency():
    print("--- اختبار كفاءة تكميم المتجهات ---")
    
    # 1. إنشاء بيانات وهمية (100 متجه ببعد 1536)
    dim = 1536
    num_vectors = 100
    original_vectors = np.random.rand(num_vectors, dim).astype(np.float32)
    
    # 2. تدريب المحرك
    quantizer = VectorQuantizer(dimension=dim, num_centroids=16)
    quantizer.train(original_vectors)
    
    # 3. حساب حجم البيانات الأصلي
    original_size = original_vectors.nbytes
    print(f"الحجم الأصلي للمتجهات: {original_size / 1024:.2f} KB")
    
    # 4. التكميم
    compressed_indices = [quantizer.quantize(v) for v in original_vectors]
    
    # 5. حساب حجم البيانات المضغوطة (Indices فقط)
    compressed_data_json = json.dumps(compressed_indices)
    compressed_size = len(compressed_data_json.encode('utf-8'))
    print(f"الحجم بعد التكميم (Indices): {compressed_size / 1024:.2f} KB")
    
    # 6. حساب نسبة الضغط
    compression_ratio = (1 - (compressed_size / original_size)) * 100
    print(f"نسبة الضغط المحققة: {compression_ratio:.2f}%")
    
    # 7. التحقق من الدقة الدلالية (الخطأ التربيعي المتوسط)
    reconstructed_vectors = np.array([quantizer.dequantize(idx) for idx in compressed_indices])
    mse = np.mean((original_vectors - reconstructed_vectors) ** 2)
    print(f"الخطأ التربيعي المتوسط (MSE): {mse:.6f}")
    
    # نسبة الضغط في JSON قد تكون أقل من 75% بسبب overhead الـ list والـ strings
    # لكنها مقارنة بـ float32 embeddings ضخمة
    assert compression_ratio > 50, f"فشل: نسبة الضغط {compression_ratio:.2f}% أقل من المتوقع"
    assert mse < 0.2, "فشل: الخطأ الدلالي كبير جداً"
    print("✅ نجح اختبار كفاءة الضغط والدقة!")

def test_memory_integration():
    print("\n--- اختبار تكامل الذاكرة مع التكميم ---")
    memory = MultimodalMemory()
    
    # إنشاء متجه وهمي
    dummy_embedding = np.random.rand(1536).tolist()
    
    # تخزين أصل مع Embedding
    dummy_file = "test_asset_tmp.txt"
    with open(dummy_file, "w") as f:
        f.write("test data")
        
    metadata = {
        "description": "صورة اختبار للضغط",
        "tags": ["test", "quantization"],
        "embedding": dummy_embedding
    }
    
    asset_id = memory.store_asset("agent_test", dummy_file, "image", metadata)
    print(f"تم تخزين الأصل بنجاح: {asset_id}")
    
    # التحقق من الفهرس
    with open(memory.index_path, "r") as f:
        index_data = json.load(f)
        asset_entry = next(a for a in index_data["assets"] if a["id"] == asset_id)
        
        assert asset_entry.get("quantized") is True
        assert "compressed_idx" in asset_entry["metadata"]
        assert "embedding" not in asset_entry["metadata"]
        print("✅ نجح التحقق من ضغط الفهرس وتوفير المساحة!")
        
    # تنظيف
    if os.path.exists(dummy_file):
        os.remove(dummy_file)

if __name__ == "__main__":
    try:
        test_quantization_efficiency()
        test_memory_integration()
    except Exception as e:
        print(f"❌ فشل الاختبار: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
