import sys
import os
import json

# إضافة المسار الجذري للمشروع
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.multimodal_sync import MultimodalSyncManager

def test_quantization_efficiency():
    print("🚀 اختبار كفاءة تكميم المتجهات (Vector Quantization)...")
    
    sync_manager = MultimodalSyncManager()
    
    # 1. إنشاء متجه أصلي (Float32)
    original_vector = [0.1234, 0.5678, 0.9012, 0.3456, 0.7890, 0.2121, 0.6543, 0.0987]
    
    # 2. التكميم (Int8)
    quantized_vector = sync_manager._quantize_vector(original_vector)
    
    # 3. إلغاء التكميم للمقارنة
    dequantized_vector = sync_manager._dequantize_vector(quantized_vector)
    
    # 4. حساب نسبة تقليل الحجم
    # في بايثون، التخزين الفعلي يختلف، لكننا نقيس الحجم المنطقي في JSON
    float_json_size = len(json.dumps(original_vector))
    int_json_size = len(json.dumps(quantized_vector))
    
    print(f"📊 حجم المتجه الأصلي (JSON): {float_json_size} bytes")
    print(f"📉 حجم المتجه المكمم (JSON): {int_json_size} bytes")
    print(f"🚀 نسبة التوفير في مساحة JSON: {((float_json_size - int_json_size) / float_json_size) * 100:.2f}%")
    
    # 5. قياس الدقة (MSE)
    mse = sum((a - b)**2 for a, b in zip(original_vector, dequantized_vector)) / len(original_vector)
    print(f"🎯 متوسط الخطأ التربيعي (MSE): {mse:.6f}")
    
    # التحقق
    if mse < 0.0001:
        print("✅ نجاح: التكميم فعال مع الحفاظ على دقة عالية جداً.")
    else:
        print("⚠️ تحذير: فقدان الدقة قد يكون ملحوظاً.")

if __name__ == "__main__":
    test_quantization_efficiency()
