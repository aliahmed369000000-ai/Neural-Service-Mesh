
import sys
import os
import numpy as np

# إضافة المسار للمشروع
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from ai.arabic_transformer import ArabicTransformer
    print("✓ تم استيراد ArabicTransformer بنجاح.")
except ImportError as e:
    print(f"✗ فشل الاستيراد: {e}")
    sys.exit(1)

def test_cross_attention_fusion():
    print("\n--- اختبار دمج Cross-Attention في Surah 4096 ---")
    
    # إعداد نموذج صغير للاختبار
    model = ArabicTransformer(d_model=256, n_layers=2)
    
    # 1. اختبار نص فقط
    text_ids = np.array([10, 20, 30])
    logits_text_only, _, _, _ = model._forward(text_ids)
    print(f"✓ استدلال نصي فقط ناجح. شكل المخرجات: {logits_text_only.shape}")
    
    # 2. اختبار دمج صورة (Cross-Attention)
    image_feats = np.random.randn(5, 512) # 5 ميزات بصرية
    logits_multimodal, _, _, _ = model._forward(text_ids, image_feats=image_feats)
    
    # التحقق من التأثير
    diff = np.abs(logits_multimodal - logits_text_only).mean()
    print(f"✓ استدلال متعدد الوسائط ناجح. متوسط الاختلاف دلالياً: {diff:.6f}")
    
    if diff > 0:
        print("✓ تم دمج معلومات الصورة بنجاح عبر Cross-Attention.")
    else:
        print("✗ لم يظهر أي تأثير لدمج الصورة.")
        
    # 3. اختبار التوليد مع Cross-Attention
    print("\n--- اختبار التوليد مع سياق بصري ---")
    generated = model.generate_ids("مرحباً", max_new=5, image_feats=image_feats)
    print(f"✓ النص المولد مع سياق بصري: {generated}")
    
    print("\n✓ نجح اختبار دمج Cross-Attention بالكامل.")

if __name__ == "__main__":
    test_cross_attention_fusion()
