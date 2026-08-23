import sys
import os
import numpy as np
import time

# إضافة مسار المشروع
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.arabic_transformer import ArabicTransformer
from ai.multimodal_kv_cache import MultimodalKVCache

def test_kv_cache_equivalence():
    print("🚀 بدء اختبار مطابقة KV Cache...")
    
    # إعداد نموذج مصغر للاختبار
    d_model = 128
    n_layers = 2
    n_heads = 4
    vocab_size = 1000
    
    model = ArabicTransformer(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=vocab_size,
        use_sparse_attn=False # نستخدم MHA الكثيف لاختبار الـ cache
    )
    
    prompt = "السلام عليكم ورحمة الله"
    ids = model.tokenizer.encode(prompt)
    
    # 1. التحقق من تطابق الـ logits في خطوة واحدة
    print("\n🔍 اختبار تطابق الـ Logits...")
    ids_all = model.tokenizer.encode(prompt)
    
    # تمريرة كاملة
    logits_full, _, _, _ = model._forward(ids_all)
    
    # تمريرة مجزأة مع Cache
    cache = MultimodalKVCache()
    # Prefill (كل الرموز عدا الأخير)
    model._forward(ids_all[:-1], past_kv=cache, use_cache=True)
    # Decode (الرمز الأخير فقط)
    logits_cached, _, _, _ = model._forward(ids_all[-1:], past_kv=cache, use_cache=True)
    
    # مقارنة آخر logits
    diff = np.abs(logits_full[-1] - logits_cached[-1]).max()
    print(f"📊 أقصى فرق في الـ Logits: {diff:.8e}")
    match = diff < 1e-10
    print(f"📊 مطابقة الـ Logits: {'✅ متطابقة' if match else '❌ غير متطابقة'}")

    # 2. قياس السرعة (توليد 20 رمز)
    print("\n⚡ قياس السرعة (توليد 20 رمز)...")
    np.random.seed(42)
    start_time = time.time()
    model.generate_ids(prompt, max_new=20, use_kv_cache=False)
    normal_duration = time.time() - start_time
    print(f"✅ التوليد العادي: {normal_duration:.4f} ثانية")
    
    np.random.seed(42)
    start_time = time.time()
    model.generate_ids(prompt, max_new=20, use_kv_cache=True)
    cached_duration = time.time() - start_time
    print(f"✅ التوليد مع Cache: {cached_duration:.4f} ثانية")
    print(f"🚀 نسبة التحسن: {(normal_duration / cached_duration - 1) * 100:.1f}%")
    
    # 3. اختبار التوافق مع الوسائط المتعددة
    print("\n📸 اختبار KV Cache مع الوسائط المتعددة...")
    image_feats = np.random.randn(1, 512)
    
    np.random.seed(42)
    gen_mm_normal = model.generate_ids(prompt, max_new=5, image_feats=image_feats, use_kv_cache=False)
    
    np.random.seed(42)
    gen_mm_cached = model.generate_ids(prompt, max_new=5, image_feats=image_feats, use_kv_cache=True)
    
    mm_match = np.array_equal(gen_mm_normal, gen_mm_cached)
    print(f"📊 مطابقة نتائج الوسائط: {'✅ متطابقة' if mm_match else '❌ غير متطابقة'}")
    
    print("\n✨ انتهى الاختبار.")

if __name__ == "__main__":
    test_kv_cache_equivalence()
