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
    # نستخدم temp=0 لجعل التوليد حتمياً (greedy) للمقارنة الدقيقة
    np.random.seed(42)
    start_time = time.time()
    gen_normal_ids = model.generate_ids(prompt, max_new=20, use_kv_cache=False, temp=0)
    normal_duration = time.time() - start_time
    print(f"✅ التوليد العادي: {normal_duration:.4f} ثانية")
    
    np.random.seed(42)
    start_time = time.time()
    gen_cached_ids = model.generate_ids(prompt, max_new=20, use_kv_cache=True, temp=0)
    cached_duration = time.time() - start_time
    print(f"✅ التوليد مع Cache: {cached_duration:.4f} ثانية")
    print(f"🚀 نسبة التحسن: {(normal_duration / cached_duration - 1) * 100:.1f}%")
    
    match_ids = np.array_equal(gen_normal_ids, gen_cached_ids)
    print(f"📊 مطابقة الرموز المولدة (Greedy): {'✅ متطابقة' if match_ids else '❌ غير متطابقة'}")

    # 3. اختبار التوافق مع الوسائط المتعددة (صورة + صوت + فيديو)
    print("\n📸 🎥 🔊 اختبار KV Cache مع الوسائط المتعددة الكاملة...")
    image_feats = np.random.randn(1, 512) * 0.1
    audio_feats = np.random.randn(1, 128) * 0.1
    video_feats = np.random.randn(5, 512) * 10.0 # تكبير التأثير للتحقق من الرصد إحصائياً
    
    np.random.seed(42)
    gen_mm_normal = model.generate_ids(prompt, max_new=5, 
                                       image_feats=image_feats, 
                                       audio_feats=audio_feats,
                                       video_feats=video_feats,
                                       use_kv_cache=False,
                                       temp=0)
    
    np.random.seed(42)
    gen_mm_cached = model.generate_ids(prompt, max_new=5, 
                                       image_feats=image_feats, 
                                       audio_feats=audio_feats,
                                       video_feats=video_feats,
                                       use_kv_cache=True,
                                       temp=0)
    
    mm_match = np.array_equal(gen_mm_normal, gen_mm_cached)
    print(f"📊 مطابقة نتائج الوسائط المتعددة: {'✅ متطابقة' if mm_match else '❌ غير متطابقة'}")
    
    # التحقق من أن الفيديو يؤثر فعلياً على المخرجات
    np.random.seed(42)
    gen_no_video = model.generate_ids(prompt, max_new=5, image_feats=image_feats, use_kv_cache=False, temp=0)
    video_effect = not np.array_equal(gen_mm_normal, gen_no_video)
    print(f"🎬 تأثير الفيديو على التوليد: {'✅ تم رصده' if video_effect else '❌ لم يُرصد'}")
    
    print("\n✨ انتهى الاختبار.")

if __name__ == "__main__":
    test_kv_cache_equivalence()
