import numpy as np
import os
import sys
from pathlib import Path

# إضافة مسار المشروع لـ sys.path
sys.path.append(str(Path(__file__).parent.parent))

from ai.arabic_transformer import ArabicTransformer
from ai.unified_memory import UnifiedMemoryManager

def test_surah_memory_retrieval():
    print("🚀 بدء اختبار ربط Surah بالذاكرة الموحدة...")
    
    # 1. إعداد الذاكرة الموحدة
    memory_dir = "/tmp/nsm_test_memory_link"
    if not os.path.exists(memory_dir):
        os.makedirs(memory_dir)
    
    memory = UnifiedMemoryManager(memory_dir, dimension=1536)
    
    # 2. إضافة خبرة للذاكرة
    experience = {
        "id": "exp_001",
        "content": "قاعدة بيانات NSM تدعم البحث الدلالي السريع.",
        "type": "technical_note"
    }
    # متجه عشوائي للتجربة
    dummy_embedding = np.random.randn(1536).tolist()
    memory.store_experience(experience, embedding=dummy_embedding)
    print("✅ تم تخزين الخبرة في الذاكرة الموحدة.")
    
    # 3. إعداد نموذج Surah (ArabicTransformer)
    # تقليل d_model و n_layers لتجنب OOM في بيئة الاختبار
    model = ArabicTransformer(n_layers=1, d_model=256, max_seq=16) 
    print("✅ تم إعداد نموذج Surah.")
    
    # 4. إجراء استعلام يحفز استرجاع الذاكرة
    query_text = "كيف تعمل قاعدة بيانات NSM؟"
    ids = model.tokenizer.encode(query_text)
    if len(ids) > 10: ids = ids[:10]
    
    # محاكاة توليد استعلام متجهي من النص (في الواقع يتم عبر نموذج embedding)
    query_vec = np.random.randn(1536).tolist()
    
    # البحث في الذاكرة
    results = memory.semantic_search(query_vec, top_k=1)
    memory_feats = None
    if results:
        # تحويل التضمينات المسترجعة إلى مصفوفة ميزات لـ CoreMatrix
        embeddings = [res["embedding"] for res in results if "embedding" in res]
        if embeddings:
            memory_feats = np.array(embeddings)
            print(f"✅ تم استرجاع {len(memory_feats)} خبرة من الذاكرة.")
    
    # 5. التمريرة الأمامية في Surah مع ميزات الذاكرة
    logits, hidden_states, risk, intent = model._forward(
        ids, 
        memory_feats=memory_feats
    )
    
    print(f"✅ نجحت التمريرة الأمامية مع الذاكرة. شكل المخرجات: {logits.shape}")
    assert logits.shape == (len(ids), 8192) # (seq_len, vocab_size)
    print("🎉 انتهى الاختبار بنجاح!")

if __name__ == "__main__":
    try:
        test_surah_memory_retrieval()
    except Exception as e:
        print(f"❌ فشل الاختبار: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
