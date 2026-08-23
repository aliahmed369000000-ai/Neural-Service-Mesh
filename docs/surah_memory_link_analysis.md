# تحليل ربط شبكة Surah بالذاكرة الموحدة (ANN)

## المواقع الرئيسية للحقن التقني:
1. **الملف المستهدف**: `ai/arabic_transformer.py`
2. **الطبقة المستهدفة**: `CoreMatrixLayer` (السطور 370-610).
3. **الآلية الحالية**: تستخدم `CoreMatrixLayer` انتباهاً متقاطعاً متفرقاً (`_sparse_cross_attention`) لدمج ميزات الصورة والصوت والفيديو.
4. **خطة الربط**:
    * إضافة `UnifiedMemoryManager` كمدخل اختياري لـ `CoreMatrixLayer`.
    * في دالة `forward` لـ `CoreMatrixLayer`:
        * إذا تم تفعيل البحث في الذاكرة، يتم تحويل المدخل `X` (النص) إلى استعلام متجهي.
        * استدعاء `unified_memory.semantic_search(query_vec)` لجلب الخبرات ذات الصلة.
        * تحويل الخبرات المسترجعة إلى ميزات وسائط إضافية (Memory Features).
        * دمج هذه الميزات عبر `_sparse_cross_attention` كأنها وسائط حقيقية.

## الملاحظات التقنية:
* `CoreMatrixLayer` تعمل بـ `d_model=4096`.
* `UnifiedMemoryManager` يعمل مع `ANNEngine` الذي يستخدم تضمينات (Embeddings) يجب أن تتوافق أبعادها أو يتم إسقاطها (Projection) لتناسب `d_model`.
* الذاكرة الموحدة تعيد `experience_data` و `distance`. يجب تحويل محتوى الخبرة (نص/متجه) إلى تمثيل يمكن لـ Surah فهمه.
