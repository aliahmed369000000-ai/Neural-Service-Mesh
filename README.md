# فرع أوزان النموذج (model-weights-v3)

هذا الفرع مخصص حصريًا لتخزين أوزان ArabicTransformer v3 (876MB).
لا علاقة له بتاريخ `main` — كل حفظة جديدة تستبدل (`commit --amend` +
`push --force`) نفس الكوميت الوحيد بدل ما تضيف كوميت جديد فوقه،
عشان حجم الفرع يفضل ثابت (~876MB) مهما كان عدد مرات الحفظ.

**position الحالي**: راجع `position.json` في هذا الفرع (يُحدَّث مع كل حفظة،
لازم يطابق `ckg_train_state_v3.json` في فرع main دايمًا).

**لاستخدامها في جلسة جديدة:**
```
git clone --branch model-weights-v3 --single-branch <repo_url> weights_only
cp -r weights_only/models/transformer_ckg_v3 <working_dir>/models/
```
