"""
DataTransformer — جسر بيانات بسيط بين عقد ExecutionEngine (core/engine.py)
=============================================================================
كانت core/engine.py تستورد `from connectors.data_transformer import
DataTransformer` رغم أن الوحدة لم تكن موجودة إطلاقاً بالمستودع (وحدة
مفقودة تماماً، وليست خطأً بسيطاً بملف موجود).

هذا التنفيذ متعمَّد أن يكون **أبسط ما يمكن**: تمرير مباشر للبيانات كما هي
بين عقدة وأخرى، بدون أي تحويل أنواع (type coercion) أو حذف حقول غير
متوقَّعة. التحقق من اكتمال الحقول المطلوبة (required fields) يحصل أصلاً
داخل BaseNode.execute() عبر NodeSchema.validate() — فلا حاجة لازدواج
ذلك المنطق هنا.

لو احتجت مستقبلاً تحويل أنواع حقيقي بين عقد بمخارج/مداخل غير متطابقة
الأسماء أو الأنواع، هذا الكلاس يحتاج توسعة عمدية مع تصميم واضح لقواعد
التحويل — لم أفترض شيئاً من ذلك هنا.
"""
from __future__ import annotations

from typing import Any, Dict


class DataTransformer:
    """تمرير مباشر (passthrough) لبيانات العقدة — بدون تحويل أو حذف حقول."""

    def transform(self, data: Dict[str, Any], schema: Any = None) -> Dict[str, Any]:
        """
        يُعيد نسخة سطحية (shallow copy) من `data` كما هي.

        `schema` (NodeSchema الخاصة بالعقدة التالية) غير مستخدَمة حالياً
        في هذا التنفيذ المبسّط — التحقق من الحقول المطلوبة يحصل أصلاً
        داخل BaseNode.execute() بعد استدعاء transform().
        """
        return dict(data)
