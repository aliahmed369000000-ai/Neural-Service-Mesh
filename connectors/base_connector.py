"""
BaseConnector — واجهة مجردة لكل "جسور البيانات" بين عقد core/engine.py
=======================================================================
كانت `connectors.base_connector` مذكورة في خريطة المراحل
(ai/validator.py: _MODULE_PHASE_MAP) كوحدة من Phase 1، لكن الملف لم
يكن موجوداً إطلاقاً بالمستودع — ولا أي كود يستورده (بخلاف حالة
DataTransformer سابقاً، التي كانت مستوردة فعلياً من core/engine.py قبل
أن تُنشأ).

الهدف من هذا الملف: تثبيت الواجهة الفعلية التي يعتمد عليها
core/engine.py ضمنياً (duck typing عبر `transformer.transform(data,
schema)`) بشكل صريح، بحيث أي connector مستقبلي (غير DataTransformer)
يلتزم بنفس التوقيع بدل الاعتماد على تطابق ضمني غير موثّق.

لا منطق تحويل فعلي هنا — فقط العقد (contract). DataTransformer هو
التنفيذ الوحيد حالياً ويرث من هذا الكلاس.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseConnector(ABC):
    """
    عقد أساسي لكل connector يمرَّر لـ ExecutionEngine (core/engine.py).

    أي subclass يجب أن يوفّر `transform(data, schema)` ويُعيد dict —
    نفس التوقيع الذي يتوقعه core/engine.py من `self._transformer`.
    """

    @abstractmethod
    def transform(self, data: Dict[str, Any], schema: Any = None) -> Dict[str, Any]:
        """
        يحوّل بيانات مخرجات عقدة إلى الشكل المتوقَّع كمدخلات للعقدة التالية.

        Args:
            data: بيانات مخرجات العقدة الحالية.
            schema: NodeSchema الخاصة بالعقدة التالية (اختياري، تُستخدم
                    من قِبل التنفيذات التي تحتاج معرفة الحقول المتوقَّعة).

        Returns:
            dict جاهز لتمريره كمدخلات للعقدة التالية.
        """
        ...
