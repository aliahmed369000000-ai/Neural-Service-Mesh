"""
Learning Orchestrator — نقطة توحيد واحدة لاستدعاء أنظمة التعلّم
===========================================================================
كان في المشروع 3 أنظمة "تعلّم" منفصلة تماماً، بلا أي تنسيق بينها، وبقواعد
بيانات مختلفة:

  1. EpisodicMemoryEngine (ai/episodic_memory.py → memory/episodic.db)
     المتصل الوحيد فعلياً بمسار الدردشة الحي قبل هذا الملف (عبر
     _record_chat_episode في streamlit_app.py).

  2. ExperienceTrainer (ai/experience_trainer.py → memory/experience.db
     عبر EpisodeStore) يدرّب NeuralCore من حلقات حقيقية، لكن هذه الحلقات
     مصمَّمة لتأتي حصراً من ReasoningPipeline.answer() (تمرير فعلي عبر
     الشبكة العصبية ينتج context_vector + decision_weights حقيقيين).
     ReasoningPipeline لا يُستدعى في أي مكان بمسار الدردشة الحي الحالي
     (الدردشة تستخدم بوابات LLM مباشرة: free_router / OpenRouter / NSM
     Agent) — لذلك EpisodeStore يبقى شبه فارغ دائماً، وزر "🎓 ابدأ
     التدريب الآن" في تبويب النواة العصبية يطبع تحذير "0 حلقة" غالباً.

  3. ConversationLearner (ai/continual_learner.py → memory/nsm_learning.db)
     نظام Q&A كامل (learn/recall/feedback) لكنه كان **يتيماً بالكامل** —
     غير مُستدعى من أي مكان في كل المشروع.

  4. KnowledgeTrainer (ai/knowledge_trainer.py) — أيضاً يتيم، مستدعيه
     الوحيد knowledge/quran_continuous_trainer.py (خيط خلفي كل 30 دقيقة)
     غير مستورَد إطلاقاً في streamlit_app.py.

قرار هذا الملف: يوحّد **نقطة الاستدعاء** لِـ #3 (ConversationLearner) داخل
مسار الدردشة الحي، لأنه الوحيد الآمن للوصل فوراً بدون اختلاق بيانات وهمية
(Q&A حقيقي = بالضبط ما صُمِّم له). #2 و#4 تُركا خارج هذا التوحيد عمداً:
وصلهما الحقيقي يتطلب إما استدعاء ReasoningPipeline ضمن كل رد حي (كلفة
زمن استجابة إضافية) أو تشغيل خيط خلفي دائم — قرارات معمارية/تشغيلية أكبر
تستحق نقاشاً منفصلاً قبل التفعيل التلقائي، لا اختلاق بيانات لتعبئتها.

الاستخدام:
    from ai.learning_orchestrator import get_orchestrator
    orch = get_orchestrator()
    orch.record_turn(query, response, domain="عام")   # بعد كل رد حي
    cached = orch.recall(query)                         # اختياري لاحقاً
    orch.feedback(query, is_positive=True)               # عند 👍/👎
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_orchestrator_singleton: Optional["LearningOrchestrator"] = None


class LearningOrchestrator:
    """نقطة استدعاء واحدة لكل أنظمة التعلّم المتصلة فعلياً بالمسار الحي.

    لا يرفع استثناءً أبداً من record_turn()/recall()/feedback() — فشل أي
    نظام فرعي لا يجوز أن يكسر تجربة المحادثة (نفس مبدأ _record_chat_episode
    في streamlit_app.py).
    """

    def __init__(self) -> None:
        self._conv_learner = None
        self._init_conversation_learner()

    def _init_conversation_learner(self) -> None:
        try:
            from ai.continual_learner import ConversationLearner
            self._conv_learner = ConversationLearner()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"LearningOrchestrator: ConversationLearner غير متاح: {e}")
            self._conv_learner = None

    def record_turn(
        self,
        query: str,
        response: str,
        domain: str = "عام",
        source: str = "conversation",
    ) -> dict:
        """يسجّل تبادل محادثة حقيقي واحد عبر كل أنظمة التعلّم المتصلة.

        يعيد ملخّصاً {"conversation_learner": bool, "quality": float} عن
        نجاح كل نظام فرعي (best-effort — لا يفشل الاستدعاء الكلي).
        """
        result: dict = {"conversation_learner": False}
        if self._conv_learner is not None and (query or "").strip() and (response or "").strip():
            try:
                quality = self._conv_learner.learn(query, response, domain=domain, source=source)
                result["conversation_learner"] = True
                result["quality"] = quality
            except Exception as e:  # noqa: BLE001
                logger.debug(f"LearningOrchestrator.record_turn: {e}")
        return result

    def recall(self, query: str, min_quality: float = 0.6) -> Optional[dict]:
        """يبحث فيما تعلّمه ConversationLearner سابقاً — جاهز للاستخدام
        كاختياري لاحقاً كطبقة كاش قبل استدعاء LLM (غير مفعّل تلقائياً بعد)."""
        if self._conv_learner is None or not (query or "").strip():
            return None
        try:
            return self._conv_learner.recall(query, min_quality=min_quality)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"LearningOrchestrator.recall: {e}")
            return None

    def feedback(self, query: str, is_positive: bool) -> bool:
        """يطبّق تقييم 👍/👎 على آخر إجابة لهذا السؤال."""
        if self._conv_learner is None or not (query or "").strip():
            return False
        try:
            return bool(self._conv_learner.feedback(query, is_positive))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"LearningOrchestrator.feedback: {e}")
            return False


def get_orchestrator() -> LearningOrchestrator:
    """singleton واحد لعملية Streamlit كاملة (وليس لكل جلسة)."""
    global _orchestrator_singleton
    if _orchestrator_singleton is None:
        _orchestrator_singleton = LearningOrchestrator()
    return _orchestrator_singleton
