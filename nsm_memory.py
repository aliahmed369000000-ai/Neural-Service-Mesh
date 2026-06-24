"""
NSM Memory — ذاكرة المحادثة الذكية
=====================================
تُضاف لـ NSMChat لتحقيق:
  1) فهم الضمائر (ركعاتها، كيف يعمل، ما أهميته)
  2) تراكم السياق عبر رسائل متعددة
  3) كشف تغيير الموضوع
  4) تذكر آخر 5 رسائل كنافذة سياق
"""
from __future__ import annotations
import re
from typing import List, Tuple, Optional
from collections import deque

# ══════════════════════════════════════════════════════════════════
# كلمات تدل على السياق (ضمائر وإشارات)
# ══════════════════════════════════════════════════════════════════
_CONTEXT_SIGNALS = [
    # ضمائر متصلة
    "ها", "هم", "هن", "ه", "ك", "كم",
    # أدوات ربط تشير للسياق
    "وكم", "وكيف", "ومتى", "وأين", "ولماذا", "وما", "وهل",
    "وكذلك", "وأيضاً", "وأيضا", "وهو", "وهي", "وهما",
    # روابط اللغة العربية
    "أيضاً", "أيضا", "كذلك", "بالإضافة", "فضلاً",
    "علاوة", "ومن", "وعن", "وفي", "وعلى",
    # إنجليزي
    "also", "too", "additionally", "furthermore",
    "and how", "and why", "and when",
]

_PRONOUN_PATTERNS = [
    r'(ركعات|صلوات|أوقات|فوائد|أنواع|أحكام|أسباب|أهمية)(ها|هم|ه)',
    r'^(وكم|وكيف|ومتى|وأين|ولماذا|وما|وهل|وهو|وهي)\b',
    r'^(و[أا]يض[اً]|وكذلك|وبالإضافة)\b',
    r'(يعمل|تعمل|تُؤدى|يُستخدم|يُحسب)(ها|ه|هم)?\??$',
]

# ══════════════════════════════════════════════════════════════════
class ConversationMemory:
    """
    نافذة سياق ذكية لـ NSMChat.
    
    تحتفظ بـ:
      - آخر 5 أزواج (سؤال، جواب، موضوع)
      - الموضوع الحالي
      - الكيانات المذكورة (أسماء، مفاهيم)
    """

    WINDOW = 5  # عدد رسائل السياق

    def __init__(self):
        self._history: deque = deque(maxlen=self.WINDOW)
        self.current_topic: str = ""
        self.current_topic_text: str = ""  # نص الموضوع للدمج
        self._entities: List[str] = []     # كلمات مفتاحية مذكورة

    # ── إضافة رسالة ──────────────────────────────────────────────
    def add(self, user_msg: str, bot_reply: str, topic: str):
        self._history.append({
            "user": user_msg,
            "bot": bot_reply,
            "topic": topic,
        })
        if topic and topic != self.current_topic:
            self.current_topic = topic
            self.current_topic_text = user_msg
        # استخراج كلمات مفيدة للسياق
        self._update_entities(user_msg)

    def _update_entities(self, text: str):
        """استخرج الكلمات المهمة (أكثر من 3 أحرف وليست أدوات)"""
        stopwords = {"ما","هو","هي","ما","من","في","على","عن","هل",
                     "كيف","متى","أين","لماذا","ماذا","كم","أي",
                     "الى","إلى","عند","مع","بعد","قبل","أو","لكن"}
        words = re.findall(r'[\u0600-\u06ff]{3,}|[a-zA-Z]{4,}', text)
        new_entities = [w for w in words if w not in stopwords]
        # أضف فقط الكلمات الجديدة (آخر 10)
        self._entities = (self._entities + new_entities)[-10:]

    # ── كشف هل الرسالة تحتاج سياق ──────────────────────────────
    def needs_context(self, user_msg: str) -> bool:
        """هل الرسالة تشير لسياق سابق؟"""
        text = user_msg.strip()

        # قصيرة جداً (أقل من 15 حرف) — غالباً تحتاج سياق
        if len(text) < 15:
            return True

        # تبدأ بـ 'و' (ربط عربي)
        if text.startswith('و') and len(text) > 2:
            return True

        # تحتوي إشارات السياق
        tl = text.lower()
        for sig in _CONTEXT_SIGNALS:
            if sig in tl:
                return True

        # أنماط الضمائر
        for pat in _PRONOUN_PATTERNS:
            if re.search(pat, text):
                return True

        return False

    # ── بناء الاستعلام المُغنى ──────────────────────────────────
    def enrich_query(self, user_msg: str) -> str:
        """
        يدمج الرسالة الحالية مع السياق السابق.
        مثال: 'وكم ركعاتها؟' + سياق 'الصلاة' → 'الصلاة وكم ركعاتها؟'
        """
        if not self.needs_context(user_msg) or not self._history:
            return user_msg

        # أخذ آخر موضوع وآخر كيانات
        ctx_parts = []

        # الموضوع الحالي
        if self.current_topic_text:
            ctx_parts.append(self.current_topic_text[:30])

        # أهم الكيانات (آخر 3)
        if self._entities:
            ctx_parts.extend(self._entities[-3:])

        # آخر سؤال (للسياق المباشر)
        if self._history:
            last_user = self._history[-1]["user"]
            ctx_parts.append(last_user[:40])

        # دمج السياق مع الرسالة الحالية
        context_str = " ".join(dict.fromkeys(ctx_parts))  # بدون تكرار
        enriched = f"{context_str} {user_msg}"
        return enriched

    # ── ملخص السياق للعرض ───────────────────────────────────────
    def context_summary(self) -> str:
        if not self._history:
            return "لا يوجد سياق سابق"
        last = self._history[-1]
        return f"الموضوع: {self.current_topic} | آخر سؤال: {last['user'][:40]}"

    # ── مسح الذاكرة ─────────────────────────────────────────────
    def clear(self):
        self._history.clear()
        self.current_topic = ""
        self.current_topic_text = ""
        self._entities = []

    # ── تاريخ المحادثة كاملاً ───────────────────────────────────
    def get_history(self) -> List[Tuple[str, str]]:
        return [(h["user"], h["bot"]) for h in self._history]

    def __len__(self):
        return len(self._history)

    def __bool__(self):
        return True  # الكائن موجود دائماً حتى لو فارغ
