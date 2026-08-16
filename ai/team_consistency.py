"""
فاحص اتساق الفريق (Team Consistency Checker) — طبقة مراجعة نهائية قبل توليف
إجابة الفريق في director الموحّد.

عندما يجيب وكيلان أو أكثر عن نفس السؤال، قد يتعارض جوابهما (أرقام متضاربة،
استنتاجات متناقضة، توصيات متعارضة). التوليف الأعمى يدمج التناقض في جواب
واحد مضلّل. هذه الوحدة:

1) تحسب مؤشر اتساق نصي بين ردود الوكلاء (تشابه جاكارد على الكلمات بعد
   تنظيف النص) وتكشف التعارضات الصريحة (أرقام/اتجاهات متضاربة في جمل قصيرة
   متقابلة).
2) إن كان الاتساق منخفضًا أو وُجدت تعارضات صريحة، يُدرَج تحذير موجّه
   داخل prompt التوليف حتى يحسم المدير التعارض بدل تجاهله، ويُضاف تنويه
   للمستخدم في نهاية الجواب المولّف (عبر بث حدث + تنبيه).
3) كل عملياتها محلية ولا تعتمد على أي API.

التكامل: تستدعيها agent_categories.py داخل _synthesize قبل بناء synth_prompt.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# حد أدنى للاتساق المقبول — تحته يُعتبر الفريق متناقضًا
_CONSISTENCY_THRESHOLD = 0.35

# عبارات اتجاهية متضاربة (نستخرج الجملة القصيرة المحيطة بها ونقارنها)
_OPPOSITE_PAIRS = (
    ("ينصح", "لا ينصح"), ("موصى", "غير موصى"), ("مفيد", "غير مفيد"),
    ("يزيد", "يقلل"), ("نعم", "لا"), ("إيجابي", "سلبي"),
    ("أفضل", "أسوأ"), ("يفضل", "يتجنب"), ("ارفع", "اخفض"),
    ("yes", "no"), ("good", "bad"), ("increase", "decrease"),
)

# أرقام (بأرقام عربية وغربية) لفحص تعارضها المباشر
_NUM_RE = re.compile(r"\d+([.,]\d+)?")


def _clean_words(text: str) -> set:
    words: List[str] = []
    for token in re.split(r"[\s.,،;:!?(){}\[\]\"'\-]+", (text or "").lower()):
        t = token.strip()
        if len(t) >= 3 and not re.match(r"^[\d.,]+$", t):
            words.append(t)
    return set(words)


def jaccard_words(a: str, b: str) -> float:
    wa, wb = _clean_words(a), _clean_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _short_sentences(text: str) -> List[str]:
    """يقسم النص إلى جمل قصيرة (مقسّمات:句号 عربي/لاتيني أو سطر جديد)."""
    parts: List[str] = []
    for segment in re.split(r"[.\n؟!؟]+", text or ""):
        s = " ".join(segment.split())
        if 8 <= len(s) <= 90:
            parts.append(s)
    return parts


def _find_opposite_clash(sentences: List[str], other_sentences: List[str]) -> Optional[str]:
    """يكشف جملتين قصيرتين متعارضتين صراحةً بين النصين (إن وُجدتا)."""
    for s in sentences:
        for s2 in other_sentences:
            if s == s2:
                continue
            for pos, neg in _OPPOSITE_PAIRS:
                p, n = pos.lower(), neg.lower()
                if (p in s.lower()) != (p in s2.lower()) and (n in s.lower()) != (n in s2.lower()):
                    # كل جملة تُظهر اتجاهًا عكسيًا للجملة الأخرى
                    return f"«{s[:70]}» ↔ «{s2[:70]}»"
    return None


def check_team_consistency(
    replies: Dict[str, str],
) -> Dict[str, Any]:
    """
    يحسب مؤشر اتساق الفريق ويكتشف التعارضات الصريحة.

    الرد: قاموس يحتوي:
    - ok: bool (هل الفريق متسق)
    - score: float متوسط أدنى تشابه ثنائي (0-1)
    - clashing_pairs: list [(agent_a, agent_b, نص التعارض)]
    - warning: نص تحذير موجّه للتوليف (فارغ إن كان كل شيء متسقًا)
    """
    keys = [k for k, v in replies.items() if v and v.strip()]
    if len(keys) < 2:
        return {"ok": True, "score": 1.0, "clashing_pairs": [], "warning": ""}

    pairs: List[Tuple[str, str, float, Optional[str]]] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ka, kb = keys[i], keys[j]
            score = jaccard_words(replies[ka], replies[kb])
            clash = _find_opposite_clash(
                _short_sentences(replies[ka]), _short_sentences(replies[kb]))
            pairs.append((ka, kb, score, clash))

    min_score = min(p[2] for p in pairs)
    clashing_pairs = [(ka, kb, cl) for ka, kb, sc, cl in pairs if cl]

    ok = min_score >= _CONSISTENCY_THRESHOLD and not clashing_pairs
    warning_parts: List[str] = []
    if clashing_pairs:
        items = "\n".join(f"- {ka} ↔ {kb}: {cl}" for ka, kb, cl in clashing_pairs)
        warning_parts.append(
            f"⚠️ تعارضات صريحة بين ردود الوكلاء (حسمها أنت كمسؤول نهائي — "
            f"لا تدمج الاتجاهين المتناقضين):\n{items}")
    if not ok:
        warning_parts.append(
            f"⚠️ مؤشر اتساق الفريق منخفض ({min_score:.2f} دون حد {_CONSISTENCY_THRESHOLD}) — "
            "الردود تتحدث عن مواضيع مختلفة أو معلومات متضاربة؛ اختر المصدر الأوثق "
            "وقدّم جوابًا واحدًا حاسمًا، أو وضّح وجهات النظر بصراحة للمستخدم.")

    return {
        "ok": ok,
        "score": min_score,
        "clashing_pairs": clashing_pairs,
        "warning": "\n\n".join(warning_parts),
    }
