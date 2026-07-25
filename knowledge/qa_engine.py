"""
qa_engine.py
============
محرك الأسئلة والأجوبة القرآني — Quran Knowledge Q&A Engine
يستخدم فقط:
  - 6236 آية قرآنية
  - 173 مفهوم في الـ CKG
  - 2149 علاقة دلالية
  - 633 جذر عربي مفهرس

لا يضيف طبقات عصبية جديدة ولا مصادر خارجية — يعمل فوق البنية الحالية فقط.
"""

from __future__ import annotations

import re
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# طبقة تعزيز عصبي اختيارية (ArabicTransformer 40M) — إضافية فوق قواعد qa_engine
# ═══════════════════════════════════════════════════════════════════════════
# مبدأ التصميم: qa_engine.py يبقى يعمل بالكامل بمفرده حتى لو فشلت هذه الطبقة
# تماماً (استيراد، تحميل أوزان، أو أي استثناء أثناء التنفيذ) — أي خطأ هنا
# يُصمَّت ويُرجَع الترتيب الأصلي دون تعديل. لا يوجد أي مسار يعتمد على نجاح
# هذه الطبقة.

_NEURAL_BOOSTER = None       # ArabicTransformer المحمَّل، أو False لو فشل نهائياً
_NEURAL_BOOST_TRIED = False  # نحاول التحميل مرة واحدة فقط (لا نعيد المحاولة كل سؤال)


def _get_neural_booster():
    """يُحمِّل ArabicTransformer مرة واحدة (lazy singleton). يُرجع None عند أي فشل."""
    global _NEURAL_BOOSTER, _NEURAL_BOOST_TRIED
    if _NEURAL_BOOST_TRIED:
        return _NEURAL_BOOSTER if _NEURAL_BOOSTER is not False else None
    _NEURAL_BOOST_TRIED = True
    try:
        from ai.arabic_transformer import ArabicTransformer
        t = ArabicTransformer()
        t.load("models/transformer_ckg_v1")
        _NEURAL_BOOSTER = t
        logger.info("[qa_engine] طبقة التعزيز العصبي (ArabicTransformer) محمَّلة بنجاح")
    except Exception as e:
        _NEURAL_BOOSTER = False
        logger.warning(f"[qa_engine] تعذّر تحميل طبقة التعزيز العصبي — "
                        f"سيعمل qa_engine بقواعده الأصلية فقط: {e}")
    return _NEURAL_BOOSTER if _NEURAL_BOOSTER is not False else None


# ═══════════════════════════════════════════════════════════════════════════
# LoRA: تدريب خفيف إضافي من ملاحظات المستخدمين (لا يمسّ الأوزان الأساسية)
# ═══════════════════════════════════════════════════════════════════════════
_LORA_ADAPTER = None
_LORA_TRIED = False
_LORA_FEEDBACK_COUNT = 0
LORA_SAVE_DIR = "models/lora_feedback_v1"
LORA_SAVE_EVERY = 10  # حفظ الـadapter كل 10 ملاحظات إيجابية


def _get_lora_adapter():
    """
    يُغلّف ArabicTransformer بـ LoRA adapters (rank=8) لتدريب خفيف جداً
    (~548K باراميتر بدل 27M) من ملاحظات المستخدمين، دون أي تعديل على
    الأوزان الأساسية المدرَّبة. يُرجع None لو الشبكة الأساسية غير متاحة.
    """
    global _LORA_ADAPTER, _LORA_TRIED
    if _LORA_TRIED:
        return _LORA_ADAPTER if _LORA_ADAPTER is not False else None
    _LORA_TRIED = True
    booster = _get_neural_booster()
    if booster is None:
        _LORA_ADAPTER = False
        return None
    try:
        from ai.lora_adapter import LoRATransformerAdapter
        import os
        lora = LoRATransformerAdapter(booster, rank=8, alpha=16.0)
        if os.path.exists(f"{LORA_SAVE_DIR}_meta.json"):
            lora.load(LORA_SAVE_DIR)
            logger.info(f"[qa_engine] LoRA adapter محمَّل من {LORA_SAVE_DIR}")
        _LORA_ADAPTER = lora
    except Exception as e:
        _LORA_ADAPTER = False
        logger.warning(f"[qa_engine] تعذّر تهيئة LoRA adapter: {e}")
    return _LORA_ADAPTER if _LORA_ADAPTER is not False else None


def record_positive_feedback(question: str, answer_summary: str = "", lr: float = 5e-3) -> bool:
    """
    يُسجِّل ملاحظة إيجابية من المستخدم على إجابة معيّنة: خطوة تدريب LoRA
    واحدة صغيرة تُعزِّز نمط السؤال (+الإجابة إن وُجدت) داخل الشبكة، دون
    أي تعديل على الأوزان الأساسية (~27M) — فقط adapters صغيرة (~548K).

    يُستدعى مستقبلاً من واجهة streamlit_app.py عند ضغط المستخدم على 👍.
    يُرجع True لو نجحت خطوة التدريب، False عند أي فشل (fallback آمن —
    لا يكسر أي شيء، فقط لا يحدث تحسين).
    """
    global _LORA_FEEDBACK_COUNT
    lora = _get_lora_adapter()
    if lora is None:
        return False
    try:
        text = (question + " " + answer_summary).strip() if answer_summary else question
        ids, targets = lora.prepare_lm_sample(text)
        if ids is None:
            return False
        lora.train_step(ids, targets, lr=lr)
        _LORA_FEEDBACK_COUNT += 1
        if _LORA_FEEDBACK_COUNT % LORA_SAVE_EVERY == 0:
            lora.save(LORA_SAVE_DIR)
            logger.info(f"[qa_engine] LoRA adapter محفوظ بعد {_LORA_FEEDBACK_COUNT} ملاحظة")
        return True
    except Exception as e:
        logger.warning(f"[qa_engine] فشل تسجيل ملاحظة LoRA: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# التوليد الحر الاختياري (Yemeni LLM) — طبقة إضافية فوق qa_engine الرمزي
# ═══════════════════════════════════════════════════════════════════════════
# نفس مبدأ _get_neural_booster: qa_engine يعمل بالكامل حتى لو فشلت هذه الطبقة
# تماماً (torch غير مثبّت، لا يوجد checkpoint، أي استثناء أثناء التوليد).
# لا يوجد أي مسار افتراضي يعتمد على نجاح هذه الطبقة — generation_mode=False
# (الافتراضي) لا يلمسها إطلاقاً.

_YEMENI_DECODER = None    # YemeniDecoder المحمَّل، أو False لو فشل نهائياً
_YEMENI_TOKENIZER = None  # YemeniTokenizer المحمَّل، أو False لو فشل نهائياً
_YEMENI_TRIED = False     # نحاول التحميل مرة واحدة فقط


def _get_yemeni_generator():
    """
    يُحمِّل YemeniDecoder + YemeniTokenizer مرة واحدة (lazy singleton).
    يُرجع (decoder, tokenizer) أو (None, None) عند أي فشل (torch غير متاح،
    لا يوجد checkpoint، إلخ). لا يرمي أي استثناء للخارج أبداً.
    """
    global _YEMENI_DECODER, _YEMENI_TOKENIZER, _YEMENI_TRIED
    if _YEMENI_TRIED:
        ok = _YEMENI_DECODER is not False and _YEMENI_TOKENIZER is not False
        return (_YEMENI_DECODER, _YEMENI_TOKENIZER) if ok else (None, None)
    _YEMENI_TRIED = True
    try:
        from ai.arabic_transformer import get_yemeni_decoder
        from ai.yemeni_tokenizer import get_yemeni_tokenizer
        _YEMENI_DECODER = get_yemeni_decoder()
        _YEMENI_TOKENIZER = get_yemeni_tokenizer()
        logger.info("[qa_engine] YemeniDecoder + YemeniTokenizer محمَّلان بنجاح "
                     "(التوليد الحر متاح — قد يكون بأوزان غير مدرَّبة إن لم "
                     "يوجد checkpoint في models/yemeni_decoder)")
    except Exception as e:
        _YEMENI_DECODER = False
        _YEMENI_TOKENIZER = False
        logger.warning(f"[qa_engine] تعذّر تحميل طبقة التوليد الحر — "
                        f"generation_mode سيُعامَل كأنه False: {e}")
    ok = _YEMENI_DECODER is not False and _YEMENI_TOKENIZER is not False
    return (_YEMENI_DECODER, _YEMENI_TOKENIZER) if ok else (None, None)


def _build_grounding_text(
    concept_matches: List[Tuple[str, float]],
    verses: List[Dict[str, Any]],
    max_concepts: int = 5,
    max_verses: int = 3,
) -> str:
    """
    يبني نص سياق تأسيسي (grounding) من نفس المعطيات الرمزية التي يحسبها
    answer_question أصلاً (مفاهيم مطابقة + آيات داعمة) — لا يستدعي أي
    مصدر خارجي جديد، فقط يعيد صياغتها كنص متصل يُغذّى للـ decoder.
    """
    parts: List[str] = []
    if concept_matches:
        names = "، ".join(c for c, _ in concept_matches[:max_concepts])
        parts.append(f"المفاهيم ذات الصلة: {names}.")
    if verses:
        for v in verses[:max_verses]:
            text = v.get("text", "")
            if text:
                parts.append(f"سورة {v.get('surah', '')} آية {v.get('ayah', '')}: {text}")
    return " ".join(parts)


def _try_generate_free_text(
    question: str,
    concept_matches: List[Tuple[str, float]],
    verses: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    top_k: int,
) -> Optional[str]:
    """
    يحاول توليد نص حر عبر YemeniDecoder، مؤسَّس (grounded) على المفاهيم
    والآيات المسترجَعة رمزياً. يُرجع None عند أي فشل (fallback آمن كامل —
    answer_question يتجاهل النتيجة ويكمل بمساره الرمزي الافتراضي).
    """
    decoder, tokenizer = _get_yemeni_generator()
    if decoder is None or tokenizer is None:
        return None
    try:
        import torch

        grounding_text = _build_grounding_text(concept_matches, verses)
        grounding_ids = tokenizer.encode(grounding_text, add_bos=False, add_eos=False) \
            if grounding_text else None
        prompt_ids = tokenizer.encode(question, add_bos=True, add_eos=False)

        grounding_tensor = (
            torch.tensor(grounding_ids, dtype=torch.int64).unsqueeze(0)
            if grounding_ids is not None and len(grounding_ids) > 0 else None
        )
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.int64).unsqueeze(0)

        with torch.no_grad():
            out_ids = decoder.generate(
                prompt_ids=prompt_tensor,
                grounding_ids=grounding_tensor,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

        generated_ids = out_ids[0].tolist()
        return tokenizer.decode(generated_ids, skip_special=True, stop_at_eos=True)
    except Exception as e:
        logger.warning(f"[qa_engine] فشل التوليد الحر (Yemeni LLM) — "
                        f"سيُستخدم المسار الرمزي فقط: {e}")
        return None


def _apply_neural_boost(
    question: str, related_concepts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    يعزّز ترتيب related_concepts دلالياً عبر مقارنة hash IDs (بدون توليد
    نص حر — انظر توثيق HashTokenizer في ai/arabic_transformer.py).
    عند أي فشل: يُرجع القائمة كما هي دون أي تعديل (fallback آمن كامل).
    """
    if not related_concepts:
        return related_concepts
    booster = _get_neural_booster()
    if booster is None:
        return related_concepts
    try:
        top = booster.predict_next(question, top_k=15)
        top_ids = [i for i, _ in top]
        top_id_set = set(top_ids)
        n = len(top_ids)

        def boost_of(name: str) -> float:
            ids = booster.tokenizer.encode(name, 3)
            if len(ids) == 0:
                return 0.0
            first_id = int(ids[0])
            if first_id not in top_id_set:
                return 0.0
            rank = top_ids.index(first_id)
            return 1.0 - (rank / n)

        for r in related_concepts:
            r["neural_boost"] = round(boost_of(r["concept"]), 4)

        # إعادة الترتيب: الوزن الأصلي يبقى العامل الأساسي، والتعزيز
        # العصبي يُستخدم فقط لكسر التعادل/تحسين طفيف (وزن صغير 0.1)
        related_concepts.sort(
            key=lambda r: (r.get("weight", 0.0) * (1.0 + 0.1 * r.get("neural_boost", 0.0))),
            reverse=True,
        )
        return related_concepts
    except Exception as e:
        logger.warning(f"[qa_engine] فشل التعزيز العصبي لهذا السؤال — "
                        f"استخدام الترتيب الأصلي: {e}")
        return related_concepts


# ═══════════════════════════════════════════════════════════════════════════
# تطبيع النص العربي (نفس منطق streamlit_app.py)
# ═══════════════════════════════════════════════════════════════════════════
_TASHKEEL  = re.compile(r'[\u064B-\u065F\u0670\u0640]')
_ALEF      = re.compile(r'[أإآٱ]')
_BOM       = re.compile(r'\ufeff')
_SPACES    = re.compile(r'\s+')


def normalize_arabic(text: str) -> str:
    text = _TASHKEEL.sub('', text)
    text = _ALEF.sub('ا', text)
    text = _BOM.sub('', text)
    text = _SPACES.sub(' ', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# كلمات أداة / وقف عربية — تُستثنى من استخراج المفاهيم من السؤال
# ═══════════════════════════════════════════════════════════════════════════
ARABIC_STOPWORDS = {
    "ما", "ماذا", "من", "هل", "كيف", "لماذا", "متى", "اين", "أين",
    "في", "على", "عن", "الى", "إلى", "مع", "هو", "هي", "هم", "نحن",
    "انت", "أنت", "انتم", "أنتم", "كان", "يكون", "قال", "يقول",
    "الذي", "التي", "الذين", "هذا", "هذه", "ذلك", "تلك",
    "لا", "لم", "لن", "قد", "كل", "بل", "أم", "او", "أو", "ثم",
    "اذا", "إذا", "حتى", "كما", "لكن", "وإن", "وان", "بين",
    "علاقة", "علاقه", "يقول", "تقول", "نص", "آية", "ايه", "ايات", "آيات",
    "القران", "القرآن", "الكريم", "حول", "بخصوص", "بشأن", "بشان",
    "موضوع", "معنى", "تفسير", "شرح", "وضح", "اشرح", "بين", "وضّح",
}

# اختصار: كلمات سؤال شائعة + حروف عطف نزيلها من بدايات/نهايات الكلمات
PREFIX_STRIP = ["وال", "بال", "فال", "كال", "لل", "ال", "و", "ف", "ب", "ل", "ك"]

# مفاهيم "إطارية" تُستخدم غالباً في صياغة السؤال نفسه ولا تمثل موضوعه
# (مثل: "ماذا يقول القرآن عن X؟") — تُخفَّض أولويتها إذا وُجد مفهوم آخر معها
META_CONCEPTS = {"قرآن", "كتاب", "وحي", "أنبياء", "رسالة"}

# عبارات استفهامية تدل على أن السؤال عن "كيان" (شخص/أمة/شخصية)
# وليس عن مفهوم عام — تُستخدم لتفعيل طبقة الكيانات المعرفية
ENTITY_QUESTION_PATTERNS = [
    "من هو", "من هي", "من هم",
    "ما هو", "ما هي",
    "من ", "ما قصة", "حدثني عن", "تحدث عن", "اخبرني عن", "أخبرني عن",
    "عرفني ب", "عرّفني ب",
]


def _strip_prefixes(word: str) -> str:
    """إزالة أل التعريف وحروف الجر/العطف الشائعة من بداية الكلمة."""
    for p in PREFIX_STRIP:
        if word.startswith(p) and len(word) - len(p) >= 2:
            return word[len(p):]
    return word


# ═══════════════════════════════════════════════════════════════════════════
# 1) استخراج المفاهيم من السؤال
# ═══════════════════════════════════════════════════════════════════════════
def extract_concepts_from_question(question: str, concepts_db: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    يحلل سؤالاً بالعربية ويستخرج المفاهيم الموجودة في الـ CKG التي تطابقه.
    يعيد قائمة (اسم المفهوم، درجة التطابق) مرتبة تنازلياً.
    """
    q_norm = normalize_arabic(question)

    # تقسيم السؤال إلى كلمات وتنظيفها من علامات الترقيم
    raw_words = re.split(r'[\s\u060C\u061F\u061B,.!?؟،؛]+', q_norm)
    words = [w for w in raw_words if w and w not in ARABIC_STOPWORDS]

    matches: Dict[str, float] = {}

    # حد أدنى لطول المفهوم لاعتباره صالحاً لمطابقة "تطابق جزئي" (substring)
    # يمنع مفاهيم قصيرة جداً (مثل "بر") من المطابقة الخاطئة داخل كلمات أطول
    MIN_LEN_FOR_SUBSTRING = 3

    # تجهيز نسخ مطبّعة من كل اسم مفهوم (مع وبدون إزالة السوابق)
    for cname, cdata in concepts_db.items():
        c_norm = normalize_arabic(cname)
        c_len  = len(c_norm)

        score = 0.0

        # (أ) تطابق المفهوم كاملاً كسلسلة فرعية من السؤال (لمفاهيم >= 3 حروف فقط)
        if c_len >= MIN_LEN_FOR_SUBSTRING and c_norm in q_norm:
            score = max(score, 1.0)

        # (ب) تطابق على مستوى الكلمات المفردة
        for w in words:
            w_clean = _strip_prefixes(w)
            if not w_clean:
                continue

            if w == c_norm or w_clean == c_norm:
                score = max(score, 1.0)
            elif c_len >= MIN_LEN_FOR_SUBSTRING and (c_norm in w or c_norm in w_clean):
                score = max(score, 0.85)
            elif (
                c_len >= MIN_LEN_FOR_SUBSTRING
                and (w_clean in c_norm or w in c_norm)
                and len(w_clean) >= MIN_LEN_FOR_SUBSTRING
                and len(w_clean) / c_len >= 0.7  # الكلمة تغطي معظم اسم المفهوم (يمنع تطابق "ايمان" مع "ايمان زواج")
            ):
                score = max(score, 0.7)
            elif c_len >= MIN_LEN_FOR_SUBSTRING and " " in c_norm:
                # (ج) مفاهيم مركّبة (تحتوي مسافة، مثل "خمر ومسكرات"):
                # نطابق على مستوى كل كلمة من كلمات المفهوم على حدة
                concept_words = [cw for cw in c_norm.split(" ") if len(cw) >= 3]
                for cw in concept_words:
                    if w_clean == cw or w == cw:
                        score = max(score, 0.8)
                    elif len(w_clean) >= 3 and cw[:3] == w_clean[:3] and abs(len(cw) - len(w_clean)) <= 2:
                        score = max(score, 0.4)
            else:
                # تشابه جذري بسيط: أول 3 حروف متطابقة (لمفاهيم وكلمات >=3 حروف)
                # بشرط أن يكون طول الكلمة والمفهوم متقاربين (يمنع تطابق كلمة قصيرة
                # مع بداية مفهوم مركّب أطول بكثير، مثل "ايمان" مع "ايمان زواج")
                if (
                    len(w_clean) >= 3 and c_len >= 3
                    and w_clean[:3] == c_norm[:3]
                    and abs(len(w_clean) - c_len) <= 2
                ):
                    score = max(score, 0.4)

        if score > 0:
            matches[cname] = score

    # ── خفض أولوية "المفاهيم الإطارية" إن وُجد مفهوم آخر غير إطاري معها ──
    non_meta = [c for c in matches if c not in META_CONCEPTS]
    if non_meta:
        for c in list(matches.keys()):
            if c in META_CONCEPTS:
                matches[c] *= 0.3

    sorted_matches = sorted(matches.items(), key=lambda x: (-x[1], -concepts_db.get(x[0], {}).get("frequency", 0)))
    return sorted_matches


# ═══════════════════════════════════════════════════════════════════════════
# 1.5) كشف أسئلة "الكيانات" (من هو/ما هي...) — لتفعيل طبقة الكيانات المعرفية
# ═══════════════════════════════════════════════════════════════════════════
def is_entity_question(question: str) -> bool:
    """
    يحدد إن كان السؤال يسأل عن "كيان" (شخص، نبي، أمة، شخصية)
    بدلاً من سؤال عام عن مفهوم أو علاقة.
    """
    q_norm = normalize_arabic(question)
    return any(q_norm.startswith(p) or f" {p}" in q_norm for p in ENTITY_QUESTION_PATTERNS)


def find_entity_match(
    concept_matches: List[Tuple[str, float]],
    entities_db: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    يبحث بين المفاهيم المستخرجة من السؤال عن أول مفهوم
    يملك إدخالاً في طبقة الكيانات المعرفية (entities.json).
    يعيد (اسم الكيان، بيانات الكيان) أو None.
    """
    for cname, _score in concept_matches:
        if cname in entities_db:
            return cname, entities_db[cname]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 2) إيجاد المفاهيم المرتبطة عبر العلاقات في CKG
# ═══════════════════════════════════════════════════════════════════════════
# الحد الأدنى لتكرار المفهوم المرتبط في القرآن ليُعتبر "ذا دلالة كافية"
# لعرضه في قائمة المفاهيم المرتبطة. مفاهيم نادرة جداً (مثل "سخرية" بـ4 تكرارات)
# قد تحصل على "weight" مرتفع كاذب رياضياً (count / min(freq)) لمجرد أن
# عدد ظهوراتها القليل تزامن مع مفهوم شائع — فنستثني هذه الحالات هنا.
MIN_RELATED_FREQUENCY = 8


def _relation_rank_score(weight: float, count: int, other_freq: int) -> float:
    """
    يحسب درجة ترتيب أكثر توازناً من "weight" الخام المخزّن في CKG.

    weight الخام = count / min(freq_a, freq_b) → ينحاز للمفاهيم النادرة
    (قاسم صغير يرفع النسبة حتى مع عدد تزامن قليل جداً).

    الدرجة الجديدة تأخذ في الحسبان أيضاً:
      - عدد مرات التزامن الفعلي (count) — أدلة أكثر = أوثق
      - تكرار المفهوم الآخر في القرآن (other_freq) — مفهوم له حضور
        حقيقي في النص، لا مجرد ذكر عابر
    """
    import math
    count_factor = math.log(count + 1) if count > 0 else 0.3  # علاقات بلا evidence (semantic/narrative) تحصل على عامل ثابت معتدل
    freq_factor  = math.log(other_freq + 1)
    return weight * count_factor * freq_factor


def find_related_concepts(
    primary_concepts: List[str],
    relations_db: Dict[str, Any],
    concepts_db: Optional[Dict[str, Any]] = None,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    يبحث في جدول العلاقات عن المفاهيم المرتبطة بالمفاهيم الأساسية
    المستخرجة من السؤال، مع ترتيب متوازن يتجنب طغيان المفاهيم
    النادرة جداً ذات "weight" مرتفع كاذب رياضياً (انظر _relation_rank_score).
    """
    concepts_db = concepts_db or {}
    primary_norm = {normalize_arabic(c) for c in primary_concepts}
    related: Dict[str, Dict[str, Any]] = {}

    for rel_key, rel in relations_db.items():
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        src_n, tgt_n = normalize_arabic(src), normalize_arabic(tgt)

        other = None
        if src_n in primary_norm and tgt_n not in primary_norm:
            other = tgt
        elif tgt_n in primary_norm and src_n not in primary_norm:
            other = src

        if other is None:
            continue

        # استثناء أسماء الجذور (root:...) من قائمة "المفاهيم المرتبطة"
        # المعروضة للمستخدم — تبقى متاحة عبر علاقات root_link لأغراض أخرى
        if other.startswith("root:"):
            continue

        other_freq = concepts_db.get(other, {}).get("frequency", 0)
        # استثناء المفاهيم النادرة جداً (دلالة ضعيفة إحصائياً)
        if concepts_db and other_freq < MIN_RELATED_FREQUENCY:
            continue

        weight = rel.get("weight", 0.0)
        count  = rel.get("count", 0)
        rtype  = rel.get("relation_type", "")
        score  = _relation_rank_score(weight, count, other_freq)

        # إن وجد المفهوم بأكثر من علاقة، نحتفظ بالأعلى بحسب الدرجة المتوازنة
        existing = related.get(other)
        if existing is None or score > existing["_score"]:
            related[other] = {
                "concept":       other,
                "weight":        weight,
                "relation_type": rtype,
                "evidence":      rel.get("evidence", []),
                "_score":        score,
            }

    ranked = sorted(related.values(), key=lambda x: -x["_score"])

    # ── ضمان تنوّع أنواع العلاقات في النتائج النهائية ──────────────────
    # الترتيب الخام قد يُهيمن عليه نوع واحد (عادة co_occurrence ذو evidence
    # كثيرة)، فتغيب علاقات semantic/narrative_sequence القيّمة موضوعياً
    # حتى لو كانت أقل توثيقاً إحصائياً. نضمن ظهور أعلى نتيجة من كل نوع
    # متاح أولاً، ثم نكمل الباقي بالترتيب العام.
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in ranked:
        by_type.setdefault(r["relation_type"], []).append(r)

    diversified: List[Dict[str, Any]] = []
    seen_concepts = set()

    # الجولة الأولى: أفضل نتيجة من كل نوع علاقة (بترتيب ظهور الأنواع
    # في القائمة الأصلية، أي الأنواع الأقوى عموماً أولاً)
    type_order = list(dict.fromkeys(r["relation_type"] for r in ranked))
    for rtype in type_order:
        candidates = by_type.get(rtype, [])
        if candidates and candidates[0]["concept"] not in seen_concepts:
            diversified.append(candidates[0])
            seen_concepts.add(candidates[0]["concept"])
        if len(diversified) >= top_k:
            break

    # الجولة الثانية: إكمال الباقي بالترتيب العام حتى الوصول لـ top_k
    if len(diversified) < top_k:
        for r in ranked:
            if r["concept"] in seen_concepts:
                continue
            diversified.append(r)
            seen_concepts.add(r["concept"])
            if len(diversified) >= top_k:
                break

    # إزالة الحقل الداخلي _score قبل الإعادة
    for r in diversified:
        r.pop("_score", None)
    return diversified[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
# 3) استرجاع الآيات الداعمة
# ═══════════════════════════════════════════════════════════════════════════
def _ref_to_surah_ayah(ref: str) -> Tuple[int, int]:
    """يحوّل مرجعاً مثل 'quran:2:153' إلى (سورة، آية)."""
    try:
        parts = ref.split(":")
        return int(parts[-2]), int(parts[-1])
    except Exception:
        return (0, 0)


def retrieve_supporting_verses(
    concept_matches: List[Tuple[str, float]],
    concepts_db: Dict[str, Any],
    ayat_by_ref: Dict[Tuple[int, int], Dict[str, Any]],
    max_verses: int = 5,
) -> List[Dict[str, Any]]:
    """
    يجمع الآيات الداعمة من حقل sources لكل مفهوم مطابق،
    مع ترتيبها بحسب قوة تطابق المفهوم.
    """
    seen_refs = set()
    verses: List[Dict[str, Any]] = []

    for cname, score in concept_matches:
        cdata = concepts_db.get(cname, {})
        sources = cdata.get("sources", [])
        for ref in sources:
            sa = _ref_to_surah_ayah(ref)
            if sa in seen_refs or sa == (0, 0):
                continue
            ayah_data = ayat_by_ref.get(sa)
            if not ayah_data:
                continue
            seen_refs.add(sa)
            verses.append({
                "surah":   sa[0],
                "ayah":    sa[1],
                "text":    ayah_data.get("text", ""),
                "concept": cname,
                "score":   score,
            })
            if len(verses) >= max_verses:
                return verses
    return verses


# ═══════════════════════════════════════════════════════════════════════════
# 4) درجة الثقة
# ═══════════════════════════════════════════════════════════════════════════
def compute_confidence(
    concept_matches: List[Tuple[str, float]],
    related_concepts: List[Dict[str, Any]],
    verses: List[Dict[str, Any]],
) -> float:
    """
    درجة ثقة مبنية على:
      - وجود مفاهيم مباشرة مطابقة (40%)
      - وجود علاقات دلالية مستنتجة (25%)
      - وجود آيات داعمة (35%)
    """
    confidence = 0.0

    if concept_matches:
        best_score = max(s for _, s in concept_matches)
        confidence += 0.40 * best_score

    if related_concepts:
        confidence += 0.25 * min(len(related_concepts) / 5, 1.0)

    if verses:
        confidence += 0.35 * min(len(verses) / 5, 1.0)

    return round(min(confidence, 1.0), 4)


# ═══════════════════════════════════════════════════════════════════════════
# 5) توليد إجابة منظمة
# ═══════════════════════════════════════════════════════════════════════════
# ── عبارات افتتاحية طبيعية بحسب المجموعة المعرفية للمفهوم الأساسي ──
CLUSTER_OPENERS = {
    "توحيد":   "في باب التوحيد وأسماء الله وصفاته، يتحدث القرآن الكريم عن",
    "عبادة":   "في باب العبادات، يبيّن القرآن الكريم أحكام ومعاني",
    "أخلاق":   "من القيم الأخلاقية التي يدعو إليها القرآن الكريم",
    "إيمان":   "في باب العقيدة والإيمان، يتناول القرآن الكريم",
    "آخرة":    "في وصف الدار الآخرة، يذكر القرآن الكريم",
    "نبوة":    "في سياق قصص الأنبياء والرسالات، يذكر القرآن الكريم",
    "معرفة":   "في مجال العلم والمعرفة، يوجّه القرآن الكريم إلى",
    "مجتمع":   "في تنظيم شؤون المجتمع، يبيّن القرآن الكريم أحكام",
    "كون":     "من آيات الله في الكون، يذكر القرآن الكريم",
    "قصص":     "من القصص القرآني، يروي القرآن الكريم خبر",
    "روح":     "في تزكية النفس وأحوالها، يتحدث القرآن الكريم عن",
    "حكم":     "في باب الأحكام والقضاء، يبيّن القرآن الكريم",
    "اقتصاد":  "في باب المعاملات المالية، ينظّم القرآن الكريم أحكام",
    "سلوك":    "من السلوكيات التي يحذّر القرآن الكريم منها أو يدعو إليها",
    "فقه":     "في الأحكام الفقهية، يبيّن القرآن الكريم حكم",
    "باطن":    "في أعمال القلوب والإيمان الباطن، يتحدث القرآن الكريم عن",
}

# علاقات لها صياغة طبيعية خاصة عند ذكرها
RELATION_PHRASES = {
    "co_occurrence":     "وترد هذه الفكرة مرتبطة في آيات عديدة بمفهوم",
    "semantic":          "وهي مرتبطة من حيث المعنى بمفهوم",
    "thematic_cluster":  "وتتكرر هذه الفكرة جنباً إلى جنب مع مفهوم",
    "root_link":         "ويتصل لفظياً بجذر",
    "narrative_sequence": "وترتبط في السياق القصصي بـ",
    "episodic_rule":     "وقد لوحظ تكرار ربطها بمفهوم",
}


def generate_entity_answer(
    question: str,
    entity_name: str,
    entity_data: Dict[str, Any],
    concept_matches: List[Tuple[str, float]],
    related_concepts: List[Dict[str, Any]],
    verses: List[Dict[str, Any]],
    concepts_db: Dict[str, Any],
) -> Dict[str, Any]:
    """
    يبني إجابة مباشرة لسؤال عن "كيان" (من هو/ما هي...) باستخدام
    طبقة الكيانات المعرفية (entities.json):

      كيان → وصف وصفات → مفاهيم مرتبطة → آيات داعمة

    بدلاً من البدء بالعلاقات الإحصائية فقط.
    """
    confidence = compute_confidence(concept_matches, related_concepts, verses)
    # سؤال كيان واضح + وجود وصف جاهز → ثقة عالية بطبيعتها
    confidence = max(confidence, 0.9)

    summary_parts = [entity_data.get("summary", "").strip()]

    attributes = entity_data.get("attributes", [])
    if attributes:
        attrs_joined = "، ".join(attributes)
        possessive = "صفاتها" if entity_data.get("gender") == "f" else "صفاته"
        summary_parts.append(f"من {possessive}: {attrs_joined}.")

    # ── المفاهيم المرتبطة: نُفضّل المرتبطة المعرّفة يدوياً في entities.json ──
    entity_related_names = entity_data.get("related_concepts", [])
    extra_related = []
    seen_related_names = {r["concept"] for r in related_concepts}
    for rname in entity_related_names:
        # تطابق مرن مع أسماء CKG (قد تختلف في التطبيع، مثل "ابرهيم" vs "إبراهيم")
        match = None
        if rname in concepts_db:
            match = rname
        else:
            rn = normalize_arabic(rname)
            match = next((c for c in concepts_db if normalize_arabic(c) == rn), None)
        if match and match not in seen_related_names and match != entity_name:
            extra_related.append({
                "concept":       match,
                "weight":        0.9,
                "relation_type": "entity_attribute",
                "evidence":      [],
            })
            seen_related_names.add(match)

    combined_related = extra_related + related_concepts

    if combined_related:
        names = "، ".join(f"«{r['concept']}»" for r in combined_related[:5])
        summary_parts.append(f"وترتبط هذه الشخصية بمفاهيم: {names}.")

    # ── ذكر الآيات بصياغة استشهادية ──
    if verses:
        refs = "، ".join(f"({v['surah']}:{v['ayah']})" for v in verses[:3])
        extra = f"، وغيرها من {len(verses)} آية" if len(verses) > 3 else ""
        summary_parts.append(f"ومن الآيات الدالة على ذلك: {refs}{extra}.")

    summary = " ".join(p for p in summary_parts if p)

    # ── تفاصيل المفاهيم الأساسية (تبقى كما هي للعرض في الواجهة) ──
    primary_details = []
    for cname, score in concept_matches[:5]:
        cdata = concepts_db.get(cname, {})
        primary_details.append({
            "name":      cname,
            "cluster":   cdata.get("cluster", "غير مصنّف"),
            "frequency": cdata.get("frequency", 0),
            "match":     score,
        })

    return {
        "question":         question,
        "summary":          summary,
        "primary_concepts": primary_details,
        "related_concepts": combined_related,
        "verses":           verses,
        "confidence":       round(min(confidence, 1.0), 4),
        "entity": {
            "name": entity_name,
            "type": entity_data.get("type", ""),
        },
    }


def generate_answer(
    question: str,
    concept_matches: List[Tuple[str, float]],
    related_concepts: List[Dict[str, Any]],
    verses: List[Dict[str, Any]],
    concepts_db: Dict[str, Any],
) -> Dict[str, Any]:
    """
    يبني إجابة منظمة بصياغة طبيعية احترافية (ملخص + مفاهيم مرتبطة +
    آيات داعمة + درجة ثقة) اعتماداً فقط على بيانات الـ CKG والقرآن الموجودة.

    الصياغة تتجنب الإشارة إلى "النظام" أو "CKG" أو الأرقام الداخلية،
    وتقدّم الإجابة كشرح معرفي مباشر يستشهد بالآيات كدليل.
    """
    confidence = compute_confidence(concept_matches, related_concepts, verses)

    if not concept_matches:
        return {
            "question":         question,
            "summary":          "لم يتم العثور على مفهوم واضح يطابق هذا السؤال في قاعدة المعرفة الحالية. "
                                 "حاول إعادة صياغة السؤال باستخدام مصطلح قرآني أوضح (مثل: الصبر، العدل، التوحيد، الصلاة).",
            "primary_concepts": [],
            "related_concepts": [],
            "verses":           [],
            "confidence":       0.0,
        }

    # ── المفاهيم الأساسية المكتشفة ──
    primary_names = [c for c, _ in concept_matches[:3]]

    # استبعاد المفاهيم "الإطارية" (مثل: قرآن، كتاب) من صياغة الملخص
    # إن وُجد معها مفهوم آخر أكثر دلالة على موضوع السؤال
    non_meta_names = [c for c in primary_names if c not in META_CONCEPTS]
    topic_names = non_meta_names if non_meta_names else primary_names

    main_concept = topic_names[0]
    main_cdata   = concepts_db.get(main_concept, {})
    main_cluster = main_cdata.get("cluster", "")
    opener = CLUSTER_OPENERS.get(main_cluster, "يتحدث القرآن الكريم عن")

    # إزالة المفاهيم الثانوية التي تشترك بكلمتها الأولى مع المفهوم الأساسي
    # أو مع مفهوم ثانوي آخر سبقه (مثل "حكمة" و"حكمة عملية")
    # لتجنب التكرار في الصياغة
    seen_first_words = {main_concept.split(" ")[0]}
    secondary_names = []
    for c in topic_names[1:]:
        first_word = c.split(" ")[0]
        if first_word in seen_first_words:
            continue
        seen_first_words.add(first_word)
        secondary_names.append(c)

    # ── بناء ملخص الإجابة بصياغة طبيعية ──
    summary_parts = []

    if not secondary_names:
        summary_parts.append(f"{opener} «{main_concept}».")
    else:
        secondary = "، ".join(f"«{c}»" for c in secondary_names)
        summary_parts.append(f"{opener} «{main_concept}»، وتتصل هذه الفكرة أيضاً بـ {secondary}.")

    # ── ذكر العلاقات المستنتجة بصياغة طبيعية ──
    if related_concepts:
        # نختار أقوى علاقة من كل نوع متاح (حتى نتنوع في الصياغة) بحد أقصى 3
        seen_types = {}
        for r in related_concepts:
            rtype = r.get("relation_type", "")
            if rtype not in seen_types and r["concept"] != main_concept:
                seen_types[rtype] = r
            if len(seen_types) >= 3:
                break

        rel_sentences = []
        for rtype, r in seen_types.items():
            phrase = RELATION_PHRASES.get(rtype, "وترتبط بمفهوم")
            target = r["concept"]
            # إزالة بادئة "root:" إن وُجدت في أسماء الجذور
            target_display = target.replace("root:", "")
            rel_sentences.append(f"{phrase} «{target_display}»")

        if rel_sentences:
            summary_parts.append("، ".join(rel_sentences) + ".")

    # ── ذكر الآيات بصياغة استشهادية ──
    if verses:
        if len(verses) == 1:
            v = verses[0]
            summary_parts.append(f"ومن الآيات الدالة على ذلك قوله تعالى في سورة {v['surah']} الآية {v['ayah']}.")
        else:
            refs = "، ".join(f"({v['surah']}:{v['ayah']})" for v in verses[:3])
            extra = f"، وغيرها من {len(verses)} آية" if len(verses) > 3 else ""
            summary_parts.append(f"ومن الآيات الدالة على ذلك: {refs}{extra}.")
    else:
        summary_parts.append("ولم يُعثر على آيات مرتبطة مباشرة بهذا المفهوم في الفهرس الحالي.")

    summary = " ".join(summary_parts)

    # ── تفاصيل المفاهيم الأساسية ──
    primary_details = []
    for cname, score in concept_matches[:5]:
        cdata = concepts_db.get(cname, {})
        primary_details.append({
            "name":      cname,
            "cluster":   cdata.get("cluster", "غير مصنّف"),
            "frequency": cdata.get("frequency", 0),
            "match":     score,
        })

    return {
        "question":         question,
        "summary":          summary,
        "primary_concepts": primary_details,
        "related_concepts": related_concepts,
        "verses":           verses,
        "confidence":       confidence,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6) الدالة الرئيسية — تجميع كل المراحل
# ═══════════════════════════════════════════════════════════════════════════
def answer_question(
    question: str,
    ckg: Dict[str, Any],
    ayat: List[Dict[str, Any]],
    max_verses: int = 5,
    max_related: int = 8,
    entities: Optional[Dict[str, Any]] = None,
    generation_mode: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
) -> Dict[str, Any]:
    """
    نقطة الدخول الرئيسية لمحرك الأسئلة والأجوبة.

    المراحل:
      0. إن كان السؤال عن "كيان" (من هو/ما هي...) ووُجد له وصف
         في طبقة الكيانات المعرفية (entities)، تُبنى الإجابة من:
         كيان → وصف وصفات → مفاهيم مرتبطة → آيات داعمة
      1. استخراج المفاهيم من السؤال
      2. البحث في العلاقات عن مفاهيم مرتبطة
      3. استرجاع الآيات الداعمة من sources
      4. توليد إجابة منظمة مع درجة ثقة
      5. (اختياري) generation_mode=True: توليد نص حر إضافي عبر YemeniDecoder،
         مؤسَّس على نفس المفاهيم/الآيات المسترجَعة في الخطوات 1-3. لا يستبدل
         أبداً الإجابة الرمزية أعلاه — يُضاف كحقل إضافي فقط، وعند أي فشل
         (لا torch، لا checkpoint، إلخ) يُتجاهل بصمت والإجابة الرمزية تبقى
         كما هي دون أي تغيير في السلوك الافتراضي (generation_mode=False).
    """
    concepts_db  = ckg.get("concepts", {})
    relations_db = ckg.get("relations", {})
    entities_db  = entities or {}

    # فهرسة الآيات بحسب (سورة، آية) لتسريع البحث
    ayat_by_ref = {(a.get("surah"), a.get("ayah")): a for a in ayat}

    # 1. استخراج المفاهيم
    concept_matches = extract_concepts_from_question(question, concepts_db)

    # 2. المفاهيم المرتبطة — نستثني المفاهيم "الإطارية" (مثل: قرآن، كتاب) من
    # بذور البحث عن العلاقات، لأنها تملك علاقات قوية عامة (مثل قرآن↔كتاب)
    # قد تطغى على علاقات المفهوم الموضوعي الفعلي للسؤال
    all_primary = [c for c, _ in concept_matches[:3]]
    non_meta_primary = [c for c in all_primary if c not in META_CONCEPTS]
    primary_names = non_meta_primary if non_meta_primary else all_primary
    related_concepts = find_related_concepts(primary_names, relations_db, concepts_db, top_k=max_related) if primary_names else []
    related_concepts = _apply_neural_boost(question, related_concepts)

    # 3. الآيات الداعمة
    verses = retrieve_supporting_verses(concept_matches, concepts_db, ayat_by_ref, max_verses=max_verses)

    # 0/4. تفعيل طبقة الكيانات المعرفية لأسئلة "من هو/ما هي..."
    result: Optional[Dict[str, Any]] = None
    if entities_db and is_entity_question(question):
        entity_match = find_entity_match(concept_matches, entities_db)
        if entity_match:
            entity_name, entity_data = entity_match
            result = generate_entity_answer(
                question, entity_name, entity_data,
                concept_matches, related_concepts, verses, concepts_db,
            )

    # 4. الإجابة المنظمة (المسار العام: مفهوم → علاقات → آيات) — إن لم يُعثر
    # على كيان مطابق أعلاه
    if result is None:
        result = generate_answer(question, concept_matches, related_concepts, verses, concepts_db)

    # 5. توليد حر اختياري (Yemeni LLM) — إضافي فقط، لا يمسّ أي من المسارين أعلاه
    result["generation_used"] = False
    result["generated_text"] = None
    if generation_mode:
        generated = _try_generate_free_text(
            question, concept_matches, verses, temperature, top_p, top_k,
        )
        if generated:
            result["generated_text"] = generated
            result["generation_used"] = True

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 7) أداة مساعدة: تشابه نصي بين سؤالين (للذاكرة التجريبية لاحقاً)
# ═══════════════════════════════════════════════════════════════════════════
def question_similarity(q1: str, q2: str) -> float:
    """
    تشابه بسيط بين سؤالين بناءً على تقاطع الكلمات (Jaccard)
    بعد التطبيع وإزالة كلمات الوقف.
    """
    def tokenize(q: str) -> set:
        norm = normalize_arabic(q)
        raw = re.split(r'[\s\u060C\u061F\u061B,.!?؟،؛]+', norm)
        return {_strip_prefixes(w) for w in raw if w and w not in ARABIC_STOPWORDS}

    t1, t2 = tokenize(q1), tokenize(q2)
    if not t1 or not t2:
        return 0.0
    inter = len(t1 & t2)
    union = len(t1 | t2)
    return round(inter / union, 4) if union else 0.0
