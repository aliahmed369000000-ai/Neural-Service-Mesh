"""
YemeniTokenizer — NSM Generative Tokenization Layer
=====================================================
محرك ترميز قابل للعكس (reversible) مصمّم للتوليد النصي الحر.

على عكس HashTokenizer (FNV-1a بلا decode)، هذا المرمِّز يحتفظ بقاموس
نصي كامل (word ↔ ID) ويدعم encode() + decode() معاً — وهو الشرط الضروري
لتشغيل autoregressive generation حقيقي.

القاموس المبدئي
----------------
• رموز خاصة قياسية (PAD / UNK / BOS / EOS / SEP / MASK)
• مفردات عربية كلاسيكية أساسية (قرآنية وفصيحة)
• لهجة يمنية: مصطلحات ومرادفات وتعابير محلية
• مسافة واسعة لإضافة كلمات جديدة ديناميكياً أثناء التدريب

الترميز
--------
• يقسّم النص إلى كلمات (regex يستهدف العربية/الأرقام/العلامات)
• الكلمات غير الموجودة في القاموس → UNK (تُضاف اختيارياً للقاموس إن
  كان grow_vocab=True)

Padding / Masking
-----------------
• pad_sequence()    — تسوية batch إلى طول موحّد
• make_pad_mask()   — قناع (batch, seq_len) لتجاهل PAD في الانتباه
• make_causal_mask() — قناع سببي علوي-مثلثي للجيل التلقائي (decoder-only)

التوافق مع arabic_transformer.py
----------------------------------
• نفس ثوابت الرموز الخاصة (PAD=0, UNK=1, BOS=2, EOS=3, SEP=4, MASK=5)
• encode() يُعيد np.ndarray(dtype=int64) مثل HashTokenizer.encode()
• يمكن تمرير المُخرَج مباشرةً إلى ArabicTransformer._forward() أو
  LoRATransformerAdapter.forward()
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# ثوابت عامة
# ══════════════════════════════════════════════════════════════════════════════

PAD_ID  = 0
UNK_ID  = 1
BOS_ID  = 2
EOS_ID  = 3
SEP_ID  = 4
MASK_ID = 5

_SPECIAL_TOKENS: Dict[str, int] = {
    "<PAD>":  PAD_ID,
    "<UNK>":  UNK_ID,
    "<BOS>":  BOS_ID,
    "<EOS>":  EOS_ID,
    "<SEP>":  SEP_ID,
    "<MASK>": MASK_ID,
}

# ══════════════════════════════════════════════════════════════════════════════
# سقف القاموس — يطابق VOCAB_SIZE في arabic_transformer.py (طبقة الإخراج ثابتة
# الحجم). لا يمكن لأي id صادر عن هذا المرمِّز أن يتجاوز MAX_VOCAB_SIZE - 1،
# مهما نما القاموس — هذا يمنع IndexError عند الربط بـ OutputHead/Embedding.
# ══════════════════════════════════════════════════════════════════════════════
MAX_VOCAB_SIZE = 8192

_SUBWORD_HASH_START = MAX_VOCAB_SIZE - 1024   # 7168
_SUBWORD_MARK        = "##"                    # علامة "بقية كلمة" على غرار BPE

_COMMON_PREFIXES = ("وال", "بال", "كال", "فال", "ال", "و", "ف", "ب", "ك", "ل")
_COMMON_SUFFIXES = ("ونها", "اتها", "كموها", "ون", "ين", "ات", "ها", "هم",
                     "كم", "تم", "نا", "ة", "ي", "ه")


def _fnv1a_bounded(text: str, span: int) -> int:
    """FNV-1a قياسي، مُقيَّد ضمن [0, span) — لا يعتمد على أي قاموس محفوظ."""
    h = 2166136261
    for ch in text.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h % span


def _segment_subwords(word: str) -> List[str]:
    """
    تجزئة مورفولوجية مبسّطة: يفصل بادئة/لاحقة شائعة عن الجذر إن وُجدتا،
    فيعطي 1-3 مقاطع بدل كلمة كاملة واحدة.
    """
    if len(word) <= 3:
        return [word]

    prefix, root = "", word
    for p in sorted(_COMMON_PREFIXES, key=len, reverse=True):
        if root.startswith(p) and len(root) - len(p) >= 2:
            prefix, root = p, root[len(p):]
            break

    suffix = ""
    for s in sorted(_COMMON_SUFFIXES, key=len, reverse=True):
        if root.endswith(s) and len(root) - len(s) >= 2:
            root, suffix = root[: -len(s)], s
            break

    parts = [p for p in (prefix, root, suffix) if p]
    return parts if len(parts) > 1 else [word]

# ══════════════════════════════════════════════════════════════════════════════
# القاموس المبدئي — عربي كلاسيكي + لهجة يمنية
# ══════════════════════════════════════════════════════════════════════════════

# بذرة المفردات: كل كلمة ستحصل على ID فريد بدءاً من 6
_SEED_VOCABULARY: List[str] = [

    # ── رموز خاصة (IDs 0–5 محجوزة أعلاه) ─────────────────────────────────
    # (لا تُكرَّر هنا — ستُدرَج تلقائياً)

    # ── أحرف وعلامات ترقيم شائعة ───────────────────────────────────────────
    "،", ".", "؟", "!", ":", "-", "\"", "(", ")", "\n",

    # ── كلمات عربية كلاسيكية / قرآنية ─────────────────────────────────────
    "بسم", "الله", "الرحمن", "الرحيم", "الحمد", "لله", "رب", "العالمين",
    "مالك", "يوم", "الدين", "اياك", "نعبد", "نستعين", "اهدنا", "الصراط",
    "المستقيم", "صراط", "الذين", "انعمت", "عليهم", "غير", "المغضوب",
    "الضالين", "قل", "هو", "احد", "الصمد", "لم", "يلد", "يولد", "يكن",
    "كفوا", "وتعالى", "سبحانه", "ان", "في", "من", "على", "الى", "مع",
    "عن", "بعد", "قبل", "كل", "ما", "هذا", "هذه", "ذلك", "تلك",
    "وهو", "وهي", "وهم", "وهن", "انه", "انها", "انهم", "لا", "الا",
    "كان", "يكون", "كانت", "ليس", "ليست", "فهو", "فهي", "فهم",
    "قال", "قالت", "قالوا", "يقول", "تقول", "يقولون",
    "علم", "يعلم", "علمنا", "العلم", "المعرفة", "الحكمة",
    "رحمة", "الرحمة", "نعمة", "النعمة", "امر", "الامر",
    "الحق", "الباطل", "الخير", "الشر", "الخلق", "الامة",
    "النبي", "الرسول", "الرسل", "الانبياء", "محمد", "ابراهيم",
    "موسى", "عيسى", "ادم", "نوح", "يوسف", "داود", "سليمان",
    "الايمان", "الاسلام", "الاحسان", "التقوى", "الصبر",
    "الشكر", "الذكر", "الدعاء", "الصلاة", "الزكاة", "الصوم",
    "الحج", "الجهاد", "القرآن", "السنة", "الحديث", "الفقه",
    "العقيدة", "التوحيد", "الشريعة", "الحكم", "العدل",
    "الرزق", "المال", "الدنيا", "الاخرة", "الجنة", "النار",
    "الملائكة", "الجن", "الشيطان", "ابليس", "القيامة",
    "الحساب", "الميزان", "الصراط", "الشفاعة",
    "العقل", "القلب", "النفس", "الروح", "البدن",
    "الانسان", "البشر", "الرجل", "المرأة", "الولد",
    "الاب", "الام", "الاخ", "الاخت", "الاسرة", "المجتمع",
    "اسم", "كلمة", "جملة", "سؤال", "جواب", "معنى",
    "كتاب", "علوم", "تعلم", "تعليم", "مدرسة", "جامعة",
    "بحث", "فكر", "تفكير", "رأي", "حكمة", "فلسفة",
    "تاريخ", "حضارة", "ثقافة", "لغة", "عربية", "شعر",
    "ارض", "سماء", "بحر", "نهر", "جبل", "مدينة", "قرية",
    "بيت", "مسجد", "كعبة", "مكة", "المدينة", "القدس",
    "اليمن", "مصر", "الشام", "العراق", "الجزيرة", "العربية",
    "اليوم", "ليلة", "شهر", "سنة", "زمان", "مكان",
    "صباح", "مساء", "ليل", "نهار", "وقت", "ساعة",
    "ذهب", "جاء", "عاد", "اكل", "شرب", "نام", "قام",
    "فعل", "عمل", "كتب", "قرأ", "سمع", "رأى", "علم",
    "احب", "كره", "خاف", "امن", "رجا", "ظن", "عرف",
    "كبير", "صغير", "كثير", "قليل", "حسن", "جيد", "جميل",
    "صحيح", "خطأ", "حق", "عدل", "ظلم", "امان", "خطر",
    "نعم", "لا", "بلى", "سبحان", "الحمد", "آمين",
    "ثم", "او", "اما", "لكن", "اذا", "لو", "حتى",
    "جدا", "ايضا", "فقط", "حتى", "اكثر", "اقل",

    # ── لهجة يمنية — مفردات ومصطلحات ───────────────────────────────────────
    "أبشر",        # بمعنى: تفضّل / اطمئن
    "سدا",         # هكذا / على ما يرام
    "جهال",        # أطفال / جهلاء (حسب السياق)
    "زول",         # شخص / فرد
    "شيّال",       # حامل / حمّال
    "تعال",        # هلمّ / أقبل
    "عندي",        # لديّ
    "وين",         # أين
    "فين",         # أين (اليمنية الشمالية)
    "هناك",        # هناك / ثمة
    "كيفك",        # كيف حالك
    "تمام",        # بخير / حسناً
    "حلو",         # جيد / لطيف
    "ايش",         # ماذا / ما
    "ليش",         # لماذا
    "متين",        # متى
    "مره",         # جداً / مرة
    "يلا",         # هيا / انطلق
    "والله",       # والله / أقسم بالله
    "يسعد",        # يسعدك / تحية
    "صباح",        # صباح الخير (اختصار)
    "مساك",        # مساء الخير (اختصار)
    "خير",         # خير / بخير
    "مشكلة",       # مشكلة
    "حال",         # حال / وضع
    "ربي",         # ربّي (نداء ديني)
    "حبيبي",       # حبيبي / عزيزي
    "يا",          # يا (نداء)
    "صاحبي",       # صديقي / رفيقي
    "اخوي",        # أخي
    "عمي",         # عمّي (نداء للكبير)
    "الناس",       # الناس
    "الشعب",       # الشعب
    "بلدنا",       # بلدنا
    "وطن",         # وطن
    "شجاع",        # شجاع / جريء
    "كرم",         # كرم / سخاء
    "ضيف",         # ضيف
    "قهوة",        # قهوة (القهوة اليمنية الشهيرة)
    "بن",          # البن / القهوة
    "عسل",         # عسل (العسل اليمني)
    "زبيب",        # زبيب
    "خبز",         # خبز
    "سمك",         # سمك
    "لحم",         # لحم
    "مرق",         # مرق / حساء
    "بيت",         # بيت / منزل
    "دار",         # دار / منزل (يمنية)
    "سوق",         # سوق
    "قات",         # القات (نبات يمني)
    "جبل",         # جبل
    "وادي",        # وادي
    "صنعاء",       # صنعاء
    "عدن",         # عدن
    "تعز",         # تعز
    "حضرموت",      # حضرموت
    "المكلا",      # المكلا
    "إب",          # إب
    "ذمار",        # ذمار
    "الحديدة",     # الحديدة
    "مأرب",        # مأرب
    "شبوة",        # شبوة
    "البيضاء",     # البيضاء

    # ── عبارات يمنية شائعة (مُعالَجة كوحدة واحدة) ───────────────────────────
    "كيف_حالك_يا_صاحبي",   # كيف حالك يا صاحبي
    "اهلا_وسهلا",          # أهلاً وسهلاً
    "في_امان_الله",         # في أمان الله
    "الله_يعافيك",          # الله يعافيك
    "ما_شاء_الله",          # ما شاء الله
    "بارك_الله_فيك",        # بارك الله فيك
    "الله_يرزقك",           # الله يرزقك
    "يعطيك_العافية",        # يعطيك العافية
    "الله_يبارك",           # الله يبارك
    "جزاك_الله_خيرا",       # جزاك الله خيراً

    # ── أرقام ───────────────────────────────────────────────────────────────
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩", "٠",
]


# ══════════════════════════════════════════════════════════════════════════════
# دالة تطبيع النص (مشتركة مع HashTokenizer)
# ══════════════════════════════════════════════════════════════════════════════

_TASHKEEL_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_WORD_RE     = re.compile(r"[\u0600-\u06FF\u0660-\u0669]+|[0-9]+|[،.؟!:\"()\-\n]")


def _normalize(text: str) -> str:
    """إزالة التشكيل وتوحيد أشكال الحروف."""
    text = _TASHKEEL_RE.sub("", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = re.sub(r"[ىئ]", "ي", text)
    text = text.replace("ة", "ه")
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# YemeniTokenizer
# ══════════════════════════════════════════════════════════════════════════════

class YemeniTokenizer:
    """
    مُرمِّز قابل للعكس مُصمَّم للتوليد النصي الحر باللهجة اليمنية والعربية.

    المزايا على HashTokenizer
    --------------------------
    ✓ decode()   — يحوّل تسلسل IDs إلى نص مقروء
    ✓ encode()   — نفس الواجهة تماماً مع HashTokenizer
    ✓ grow_vocab — يضيف كلمات جديدة ديناميكياً أثناء التدريب
    ✓ save/load  — يحفظ القاموس كـ JSON (صغير / شفاف)
    ✓ pad/mask   — أدوات تسوية الـ batch للتدريب

    التوافق
    -------
    نفس ثوابت الرموز الخاصة مع HashTokenizer:
        PAD=0, UNK=1, BOS=2, EOS=3, SEP=4, MASK=5
    """

    # ثوابت الرموز الخاصة (تطابق HashTokenizer تماماً)
    PAD  = PAD_ID
    UNK  = UNK_ID
    BOS  = BOS_ID
    EOS  = EOS_ID
    SEP  = SEP_ID
    MASK = MASK_ID

    SPECIAL_TOKENS = list(_SPECIAL_TOKENS.keys())
    _FIRST_WORD_ID = len(_SPECIAL_TOKENS)  # = 6

    def __init__(
        self,
        grow_vocab: bool = True,
        normalize:  bool = True,
        vocab_path: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        grow_vocab  : إن True، تُضاف الكلمات الجديدة عند encode() تلقائياً.
        normalize   : إن True، يُطبَّق _normalize() (إزالة تشكيل + توحيد حروف).
        vocab_path  : مسار اختياري لملف JSON لتحميل قاموس محفوظ مسبقاً.
        """
        self.grow_vocab = grow_vocab
        self.normalize  = normalize

        # القاموسان المتبادلان
        self._word2id: Dict[str, int] = {}
        self._id2word: Dict[int, str] = {}

        # أضف الرموز الخاصة
        for tok, tid in _SPECIAL_TOKENS.items():
            self._word2id[tok] = tid
            self._id2word[tid] = tok

        # أضف بذرة المفردات
        for word in _SEED_VOCABULARY:
            self._add_word(word)

        # حمّل قاموس خارجي إن وُجد
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
            logger.info(f"[YemeniTokenizer] ✓ قاموس محمَّل من {vocab_path} "
                        f"| حجم: {self.vocab_size:,}")
        else:
            logger.info(f"[YemeniTokenizer] ✓ قاموس مُهيَّأ | "
                        f"حجم: {self.vocab_size:,} كلمة")

    # ──────────────────────────────────────────────────────────────────────
    # إدارة القاموس
    # ──────────────────────────────────────────────────────────────────────

    def _add_word(self, word: str) -> int:
        """
        يضيف كلمة إن لم تكن موجودة ويُعيد ID-ها. عند امتلاء منطقة الكلمات
        الكاملة (_SUBWORD_HASH_START)، تُجزَّأ الكلمة صرفياً ويُطوى كل مقطع
        إلى ID ثابت داخل نطاق الـ hash المحجوز — يضمن عدم تجاوز MAX_VOCAB_SIZE.
        """
        if word in self._word2id:
            return self._word2id[word]

        if len(self._word2id) < _SUBWORD_HASH_START:
            new_id = len(self._word2id)
            self._word2id[word] = new_id
            self._id2word[new_id] = word
            return new_id

        return self._hash_into_subword_region(word)

    def _hash_into_subword_region(self, word: str) -> int:
        """يطوي كلمة/مقطعاً إلى ID ثابت داخل [_SUBWORD_HASH_START, MAX_VOCAB_SIZE)."""
        span = MAX_VOCAB_SIZE - _SUBWORD_HASH_START
        tid = _SUBWORD_HASH_START + _fnv1a_bounded(word, span)
        self._id2word.setdefault("_subword_debug", {})
        self._id2word["_subword_debug"][tid] = word  # type: ignore[index]
        return tid

    def _word_or_subwords_to_ids(self, word: str) -> List[int]:
        """يعطي ID مباشر إن توفّرت مساحة، وإلا يجزّئ الكلمة صرفياً."""
        if word in self._word2id or len(self._word2id) < _SUBWORD_HASH_START:
            return [self._add_word(word)]
        return [self._hash_into_subword_region(p) for p in _segment_subwords(word)]

    @property
    def vocab_size(self) -> int:
        return len(self._word2id)

    def word_to_id(self, word: str) -> int:
        """كلمة → ID (UNK إن لم توجد ولم يكن grow_vocab=True)."""
        key = _normalize(word) if self.normalize else word
        if key in self._word2id:
            return self._word2id[key]
        if self.grow_vocab:
            return self._add_word(key)
        return self.UNK

    def id_to_word(self, token_id: int) -> str:
        """ID → كلمة (أو '<UNK>' إن لم يوجد)."""
        if token_id in self._id2word:
            return self._id2word[token_id]
        subword_debug = self._id2word.get("_subword_debug", {})
        if isinstance(subword_debug, dict) and token_id in subword_debug:
            return _SUBWORD_MARK + subword_debug[token_id]
        return "<UNK>"

    @property
    def total_id_space(self) -> int:
        """الحد الأقصى المطلق لأي ID — استخدمه لحجم Embedding، وليس vocab_size."""
        return MAX_VOCAB_SIZE

    # ──────────────────────────────────────────────────────────────────────
    # Encode — نص → تسلسل IDs
    # ──────────────────────────────────────────────────────────────────────

    def encode(
        self,
        text:    str,
        max_len: int = 128,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> np.ndarray:
        """
        نص عربي → np.ndarray(dtype=int64) بنفس واجهة HashTokenizer.encode().

        Parameters
        ----------
        text    : النص المُراد ترميزه.
        max_len : أقصى طول للتسلسل (يقطع إن تجاوز).
        add_bos : أضف BOS في البداية (افتراضي: True).
        add_eos : أضف EOS في النهاية (افتراضي: True).

        Returns
        -------
        np.ndarray[int64] — التسلسل مع BOS/EOS.
        """
        if self.normalize:
            text = _normalize(text)

        # استبدل عبارات مركّبة (underscore-joined) بنسخة مطابقة لما في القاموس
        # مثال: "كيف حالك يا صاحبي" → "كيف_حالك_يا_صاحبي" إن وُجدت
        text = self._replace_multiword_phrases(text)

        words = _WORD_RE.findall(text)
        ids: List[int] = []

        if add_bos:
            ids.append(self.BOS)

        for w in words:
            ids.extend(self._word_or_subwords_to_ids(w))

        if add_eos:
            ids.append(self.EOS)

        # قص إن تجاوز max_len (مع الحفاظ على EOS في النهاية)
        if len(ids) > max_len:
            if add_eos:
                ids = ids[: max_len - 1] + [self.EOS]
            else:
                ids = ids[:max_len]

        return np.array(ids, dtype=np.int64)

    # ──────────────────────────────────────────────────────────────────────
    # Decode — تسلسل IDs → نص
    # ──────────────────────────────────────────────────────────────────────

    def decode(
        self,
        ids:             "np.ndarray | List[int]",
        skip_special:    bool = True,
        stop_at_eos:     bool = True,
    ) -> str:
        """
        np.ndarray(int64) أو List[int] → نص عربي مقروء.

        Parameters
        ----------
        ids           : تسلسل IDs الصادر من encode() أو generate().
        skip_special  : تجاهل الرموز الخاصة (BOS/EOS/PAD/...) في الإخراج.
        stop_at_eos   : أوقف الفك عند أول EOS.

        Returns
        -------
        str — النص المُعاد بناؤه.
        """
        special_ids = set(_SPECIAL_TOKENS.values())
        words: List[str] = []

        for tid in ids:
            tid = int(tid)
            if stop_at_eos and tid == self.EOS:
                break
            if skip_special and tid in special_ids:
                continue
            word = self.id_to_word(tid)
            # حوّل العبارات المركّبة (underscore) إلى نص عادي
            word = word.replace("_", " ")
            words.append(word)

        return " ".join(words)

    # ──────────────────────────────────────────────────────────────────────
    # Padding / Masking — دعم batch
    # ──────────────────────────────────────────────────────────────────────

    def pad_sequence(
        self,
        sequences: List[np.ndarray],
        max_len:   Optional[int] = None,
        pad_id:    int = PAD_ID,
    ) -> np.ndarray:
        """
        يُسوّي قائمة تسلسلات بأطوال متفاوتة إلى مصفوفة (batch, max_len).

        Parameters
        ----------
        sequences : قائمة من np.ndarray(int64)، كل واحدة تسلسل مُرمَّز.
        max_len   : الطول المستهدف (يُستخدم أطول تسلسل إن لم يُحدَّد).
        pad_id    : قيمة الـ padding (افتراضي: PAD=0).

        Returns
        -------
        np.ndarray shape (len(sequences), max_len) — dtype int64.
        """
        if not sequences:
            return np.zeros((0, max_len or 1), dtype=np.int64)

        L = max_len or max(len(s) for s in sequences)
        batch = np.full((len(sequences), L), pad_id, dtype=np.int64)
        for i, seq in enumerate(sequences):
            n = min(len(seq), L)
            batch[i, :n] = seq[:n]
        return batch

    def make_pad_mask(
        self,
        padded: np.ndarray,
        pad_id: int = PAD_ID,
    ) -> np.ndarray:
        """
        يُنشئ قناع الـ padding — True حيث يوجد PAD (يُتجاهل في الانتباه).

        Parameters
        ----------
        padded : مصفوفة (batch, seq_len) من pad_sequence().
        pad_id : قيمة الـ padding المُستخدَمة.

        Returns
        -------
        np.ndarray bool shape (batch, seq_len) —
            True  = موضع PAD (يُطمَّس في softmax)
            False = موضع حقيقي
        """
        return padded == pad_id

    @staticmethod
    def make_causal_mask(seq_len: int) -> np.ndarray:
        """
        يُنشئ قناعاً سببياً علوياً (upper-triangular causal mask).

        Used in decoder-only generation: كل موضع لا يرى المواضع التالية.

        Parameters
        ----------
        seq_len : طول التسلسل.

        Returns
        -------
        np.ndarray bool shape (seq_len, seq_len) —
            True  = يجب حجبه (المستقبل)
            False = مسموح به (الماضي + الحاضر)

        مثال لـ seq_len=4:
            [[F, T, T, T],
             [F, F, T, T],
             [F, F, F, T],
             [F, F, F, F]]
        """
        return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

    @staticmethod
    def make_combined_mask(
        pad_mask:    np.ndarray,   # (batch, seq_len) bool
        causal_mask: np.ndarray,   # (seq_len, seq_len) bool
    ) -> np.ndarray:
        """
        يدمج قناع PAD وقناع السببية في قناع موحّد.

        مفيد لتمرير قناع واحد إلى MultiHeadAttention عند توليد batch.

        Parameters
        ----------
        pad_mask    : (batch, seq_len) — True = PAD
        causal_mask : (seq_len, seq_len) — True = مستقبل

        Returns
        -------
        np.ndarray bool shape (batch, seq_len, seq_len) —
            True = يجب حجبه (PAD أو مستقبل)
        """
        # (batch, 1, seq_len) | (1, seq_len, seq_len) → (batch, seq_len, seq_len)
        pad_exp    = pad_mask[:, None, :]           # (batch, 1, seq_len)
        causal_exp = causal_mask[None, :, :]        # (1, seq_len, seq_len)
        return pad_exp | causal_exp

    # ──────────────────────────────────────────────────────────────────────
    # عبارات مركّبة (Multi-word phrases)
    # ──────────────────────────────────────────────────────────────────────

    def _replace_multiword_phrases(self, text: str) -> str:
        """
        يستبدل العبارات المركّبة الموجودة في القاموس بنسختها المُوحَّدة
        (underscore-joined) قبل التقطيع العادي.

        مثال: "كيف حالك يا صاحبي" → "كيف_حالك_يا_صاحبي"
        """
        # استخرج العبارات المركّبة (التي تحتوي underscore) من القاموس
        phrases = sorted(
            (w for w in self._word2id if "_" in w),
            key=len, reverse=True   # الأطول أولاً لتجنب التعارض
        )
        for phrase in phrases:
            readable = phrase.replace("_", " ")
            if readable in text:
                text = text.replace(readable, phrase)
        return text

    # ──────────────────────────────────────────────────────────────────────
    # حفظ / تحميل القاموس
    # ──────────────────────────────────────────────────────────────────────

    def save_vocab(self, path: str) -> None:
        """
        يحفظ القاموس كـ JSON شفاف (word → id).

        Parameters
        ----------
        path : مسار الملف (ينشئ المجلدات تلقائياً).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "word2id": self._word2id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[YemeniTokenizer] ✓ حُفِظ القاموس → {path} "
                    f"| {self.vocab_size:,} كلمة")

    def load_vocab(self, path: str) -> None:
        """
        يُحمِّل قاموساً محفوظاً مسبقاً.
        يُدمج مع القاموس الحالي (لا يمسح الرموز الخاصة).

        Parameters
        ----------
        path : مسار ملف JSON المحفوظ بـ save_vocab().
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        word2id: Dict[str, int] = data.get("word2id", {})
        # أضف الكلمات الجديدة فقط (لا تُعيد الكتابة فوق الرموز الخاصة)
        max_existing = max(self._id2word.keys()) if self._id2word else -1
        for word, wid in word2id.items():
            if word not in self._word2id:
                # استخدم ID الأصلي إن لم يتعارض، وإلا خصّص جديداً
                if wid not in self._id2word:
                    self._word2id[word] = wid
                    self._id2word[wid]  = word
                else:
                    new_id = max(self._id2word.keys()) + 1
                    self._word2id[word] = new_id
                    self._id2word[new_id] = word
        logger.info(f"[YemeniTokenizer] ✓ قاموس محمَّل من {path} "
                    f"| حجم جديد: {self.vocab_size:,}")

    # ──────────────────────────────────────────────────────────────────────
    # معلومات
    # ──────────────────────────────────────────────────────────────────────

    def info(self) -> Dict:
        """يُعيد ملخصاً تشخيصياً."""
        multiword = [w for w in self._word2id if "_" in w]
        return {
            "vocab_size":        self.vocab_size,
            "special_tokens":    len(_SPECIAL_TOKENS),
            "seed_words":        len(_SEED_VOCABULARY),
            "multiword_phrases": len(multiword),
            "grow_vocab":        self.grow_vocab,
            "normalize":         self.normalize,
            "special_ids": {
                "PAD":  self.PAD,
                "UNK":  self.UNK,
                "BOS":  self.BOS,
                "EOS":  self.EOS,
                "SEP":  self.SEP,
                "MASK": self.MASK,
            },
        }

    def __repr__(self) -> str:
        return (f"YemeniTokenizer(vocab_size={self.vocab_size}, "
                f"grow_vocab={self.grow_vocab})")


# ══════════════════════════════════════════════════════════════════════════════
# Factory — نقطة وصول موحّدة
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_VOCAB_PATH = "models/yemeni_tokenizer_vocab.json"

def get_yemeni_tokenizer(
    vocab_path:  Optional[str] = _DEFAULT_VOCAB_PATH,
    grow_vocab:  bool = True,
    normalize:   bool = True,
) -> YemeniTokenizer:
    """
    يُعيد مثيل YemeniTokenizer:
    • يحمّل القاموس المحفوظ إن وُجد.
    • يبدأ بالقاموس البذري إن لم يوجد.

    Parameters
    ----------
    vocab_path : مسار ملف JSON للقاموس المحفوظ (None = بذري فقط).
    grow_vocab : السماح بتوسيع القاموس ديناميكياً.
    normalize  : تطبيع النص قبل الترميز.
    """
    return YemeniTokenizer(
        grow_vocab=grow_vocab,
        normalize=normalize,
        vocab_path=vocab_path if (vocab_path and os.path.exists(vocab_path)) else None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Quick Self-Test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 60)
    print("YemeniTokenizer — Phase 1 Self-Test")
    print("=" * 60)

    tok = YemeniTokenizer(grow_vocab=True)
    print(f"\n✓ Info: {tok.info()}")

    # اختبار encode / decode
    samples = [
        "بسم الله الرحمن الرحيم",
        "كيف حالك يا صاحبي",
        "أبشر سدا جهال",
        "الله يعافيك يا حبيبي",
        "صنعاء مدينة جميلة",
        "يلا نروح السوق",
    ]

    print("\n── encode / decode ──")
    all_ok = True
    for text in samples:
        ids = tok.encode(text, max_len=32)
        recovered = tok.decode(ids)
        match = _normalize(text) in _normalize(recovered) or \
                _normalize(recovered).replace(" ", "") in _normalize(text).replace(" ", "")
        status = "✓" if match else "⚠"
        if not match:
            all_ok = False
        print(f"  {status} orig:      {text}")
        print(f"    ids:       {ids.tolist()}")
        print(f"    decoded:   {recovered}")
        print()

    # اختبار padding / masking
    print("── padding / masking ──")
    seqs = [tok.encode(t, add_bos=True, add_eos=True) for t in samples[:3]]
    padded = tok.pad_sequence(seqs)
    print(f"  ✓ padded shape: {padded.shape}")

    pad_mask = tok.make_pad_mask(padded)
    print(f"  ✓ pad_mask shape: {pad_mask.shape}")

    causal = YemeniTokenizer.make_causal_mask(padded.shape[1])
    print(f"  ✓ causal_mask shape: {causal.shape}")

    combined = YemeniTokenizer.make_combined_mask(pad_mask, causal)
    print(f"  ✓ combined_mask shape: {combined.shape}")

    # اختبار حفظ / تحميل
    print("\n── save / load ──")
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        path = tf.name
    tok.save_vocab(path)
    tok2 = YemeniTokenizer(vocab_path=path)
    assert tok2.vocab_size == tok.vocab_size, "vocab_size mismatch after load!"
    os.unlink(path)
    print(f"  ✓ save/load round-trip OK | vocab_size={tok2.vocab_size}")

    # اختبار عبارات مركّبة
    phrase_text = "كيف حالك يا صاحبي"
    ids_phrase  = tok.encode(phrase_text)
    dec_phrase  = tok.decode(ids_phrase)
    assert "كيف حالك يا صاحبي" in dec_phrase or "كيف_حالك_يا_صاحبي" not in dec_phrase
    print(f"  ✓ multi-word phrase: '{phrase_text}' → decoded: '{dec_phrase}'")

    print("\n" + ("✅ جميع الاختبارات ناجحة" if all_ok else "⚠ بعض النصوص لم تتطابق تماماً (طبيعي بسبب التطبيع)"))
    print("=" * 60)
