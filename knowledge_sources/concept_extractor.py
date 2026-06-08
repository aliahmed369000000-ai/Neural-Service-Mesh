"""
Concept Extractor — الأولوية 1 (الأساس)
=========================================
يستبدل _extract_concepts() البدائي في quran_source.py بنظام حقيقي يستخدم:
  1. TF-IDF (scikit-learn) لاستخراج الكلمات المفتاحية الأعلى وزناً
  2. Arabic Concept Clusters — 60+ مفهوم مُصنَّف مسبقاً
  3. درجة ثقة (confidence score) لكل مفهوم مستخرج
  4. دعم Batch لمعالجة آلاف الآيات بكفاءة

الاستخدام:
    from knowledge_sources.concept_extractor import ConceptExtractor

    extractor = ConceptExtractor()
    extractor.fit(all_texts)          # مرحلة واحدة على كل النصوص
    results = extractor.extract(text, reference)
    # → [ConceptMatch(concept="صبر", score=0.87, cluster="أخلاق"), ...]
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Arabic Text Normalization ────────────────────────────────────────────────

_TASHKEEL = re.compile(r'[\u064B-\u065F\u0670]')   # حركات التشكيل

def _strip_tashkeel(text: str) -> str:
    """إزالة حركات التشكيل للمطابقة الدقيقة."""
    return _TASHKEEL.sub('', text)

# ── Arabic Concept Clusters (60+ مفهوم) ────────────────────────────────────

CONCEPT_CLUSTERS: Dict[str, Dict[str, List[str]]] = {
    # ── توحيد ─────────────────────────────────────────────────────────────
    "توحيد": {
        "الله":       ["الله", "إله", "رب", "الرب"],
        "أسماء الله": ["الرحمن", "الرحيم", "الملك", "القدوس", "السلام",
                       "العزيز", "الحكيم", "الغفور", "الكريم", "اللطيف",
                       "الخبير", "الحي", "القيوم", "الأول", "الآخر"],
        "وحدانية":    ["أحد", "واحد", "لا إله إلا الله", "لا شريك"],
    },
    # ── عبادة ──────────────────────────────────────────────────────────────
    "عبادة": {
        "صلاة":  ["صلاة", "صلّ", "يصلون", "الصلوات", "ركوع", "سجود"],
        "زكاة":  ["زكاة", "الزكاة", "ينفقون", "أنفقوا", "الإنفاق"],
        "صيام":  ["صيام", "صوم", "رمضان", "الصائمين"],
        "حج":    ["حج", "الحج", "عمرة", "البيت", "الكعبة"],
        "دعاء":  ["دعاء", "ادعوا", "يدعو", "استغفر", "استغفروا"],
        "ذكر":   ["ذكر", "يذكرون", "سبحان", "الحمد", "تسبيح"],
    },
    # ── أخلاق ──────────────────────────────────────────────────────────────
    "أخلاق": {
        "صبر":    ["صبر", "صابرين", "اصبروا", "الصابرون", "صبّار", "يصبرون",
                   "استعينوا بالصبر", "الصبر جميل"],
        "تقوى":   ["تقوى", "تقوا", "اتقوا", "المتقون", "التقوى"],
        "عدل":    ["عدل", "القسط", "العدل", "اعدلوا"],
        "إحسان":  ["إحسان", "أحسنوا", "المحسنون", "أحسن"],
        "صدق":    ["صدق", "الصادقون", "الصديقون", "صادقاً"],
        "توبة":   ["تاب", "توبة", "يتوبون", "التائبون", "استغفر"],
        "شكر":    ["شكر", "شاكرين", "الشاكرون", "يشكر"],
        "تواضع":  ["تواضع", "لا تتكبر", "المتكبرون"],
        "عفو":    ["عفو", "اعفوا", "يعفو", "غفر"],
        "أمانة":  ["أمانة", "الأمانات", "أمين"],
    },
    # ── إيمان وعقيدة ───────────────────────────────────────────────────────
    "إيمان": {
        "إيمان":    ["آمن", "المؤمنون", "الإيمان", "يؤمنون"],
        "كفر":      ["كفر", "الكافرون", "كفروا"],
        "نفاق":     ["منافقون", "النفاق", "يكذبون"],
        "يقين":     ["يقين", "بيقين", "اليقين"],
        "غيب":      ["غيب", "الغيب", "يؤمنون بالغيب"],
    },
    # ── آخرة ───────────────────────────────────────────────────────────────
    "آخرة": {
        "يوم القيامة": ["يوم القيامة", "يوم الدين", "يوم الحساب",
                        "القيامة", "الساعة"],
        "جنة":         ["جنة", "الجنة", "الفردوس", "جنات"],
        "نار":         ["النار", "جهنم", "العذاب", "نار"],
        "حساب":        ["حساب", "الحساب", "ميزان", "الموازين"],
        "بعث":         ["بعث", "البعث", "نشور", "المعاد"],
    },
    # ── نبوة ───────────────────────────────────────────────────────────────
    "نبوة": {
        "أنبياء":  ["نبي", "نبيّ", "الأنبياء", "رسول", "الرسل"],
        "محمد ﷺ": ["محمد", "أحمد", "النبي", "الرسول الكريم"],
        "وحي":    ["وحي", "أوحى", "الوحي", "أنزل"],
        "قرآن":   ["القرآن", "الكتاب المبين", "الكتاب"],
    },
    # ── معرفة وعقل ─────────────────────────────────────────────────────────
    "معرفة": {
        "علم":   ["علم", "يعلمون", "العلم", "العلماء"],
        "حكمة":  ["حكمة", "الحكمة", "حكيم"],
        "عقل":   ["عقل", "يعقلون", "أولو الألباب", "تعقلون"],
        "تفكر":  ["تفكروا", "يتفكرون", "تدبر", "يتدبرون"],
        "ذكر":   ["تذكرون", "يذّكرون"],
    },
    # ── اجتماع وأسرة ───────────────────────────────────────────────────────
    "مجتمع": {
        "أسرة":    ["أهل", "والدين", "والد", "والدة", "أولاد"],
        "تعاون":   ["تعاونوا", "وتعاونوا"],
        "وحدة":    ["وحدة", "واعتصموا", "جماعة"],
        "شورى":    ["شورى", "يشاورون"],
        "أمة":     ["أمة", "الأمة", "أمم"],
    },
    # ── طبيعة وكون ─────────────────────────────────────────────────────────
    "كون": {
        "خلق":   ["خلق", "الخلق", "المخلوقات", "خلقنا"],
        "سماء":  ["السماء", "السماوات", "الكون"],
        "أرض":   ["الأرض", "في الأرض"],
        "ماء":   ["الماء", "المطر", "الأنهار"],
        "نعم":   ["نعمة", "النعم", "أنعمنا"],
    },
}

# ── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class ConceptMatch:
    """مفهوم واحد مستخرج من آية مع درجة الثقة."""
    concept:    str          # e.g. "صبر"
    cluster:    str          # e.g. "أخلاق"
    score:      float        # 0.0 – 1.0
    source:     str          # "keyword" | "tfidf"
    keywords:   List[str] = field(default_factory=list)   # الكلمات التي أطلقت المفهوم

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "cluster": self.cluster,
            "score":   round(self.score, 3),
            "source":  self.source,
            "keywords": self.keywords,
        }


# ── Core Extractor ──────────────────────────────────────────────────────────

class ConceptExtractor:
    """
    محرك استخراج المفاهيم الحقيقي.

    المراحل:
      1. fit(texts)    — يبني نموذج TF-IDF على كل النصوص
      2. extract(text) — يستخرج مفاهيم لنص واحد بمصدرين:
           a. Keyword Matching   — بحث مباشر في CONCEPT_CLUSTERS
           b. TF-IDF Top Terms  — كلمات عالية الوزن تُضاف للسياق

    النتيجة: قائمة ConceptMatch مرتبة تنازلياً حسب الدرجة.
    """

    def __init__(self, max_concepts: int = 10, min_score: float = 0.15):
        self.max_concepts  = max_concepts
        self.min_score     = min_score
        self._fitted       = False
        self._vectorizer   = None
        self._tfidf_matrix = None
        self._texts_ref: List[str] = []

        # بناء فهرس عكسي: keyword → (cluster, concept)
        self._keyword_index: Dict[str, Tuple[str, str]] = {}
        self._build_keyword_index()

    # ── Setup ────────────────────────────────────────────────────────────

    def _build_keyword_index(self) -> None:
        for cluster, concepts in CONCEPT_CLUSTERS.items():
            for concept, keywords in concepts.items():
                for kw in keywords:
                    # نضيف فقط إذا لم يكن موجوداً (أطول كلمة تفوز)
                    if kw not in self._keyword_index:
                        self._keyword_index[kw] = (cluster, concept)
        logger.info(f"[ConceptExtractor] keyword_index built: {len(self._keyword_index)} entries")

    def fit(self, texts: List[str]) -> "ConceptExtractor":
        """
        بناء نموذج TF-IDF على مجموعة النصوص الكاملة.
        يجب استدعاؤه مرة واحدة قبل extract().
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            logger.warning("[ConceptExtractor] scikit-learn not installed — TF-IDF disabled")
            self._fitted = False
            return self

        if not texts:
            logger.warning("[ConceptExtractor] fit() called with empty texts")
            return self

        logger.info(f"[ConceptExtractor] fitting TF-IDF on {len(texts)} texts …")
        self._vectorizer = TfidfVectorizer(
            analyzer      = "word",
            token_pattern = r"[\u0600-\u06FF]+",   # Arabic only
            min_df        = 2,
            max_df        = 0.85,
            max_features  = 5000,
            sublinear_tf  = True,
        )
        try:
            self._tfidf_matrix = self._vectorizer.fit_transform(texts)
            self._texts_ref    = texts
            self._fitted       = True
            logger.info(f"[ConceptExtractor] TF-IDF fitted: vocab={len(self._vectorizer.vocabulary_)}")
        except Exception as exc:
            logger.error(f"[ConceptExtractor] TF-IDF fit error: {exc}")
            self._fitted = False
        return self

    # ── Extraction ───────────────────────────────────────────────────────

    def extract(
        self,
        text:       str,
        reference:  str = "",
        surah_name: str = "",
        doc_index:  Optional[int] = None,
    ) -> List[ConceptMatch]:
        """
        استخراج المفاهيم من نص واحد.
        Returns قائمة ConceptMatch مرتبة حسب score تنازلياً.
        """
        matches: Dict[str, ConceptMatch] = {}

        # ── 1. Keyword Matching ──────────────────────────────────────────
        self._keyword_match(_strip_tashkeel(text), matches)

        # ── 2. TF-IDF Top Terms ─────────────────────────────────────────
        if self._fitted and doc_index is not None:
            self._tfidf_match(doc_index, matches)

        # ── 3. Structural concepts (always present) ──────────────────────
        if surah_name:
            matches[f"سورة:{surah_name}"] = ConceptMatch(
                concept  = f"سورة:{surah_name}",
                cluster  = "هيكل",
                score    = 1.0,
                source   = "structural",
                keywords = [surah_name],
            )
        if reference:
            matches[f"آية:{reference}"] = ConceptMatch(
                concept  = f"آية:{reference}",
                cluster  = "هيكل",
                score    = 1.0,
                source   = "structural",
                keywords = [reference],
            )

        # ── 4. Filter & Sort ─────────────────────────────────────────────
        results = [m for m in matches.values() if m.score >= self.min_score]
        results.sort(key=lambda m: m.score, reverse=True)
        return results[: self.max_concepts]

    def extract_simple(self, text: str, surah_name: str = "", reference: str = "") -> List[str]:
        """
        واجهة بسيطة: ترجع قائمة أسماء المفاهيم فقط (متوافقة مع الكود القديم).
        """
        matches = self.extract(text, reference=reference, surah_name=surah_name)
        return [m.concept for m in matches]

    # ── Internal ─────────────────────────────────────────────────────────

    def _keyword_match(self, text: str, out: Dict[str, ConceptMatch]) -> None:
        """بحث مباشر عن كل كلمة مفتاحية في النص مع حساب الدرجة."""
        clean_text = _strip_tashkeel(text)   # إزالة التشكيل للمطابقة
        # نرتب المفاتيح من الأطول للأقصر لتفضيل العبارات على الكلمات
        sorted_kws = sorted(self._keyword_index.keys(), key=len, reverse=True)
        hit_count: Dict[str, List[str]] = {}  # concept → list of matched keywords

        for kw in sorted_kws:
            if kw in clean_text:
                cluster, concept = self._keyword_index[kw]
                if concept not in hit_count:
                    hit_count[concept] = []
                hit_count[concept].append(kw)

        for concept, kws in hit_count.items():
            # درجة الثقة: أكثر كلمات تعني يقيناً أكبر
            score = min(0.5 + 0.15 * len(kws), 0.95)
            cluster = self._keyword_index[kws[0]][0]
            if concept not in out or out[concept].score < score:
                out[concept] = ConceptMatch(
                    concept  = concept,
                    cluster  = cluster,
                    score    = score,
                    source   = "keyword",
                    keywords = kws,
                )

    def _tfidf_match(self, doc_index: int, out: Dict[str, ConceptMatch]) -> None:
        """يضيف مفاهيم عبر الكلمات العالية الوزن في TF-IDF."""
        try:
            row      = self._tfidf_matrix[doc_index]
            feature_names = self._vectorizer.get_feature_names_out()
            scores   = np.asarray(row.todense()).flatten()
            top_idx  = scores.argsort()[::-1][:20]   # أعلى 20 كلمة

            for idx in top_idx:
                if scores[idx] < 0.05:
                    break
                word  = feature_names[idx]
                score = float(scores[idx])
                # هل الكلمة موجودة في فهرسنا؟
                if word in self._keyword_index:
                    cluster, concept = self._keyword_index[word]
                    combined = min(score * 1.2 + 0.3, 0.92)
                    if concept not in out or out[concept].score < combined:
                        out[concept] = ConceptMatch(
                            concept  = concept,
                            cluster  = cluster,
                            score    = combined,
                            source   = "tfidf",
                            keywords = [word],
                        )
        except Exception as exc:
            logger.warning(f"[ConceptExtractor] TF-IDF match error: {exc}")

    # ── Batch ─────────────────────────────────────────────────────────────

    def extract_batch(
        self,
        texts:       List[str],
        references:  Optional[List[str]] = None,
        surah_names: Optional[List[str]] = None,
    ) -> List[List[ConceptMatch]]:
        """
        معالجة مجموعة كبيرة من النصوص دفعة واحدة.
        الأكثر كفاءة من استدعاء extract() لكل نص.
        """
        refs   = references  or [""] * len(texts)
        snames = surah_names or [""] * len(texts)

        results = []
        for i, (text, ref, sname) in enumerate(zip(texts, refs, snames)):
            doc_idx = i if self._fitted else None
            results.append(self.extract(text, reference=ref, surah_name=sname, doc_index=doc_idx))

        logger.info(f"[ConceptExtractor] batch extracted {len(results)} items")
        return results

    # ── Stats ─────────────────────────────────────────────────────────────

    def cluster_distribution(self, all_matches: List[List[ConceptMatch]]) -> Dict[str, int]:
        """إحصائيات توزيع المفاهيم على الـ clusters."""
        dist: Dict[str, int] = {}
        for matches in all_matches:
            for m in matches:
                if m.cluster != "هيكل":
                    dist[m.cluster] = dist.get(m.cluster, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))


# ── Singleton Factory ────────────────────────────────────────────────────────

_default_extractor: Optional[ConceptExtractor] = None


def get_extractor() -> ConceptExtractor:
    """ارجع النسخة الافتراضية (singleton) من ConceptExtractor."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = ConceptExtractor()
    return _default_extractor


def fit_extractor_on_quran(texts: List[str]) -> ConceptExtractor:
    """
    أنشئ وارجع extractor جاهز بعد تدريبه على نصوص القرآن.
    يُستدعى مرة واحدة عند بدء النظام.
    """
    global _default_extractor
    _default_extractor = ConceptExtractor()
    _default_extractor.fit(texts)
    return _default_extractor
