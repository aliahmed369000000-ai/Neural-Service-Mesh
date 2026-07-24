"""
Generic Concept Extractor — إضافي بالكامل، لا يمسّ concept_extractor.py
=========================================================================
نسخة عامة من ConceptExtractor لأي دومين عربي (مو الإسلاميات فقط):
بدل الاعتماد على CONCEPT_CLUSTERS المُعرَّفة يدوياً (60+ مفهوم إسلامي
مكتوب باليد)، هذا المستخرج يكتشف العناقيد تلقائياً من النص نفسه عبر:

  1. TF-IDF على كل مستندات الدومين (نفس أسلوب concept_extractor.py)
  2. KMeans على متجهات المستندات → يكتشف K عنقود موضوعي تلقائياً
  3. اسم كل عنقود يُشتق من أعلى كلمتين وزناً في مركز العنقود (تلقائياً،
     بدون تدخل يدوي)
  4. لكل مستند: أعلى كلمات TF-IDF فيه = "مفاهيمه"، وعنقوده = عنقود
     المستند الذي حُدِّد بالخطوة 2

الفائدة: يعمل على أي دومين جديد (ويكيبيديا، أدب، تاريخ...) بدون كتابة
قوائم كلمات مفتاحية يدوية لكل دومين — فقط أعطه نصوصاً وسيكتشف البنية.

التوافق: يُنتج نفس ConceptMatch المستخدمة في concept_extractor.py،
ونفس شكل extract_batch()، فهو بديل قابل للتركيب (drop-in) في نفس
مسار ingest_batch() الموجود في cognitive_graph.py دون أي تعديل هناك.

الاستخدام:
    from knowledge_sources.generic_concept_extractor import GenericConceptExtractor

    extractor = GenericConceptExtractor(n_clusters=12)
    extractor.fit(all_texts)
    all_matches = extractor.extract_batch(texts, references=refs, group_names=groups)
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import numpy as np

# نعيد استخدام ConceptMatch نفسها (لا تكرار، توافق كامل مع بقية النظام)
from knowledge_sources.concept_extractor import ConceptMatch

logger = logging.getLogger(__name__)

_TASHKEEL = re.compile(r'[\u064B-\u065F\u0670]')


def _strip_tashkeel(text: str) -> str:
    return _TASHKEEL.sub('', text)


# كلمات توقف عربية شائعة (أدوات ربط/ظرف/ضمائر) — تُستبعد من TF-IDF حتى لا
# تتصدر أسماء العناقيد أو المفاهيم المستخرجة بكلمات وظيفية لا معنى دلالياً لها.
ARABIC_STOPWORDS: List[str] = [
    "من", "في", "على", "إلى", "عن", "مع", "بين", "دون", "عند", "حتى",
    "كان", "كانت", "كانوا", "يكون", "تكون", "أصبح", "أصبحت", "صار",
    "الذي", "التي", "الذين", "اللذان", "اللتان", "هذا", "هذه", "ذلك", "تلك",
    "أن", "إن", "لا", "ما", "لم", "لن", "قد", "لقد", "كل", "بعض", "غير",
    "ثم", "أو", "إذا", "حيث", "هو", "هي", "هم", "أنت", "أنا", "نحن",
    "كما", "بعد", "قبل", "فوق", "تحت", "أيضا", "أيضاً", "إلا", "لكن",
    "لكنه", "لكنها", "كانا", "يوجد", "توجد", "وجود", "غيره", "غيرها",
    "عشر", "عشرة", "آلاف", "ألف", "مئة", "مائة", "أكبر", "أصغر", "بشكل",
    "مناطق", "منطقة", "خلال", "عبر", "نحو", "حول", "لها", "له", "به", "بها",
]


def _tokenize_ar(text: str) -> List[str]:
    """
    استخراج كلمات عربية نظيفة: نطاق [\\u0621-\\u064A] (حروف عربية فقط،
    يستبعد الفاصلة العربية وعلامات الترقيم الواقعة في 0600-0620 والتشكيل
    في 064B-065F)، مع تجريد بادئتي "و" و"ال" الشائعتين لتقليل تكرار نفس
    الكلمة بصيغتين مختلفتين (و+تعد مقابل تعد).
    """
    tokens = re.findall(r"[\u0621-\u064A]+", text)
    out = []
    for t in tokens:
        if t.startswith("وال") and len(t) > 5:
            t = t[3:]
        elif t.startswith("و") and len(t) > 3:
            t = t[1:]
        elif t.startswith("ال") and len(t) > 4:
            t = t[2:]
        if len(t) > 1 and t not in ARABIC_STOPWORDS:
            out.append(t)
    return out


class GenericConceptExtractor:
    """
    مستخرج مفاهيم عام بتصنيف تلقائي (بدون قوائم كلمات يدوية).

    المراحل:
      1. fit(texts)     — TF-IDF + KMeans لاكتشاف K عنقود موضوعي تلقائياً
      2. extract_batch  — لكل مستند: أعلى كلمات TF-IDF (=مفاهيمه)
                           + عنقوده المكتشَف (=تصنيفه)
    """

    def __init__(
        self,
        max_concepts_per_doc: int = 8,
        min_score: float = 0.12,
        n_clusters: int = 12,
        terms_per_cluster_label: int = 2,
    ):
        self.max_concepts_per_doc = max_concepts_per_doc
        self.min_score = min_score
        self.n_clusters = n_clusters
        self.terms_per_cluster_label = terms_per_cluster_label

        self._fitted = False
        self._vectorizer = None
        self._tfidf_matrix = None
        self._doc_cluster: Optional[np.ndarray] = None
        self._cluster_labels: Dict[int, str] = {}

    # ── Setup ────────────────────────────────────────────────────────────
    def fit(self, texts: List[str]) -> "GenericConceptExtractor":
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
        except ImportError:
            logger.warning("[GenericConceptExtractor] scikit-learn غير مثبت — التصنيف معطّل")
            self._fitted = False
            return self

        if not texts:
            logger.warning("[GenericConceptExtractor] fit() استُدعيت بنصوص فارغة")
            return self

        clean_texts = [_strip_tashkeel(t) for t in texts]

        logger.info(f"[GenericConceptExtractor] تدريب TF-IDF على {len(texts)} نص …")
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=_tokenize_ar,
            preprocessor=lambda x: x,   # نعطّل المعالجة المسبقة الافتراضية (lowercase غير مفيد للعربية)
            token_pattern=None,
            min_df=2,
            max_df=0.9,
            max_features=8000,
            sublinear_tf=True,
        )
        try:
            self._tfidf_matrix = self._vectorizer.fit_transform(clean_texts)
        except Exception as exc:
            logger.error(f"[GenericConceptExtractor] فشل TF-IDF: {exc}")
            self._fitted = False
            return self

        # عدد العناقيد لا يتجاوز عدد المستندات
        k = max(1, min(self.n_clusters, len(texts)))
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            self._doc_cluster = km.fit_predict(self._tfidf_matrix)
            feature_names = self._vectorizer.get_feature_names_out()
            for cid, center in enumerate(km.cluster_centers_):
                top_idx = center.argsort()[::-1][: self.terms_per_cluster_label]
                top_terms = [feature_names[i] for i in top_idx if center[i] > 0]
                label = "_".join(top_terms) if top_terms else f"عنقود_{cid}"
                self._cluster_labels[cid] = label
            logger.info(
                f"[GenericConceptExtractor] اكتُشف {k} عنقود: "
                f"{list(self._cluster_labels.values())}"
            )
        except Exception as exc:
            logger.error(f"[GenericConceptExtractor] فشل KMeans: {exc}")
            self._doc_cluster = np.zeros(len(texts), dtype=int)
            self._cluster_labels = {0: "عام"}

        self._fitted = True
        return self

    # ── Extraction ───────────────────────────────────────────────────────
    def extract(
        self,
        doc_index: int,
        reference: str = "",
        group_name: str = "",
    ) -> List[ConceptMatch]:
        matches: Dict[str, ConceptMatch] = {}

        if self._fitted and doc_index is not None:
            self._tfidf_top_terms(doc_index, matches)

        # عقد هيكلية (نفس اصطلاح "هيكل" المستخدم في concept_extractor.py
        # — تُستبعد تلقائياً من العلاقات الدلالية عبر ingest_from_concept_matches)
        if group_name:
            matches[f"مجموعة:{group_name}"] = ConceptMatch(
                concept=f"مجموعة:{group_name}", cluster="هيكل",
                score=1.0, source="structural", keywords=[group_name],
            )
        if reference:
            matches[f"مرجع:{reference}"] = ConceptMatch(
                concept=f"مرجع:{reference}", cluster="هيكل",
                score=1.0, source="structural", keywords=[reference],
            )

        results = [m for m in matches.values() if m.score >= self.min_score or m.cluster == "هيكل"]
        results.sort(key=lambda m: m.score, reverse=True)
        return results[: self.max_concepts_per_doc + 2]  # +2 هامش للعقد الهيكلية

    def extract_batch(
        self,
        texts: List[str],
        references: Optional[List[str]] = None,
        group_names: Optional[List[str]] = None,
    ) -> List[List[ConceptMatch]]:
        refs = references or [""] * len(texts)
        groups = group_names or [""] * len(texts)

        results = []
        for i in range(len(texts)):
            doc_idx = i if self._fitted else None
            results.append(self.extract(doc_idx, reference=refs[i], group_name=groups[i]))
        logger.info(f"[GenericConceptExtractor] batch استخرج {len(results)} عنصر")
        return results

    # ── Internal ─────────────────────────────────────────────────────────
    def _tfidf_top_terms(self, doc_index: int, out: Dict[str, ConceptMatch]) -> None:
        try:
            row = self._tfidf_matrix[doc_index]
            feature_names = self._vectorizer.get_feature_names_out()
            scores = np.asarray(row.todense()).flatten()
            top_idx = scores.argsort()[::-1][: self.max_concepts_per_doc]

            cluster_id = int(self._doc_cluster[doc_index]) if self._doc_cluster is not None else 0
            cluster_label = self._cluster_labels.get(cluster_id, "عام")

            for idx in top_idx:
                if scores[idx] < self.min_score:
                    break
                word = feature_names[idx]
                score = float(min(scores[idx] + 0.3, 0.97))  # نفس منطق تعزيز الدرجة في concept_extractor.py
                out[word] = ConceptMatch(
                    concept=word, cluster=cluster_label,
                    score=score, source="tfidf_auto", keywords=[word],
                )
        except Exception as exc:
            logger.warning(f"[GenericConceptExtractor] خطأ TF-IDF: {exc}")

    # ── Stats ─────────────────────────────────────────────────────────────
    def cluster_distribution(self, all_matches: List[List[ConceptMatch]]) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for matches in all_matches:
            for m in matches:
                if m.cluster != "هيكل":
                    dist[m.cluster] = dist.get(m.cluster, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))
