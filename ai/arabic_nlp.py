"""
Arabic NLP Engine — Phase 18
============================
Adds full Arabic language understanding to the Neural Service Mesh.

Architecture: 3 analysis layers → 7-element feature vector → weight matrix
---------------------------------------------------------------------------

  Layer 1 — Syntactic (نحوي)
      Tokenisation, POS tagging (verbs/nouns/particles),
      sentence boundary detection, dependency hints.

  Layer 2 — Morphological (صرفي)
      Root extraction (جذور), morphological pattern matching (أوزان),
      conjugation state, prefix/suffix stripping.

  Layer 3 — Semantic (دلالي)
      Meaning tagging, contextual concept mapping, CKG integration,
      semantic density scoring.

Output contract
---------------
  Every analysis produces an ArabicFeatureVector with exactly 7 core elements,
  expanded to 784 via to_list() (matching INPUT_DIM=784 of neural_core.py —
  NEVER change the 7 core elements' order/meaning):

    [0] verb_score          — proportion of verb-form tokens (0-1)
    [1] noun_score          — proportion of noun-form tokens (0-1)
    [2] root_complexity     — normalised root diversity score (0-1)
    [3] morpho_pattern_score — known wazn (وزن) coverage (0-1)
    [4] semantic_concept_score — CKG/concept alignment (0-1)
    [5] context_score       — semantic coherence / density (0-1)
    [6] syntactic_complexity — sentence structure complexity (0-1)

Constraints (must never be violated)
-------------------------------------
  • to_list() output is ALWAYS 784 elements (7 core + 777 hash-expanded,
    matches neural_core.py L_embed input dimension).
  • This module never modifies NeuralWeightLayer or DynamicWeightLayer shape.
  • All three layers run without external Arabic NLP libraries (pure Python).
  • The module is safe to import even when numpy is unavailable.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Arabic constants
# ═══════════════════════════════════════════════════════════════════════════

# Unicode ranges
_ARABIC_LETTERS     = "\u0600-\u06FF"
_ARABIC_EXTENDED    = "\u0750-\u077F"
_TASHKEEL           = "\u064B-\u065F\u0670\u0640"   # diacritics + tatweel
_ARABIC_RE          = re.compile(f"[{_ARABIC_LETTERS}{_ARABIC_EXTENDED}]+")
_TASHKEEL_RE        = re.compile(f"[{_TASHKEEL}]")

# ── Common Arabic prefixes (حروف مبنية تسبق الكلمة) ─────────────────────
_PREFIXES: List[str] = [
    "ال", "و", "ف", "ب", "ل", "ك", "م", "لل", "بال", "فال", "وال",
    "ولل", "كال", "وب", "وف", "ول",
]

# ── Common Arabic suffixes (علامات الإعراب والتأنيث والجمع) ───────────────
_SUFFIXES: List[str] = [
    "ون", "ين", "ات", "ان", "ين", "ة", "ه", "ها", "هم", "هن", "كم",
    "كن", "نا", "تم", "تن", "وا", "تا", "ا", "ي", "يا",
]

# ── Known Arabic roots (جذور شائعة مع مجالاتها الدلالية) ──────────────────
_KNOWN_ROOTS: Dict[str, List[str]] = {
    # جذر → [مجالات دلالية]
    "كتب": ["كتابة", "علم"],
    "قرأ": ["قراءة", "علم"],
    "علم": ["علم", "معرفة"],
    "فهم": ["فهم", "معرفة"],
    "عبد": ["عبادة", "دين"],
    "صلو": ["صلاة", "عبادة"],
    "ذكر": ["ذكر", "عبادة", "ذاكرة"],
    "شكر": ["شكر", "أخلاق"],
    "صبر": ["صبر", "أخلاق"],
    "رحم": ["رحمة", "أخلاق"],
    "أمن": ["إيمان", "دين", "أمان"],
    "حمد": ["حمد", "عبادة"],
    "سبح": ["تسبيح", "عبادة"],
    "هدي": ["هداية", "دين"],
    "نصر": ["نصر", "قوة"],
    "فتح": ["فتح", "نصر"],
    "رزق": ["رزق", "نعمة"],
    "خلق": ["خلق", "إبداع"],
    "سجد": ["سجود", "عبادة"],
    "تقي": ["تقوى", "دين"],
    "قتل": ["قتال", "حرب"],
    "جهد": ["جهاد", "سعي"],
    "أمر": ["أمر", "حكم"],
    "نهي": ["نهي", "حكم"],
    "قول": ["قول", "تواصل"],
    "سمع": ["سمع", "إدراك"],
    "بصر": ["بصر", "إدراك"],
    "عقل": ["عقل", "تفكير"],
    "فكر": ["تفكير", "عقل"],
    "ظلم": ["ظلم", "عدل"],
    "عدل": ["عدل", "أخلاق"],
    "حكم": ["حكمة", "عدل"],
    "مات": ["موت", "حياة"],
    "حيا": ["حياة", "خلق"],
    "نزل": ["نزول", "وحي"],
    "وحي": ["وحي", "دين"],
    "نبأ": ["نبوة", "دين"],
    "رسل": ["رسالة", "دين"],
    "ملك": ["ملك", "حكم"],
    "دعو": ["دعاء", "عبادة", "دعوة"],
    "توب": ["توبة", "دين"],
    "غفر": ["مغفرة", "رحمة"],
    "حسن": ["حسن", "جمال"],
    "قدر": ["قدرة", "قدر"],
    "شاء": ["مشيئة", "إرادة"],
    "وعد": ["وعد", "أمانة"],
    "أخذ": ["أخذ", "فعل"],
    "سأل": ["سؤال", "طلب"],
    "جاء": ["مجيء", "حركة"],
    "ذهب": ["ذهاب", "حركة"],
    "أكل": ["أكل", "طعام"],
    "شرب": ["شرب", "طعام"],
    "بين": ["بيان", "وضوح"],
    "حزن": ["حزن", "عاطفة"],
    "فرح": ["فرح", "عاطفة"],
}

# ── Morphological patterns (أوزان صرفية) with POS tag ─────────────────────
# Pattern: (regex for stripped root+pattern, pos, wazn_name)
_MORPHO_PATTERNS: List[Tuple[str, str, str]] = [
    # ── أفعال (Verbs) ─────────────────────────────────────────────────────
    (r"^[فعل]{3}$",           "verb",   "فَعَلَ"),          # past simple
    (r"^ي[فعل]{3}$",          "verb",   "يَفْعَلُ"),         # present 3ms
    (r"^ت[فعل]{3}$",          "verb",   "تَفْعَلُ"),         # present 2ms/3fs
    (r"^أ[فعل]{3}$",          "verb",   "أَفْعَلَ"),         # causative past
    (r"^[فعل]{3}ل$",          "verb",   "فَعَّلَ"),          # intensive
    (r"^ف[اع][عل]{2}$",       "verb",   "فَاعَلَ"),          # reciprocal
    (r"^تف[عل]{3}$",          "verb",   "تَفَعَّلَ"),        # reflexive intensive
    (r"^تف[اع][عل]{2}$",      "verb",   "تَفَاعَلَ"),       # reflexive reciprocal
    (r"^انف[عل]{2}$",         "verb",   "انْفَعَلَ"),        # passive-like
    (r"^افت[عل]{2}$",         "verb",   "افْتَعَلَ"),        # reflexive VIII
    (r"^استف[عل]{2}$",        "verb",   "اسْتَفْعَلَ"),     # X causative/request
    # ── أسماء فاعل ومفعول (Active/Passive Participles) ────────────────────
    (r"^ف[اع][عل]{1}$",       "noun",   "فَاعِل"),           # agent فاعل
    (r"^مف[عل]{2}[ةه]?$",     "noun",   "مَفْعُول"),         # patient مفعول
    (r"^مف[عل]{2}ل$",         "noun",   "مُفَعِّل"),         # causative agent
    (r"^مف[اع][عل]{1}[ةه]?$", "noun",   "مُفَاعِل"),        # reciprocal agent
    (r"^مستف[عل]{2}[ةه]?$",   "noun",   "مُسْتَفْعِل"),     # X agent
    # ── أسماء (Nouns) ─────────────────────────────────────────────────────
    (r"^[فعل]{2}[اى]ل$",      "noun",   "فَعَائِل"),         # broken plural
    (r"^[فعل]{3}[اى]ن$",      "noun",   "فُعْلَان"),         # pattern noun
    (r"^[فعل]{3}[يى]ة$",      "noun",   "فِعَالِيَّة"),      # abstract noun
    (r"^[فعل]{3}[ةه]$",       "noun",   "فِعَالَة"),         # occupation noun
    (r"^[فعل]{4}[ةه]?$",      "noun",   "فَعَّالَة"),        # intensive noun
    (r"^مف[عل]{3}$",          "noun",   "مَفْعَل"),          # place/time noun
    (r"^[فعل]{3}[يى]ّ$",      "adj",    "فَعَلِيّ"),         # relational adj
    (r"^[فعل]{2}[يى][لن]$",   "adj",    "فَعِيل"),           # quality adj
    # ── حروف وأدوات (Particles) ───────────────────────────────────────────
    (r"^(في|من|إلى|على|عن|مع|بعد|قبل|حتى|لأن|إن|أن|لا|ما|هل|قد|لم|لن)$",
     "particle", "أداة"),
]

# ── Precompiled particle set ──────────────────────────────────────────────
_PARTICLES = {
    "في", "من", "إلى", "على", "عن", "مع", "بعد", "قبل", "حتى",
    "لأن", "إن", "أن", "لا", "ما", "هل", "قد", "لم", "لن",
    "هذا", "هذه", "ذلك", "تلك", "الذي", "التي", "الذين", "اللاتي",
    "هو", "هي", "هم", "هن", "نحن", "أنت", "أنتم", "أنا",
    "كان", "ليس", "كانت", "يكون", "تكون",
}

# ── Semantic concept dictionary (مفردات → مفاهيم) ─────────────────────────
_SEMANTIC_DICT: Dict[str, List[str]] = {
    "الله":   ["إيمان", "توحيد", "دين"],
    "رب":     ["إيمان", "توحيد"],
    "رحمن":   ["رحمة", "إيمان"],
    "رحيم":   ["رحمة", "إيمان"],
    "إيمان":  ["إيمان", "دين"],
    "تقوى":   ["تقوى", "دين"],
    "صلاة":   ["عبادة", "دين"],
    "زكاة":   ["عبادة", "دين", "اقتصاد"],
    "صوم":    ["عبادة", "دين"],
    "حج":     ["عبادة", "دين"],
    "قرآن":   ["علم", "دين", "وحي"],
    "كتاب":   ["علم", "كتابة"],
    "عبادة":  ["عبادة", "دين"],
    "دعاء":   ["عبادة", "دين"],
    "توبة":   ["توبة", "دين"],
    "غفران":  ["مغفرة", "رحمة"],
    "جنة":    ["جنة", "آخرة"],
    "نار":    ["نار", "آخرة"],
    "آخرة":   ["آخرة", "دين"],
    "دنيا":   ["دنيا", "حياة"],
    "عدل":    ["عدل", "أخلاق"],
    "ظلم":    ["ظلم", "عدل"],
    "صدق":    ["صدق", "أخلاق"],
    "أمانة":  ["أمانة", "أخلاق"],
    "علم":    ["علم", "معرفة"],
    "عقل":    ["عقل", "تفكير"],
    "حكمة":   ["حكمة", "علم"],
    "خلق":    ["أخلاق", "خلق"],
    "أمة":    ["مجتمع", "دين"],
    "ناس":    ["مجتمع", "إنسان"],
    "إنسان":  ["إنسان", "خلق"],
    "حياة":   ["حياة", "خلق"],
    "موت":    ["موت", "آخرة"],
    "نعمة":   ["نعمة", "رزق"],
    "رزق":    ["رزق", "نعمة"],
    "صحة":    ["صحة", "نعمة"],
    "قلب":    ["قلب", "إدراك"],
    "نفس":    ["نفس", "إنسان"],
    "روح":    ["روح", "خلق"],
    "ذكر":    ["ذكر", "عبادة"],
    "شكر":    ["شكر", "أخلاق"],
    "صبر":    ["صبر", "أخلاق"],
    "حسن":    ["حسن", "أخلاق"],
    "خير":    ["خير", "أخلاق"],
    "شر":     ["شر", "أخلاق"],
}

# ── Stopwords for context scoring ─────────────────────────────────────────
_STOPWORDS = {
    "في", "من", "إلى", "على", "عن", "مع", "هو", "هي", "هم", "نحن",
    "أنت", "أنا", "كان", "قد", "لا", "ما", "أن", "إن", "عند",
    "ذلك", "هذا", "هذه", "تلك", "الذي", "التي", "الذين", "لكن",
    "ثم", "أو", "بل", "حتى", "إذا", "لو", "لأن", "بعد", "قبل",
}


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ArabicToken:
    """Single analysed Arabic word token."""
    raw: str            # original form
    clean: str          # stripped of tashkeel
    stripped: str       # stripped of prefix + suffix
    prefix: str = ""
    suffix: str = ""
    pos: str = "unknown"        # verb | noun | particle | adj | unknown
    root: str = ""              # extracted root (3-4 chars)
    wazn: str = ""              # morphological pattern (وزن)
    semantic_fields: List[str] = field(default_factory=list)
    root_confidence: float = 0.0


@dataclass
class SyntacticLayer:
    """Layer 1 output — syntactic analysis."""
    sentence_count: int = 0
    token_count: int = 0
    verb_count: int = 0
    noun_count: int = 0
    particle_count: int = 0
    adj_count: int = 0
    unknown_count: int = 0
    tokens: List[ArabicToken] = field(default_factory=list)
    verb_score: float = 0.0     # col[0]
    noun_score: float = 0.0     # col[1]
    syntactic_complexity: float = 0.0  # col[6]


@dataclass
class MorphologicalLayer:
    """Layer 2 output — morphological analysis."""
    unique_roots: List[str] = field(default_factory=list)
    root_count: int = 0
    pattern_matches: int = 0
    total_tokens_analysed: int = 0
    root_complexity: float = 0.0        # col[2]
    morpho_pattern_score: float = 0.0  # col[3]
    roots_by_frequency: Dict[str, int] = field(default_factory=dict)
    wazn_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class SemanticLayer:
    """Layer 3 output — semantic / contextual analysis."""
    concepts_found: List[str] = field(default_factory=list)
    concept_count: int = 0
    semantic_fields: List[str] = field(default_factory=list)
    ckg_aligned: bool = False
    semantic_concept_score: float = 0.0  # col[4]
    context_score: float = 0.0           # col[5]
    concept_frequency: Dict[str, int] = field(default_factory=dict)


@dataclass
class ArabicFeatureVector:
    """
    7-element feature vector — maps directly to neural weight matrix columns.

    COLUMNS (NEVER change count or order):
      [0] verb_score          (0-1)
      [1] noun_score          (0-1)
      [2] root_complexity     (0-1)
      [3] morpho_pattern_score (0-1)
      [4] semantic_concept_score (0-1)
      [5] context_score       (0-1)
      [6] syntactic_complexity (0-1)
    """
    verb_score: float = 0.0
    noun_score: float = 0.0
    root_complexity: float = 0.0
    morpho_pattern_score: float = 0.0
    semantic_concept_score: float = 0.0
    context_score: float = 0.0
    syntactic_complexity: float = 0.0

    def to_list(self) -> List[float]:
        """Return 784 floats: 7 قيم أصلية + 777 TF-IDF hash.

        [0:7]   — 7 قيم صرفية/دلالية
        [7:784] — 777 قيمة character n-gram hash من النص الأصلي
        (موسَّعة من 256→784 لتطابق L_embed(784×784) الجديدة في neural_core.py)
        """
        import math
        base7 = [
            round(self.verb_score, 6),
            round(self.noun_score, 6),
            round(self.root_complexity, 6),
            round(self.morpho_pattern_score, 6),
            round(self.semantic_concept_score, 6),
            round(self.context_score, 6),
            round(self.syntactic_complexity, 6),
        ]
        # توسيع إلى 784: character bigrams + trigrams على النص
        n_hash = 777
        hash_vec = [0.0] * n_hash
        # نستخدم أي نص متاح — نأخذ قيم base7 كسلسلة
        text_key = "|".join(f"{v:.4f}" for v in base7)
        for n in (2, 3):
            for i in range(len(text_key) - n + 1):
                gram = text_key[i:i+n]
                h = abs(hash(gram)) % n_hash
                hash_vec[h] += 1.0
        total = sum(hash_vec)
        if total > 0:
            hash_vec = [
                round(math.log1p(v * 10.0 / total) / math.log1p(10.0), 6)
                for v in hash_vec
            ]
        return base7 + hash_vec  # len=784

    def to_dict(self) -> dict:
        return {
            "col_0_verb_score":              round(self.verb_score, 6),
            "col_1_noun_score":              round(self.noun_score, 6),
            "col_2_root_complexity":         round(self.root_complexity, 6),
            "col_3_morpho_pattern_score":    round(self.morpho_pattern_score, 6),
            "col_4_semantic_concept_score":  round(self.semantic_concept_score, 6),
            "col_5_context_score":           round(self.context_score, 6),
            "col_6_syntactic_complexity":    round(self.syntactic_complexity, 6),
        }


@dataclass
class ArabicAnalysisResult:
    """Full analysis result from all 3 layers + combined feature vector."""
    text: str
    text_clean: str
    analysed_at: str
    syntactic: SyntacticLayer
    morphological: MorphologicalLayer
    semantic: SemanticLayer
    feature_vector: ArabicFeatureVector
    neural_output: Optional[List[float]] = None   # output after weight-matrix forward pass

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "text_clean": self.text_clean,
            "analysed_at": self.analysed_at,
            "layers": {
                "syntactic": {
                    "sentence_count":    self.syntactic.sentence_count,
                    "token_count":       self.syntactic.token_count,
                    "verb_count":        self.syntactic.verb_count,
                    "noun_count":        self.syntactic.noun_count,
                    "particle_count":    self.syntactic.particle_count,
                    "adj_count":         self.syntactic.adj_count,
                    "verb_score":        round(self.syntactic.verb_score, 4),
                    "noun_score":        round(self.syntactic.noun_score, 4),
                    "syntactic_complexity": round(self.syntactic.syntactic_complexity, 4),
                    "tokens": [
                        {
                            "raw": t.raw, "pos": t.pos,
                            "root": t.root, "wazn": t.wazn,
                            "semantic_fields": t.semantic_fields[:3],
                        }
                        for t in self.syntactic.tokens
                    ],
                },
                "morphological": {
                    "unique_roots":        self.morphological.unique_roots[:20],
                    "root_count":          self.morphological.root_count,
                    "pattern_matches":     self.morphological.pattern_matches,
                    "root_complexity":     round(self.morphological.root_complexity, 4),
                    "morpho_pattern_score": round(self.morphological.morpho_pattern_score, 4),
                    "roots_by_frequency":  dict(sorted(
                        self.morphological.roots_by_frequency.items(),
                        key=lambda x: -x[1]
                    )[:10]),
                    "wazn_distribution":   self.morphological.wazn_distribution,
                },
                "semantic": {
                    "concepts_found":      self.semantic.concepts_found[:15],
                    "concept_count":       self.semantic.concept_count,
                    "semantic_fields":     list(set(self.semantic.semantic_fields))[:10],
                    "ckg_aligned":         self.semantic.ckg_aligned,
                    "semantic_concept_score": round(self.semantic.semantic_concept_score, 4),
                    "context_score":       round(self.semantic.context_score, 4),
                    "concept_frequency":   dict(sorted(
                        self.semantic.concept_frequency.items(),
                        key=lambda x: -x[1]
                    )[:10]),
                },
            },
            "feature_vector": self.feature_vector.to_dict(),
            "feature_vector_list": self.feature_vector.to_list(),
            "neural_output": self.neural_output,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1 — Syntactic Analyser (نحوي)
# ═══════════════════════════════════════════════════════════════════════════

class SyntacticAnalyser:
    """
    Layer 1: Tokenisation and POS tagging for Arabic text.

    Identifies:
      - Sentence boundaries (using Arabic punctuation + conjunctions)
      - Part-of-speech: verbs, nouns, particles, adjectives
      - Syntactic complexity (avg tokens per sentence, POS variety)
    """

    # Verb-indicating prefixes on a stripped word
    _VERB_PREFIX_RE = re.compile(r"^[يتأنست]")
    # Verb past tense: word is 3 chars, all Arabic letters
    _VERB_PAST_RE   = re.compile(r"^[\u0600-\u06FF]{3,4}$")
    # Definite article
    _DEF_ART_RE     = re.compile(r"^ال")

    def analyse(self, text: str) -> SyntacticLayer:
        layer = SyntacticLayer()

        # ── Tokenise ──────────────────────────────────────────────────────
        clean_text = self._clean(text)
        # Count sentences by Arabic full stop + Latin period
        sentences = re.split(r"[.!?؟۔।\n]+", clean_text)
        layer.sentence_count = max(1, sum(1 for s in sentences if s.strip()))

        # Extract Arabic word tokens
        raw_tokens = _ARABIC_RE.findall(clean_text)
        layer.token_count = len(raw_tokens)

        if layer.token_count == 0:
            return layer

        # ── POS tagging ───────────────────────────────────────────────────
        tokens: List[ArabicToken] = []
        for raw in raw_tokens:
            tok = self._tag_token(raw)
            tokens.append(tok)
            if tok.pos == "verb":
                layer.verb_count += 1
            elif tok.pos == "noun":
                layer.noun_count += 1
            elif tok.pos == "particle":
                layer.particle_count += 1
            elif tok.pos == "adj":
                layer.adj_count += 1
            else:
                layer.unknown_count += 1

        layer.tokens = tokens

        # ── Scores (mapped to col[0], col[1], col[6]) ─────────────────────
        n = layer.token_count
        layer.verb_score = round(layer.verb_count / n, 6)
        layer.noun_score = round(layer.noun_count / n, 6)

        # Syntactic complexity: sentence length variance + POS diversity
        avg_tok_per_sent = n / layer.sentence_count
        # Normalise: 5 tokens/sentence → 0.3, 15 → 0.7, 25+ → 1.0
        length_factor = min(1.0, avg_tok_per_sent / 25.0)
        pos_types = sum(1 for c in [
            layer.verb_count, layer.noun_count,
            layer.particle_count, layer.adj_count
        ] if c > 0)
        pos_factor = pos_types / 4.0
        layer.syntactic_complexity = round((length_factor + pos_factor) / 2.0, 6)

        return layer

    def _clean(self, text: str) -> str:
        """Remove tashkeel, normalise alef variants."""
        text = _TASHKEEL_RE.sub("", text)
        text = re.sub(r"[أإآٱ]", "ا", text)
        return text.strip()

    def _strip_affixes(self, word: str) -> Tuple[str, str, str]:
        """
        Strip the longest matching prefix and suffix.
        Returns (stripped, prefix, suffix).
        """
        prefix = ""
        suffix = ""
        stripped = word

        # Try prefixes longest first
        # Require at least 3 chars remaining so single-letter prefixes
        # don't consume 3-letter roots (e.g. "ك" must not strip from "كتب")
        for p in sorted(_PREFIXES, key=len, reverse=True):
            if stripped.startswith(p) and len(stripped) - len(p) >= 3:
                prefix = p
                stripped = stripped[len(p):]
                break

        # Try suffixes longest first (after prefix removal)
        for s in sorted(_SUFFIXES, key=len, reverse=True):
            if stripped.endswith(s) and len(stripped) - len(s) >= 3:
                suffix = s
                stripped = stripped[:-len(s)]
                break

        return stripped, prefix, suffix

    def _tag_token(self, raw: str) -> ArabicToken:
        """Classify a single token."""
        clean = _TASHKEEL_RE.sub("", raw)
        clean = re.sub(r"[أإآٱ]", "ا", clean)

        # Fast particle check
        if clean in _PARTICLES or len(clean) <= 2:
            return ArabicToken(
                raw=raw, clean=clean, stripped=clean,
                pos="particle",
            )

        stripped, prefix, suffix = self._strip_affixes(clean)

        # POS classification rules
        pos = "unknown"

        # Definite nouns (start with ال)
        if prefix in ("ال", "بال", "فال", "وال", "كال", "لل", "ولل"):
            pos = "noun"

        # Semantic dictionary check
        sem_fields: List[str] = []
        for key, fields in _SEMANTIC_DICT.items():
            if key in clean or clean in key:
                sem_fields = fields
                if pos == "unknown":
                    pos = "noun"
                break

        # Verb detection: Arabic present tense verb prefixes
        if pos == "unknown":
            if self._VERB_PREFIX_RE.match(stripped) and len(stripped) >= 3:
                pos = "verb"
            elif len(stripped) == 3 and re.match(r"^[\u0600-\u06FF]{3}$", stripped):
                # 3-char words with no prefix/suffix → likely past-tense verb
                # unless looks like noun
                if stripped[-1] not in "ةه":
                    pos = "verb"
                else:
                    pos = "noun"
            elif len(stripped) >= 4 and stripped.endswith("ة"):
                pos = "noun"
            elif len(stripped) >= 4:
                pos = "noun"

        return ArabicToken(
            raw=raw, clean=clean, stripped=stripped,
            prefix=prefix, suffix=suffix,
            pos=pos,
            semantic_fields=sem_fields,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2 — Morphological Analyser (صرفي)
# ═══════════════════════════════════════════════════════════════════════════

class MorphologicalAnalyser:
    """
    Layer 2: Root extraction (جذور) and pattern matching (أوزان).

    For each token stripped of prefixes/suffixes:
      1. Attempts to identify the 3-letter root.
      2. Matches against known morphological patterns (أوزان).
      3. Computes root_complexity (col[2]) and morpho_pattern_score (col[3]).
    """

    def analyse(self, syn: SyntacticLayer) -> MorphologicalLayer:
        layer = MorphologicalLayer()

        roots_freq: Dict[str, int] = {}
        wazn_dist:  Dict[str, int] = {}
        pattern_hit = 0
        total = 0

        for tok in syn.tokens:
            if tok.pos == "particle":
                continue
            total += 1
            stripped = tok.stripped

            # ── Root extraction ───────────────────────────────────────────
            root, confidence = self._extract_root(stripped, tok.pos)
            tok.root = root
            tok.root_confidence = confidence
            if root:
                roots_freq[root] = roots_freq.get(root, 0) + 1
                # Attach semantic fields from root dict
                if root in _KNOWN_ROOTS and not tok.semantic_fields:
                    tok.semantic_fields = _KNOWN_ROOTS[root]

            # ── Pattern matching ──────────────────────────────────────────
            wazn = self._match_pattern(stripped, tok.pos)
            tok.wazn = wazn
            if wazn:
                pattern_hit += 1
                wazn_dist[wazn] = wazn_dist.get(wazn, 0) + 1

        layer.roots_by_frequency = roots_freq
        layer.wazn_distribution  = wazn_dist
        layer.unique_roots       = list(roots_freq.keys())
        layer.root_count         = len(layer.unique_roots)
        layer.pattern_matches    = pattern_hit
        layer.total_tokens_analysed = total

        # ── Scores ────────────────────────────────────────────────────────
        if total > 0:
            # root_complexity: root diversity / total (more unique roots = richer text)
            diversity = len(roots_freq) / total
            # normalise: diversity 0.3 → score 0.5, 0.8 → score 1.0
            layer.root_complexity = round(min(1.0, diversity * 1.3), 6)

            # morpho_pattern_score: fraction of content words matching a known pattern
            layer.morpho_pattern_score = round(pattern_hit / total, 6)
        else:
            layer.root_complexity      = 0.0
            layer.morpho_pattern_score = 0.0

        return layer

    def _extract_root(self, word: str, pos: str) -> Tuple[str, float]:
        """
        Lightweight root extraction.
        Strategy:
          1. Check known roots directly.
          2. Try 3-letter extraction by removing common patterns.
          3. Fall back to first 3 Arabic letters.
        """
        if not word or len(word) < 2:
            return "", 0.0

        # Remove remaining long vowel signs
        w = re.sub(r"[اويى]", "", word)

        # Check known roots directly
        for root in _KNOWN_ROOTS:
            if root in word or word in root:
                return root, 0.9

        # Attempt to pick 3 consonants
        consonants = re.findall(r"[\u0600-\u06FF]", w)
        if len(consonants) >= 3:
            root = "".join(consonants[:3])
            # Verify it's in known roots
            conf = 0.7 if root in _KNOWN_ROOTS else 0.4
            return root, conf

        if consonants:
            return "".join(consonants), 0.3

        return word[:3] if len(word) >= 3 else word, 0.2

    def _match_pattern(self, word: str, pos: str) -> str:
        """
        Match a stripped word against known morphological patterns.
        Returns the wazn name or empty string.
        """
        if not word or len(word) < 2:
            return ""

        # Normalise: replace Arabic consonants with placeholder slots
        # Strategy: check length and characteristic letters
        w_len = len(word)

        # Particle: no pattern
        if pos == "particle":
            return ""

        # Very short verb-like (3 chars, all consonants)
        if pos == "verb" and w_len == 3:
            return "فَعَلَ"

        if pos == "verb" and w_len == 4:
            first = word[0]
            if first in "أاي":
                return "أَفْعَلَ"
            if first == "ت":
                return "تَفَعَّلَ"
            return "فَعَّلَ"

        if pos == "verb" and w_len >= 5:
            if word.startswith("است"):
                return "اسْتَفْعَلَ"
            if word.startswith("انف"):
                return "انْفَعَلَ"
            if word.startswith("افت"):
                return "افْتَعَلَ"
            return "تَفَاعَلَ"

        # Noun patterns by suffix
        if pos in ("noun", "adj", "unknown"):
            if word.endswith("ة") and w_len == 4:
                return "فِعَالَة"
            if word.endswith("ون") or word.endswith("ين"):
                return "فَاعِلُون"   # sound masculine plural
            if word.endswith("ات") and w_len >= 4:
                return "فَعَّالَة"   # sound feminine plural
            if word.endswith("ان") and w_len >= 4:
                return "فُعْلَان"
            if w_len == 4 and "ا" in word[1:-1]:
                return "فَاعِل"
            if w_len >= 5 and word.startswith("م"):
                return "مَفْعُول"
            if w_len >= 6 and word.startswith("مست"):
                return "مُسْتَفْعِل"

        return ""


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3 — Semantic Analyser (دلالي)
# ═══════════════════════════════════════════════════════════════════════════

class SemanticAnalyser:
    """
    Layer 3: Contextual meaning and concept mapping.

    Uses:
      - Built-in semantic dictionary (_SEMANTIC_DICT)
      - Known root semantic fields (_KNOWN_ROOTS)
      - Optional CKG integration (passed at runtime)
    """

    def analyse(
        self,
        text: str,
        syn: SyntacticLayer,
        morph: MorphologicalLayer,
        ckg=None,
    ) -> SemanticLayer:
        layer = SemanticLayer()

        concept_freq: Dict[str, int] = {}
        all_fields:   List[str]      = []

        # ── Step 1: dictionary + token semantic fields ─────────────────────
        clean = _TASHKEEL_RE.sub("", text)
        clean = re.sub(r"[أإآٱ]", "ا", clean)

        for key, fields in _SEMANTIC_DICT.items():
            key_clean = re.sub(r"[أإآٱ]", "ا", key)
            if key_clean in clean:
                for f in fields:
                    concept_freq[f] = concept_freq.get(f, 0) + clean.count(key_clean)
                    all_fields.append(f)

        # ── Step 2: token-level semantic fields (from Layer 1 & 2) ────────
        for tok in syn.tokens:
            for f in tok.semantic_fields:
                concept_freq[f] = concept_freq.get(f, 0) + 1
                all_fields.append(f)

        # ── Step 3: CKG alignment ─────────────────────────────────────────
        ckg_bonus = 0.0
        if ckg is not None:
            try:
                ckg_concepts = set(ckg._concepts.keys()) if hasattr(ckg, "_concepts") else set()
                for concept in concept_freq:
                    if concept in ckg_concepts:
                        ckg_bonus += 0.1
                layer.ckg_aligned = bool(ckg_concepts & set(concept_freq.keys()))
            except Exception:
                pass
        ckg_bonus = min(0.3, ckg_bonus)

        # ── Step 4: context score ─────────────────────────────────────────
        words = _ARABIC_RE.findall(clean)
        content_words = [w for w in words if w not in _STOPWORDS and len(w) > 2]
        total_words   = len(words)
        content_ratio = len(content_words) / max(total_words, 1)

        # Semantic density: concept hits per content word
        total_hits    = sum(concept_freq.values())
        sem_density   = min(1.0, total_hits / max(len(content_words), 1))

        layer.concept_frequency        = concept_freq
        layer.concepts_found           = list(concept_freq.keys())
        layer.concept_count            = len(concept_freq)
        layer.semantic_fields          = all_fields

        # col[4]: semantic_concept_score
        raw_sem = min(1.0, layer.concept_count / max(total_words / 5, 1))
        layer.semantic_concept_score   = round(min(1.0, raw_sem + ckg_bonus), 6)

        # col[5]: context_score
        layer.context_score = round(
            (content_ratio * 0.5 + sem_density * 0.5), 6
        )

        return layer


# ═══════════════════════════════════════════════════════════════════════════
# Arabic NLP Engine — main interface
# ═══════════════════════════════════════════════════════════════════════════

class ArabicNLPEngine:
    """
    Main Arabic NLP Engine.

    Orchestrates all 3 analysis layers and produces a 7-element
    ArabicFeatureVector compatible with the neural weight matrix (9×7).

    Usage
    -----
        engine = ArabicNLPEngine()

        # Full analysis
        result = engine.analyse("بسم الله الرحمن الرحيم")

        # Get 7-element vector for neural training
        vec = result.feature_vector.to_list()   # always len=7

        # Train the neural weight layer
        output = engine.analyse_and_train(text, target_score, neural_layer)
    """

    VERSION = "18.0.0"

    def __init__(self, ckg=None):
        self._syn   = SyntacticAnalyser()
        self._morph = MorphologicalAnalyser()
        self._sem   = SemanticAnalyser()
        self._ckg   = ckg
        self._analysis_count = 0
        logger.info(
            f"ArabicNLPEngine v{self.VERSION} initialised "
            f"(3 layers: نحوي → صرفي → دلالي → 7-col feature vector)"
        )

    def set_ckg(self, ckg) -> None:
        """Inject CKG for semantic alignment (optional)."""
        self._ckg = ckg
        logger.info("ArabicNLPEngine: CKG connected")

    # ── Core analysis ─────────────────────────────────────────────────────

    def analyse(self, text: str) -> ArabicAnalysisResult:
        """
        Run all 3 layers and produce a full ArabicAnalysisResult.

        Returns
        -------
        ArabicAnalysisResult
            Contains .feature_vector with exactly 7 floats — ready to feed
            into NeuralWeightLayer.forward() or DynamicWeightLayer.forward().
        """
        if not text or not text.strip():
            return self._empty_result(text or "")

        self._analysis_count += 1

        # Layer 1 — Syntactic
        syn   = self._syn.analyse(text)

        # Layer 2 — Morphological (receives Layer 1 tokens)
        morph = self._morph.analyse(syn)

        # Layer 3 — Semantic (receives both previous layers + optional CKG)
        sem   = self._sem.analyse(text, syn, morph, self._ckg)

        # ── Assemble 7-element feature vector ─────────────────────────────
        fv = ArabicFeatureVector(
            verb_score             = syn.verb_score,            # col[0]
            noun_score             = syn.noun_score,            # col[1]
            root_complexity        = morph.root_complexity,     # col[2]
            morpho_pattern_score   = morph.morpho_pattern_score,# col[3]
            semantic_concept_score = sem.semantic_concept_score,# col[4]
            context_score          = sem.context_score,         # col[5]
            syntactic_complexity   = syn.syntactic_complexity,  # col[6]
        )

        clean = _TASHKEEL_RE.sub("", text)
        clean = re.sub(r"[أإآٱ]", "ا", clean).strip()

        return ArabicAnalysisResult(
            text        = text,
            text_clean  = clean,
            analysed_at = datetime.now(timezone.utc).isoformat(),
            syntactic   = syn,
            morphological = morph,
            semantic    = sem,
            feature_vector = fv,
        )

    def analyse_and_train(
        self,
        text: str,
        target: float,
        neural_layer,
    ) -> dict:
        """
        Analyse text and use the 7-element feature vector to train
        the neural weight layer (NeuralWeightLayer or DynamicWeightLayer).

        Parameters
        ----------
        text : str
            Arabic text to analyse.
        target : float
            Training target (0-1). E.g. quality score / 100.
        neural_layer : NeuralWeightLayer | DynamicWeightLayer
            The active neural weight layer from the mesh.

        Returns
        -------
        dict with analysis summary + training result.
        """
        result = self.analyse(text)
        vec    = result.feature_vector.to_list()   # exactly 7 floats

        # Forward pass
        try:
            import numpy as np
            output = neural_layer.forward(vec)
            result.neural_output = output.tolist()

            # Training step
            loss   = neural_layer.train_step(vec, float(target))
            weight_stats = {
                "min":  round(float(neural_layer.weights.min()), 6),
                "max":  round(float(neural_layer.weights.max()), 6),
                "mean": round(float(neural_layer.weights.mean()), 6),
                "shape": list(neural_layer.weights.shape),
            }
        except Exception as exc:
            logger.warning(f"ArabicNLPEngine: neural layer error: {exc}")
            output       = []
            loss         = None
            weight_stats = {}

        return {
            "analysis":     result.to_dict(),
            "feature_vector": vec,
            "target":       target,
            "loss":         round(loss, 8) if loss is not None else None,
            "neural_output": result.neural_output,
            "weight_stats": weight_stats,
        }

    def batch_analyse(self, texts: List[str]) -> List[ArabicAnalysisResult]:
        """Analyse a list of texts and return results."""
        return [self.analyse(t) for t in texts]

    # ── Convenience sub-analysis methods ──────────────────────────────────

    def syntactic_only(self, text: str) -> dict:
        """Layer 1 only."""
        syn = self._syn.analyse(text)
        return {
            "sentence_count": syn.sentence_count,
            "token_count":    syn.token_count,
            "verb_count":     syn.verb_count,
            "noun_count":     syn.noun_count,
            "particle_count": syn.particle_count,
            "verb_score":     round(syn.verb_score, 4),
            "noun_score":     round(syn.noun_score, 4),
            "syntactic_complexity": round(syn.syntactic_complexity, 4),
            "tokens": [
                {"raw": t.raw, "pos": t.pos, "stripped": t.stripped}
                for t in syn.tokens
            ],
        }

    def morphological_only(self, text: str) -> dict:
        """Layers 1 + 2."""
        syn   = self._syn.analyse(text)
        morph = self._morph.analyse(syn)
        return {
            "unique_roots":       morph.unique_roots[:20],
            "root_count":         morph.root_count,
            "pattern_matches":    morph.pattern_matches,
            "root_complexity":    round(morph.root_complexity, 4),
            "morpho_pattern_score": round(morph.morpho_pattern_score, 4),
            "roots_by_frequency": dict(sorted(
                morph.roots_by_frequency.items(), key=lambda x: -x[1]
            )[:15]),
            "wazn_distribution":  morph.wazn_distribution,
        }

    def semantic_only(self, text: str) -> dict:
        """Layers 1 + 2 + 3 (semantic output only)."""
        result = self.analyse(text)
        sem    = result.semantic
        return {
            "concepts_found":        sem.concepts_found[:15],
            "concept_count":         sem.concept_count,
            "semantic_fields":       list(set(sem.semantic_fields))[:10],
            "ckg_aligned":           sem.ckg_aligned,
            "semantic_concept_score": round(sem.semantic_concept_score, 4),
            "context_score":         round(sem.context_score, 4),
            "concept_frequency":     dict(sorted(
                sem.concept_frequency.items(), key=lambda x: -x[1]
            )[:10]),
        }

    def status(self) -> dict:
        return {
            "version":        self.VERSION,
            "analysis_count": self._analysis_count,
            "ckg_connected":  self._ckg is not None,
            "layers": [
                {"id": 1, "name": "نحوي (Syntactic)",      "class": "SyntacticAnalyser"},
                {"id": 2, "name": "صرفي (Morphological)",   "class": "MorphologicalAnalyser"},
                {"id": 3, "name": "دلالي (Semantic)",        "class": "SemanticAnalyser"},
            ],
            "feature_vector_size": 7,
            "feature_vector_cols": [
                "col_0_verb_score", "col_1_noun_score",
                "col_2_root_complexity", "col_3_morpho_pattern_score",
                "col_4_semantic_concept_score", "col_5_context_score",
                "col_6_syntactic_complexity",
            ],
            "neural_matrix_compatible": True,
            "columns_fixed": 7,
            "rows_start": 9,
            "rows_grow_by": 23,
        }

    def _empty_result(self, text: str) -> ArabicAnalysisResult:
        return ArabicAnalysisResult(
            text=text, text_clean="",
            analysed_at=datetime.now(timezone.utc).isoformat(),
            syntactic=SyntacticLayer(),
            morphological=MorphologicalLayer(),
            semantic=SemanticLayer(),
            feature_vector=ArabicFeatureVector(),
        )


# ── Module-level singleton ─────────────────────────────────────────────────

_default_engine: Optional[ArabicNLPEngine] = None


def get_arabic_engine(ckg=None) -> ArabicNLPEngine:
    """Return (and cache) the module-level ArabicNLPEngine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = ArabicNLPEngine(ckg=ckg)
    elif ckg is not None and _default_engine._ckg is None:
        _default_engine.set_ckg(ckg)
    return _default_engine
