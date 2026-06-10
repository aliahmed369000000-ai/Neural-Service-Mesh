"""
Arabic Concept Discovery  (Cognitive Layer)
============================================
Automatically discovers Arabic concepts from any corpus (Quran, books, etc.)
by analysing token frequency and extracting trilateral roots.

No external libraries required. Pure Python 3.8+.

What this does
--------------
1. Scans all quran_chunk_*.json files
2. Counts every Arabic token frequency across the full corpus
3. Any token appearing 10+ times = concept candidate
4. Extracts the trilateral root (3-letter root) for each token
5. Groups token families under one root concept
6. Merges results into ConceptDictionary
7. Saves knowledge/arabic_roots_index.json

Usage
-----
    from knowledge_sources.arabic_concept_discovery import ArabicConceptDiscovery
    discovery = ArabicConceptDiscovery()
    results   = discovery.run()
    print(results["total_roots"])      # 400+
    print(results["total_tokens"])     # 1500+
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Stop Words — filtered out before concept discovery
# ─────────────────────────────────────────────────────────────────────────────

ARABIC_STOP_WORDS: Set[str] = {
    # Prepositions
    "من", "في", "على", "إلى", "عن", "مع", "عند", "حتى", "منذ", "خلال",
    # Pronouns
    "هو", "هي", "هم", "هن", "أنت", "أنتم", "أنا", "نحن", "انا",
    # Conjunctions
    "أن", "إن", "ان", "ان", "لأن", "لان", "حين", "إذا", "اذا",
    "ثم", "أو", "او", "بل", "لكن", "لو", "كي",
    # Negation
    "لا", "ما", "لم", "لن", "ليس",
    # Common verbs (too frequent to be meaningful)
    "كان", "قال", "قالوا", "يكون", "كانت", "كانوا",
    # Relative pronouns
    "الذي", "التي", "الذين", "اللواتي", "اللتان",
    # Question words
    "ما", "من", "كيف", "أين", "اين", "متى", "لماذا", "هل",
    # Common particles
    "وما", "فما", "وهو", "وهي", "وهم", "لهم", "لكم", "لنا",
    "منهم", "منكم", "منها", "عليه", "عليهم", "عليك", "عليكم",
    "بهم", "بكم", "بها", "بنا", "فيه", "فيها", "فيهم", "فيكم",
    "فإن", "فلا", "ولا", "وقد", "فقد", "لقد",
    "إنه", "إنها", "إنهم", "انه", "انها", "انهم",
    # Numbers / demonstratives
    "هذا", "هذه", "هؤلاء", "ذلك", "تلك", "أولئك",
    "كل", "بعض", "غير", "مثل",
    # Short meaningless tokens
    "قد", "إذ", "اذ", "أي", "اي", "يا",
}

# Normalise stop words (remove diacritics, unify letters)
def _norm(w: str) -> str:
    w = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', w)   # tashkeel
    w = re.sub(r'[أإآٱ]', 'ا', w)                        # alef
    w = w.replace('ى', 'ي').replace('ة', 'ه')
    return w

_STOP_WORDS_NORM: Set[str] = {_norm(w) for w in ARABIC_STOP_WORDS}

# ─────────────────────────────────────────────────────────────────────────────
#  Trilateral Root Extractor
# ─────────────────────────────────────────────────────────────────────────────

# Ordered prefix list — longer prefixes first to avoid partial stripping
_PREFIXES = [
    "وال", "فال", "بال", "كال", "ولل", "فلل",
    "وب", "فب", "وك", "فك", "ول", "فل",
    "ال", "و", "ف", "ب", "ك", "ل", "س",
]

# Ordered suffix list — longer suffixes first
_SUFFIXES = [
    "ونني", "وني", "تني", "ينني", "تموني",
    "ونه", "ونها", "ونهم", "ونكم",
    "تموه", "تموها", "تموهم",
    "ون", "ين", "ان", "ات", "تم", "تن",
    "نا", "ها", "هم", "هن", "كم", "كن",
    "يه", "ية", "يا",
    "ه", "ة", "ي", "ا", "ن",
]


def extract_root(token: str) -> str:
    """
    Extract an approximate trilateral root from a normalised Arabic token.

    Returns the root string (3-4 chars ideally) or the original token
    if extraction produces something too short or too long.
    """
    word = token

    # Strip prefixes
    for prefix in _PREFIXES:
        if word.startswith(prefix) and len(word) - len(prefix) >= 3:
            word = word[len(prefix):]
            break

    # Strip suffixes
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[:-len(suffix)]
            break

    # Accept 3 or 4 char results as root candidates
    if 3 <= len(word) <= 4:
        return word

    # If still too long, return first 3 chars as approximate root
    if len(word) > 4:
        return word[:3]

    # Too short — return original token
    return token


# ─────────────────────────────────────────────────────────────────────────────
#  ArabicConceptDiscovery
# ─────────────────────────────────────────────────────────────────────────────

class ArabicConceptDiscovery:
    """
    Discovers Arabic concepts automatically from the Quran corpus.

    Parameters
    ----------
    knowledge_dir     : path to knowledge/ folder containing quran chunks
    roots_index_path  : where to save arabic_roots_index.json
    min_frequency     : minimum token frequency to qualify as a concept (default 10)
    min_token_length  : ignore tokens shorter than this (default 3)
    """

    def __init__(
        self,
        knowledge_dir:    Path = Path("./knowledge"),
        roots_index_path: Path = Path("./knowledge/arabic_roots_index.json"),
        min_frequency:    int  = 10,
        min_token_length: int  = 3,
    ):
        self._knowledge_dir    = Path(knowledge_dir)
        self._roots_index_path = Path(roots_index_path)
        self._min_frequency    = min_frequency
        self._min_token_length = min_token_length

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """
        Full discovery pipeline. Returns summary statistics.

        Steps:
          1. Load all Quran ayahs from chunk files
          2. Count token frequencies
          3. Filter stop words and short tokens
          4. Extract roots and group token families
          5. Save roots index
          6. Merge into ConceptDictionary
          7. Inject into CognitiveGraph
        """
        logger.info("[ArabicConceptDiscovery] Starting full discovery run...")

        # Step 1: Load corpus
        ayahs = self._load_corpus()
        logger.info(f"[ArabicConceptDiscovery] Loaded {len(ayahs)} ayahs")

        # Step 2: Count frequencies
        token_counts = self._count_tokens(ayahs)
        logger.info(f"[ArabicConceptDiscovery] {len(token_counts)} unique tokens found")

        # Step 3: Filter
        candidates = {
            token: count
            for token, count in token_counts.items()
            if count >= self._min_frequency
            and len(token) >= self._min_token_length
            and token not in _STOP_WORDS_NORM
        }
        logger.info(f"[ArabicConceptDiscovery] {len(candidates)} candidate concepts after filtering")

        # Step 4: Extract roots and build families
        root_families = self._build_root_families(candidates)
        logger.info(f"[ArabicConceptDiscovery] {len(root_families)} unique roots discovered")

        # Step 5: Save roots index
        self._save_roots_index(root_families)

        # Step 6: Merge into ConceptDictionary
        merged = self._merge_into_concept_dictionary(root_families)

        # Step 7: Inject into CognitiveGraph
        injected = self._inject_into_cognitive_graph(root_families)

        results = {
            "total_ayahs":         len(ayahs),
            "total_unique_tokens": len(token_counts),
            "total_candidates":    len(candidates),
            "total_roots":         len(root_families),
            "total_tokens_mapped": sum(len(f["tokens"]) for f in root_families.values()),
            "merged_into_dict":    merged,
            "injected_into_ckg":   injected,
            "roots_index_path":    str(self._roots_index_path),
        }

        logger.info(
            f"[ArabicConceptDiscovery] Done — "
            f"{results['total_roots']} roots, "
            f"{results['total_tokens_mapped']} tokens mapped"
        )
        return results

    def get_concept_for_word(self, word: str) -> Optional[str]:
        """
        Given any Arabic word, return the root concept it belongs to.
        Returns None if not found in the index.
        """
        if not self._roots_index_path.exists():
            return None

        norm_word = _norm(word)
        index = json.loads(self._roots_index_path.read_text(encoding="utf-8"))

        # Direct root match
        if norm_word in index:
            return norm_word

        # Search in token lists
        for root, data in index.items():
            if norm_word in data.get("tokens", []):
                return root

        # Try extracting root and matching
        root = extract_root(norm_word)
        if root in index:
            return root

        return None

    def load_roots_index(self) -> Dict[str, Any]:
        """Load and return the saved roots index."""
        if not self._roots_index_path.exists():
            return {}
        return json.loads(self._roots_index_path.read_text(encoding="utf-8"))

    # ── Internal ───────────────────────────────────────────────────────────

    def _load_corpus(self) -> List[str]:
        """Load all ayah texts from quran_chunk_*.json files."""
        ayahs: List[str] = []
        chunk_files = sorted(self._knowledge_dir.glob("quran_chunk_*.json"))

        if not chunk_files:
            logger.warning(
                f"[ArabicConceptDiscovery] No chunk files found in {self._knowledge_dir}"
            )
            return ayahs

        for chunk_file in chunk_files:
            try:
                data = json.loads(chunk_file.read_text(encoding="utf-8"))
                items = data if isinstance(data, list) else data.get("items", [])
                for item in items:
                    text = ""
                    if isinstance(item, dict):
                        text = (
                            item.get("content", "")
                            or item.get("text", "")
                            or item.get("arabic", "")
                            or item.get("raw_content", "")
                        )
                    elif isinstance(item, str):
                        text = item
                    if text:
                        ayahs.append(text)
            except Exception as exc:
                logger.warning(f"[ArabicConceptDiscovery] Error reading {chunk_file}: {exc}")

        return ayahs

    def _count_tokens(self, texts: List[str]) -> Counter:
        """Tokenize all texts and return frequency counter."""
        counter: Counter = Counter()
        non_arabic = re.compile(r'[^\u0600-\u06FF]')
        multi_space = re.compile(r'\s+')

        for text in texts:
            # Normalise
            norm = _norm(text)
            # Keep only Arabic chars
            norm = non_arabic.sub(' ', norm)
            norm = multi_space.sub(' ', norm).strip()
            tokens = [t for t in norm.split(' ') if t]
            counter.update(tokens)

        return counter

    def _build_root_families(
        self, candidates: Dict[str, int]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group candidate tokens by their extracted root.

        Returns dict: root → {tokens, total_frequency, category}
        """
        root_map: Dict[str, Dict[str, Any]] = {}

        for token, freq in sorted(candidates.items(), key=lambda x: -x[1]):
            root = extract_root(token)

            if root not in root_map:
                root_map[root] = {
                    "tokens":          [],
                    "total_frequency": 0,
                    "top_token":       token,
                    "top_frequency":   freq,
                    "category":        "مكتشف_تلقائياً",
                }

            root_map[root]["tokens"].append(token)
            root_map[root]["total_frequency"] += freq

            # Track the most frequent token as representative
            if freq > root_map[root]["top_frequency"]:
                root_map[root]["top_token"]     = token
                root_map[root]["top_frequency"] = freq

        return root_map

    def _save_roots_index(self, root_families: Dict[str, Dict[str, Any]]) -> None:
        """Save the roots index to disk atomically."""
        output = {}
        for root, data in root_families.items():
            output[root] = {
                "tokens":          data["tokens"],
                "frequency":       data["total_frequency"],
                "top_token":       data["top_token"],
                "concept_name":    root,
                "category":        data["category"],
            }

        tmp = self._roots_index_path.with_suffix(".tmp")
        try:
            self._roots_index_path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(output, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(str(tmp), str(self._roots_index_path))
            logger.info(
                f"[ArabicConceptDiscovery] Saved {len(output)} roots to {self._roots_index_path}"
            )
        except Exception as exc:
            logger.error(f"[ArabicConceptDiscovery] Failed to save roots index: {exc}")
            tmp.unlink(missing_ok=True)

    def _merge_into_concept_dictionary(
        self, root_families: Dict[str, Dict[str, Any]]
    ) -> int:
        """
        Merge discovered roots into the live ConceptExtractor keyword index.
        Manually defined concepts take priority (existing keys are skipped).
        Returns number of new concepts added.
        """
        try:
            from knowledge_sources.concept_extractor import get_extractor
            extractor = get_extractor()

            # Build set of already-known concept keys from the keyword index
            existing_concepts: set = set()
            if hasattr(extractor, "_keyword_index"):
                existing_concepts = {concept for (_, concept) in extractor._keyword_index.values()}

            added = 0
            for root, data in root_families.items():
                if root in existing_concepts:
                    continue  # manual definition takes priority

                # Inject discovered tokens into the keyword index
                if hasattr(extractor, "_keyword_index"):
                    for token in data["tokens"][:10]:   # top 10 tokens per root
                        extractor._keyword_index[token] = ("مكتشف_تلقائياً", root)
                    added += 1

            logger.info(
                f"[ArabicConceptDiscovery] Merged {added} new concepts into ConceptExtractor index"
            )
            return added

        except Exception as exc:
            logger.warning(f"[ArabicConceptDiscovery] Could not merge into dictionary: {exc}")
            return 0

    def _inject_into_cognitive_graph(
        self, root_families: Dict[str, Dict[str, Any]]
    ) -> int:
        """
        Inject discovered concepts into the CognitiveGraph with their frequencies.
        Returns number of concepts injected.
        """
        try:
            from knowledge.cognitive_graph import get_cognitive_graph
            cg = get_cognitive_graph()
            injected = 0

            for root, data in root_families.items():
                # add_concept(name, cluster, source) — touch frequency each call
                freq = data.get("total_frequency", 1)
                for _ in range(min(freq, 20)):   # cap at 20 touches to avoid slowness
                    cg.add_concept(
                        name    = root,
                        cluster = "مكتشف_تلقائياً",
                        source  = "arabic_discovery",
                    )
                injected += 1

            cg.save()
            logger.info(
                f"[ArabicConceptDiscovery] Injected {injected} concepts into CognitiveGraph"
            )
            return injected

        except Exception as exc:
            logger.warning(f"[ArabicConceptDiscovery] Could not inject into CKG: {exc}")
            return 0


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience runner
# ─────────────────────────────────────────────────────────────────────────────

def run_discovery(min_frequency: int = 10) -> Dict[str, Any]:
    """
    Module-level convenience function.
    Call this from quran_continuous_trainer.py after each batch.
    """
    discovery = ArabicConceptDiscovery(min_frequency=min_frequency)
    return discovery.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_discovery()
    print("\n=== Arabic Concept Discovery Results ===")
    print(f"  Ayahs processed  : {results['total_ayahs']}")
    print(f"  Unique tokens    : {results['total_unique_tokens']}")
    print(f"  Concept candidates: {results['total_candidates']}")
    print(f"  Roots discovered : {results['total_roots']}")
    print(f"  Tokens mapped    : {results['total_tokens_mapped']}")
    print(f"  Added to dict    : {results['merged_into_dict']}")
    print(f"  Added to CKG     : {results['injected_into_ckg']}")
    print(f"  Index saved to   : {results['roots_index_path']}")
