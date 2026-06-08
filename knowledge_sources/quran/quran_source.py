"""
Knowledge Sources — Quran Source
==================================
The Holy Quran is the FIRST official knowledge source in the system.

Design Principles:
  ┌─────────────────────────────────────────────────────────────────┐
  │  SOURCE DATA        ← READ ONLY — NEVER MODIFIED               │
  │  ─────────────────────────────────────────────────────────────  │
  │  DERIVED KNOWLEDGE  ← System may build concepts/relations here  │
  └─────────────────────────────────────────────────────────────────┘

  - trust_score = 1.0  (absolute maximum)
  - access_mode = READ_ONLY
  - allow_raw_modification = False
  - source_type = SCRIPTURE
  - update_frequency = STATIC (the text never changes)

Data Loading Strategy
---------------------
1. First tries to load a local Quran JSON file from:
      knowledge_sources/quran/data/quran.json
2. If not present, fetches from the open Quran API:
      https://api.alquran.cloud/v1/quran/quran-uthmani
3. Caches the data locally for offline use.

Each Ayah becomes one KnowledgeItem:
  raw_content   = Arabic text (protected)
  raw_reference = "surah_number:ayah_number"  e.g. "2:255"
  derived_tags  = [surah_name, juz, makkiyah/madaniyah]

The system ONLY stores derived knowledge (concepts, topics, themes).
The original Arabic text is stored for reference but flagged as immutable.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

from knowledge_sources.source_metadata import (
    KnowledgeItem, SourceMetadata, SourceType, UpdateFrequency,
    AccessMode, SourceStatus
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

QURAN_SOURCE_ID   = "quran-uthmani-v1"
QURAN_SOURCE_NAME = "القرآن الكريم"
_DATA_DIR         = Path("./knowledge_sources/quran/data")
_LOCAL_FILE       = _DATA_DIR / "quran.json"
_API_URL          = "https://api.alquran.cloud/v1/quran/quran-uthmani"
_API_TIMEOUT      = 30   # seconds

# Known Surah names (1-114) for tagging
_SURAH_NAMES = [
    "", "الفاتحة", "البقرة", "آل عمران", "النساء", "المائدة",
    "الأنعام", "الأعراف", "الأنفال", "التوبة", "يونس",
    "هود", "يوسف", "الرعد", "إبراهيم", "الحجر",
    "النحل", "الإسراء", "الكهف", "مريم", "طه",
    "الأنبياء", "الحج", "المؤمنون", "النور", "الفرقان",
    "الشعراء", "النمل", "القصص", "العنكبوت", "الروم",
    "لقمان", "السجدة", "الأحزاب", "سبأ", "فاطر",
    "يس", "الصافات", "ص", "الزمر", "غافر",
    "فصلت", "الشورى", "الزخرف", "الدخان", "الجاثية",
    "الأحقاف", "محمد", "الفتح", "الحجرات", "ق",
    "الذاريات", "الطور", "النجم", "القمر", "الرحمن",
    "الواقعة", "الحديد", "المجادلة", "الحشر", "الممتحنة",
    "الصف", "الجمعة", "المنافقون", "التغابن", "الطلاق",
    "التحريم", "الملك", "القلم", "الحاقة", "المعارج",
    "نوح", "الجن", "المزمل", "المدثر", "القيامة",
    "الإنسان", "المرسلات", "النبأ", "النازعات", "عبس",
    "التكوير", "الانفطار", "المطففين", "الانشقاق", "البروج",
    "الطارق", "الأعلى", "الغاشية", "الفجر", "البلد",
    "الشمس", "الليل", "الضحى", "الشرح", "التين",
    "العلق", "القدر", "البينة", "الزلزلة", "العاديات",
    "القارعة", "التكاثر", "العصر", "الهمزة", "الفيل",
    "قريش", "الماعون", "الكوثر", "الكافرون", "النصر",
    "المسد", "الإخلاص", "الفلق", "الناس",
]


# ── Metadata Factory ───────────────────────────────────────────────────────

def build_quran_metadata() -> SourceMetadata:
    """Create the canonical SourceMetadata for the Quran source."""
    return SourceMetadata(
        id                     = QURAN_SOURCE_ID,
        name                   = QURAN_SOURCE_NAME,
        description            = (
            "القرآن الكريم — المصدر الأول للمعرفة في النظام. "
            "نص ثابت محمي، لا يُعدَّل أصله أبداً. "
            "يُسمح للنظام ببناء علاقات ومفاهيم مستخرجة فقط."
        ),
        source_type            = SourceType.SCRIPTURE,
        access_mode            = AccessMode.READ_ONLY,
        trust_score            = 1.0,
        base_trust             = 1.0,
        requires_citation      = True,
        update_frequency       = UpdateFrequency.STATIC,
        allow_raw_modification = False,   # ABSOLUTE — never change the Quran text
        store_raw_data         = True,    # Keep the text for reference
        store_derived_knowledge = True,   # System builds concepts here
        language               = "ar",
        version                = "uthmani-1.0",
        tags                   = ["قرآن", "إسلام", "عربي", "scripture", "arabic"],
        config                 = {
            "local_file":  str(_LOCAL_FILE),
            "api_url":     _API_URL,
            "edition":     "quran-uthmani",
            "total_surahs": 114,
            "total_ayahs":  6236,
        },
    )


# ── Data Loader ────────────────────────────────────────────────────────────

class QuranLoader:
    """
    Loads Quran data from local cache or remote API.

    Never modifies the fetched text. Always stores a local copy so
    the system can run offline after the first sync.
    """

    def __init__(self, local_file: Path = _LOCAL_FILE, api_url: str = _API_URL):
        self._local  = local_file
        self._api    = api_url

    def load(self) -> Optional[Dict[str, Any]]:
        """Load Quran data. Returns raw API/JSON structure or None."""
        # Try local first
        if self._local.exists():
            logger.info(f"[QuranLoader] loading from local cache: {self._local}")
            try:
                with open(self._local, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning(f"[QuranLoader] local load error: {exc}")

        # Try API
        logger.info(f"[QuranLoader] fetching from API: {self._api}")
        return self._fetch_from_api()

    def _fetch_from_api(self) -> Optional[Dict[str, Any]]:
        try:
            req  = Request(self._api, headers={"User-Agent": "NeuralServiceMesh/1.0"})
            with urlopen(req, timeout=_API_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            if data.get("code") == 200 and data.get("data"):
                self._cache_locally(data["data"])
                return data["data"]
            else:
                logger.error(f"[QuranLoader] API returned unexpected structure")
                return None

        except URLError as exc:
            logger.warning(f"[QuranLoader] network error: {exc} — will use offline data")
            return None
        except Exception as exc:
            logger.error(f"[QuranLoader] fetch error: {exc}")
            return None

    def _cache_locally(self, data: Dict[str, Any]) -> None:
        try:
            self._local.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._local.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            tmp.replace(self._local)
            logger.info(f"[QuranLoader] cached {self._local}")
        except Exception as exc:
            logger.warning(f"[QuranLoader] cache write error: {exc}")


# ── Feeder ─────────────────────────────────────────────────────────────────

class QuranFeeder:
    """
    Converts Quran data into KnowledgeItems.

    Each Ayah → one KnowledgeItem.
    raw_content is the protected Arabic text.
    derived_* fields contain system-extractable information.

    The feeder supports partial loading (surah range) to allow
    incremental sync without overwhelming the system.
    """

    def __init__(
        self,
        loader:        Optional[QuranLoader] = None,
        surah_start:   int = 1,
        surah_end:     int = 114,
        max_items:     Optional[int] = None,
    ):
        self._loader      = loader or QuranLoader()
        self._surah_start = surah_start
        self._surah_end   = surah_end
        self._max_items   = max_items

    def fetch(self) -> List[KnowledgeItem]:
        """Main entry point — returns list of KnowledgeItems for the Quran."""
        data = self._loader.load()
        if not data:
            logger.warning("[QuranFeeder] no data available — returning empty list")
            return []

        items   = []
        surahs  = data.get("surahs", {}).get("references", []) or data.get("surahs", [])

        for surah in surahs:
            surah_num = surah.get("number", 0)
            if not (self._surah_start <= surah_num <= self._surah_end):
                continue

            surah_name    = surah.get("name", _SURAH_NAMES[surah_num] if surah_num <= 114 else "")
            surah_english = surah.get("englishName", "")
            revelation    = surah.get("revelationType", "")   # Meccan / Medinan

            for ayah in surah.get("ayahs", []):
                ayah_num = ayah.get("numberInSurah", 0)
                text     = ayah.get("text", "").strip()
                juz      = ayah.get("juz", 0)

                if not text:
                    continue

                reference = f"{surah_num}:{ayah_num}"

                # Build derived tags from Quran metadata
                tags = [
                    surah_name,
                    surah_english,
                    f"سورة-{surah_num}",
                    f"جزء-{juz}",
                    revelation.lower() if revelation else "",
                    "قرآن", "آية",
                ]
                tags = [t for t in tags if t]

                item = KnowledgeItem(
                    item_id        = f"quran:{reference}",
                    source_id      = QURAN_SOURCE_ID,
                    source_type    = SourceType.SCRIPTURE,
                    # ── Raw (PROTECTED) ───────────────────────────────
                    raw_content    = text,          # Original Arabic — READ ONLY
                    raw_reference  = reference,     # "2:255"
                    raw_language   = "ar",
                    # ── Derived (system may write) ────────────────────
                    derived_tags      = tags,
                    derived_concepts  = self._extract_concepts(text, surah_name, reference),
                    derived_summary   = f"آية {ayah_num} من سورة {surah_name}",
                    # ── Trust ─────────────────────────────────────────
                    trust_score    = 1.0,
                )

                items.append(item)

                if self._max_items and len(items) >= self._max_items:
                    logger.info(f"[QuranFeeder] reached max_items={self._max_items}")
                    return items

        logger.info(f"[QuranFeeder] produced {len(items)} items")
        return items

    def _extract_concepts(
        self, text: str, surah_name: str, reference: str
    ) -> List[str]:
        """
        Extract basic concepts from an ayah for the knowledge graph.
        This is DERIVED knowledge — the system adds this, not the original text.

        In a full implementation, this would use NLP / embedding models.
        Here we use a simple keyword approach as a foundation.
        """
        concepts = [f"سورة:{surah_name}", f"آية:{reference}"]

        # Topic keywords commonly found in Quran
        topic_map = {
            "الله":     "توحيد",
            "الحمن":     "رحمة",
            "الرحيم":     "رحمة",
            "جنة":      "آخرة",
            "نار":      "آخرة",
            "يوم القيامة": "آخرة",
            "صلاة":     "عبادة",
            "زكاة":     "عبادة",
            "صيام":     "عبادة",
            "تقوى":     "أخلاق",
            "علم":      "معرفة",
            "عقل":      "معرفة",
            "رسول":     "نبوة",
            "نبي":      "نبوة",
            "كتاب":     "وحي",
            "قرآن":     "وحي",
        }

        for keyword, concept in topic_map.items():
            if keyword in text and concept not in concepts:
                concepts.append(concept)

        return concepts[:10]  # Cap to avoid bloat


# ── Convenience Factory ────────────────────────────────────────────────────

def create_quran_source(
    surah_start: int = 1,
    surah_end:   int = 114,
    max_items:   Optional[int] = None,
):
    """
    Returns (SourceMetadata, feeder_callable) ready to pass to SourceManager.

    Usage:
        meta, feeder = create_quran_source()
        source_manager.register_source(meta, feeder)
        source_manager.sync_source(meta.id)
    """
    meta   = build_quran_metadata()
    loader = QuranLoader()
    feeder_obj = QuranFeeder(
        loader      = loader,
        surah_start = surah_start,
        surah_end   = surah_end,
        max_items   = max_items,
    )
    return meta, feeder_obj.fetch
