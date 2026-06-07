"""
Phase 15 – Immune System
Digital immune layer that inspects every incoming data item before
it is admitted to memory.

Responsibilities:
  • Classify sources as Trusted / Unknown / Blacklisted
  • Detect contradictions with stored facts (flag if conflict > 70 %)
  • Auto-block flood attacks: ≥ 1 000 identical items from unknown
    sources within a 60-second window
  • Quarantine suspicious items for manual review
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict, deque
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations & constants
# ---------------------------------------------------------------------------

class SourceTrust(str, Enum):
    TRUSTED     = "Trusted"
    UNKNOWN     = "Unknown"
    BLACKLISTED = "Blacklisted"


FLOOD_THRESHOLD   = 1_000   # identical items from unknown sources …
FLOOD_WINDOW_SEC  = 60      # … within this many seconds
CONFLICT_FLAG_PCT = 0.70    # raise flag if contradiction ratio exceeds this


# ---------------------------------------------------------------------------
# ImmuneSystem
# ---------------------------------------------------------------------------

class ImmuneSystem:
    """
    Inspects incoming data items and decides their fate.

    Parameters
    ----------
    knowledge_store : optional
        Object with a .get(key) method used to look up stored facts.
        If None, contradiction detection is skipped.
    """

    def __init__(self, knowledge_store=None):
        self._ks = knowledge_store

        # Source classification registry
        self._trusted:     set = {"system", "core", "bootstrap", "internal_sensor"}
        self._blacklisted: set = set()

        # Quarantine: item_id -> item dict
        self._quarantine: Dict[str, dict] = {}

        # Flood detection: content_hash -> deque of (timestamp, source)
        self._flood_ledger: Dict[str, Deque[Tuple[float, str]]] = defaultdict(deque)

        # Blocked hashes (auto-blocked due to flood)
        self._blocked_hashes: set = set()

        # Counters
        self._total_inspected:  int = 0
        self._total_blocked:    int = 0
        self._total_quarantined: int = 0
        self._total_passed:     int = 0

        # Current aggregate threat level [0.0 – 1.0]
        self._threat_level: float = 0.0

        logger.info("ImmuneSystem initialised")

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def trust_source(self, source: str) -> None:
        name = source.lower()
        self._trusted.add(name)
        self._blacklisted.discard(name)

    def blacklist_source(self, source: str) -> None:
        name = source.lower()
        self._blacklisted.add(name)
        self._trusted.discard(name)

    def classify_source(self, source: str) -> SourceTrust:
        name = source.lower()
        if name in self._blacklisted:
            return SourceTrust.BLACKLISTED
        if name in self._trusted:
            return SourceTrust.TRUSTED
        return SourceTrust.UNKNOWN

    # ------------------------------------------------------------------
    # Core inspection
    # ------------------------------------------------------------------

    def inspect(self, data: dict) -> dict:
        """
        Examine a single incoming data item.

        Returns a result dict with keys:
            allowed        (bool)
            action         (str)  – 'pass' | 'quarantine' | 'block'
            source_trust   (str)  – Trusted / Unknown / Blacklisted
            threat_score   (float) – 0.0 to 1.0
            flags          (list[str])
            item_id        (str)
        """
        self._total_inspected += 1

        source  = str(data.get("source", "unknown")).lower()
        content = str(data.get("content", ""))
        item_id = self._make_id(data)

        flags: List[str] = []
        threat: float    = 0.0

        trust = self.classify_source(source)

        # --- Rule 1: blacklisted source → immediate block ---------------
        if trust == SourceTrust.BLACKLISTED:
            threat = 1.0
            flags.append("blacklisted_source")
            self._total_blocked += 1
            self._update_threat(threat)
            return self._result(item_id, allowed=False, action="block",
                                trust=trust, threat=threat, flags=flags)

        # --- Rule 2: flood detection ------------------------------------
        content_hash = self._hash(content)
        if content_hash in self._blocked_hashes:
            threat = 0.95
            flags.append("flood_blocked_hash")
            self._total_blocked += 1
            self._update_threat(threat)
            return self._result(item_id, allowed=False, action="block",
                                trust=trust, threat=threat, flags=flags)

        if trust == SourceTrust.UNKNOWN:
            flood_hit = self._register_flood(content_hash, source)
            if flood_hit:
                self._blocked_hashes.add(content_hash)
                threat = 0.95
                flags.append("flood_attack_detected")
                self._total_blocked += 1
                self._update_threat(threat)
                return self._result(item_id, allowed=False, action="block",
                                    trust=trust, threat=threat, flags=flags)

        # --- Rule 3: contradiction detection ----------------------------
        conflict_ratio = self._check_contradictions(content)
        if conflict_ratio > 0:
            threat = max(threat, conflict_ratio)
            if conflict_ratio > CONFLICT_FLAG_PCT:
                flags.append(f"high_conflict_{conflict_ratio:.0%}")
                # Quarantine rather than outright block
                self._quarantine[item_id] = {**data,
                                              "_item_id": item_id,
                                              "_conflict_ratio": conflict_ratio}
                self._total_quarantined += 1
                self._update_threat(threat)
                return self._result(item_id, allowed=False, action="quarantine",
                                    trust=trust, threat=threat, flags=flags)
            else:
                flags.append(f"minor_conflict_{conflict_ratio:.0%}")

        # --- Unknown-source penalty -------------------------------------
        if trust == SourceTrust.UNKNOWN:
            threat = max(threat, 0.30)
            flags.append("unknown_source")

        # --- Allow ------------------------------------------------------
        self._total_passed += 1
        self._update_threat(threat)
        return self._result(item_id, allowed=True, action="pass",
                            trust=trust, threat=threat, flags=flags)

    def quarantine(self, item_id: str) -> None:
        """Manually move an item (by its item_id) into quarantine."""
        if item_id not in self._quarantine:
            self._quarantine[item_id] = {"_item_id": item_id, "_manual": True}
            self._total_quarantined += 1
            logger.warning("Item %s manually quarantined", item_id)

    def release_from_quarantine(self, item_id: str) -> Optional[dict]:
        """Remove and return an item from quarantine (approve it)."""
        return self._quarantine.pop(item_id, None)

    def get_threat_level(self) -> float:
        """Current aggregate threat level in [0.0, 1.0]."""
        return round(self._threat_level, 4)

    def summary(self) -> dict:
        return {
            "engine": "ImmuneSystem",
            "total_inspected":    self._total_inspected,
            "total_passed":       self._total_passed,
            "total_quarantined":  self._total_quarantined,
            "total_blocked":      self._total_blocked,
            "quarantine_size":    len(self._quarantine),
            "blocked_hashes":     len(self._blocked_hashes),
            "trusted_sources":    len(self._trusted),
            "blacklisted_sources": len(self._blacklisted),
            "current_threat_level": self.get_threat_level(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_flood(self, content_hash: str, source: str) -> bool:
        """Return True if flood threshold has been exceeded."""
        now = time.monotonic()
        dq  = self._flood_ledger[content_hash]

        # Evict entries outside window
        while dq and dq[0][0] < now - FLOOD_WINDOW_SEC:
            dq.popleft()

        dq.append((now, source))

        # Count entries from unknown sources only
        unknown_count = sum(
            1 for _, s in dq
            if s not in self._trusted and s not in self._blacklisted
        )
        if unknown_count >= FLOOD_THRESHOLD:
            logger.warning(
                "Flood detected: %d identical items from unknown sources "
                "within %ds for hash %s", unknown_count, FLOOD_WINDOW_SEC, content_hash
            )
            return True
        return False

    def _check_contradictions(self, content: str) -> float:
        """
        Returns a contradiction ratio in [0.0, 1.0].
        Requires a knowledge_store with iterable fact entries.
        """
        if self._ks is None:
            return 0.0
        try:
            facts: Any = self._ks.get("facts") or []
            if not facts:
                return 0.0
            content_lower = content.lower()
            contradictions = 0
            checked = 0
            for fact in facts:
                fact_text = str(fact.get("content", "") if isinstance(fact, dict) else fact).lower()
                if not fact_text:
                    continue
                checked += 1
                # Simple heuristic: if fact negates a key phrase in content
                if self._texts_contradict(content_lower, fact_text):
                    contradictions += 1
            return (contradictions / checked) if checked else 0.0
        except Exception as exc:
            logger.debug("Contradiction check failed: %s", exc)
            return 0.0

    @staticmethod
    def _texts_contradict(a: str, b: str) -> bool:
        """
        Lightweight contradiction heuristic.
        True when one text contains 'not <word>' and the other contains that
        word without the negation (or vice-versa).
        """
        negation_markers = [" not ", " never ", " no ", " false ", " incorrect "]
        for marker in negation_markers:
            if marker in a and marker not in b:
                # Check for shared keyword
                words_a = set(a.split())
                words_b = set(b.split())
                if len(words_a & words_b) >= 3:
                    return True
        return False

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]

    @staticmethod
    def _make_id(data: dict) -> str:
        key = f"{data.get('source','')}|{data.get('content','')}|{data.get('timestamp','')}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _update_threat(self, sample: float) -> None:
        """Exponential moving average of threat level (α = 0.1)."""
        self._threat_level = 0.9 * self._threat_level + 0.1 * sample

    @staticmethod
    def _result(item_id, *, allowed, action, trust, threat, flags) -> dict:
        return {
            "item_id":      item_id,
            "allowed":      allowed,
            "action":       action,
            "source_trust": trust.value,
            "threat_score": round(threat, 4),
            "flags":        flags,
        }


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    immune = ImmuneSystem()
    immune.blacklist_source("spam_bot")

    items = [
        {"source": "system",    "content": "Core boot complete.",            "timestamp": "2026-06-07T00:00:00Z"},
        {"source": "unknown",   "content": "Breaking news from nowhere.",     "timestamp": "2026-06-07T01:00:00Z"},
        {"source": "spam_bot",  "content": "Buy cheap memory upgrades now!",  "timestamp": "2026-06-07T01:05:00Z"},
        {"source": "verified",  "content": "v14.0.0 released successfully.",  "timestamp": "2026-06-07T02:00:00Z"},
    ]

    print("=== ImmuneSystem Demo ===")
    for item in items:
        result = immune.inspect(item)
        print(f"  [{result['action'].upper():10s}] trust={result['source_trust']:12s} "
              f"threat={result['threat_score']:.2f}  flags={result['flags']}")

    print("\n--- Flood test (1000 identical unknown items) ---")
    flood_item = {"source": "bot_farm", "content": "IDENTICAL FLOOD MESSAGE"}
    blocked = 0
    for _ in range(1001):
        r = immune.inspect(flood_item)
        if not r["allowed"]:
            blocked += 1
    print(f"  Blocked {blocked} flood items out of 1001")

    print("\n--- Summary ---")
    for k, v in immune.summary().items():
        print(f"  {k}: {v}")
