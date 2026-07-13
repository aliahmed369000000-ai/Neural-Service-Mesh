"""
Knowledge Sources — Source Validator
======================================
Every piece of information passes through this validator BEFORE
it is allowed to enter the memory engine or knowledge store.

Validation checks:
  1. Completeness   — required fields present and non-empty
  2. Duplication    — hash-based fingerprint deduplication
  3. Corruption     — encoding, length, structure sanity
  4. Trustworthiness — source trust score threshold

The validator is STATEFUL (maintains seen-hashes) so it can detect
duplicates within a session. On restart, hashes reload from disk.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from knowledge_sources.source_metadata import (
    KnowledgeItem, SourceMetadata, SourceType
)

logger = logging.getLogger(__name__)

_HASH_STORE_PATH = Path("./data/ks_seen_hashes.json")
_MIN_CONTENT_LEN = 3
_MAX_CONTENT_LEN = 500_000   # 500 KB per item


# ── Validation Result ──────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    passed:          bool = False
    score:           float = 0.0        # 0–100 overall validation confidence
    checks:          Dict[str, bool] = field(default_factory=dict)
    warnings:        List[str]       = field(default_factory=list)
    errors:          List[str]       = field(default_factory=list)
    fingerprint:     str             = ""
    is_duplicate:    bool            = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Validator ──────────────────────────────────────────────────────────────

class SourceValidator:
    """
    Stateful validator for all incoming KnowledgeItems.

    Thread-safe. Persists seen fingerprints to disk so duplicates are
    detected across restarts for long-running sync sources.
    """

    def __init__(
        self,
        min_trust_threshold: float = 0.3,
        deduplicate: bool = True,
        hash_store_path: str = str(_HASH_STORE_PATH),
    ):
        self._min_trust    = min_trust_threshold
        self._deduplicate  = deduplicate
        self._hash_path    = Path(hash_store_path)
        self._seen_hashes: Set[str] = set()
        self._lock         = threading.Lock()
        self._total_validated  = 0
        self._total_rejected   = 0
        self._load_hashes()
        logger.info("[SourceValidator] initialised")

    # ── Persistence ────────────────────────────────────────────────────────

    def _load_hashes(self) -> None:
        try:
            if self._hash_path.exists():
                with open(self._hash_path) as f:
                    data = json.load(f)
                self._seen_hashes = set(data.get("hashes", []))
                logger.info(f"[SourceValidator] loaded {len(self._seen_hashes)} fingerprints")
        except Exception as exc:
            logger.warning(f"[SourceValidator] hash load error: {exc}")

    def _persist_hashes(self) -> None:
        try:
            self._hash_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._hash_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump({"hashes": list(self._seen_hashes),
                           "updated_at": datetime.now(timezone.utc).isoformat()}, f)
            tmp.replace(self._hash_path)
        except Exception as exc:
            logger.warning(f"[SourceValidator] hash persist error: {exc}")

    # ── Fingerprinting ─────────────────────────────────────────────────────

    @staticmethod
    def fingerprint(item: KnowledgeItem) -> str:
        """SHA-256 of (source_id + raw_reference + raw_content[:512])."""
        blob = f"{item.source_id}|{item.raw_reference}|{item.raw_content[:512]}"
        return hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()

    # ── Main Validate Method ───────────────────────────────────────────────

    def validate(
        self,
        item: KnowledgeItem,
        source: Optional[SourceMetadata] = None,
    ) -> ValidationResult:
        """
        Validate a KnowledgeItem before ingestion.

        Returns a ValidationResult with .passed indicating go/no-go.
        """
        result = ValidationResult()
        fp     = self.fingerprint(item)
        result.fingerprint = fp

        with self._lock:
            self._total_validated += 1

            # 1. Completeness check
            ok_complete, warnings_c, errors_c = self._check_completeness(item)
            result.checks["completeness"] = ok_complete
            result.warnings.extend(warnings_c)
            result.errors.extend(errors_c)

            # 2. Corruption check
            ok_corrupt, warnings_k, errors_k = self._check_corruption(item)
            result.checks["corruption"] = ok_corrupt
            result.warnings.extend(warnings_k)
            result.errors.extend(errors_k)

            # 3. Duplication check
            ok_dup = True
            if self._deduplicate:
                if fp in self._seen_hashes:
                    result.is_duplicate = True
                    ok_dup = False
                    result.warnings.append(f"Duplicate fingerprint: {fp[:16]}…")
                result.checks["duplication"] = ok_dup
            else:
                result.checks["duplication"] = True

            # 4. Trustworthiness check
            ok_trust, warnings_t, errors_t = self._check_trust(item, source)
            result.checks["trustworthiness"] = ok_trust
            result.warnings.extend(warnings_t)
            result.errors.extend(errors_t)

            # ── Overall decision ─────────────────────────────────────────
            all_passed = all(result.checks.values()) and not result.errors
            result.passed = all_passed

            if all_passed:
                # Register fingerprint
                self._seen_hashes.add(fp)
                if len(self._seen_hashes) % 100 == 0:
                    self._persist_hashes()
            else:
                self._total_rejected += 1

            # ── Compute validation score (0–100) ─────────────────────────
            check_score   = sum(result.checks.values()) / max(len(result.checks), 1)
            penalty       = len(result.errors) * 10 + len(result.warnings) * 3
            result.score  = max(0.0, round(check_score * 100 - penalty, 2))

            return result

    def validate_batch(
        self,
        items: List[KnowledgeItem],
        source: Optional[SourceMetadata] = None,
    ) -> Tuple[List[KnowledgeItem], List[ValidationResult]]:
        """Validate a list; returns (accepted_items, all_results)."""
        accepted = []
        results  = []
        for item in items:
            res = self.validate(item, source)
            results.append(res)
            if res.passed:
                accepted.append(item)
        logger.info(
            f"[SourceValidator] batch: {len(accepted)}/{len(items)} accepted"
        )
        return accepted, results

    # ── Individual Checks ──────────────────────────────────────────────────

    def _check_completeness(
        self, item: KnowledgeItem
    ) -> Tuple[bool, List[str], List[str]]:
        warnings, errors = [], []

        if not item.source_id:
            errors.append("Missing source_id")
        if not item.raw_content or len(item.raw_content.strip()) < _MIN_CONTENT_LEN:
            errors.append(f"raw_content too short (min {_MIN_CONTENT_LEN} chars)")
        if not item.raw_reference:
            warnings.append("raw_reference is empty — provenance will be weak")
        if not item.item_id:
            errors.append("Missing item_id")

        return len(errors) == 0, warnings, errors

    def _check_corruption(
        self, item: KnowledgeItem
    ) -> Tuple[bool, List[str], List[str]]:
        warnings, errors = [], []

        content = item.raw_content or ""

        # Length sanity
        if len(content) > _MAX_CONTENT_LEN:
            errors.append(
                f"raw_content exceeds max length ({len(content)} > {_MAX_CONTENT_LEN})"
            )

        # Null bytes
        if "\x00" in content:
            errors.append("raw_content contains null bytes (binary corruption)")

        # Encoding check
        try:
            content.encode("utf-8")
        except UnicodeEncodeError as e:
            errors.append(f"Encoding error: {e}")

        # Repetition check (>90% same char is suspicious)
        if content and len(set(content)) == 1 and len(content) > 20:
            errors.append("raw_content is all the same character (corrupted)")

        # Whitespace-only
        if content and not content.strip():
            errors.append("raw_content is whitespace only")

        return len(errors) == 0, warnings, errors

    def _check_trust(
        self,
        item: KnowledgeItem,
        source: Optional[SourceMetadata],
    ) -> Tuple[bool, List[str], List[str]]:
        warnings, errors = [], []

        effective_trust = item.trust_score
        if source:
            effective_trust = max(effective_trust, source.trust_score)

        if effective_trust < self._min_trust:
            errors.append(
                f"Trust score {effective_trust:.2f} below threshold {self._min_trust:.2f}"
            )

        # Scripture sources get a trust floor of 0.99
        if item.source_type == SourceType.SCRIPTURE and effective_trust < 0.99:
            warnings.append(
                "Scripture source trust should be 1.0 — check source registration"
            )

        return len(errors) == 0, warnings, errors

    # ── Admin ──────────────────────────────────────────────────────────────

    def flush_hashes(self) -> None:
        """Persist all seen hashes to disk now."""
        with self._lock:
            self._persist_hashes()

    def clear_hashes(self) -> None:
        """Clear deduplication memory (useful for full re-sync)."""
        with self._lock:
            self._seen_hashes.clear()
        logger.info("[SourceValidator] fingerprint cache cleared")

    def summary(self) -> Dict[str, Any]:
        return {
            "total_validated":   self._total_validated,
            "total_rejected":    self._total_rejected,
            "seen_fingerprints": len(self._seen_hashes),
            "min_trust_threshold": self._min_trust,
            "deduplication_enabled": self._deduplicate,
        }
