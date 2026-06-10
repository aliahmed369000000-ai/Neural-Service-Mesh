"""
Quran Continuous Trainer
========================
Runs on startup and every 30 minutes in a background thread.

Pipeline per batch (100 ayahs):
  1. Read cursor from data/quran_training_cursor.json
  2. Load ayahs from knowledge/quran_chunk_*.json from cursor position
  3. Extract concepts per ayah (ConceptExtractor)
  4. Ingest concepts + relations into live CKG (ingest_batch)
  5. Run RelationInferencer on the live CKG
  6. Save knowledge/cognitive_graph.json
  7. Take top-3 concepts from batch → one KnowledgeTrainer step → mesh.db row
  8. Save checkpoints/neural_weights.npy
  9. Update data/quran_training_cursor.json
 10. Repeat until all 6236 ayahs processed, then reschedule in 30 min
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_CHUNK_DIR    = Path("./knowledge")
_CURSOR_PATH  = Path("./data/quran_training_cursor.json")
_WEIGHTS_DIR  = Path("./checkpoints")
_WEIGHTS_PATH = _WEIGHTS_DIR / "neural_weights.npy"
_INTERVAL_SEC = 30 * 60   # 30 minutes between full-pass reschedules
_BATCH_SIZE   = 100


# ── Helpers ────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_cursor() -> Dict:
    if _CURSOR_PATH.exists():
        try:
            with open(_CURSOR_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "last_ayah_processed":  0,
        "total_concepts":       0,
        "total_relations":      0,
        "last_run":             None,
        "total_training_steps": 0,
    }


def _save_cursor(cursor: Dict) -> None:
    _CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(_CURSOR_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cursor, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _CURSOR_PATH)


def _load_all_ayahs() -> List[Dict]:
    """Load every ayah from quran_chunk_*.json in sorted order."""
    chunks = sorted(_CHUNK_DIR.glob("quran_chunk_*.json"))
    ayahs: List[Dict] = []
    for chunk_path in chunks:
        try:
            with open(chunk_path, encoding="utf-8") as f:
                ayahs.extend(json.load(f))
        except Exception as exc:
            logger.warning(f"[QCT] Could not load {chunk_path}: {exc}")
    return ayahs


def _save_weights(mesh) -> bool:
    """Save neural_layer weights to checkpoints/neural_weights.npy."""
    try:
        layer = getattr(mesh, "neural_layer", None)
        if layer is None:
            return False
        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        layer.save(str(_WEIGHTS_PATH))
        return True
    except Exception as exc:
        logger.warning(f"[QCT] Weight save failed: {exc}")
        return False


# ── Core batch-processing logic ────────────────────────────────────────────

def _process_batch(
    batch:    List[Dict],
    ckg,
    inferencer_cls,
    trainer,
    cursor:   Dict,
) -> Dict:
    """
    Process one batch of ayahs.
    Returns updated stats to merge into cursor.
    """
    from knowledge_sources.concept_extractor import ConceptExtractor

    extractor = ConceptExtractor(max_concepts=8, min_score=0.12)

    # ── 1. Extract concepts for every ayah ───────────────────────────────
    all_matches: List[List] = []
    references:  List[str]  = []
    concept_freq: Dict[str, Dict] = {}   # concept_name → {cluster, count}

    for ayah in batch:
        text = ayah.get("text_norm") or ayah.get("text", "")
        ref  = f"{ayah.get('surah', 0)}:{ayah.get('ayah', 0)}"
        matches = extractor.extract(text, reference=ref)
        all_matches.append(matches)
        references.append(ref)

        # Accumulate concept frequency for this batch
        for m in matches:
            if m.cluster == "هيكل":
                continue
            if m.concept not in concept_freq:
                concept_freq[m.concept] = {"cluster": m.cluster, "count": 0}
            concept_freq[m.concept]["count"] += 1

    # ── 2. Ingest into live CKG ───────────────────────────────────────────
    result = ckg.ingest_batch(all_matches, references=references, auto_save=False)

    # ── 3. Run RelationInferencer ─────────────────────────────────────────
    try:
        inf = inferencer_cls(ckg)
        inf.run(verbose=False)
    except Exception as exc:
        logger.warning(f"[QCT] RelationInferencer error: {exc}")

    # ── 4. Save CKG to disk ───────────────────────────────────────────────
    try:
        ckg.save()
    except Exception as exc:
        logger.warning(f"[QCT] CKG save error: {exc}")

    # ── 5. KnowledgeTrainer step — top-3 concepts from batch ─────────────
    train_steps_added = 0
    if trainer is not None and concept_freq:
        top3 = sorted(concept_freq.items(), key=lambda x: -x[1]["count"])[:3]
        items = [
            {
                "concept":     name,
                "text":        f"مفهوم قرآني: {name}",
                "cluster":     info["cluster"],
                "importance":  min(info["count"] / 10.0, 1.0),
                "certainty":   0.85,
                "abstraction": 0.5,
            }
            for name, info in top3
        ]
        try:
            res = trainer.train_domain("quran", items)
            train_steps_added = res.get("train_steps", 0)
        except Exception as exc:
            logger.warning(f"[QCT] KnowledgeTrainer step error: {exc}")

    return {
        "concepts_added":    result.get("concepts_added", 0),
        "relations_added":   result.get("relations_added", 0),
        "total_concepts":    result.get("total_concepts",  ckg.concept_count()),
        "total_relations":   result.get("total_relations", ckg.relation_count()),
        "train_steps_added": train_steps_added,
    }


# ── Main trainer class ─────────────────────────────────────────────────────

class QuranContinuousTrainer:
    """
    Background trainer that continuously processes Quran ayahs.

    Usage:
        qct = QuranContinuousTrainer(mesh)
        qct.start()   # begins background thread
        qct.stop()    # graceful shutdown
    """

    def __init__(self, mesh):
        self._mesh    = mesh
        self._thread: Optional[threading.Thread] = None
        self._stop    = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("[QCT] Already running — ignoring start()")
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="QuranContinuousTrainer",
            daemon=True,
        )
        self._thread.start()
        logger.info("[QCT] Background thread started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("[QCT] Stopped")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_full_pass()
            except Exception as exc:
                logger.error(f"[QCT] Unhandled error in loop: {exc}", exc_info=True)

            # Wait 30 minutes before re-strengthening
            logger.info(f"[QCT] Full pass complete. Next run in {_INTERVAL_SEC // 60} min.")
            self._stop.wait(_INTERVAL_SEC)

    def _run_full_pass(self) -> None:
        """Process all ayahs from cursor position, then reset to 0 for next pass."""
        from knowledge.cognitive_graph import get_ckg
        from knowledge.relation_inferencer import RelationInferencer
        from ai.knowledge_trainer import KnowledgeTrainer

        ckg = get_ckg()
        trainer: Optional[KnowledgeTrainer] = None
        try:
            trainer = KnowledgeTrainer(self._mesh)
        except Exception as exc:
            logger.warning(f"[QCT] KnowledgeTrainer unavailable: {exc}")

        ayahs = _load_all_ayahs()
        if not ayahs:
            logger.warning("[QCT] No ayahs found — skipping pass")
            return

        cursor = _load_cursor()
        start  = cursor.get("last_ayah_processed", 0)

        # If we've already finished a full pass, restart from 0 to re-strengthen
        if start >= len(ayahs):
            logger.info("[QCT] All ayahs processed — restarting from 0 to strengthen relations")
            start = 0

        logger.info(
            f"[QCT] Starting pass: ayahs={len(ayahs)} "
            f"start={start} batch={_BATCH_SIZE}"
        )

        pos = start
        while pos < len(ayahs) and not self._stop.is_set():
            batch = ayahs[pos: pos + _BATCH_SIZE]
            if not batch:
                break

            stats = _process_batch(batch, ckg, RelationInferencer, trainer, cursor)

            # ── Save weights ──────────────────────────────────────────────
            _save_weights(self._mesh)

            # ── Update cursor ─────────────────────────────────────────────
            pos += len(batch)
            cursor["last_ayah_processed"]   = pos
            cursor["total_concepts"]        = stats["total_concepts"]
            cursor["total_relations"]       = stats["total_relations"]
            cursor["last_run"]              = _now_iso()
            cursor["total_training_steps"]  = (
                cursor.get("total_training_steps", 0) + stats["train_steps_added"]
            )
            _save_cursor(cursor)

            logger.info(
                f"[QCT] Batch done: pos={pos}/{len(ayahs)} "
                f"concepts={stats['total_concepts']} "
                f"relations={stats['total_relations']} "
                f"new_train_steps={stats['train_steps_added']}"
            )

        # Reset cursor position so next scheduled run re-processes from 0
        cursor["last_ayah_processed"] = len(ayahs)
        _save_cursor(cursor)
        logger.info(
            f"[QCT] Pass complete: total_concepts={cursor['total_concepts']} "
            f"total_relations={cursor['total_relations']} "
            f"total_training_steps={cursor['total_training_steps']}"
        )
