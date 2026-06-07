"""
Phase 15 – Brain Checkpoint
============================
الهوية الدائمة عبر إعادة التشغيل.

يحفظ حالة الدماغ الكاملة على القرص ويسترجعها تلقائياً عند التشغيل.

ما يحفظه:
  • Neural weights (أوزان الشبكة العصبية)
  • Episodic memory snapshot (لقطة الذاكرة الإيبيسودية)
  • World model state  (حالة نموذج العالم)
  • System DNA snapshot (بصمة الحمض النووي الرقمي)
  • Metadata: version, timestamp, fitness

سياسة الـ rollback:
  • يحتفظ بآخر 5 نسخ
  • يرتّبها بالتاريخ ويحذف الأقدم تلقائياً
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT_DIR = "./checkpoints"
MAX_CHECKPOINTS        = 5          # أقصى عدد نسخ محفوظة
DEFAULT_INTERVAL_MIN   = 10         # دقائق بين كل حفظ تلقائي


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_tag() -> str:
    """Timestamp tag suitable for filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_serialize(obj: Any) -> Any:
    """Convert numpy arrays / non-JSON types to serialisable form."""
    if obj is None:
        return None
    # numpy ndarray
    if hasattr(obj, "tolist"):
        return obj.tolist()
    # dict → recurse
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    # list / tuple → recurse
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(i) for i in obj]
    # scalar fallback
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ---------------------------------------------------------------------------
# BrainCheckpoint
# ---------------------------------------------------------------------------

class BrainCheckpoint:
    """
    Manages saving and loading of the full brain state.

    Parameters
    ----------
    checkpoint_dir : str
        Directory where checkpoint files are stored.
    max_checkpoints : int
        Maximum number of rolling checkpoints to keep.
    """

    def __init__(
        self,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        max_checkpoints: int = MAX_CHECKPOINTS,
    ):
        self._dir            = checkpoint_dir
        self._max            = max_checkpoints
        self._lock           = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._running        = False
        self._saves_count    = 0
        self._loads_count    = 0
        self._last_save_path: Optional[str] = None
        self._last_save_ts:   Optional[str] = None

        os.makedirs(self._dir, exist_ok=True)
        logger.info(f"BrainCheckpoint ready  dir={self._dir}  max={self._max}")

    # ── Core: Save ─────────────────────────────────────────────────────────

    def save(self, mesh) -> str:
        """
        Save the complete brain state to disk.

        Parameters
        ----------
        mesh : NeuralServiceMesh
            The live mesh instance to snapshot.

        Returns
        -------
        str
            Absolute path to the saved checkpoint file.
        """
        tag  = _ts_tag()
        name = f"brain_checkpoint_{tag}.json"
        path = os.path.join(self._dir, name)

        snapshot = self._extract_state(mesh)
        payload  = {
            "version":   getattr(mesh, "VERSION", "unknown"),
            "saved_at":  _now_iso(),
            "tag":       tag,
            "state":     snapshot,
        }

        with self._lock:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(_safe_serialize(payload), fh, indent=2)

            self._saves_count    += 1
            self._last_save_path  = path
            self._last_save_ts    = _now_iso()

            self._prune()

        logger.info(f"[BrainCheckpoint] Saved → {path}")
        return path

    # ── Core: Load ─────────────────────────────────────────────────────────

    def load(self, path: Optional[str] = None) -> dict:
        """
        Load a brain state snapshot.

        Parameters
        ----------
        path : str | None
            Path to a specific checkpoint file.  If None, the latest
            checkpoint in the directory is loaded automatically.

        Returns
        -------
        dict
            Full payload: { version, saved_at, tag, state }
        """
        if path is None:
            path = self._latest_path()
            if path is None:
                logger.warning("[BrainCheckpoint] No checkpoint found — starting fresh.")
                return {}

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self._loads_count += 1
        logger.info(f"[BrainCheckpoint] Loaded ← {path}  (version={payload.get('version')})")
        return payload

    # ── Auto-Save ──────────────────────────────────────────────────────────

    def auto_save_start(self, interval_minutes: float = DEFAULT_INTERVAL_MIN, mesh=None):
        """
        Start periodic auto-save in a background thread.

        Parameters
        ----------
        interval_minutes : float
            How often to save (in minutes).
        mesh : NeuralServiceMesh
            The mesh instance to snapshot.  Can be updated later via
            set_mesh().
        """
        if self._running:
            logger.warning("[BrainCheckpoint] auto_save already running.")
            return

        self._mesh     = mesh
        self._interval = interval_minutes * 60.0
        self._running  = True
        self._schedule_next()
        logger.info(
            f"[BrainCheckpoint] Auto-save started  interval={interval_minutes:.1f} min"
        )

    def auto_save_stop(self):
        """Stop the periodic auto-save timer."""
        self._running = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        logger.info("[BrainCheckpoint] Auto-save stopped.")

    def set_mesh(self, mesh):
        """Update the mesh reference used by auto-save."""
        self._mesh = mesh

    def _schedule_next(self):
        if not self._running:
            return
        self._timer = threading.Timer(self._interval, self._auto_tick)
        self._timer.daemon = True
        self._timer.start()

    def _auto_tick(self):
        try:
            if hasattr(self, "_mesh") and self._mesh is not None:
                self.save(self._mesh)
        except Exception as exc:
            logger.error(f"[BrainCheckpoint] Auto-save error: {exc}")
        finally:
            self._schedule_next()

    # ── List & Prune ───────────────────────────────────────────────────────

    def list_checkpoints(self) -> List[dict]:
        """
        Return metadata for all saved checkpoints, sorted newest-first.

        Returns
        -------
        list[dict]
            Each entry: { path, tag, size_kb, saved_at }
        """
        files = sorted(
            [
                f for f in os.listdir(self._dir)
                if f.startswith("brain_checkpoint_") and f.endswith(".json")
            ],
            reverse=True,
        )
        results = []
        for fname in files:
            fpath = os.path.join(self._dir, fname)
            size_kb = round(os.path.getsize(fpath) / 1024.0, 2)
            # Quick read of saved_at without loading full state
            saved_at = "unknown"
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                    saved_at = meta.get("saved_at", "unknown")
                    version  = meta.get("version", "?")
            except Exception:
                version = "?"
            results.append({
                "path":     fpath,
                "filename": fname,
                "version":  version,
                "saved_at": saved_at,
                "size_kb":  size_kb,
            })
        return results

    def _latest_path(self) -> Optional[str]:
        """Return path of the most recent checkpoint, or None."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return checkpoints[0]["path"]

    def _prune(self):
        """Remove oldest checkpoints beyond self._max."""
        checkpoints = self.list_checkpoints()
        while len(checkpoints) > self._max:
            oldest = checkpoints.pop()
            try:
                os.remove(oldest["path"])
                logger.info(f"[BrainCheckpoint] Pruned old checkpoint: {oldest['filename']}")
            except OSError as exc:
                logger.warning(f"[BrainCheckpoint] Could not prune {oldest['path']}: {exc}")

    # ── State Extraction ───────────────────────────────────────────────────

    def _extract_state(self, mesh) -> dict:
        """
        Pull the most important state blobs from the mesh.

        Designed to be forward-compatible: any attribute that doesn't
        exist is silently skipped.
        """
        state: Dict[str, Any] = {}

        # --- Neural weights ---
        try:
            nl = getattr(mesh, "neural_layer", None)
            if nl is not None:
                state["neural_weights"] = {
                    "weights":     getattr(nl, "weights", None),
                    "train_steps": getattr(nl, "_train_steps", 0),
                }
        except Exception as exc:
            logger.debug(f"[Checkpoint] neural_weights skip: {exc}")

        # --- Dynamic weight layer ---
        try:
            dl = getattr(mesh, "dynamic_layer", None)
            if dl is not None and hasattr(dl, "summary"):
                state["dynamic_layer"] = dl.summary()
        except Exception as exc:
            logger.debug(f"[Checkpoint] dynamic_layer skip: {exc}")

        # --- Deep routing network ---
        try:
            dn = getattr(mesh, "deep_network", None)
            if dn is not None and hasattr(dn, "summary"):
                state["deep_network"] = dn.summary()
                # Try to grab raw weight matrices
                if hasattr(dn, "_layers"):
                    state["deep_network"]["raw_layers"] = [
                        {"W": getattr(l, "W", None), "b": getattr(l, "b", None)}
                        for l in dn._layers
                    ]
        except Exception as exc:
            logger.debug(f"[Checkpoint] deep_network skip: {exc}")

        # --- Episodic memory ---
        try:
            em = getattr(mesh, "episodic_memory", None)
            if em is not None and hasattr(em, "summary"):
                state["episodic_memory"] = em.summary()
                # Grab recent episodes list if available
                if hasattr(em, "_episodic"):
                    recent = list(em._episodic)[-200:]  # last 200
                    state["episodic_memory"]["recent_episodes"] = [
                        e.__dict__ if hasattr(e, "__dict__") else str(e)
                        for e in recent
                    ]
        except Exception as exc:
            logger.debug(f"[Checkpoint] episodic_memory skip: {exc}")

        # --- World model ---
        try:
            env = getattr(mesh, "env_model", None)
            if env is not None and hasattr(env, "get_world_summary"):
                state["world_model"] = env.get_world_summary()
            elif env is not None and hasattr(env, "summary"):
                state["world_model"] = env.summary()
        except Exception as exc:
            logger.debug(f"[Checkpoint] world_model skip: {exc}")

        # --- System DNA ---
        try:
            dna = getattr(mesh, "system_dna", None)
            if dna is not None and hasattr(dna, "get_dna_snapshot"):
                state["system_dna"] = dna.get_dna_snapshot()
            elif dna is not None and hasattr(dna, "summary"):
                state["system_dna"] = dna.summary()
        except Exception as exc:
            logger.debug(f"[Checkpoint] system_dna skip: {exc}")

        # --- Self-awareness ---
        try:
            sa = getattr(mesh, "self_awareness", None)
            if sa is not None and hasattr(sa, "summary"):
                state["self_awareness"] = sa.summary()
        except Exception as exc:
            logger.debug(f"[Checkpoint] self_awareness skip: {exc}")

        # --- Structural evolution DNA ---
        try:
            se = getattr(mesh, "structural_evolution", None)
            if se is not None and hasattr(se, "current_architecture"):
                state["structural_architecture"] = se.current_architecture()
        except Exception as exc:
            logger.debug(f"[Checkpoint] structural_evolution skip: {exc}")

        # --- Knowledge store snapshot (top-level keys only) ---
        try:
            ks = getattr(mesh, "knowledge", None)
            if ks is not None and hasattr(ks, "keys"):
                state["knowledge_keys"] = list(ks.keys())[:500]
        except Exception as exc:
            logger.debug(f"[Checkpoint] knowledge_keys skip: {exc}")

        # --- VERSION + timestamp ---
        state["meta"] = {
            "version":    getattr(mesh, "VERSION", "unknown"),
            "saved_at":   _now_iso(),
            "saves_done": self._saves_count + 1,
        }

        return state

    # ── Summary ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        checkpoints = self.list_checkpoints()
        return {
            "component":       "BrainCheckpoint",
            "checkpoint_dir":  self._dir,
            "max_checkpoints": self._max,
            "saved_count":     self._saves_count,
            "loaded_count":    self._loads_count,
            "auto_save":       self._running,
            "interval_min":    getattr(self, "_interval", 0) / 60.0,
            "checkpoints":     checkpoints,
            "latest":          checkpoints[0] if checkpoints else None,
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, json

    print("=" * 60)
    print("  BrainCheckpoint — Self-Test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        bc = BrainCheckpoint(checkpoint_dir=tmpdir, max_checkpoints=3)

        # Fake mesh object
        class FakeMesh:
            VERSION = "15.0.0"

        mesh = FakeMesh()

        paths = []
        for i in range(4):
            p = bc.save(mesh)
            paths.append(p)
            print(f"  Saved [{i+1}]: {os.path.basename(p)}")
            time.sleep(0.05)

        checkpoints = bc.list_checkpoints()
        print(f"\n  Stored checkpoints (max={bc._max}): {len(checkpoints)}")
        assert len(checkpoints) <= 3, "Prune failed!"

        data = bc.load()
        print(f"  Loaded version : {data.get('version')}")
        print(f"  Summary        : saves={bc._saves_count}  loads={bc._loads_count}")

        print("\n  summary():")
        s = bc.summary()
        print(f"    auto_save={s['auto_save']}  saved_count={s['saved_count']}")

    print("\n✓ BrainCheckpoint self-test PASSED")
