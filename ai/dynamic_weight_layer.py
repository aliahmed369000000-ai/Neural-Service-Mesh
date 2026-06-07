"""
Phase 9 — Axis 2: Dynamic Self-Growing Weight Layer
=====================================================
Replaces the fixed 9×7 NeuralWeightLayer (Phase 8) with a layer that
**grows its own weight matrix** when performance plateaus.

Growth rules
------------
  • Monitor the loss history in a rolling window.
  • If the *improvement* over the last PLATEAU_CHECK_STEPS steps
    is below PLATEAU_THRESHOLD (< 1% improvement), the network is
    considered to have plateaued.
  • On plateau: add GROW_ROWS new rows AND GROW_COLS new columns,
    initialising the new weights with Xavier/Glorot uniform noise.
  • A minimum cool-down of COOLDOWN_STEPS prevents over-expansion.
  • Maximum dimensions are capped at MAX_ROWS × MAX_COLS to prevent
    unbounded growth.

Growth trajectory (from the images):
    Start:         7×9
    After 100 rt:  12×9
    After 500 rt:  18×13
    After 1000 rt: 27×19

The class is a drop-in superset of NeuralWeightLayer:
  • forward(x)            — same signature, adapts to current shape
  • train_step(x, target) — same signature, triggers growth check
  • save/load             — stores shape metadata alongside weights
  • summary()             — extended with growth_events field

Backward compatibility
----------------------
`extract_routing_weights(layer)` from ai.neural_weights continues to
work because this class exposes the same `.weights` attribute and the
same first-row semantics.
"""
from __future__ import annotations

import logging
import math
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Growth configuration ──────────────────────────────────────────────────────
INITIAL_ROWS = 7       # matches image: "تبدأ: 7×9"
INITIAL_COLS = 9       # feature dim from RichDataCollector output
GROW_ROWS = 3          # add 3 rows on plateau
GROW_COLS = 2          # add 2 cols on plateau (expands capacity)
MAX_ROWS = 50
MAX_COLS = 30
PLATEAU_CHECK_STEPS = 20   # look at last 20 training steps
PLATEAU_THRESHOLD = 0.01   # < 1% improvement triggers growth
COOLDOWN_STEPS = 30        # minimum steps between two growths


class DynamicWeightLayer:
    """
    Phase 9 Axis-2: Self-growing neural weight layer.

    The weight matrix starts at INITIAL_ROWS × INITIAL_COLS and
    expands automatically when learning stagnates.

    Parameters
    ----------
    initial_rows : int
        Starting row count (default: INITIAL_ROWS = 7).
    initial_cols : int
        Starting col count / input feature dim (default: INITIAL_COLS = 9).
    learning_rate : float
        Step size for gradient updates. Default 0.01.
    name : str
        Human-readable label.
    """

    def __init__(
        self,
        initial_rows: int = INITIAL_ROWS,
        initial_cols: int = INITIAL_COLS,
        learning_rate: float = 0.01,
        name: str = "dynamic_weight_layer",
    ):
        self._rows = initial_rows
        self._cols = initial_cols
        self.learning_rate = learning_rate
        self.name = name

        # Initialise with Xavier uniform
        self.weights: np.ndarray = self._xavier_init(initial_rows, initial_cols)

        # Training bookkeeping
        self._train_steps: int = 0
        self._last_loss: Optional[float] = None
        self._loss_history: deque = deque(maxlen=500)
        self._growth_events: List[dict] = []
        self._steps_since_growth: int = 0

        logger.info(
            f"DynamicWeightLayer '{self.name}' initialised — "
            f"shape=({self._rows}×{self._cols})  lr={self.learning_rate}"
        )

    # ── Shape property ────────────────────────────────────────────────────

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._rows, self._cols)

    @property
    def SHAPE(self) -> Tuple[int, int]:
        """Compatibility alias for code that reads NeuralWeightLayer.SHAPE."""
        return self.shape

    # ── Xavier initialisation ─────────────────────────────────────────────

    @staticmethod
    def _xavier_init(rows: int, cols: int) -> np.ndarray:
        """Xavier/Glorot uniform initialisation: U[-limit, limit]."""
        limit = math.sqrt(6.0 / (rows + cols))
        return np.random.uniform(-limit, limit, size=(rows, cols)).astype(np.float64)

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Pass input vector through the dynamic weight matrix.

        If `x` is shorter than `self._cols`, it is zero-padded.
        If `x` is longer, it is truncated (shouldn't normally happen).

        Returns np.ndarray of shape (rows,).
        """
        x_arr = np.array(x, dtype=np.float64)
        # Pad or truncate to match current column count
        if x_arr.shape[0] < self._cols:
            x_arr = np.pad(x_arr, (0, self._cols - x_arr.shape[0]))
        elif x_arr.shape[0] > self._cols:
            x_arr = x_arr[:self._cols]

        output = self.weights @ x_arr          # (rows, cols) × (cols,) → (rows,)
        return np.maximum(0.0, output)         # ReLU activation

    # ── Training step ─────────────────────────────────────────────────────

    def train_step(
        self,
        input_vector: Union[List[float], np.ndarray],
        target: float,
    ) -> float:
        """
        Single supervised training step (MSE loss, gradient descent).

        Also checks for plateau and triggers growth when needed.

        Returns
        -------
        float — MSE loss before the update.
        """
        x = np.array(input_vector, dtype=np.float64)
        # Adapt input to current column count
        if x.shape[0] < self._cols:
            x = np.pad(x, (0, self._cols - x.shape[0]))
        elif x.shape[0] > self._cols:
            x = x[:self._cols]

        output = self.forward(x)
        target_vec = np.full(self._rows, target, dtype=np.float64)
        error = output - target_vec
        loss = float(np.mean(error ** 2))

        # Gradient of MSE w.r.t. weights
        grad = np.outer(error, x) / self._rows
        self.weights -= self.learning_rate * grad
        self.weights = np.clip(self.weights, -5.0, 5.0)

        self._train_steps += 1
        self._last_loss = loss
        self._loss_history.append(loss)
        self._steps_since_growth += 1

        # Check plateau every PLATEAU_CHECK_STEPS steps
        if (self._train_steps % PLATEAU_CHECK_STEPS == 0 and
                self._steps_since_growth >= COOLDOWN_STEPS):
            if self._is_plateauing():
                self._grow()

        logger.debug(
            f"DynamicWeightLayer train_step #{self._train_steps}  "
            f"loss={loss:.6f}  shape=({self._rows}×{self._cols})"
        )
        return loss

    # ── Plateau detection ─────────────────────────────────────────────────

    def _is_plateauing(self) -> bool:
        """
        Return True if the recent loss improvement is below PLATEAU_THRESHOLD.

        Compares mean loss of the older half vs the recent half of the
        last PLATEAU_CHECK_STEPS * 2 entries.
        """
        if len(self._loss_history) < PLATEAU_CHECK_STEPS * 2:
            return False
        window = list(self._loss_history)[-PLATEAU_CHECK_STEPS * 2:]
        mid = len(window) // 2
        mean_older  = float(np.mean(window[:mid]))
        mean_recent = float(np.mean(window[mid:]))
        if mean_older == 0.0:
            return False
        improvement = (mean_older - mean_recent) / mean_older
        logger.debug(
            f"Plateau check: older_loss={mean_older:.6f}  "
            f"recent_loss={mean_recent:.6f}  improvement={improvement:.4f}"
        )
        return improvement < PLATEAU_THRESHOLD

    # ── Growth ────────────────────────────────────────────────────────────

    def _grow(self) -> None:
        """
        Expand the weight matrix by adding GROW_ROWS rows and GROW_COLS cols.
        New weights are Xavier-initialised.
        """
        old_shape = (self._rows, self._cols)

        new_rows = min(self._rows + GROW_ROWS, MAX_ROWS)
        new_cols = min(self._cols + GROW_COLS, MAX_COLS)

        if new_rows == self._rows and new_cols == self._cols:
            logger.info(
                f"DynamicWeightLayer '{self.name}': already at maximum size "
                f"({MAX_ROWS}×{MAX_COLS}), skipping growth."
            )
            return

        # New weight matrix: copy existing + xavier-init new cells
        new_weights = np.zeros((new_rows, new_cols), dtype=np.float64)
        new_weights[:self._rows, :self._cols] = self.weights

        # Fill new rows
        if new_rows > self._rows:
            extra_rows = self._xavier_init(new_rows - self._rows, new_cols)
            new_weights[self._rows:, :] = extra_rows

        # Fill new columns for existing rows
        if new_cols > self._cols:
            extra_cols = self._xavier_init(self._rows, new_cols - self._cols)
            new_weights[:self._rows, self._cols:] = extra_cols

        self.weights = new_weights
        self._rows = new_rows
        self._cols = new_cols
        self._steps_since_growth = 0

        event = {
            "step": self._train_steps,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_shape": list(old_shape),
            "new_shape": [new_rows, new_cols],
            "trigger": "plateau",
            "loss_at_growth": round(self._last_loss, 8) if self._last_loss else None,
        }
        self._growth_events.append(event)

        logger.info(
            f"DynamicWeightLayer '{self.name}' GREW: "
            f"{old_shape[0]}×{old_shape[1]} → {new_rows}×{new_cols}  "
            f"(step {self._train_steps})"
        )

    # ── Compatibility with extract_routing_weights() ──────────────────────

    def get_routing_row(self) -> np.ndarray:
        """Return first row, first 4 cols — compatible with extract_routing_weights."""
        return self.weights[0, :min(4, self._cols)]

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> str:
        """Save weights and metadata to <path>.npy and <path>.meta.npy."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), self.weights)
        # Save shape as metadata
        meta = np.array([self._rows, self._cols, self._train_steps], dtype=np.int64)
        np.save(str(p).replace(".npy", "_meta.npy"), meta)
        logger.info(
            f"DynamicWeightLayer '{self.name}' saved → {p.resolve()}  "
            f"shape=({self._rows}×{self._cols})"
        )
        return str(p.resolve())

    def load(self, path: str) -> None:
        """Load weights (and optionally metadata) from .npy file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"DynamicWeightLayer.load: not found: {p}")
        loaded = np.load(str(p))
        self.weights = loaded.astype(np.float64)
        self._rows, self._cols = self.weights.shape

        # Try to load metadata
        meta_path = str(p).replace(".npy", "_meta.npy")
        if os.path.exists(meta_path):
            try:
                meta = np.load(meta_path)
                self._train_steps = int(meta[2])
            except Exception:
                pass

        logger.info(
            f"DynamicWeightLayer '{self.name}' loaded ← {p.resolve()}  "
            f"shape=({self._rows}×{self._cols})  steps={self._train_steps}"
        )

    # ── Introspection ─────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Extended summary including growth history."""
        return {
            "name": self.name,
            "shape": [self._rows, self._cols],
            "learning_rate": self.learning_rate,
            "train_steps": self._train_steps,
            "last_loss": round(self._last_loss, 8) if self._last_loss is not None else None,
            "growth_events_count": len(self._growth_events),
            "growth_events": self._growth_events[-5:],  # last 5 growths
            "weight_stats": {
                "min":  round(float(self.weights.min()), 6),
                "max":  round(float(self.weights.max()), 6),
                "mean": round(float(self.weights.mean()), 6),
                "std":  round(float(self.weights.std()), 6),
            },
            "plateau_check": {
                "window_steps":  PLATEAU_CHECK_STEPS,
                "threshold":     PLATEAU_THRESHOLD,
                "cooldown":      COOLDOWN_STEPS,
                "steps_since_growth": self._steps_since_growth,
            },
            "max_size": f"{MAX_ROWS}×{MAX_COLS}",
        }

    def get_weights_list(self) -> List[List[float]]:
        """Return weight matrix as plain Python list of lists."""
        return self.weights.tolist()

    def __repr__(self) -> str:
        return (
            f"<DynamicWeightLayer name='{self.name}' "
            f"shape=({self._rows}×{self._cols}) "
            f"steps={self._train_steps} "
            f"growths={len(self._growth_events)}>"
        )


# ── Routing weight extractor (Phase 8 API compatible) ────────────────────────

def extract_routing_weights_dynamic(layer: DynamicWeightLayer) -> dict:
    """
    Derive the 4 RoutingEngine weight scalars from a DynamicWeightLayer.

    Mirrors the Phase 8 `extract_routing_weights()` function so existing
    callers work unchanged.
    """
    row0 = layer.weights[0, :min(4, layer._cols)]
    total = float(row0.sum())
    if total <= 0.0:
        logger.warning("extract_routing_weights_dynamic: row0 sum ≤ 0, using defaults")
        return {
            "W_SEMANTIC": 0.30,
            "W_SCORE":    0.35,
            "W_MEMORY":   0.25,
            "W_TOPOLOGY": 0.10,
        }
    # Pad to 4 elements if fewer columns
    normed_arr = (row0 / total)
    while len(normed_arr) < 4:
        normed_arr = np.append(normed_arr, 0.0)
        total = float(normed_arr.sum())
        if total > 0:
            normed_arr = normed_arr / total
    normed = normed_arr[:4].tolist()
    return {
        "W_SEMANTIC": round(normed[0], 6),
        "W_SCORE":    round(normed[1], 6),
        "W_MEMORY":   round(normed[2], 6),
        "W_TOPOLOGY": round(normed[3], 6),
    }


# ── Module-level singleton ────────────────────────────────────────────────────

_default_dynamic_layer: Optional[DynamicWeightLayer] = None


def get_default_dynamic_layer(
    weights_path: str = "models/classifiers/dynamic_weights.npy",
) -> DynamicWeightLayer:
    """
    Return (and cache) the module-level default DynamicWeightLayer.

    Loads persisted weights from `weights_path` if the file exists.
    """
    global _default_dynamic_layer
    if _default_dynamic_layer is None:
        _default_dynamic_layer = DynamicWeightLayer(name="default_dynamic_layer")
        if os.path.exists(weights_path):
            try:
                _default_dynamic_layer.load(weights_path)
                logger.info(f"DynamicWeightLayer restored from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load dynamic weights from {weights_path}: {e}")
    return _default_dynamic_layer
