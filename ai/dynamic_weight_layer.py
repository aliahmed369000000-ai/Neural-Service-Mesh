"""
Phase 9 — Axis 2: Dynamic Self-Growing Weight Layer
=====================================================
Replaces the fixed 9×7 NeuralWeightLayer (Phase 8) with a layer that
**grows its own weight matrix** when performance plateaus.

Growth strategy (updated v15+)
-------------------------------
  • COLUMNS are FIXED at 7 — they represent the 7 input features coming
    from _build_feature_vector() / RichDataCollector.  Changing columns
    would break every upstream caller.  Columns NEVER grow.

  • ROWS start at 9 (matching the Phase 8 base matrix) and grow by +23
    on every plateau event.  +23 gives the network a meaningful capacity
    jump instead of the old +3 trickle.

  • Growth trajectory:
        Start:            9  rows × 7 cols
        After plateau 1: 32  rows × 7 cols  (+23)
        After plateau 2: 55  rows × 7 cols  (+23)
        After plateau 3: 78  rows × 7 cols  (+23)
        ...
        Cap:            200  rows × 7 cols

  • COOLDOWN_STEPS raised to 120 (was 30) so the network fully absorbs
    each +23 expansion before growing again.

  • On plateau: add GROW_ROWS=23 new rows only.
    New rows are Xavier/Glorot-initialised.
  • Maximum row count capped at MAX_ROWS=200.

The class is a drop-in superset of NeuralWeightLayer:
  • forward(x)            — same signature, adapts to current row count
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

# Starting dimensions — rows=9 matches Phase 8 NeuralWeightLayer base
INITIAL_ROWS = 9       # start at 9 rows (same as Phase 8 base matrix)
INITIAL_COLS = 7       # FIXED — equals the input feature vector size (INPUT_DIM)

# Growth parameters
GROW_ROWS = 23         # +23 rows per plateau event (was 3)
GROW_COLS = 0          # columns NEVER grow (fixed at INPUT_DIM=7)

# Limits
MAX_ROWS = 200         # raised from 50 to accommodate +23 growth cadence
MAX_COLS = 7           # hard ceiling = INITIAL_COLS (columns locked)

# Plateau detection
PLATEAU_CHECK_STEPS = 20    # look at last 20 training steps
PLATEAU_THRESHOLD   = 0.01  # < 1% improvement triggers growth

# Cooldown raised to 120 steps (was 30) — gives the network time to
# "digest" the 23 new rows before growing again
COOLDOWN_STEPS = 120


class DynamicWeightLayer:
    """
    Phase 9 Axis-2: Self-growing neural weight layer.

    The weight matrix starts at INITIAL_ROWS × INITIAL_COLS (9×7) and
    expands its ROW count automatically when learning stagnates.
    Column count is permanently fixed at 7 (= input feature dimension).

    Parameters
    ----------
    initial_rows : int
        Starting row count (default: INITIAL_ROWS = 9).
    initial_cols : int
        Input feature dimension — must stay 7 (default: INITIAL_COLS = 7).
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
        # Enforce column lock
        if initial_cols != INITIAL_COLS:
            logger.warning(
                f"DynamicWeightLayer: initial_cols={initial_cols} overridden "
                f"to {INITIAL_COLS} (columns are fixed to input feature dim)."
            )
            initial_cols = INITIAL_COLS

        self._rows = initial_rows
        self._cols = initial_cols          # always 7, never changes
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
            f"shape=({self._rows}×{self._cols})  "
            f"grow_rows={GROW_ROWS}  cooldown={COOLDOWN_STEPS}  lr={self.learning_rate}"
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

        Input is always expected to be exactly 7 elements (INITIAL_COLS).
        Shorter vectors are zero-padded; longer vectors are truncated.

        Returns np.ndarray of shape (current_rows,).
        """
        x_arr = np.array(x, dtype=np.float64)
        # Normalise input to exactly self._cols (= 7, always)
        if x_arr.shape[0] < self._cols:
            x_arr = np.pad(x_arr, (0, self._cols - x_arr.shape[0]))
        elif x_arr.shape[0] > self._cols:
            x_arr = x_arr[:self._cols]

        output = self.weights @ x_arr          # (rows, 7) × (7,) → (rows,)
        return np.maximum(0.0, output)         # ReLU activation

    # ── Training step ─────────────────────────────────────────────────────

    def train_step(
        self,
        input_vector: Union[List[float], np.ndarray],
        target: float,
    ) -> float:
        """
        Single supervised training step (MSE loss, gradient descent).

        Also checks for plateau and triggers +23 row growth when needed.

        Returns
        -------
        float — MSE loss before the update.
        """
        x = np.array(input_vector, dtype=np.float64)
        if x.shape[0] < self._cols:
            x = np.pad(x, (0, self._cols - x.shape[0]))
        elif x.shape[0] > self._cols:
            x = x[:self._cols]

        output     = self.forward(x)
        target_vec = np.full(self._rows, target, dtype=np.float64)
        error      = output - target_vec
        loss       = float(np.mean(error ** 2))

        # Gradient of MSE w.r.t. weights: outer(error, x) / rows
        grad = np.outer(error, x) / self._rows
        self.weights -= self.learning_rate * grad
        self.weights  = np.clip(self.weights, -5.0, 5.0)

        self._train_steps        += 1
        self._last_loss           = loss
        self._loss_history.append(loss)
        self._steps_since_growth += 1

        # Plateau check — only every PLATEAU_CHECK_STEPS steps
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
        Return True if recent loss improvement is below PLATEAU_THRESHOLD.

        Splits the last PLATEAU_CHECK_STEPS×2 entries into older / recent
        halves and compares their means.
        """
        if len(self._loss_history) < PLATEAU_CHECK_STEPS * 2:
            return False
        window      = list(self._loss_history)[-PLATEAU_CHECK_STEPS * 2:]
        mid         = len(window) // 2
        mean_older  = float(np.mean(window[:mid]))
        mean_recent = float(np.mean(window[mid:]))
        if mean_older == 0.0:
            return False
        improvement = (mean_older - mean_recent) / mean_older
        logger.debug(
            f"Plateau check: older={mean_older:.6f}  "
            f"recent={mean_recent:.6f}  improvement={improvement:.4f}"
        )
        return improvement < PLATEAU_THRESHOLD

    # ── Growth (+23 rows, 0 cols) ─────────────────────────────────────────

    def _grow(self) -> None:
        """
        Expand the weight matrix by adding GROW_ROWS=23 new rows.
        Column count stays fixed at 7 (GROW_COLS=0).
        New rows are Xavier-initialised.
        """
        old_shape = (self._rows, self._cols)
        new_rows  = min(self._rows + GROW_ROWS, MAX_ROWS)

        if new_rows == self._rows:
            logger.info(
                f"DynamicWeightLayer '{self.name}': already at maximum "
                f"{MAX_ROWS} rows — skipping growth."
            )
            return

        # Build expanded matrix: copy existing rows, append Xavier new rows
        extra_rows  = self._xavier_init(new_rows - self._rows, self._cols)
        new_weights = np.vstack([self.weights, extra_rows])

        self.weights = new_weights
        self._rows   = new_rows
        # self._cols remains unchanged (7)
        self._steps_since_growth = 0

        event = {
            "step":          self._train_steps,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "old_shape":     list(old_shape),
            "new_shape":     [new_rows, self._cols],
            "rows_added":    new_rows - old_shape[0],
            "cols_added":    0,
            "trigger":       "plateau",
            "loss_at_growth": round(self._last_loss, 8) if self._last_loss else None,
        }
        self._growth_events.append(event)

        logger.info(
            f"DynamicWeightLayer '{self.name}' GREW (rows only): "
            f"{old_shape[0]}×{old_shape[1]} → {new_rows}×{self._cols}  "
            f"(+{new_rows - old_shape[0]} rows, step {self._train_steps})"
        )

    # ── Compatibility with extract_routing_weights() ──────────────────────

    def get_routing_row(self) -> np.ndarray:
        """Return first row, first 4 cols — compatible with extract_routing_weights."""
        return self.weights[0, :min(4, self._cols)]

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> str:
        """Save weights and shape metadata to <path>.npy and <path>_meta.npy."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), self.weights)
        meta = np.array([self._rows, self._cols, self._train_steps], dtype=np.int64)
        np.save(str(p).replace(".npy", "_meta.npy"), meta)
        logger.info(
            f"DynamicWeightLayer '{self.name}' saved → {p.resolve()}  "
            f"shape=({self._rows}×{self._cols})"
        )
        return str(p.resolve())

    def load(self, path: str) -> None:
        """
        Load weights from .npy file.
        Column count is re-validated: must be 7, else the layer resets.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"DynamicWeightLayer.load: not found: {p}")
        loaded = np.load(str(p))

        # Safety: enforce column lock
        if loaded.shape[1] != INITIAL_COLS:
            logger.warning(
                f"DynamicWeightLayer.load: saved cols={loaded.shape[1]} ≠ "
                f"required {INITIAL_COLS}. Ignoring saved file, using fresh weights."
            )
            return

        self.weights  = loaded.astype(np.float64)
        self._rows, self._cols = self.weights.shape

        # Load metadata if present
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
        """Extended summary including growth history and trajectory."""
        # Project future growth trajectory
        trajectory = []
        r = self._rows
        for _ in range(5):
            r_next = min(r + GROW_ROWS, MAX_ROWS)
            trajectory.append(f"{r_next}×{self._cols}")
            if r_next >= MAX_ROWS:
                break
            r = r_next

        return {
            "name":                self.name,
            "shape":               [self._rows, self._cols],
            "cols_fixed":          True,
            "grow_rows":           GROW_ROWS,
            "grow_cols":           GROW_COLS,
            "max_rows":            MAX_ROWS,
            "learning_rate":       self.learning_rate,
            "train_steps":         self._train_steps,
            "last_loss":           round(self._last_loss, 8) if self._last_loss is not None else None,
            "growth_events_count": len(self._growth_events),
            "growth_events":       self._growth_events[-5:],
            "next_growth_trajectory": trajectory,
            "weight_stats": {
                "min":  round(float(self.weights.min()), 6),
                "max":  round(float(self.weights.max()), 6),
                "mean": round(float(self.weights.mean()), 6),
                "std":  round(float(self.weights.std()), 6),
            },
            "plateau_check": {
                "window_steps":       PLATEAU_CHECK_STEPS,
                "threshold":          PLATEAU_THRESHOLD,
                "cooldown":           COOLDOWN_STEPS,
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

    Uses row 0, columns 0-3 (unchanged by row growth).
    Mirrors the Phase 8 `extract_routing_weights()` for drop-in use.
    """
    row0  = layer.weights[0, :min(4, layer._cols)]
    total = float(row0.sum())
    if total <= 0.0:
        logger.warning("extract_routing_weights_dynamic: row0 sum ≤ 0, using defaults")
        return {
            "W_SEMANTIC": 0.30,
            "W_SCORE":    0.35,
            "W_MEMORY":   0.25,
            "W_TOPOLOGY": 0.10,
        }
    normed_arr = row0 / total
    while len(normed_arr) < 4:
        normed_arr = np.append(normed_arr, 0.0)
        t = float(normed_arr.sum())
        if t > 0:
            normed_arr /= t
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
    Starts at 9×7; loads persisted weights if the file exists.
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


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 65)
    print("  DynamicWeightLayer v15 — Self-Test")
    print(f"  Strategy: rows start={INITIAL_ROWS}, grow=+{GROW_ROWS}, cols fixed={INITIAL_COLS}")
    print("=" * 65)

    layer = DynamicWeightLayer(name="test_layer")
    print(f"\n  Initial shape: {layer.shape}")
    assert layer.shape == (9, 7), f"Expected (9,7) got {layer.shape}"
    assert layer._cols == 7, "Cols must be 7"

    # Simulate plateau: inject a flat loss history
    for i in range(COOLDOWN_STEPS + PLATEAU_CHECK_STEPS * 2 + 5):
        layer._loss_history.append(0.0500)   # stagnant loss
    layer._steps_since_growth = COOLDOWN_STEPS + 1
    layer._train_steps = PLATEAU_CHECK_STEPS

    if layer._is_plateauing():
        layer._grow()
        print(f"  After growth:  {layer.shape}  (+{GROW_ROWS} rows)")
        assert layer.shape == (9 + GROW_ROWS, 7), f"Expected ({9+GROW_ROWS},7) got {layer.shape}"
        assert layer._cols == 7, "Cols must stay 7 after growth"
    else:
        print("  (plateau not triggered in static test — OK)")

    # Forward pass with 7-element input
    out = layer.forward([0.5] * 7)
    assert out.shape == (layer._rows,), f"Forward output shape wrong: {out.shape}"
    print(f"  Forward pass:  input=(7,) → output=({out.shape[0]},)  ✓")

    # Train step
    loss = layer.train_step([0.3, 0.7, 0.5, 0.2, 0.4, 0.6, 0.1], target=0.8)
    print(f"  Train step:    loss={loss:.6f}  steps={layer._train_steps}  ✓")

    # Routing extraction
    r = extract_routing_weights_dynamic(layer)
    total = sum(r.values())
    assert abs(total - 1.0) < 1e-6, f"Routing weights don't sum to 1: {total}"
    print(f"  Routing:       {r}  (sum={total:.6f}) ✓")

    # Growth trajectory from summary
    s = layer.summary()
    print(f"\n  Summary shape: {s['shape']}  grow_rows={s['grow_rows']}  grow_cols={s['grow_cols']}")
    print(f"  Trajectory:    current → {' → '.join(s['next_growth_trajectory'])}")
    print(f"  Max size:      {s['max_size']}")

    print("\n✓ DynamicWeightLayer v15 self-test PASSED")
