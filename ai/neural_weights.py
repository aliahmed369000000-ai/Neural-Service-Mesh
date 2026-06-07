"""
Phase 8 — Real Neural Weights
==============================
Provides a genuine numpy-backed weight matrix for the RoutingEngine.
The weight matrix (9 rows × 7 columns) replaces the hard-coded W_SEMANTIC / W_SCORE /
W_MEMORY / W_TOPOLOGY scalar constants with learnable parameters derived
from real execution data.

Architecture
------------
  NeuralWeightLayer
    ├── weights  : np.ndarray  shape=(9, 7)
    ├── forward(x)           → weighted output vector
    ├── update(delta)        → gradient-free weight update (learning step)
    ├── train_step(input_vector, target) → compute error & call update()
    ├── save(path)           → persist to .npy file
    └── load(path)           → restore from .npy file

  extract_routing_weights(layer)
    Returns the 4 scalars (W_SEMANTIC, W_SCORE, W_MEMORY, W_TOPOLOGY)
    derived from the first row of the weight matrix, normalised to sum=1.

Growth strategy (Phase 9+)
--------------------------
  Columns are ALWAYS 7 — fixed to the feature vector size.
  Row growth is handled by DynamicWeightLayer with GROW_ROWS=23,
  starting from 9 rows.  Trajectory: 9 → 32 → 55 → 78 → ...
  Max rows capped at 200 (see dynamic_weight_layer.py).

Usage (standalone)
------------------
    from ai.neural_weights import NeuralWeightLayer
    layer = NeuralWeightLayer()
    out   = layer.forward([0.5, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7])
    layer.train_step([0.5, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7], target=1.0)
    layer.save("models/classifiers/routing_weights.npy")

Integration with RoutingEngine
-------------------------------
    from ai.neural_weights import NeuralWeightLayer, extract_routing_weights
    layer   = NeuralWeightLayer()
    weights = extract_routing_weights(layer)
    # weights == {"W_SEMANTIC": ..., "W_SCORE": ..., "W_MEMORY": ..., "W_TOPOLOGY": ...}
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Initial weight matrix (9 rows × 7 columns) ──────────────────────────────
# Columns: FIXED at 7  (=INPUT_DIM from DeepRoutingNetwork / feature vector size)
# Rows:    START at 9  — growth happens on rows only (+23 per plateau event)
# Row 0 encodes the 4 routing-weight scalars (first 4 columns used).
# Rows 1-8 are additional feature-transformation layers.
_INITIAL_WEIGHTS: List[List[float]] = [
    [0.2,  0.3,  0.5,  0.7,  0.11, 0.15, 0.17],
    [0.13, 0.31, 0.3,  0.1,  0.12, 0.23, 0.16],
    [0.13, 0.14, 0.3,  0.1,  0.15, 0.11, 0.1 ],
    [0.11, 0.35, 0.19, 0.2,  0.1,  0.12, 0.4 ],
    [0.121, 0.31, 0.13, 0.2,  0.1,  0.32, 0.4 ],
    [0.11, 0.3,  0.4,  0.1,  0.4,  0.13, 0.1 ],
    [0.25, 0.24, 0.3,  0.1,  0.11, 0.2,  0.29],
    [0.2,  0.5,  0.15, 0.10, 0.1,  0.3,  0.26],
    [0.0,  0.4,  0.6,  0.3,  0.3,  0.11, 0.5 ],
]


class NeuralWeightLayer:
    """
    A single dense weight layer backed by a (9 × 7) numpy matrix.

    Design contract
    ---------------
    • COLUMNS are FIXED at 7 — they represent the 7 input features and
      must never change (breaking change to the whole pipeline).
    • ROWS start at 9 and grow in steps of +23 via DynamicWeightLayer.
      This class (Phase 8) keeps its shape fixed; growth lives in Phase 9.

    Parameters
    ----------
    initial_weights : array-like, optional
        Seed weights with shape (9, 7).  Defaults to the Phase-8 matrix.
    learning_rate : float
        Step size for `update()`.  Default 0.01.
    name : str
        Human-readable label stored in saved artefacts.
    """

    SHAPE = (9, 7)  # rows=9 (fixed base), cols=7 (NEVER change cols)

    def __init__(
        self,
        initial_weights: Optional[Union[np.ndarray, List[List[float]]]] = None,
        learning_rate: float = 0.01,
        name: str = "routing_weight_layer",
    ):
        if initial_weights is not None:
            w = np.array(initial_weights, dtype=np.float64)
            if w.shape != self.SHAPE:
                raise ValueError(
                    f"NeuralWeightLayer expects shape {self.SHAPE}, got {w.shape}"
                )
            self.weights: np.ndarray = w.copy()
        else:
            self.weights = np.array(_INITIAL_WEIGHTS, dtype=np.float64)

        self.learning_rate: float = learning_rate
        self.name: str = name
        self._train_steps: int = 0
        self._last_loss: Optional[float] = None

        logger.info(
            f"NeuralWeightLayer '{self.name}' initialised — "
            f"shape={self.weights.shape}  lr={self.learning_rate}"
        )

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Pass an input vector through the weight matrix.

        Parameters
        ----------
        x : array-like of length 7
            Input feature vector (7 elements matching the column count).

        Returns
        -------
        np.ndarray  shape=(rows,)
            One activation per row of the weight matrix (9 for base layer).
        """
        x_arr = np.array(x, dtype=np.float64)
        if x_arr.shape != (self.SHAPE[1],):
            raise ValueError(
                f"forward() expects input of length {self.SHAPE[1]}, got {x_arr.shape}"
            )
        output = self.weights @ x_arr          # (9, 7) × (7,) → (9,)
        activated = self._relu(output)
        return activated

    # ── Weight update (gradient-free, simple delta rule) ─────────────────

    def update(self, delta: Union[float, np.ndarray]) -> None:
        """
        Apply a simple additive update to every weight.

        Parameters
        ----------
        delta : float or ndarray shape=(9, 7)
            Scalar → broadcast to every element.
            Array  → element-wise addition (must match SHAPE).
        """
        d = np.array(delta, dtype=np.float64)
        if d.ndim == 0:                         # scalar
            self.weights += float(d) * self.learning_rate
        elif d.shape == self.SHAPE:
            self.weights += d * self.learning_rate
        else:
            raise ValueError(
                f"update() delta shape {d.shape} incompatible with weights {self.SHAPE}"
            )
        # Clip to keep weights in a safe range
        self.weights = np.clip(self.weights, -5.0, 5.0)
        logger.debug(f"NeuralWeightLayer '{self.name}' weights updated")

    # ── Training step ─────────────────────────────────────────────────────

    def train_step(
        self,
        input_vector: Union[List[float], np.ndarray],
        target: float,
    ) -> float:
        """
        Perform a single supervised training step.

        Uses a simple mean-squared-error loss:
            loss = mean((output - target)²)
        and updates weights proportionally to the error.

        Parameters
        ----------
        input_vector : array-like, length 7
        target : float
            Desired scalar output (e.g. a normalised route score in [0, 1]).

        Returns
        -------
        float
            The MSE loss before the update.
        """
        x = np.array(input_vector, dtype=np.float64)
        output = self.forward(x)                            # shape (9,)
        target_vec = np.full(self.SHAPE[0], target, dtype=np.float64)

        error = output - target_vec                         # (9,)
        loss = float(np.mean(error ** 2))

        # Gradient of MSE w.r.t. weights: outer product (error, x)
        grad = np.outer(error, x) / self.SHAPE[0]          # (9, 7)
        self.update(-grad)                                  # gradient descent

        self._train_steps += 1
        self._last_loss = loss
        logger.debug(
            f"train_step #{self._train_steps}  loss={loss:.6f}  "
            f"target={target:.4f}"
        )
        return loss

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> str:
        """
        Save the weight matrix to a .npy file.

        Parameters
        ----------
        path : str
            Destination file path (should end in .npy).

        Returns
        -------
        str
            The resolved absolute path that was written.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), self.weights)
        logger.info(f"NeuralWeightLayer '{self.name}' saved → {p.resolve()}")
        return str(p.resolve())

    def load(self, path: str) -> None:
        """
        Load a weight matrix from a .npy file.

        Parameters
        ----------
        path : str
            Source .npy file.  Must contain an array of shape (9, 7).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"NeuralWeightLayer.load: file not found: {p}")
        loaded = np.load(str(p))
        if loaded.shape != self.SHAPE:
            raise ValueError(
                f"Loaded weights shape {loaded.shape} ≠ expected {self.SHAPE}"
            )
        self.weights = loaded.astype(np.float64)
        logger.info(f"NeuralWeightLayer '{self.name}' loaded ← {p.resolve()}")

    # ── Introspection ─────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a serialisable summary of the layer state."""
        return {
            "name": self.name,
            "shape": list(self.SHAPE),
            "learning_rate": self.learning_rate,
            "train_steps": self._train_steps,
            "last_loss": round(self._last_loss, 8) if self._last_loss is not None else None,
            "weight_stats": {
                "min":  round(float(self.weights.min()), 6),
                "max":  round(float(self.weights.max()), 6),
                "mean": round(float(self.weights.mean()), 6),
                "std":  round(float(self.weights.std()), 6),
            },
        }

    def get_weights_list(self) -> List[List[float]]:
        """Return the weight matrix as a plain Python list of lists."""
        return self.weights.tolist()

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def __repr__(self) -> str:
        return (
            f"<NeuralWeightLayer name='{self.name}' "
            f"shape={self.SHAPE} steps={self._train_steps}>"
        )


# ── Routing weight extractor ──────────────────────────────────────────────────

def extract_routing_weights(layer: NeuralWeightLayer) -> dict:
    """
    Derive the 4 RoutingEngine weight scalars from a NeuralWeightLayer.

    Uses the first row of the weight matrix (columns 0-3) and normalises
    them so they sum to 1.0, preserving relative magnitudes.

    Returns
    -------
    dict with keys: W_SEMANTIC, W_SCORE, W_MEMORY, W_TOPOLOGY
    """
    row0 = layer.weights[0, :4]                   # first row, first 4 cols
    total = float(row0.sum())
    if total <= 0.0:
        logger.warning("extract_routing_weights: row0 sum ≤ 0, using defaults")
        return {
            "W_SEMANTIC": 0.30,
            "W_SCORE":    0.35,
            "W_MEMORY":   0.25,
            "W_TOPOLOGY": 0.10,
        }
    normed = (row0 / total).tolist()
    return {
        "W_SEMANTIC": round(normed[0], 6),
        "W_SCORE":    round(normed[1], 6),
        "W_MEMORY":   round(normed[2], 6),
        "W_TOPOLOGY": round(normed[3], 6),
    }


# ── Module-level default layer (singleton pattern) ───────────────────────────

_default_layer: Optional[NeuralWeightLayer] = None


def get_default_layer(weights_path: str = "models/classifiers/routing_weights.npy") -> NeuralWeightLayer:
    """
    Return (and cache) the module-level default NeuralWeightLayer.

    If a saved .npy file exists at `weights_path`, it is loaded automatically.
    Otherwise the Phase-8 initial matrix is used.
    """
    global _default_layer
    if _default_layer is None:
        _default_layer = NeuralWeightLayer(name="default_routing_layer")
        if os.path.exists(weights_path):
            try:
                _default_layer.load(weights_path)
                logger.info(f"Default NeuralWeightLayer restored from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load weights from {weights_path}: {e}")
    return _default_layer
