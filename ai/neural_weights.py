"""
Phase 8+ — Real Neural Weights (108 × 7 matrix, bias=0.6)
==========================================================
Provides a numpy-backed weight matrix for the RoutingEngine.

Architecture
------------
  NeuralWeightLayer
    ├── weights  : np.ndarray  shape=(108, 7)
    ├── bias     : float       default=0.6
    ├── forward(x)           → weighted output vector
    ├── update(delta)        → gradient-free weight update (learning step)
    ├── train_step(input_vector, target) → compute error & call update()
    ├── save(path)           → persist weights to .npy file
    └── load(path)           → restore from .npy file (accepts any (N,7) shape)

  extract_routing_weights(layer)
    Returns the 4 scalars (W_SEMANTIC, W_SCORE, W_MEMORY, W_TOPOLOGY)
    derived from the first row of the weight matrix, normalised to sum=1.

Growth strategy
---------------
  Columns are ALWAYS 7 — fixed to the feature vector size.
  Rows are now 108 (expanded from Phase-8 base of 9).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Default bias ─────────────────────────────────────────────────────────────
DEFAULT_BIAS: float = 0.6

# ── Initial weight matrix (108 rows × 7 columns) ─────────────────────────────
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
    [0.9,  0.18,  0.21,  0.1,  0.9, 0.4, 0.31],
    [0.5, 0.8, 0.11,  0.1,  0.14, 0.5, 0.16],
    [0.13, 0.2, 0.18,  0.1,  0.9, 0.4, 0.9 ],
    [0.51, 0.37, 0.3, 0.1,  0.11,  0.5, 0.1 ],
    [0.26, 0.65, 0.27, 0.1,  0.15,  0.5, 0.12 ],
    [0.26, 0.21,  0.22,  0.1,  0.9,  0.4, 0.3 ],
    [0.5, 0.14, 0.5,  0.1,  0.3, 0.2,  0.1],
    [0.1,  0.3,  0.22, 0.5, 0.8,  0.74,  0.17],
    [0.12,  0.3,  0.4,  0.1,  0.4,  0.19, 0.1 ],
    [0.31,  0.5,  0.6,  0.19,  0.3, 0.7, 0.2],
    [0.21, 0.4, 0.1,  0.2,  0.11, 0.21, 0.2],
    [0.11, 0.11, 0.16,  0.1,  0.21, 0.3, 0.15 ],
    [0.11, 0.38, 0.19, 0.2,  0.1,  0.12, 0.4 ],
    [0.17, 0.6, 0.9, 0.3,  0.4,  0.13, 0.6 ],
    [0.1, 0.25,  0.42,  0.5,  0.27,  0.33, 0.4 ],
    [0.12, 0.82, 0.36,  0.3,  0.9, 0.1,  0.26],
    [0.2,  0.21,  0.3, 0.5, 0.6,  0.1,  0.13],
    [0.2,  0.22,  0.22,  0.7,  0.8,  0.3, 0.1 ],
    [0.2,  0.28,  0.3,  0.5,  0.6, 0.1, 0.23],
    [0.3, 0.2, 0.3,  0.1,  0.12, 0.13, 0.32],
    [0.25, 0.26, 0.3,  0.1,  0.2, 0.4, 0.13 ],
    [0.13, 0.15, 0.3, 0.1,  0.6,  0.1, 0.1 ],
    [0.11, 0.21, 0.3, 0.1,  0.4,  0.1, 0.18 ],
    [0.18, 0.9,  0.17,  0.2,  0.1,  0.12, 0.4 ],
    [0.13, 0.3, 0.4,  0.1,  0.4, 0.16,  0.1],
    [0.7,  0.3,  0.4, 0.1, 0.25,  0.11,  0.1],
    [0.2,  0.11,  0.14,  0.2,  0.1,  0.9, 0.21 ],
    [0.5,  0.1,  0.1,  0.2,  0.6, 0.6, 0.1],
    [0.36, 0.31, 0.2,  0.3,  0.11, 0.31, 0.17],
    [0.31, 0.9, 0.11,  0.6,  0.4, 0.1, 0.1 ],
    [0.42, 0.1, 0.2, 0.6,  0.4,  0.1, 0.1 ],
    [0.29, 0.1, 0.2, 0.8,  0.4,  0.1, 0.12 ],
    [0.2, 0.1,  0.2,  0.9,  0.5,  0.12, 0.1 ],
    [0.31, 0.1, 0.2,  0.2,  0.11, 0.5,  0.8],
    [0.21,  0.1,  0.2, 0.2, 0.11,  0.5,  0.9],
    [0.7,  0.1,  0.2,  0.6,  0.4,  0.1, 0.1 ],
    [0.11,  0.1,  0.2,  0.6,  0.4, 0.1, 0.1],
    [0.18, 0.23, 0.6,  0.1,  0.4, 0.24, 0.1],
    [0.44, 0.6, 0.15,  0.1,  0.17, 0.5, 0.1 ],
    [0.1, 0.75, 0.12, 0.2,  0.1,  0.19, 0.4 ],
    [0.16, 0.32, 0.12, 0.3,  0.14,  0.6, 0.13 ],
    [0.9, 0.4,  0.5,  0.2,  0.6,  0.7, 0.7 ],
    [0.6, 0.5, 0.5,  0.2,  0.2, 0.3,  0.4],
    [0.8,  0.7,  0.6, 0.2, 0.12,  0.9,  0.11],
    [0.9,  0.26,  0.21,  0.1,  0.4,  0.3, 0.12 ],
    [0.8,  0.4,  0.5,  0.1,  0.8, 0.4, 0.4],
    [0.3, 0.4, 0.11,  0.1,  0.2, 0.7, 0.14],
    [0.1, 0.3, 0.1,  0.2,  0.4, 0.11, 0.1 ],
    [0.3, 0.6, 0.4, 0.1,  0.7,  0.6, 0.1 ],
    [0.1, 0.3, 0.1, 0.2,  0.4,  0.11, 0.1 ],
    [0.89, 0.78,  0.13,  0.2,  0.1,  0.19, 0.4 ],
    [0.1, 0.3, 0.1,  0.2,  0.4, 0.11,  0.1],
    [0.2,  0.4,  0.1, 0.2, 0.1,  0.11,  0.1],
    [0.31,  0.56,  0.11,  0.1,  0.1,  0.2, 0.23 ],
    [0.2,  0.4,  0.1,  0.2,  0.4, 0.11, 0.1],
    [0.22, 0.41, 0.18,  0.2,  0.1, 0.17, 0.4],
    [0.5, 0.14, 0.12,  0.2,  0.1, 0.13, 0.4 ],
    [0.33, 0.16, 0.15, 0.1,  0.8,  0.1, 0.12 ],
    [0.12, 0.9, 0.7, 0.3,  0.11,  0.5, 0.11 ],
    [0.1, 0.3,  0.1,  0.1,  0.7,  0.4, 0.6 ],
    [0.1, 0.17, 0.1,  0.2,  0.6, 0.1,  0.12],
    [0.6,  0.12,  0.19, 0.1, 0.16,  0.41,  0.21],
    [0.14,  0.5,  0.16,  0.3,  0.6,  0.15, 0.12 ],
    [0.4,  0.15,  0.8,  0.2, 0.1, 0.11, 0.4],
    [0.9, 0.13, 0.8,  0.2,  0.1, 0.4, 0.4],
    [0.5, 0.3, 0.6,  0.2,  0.8, 0.16, 0.16 ],
    [0.13, 0.15, 0.18, 0.3,  0.5,  0.4, 0.1 ],
    [0.6, 0.2, 0.4, 0.2,  0.7,  0.8, 0.5 ],
    [0.5, 0.8,  0.2,  0.6,  0.3,  0.4, 0.5 ],
    [0.4, 0.6, 0.4,  0.2,  0.1, 0.7,  0.14],
    [0.3,  0.4,  0.11, 0.1, 0.8,  0.5,  0.6],
    [0.8,  0.8,  0.7,  0.1,  0.12,  0.12, 0.11 ],
    [0.6,  0.8,  0.7,  0.1,  0.1, 0.15, 0.14],
    [0.23, 0.21, 0.6,  0.1,  0.2, 0.11, 0.9],
    [0.6, 0.12, 0.7,  0.1,  0.9, 0.14, 0.9 ],
    [0.4, 0.6, 0.5, 0.2,  0.4,  0.11, 0.4 ],
    [0.4, 0.16, 0.5, 0.2,  0.7,  0.18, 0.14 ],
    [0.1, 0.3,  0.6,  0.4,  0.4,  0.11, 0.5 ],
    [0.2, 0.7, 0.7,  0.3,  0.5, 0.4,  0.1],
    [0.11,  0.14,  0.11, 0.2, 0.3,  0.11,  0.1],
    [0.5,  0.5,  0.6,  0.2,  0.16,  0.4, 0.8 ],
    [0.6,  0.9,  0.5,  0.2,  0.11, 0.11, 0.11],
    [0.3, 0.8, 0.2,  0.2,  0.4, 0.11, 0.5],
    [0.9, 0.5, 0.1,  0.2,  0.6, 0.15, 0.8 ],
    [0.9, 0.7, 0.3, 0.1,  0.4,  0.8, 0.11 ],
    [0.5, 0.18, 0.13, 0.2,  0.5,  0.4, 0.2 ],
    [0.7, 0.0,  0.8,  0.1,  0.15,  0.9, 0.6 ],
    [0.1, 0.17, 0.2,  0.1,  0.11, 0.1,  0.9],
    [0.2,  0.41,  0.2, 0.6, 0.3,  0.24,  0.19],
    [0.9,  0.7,  0.11,  0.1,  0.1,  0.8, 0.19 ],
    [0.3,  0.11,  0.4,  0.2,  0.6, 0.7, 0.7],
    [0.12, 0.4, 0.1,  0.1,  0.7, 0.2, 0.7],
    [0.4, 0.1, 0.6,  0.1,  0.4, 0.9, 0.3 ],
    [0.7, 0.25, 0.11, 0.2,  0.11,  0.26, 0.0 ],
    [0.3, 0.2, 0.8, 0.1,  0.2,  0.3, 0.7 ],
    [0.4, 0.18,  0.3,  0.1,  0.7,  0.4, 0.11 ],
    [0.0, 0.7, 0.5,  0.2,  0.3, 0.13,  0.4],
    [0.17,  0.5,  0.17, 0.1, 0.4,  0.5,  0.6],
    [0.0,  0.11,  0.0,  0.1,  0.7,  0.13, 0.0 ],
]


class NeuralWeightLayer:
    """
    Neural weight layer backed by a (108 × 7) numpy matrix with scalar bias.

    Design contract
    ---------------
    • COLUMNS are FIXED at 7 — they represent the 7 input features.
    • ROWS are 108 (expanded matrix imported externally).
    • bias is a scalar applied after the matrix multiply (default 0.6).

    Parameters
    ----------
    initial_weights : array-like, optional
        Seed weights with shape (N, 7).  Defaults to the 108-row matrix.
    bias : float
        Scalar bias added to every activation.  Default 0.6.
    learning_rate : float
        Step size for `update()`.  Default 0.01.
    name : str
        Human-readable label stored in saved artefacts.
    """

    COLS = 7

    def __init__(
        self,
        initial_weights: Optional[Union[np.ndarray, List[List[float]]]] = None,
        bias: float = DEFAULT_BIAS,
        learning_rate: float = 0.01,
        name: str = "routing_weight_layer",
    ):
        if initial_weights is not None:
            w = np.array(initial_weights, dtype=np.float64)
            if w.ndim != 2 or w.shape[1] != self.COLS:
                raise ValueError(
                    f"NeuralWeightLayer expects shape (N, {self.COLS}), got {w.shape}"
                )
            self.weights: np.ndarray = w.copy()
        else:
            self.weights = np.array(_INITIAL_WEIGHTS, dtype=np.float64)

        self.bias: float = float(bias)
        self.learning_rate: float = learning_rate
        self.name: str = name
        self._train_steps: int = 0
        self._last_loss: Optional[float] = None

        logger.info(
            f"NeuralWeightLayer '{self.name}' initialised — "
            f"shape={self.weights.shape}  bias={self.bias}  lr={self.learning_rate}"
        )

    # ── Shape property (compatibility with code that reads .SHAPE) ────────

    @property
    def SHAPE(self):
        return self.weights.shape

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Pass an input vector through the weight matrix.

        Parameters
        ----------
        x : array-like of length 7

        Returns
        -------
        np.ndarray  shape=(rows,)
        """
        x_arr = np.array(x, dtype=np.float64)
        if x_arr.shape != (self.COLS,):
            raise ValueError(
                f"forward() expects input of length {self.COLS}, got {x_arr.shape}"
            )
        output = self.weights @ x_arr + self.bias   # (N, 7) × (7,) + scalar → (N,)
        activated = self._relu(output)
        return activated

    # ── Weight update (gradient-free, simple delta rule) ─────────────────

    def update(self, delta: Union[float, np.ndarray]) -> None:
        """
        Apply a simple additive update to every weight.

        Parameters
        ----------
        delta : float or ndarray shape=(N, 7)
        """
        d = np.array(delta, dtype=np.float64)
        if d.ndim == 0:
            self.weights += float(d) * self.learning_rate
        elif d.shape == self.weights.shape:
            self.weights += d * self.learning_rate
        else:
            raise ValueError(
                f"update() delta shape {d.shape} incompatible with weights {self.weights.shape}"
            )
        self.weights = np.clip(self.weights, -5.0, 5.0)
        logger.debug(f"NeuralWeightLayer '{self.name}' weights updated")

    # ── Training step ─────────────────────────────────────────────────────

    def train_step(
        self,
        input_vector: Union[List[float], np.ndarray],
        target: float,
    ) -> float:
        """
        Perform a single supervised training step (MSE loss).

        Parameters
        ----------
        input_vector : array-like, length 7
        target : float

        Returns
        -------
        float  MSE loss before the update.
        """
        x = np.array(input_vector, dtype=np.float64)
        output = self.forward(x)                                # shape (N,)
        n_rows = self.weights.shape[0]
        target_vec = np.full(n_rows, target, dtype=np.float64)

        error = output - target_vec                             # (N,)
        loss = float(np.mean(error ** 2))

        grad = np.outer(error, x) / n_rows                     # (N, 7)
        self.update(-grad)

        self._train_steps += 1
        self._last_loss = loss
        logger.debug(
            f"train_step #{self._train_steps}  loss={loss:.6f}  target={target:.4f}"
        )
        return loss

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> str:
        """Save the weight matrix to a .npy file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), self.weights)
        logger.info(f"NeuralWeightLayer '{self.name}' saved → {p.resolve()}")
        return str(p.resolve())

    def save_bias(self, path: str) -> str:
        """Save the bias scalar to a .npy file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), np.array(self.bias, dtype=np.float64))
        logger.info(f"NeuralWeightLayer '{self.name}' bias saved → {p.resolve()}")
        return str(p.resolve())

    def load(self, path: str) -> None:
        """
        Load a weight matrix from a .npy file.

        Accepts any shape (N, 7) — not restricted to the original (9, 7).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"NeuralWeightLayer.load: file not found: {p}")
        loaded = np.load(str(p))
        if loaded.ndim != 2 or loaded.shape[1] != self.COLS:
            raise ValueError(
                f"Loaded weights shape {loaded.shape} — must be (N, {self.COLS})"
            )
        self.weights = loaded.astype(np.float64)
        logger.info(
            f"NeuralWeightLayer '{self.name}' loaded ← {p.resolve()}  "
            f"shape={self.weights.shape}"
        )

    def load_bias(self, path: str) -> None:
        """Load bias scalar from a .npy file."""
        p = Path(path)
        if not p.exists():
            return
        self.bias = float(np.load(str(p)))
        logger.info(f"NeuralWeightLayer '{self.name}' bias loaded ← {p.resolve()}  bias={self.bias}")

    # ── Introspection ─────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a serialisable summary of the layer state."""
        return {
            "name": self.name,
            "shape": list(self.weights.shape),
            "bias": self.bias,
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
            f"shape={self.weights.shape} bias={self.bias} steps={self._train_steps}>"
        )


# ── Routing weight extractor ──────────────────────────────────────────────────

def extract_routing_weights(layer: NeuralWeightLayer) -> dict:
    """
    Derive the 4 RoutingEngine weight scalars from a NeuralWeightLayer.

    Uses the first row of the weight matrix (columns 0-3) and normalises
    them so they sum to 1.0.
    """
    row0 = layer.weights[0, :4]
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
    Otherwise the 108-row initial matrix is used.
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
