"""
Phase 9 — Axis 3: Single-Layer Routing Network (v17 — single layer)
=====================================================================
This module now implements a **single fully-connected layer** instead
of the previous 3-layer deep network. The network is intentionally kept
as ONE layer with a FIXED shape contract:

Column/input dimension contract
---------------------------------------
  INPUT_DIM=7 is FIXED:
    • DeepRoutingNetwork  Layer-1  shape=(108, 7) — cols FIXED at 7,
      rows start at 108 and grow only by +6 per trigger (max 200).

Architecture (v17 — single layer):

    Input (7 features — from RichDataCollector)
        ↓
    Layer 1: 108×7   — relu  (rows grow +6 up to max 200, cols FIXED=7)
        ↓
    Output: 108-dim activation vector
        → routing weights (W_SEMANTIC, W_SCORE, W_MEMORY, W_TOPOLOGY)
          are derived from rows 0-3 of the output, normalised to sum=1
          (same convention as the Phase 8 `extract_routing_weights`).

The single layer:
  - Stores a weight matrix of shape (out_dim × in_dim) = (108 × 7)
  - All biases are fixed at 0.6
  - Uses ReLU activation
  - Supports gradient-based training via backpropagation (single layer)

Growth rules (the only layer):
  - Columns (7) are FIXED and must never change
  - Rows grow by +6 per trigger
  - Max rows: 200
  - Bias for newly added rows is also 0.6

Backward compatibility
----------------------
`get_default_deep_network()` returns a singleton instance.
`extract_deep_routing_weights(net)` returns the same 4-key dict as the
Phase 8 `extract_routing_weights(layer)`.
The public API (`forward`, `train_step`, `train_batch`,
`predict_routing_weights`, `grow`, `load_custom_weights`, `save`, `load`,
`summary`, `architecture_str`, `.layers[0]`, `.weights`, `.SHAPE`) is kept
unchanged so existing callers (RoutingEngine, KnowledgeTrainer, main.py)
continue to work without modification.
"""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Architecture definition ───────────────────────────────────────────────────
# Single-layer architecture (v17):
#   Layer 1: 108×7  relu   — rows growable (+6, NO upper limit), cols=7 FIXED, bias=0.6
LAYER_CONFIGS: List[Tuple[int, int, str]] = [
    (108, 7, "relu"),    # Layer 1: 7 inputs → 108 nodes
]

INPUT_DIM     = 7
OUTPUT_DIM    = 4
LEARNING_RATE = 0.005
WEIGHTS_DIR   = "models/classifiers"

# Layer growth parameters
L1_INITIAL_ROWS = 108
L1_GROW_BY      = 6
L1_MAX_ROWS     = None   # NO upper limit — rows grow indefinitely
L1_COLS         = 7      # FIXED — must never change

# ── Plateau-based auto-growth (يتطور عند انخفاض القدرة على الفهم) ────────────
# When the training loss stops improving (the network's "ability to
# understand" the data plateaus), the single layer grows by +6 rows.
PLATEAU_WINDOW    = 50     # loss-history window size used to detect plateau
PLATEAU_THRESHOLD = 0.01   # min relative improvement to NOT be a plateau
PLATEAU_COOLDOWN  = 200    # min train_steps between growth events

# ── Initial Layer 1 weights (108×7) provided by user ─────────────────────────
_L1_INITIAL_WEIGHTS: List[List[float]] = [
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


def _xavier_init(rows: int, cols: int) -> np.ndarray:
    limit = math.sqrt(6.0 / (rows + cols))
    return np.random.uniform(-limit, limit, size=(rows, cols)).astype(np.float64)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_deriv(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max()
    exp_x = np.exp(shifted)
    total = exp_x.sum()
    if total == 0:
        return np.ones(len(x)) / len(x)
    return exp_x / total


class DenseLayer:
    """
    A single fully-connected layer with configurable activation.

    Stores:
      weights : (out_dim, in_dim)
      biases  : (out_dim,)   — initialised to 0.6
    """

    def __init__(self, out_dim: int, in_dim: int,
                 activation: str = "relu", name: str = ""):
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.activation = activation
        self.name = name or f"layer_{out_dim}x{in_dim}"

        self.weights: np.ndarray = _xavier_init(out_dim, in_dim)
        self.biases: np.ndarray = np.full(out_dim, 0.6, dtype=np.float64)

        # Cached values for backprop
        self._last_input: Optional[np.ndarray] = None
        self._last_pre_act: Optional[np.ndarray] = None
        self._last_output: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: W·x + b → activation."""
        self._last_input = x.copy()
        pre = self.weights @ x + self.biases
        self._last_pre_act = pre.copy()

        if self.activation == "relu":
            out = _relu(pre)
        elif self.activation == "softmax":
            out = _softmax(pre)
        else:
            out = pre  # linear

        self._last_output = out.copy()
        return out

    def backward(
        self,
        grad_output: np.ndarray,
        learning_rate: float,
    ) -> np.ndarray:
        """
        Backpropagate through this layer.

        Updates weights and biases in-place.
        Returns gradient to pass to the previous layer.
        """
        if self.activation == "relu":
            grad_pre = grad_output * _relu_deriv(self._last_pre_act)
        elif self.activation == "softmax":
            # For MSE + softmax: simplified gradient (treat as linear)
            grad_pre = grad_output
        else:
            grad_pre = grad_output

        # Weight gradient: outer product of pre-activation grad and input
        grad_w = np.outer(grad_pre, self._last_input)
        grad_b = grad_pre
        grad_x = self.weights.T @ grad_pre

        # Gradient descent update
        self.weights -= learning_rate * grad_w
        self.biases  -= learning_rate * grad_b

        # Clip for stability
        self.weights = np.clip(self.weights, -5.0, 5.0)

        return grad_x

    def summary(self) -> dict:
        return {
            "name":       self.name,
            "shape":      [self.out_dim, self.in_dim],
            "activation": self.activation,
            "weight_stats": {
                "min":  round(float(self.weights.min()), 6),
                "max":  round(float(self.weights.max()), 6),
                "mean": round(float(self.weights.mean()), 6),
                "std":  round(float(self.weights.std()), 6),
            },
        }

    def save(self, path_prefix: str) -> None:
        np.save(f"{path_prefix}_weights.npy", self.weights)
        np.save(f"{path_prefix}_biases.npy",  self.biases)

    def load(self, path_prefix: str) -> None:
        self.weights = np.load(f"{path_prefix}_weights.npy").astype(np.float64)
        self.biases  = np.load(f"{path_prefix}_biases.npy").astype(np.float64)
        self.out_dim = self.weights.shape[0]
        self.in_dim  = self.weights.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
#  Deep Routing Network
# ─────────────────────────────────────────────────────────────────────────────

class DeepRoutingNetwork:
    """
    Phase 9 Axis-3: Deep multi-layer routing weight predictor (v16).

    Architecture (3 layers):
        Input (7) → Layer1 108×7 relu → Layer2 9×108 relu → Layer3 4×9 softmax

    Layer 1 growth rules:
      - Columns (7) are FIXED and must never change
      - Rows grow by +6 per trigger (call grow())
      - Max rows: 200
      - When Layer 1 grows, Layer 2's in_dim expands in lockstep

    The output is always 4 values that sum to 1.0, directly usable as
    (W_SEMANTIC, W_SCORE, W_MEMORY, W_TOPOLOGY).

    Parameters
    ----------
    learning_rate : float  default 0.005
    name : str
    """

    def __init__(
        self,
        learning_rate: float = LEARNING_RATE,
        name: str = "deep_routing_network",
        plateau_window: int = PLATEAU_WINDOW,
        plateau_threshold: float = PLATEAU_THRESHOLD,
        plateau_cooldown: int = PLATEAU_COOLDOWN,
    ):
        self.name = name
        self.learning_rate = learning_rate
        self._train_steps = 0
        self._last_loss: Optional[float] = None
        self._loss_history: List[float] = []

        # ── إعدادات النمو عند الركود (انخفاض القدرة على الفهم) ───────────
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.plateau_cooldown = plateau_cooldown
        self._steps_since_growth = 0
        self._growth_events: List[dict] = []

        # Build the single layer: 108×7 relu, bias=0.6
        l1 = DenseLayer(L1_INITIAL_ROWS, L1_COLS, "relu",
                        name=f"L1_{L1_INITIAL_ROWS}x{L1_COLS}_relu")
        l1.weights = np.array(_L1_INITIAL_WEIGHTS, dtype=np.float64)
        l1.biases  = np.full(L1_INITIAL_ROWS, 0.6, dtype=np.float64)

        self.layers: List[DenseLayer] = [l1]

        total_params = self._count_params()
        logger.info(
            f"DeepRoutingNetwork '{self.name}' initialised — "
            f"1 layer: 7→108(relu) | "
            f"total parameters: {total_params}"
        )

    # ── Growth (single layer rows) ────────────────────────────────────────

    def grow(self) -> bool:
        """
        Grow the single layer by +6 rows. There is NO upper limit on rows —
        the layer can grow indefinitely.

        Columns (7) are never touched. Bias for new rows is 0.6.

        Returns True (growth always succeeds; no max-rows ceiling).
        """
        l1 = self.layers[0]
        current_rows = l1.out_dim
        new_rows = current_rows + L1_GROW_BY
        added    = L1_GROW_BY

        # Extend layer weights (new rows xavier-init) and biases (0.6)
        new_w = _xavier_init(added, L1_COLS)
        l1.weights = np.vstack([l1.weights, new_w])
        l1.biases  = np.concatenate([l1.biases, np.full(added, 0.6)])
        l1.out_dim = new_rows
        l1.name    = f"L1_{new_rows}x{L1_COLS}_relu"

        logger.info(
            f"DeepRoutingNetwork.grow(): layer {current_rows}→{new_rows} rows | "
            f"total params now: {self._count_params()}"
        )
        return True

    # ── Custom weight loader ──────────────────────────────────────────────

    def load_custom_weights(
        self,
        matrix: np.ndarray,
        layer_index: int = 0,
    ) -> None:
        """
        Load a custom weight matrix into the single layer.

        Enforces cols=7 (FIXED). The layer's row count is set to match
        the matrix — there is no upper limit on rows.

        Parameters
        ----------
        matrix : np.ndarray  shape (out_dim, in_dim)
        layer_index : int  must be 0 (single-layer network)

        Raises
        ------
        ValueError  if shape is incompatible with fixed constraints.
        """
        if layer_index != 0:
            raise ValueError(
                f"This network has a single layer; layer_index must be 0, "
                f"got {layer_index}."
            )

        m = np.array(matrix, dtype=np.float64)
        layer = self.layers[0]

        # Enforce fixed column count
        if m.ndim != 2 or m.shape[1] != L1_COLS:
            raise ValueError(
                f"Layer columns are FIXED at {L1_COLS}. "
                f"Got shape {m.shape}."
            )
        new_rows = m.shape[0]
        layer.weights = m
        layer.biases  = np.full(new_rows, 0.6, dtype=np.float64)
        layer.out_dim = new_rows
        layer.name    = f"L1_{new_rows}x{L1_COLS}_relu"

        logger.info(
            f"load_custom_weights: layer 0 ('{layer.name}') "
            f"loaded shape {m.shape} | "
            f"total params now: {self._count_params()}"
        )

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Run the forward pass through the single layer.

        Parameters
        ----------
        x : array-like of length INPUT_DIM (7)
            Normalised feature vector from RichDataCollector.

        Returns
        -------
        np.ndarray shape (N,) — ReLU activations, N = current row count
        (108 by default, grows by +6 up to 200).
        """
        h = np.array(x, dtype=np.float64)
        # Pad or truncate to INPUT_DIM (7 — FIXED)
        if h.shape[0] < INPUT_DIM:
            h = np.pad(h, (0, INPUT_DIM - h.shape[0]))
        elif h.shape[0] > INPUT_DIM:
            h = h[:INPUT_DIM]

        return self.layers[0].forward(h)

    def predict_routing_weights(self, x: Union[List[float], np.ndarray]) -> dict:
        """
        Forward pass returning the 4-key routing weights dict.

        The 4 routing scalars are derived from rows 0-3 of the layer's
        output, normalised so they sum to 1 (same convention as the
        Phase 8 `extract_routing_weights`).

        Returns
        -------
        dict with keys: W_SEMANTIC, W_SCORE, W_MEMORY, W_TOPOLOGY
        """
        out = self.forward(x)
        head = out[:OUTPUT_DIM]
        total = float(head.sum())
        if total <= 0.0:
            return {
                "W_SEMANTIC": 0.30,
                "W_SCORE":    0.35,
                "W_MEMORY":   0.25,
                "W_TOPOLOGY": 0.10,
            }
        normed = (head / total).tolist()
        return {
            "W_SEMANTIC": round(normed[0], 6),
            "W_SCORE":    round(normed[1], 6),
            "W_MEMORY":   round(normed[2], 6),
            "W_TOPOLOGY": round(normed[3], 6),
        }

    # ── Training step (single-layer backpropagation) ──────────────────────

    def train_step(
        self,
        input_vector: Union[List[float], np.ndarray],
        target: float,
    ) -> float:
        """
        Supervised training step using single-layer backpropagation.

        Loss: MSE between the layer's output vector (N,) and `target`
        broadcast to all N rows.

        Parameters
        ----------
        input_vector : array-like length 7
        target : float  in [0, 1]
            Desired composite routing quality score.

        Returns
        -------
        float — MSE loss before the update.
        """
        x = np.array(input_vector, dtype=np.float64)
        if x.shape[0] < INPUT_DIM:
            x = np.pad(x, (0, INPUT_DIM - x.shape[0]))
        elif x.shape[0] > INPUT_DIM:
            x = x[:INPUT_DIM]

        # Forward
        output = self.forward(x)

        # Target broadcast to every row
        n_rows = output.shape[0]
        target_vec = np.full(n_rows, float(target), dtype=np.float64)

        error = output - target_vec
        loss = float(np.mean(error ** 2))

        # Backward through the single layer
        grad = 2.0 * error / n_rows
        self.layers[0].backward(grad, self.learning_rate)

        self._train_steps += 1
        self._steps_since_growth += 1
        self._last_loss = loss
        self._loss_history.append(loss)
        if len(self._loss_history) > 1000:
            self._loss_history = self._loss_history[-1000:]

        # ── نمو عند ركود التعلّم (انخفاض القدرة على الفهم) ───────────────
        # يحدث *بين* خطوات التدريب فقط، فلا يكسر صحة التدرجات لهذه الخطوة.
        grew = self.evolve_if_plateau()

        logger.debug(
            f"DeepRoutingNetwork train_step #{self._train_steps}  "
            f"loss={loss:.6f}  grew={grew}"
        )
        return loss

    # ── التطوّر عند الركود (يتطور عند انخفاض القدرة على الفهم) ─────────────

    def _is_plateauing(self) -> bool:
        """
        يفحص إن كانت الخسارة قد ركدت: يقارن متوسط الخسارة في النافذة
        الأقدم بمتوسط النافذة الأحدث (كل نافذة حجمها `plateau_window`).
        إذا كان التحسّن أقل من `plateau_threshold`، فالشبكة "لم تعد تفهم"
        البيانات بشكل أفضل وتحتاج إلى نمو.
        """
        hist = self._loss_history
        w = self.plateau_window
        if len(hist) < w * 2:
            return False
        window = hist[-w * 2:]
        mean_older = float(np.mean(window[:w]))
        mean_recent = float(np.mean(window[w:]))
        if mean_older == 0.0:
            return False
        improvement = (mean_older - mean_recent) / mean_older
        return improvement < self.plateau_threshold

    def evolve_if_plateau(self) -> bool:
        """
        إذا ركدت الخسارة (انخفضت القدرة على الفهم) بعد `plateau_cooldown`
        خطوة من آخر نمو، تنمو الطبقة الوحيدة بمقدار `L1_GROW_BY` صفوف
        (بدون حد أقصى).

        Returns: True إذا حدث نمو.
        """
        if self._steps_since_growth < self.plateau_cooldown:
            return False
        if not self._is_plateauing():
            return False

        old_rows = self.layers[0].out_dim
        self.grow()
        self._steps_since_growth = 0

        event = {
            "step": self._train_steps,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_rows": old_rows,
            "new_rows": self.layers[0].out_dim,
            "loss_at_growth": self._last_loss,
        }
        self._growth_events.append(event)
        logger.info(
            f"DeepRoutingNetwork.evolve_if_plateau(): plateau detected at "
            f"step {self._train_steps} — grew {old_rows}→{self.layers[0].out_dim} rows"
        )
        return True

    def train_batch(
        self,
        vectors: List[List[float]],
        targets: List[float],
    ) -> float:
        """
        Train on a batch of samples. Returns average loss.
        """
        if not vectors:
            return 0.0
        total_loss = sum(
            self.train_step(v, t)
            for v, t in zip(vectors, targets)
        )
        return total_loss / len(vectors)

    # ── Persistence (BrainCheckpoint compatible) ──────────────────────────

    def save(self, directory: str = WEIGHTS_DIR) -> str:
        """
        Save the layer weights to `directory/deep_network_layer_1_*.npy`.
        Also saves the current row count so load() can restore growable shape.
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for i, layer in enumerate(self.layers):
            prefix = str(d / f"deep_network_layer_{i+1}")
            layer.save(prefix)
        # Save training state + current row count (for growth persistence)
        state = np.array([
            self._train_steps,
            self._last_loss or 0.0,
            self.layers[0].out_dim,   # current rows
        ], dtype=np.float64)
        np.save(str(d / "deep_network_state.npy"), state)
        logger.info(
            f"DeepRoutingNetwork saved to {directory}  "
            f"steps={self._train_steps}  "
            f"rows={self.layers[0].out_dim}"
        )
        return str(d.resolve())

    def load(self, directory: str = WEIGHTS_DIR) -> None:
        """
        Load the layer weights from `directory`.
        Restores the row count from saved state.
        """
        d = Path(directory)
        for i, layer in enumerate(self.layers):
            prefix = str(d / f"deep_network_layer_{i+1}")
            w_path = f"{prefix}_weights.npy"
            if not os.path.exists(w_path):
                logger.warning(
                    f"DeepRoutingNetwork: layer {i+1} weights not found at {w_path}"
                )
                continue
            try:
                layer.load(prefix)
            except Exception as e:
                logger.warning(f"DeepRoutingNetwork: failed to load layer {i+1}: {e}")
        # Load training state
        state_path = str(d / "deep_network_state.npy")
        if os.path.exists(state_path):
            state = np.load(state_path)
            self._train_steps = int(state[0])
            self._last_loss = float(state[1]) if state[1] != 0.0 else None
            # Layer dims already restored by DenseLayer.load() reading .npy shape
        logger.info(
            f"DeepRoutingNetwork loaded from {directory}  "
            f"steps={self._train_steps}  "
            f"rows={self.layers[0].out_dim}"
        )

    # ── Compatibility layer (RoutingEngine / DynamicWeightLayer API) ──────

    @property
    def weights(self) -> np.ndarray:
        """
        Compatibility shim: returns the single layer's weights so code
        that reads `net.weights` (e.g., RoutingEngine, extract_routing_weights)
        works. Shape: (rows, 7) — cols always 7.
        """
        return self.layers[0].weights

    @property
    def SHAPE(self) -> Tuple[int, int]:
        """Compatibility shim — (rows, 7)."""
        return (self.layers[0].out_dim, self.layers[0].in_dim)

    def get_weights_list(self) -> List[List[float]]:
        """Return the layer's weights as list of lists (Phase 8 compatible)."""
        return self.layers[0].weights.tolist()

    def get_all_weights(self) -> Dict[str, List[List[float]]]:
        """Return the layer's weights (kept as dict for API compatibility)."""
        return {
            layer.name: layer.weights.tolist()
            for layer in self.layers
        }

    # ── Introspection ─────────────────────────────────────────────────────

    def _count_params(self) -> int:
        return sum(l.weights.size + l.biases.size for l in self.layers)

    def architecture_str(self) -> str:
        """Human-readable architecture description."""
        parts = [f"Input ({INPUT_DIM})"]
        for layer in self.layers:
            parts.append(f"{layer.name} ({layer.in_dim}→{layer.out_dim})")
        parts.append(f"Routing weights ({OUTPUT_DIM}, derived from output rows 0-3)")
        return " → ".join(parts)

    def summary(self) -> dict:
        recent_losses = self._loss_history[-50:] if self._loss_history else []
        total_params = self._count_params()
        return {
            "name": self.name,
            "architecture": self.architecture_str(),
            "layer_count": len(self.layers),
            "layers": [l.summary() for l in self.layers],
            "total_parameters": total_params,
            "layer1_rows": self.layers[0].out_dim,
            "layer1_cols_fixed": L1_COLS,
            "layer1_max_rows": None,  # NO upper limit — grows indefinitely
            "layer1_grow_by": L1_GROW_BY,
            "train_steps": self._train_steps,
            "last_loss": round(self._last_loss, 8) if self._last_loss else None,
            "avg_recent_loss": round(
                sum(recent_losses) / len(recent_losses), 8
            ) if recent_losses else None,
            "learning_rate": self.learning_rate,
            "plateau_window": self.plateau_window,
            "plateau_threshold": self.plateau_threshold,
            "plateau_cooldown": self.plateau_cooldown,
            "steps_since_growth": self._steps_since_growth,
            "growth_events": len(self._growth_events),
        }

    def __repr__(self) -> str:
        return (
            f"<DeepRoutingNetwork '{self.name}'  "
            f"layers={len(self.layers)}  "
            f"params={self._count_params()}  "
            f"L1_shape=({self.layers[0].out_dim},{self.layers[0].in_dim})  "
            f"steps={self._train_steps}>"
        )


# ── Routing weight extractor (Phase 8 API compatible) ────────────────────────

def extract_deep_routing_weights(net: DeepRoutingNetwork) -> dict:
    """
    Derive routing scalars from a DeepRoutingNetwork.

    Unlike the Phase 8 extractor (which reads from the weight matrix),
    this does a full forward pass with a *neutral* input vector so that
    all layers contribute to the output.
    """
    # Neutral input: all 0.5 (represents an average-quality route)
    neutral = np.full(INPUT_DIM, 0.5, dtype=np.float64)
    try:
        return net.predict_routing_weights(neutral)
    except Exception as e:
        logger.warning(f"extract_deep_routing_weights failed: {e}")
        return {
            "W_SEMANTIC": 0.30,
            "W_SCORE":    0.35,
            "W_MEMORY":   0.25,
            "W_TOPOLOGY": 0.10,
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_default_deep_network: Optional[DeepRoutingNetwork] = None


def get_default_deep_network(
    weights_dir: str = WEIGHTS_DIR,
) -> DeepRoutingNetwork:
    """
    Return (and cache) the module-level default DeepRoutingNetwork.

    Loads persisted weights from `weights_dir` if available.
    """
    global _default_deep_network
    if _default_deep_network is None:
        _default_deep_network = DeepRoutingNetwork(name="default_deep_router")
        layer1_path = os.path.join(weights_dir, "deep_network_layer_1_weights.npy")
        if os.path.exists(layer1_path):
            try:
                _default_deep_network.load(weights_dir)
                logger.info(f"DeepRoutingNetwork restored from {weights_dir}")
            except Exception as e:
                logger.warning(f"Could not load deep network from {weights_dir}: {e}")
    return _default_deep_network
