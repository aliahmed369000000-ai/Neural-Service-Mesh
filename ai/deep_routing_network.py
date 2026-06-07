"""
Phase 9 — Axis 3: Deep Multi-Layer Neural Network
==================================================
Elevates the project from a *linear model* (single weight matrix) to a
genuine **deep neural network** with multiple learned layers.

Architecture (matches the images exactly):

    Input (7 features — from RichDataCollector)
        ↓
    Layer 1: 9×7   — تتعلم الأنماط البسيطة
        ↓
    Layer 2: 16×9  — تتعلم الأنماط المركبة
        ↓
    Layer 3: 8×16  — تضغط المعرفة
        ↓
    Output: 4 routing weights (W_SEMANTIC, W_SCORE, W_MEMORY, W_TOPOLOGY)

Each layer:
  - Stores a weight matrix of shape (out_dim × in_dim)
  - Uses ReLU activation (except the final output layer which uses Softmax
    so the 4 routing weights always sum to 1)
  - Supports gradient-based training via backpropagation through all layers

The DeepRoutingNetwork integrates with both the existing RoutingEngine
(by providing the same `extract_routing_weights()` interface) and with
the Phase 9 DynamicWeightLayer (which can replace Layer 1 for
self-growing capability).

Backward compatibility
----------------------
`get_deep_network()` returns a singleton instance.
`extract_deep_routing_weights(net)` returns the same 4-key dict as the
Phase 8 `extract_routing_weights(layer)`.
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
# Each tuple: (output_dim, input_dim, activation)
LAYER_CONFIGS: List[Tuple[int, int, str]] = [
    (9,  7,  "relu"),    # Layer 1: simple patterns
    (16, 9,  "relu"),    # Layer 2: composite patterns
    (8,  16, "relu"),    # Layer 3: knowledge compression
    (4,  8,  "softmax"), # Output:  routing weights (sum to 1)
]

INPUT_DIM  = 7
OUTPUT_DIM = 4
LEARNING_RATE = 0.005
WEIGHTS_DIR = "models/classifiers"


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
      biases  : (out_dim,)
    """

    def __init__(self, out_dim: int, in_dim: int,
                 activation: str = "relu", name: str = ""):
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.activation = activation
        self.name = name or f"layer_{out_dim}x{in_dim}"

        self.weights: np.ndarray = _xavier_init(out_dim, in_dim)
        self.biases: np.ndarray = np.zeros(out_dim, dtype=np.float64)

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


# ─────────────────────────────────────────────────────────────────────────────
#  Deep Routing Network
# ─────────────────────────────────────────────────────────────────────────────

class DeepRoutingNetwork:
    """
    Phase 9 Axis-3: Deep multi-layer routing weight predictor.

    Architecture:
        Input (7) → Layer1 9×7 → Layer2 16×9 → Layer3 8×16 → Output 4 (softmax)

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
    ):
        self.name = name
        self.learning_rate = learning_rate
        self._train_steps = 0
        self._last_loss: Optional[float] = None
        self._loss_history: List[float] = []

        # Build layers
        self.layers: List[DenseLayer] = []
        for i, (out_dim, in_dim, act) in enumerate(LAYER_CONFIGS):
            layer = DenseLayer(
                out_dim, in_dim, act,
                name=f"L{i+1}_{out_dim}x{in_dim}_{act}"
            )
            self.layers.append(layer)

        logger.info(
            f"DeepRoutingNetwork '{self.name}' initialised — "
            f"{len(self.layers)} layers: "
            + " → ".join(
                f"{l.in_dim}→{l.out_dim}" for l in self.layers
            )
        )

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Run the full forward pass through all layers.

        Parameters
        ----------
        x : array-like of length INPUT_DIM (7)
            Normalised feature vector from RichDataCollector.

        Returns
        -------
        np.ndarray shape (4,) — softmax routing weights summing to 1.
        """
        h = np.array(x, dtype=np.float64)
        # Pad or truncate to INPUT_DIM
        if h.shape[0] < INPUT_DIM:
            h = np.pad(h, (0, INPUT_DIM - h.shape[0]))
        elif h.shape[0] > INPUT_DIM:
            h = h[:INPUT_DIM]

        for layer in self.layers:
            h = layer.forward(h)
        return h

    def predict_routing_weights(self, x: Union[List[float], np.ndarray]) -> dict:
        """
        Forward pass returning the 4-key routing weights dict.

        Returns
        -------
        dict with keys: W_SEMANTIC, W_SCORE, W_MEMORY, W_TOPOLOGY
        """
        out = self.forward(x)
        return {
            "W_SEMANTIC": round(float(out[0]), 6),
            "W_SCORE":    round(float(out[1]), 6),
            "W_MEMORY":   round(float(out[2]), 6),
            "W_TOPOLOGY": round(float(out[3]), 6),
        }

    # ── Training step (full backpropagation) ─────────────────────────────

    def train_step(
        self,
        input_vector: Union[List[float], np.ndarray],
        target: float,
    ) -> float:
        """
        Supervised training step using full backpropagation.

        Loss: MSE between output vector and target broadcast to (4,).

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

        # Target broadcast: uniform distribution weighted toward target score
        target_vec = np.array([target * 0.30, target * 0.35,
                                target * 0.25, target * 0.10], dtype=np.float64)
        target_sum = target_vec.sum()
        if target_sum > 0:
            target_vec = target_vec / target_sum

        error = output - target_vec
        loss = float(np.mean(error ** 2))

        # Backward through all layers in reverse
        grad = 2.0 * error / OUTPUT_DIM
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.learning_rate)

        self._train_steps += 1
        self._last_loss = loss
        self._loss_history.append(loss)
        if len(self._loss_history) > 1000:
            self._loss_history = self._loss_history[-1000:]

        logger.debug(
            f"DeepRoutingNetwork train_step #{self._train_steps}  "
            f"loss={loss:.6f}  output={[round(v, 3) for v in output.tolist()]}"
        )
        return loss

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

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, directory: str = WEIGHTS_DIR) -> str:
        """Save all layer weights to `directory/deep_network_layer_N_*.npy`."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for i, layer in enumerate(self.layers):
            prefix = str(d / f"deep_network_layer_{i+1}")
            layer.save(prefix)
        # Save training state
        state = np.array([self._train_steps,
                          self._last_loss or 0.0], dtype=np.float64)
        np.save(str(d / "deep_network_state.npy"), state)
        logger.info(
            f"DeepRoutingNetwork saved to {directory}  "
            f"steps={self._train_steps}"
        )
        return str(d.resolve())

    def load(self, directory: str = WEIGHTS_DIR) -> None:
        """Load all layer weights from `directory`."""
        d = Path(directory)
        for i, layer in enumerate(self.layers):
            prefix = str(d / f"deep_network_layer_{i+1}")
            w_path = f"{prefix}_weights.npy"
            if not os.path.exists(w_path):
                logger.warning(f"DeepRoutingNetwork: layer {i+1} weights not found at {w_path}")
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
        logger.info(
            f"DeepRoutingNetwork loaded from {directory}  "
            f"steps={self._train_steps}"
        )

    # ── Compatibility layer (Phase 8 API) ─────────────────────────────────

    @property
    def weights(self) -> np.ndarray:
        """
        Compatibility shim: returns Layer 1's weights so code that reads
        `layer.weights` still works (e.g., extract_routing_weights from Phase 8).
        """
        return self.layers[0].weights

    @property
    def SHAPE(self) -> Tuple[int, int]:
        """Compatibility shim."""
        return (self.layers[0].out_dim, self.layers[0].in_dim)

    def get_weights_list(self) -> List[List[float]]:
        """Return Layer 1 weights as list of lists (Phase 8 compatible)."""
        return self.layers[0].weights.tolist()

    def get_all_weights(self) -> Dict[str, List[List[float]]]:
        """Return all layers' weights."""
        return {
            layer.name: layer.weights.tolist()
            for layer in self.layers
        }

    # ── Introspection ─────────────────────────────────────────────────────

    def architecture_str(self) -> str:
        """Human-readable architecture description."""
        parts = [f"Input ({INPUT_DIM})"]
        for layer in self.layers:
            parts.append(f"{layer.name} ({layer.in_dim}→{layer.out_dim})")
        parts.append(f"Output ({OUTPUT_DIM} routing weights)")
        return " → ".join(parts)

    def summary(self) -> dict:
        recent_losses = self._loss_history[-50:] if self._loss_history else []
        return {
            "name": self.name,
            "architecture": self.architecture_str(),
            "layer_count": len(self.layers),
            "layers": [l.summary() for l in self.layers],
            "total_parameters": sum(
                l.weights.size + l.biases.size for l in self.layers
            ),
            "train_steps": self._train_steps,
            "last_loss": round(self._last_loss, 8) if self._last_loss else None,
            "avg_recent_loss": round(
                sum(recent_losses) / len(recent_losses), 8
            ) if recent_losses else None,
            "learning_rate": self.learning_rate,
        }

    def __repr__(self) -> str:
        return (
            f"<DeepRoutingNetwork '{self.name}'  "
            f"layers={len(self.layers)}  "
            f"params={sum(l.weights.size for l in self.layers)}  "
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
