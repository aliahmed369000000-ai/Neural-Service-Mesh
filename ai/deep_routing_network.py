"""
Phase 9 — Axis 3: Deep Routing Network (v18 — three layers, expanded)
======================================================================
Upgraded from v17 (single layer 7→108) to a proper 3-layer network:

Architecture (v18):
    Input (128 features — CKG concept vector)
        ↓
    Layer 1: 512×128  relu   (rows growable +8, no upper limit)
        ↓
    Layer 2: 256×512  relu
        ↓
    Layer 3:  16×256  softmax
        ↓
    Output: 16-dim routing weight vector
      → 4 primary weights (W_SEMANTIC, W_SCORE, W_MEMORY, W_TOPOLOGY)
        derived from rows 0-3, normalised to sum=1 (backward compatible).

Backward compatibility
----------------------
ALL public API preserved: forward(), train_step(), train_batch(),
predict_routing_weights(), grow(), load_custom_weights(), save(), load(),
summary(), architecture_str(), .layers[0], .weights, .SHAPE,
get_default_deep_network(), extract_deep_routing_weights()

Old 7-feature vectors are zero-padded to 256 automatically.
Old v17 weights (108×7 or 112×7 from nnn_112.csv) are migrated via migrate_weights().
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

# ── Architecture constants (v18) ──────────────────────────────────────────────
INPUT_DIM      = 256   # موحّد: 7 دلالي + 249 TF-IDF hash
HIDDEN1_DIM    = 112   # L1: 112×256
HIDDEN2_DIM    = 32    # L2: 32×112
OUTPUT_DIM     = 4     # L3: 4×32 → 4 routing weights
LEARNING_RATE  = 0.003
WEIGHTS_DIR    = "models/classifiers"

L1_INITIAL_ROWS = 112  # موحّد مع neural_core
L1_GROW_BY      = 8
L1_MAX_ROWS     = None
L1_COLS         = INPUT_DIM  # FIXED at 128

PLATEAU_WINDOW    = 50
PLATEAU_THRESHOLD = 0.01
PLATEAU_COOLDOWN  = 200

LAYER_CONFIGS: List[Tuple[int, int, str]] = [
    (HIDDEN1_DIM, INPUT_DIM,   "relu"),
    (HIDDEN2_DIM, HIDDEN1_DIM, "relu"),
    (OUTPUT_DIM,  HIDDEN2_DIM, "softmax"),
]


def _xavier_init(rows: int, cols: int) -> np.ndarray:
    limit = math.sqrt(6.0 / (rows + cols))
    return np.random.uniform(-limit, limit, size=(rows, cols)).astype(np.float64)


def _he_init(rows: int, cols: int) -> np.ndarray:
    std = math.sqrt(2.0 / cols)
    return np.random.normal(0.0, std, size=(rows, cols)).astype(np.float64)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_deriv(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max()
    exp_x = np.exp(np.clip(shifted, -500, 500))
    total = exp_x.sum()
    if total == 0:
        return np.ones(len(x)) / len(x)
    return exp_x / total


class DenseLayer:
    def __init__(self, out_dim: int, in_dim: int,
                 activation: str = "relu", name: str = ""):
        self.out_dim = out_dim
        self.in_dim  = in_dim
        self.activation = activation
        self.name = name or f"layer_{out_dim}x{in_dim}"

        if activation == "relu":
            self.weights: np.ndarray = _he_init(out_dim, in_dim)
        else:
            self.weights: np.ndarray = _xavier_init(out_dim, in_dim)
        self.biases: np.ndarray = np.zeros(out_dim, dtype=np.float64)

        self._last_input:   Optional[np.ndarray] = None
        self._last_pre_act: Optional[np.ndarray] = None
        self._last_output:  Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_input = x.copy()
        pre = self.weights @ x + self.biases
        self._last_pre_act = pre.copy()
        if self.activation == "relu":
            out = _relu(pre)
        elif self.activation == "softmax":
            out = _softmax(pre)
        else:
            out = pre
        self._last_output = out.copy()
        return out

    def backward(self, grad_output: np.ndarray,
                 learning_rate: float) -> np.ndarray:
        if self.activation == "relu":
            grad_pre = grad_output * _relu_deriv(self._last_pre_act)
        else:
            grad_pre = grad_output
        grad_w = np.outer(grad_pre, self._last_input)
        grad_x = self.weights.T @ grad_pre
        self.weights -= learning_rate * grad_w
        self.biases  -= learning_rate * grad_pre
        self.weights  = np.clip(self.weights, -5.0, 5.0)
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


class DeepRoutingNetwork:
    """
    Phase 9 Axis-3: Three-layer routing network (v18).
    Input(128) → L1(512,relu) → L2(256,relu) → L3(16,softmax)
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
        self.plateau_window    = plateau_window
        self.plateau_threshold = plateau_threshold
        self.plateau_cooldown  = plateau_cooldown
        self._steps_since_growth = 0
        self._growth_events: List[dict] = []

        l1 = DenseLayer(L1_INITIAL_ROWS, INPUT_DIM,   "relu",
                        name=f"L1_{L1_INITIAL_ROWS}x{INPUT_DIM}_relu")
        l2 = DenseLayer(HIDDEN2_DIM, L1_INITIAL_ROWS, "relu",
                        name=f"L2_{HIDDEN2_DIM}x{L1_INITIAL_ROWS}_relu")
        l3 = DenseLayer(OUTPUT_DIM,  HIDDEN2_DIM,     "softmax",
                        name=f"L3_{OUTPUT_DIM}x{HIDDEN2_DIM}_softmax")

        self.layers: List[DenseLayer] = [l1, l2, l3]

        logger.info(
            f"DeepRoutingNetwork '{self.name}' v18 — "
            f"128→512(relu)→256(relu)→16(softmax) | "
            f"params: {self._count_params():,}"
        )

    def grow(self) -> bool:
        l1, l2 = self.layers[0], self.layers[1]
        old_rows = l1.out_dim
        added    = L1_GROW_BY
        new_rows = old_rows + added

        new_w1 = _he_init(added, l1.in_dim)
        l1.weights = np.vstack([l1.weights, new_w1])
        l1.biases  = np.concatenate([l1.biases, np.zeros(added)])
        l1.out_dim = new_rows
        l1.name    = f"L1_{new_rows}x{l1.in_dim}_relu"

        new_cols2 = _he_init(l2.out_dim, added)
        l2.weights = np.hstack([l2.weights, new_cols2])
        l2.in_dim  = new_rows
        l2.name    = f"L2_{l2.out_dim}x{new_rows}_relu"

        logger.info(
            f"grow(): L1 {old_rows}→{new_rows} | "
            f"total params: {self._count_params():,}"
        )
        return True

    def migrate_weights(self, old_weights: np.ndarray) -> None:
        """Migrate old weights (any shape: 108×7, 112×7, etc.) into new L1 (128-dim).
        الأوزان المدروسة من nnn_112.csv (112×7) تُدمج تلقائياً في أول 112 صف من L1.
        """
        old = np.array(old_weights, dtype=np.float64)
        r = min(old.shape[0], self.layers[0].out_dim)
        c = min(old.shape[1], self.layers[0].in_dim)
        self.layers[0].weights[:r, :c] = old[:r, :c]
        logger.info(f"migrate_weights: copied {r}×{c} old weights into L1")

    def load_custom_weights(self, matrix: np.ndarray,
                            layer_index: int = 0) -> None:
        m = np.array(matrix, dtype=np.float64)
        if layer_index >= len(self.layers):
            raise ValueError(f"layer_index {layer_index} out of range")
        layer = self.layers[layer_index]
        if layer_index == 0 and m.ndim == 2 and m.shape[1] != INPUT_DIM:
            raise ValueError(f"L1 cols FIXED at {INPUT_DIM}. Got {m.shape}.")
        layer.weights = m
        layer.biases  = np.zeros(m.shape[0], dtype=np.float64)
        layer.out_dim = m.shape[0]
        layer.in_dim  = m.shape[1]

    def forward(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        h = np.array(x, dtype=np.float64).ravel()
        if h.shape[0] < INPUT_DIM:
            h = np.pad(h, (0, INPUT_DIM - h.shape[0]))
        elif h.shape[0] > INPUT_DIM:
            h = h[:INPUT_DIM]
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def predict_routing_weights(
        self, x: Union[List[float], np.ndarray]
    ) -> dict:
        out = self.forward(x)
        head = out[:4]
        total = float(head.sum())
        if total <= 0.0:
            return {"W_SEMANTIC": 0.30, "W_SCORE": 0.35,
                    "W_MEMORY": 0.25, "W_TOPOLOGY": 0.10}
        normed = (head / total).tolist()
        return {
            "W_SEMANTIC": round(normed[0], 6),
            "W_SCORE":    round(normed[1], 6),
            "W_MEMORY":   round(normed[2], 6),
            "W_TOPOLOGY": round(normed[3], 6),
        }

    def train_step(
        self,
        input_vector: Union[List[float], np.ndarray],
        target: float,
    ) -> float:
        x = np.array(input_vector, dtype=np.float64).ravel()
        if x.shape[0] < INPUT_DIM:
            x = np.pad(x, (0, INPUT_DIM - x.shape[0]))
        elif x.shape[0] > INPUT_DIM:
            x = x[:INPUT_DIM]

        output = self.forward(x)
        target_vec = np.full(OUTPUT_DIM, float(target), dtype=np.float64)
        error = output - target_vec
        loss  = float(np.mean(error ** 2))

        grad = 2.0 * error / OUTPUT_DIM
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.learning_rate)

        self._train_steps += 1
        self._steps_since_growth += 1
        self._last_loss = loss
        self._loss_history.append(loss)
        if len(self._loss_history) > 1000:
            self._loss_history = self._loss_history[-1000:]
        self.evolve_if_plateau()
        return loss

    def train_batch(self, vectors: List[List[float]],
                    targets: List[float]) -> float:
        if not vectors:
            return 0.0
        return sum(self.train_step(v, t) for v, t in zip(vectors, targets)) / len(vectors)

    def _is_plateauing(self) -> bool:
        hist = self._loss_history
        w = self.plateau_window
        if len(hist) < w * 2:
            return False
        window = hist[-w * 2:]
        mean_older  = float(np.mean(window[:w]))
        mean_recent = float(np.mean(window[w:]))
        if mean_older == 0.0:
            return False
        return (mean_older - mean_recent) / mean_older < self.plateau_threshold

    def evolve_if_plateau(self) -> bool:
        if self._steps_since_growth < self.plateau_cooldown:
            return False
        if not self._is_plateauing():
            return False
        old_rows = self.layers[0].out_dim
        self.grow()
        self._steps_since_growth = 0
        self._growth_events.append({
            "step": self._train_steps,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_rows": old_rows,
            "new_rows": self.layers[0].out_dim,
            "loss_at_growth": self._last_loss,
        })
        return True

    def save(self, directory: str = WEIGHTS_DIR) -> str:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for i, layer in enumerate(self.layers):
            layer.save(str(d / f"deep_network_layer_{i+1}"))
        np.save(str(d / "deep_network_state.npy"), np.array([
            self._train_steps, self._last_loss or 0.0,
            self.layers[0].out_dim, INPUT_DIM, OUTPUT_DIM,
        ], dtype=np.float64))
        return str(d.resolve())

    def load(self, directory: str = WEIGHTS_DIR) -> None:
        d = Path(directory)
        for i, layer in enumerate(self.layers):
            prefix = str(d / f"deep_network_layer_{i+1}")
            if os.path.exists(f"{prefix}_weights.npy"):
                try:
                    layer.load(prefix)
                except Exception as e:
                    logger.warning(f"layer {i+1} load failed: {e}")
        state_path = str(d / "deep_network_state.npy")
        if os.path.exists(state_path):
            state = np.load(state_path)
            self._train_steps = int(state[0])
            self._last_loss = float(state[1]) if state[1] != 0.0 else None

    @property
    def weights(self) -> np.ndarray:
        return self.layers[0].weights

    @property
    def SHAPE(self) -> Tuple[int, int]:
        return (self.layers[0].out_dim, self.layers[0].in_dim)

    def get_weights_list(self) -> List[List[float]]:
        return self.layers[0].weights.tolist()

    def get_all_weights(self) -> Dict[str, List[List[float]]]:
        return {l.name: l.weights.tolist() for l in self.layers}

    def _count_params(self) -> int:
        return sum(l.weights.size + l.biases.size for l in self.layers)

    def architecture_str(self) -> str:
        parts = [f"Input({INPUT_DIM})"]
        for layer in self.layers:
            parts.append(f"{layer.name}")
        parts.append(f"RoutingWeights({OUTPUT_DIM}→first4=W_SEMANTIC/SCORE/MEMORY/TOPOLOGY)")
        return " → ".join(parts)

    def summary(self) -> dict:
        recent = self._loss_history[-50:] if self._loss_history else []
        return {
            "name": self.name, "version": "v18",
            "architecture": self.architecture_str(),
            "layer_count": len(self.layers),
            "layers": [l.summary() for l in self.layers],
            "total_parameters": self._count_params(),
            "input_dim": INPUT_DIM, "output_dim": OUTPUT_DIM,
            "layer1_rows": self.layers[0].out_dim,
            "layer1_cols_fixed": INPUT_DIM,
            "layer1_grow_by": L1_GROW_BY,
            "train_steps": self._train_steps,
            "last_loss": round(self._last_loss, 8) if self._last_loss else None,
            "avg_recent_loss": round(sum(recent)/len(recent), 8) if recent else None,
            "learning_rate": self.learning_rate,
            "steps_since_growth": self._steps_since_growth,
            "growth_events": len(self._growth_events),
        }

    def __repr__(self) -> str:
        return (
            f"<DeepRoutingNetwork '{self.name}' v18  "
            f"params={self._count_params():,}  "
            f"arch=128→{self.layers[0].out_dim}→"
            f"{self.layers[1].out_dim}→{self.layers[2].out_dim}  "
            f"steps={self._train_steps}>"
        )


# ── CKG query encoder ─────────────────────────────────────────────────────────

def encode_query_to_ckg_vector(
    query: str,
    ckg_concepts: dict,
    dim: int = INPUT_DIM,
) -> np.ndarray:
    """
    Encode an Arabic query into a 128-dim CKG concept vector (L2-normalised).
    Each position = one CKG concept, weighted by name-match × strength.
    """
    import re
    def clean(t: str) -> str:
        t = re.sub(r'[ٱ]', 'ا', t)
        t = re.sub(r'[ًٌٍَُِّْٰ]', '', t)
        t = re.sub(r'[^\u0600-\u06FF\s]', ' ', t)
        return t.strip()

    q_words = set(clean(query).split())
    vec = np.zeros(dim, dtype=np.float64)
    for i, (name, meta) in enumerate(list(ckg_concepts.items())[:dim]):
        name_clean = clean(name)
        strength   = meta.get("strength", 0.1)
        freq_norm  = min(1.0, meta.get("frequency", 1) / 500)
        match = 1.0 if (name_clean in q_words or
                        any(w in name_clean for w in q_words if len(w) >= 3)) else 0.0
        vec[i] = match * strength * (1.0 + freq_norm)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ── Routing weight extractor (Phase 8 API compatible) ────────────────────────

def extract_deep_routing_weights(net: DeepRoutingNetwork) -> dict:
    neutral = np.full(INPUT_DIM, 0.5, dtype=np.float64)
    try:
        return net.predict_routing_weights(neutral)
    except Exception as e:
        logger.warning(f"extract_deep_routing_weights failed: {e}")
        return {"W_SEMANTIC": 0.30, "W_SCORE": 0.35,
                "W_MEMORY": 0.25, "W_TOPOLOGY": 0.10}


# ── Module-level singleton ────────────────────────────────────────────────────

_default_deep_network: Optional[DeepRoutingNetwork] = None


def get_default_deep_network(weights_dir: str = WEIGHTS_DIR) -> DeepRoutingNetwork:
    global _default_deep_network
    if _default_deep_network is None:
        _default_deep_network = DeepRoutingNetwork(name="default_deep_router")
        layer1_path = os.path.join(weights_dir, "deep_network_layer_1_weights.npy")
        if os.path.exists(layer1_path):
            try:
                old_w = np.load(layer1_path).astype(np.float64)
                if old_w.shape[1] <= 7:
                    # v17 weights → migrate
                    _default_deep_network.migrate_weights(old_w)
                    logger.info(f"Migrated v17 weights {old_w.shape} into v18 L1")
                else:
                    _default_deep_network.load(weights_dir)
                    logger.info(f"DeepRoutingNetwork v18 restored from {weights_dir}")
            except Exception as e:
                logger.warning(f"Could not load weights: {e}")
    return _default_deep_network
