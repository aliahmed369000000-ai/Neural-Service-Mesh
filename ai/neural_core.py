"""
Neural Core — One Neural Core (إعادة كتابة كاملة)
====================================================
شبكة عصبية صحيحة رياضياً (numpy نظيف) + ذاكرة عصبية ترابطية (RAG-style).

تصحيح الأخطاء الرياضية من الإصدارات السابقة:
  • تدرّج ReLU كان يُحسب بدون ضرب في relu'(pre-activation) في الطبقات
    المفردة (NeuralWeightLayer / DynamicWeightLayer القديمة) — تم تصحيحه.
  • bias لم يكن يُحدَّث أبداً — الآن يُحدَّث بشكل صحيح: grad_b = grad_pre.
  • تدرّج softmax كان يُعامَل كخطي (تقريب خاطئ) — الآن نستخدم اشتقاق
    Jacobian الصحيح لـ softmax مع MSE:
        dL/dz_i = sum_j (dL/dout_j) * out_j * (delta_ij - out_i)
  • آلية "النمو" (plateau growth) أصبحت اختيارية، صريحة، ومُسجَّلة،
    ولا تتدخل في صحة التدرجات (التوسعة بعد التدريب فقط، تُهيَّأ
    أوزانها بـ Xavier ولا تكسر سلسلة backprop لأنها لا تُستخدم إلا
    في الخطوة التالية).

المكوّنات:
  1. DenseLayer        — طبقة كاملة الاتصال + backprop صحيح (relu/softmax/linear/tanh/sigmoid)
  2. NeuralNetwork     — شبكة متعددة الطبقات قابلة للتشكيل (MLP عام)
  3. AssociativeMemory — ذاكرة عصبية: تخزين متجهات + استرجاع بالتشابه (RAG)
  4. NeuralCore        — الواجهة الموحّدة: "يتعلم / يتذكر / يتطور"
  5. طبقات توافق خلفي: NeuralWeightLayer / DynamicWeightLayer (API قديم
     لكن بحسابات رياضية صحيحة الآن، تُبنى فوق DenseLayer/NeuralNetwork)
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("NeuralCore")

ArrayLike = Union[Sequence[float], np.ndarray]


# ════════════════════════════════════════════════════════════════════════
# دوال التفعيل ومشتقاتها (مُعرَّفة بالنسبة لـ pre-activation z)
# ════════════════════════════════════════════════════════════════════════

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    """مشتقة ReLU بالنسبة لـ z (pre-activation)."""
    return (z > 0.0).astype(np.float64)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def sigmoid_grad_from_output(out: np.ndarray) -> np.ndarray:
    """مشتقة sigmoid بالنسبة لـ z، معبَّرة عبر الخرج out=sigmoid(z)."""
    return out * (1.0 - out)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_grad_from_output(out: np.ndarray) -> np.ndarray:
    return 1.0 - out ** 2


def softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - np.max(z)
    exp_z = np.exp(shifted)
    total = exp_z.sum()
    if total == 0:
        return np.ones_like(z) / len(z)
    return exp_z / total


def linear(z: np.ndarray) -> np.ndarray:
    return z


ACTIVATIONS = {"relu", "sigmoid", "tanh", "softmax", "linear"}


# ════════════════════════════════════════════════════════════════════════
# 1) DenseLayer — طبقة كاملة الاتصال مع backprop صحيح رياضياً
# ════════════════════════════════════════════════════════════════════════

class DenseLayer:
    """
    طبقة Dense: out = activation(W @ x + b)

    W : (out_dim, in_dim)
    b : (out_dim,)

    التدرجات (Backward) مُشتقّة بشكل صحيح لكل دالة تفعيل مدعومة:

      - relu    : dL/dz = dL/dout ⊙ relu'(z)
      - sigmoid : dL/dz = dL/dout ⊙ out ⊙ (1-out)
      - tanh    : dL/dz = dL/dout ⊙ (1 - out²)
      - linear  : dL/dz = dL/dout
      - softmax : dL/dz_i = Σ_j dL/dout_j * out_j * (δ_ij - out_i)
                  (Jacobian الكامل لـ softmax — صحيح مع أي دالة خسارة عليا)

    بعد الحصول على dL/dz:
      dL/dW = outer(dL/dz, x)
      dL/db = dL/dz
      dL/dx = W.T @ (dL/dz)     ← يُمرَّر للطبقة السابقة
    """

    def __init__(self, out_dim: int, in_dim: int, activation: str = "relu",
                 name: str = "", weight_init: str = "xavier",
                 seed: Optional[int] = None):
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation غير معروفة: {activation}")

        self.out_dim = int(out_dim)
        self.in_dim = int(in_dim)
        self.activation = activation
        self.name = name or f"dense_{out_dim}x{in_dim}_{activation}"

        rng = np.random.default_rng(seed)
        if weight_init == "xavier":
            limit = math.sqrt(6.0 / (in_dim + out_dim))
            self.W = rng.uniform(-limit, limit, size=(out_dim, in_dim)).astype(np.float64)
        elif weight_init == "zeros":
            self.W = np.zeros((out_dim, in_dim), dtype=np.float64)
        else:
            self.W = rng.normal(0, 0.1, size=(out_dim, in_dim)).astype(np.float64)

        self.b = np.zeros(out_dim, dtype=np.float64)

        # ── حالة مخبأة للـ backward ──
        self._x: Optional[np.ndarray] = None
        self._z: Optional[np.ndarray] = None
        self._out: Optional[np.ndarray] = None

        # ── Adam optimizer state (اختياري) ──
        self._m_W = np.zeros_like(self.W)
        self._v_W = np.zeros_like(self.W)
        self._m_b = np.zeros_like(self.b)
        self._v_b = np.zeros_like(self.b)
        self._adam_t = 0

    # ── Forward ──────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        z = self.W @ x + self.b
        self._z = z

        if self.activation == "relu":
            out = relu(z)
        elif self.activation == "sigmoid":
            out = sigmoid(z)
        elif self.activation == "tanh":
            out = tanh(z)
        elif self.activation == "softmax":
            out = softmax(z)
        else:
            out = linear(z)

        self._out = out
        return out

    # ── Backward (صحيح رياضياً) ──────────────────────────────────────

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        d_out : dL/dout لهذه الطبقة (من الطبقة التالية أو من دالة الخسارة)

        Returns: dL/dx (يُمرَّر للطبقة السابقة)
        ويحسب dL/dW و dL/db ويخزّنها في self._grad_W / self._grad_b
        دون تحديث الأوزان (التحديث في خطوة منفصلة عبر apply_gradients).
        """
        z, out, x = self._z, self._out, self._x

        if self.activation == "relu":
            d_z = d_out * relu_grad(z)
        elif self.activation == "sigmoid":
            d_z = d_out * sigmoid_grad_from_output(out)
        elif self.activation == "tanh":
            d_z = d_out * tanh_grad_from_output(out)
        elif self.activation == "softmax":
            # Jacobian الكامل لـ softmax: d_z_i = sum_j d_out_j * out_j * (delta_ij - out_i)
            # = out * (d_out - sum(d_out * out))
            s = float(np.dot(d_out, out))
            d_z = out * (d_out - s)
        else:  # linear
            d_z = d_out

        self._grad_W = np.outer(d_z, x)
        self._grad_b = d_z

        d_x = self.W.T @ d_z
        return d_x

    # ── تحديث الأوزان ─────────────────────────────────────────────────

    def apply_gradients(self, learning_rate: float, clip: float = 5.0,
                         optimizer: str = "sgd",
                         beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        if optimizer == "sgd":
            self.W -= learning_rate * self._grad_W
            self.b -= learning_rate * self._grad_b
        elif optimizer == "adam":
            self._adam_t += 1
            t = self._adam_t

            self._m_W = beta1 * self._m_W + (1 - beta1) * self._grad_W
            self._v_W = beta2 * self._v_W + (1 - beta2) * (self._grad_W ** 2)
            m_hat_W = self._m_W / (1 - beta1 ** t)
            v_hat_W = self._v_W / (1 - beta2 ** t)
            self.W -= learning_rate * m_hat_W / (np.sqrt(v_hat_W) + eps)

            self._m_b = beta1 * self._m_b + (1 - beta1) * self._grad_b
            self._v_b = beta2 * self._v_b + (1 - beta2) * (self._grad_b ** 2)
            m_hat_b = self._m_b / (1 - beta1 ** t)
            v_hat_b = self._v_b / (1 - beta2 ** t)
            self.b -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + eps)
        else:
            raise ValueError(f"optimizer غير معروف: {optimizer}")

        if clip is not None:
            np.clip(self.W, -clip, clip, out=self.W)
            np.clip(self.b, -clip, clip, out=self.b)

    # ── إدارة الأبعاد (نمو صريح، لا يكسر backprop لأنه بين الخطوات) ────

    def grow_out(self, add_rows: int, seed: Optional[int] = None) -> None:
        """يضيف `add_rows` صفوف جديدة (نيورونات خرج إضافية) بتهيئة Xavier."""
        if add_rows <= 0:
            return
        rng = np.random.default_rng(seed)
        limit = math.sqrt(6.0 / (self.in_dim + self.out_dim + add_rows))
        new_W = rng.uniform(-limit, limit, size=(add_rows, self.in_dim)).astype(np.float64)
        new_b = np.zeros(add_rows, dtype=np.float64)

        self.W = np.vstack([self.W, new_W])
        self.b = np.concatenate([self.b, new_b])
        self.out_dim += add_rows

        self._m_W = np.zeros_like(self.W)
        self._v_W = np.zeros_like(self.W)
        self._m_b = np.zeros_like(self.b)
        self._v_b = np.zeros_like(self.b)

    def grow_in(self, add_cols: int, seed: Optional[int] = None) -> None:
        """يضيف `add_cols` أعمدة جديدة (يستخدم عند نمو الطبقة السابقة)."""
        if add_cols <= 0:
            return
        rng = np.random.default_rng(seed)
        limit = math.sqrt(6.0 / (self.in_dim + add_cols + self.out_dim))
        new_cols = rng.uniform(-limit, limit, size=(self.out_dim, add_cols)).astype(np.float64)
        self.W = np.hstack([self.W, new_cols])
        self.in_dim += add_cols

        self._m_W = np.zeros_like(self.W)
        self._v_W = np.zeros_like(self.W)

    # ── حفظ/تحميل ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "activation": self.activation,
            "out_dim": self.out_dim,
            "in_dim": self.in_dim,
            "W": self.W.tolist(),
            "b": self.b.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DenseLayer":
        layer = cls(d["out_dim"], d["in_dim"], d["activation"], d.get("name", ""))
        layer.W = np.array(d["W"], dtype=np.float64)
        layer.b = np.array(d["b"], dtype=np.float64)
        return layer

    def summary(self) -> dict:
        return {
            "name": self.name,
            "shape": [self.out_dim, self.in_dim],
            "activation": self.activation,
            "params": int(self.W.size + self.b.size),
            "weight_stats": {
                "min": round(float(self.W.min()), 6),
                "max": round(float(self.W.max()), 6),
                "mean": round(float(self.W.mean()), 6),
                "std": round(float(self.W.std()), 6),
            },
            "bias_stats": {
                "min": round(float(self.b.min()), 6),
                "max": round(float(self.b.max()), 6),
                "mean": round(float(self.b.mean()), 6),
            },
        }

    def __repr__(self) -> str:
        return f"<DenseLayer '{self.name}' ({self.in_dim}→{self.out_dim}) act={self.activation}>"


# ════════════════════════════════════════════════════════════════════════
# دوال الخسارة ومشتقاتها (dL/dout)
# ════════════════════════════════════════════════════════════════════════

def mse_loss(out: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Mean Squared Error.
      L = mean((out - target)^2)
      dL/dout = 2*(out - target) / N
    """
    error = out - target
    loss = float(np.mean(error ** 2))
    grad = (2.0 / out.size) * error
    return loss, grad


def cross_entropy_loss(out: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> Tuple[float, np.ndarray]:
    """
    Categorical Cross-Entropy (يُستخدم عادة بعد softmax).
      L = -Σ target_i * log(out_i)
      dL/dout_i = -target_i / out_i
    (يُمرَّر إلى DenseLayer.backward الذي يطبّق Jacobian softmax الصحيح)
    """
    out_clipped = np.clip(out, eps, 1.0)
    loss = float(-np.sum(target * np.log(out_clipped)))
    grad = -target / out_clipped
    return loss, grad


LOSS_FUNCTIONS = {"mse": mse_loss, "cross_entropy": cross_entropy_loss}


# ════════════════════════════════════════════════════════════════════════
# 2) NeuralNetwork — شبكة متعددة الطبقات (MLP عام)
# ════════════════════════════════════════════════════════════════════════

class NeuralNetwork:
    """
    شبكة عصبية متعددة الطبقات بـ backpropagation صحيح رياضياً عبر كل الطبقات.

    مثال:
        net = NeuralNetwork(layer_dims=[7, 64, 32, 4],
                             activations=["relu", "relu", "softmax"],
                             learning_rate=0.01, loss="mse")
        out  = net.forward(x)
        loss = net.train_step(x, target)
    """

    def __init__(self, layer_dims: List[int], activations: List[str],
                 learning_rate: float = 0.01, loss: str = "mse",
                 optimizer: str = "adam", name: str = "neural_network",
                 seed: Optional[int] = None):
        if len(activations) != len(layer_dims) - 1:
            raise ValueError(
                f"عدد دوال التفعيل ({len(activations)}) يجب أن يساوي "
                f"عدد الطبقات - 1 ({len(layer_dims) - 1})"
            )
        if loss not in LOSS_FUNCTIONS:
            raise ValueError(f"دالة خسارة غير معروفة: {loss}")

        self.name = name
        self.learning_rate = learning_rate
        self.loss_name = loss
        self.optimizer = optimizer
        self.layer_dims = list(layer_dims)

        self.layers: List[DenseLayer] = []
        for i, act in enumerate(activations):
            layer_seed = None if seed is None else seed + i
            self.layers.append(
                DenseLayer(layer_dims[i + 1], layer_dims[i], act,
                           name=f"L{i+1}_{layer_dims[i+1]}x{layer_dims[i]}_{act}",
                           seed=layer_seed)
            )

        self.input_dim = layer_dims[0]
        self.output_dim = layer_dims[-1]

        # ── سجل التدريب ──
        self._train_steps = 0
        self._last_loss: Optional[float] = None
        self._loss_history: deque = deque(maxlen=2000)

        logger.info(
            f"NeuralNetwork '{self.name}' initialised — "
            f"{' → '.join(str(d) for d in layer_dims)} | "
            f"activations={activations} | loss={loss} | lr={learning_rate} | "
            f"params={self.param_count()}"
        )

    # ── Forward ──────────────────────────────────────────────────────

    def _prep_input(self, x: ArrayLike) -> np.ndarray:
        arr = np.array(x, dtype=np.float64)
        if arr.shape[0] < self.input_dim:
            arr = np.pad(arr, (0, self.input_dim - arr.shape[0]))
        elif arr.shape[0] > self.input_dim:
            arr = arr[: self.input_dim]
        return arr

    def forward(self, x: ArrayLike) -> np.ndarray:
        h = self._prep_input(x)
        for layer in self.layers:
            h = layer.forward(h)
        return h

    # ── Train step (backprop كامل وصحيح) ──────────────────────────────

    def train_step(self, x: ArrayLike, target: ArrayLike) -> float:
        """
        خطوة تدريب واحدة:
          1. forward كامل
          2. حساب الخسارة + dL/dout
          3. backward عبر كل الطبقات بالعكس (chain rule كامل)
          4. تحديث كل طبقة (Adam أو SGD)

        Returns: قيمة الخسارة قبل التحديث.
        """
        out = self.forward(x)

        target_arr = np.array(target, dtype=np.float64)
        if target_arr.shape != out.shape:
            target_arr = np.broadcast_to(target_arr, out.shape).astype(np.float64) \
                if target_arr.size == 1 else target_arr.reshape(out.shape)

        loss_fn = LOSS_FUNCTIONS[self.loss_name]
        loss, d_out = loss_fn(out, target_arr)

        # backward chain — من الطبقة الأخيرة إلى الأولى
        grad = d_out
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        # تحديث كل الطبقات
        for layer in self.layers:
            layer.apply_gradients(self.learning_rate, optimizer=self.optimizer)

        self._train_steps += 1
        self._last_loss = loss
        self._loss_history.append(loss)
        return loss

    def train_batch(self, samples: List[Tuple[ArrayLike, ArrayLike]]) -> float:
        """تدريب على دفعة (متوسط الخسارة)."""
        if not samples:
            return 0.0
        total = sum(self.train_step(x, t) for x, t in samples)
        return total / len(samples)

    # ── نمو صريح للشبكة ─────────────────────────────────────────────

    def grow_layer(self, layer_index: int, add_units: int, seed: Optional[int] = None) -> None:
        """
        يزيد عدد نيورونات طبقة `layer_index` بمقدار `add_units`،
        ويزامن in_dim للطبقة التالية تلقائياً (إن وُجدت).
        لا يُستدعى أثناء train_step — خطوة صريحة بين الدفعات.
        """
        if not (0 <= layer_index < len(self.layers)):
            raise ValueError("layer_index خارج المدى")
        layer = self.layers[layer_index]
        layer.grow_out(add_units, seed=seed)
        self.layer_dims[layer_index + 1] += add_units
        layer.name = f"L{layer_index+1}_{layer.out_dim}x{layer.in_dim}_{layer.activation}"

        if layer_index + 1 < len(self.layers):
            next_layer = self.layers[layer_index + 1]
            next_layer.grow_in(add_units, seed=seed)
            next_layer.name = f"L{layer_index+2}_{next_layer.out_dim}x{next_layer.in_dim}_{next_layer.activation}"

        logger.info(
            f"NeuralNetwork '{self.name}': نمو الطبقة {layer_index} "
            f"(+{add_units} وحدة) → shape={layer.W.shape} | "
            f"params={self.param_count()}"
        )

    # ── حفظ/تحميل ────────────────────────────────────────────────────

    def save(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "layer_dims": self.layer_dims,
            "learning_rate": self.learning_rate,
            "loss_name": self.loss_name,
            "optimizer": self.optimizer,
            "train_steps": self._train_steps,
            "last_loss": self._last_loss,
            "layers": [l.to_dict() for l in self.layers],
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info(f"NeuralNetwork '{self.name}' saved → {p.resolve()}")
        return str(p.resolve())

    @classmethod
    def load(cls, path: str) -> "NeuralNetwork":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        activations = [l["activation"] for l in data["layers"]]
        net = cls(data["layer_dims"], activations,
                  learning_rate=data["learning_rate"], loss=data["loss_name"],
                  optimizer=data.get("optimizer", "adam"), name=data["name"])
        net.layers = [DenseLayer.from_dict(l) for l in data["layers"]]
        net._train_steps = data.get("train_steps", 0)
        net._last_loss = data.get("last_loss")
        return net

    # ── معلومات ──────────────────────────────────────────────────────

    def param_count(self) -> int:
        return sum(l.W.size + l.b.size for l in self.layers)

    def architecture_str(self) -> str:
        parts = [f"Input({self.input_dim})"]
        for layer in self.layers:
            parts.append(f"{layer.name}")
        return " → ".join(parts)

    def summary(self) -> dict:
        recent = list(self._loss_history)[-50:]
        return {
            "name": self.name,
            "architecture": self.architecture_str(),
            "layer_dims": self.layer_dims,
            "loss_function": self.loss_name,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "train_steps": self._train_steps,
            "last_loss": round(self._last_loss, 8) if self._last_loss is not None else None,
            "avg_recent_loss": round(sum(recent) / len(recent), 8) if recent else None,
            "total_parameters": self.param_count(),
            "layers": [l.summary() for l in self.layers],
        }

    def __repr__(self) -> str:
        return (f"<NeuralNetwork '{self.name}' {self.architecture_str()} "
                f"params={self.param_count()} steps={self._train_steps}>")


# ════════════════════════════════════════════════════════════════════════
# 3) AssociativeMemory — ذاكرة عصبية ترابطية (Embedding + Retrieval / RAG)
# ════════════════════════════════════════════════════════════════════════

class AssociativeMemory:
    """
    ذاكرة عصبية: تخزّن (متجه embedding، بيانات وصفية) وتسترجع أقرب
    العناصر لأي متجه استعلام بالتشابه الجيب-تمامي (cosine similarity).

    هذا هو رابط "الذاكرة العصبية بالمعرفة": أي معرفة جديدة (نص/مفهوم)
    تُحوَّل إلى متجه عبر NeuralNetwork (طبقة تمثيل/encoder) أو عبر
    VectorEncoder الموجود مسبقاً، وتُخزَّن هنا. عند الاستعلام بمتجه
    سياق حالي، تُسترجع أقرب N ذكريات — أسلوب RAG.

    التخزين قابل للحفظ/التحميل كـ JSON (متجهات + بيانات وصفية).
    """

    def __init__(self, dim: int, capacity: int = 5000, name: str = "associative_memory"):
        self.dim = int(dim)
        self.capacity = int(capacity)
        self.name = name

        self._vectors: np.ndarray = np.zeros((0, self.dim), dtype=np.float64)
        self._meta: List[dict] = []

    # ── إضافة ذكرى ───────────────────────────────────────────────────

    def remember(self, vector: ArrayLike, metadata: Optional[dict] = None) -> int:
        """
        يخزّن متجهاً جديداً (يُطبَّع تلقائياً) مع بيانات وصفية اختيارية.
        إن تجاوزت السعة capacity، تُحذف أقدم ذكرى (FIFO).
        Returns: فهرس الذكرى المخزَّنة.
        """
        v = np.array(vector, dtype=np.float64)
        if v.shape[0] != self.dim:
            if v.shape[0] < self.dim:
                v = np.pad(v, (0, self.dim - v.shape[0]))
            else:
                v = v[: self.dim]

        norm = np.linalg.norm(v)
        v_norm = v / norm if norm > 0 else v

        meta = dict(metadata or {})
        meta.setdefault("stored_at", datetime.now(timezone.utc).isoformat())
        meta.setdefault("raw_vector", v.tolist())

        if self._vectors.shape[0] >= self.capacity:
            self._vectors = self._vectors[1:]
            self._meta = self._meta[1:]

        self._vectors = np.vstack([self._vectors, v_norm.reshape(1, -1)])
        self._meta.append(meta)
        return len(self._meta) - 1

    # ── استرجاع (RAG) ────────────────────────────────────────────────

    def recall(self, query_vector: ArrayLike, top_k: int = 5,
               min_similarity: float = 0.0) -> List[dict]:
        """
        يسترجع أقرب `top_k` ذكريات لمتجه الاستعلام بالتشابه الجيب-تمامي.

        Returns: قائمة من dict تحتوي:
          { "index": int, "similarity": float, "metadata": dict }
        مرتّبة تنازلياً بالتشابه.
        """
        if self._vectors.shape[0] == 0:
            return []

        q = np.array(query_vector, dtype=np.float64)
        if q.shape[0] != self.dim:
            if q.shape[0] < self.dim:
                q = np.pad(q, (0, self.dim - q.shape[0]))
            else:
                q = q[: self.dim]

        q_norm_val = np.linalg.norm(q)
        q_norm = q / q_norm_val if q_norm_val > 0 else q

        sims = self._vectors @ q_norm  # cosine similarity (كل المتجهات مُطبَّعة)

        order = np.argsort(-sims)
        results = []
        for idx in order[:max(top_k * 3, top_k)]:
            sim = float(sims[idx])
            if sim < min_similarity:
                continue
            results.append({
                "index": int(idx),
                "similarity": round(sim, 6),
                "metadata": self._meta[idx],
            })
            if len(results) >= top_k:
                break
        return results

    # ── حفظ/تحميل ────────────────────────────────────────────────────

    def save(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "dim": self.dim,
            "capacity": self.capacity,
            "vectors": self._vectors.tolist(),
            "meta": self._meta,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return str(p.resolve())

    @classmethod
    def load(cls, path: str) -> "AssociativeMemory":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        mem = cls(data["dim"], data.get("capacity", 5000), data.get("name", "associative_memory"))
        mem._vectors = np.array(data["vectors"], dtype=np.float64)
        mem._meta = data["meta"]
        return mem

    def summary(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "capacity": self.capacity,
            "stored": len(self._meta),
            "usage_pct": round(100.0 * len(self._meta) / self.capacity, 2) if self.capacity else 0.0,
        }

    def __len__(self) -> int:
        return len(self._meta)

    def __repr__(self) -> str:
        return f"<AssociativeMemory '{self.name}' dim={self.dim} stored={len(self._meta)}/{self.capacity}>"


# ════════════════════════════════════════════════════════════════════════
# 4) NeuralCore — "One Neural Core": يتعلم / يتذكر / يتطور
# ════════════════════════════════════════════════════════════════════════

DEFAULT_INPUT_DIM = 7          # ثابت — يطابق VectorEncoder في knowledge_trainer
DEFAULT_HIDDEN_DIMS = [64, 32]
DEFAULT_OUTPUT_DIM = 4         # W_SEMANTIC, W_SCORE, W_MEMORY, W_TOPOLOGY


class NeuralCore:
    """
    النواة العصبية الموحَّدة الواحدة.

    تجمع:
      - net    : NeuralNetwork  — التعلّم (forward + backprop صحيح)
      - memory : AssociativeMemory — التذكّر (تخزين/استرجاع بالتشابه)
      - evolve : نمو الشبكة عند ركود التعلّم (plateau) — صريح ومُسجَّل

    "يتعلم"   → train_step() / train_batch() عبر backprop صحيح رياضياً.
    "يتذكر"   → remember() / recall(): تخزين متجه + ربطه باستعلام السياق.
    "يتطور"   → evolve_if_plateau(): فحص ركود الخسارة ونمو الشبكة عند الحاجة.
    "ربط الذاكرة بالمعرفة" → train_and_remember(): كل خطوة تدريب تُخزَّن
        أيضاً كذكرى (متجه الإدخال + الخرج + البيانات الوصفية)، فيصبح
        أي استدعاء recall() قادراً على استرجاع أمثلة تدريب سابقة ذات
        صلة بالسياق الحالي (RAG على ذاكرة التدريب نفسها).

    Parameters
    ----------
    input_dim, hidden_dims, output_dim : أبعاد الشبكة (MLP)
    activations : دوال تفعيل الطبقات المخفية + الإخراج (طول = عدد الطبقات)
    loss : "mse" أو "cross_entropy"
    learning_rate : معدل التعلّم
    optimizer : "sgd" أو "adam"
    memory_capacity : أقصى عدد ذكريات
    plateau_window, plateau_threshold, plateau_cooldown, grow_units, max_hidden_width:
        إعدادات التطوّر (evolution / growth)
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        activations: Optional[List[str]] = None,
        loss: str = "mse",
        learning_rate: float = 0.01,
        optimizer: str = "adam",
        memory_capacity: int = 5000,
        plateau_window: int = 50,
        plateau_threshold: float = 0.01,
        plateau_cooldown: int = 200,
        grow_units: int = 8,
        max_hidden_width: int = 256,
        name: str = "neural_core",
        seed: Optional[int] = None,
    ):
        hidden_dims = hidden_dims if hidden_dims is not None else list(DEFAULT_HIDDEN_DIMS)
        layer_dims = [input_dim] + hidden_dims + [output_dim]

        if activations is None:
            activations = ["relu"] * len(hidden_dims) + ["softmax"]

        self.name = name
        self.net = NeuralNetwork(layer_dims, activations, learning_rate=learning_rate,
                                  loss=loss, optimizer=optimizer, name=f"{name}_net", seed=seed)
        self.memory = AssociativeMemory(dim=input_dim, capacity=memory_capacity,
                                         name=f"{name}_memory")

        # ── إعدادات التطوّر ──
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.plateau_cooldown = plateau_cooldown
        self.grow_units = grow_units
        self.max_hidden_width = max_hidden_width
        self._steps_since_growth = 0
        self._growth_events: List[dict] = []

        logger.info(f"NeuralCore '{self.name}' initialised — {self.net.architecture_str()}")

    # ── واجهة التعلّم ────────────────────────────────────────────────

    def forward(self, x: ArrayLike) -> np.ndarray:
        return self.net.forward(x)

    def train_step(self, x: ArrayLike, target: ArrayLike) -> float:
        """خطوة تدريب واحدة (backprop صحيح كامل) + فحص تطوّر اختياري."""
        loss = self.net.train_step(x, target)
        self._steps_since_growth += 1
        return loss

    def train_batch(self, samples: List[Tuple[ArrayLike, ArrayLike]]) -> float:
        return self.net.train_batch(samples)

    # ── واجهة الذاكرة (يتذكر) ────────────────────────────────────────

    def remember(self, vector: ArrayLike, metadata: Optional[dict] = None) -> int:
        """يخزّن متجهاً في الذاكرة الترابطية مع بيانات وصفية."""
        return self.memory.remember(vector, metadata)

    def recall(self, query_vector: ArrayLike, top_k: int = 5,
               min_similarity: float = 0.0) -> List[dict]:
        """يسترجع أقرب الذكريات لمتجه سياق حالي (RAG)."""
        return self.memory.recall(query_vector, top_k=top_k, min_similarity=min_similarity)

    # ── ربط الذاكرة بالمعرفة: تدريب + تذكّر في خطوة واحدة ────────────

    def train_and_remember(self, x: ArrayLike, target: ArrayLike,
                            metadata: Optional[dict] = None) -> Dict[str, object]:
        """
        خطوة موحّدة:
          1. train_step(x, target) — تعلّم حقيقي عبر backprop.
          2. forward(x) بعد التحديث — للحصول على الخرج الجديد.
          3. remember(x, metadata + output) — ربط متجه الإدخال (المعرفة)
             بخرج الشبكة الحالي في الذاكرة الترابطية، فيصبح قابلاً
             للاسترجاع عبر recall() لاحقاً بالاستعلام بمتجهات سياق مشابهة.

        Returns: { "loss": float, "output": list, "memory_index": int }
        """
        loss = self.train_step(x, target)
        output = self.forward(x)

        meta = dict(metadata or {})
        meta["output"] = output.tolist()
        meta["loss"] = loss
        meta["train_step"] = self.net._train_steps

        idx = self.remember(x, meta)

        # فحص ركود التعلّم وتطوّر الشبكة عند الحاجة (بين الخطوات، لا يكسر backprop)
        grew = self.evolve_if_plateau()

        return {"loss": loss, "output": output.tolist(), "memory_index": idx, "grew": grew}

    # ── التطوّر (يتطور) ──────────────────────────────────────────────

    def _is_plateauing(self) -> bool:
        hist = list(self.net._loss_history)
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
        يفحص ركود الخسارة (نافذة `plateau_window`، بعد `plateau_cooldown`
        خطوة من آخر نمو). إذا كان التحسّن أقل من `plateau_threshold`،
        تنمو أول طبقة مخفية بمقدار `grow_units` (حتى `max_hidden_width`).

        النمو يحدث *بين* خطوات التدريب فقط (لا يُستدعى داخل backward)،
        لذا لا يكسر صحة التدرجات لأي خطوة سابقة أو لاحقة.

        Returns: True إذا حدث نمو.
        """
        if self._steps_since_growth < self.plateau_cooldown:
            return False
        if not self._is_plateauing():
            return False
        if len(self.net.layers) < 2:
            return False

        # ابحث عن أول طبقة مخفية (غير الأخيرة) لم تصل الحد الأقصى
        for i, layer in enumerate(self.net.layers[:-1]):
            if layer.out_dim < self.max_hidden_width:
                add = min(self.grow_units, self.max_hidden_width - layer.out_dim)
                old_dim = layer.out_dim
                self.net.grow_layer(i, add)
                self._steps_since_growth = 0
                event = {
                    "step": self.net._train_steps,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "layer_index": i,
                    "old_units": old_dim,
                    "new_units": layer.out_dim,
                    "loss_at_growth": self.net._last_loss,
                }
                self._growth_events.append(event)
                logger.info(f"NeuralCore '{self.name}' EVOLVED: layer {i} "
                            f"{old_dim}→{layer.out_dim} units (step {self.net._train_steps})")
                return True
        return False

    # ── حفظ/تحميل النواة بالكامل ──────────────────────────────────────

    def save(self, directory: str) -> str:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.net.save(str(d / "network.json"))
        self.memory.save(str(d / "memory.json"))
        state = {
            "name": self.name,
            "plateau_window": self.plateau_window,
            "plateau_threshold": self.plateau_threshold,
            "plateau_cooldown": self.plateau_cooldown,
            "grow_units": self.grow_units,
            "max_hidden_width": self.max_hidden_width,
            "steps_since_growth": self._steps_since_growth,
            "growth_events": self._growth_events,
        }
        with open(d / "core_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        logger.info(f"NeuralCore '{self.name}' saved → {d.resolve()}")
        return str(d.resolve())

    @classmethod
    def load(cls, directory: str) -> "NeuralCore":
        d = Path(directory)
        core = cls.__new__(cls)
        core.net = NeuralNetwork.load(str(d / "network.json"))
        core.memory = AssociativeMemory.load(str(d / "memory.json"))

        state_path = d / "core_state.json"
        if state_path.exists():
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
            core.name = state.get("name", "neural_core")
            core.plateau_window = state.get("plateau_window", 50)
            core.plateau_threshold = state.get("plateau_threshold", 0.01)
            core.plateau_cooldown = state.get("plateau_cooldown", 200)
            core.grow_units = state.get("grow_units", 8)
            core.max_hidden_width = state.get("max_hidden_width", 256)
            core._steps_since_growth = state.get("steps_since_growth", 0)
            core._growth_events = state.get("growth_events", [])
        else:
            core.name = "neural_core"
            core.plateau_window = 50
            core.plateau_threshold = 0.01
            core.plateau_cooldown = 200
            core.grow_units = 8
            core.max_hidden_width = 256
            core._steps_since_growth = 0
            core._growth_events = []

        logger.info(f"NeuralCore '{core.name}' loaded ← {d.resolve()}")
        return core

    # ── معلومات ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "name": self.name,
            "network": self.net.summary(),
            "memory": self.memory.summary(),
            "evolution": {
                "plateau_window": self.plateau_window,
                "plateau_threshold": self.plateau_threshold,
                "plateau_cooldown": self.plateau_cooldown,
                "grow_units": self.grow_units,
                "max_hidden_width": self.max_hidden_width,
                "steps_since_growth": self._steps_since_growth,
                "growth_events_count": len(self._growth_events),
                "growth_events": self._growth_events[-5:],
            },
        }

    def __repr__(self) -> str:
        return (f"<NeuralCore '{self.name}' net={self.net.architecture_str()} "
                f"memory={len(self.memory)}/{self.memory.capacity}>")


# ════════════════════════════════════════════════════════════════════════
# 5) طبقات توافق خلفي (Backward-compatible shims)
#    نفس الواجهة القديمة (forward/train_step/.weights/.SHAPE/summary)
#    لكن backprop ومشتقات صحيحة رياضياً الآن.
# ════════════════════════════════════════════════════════════════════════

class NeuralWeightLayer:
    """
    توافق خلفي مع ai.neural_weights.NeuralWeightLayer القديمة.

    الفرق عن النسخة القديمة:
      - forward: نفس الصيغة output = ReLU(W @ x + bias) لكن bias الآن
        متجه (out_dim,) بدلاً من سكالار واحد — يُهيَّأ كله بنفس القيمة
        الافتراضية 0.6 للحفاظ على التوافق العددي عند bias=0.6 الافتراضي.
      - train_step: backprop صحيح — التدرج يُضرب بمشتقة ReLU، وbias يُحدَّث.
    """

    COLS = 7

    def __init__(self, initial_weights: Optional[ArrayLike] = None,
                 bias: float = 0.6, learning_rate: float = 0.01,
                 name: str = "routing_weight_layer"):
        if initial_weights is not None:
            w = np.array(initial_weights, dtype=np.float64)
            if w.ndim != 2 or w.shape[1] != self.COLS:
                raise ValueError(f"شكل الأوزان يجب أن يكون (N, {self.COLS}), حصلت على {w.shape}")
            rows = w.shape[0]
        else:
            rows = 108
            w = None

        self._layer = DenseLayer(rows, self.COLS, "relu", name=name)
        if w is not None:
            self._layer.W = w.copy()
        self._layer.b = np.full(rows, float(bias), dtype=np.float64)

        self.bias = float(bias)
        self.learning_rate = learning_rate
        self.name = name
        self._train_steps = 0
        self._last_loss: Optional[float] = None

    @property
    def weights(self) -> np.ndarray:
        return self._layer.W

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        self._layer.W = np.array(value, dtype=np.float64)

    @property
    def SHAPE(self):
        return self._layer.W.shape

    def forward(self, x: ArrayLike) -> np.ndarray:
        x_arr = np.array(x, dtype=np.float64)
        if x_arr.shape != (self.COLS,):
            raise ValueError(f"forward() يتوقع طول {self.COLS}، حصل على {x_arr.shape}")
        return self._layer.forward(x_arr)

    def train_step(self, input_vector: ArrayLike, target: float) -> float:
        """خطوة تدريب صحيحة: MSE + backprop مع مشتقة ReLU، تحديث W و b."""
        x = np.array(input_vector, dtype=np.float64)
        out = self._layer.forward(x)
        target_vec = np.full(self._layer.out_dim, float(target), dtype=np.float64)

        loss, d_out = mse_loss(out, target_vec)
        self._layer.backward(d_out)
        self._layer.apply_gradients(self.learning_rate, optimizer="sgd")

        self._train_steps += 1
        self._last_loss = loss
        return loss

    def update(self, delta: Union[float, np.ndarray]) -> None:
        """تحديث إضافي مباشر (متوافق مع API القديم)."""
        d = np.array(delta, dtype=np.float64)
        if d.ndim == 0:
            self._layer.W += float(d) * self.learning_rate
        elif d.shape == self._layer.W.shape:
            self._layer.W += d * self.learning_rate
        else:
            raise ValueError(f"شكل delta {d.shape} غير متوافق مع {self._layer.W.shape}")
        np.clip(self._layer.W, -5.0, 5.0, out=self._layer.W)

    def save(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), self._layer.W)
        return str(p.resolve())

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        loaded = np.load(str(p))
        if loaded.ndim != 2 or loaded.shape[1] != self.COLS:
            raise ValueError(f"شكل غير متوافق: {loaded.shape}")
        self._layer.W = loaded.astype(np.float64)
        self._layer.out_dim = loaded.shape[0]
        self._layer.b = np.full(loaded.shape[0], self.bias, dtype=np.float64)

    def get_weights_list(self) -> List[List[float]]:
        return self._layer.W.tolist()

    def summary(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self._layer.W.shape),
            "bias": self.bias,
            "learning_rate": self.learning_rate,
            "train_steps": self._train_steps,
            "last_loss": round(self._last_loss, 8) if self._last_loss is not None else None,
            "weight_stats": self._layer.summary()["weight_stats"],
        }

    def __repr__(self) -> str:
        return f"<NeuralWeightLayer(fixed) name='{self.name}' shape={self._layer.W.shape} steps={self._train_steps}>"


class DynamicWeightLayer:
    """
    توافق خلفي مع ai.dynamic_weight_layer.DynamicWeightLayer القديمة.

    الفرق عن النسخة القديمة:
      - forward: نفس output = ReLU(W @ x) لكن الآن مع bias=0 صريح
        (النسخة القديمة لم تكن تملك bias أساساً، فهذا متوافق تماماً).
      - train_step: backprop صحيح (تدرج × مشتقة ReLU)، النمو (+GROW_ROWS)
        يبقى كما هو لكن مُسجَّل بنفس الآلية مع توافق الأبعاد.
    """

    INITIAL_ROWS = 9
    INITIAL_COLS = 7
    GROW_ROWS = 23
    MAX_ROWS = 200
    PLATEAU_CHECK_STEPS = 20
    PLATEAU_THRESHOLD = 0.01
    COOLDOWN_STEPS = 120

    def __init__(self, initial_rows: int = INITIAL_ROWS, initial_cols: int = INITIAL_COLS,
                 learning_rate: float = 0.01, name: str = "dynamic_weight_layer"):
        if initial_cols != self.INITIAL_COLS:
            initial_cols = self.INITIAL_COLS

        self._layer = DenseLayer(initial_rows, initial_cols, "relu", name=name)
        self._layer.b = np.zeros(initial_rows, dtype=np.float64)  # لا bias (مطابق للقديم)

        self.learning_rate = learning_rate
        self.name = name
        self._train_steps = 0
        self._last_loss: Optional[float] = None
        self._loss_history: deque = deque(maxlen=500)
        self._growth_events: List[dict] = []
        self._steps_since_growth = 0

    @property
    def weights(self) -> np.ndarray:
        return self._layer.W

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        self._layer.W = np.array(value, dtype=np.float64)

    @property
    def shape(self) -> Tuple[int, int]:
        return self._layer.W.shape

    @property
    def SHAPE(self) -> Tuple[int, int]:
        return self.shape

    @property
    def _rows(self) -> int:
        return self._layer.out_dim

    @property
    def _cols(self) -> int:
        return self._layer.in_dim

    def forward(self, x: ArrayLike) -> np.ndarray:
        x_arr = np.array(x, dtype=np.float64)
        if x_arr.shape[0] < self._cols:
            x_arr = np.pad(x_arr, (0, self._cols - x_arr.shape[0]))
        elif x_arr.shape[0] > self._cols:
            x_arr = x_arr[: self._cols]
        return self._layer.forward(x_arr)

    def train_step(self, input_vector: ArrayLike, target: float) -> float:
        x = np.array(input_vector, dtype=np.float64)
        if x.shape[0] < self._cols:
            x = np.pad(x, (0, self._cols - x.shape[0]))
        elif x.shape[0] > self._cols:
            x = x[: self._cols]

        out = self._layer.forward(x)
        target_vec = np.full(self._rows, float(target), dtype=np.float64)

        loss, d_out = mse_loss(out, target_vec)
        self._layer.backward(d_out)
        self._layer.apply_gradients(self.learning_rate, optimizer="sgd")

        self._train_steps += 1
        self._last_loss = loss
        self._loss_history.append(loss)
        self._steps_since_growth += 1

        if (self._train_steps % self.PLATEAU_CHECK_STEPS == 0 and
                self._steps_since_growth >= self.COOLDOWN_STEPS):
            if self._is_plateauing():
                self._grow()

        return loss

    def _is_plateauing(self) -> bool:
        if len(self._loss_history) < self.PLATEAU_CHECK_STEPS * 2:
            return False
        window = list(self._loss_history)[-self.PLATEAU_CHECK_STEPS * 2:]
        mid = len(window) // 2
        mean_older = float(np.mean(window[:mid]))
        mean_recent = float(np.mean(window[mid:]))
        if mean_older == 0.0:
            return False
        return (mean_older - mean_recent) / mean_older < self.PLATEAU_THRESHOLD

    def _grow(self) -> None:
        old_rows = self._rows
        new_rows = min(old_rows + self.GROW_ROWS, self.MAX_ROWS)
        if new_rows == old_rows:
            return
        added = new_rows - old_rows
        self._layer.grow_out(added)
        self._steps_since_growth = 0
        self._growth_events.append({
            "step": self._train_steps,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_shape": [old_rows, self._cols],
            "new_shape": [new_rows, self._cols],
            "rows_added": added,
            "trigger": "plateau",
            "loss_at_growth": round(self._last_loss, 8) if self._last_loss else None,
        })

    def get_routing_row(self) -> np.ndarray:
        return self._layer.W[0, :min(4, self._cols)]

    def save(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), self._layer.W)
        meta = np.array([self._rows, self._cols, self._train_steps], dtype=np.int64)
        np.save(str(p).replace(".npy", "_meta.npy"), meta)
        return str(p.resolve())

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        loaded = np.load(str(p))
        if loaded.shape[1] != self.INITIAL_COLS:
            return
        self._layer.W = loaded.astype(np.float64)
        self._layer.out_dim, self._layer.in_dim = loaded.shape
        self._layer.b = np.zeros(loaded.shape[0], dtype=np.float64)
        meta_path = str(p).replace(".npy", "_meta.npy")
        if os.path.exists(meta_path):
            try:
                meta = np.load(meta_path)
                self._train_steps = int(meta[2])
            except Exception:
                pass

    def get_weights_list(self) -> List[List[float]]:
        return self._layer.W.tolist()

    def summary(self) -> dict:
        trajectory = []
        r = self._rows
        for _ in range(5):
            r_next = min(r + self.GROW_ROWS, self.MAX_ROWS)
            trajectory.append(f"{r_next}×{self._cols}")
            if r_next >= self.MAX_ROWS:
                break
            r = r_next
        return {
            "name": self.name,
            "shape": [self._rows, self._cols],
            "cols_fixed": True,
            "grow_rows": self.GROW_ROWS,
            "max_rows": self.MAX_ROWS,
            "learning_rate": self.learning_rate,
            "train_steps": self._train_steps,
            "last_loss": round(self._last_loss, 8) if self._last_loss is not None else None,
            "growth_events_count": len(self._growth_events),
            "growth_events": self._growth_events[-5:],
            "next_growth_trajectory": trajectory,
            "weight_stats": self._layer.summary()["weight_stats"],
            "plateau_check": {
                "window_steps": self.PLATEAU_CHECK_STEPS,
                "threshold": self.PLATEAU_THRESHOLD,
                "cooldown": self.COOLDOWN_STEPS,
                "steps_since_growth": self._steps_since_growth,
            },
            "max_size": f"{self.MAX_ROWS}×{self.INITIAL_COLS}",
        }

    def __repr__(self) -> str:
        return (f"<DynamicWeightLayer(fixed) name='{self.name}' "
                f"shape=({self._rows}×{self._cols}) steps={self._train_steps} "
                f"growths={len(self._growth_events)}>")


# ── دوال استخراج أوزان التوجيه (متوافقة مع الواجهة القديمة) ──────────────

def extract_routing_weights(layer: Union[NeuralWeightLayer, DynamicWeightLayer, "NeuralCore"]) -> dict:
    """يستخرج 4 أوزان توجيه (تجمع=1) من الصف الأول لمصفوفة الأوزان."""
    w = layer.weights if hasattr(layer, "weights") else layer.net.layers[0].W
    row0 = np.asarray(w)[0, :4]
    total = float(row0.sum())
    if total <= 0.0:
        return {"W_SEMANTIC": 0.30, "W_SCORE": 0.35, "W_MEMORY": 0.25, "W_TOPOLOGY": 0.10}
    normed = (row0 / total).tolist()
    return {
        "W_SEMANTIC": round(normed[0], 6),
        "W_SCORE": round(normed[1], 6),
        "W_MEMORY": round(normed[2], 6),
        "W_TOPOLOGY": round(normed[3], 6),
    }


# ════════════════════════════════════════════════════════════════════════
# Singleton الافتراضي
# ════════════════════════════════════════════════════════════════════════

_default_core: Optional[NeuralCore] = None


def get_default_core(directory: str = "models/neural_core") -> NeuralCore:
    """يُعيد (ويُخزِّن) نسخة NeuralCore الافتراضية الوحيدة."""
    global _default_core
    if _default_core is None:
        if os.path.exists(os.path.join(directory, "network.json")):
            try:
                _default_core = NeuralCore.load(directory)
                logger.info(f"NeuralCore restored from {directory}")
            except Exception as e:
                logger.warning(f"Could not load NeuralCore from {directory}: {e}")
        if _default_core is None:
            _default_core = NeuralCore(name="default_neural_core")
    return _default_core
