"""
Phase 10 — Multimodal Unified Network (v1)
============================================
شبكة موحّدة تدمج 3 وسائط:
    نص (CKG)  128   ─┐
    صورة آية  784   ─┼─ concat(1040) ─→ دمج خطي ─→ تمثيل موحّد (784) ─→ شبكة نامية ذاتيًا
    صوت تلاوة 128   ─┘                                                       ↓
                                                            ┌───────────────┴───────────────┐
                                                        رأس Agent (قرار/فعل)        رأس تحليل البيانات

ملاحظات تصميم مهمة:
- هذا ملف **مستقل تمامًا** ولا يلمس ai/deep_routing_network.py (يبقى كما هو، يُستخدم في
  4 ملفات أخرى من مشروعك: main.py, routing_engine.py, knowledge_trainer.py, signal_stream.py).
- التمثيل الموحّد (784) يمكن إعادة استخدامه كمدخل إضافي لشبكة التوجيه القديمة عبر
  to_routing_vector() أدناه (تُرجع 128 بُعد متوافقة، دون تعديل تلك الشبكة).
- "الدردشة" الفعلية (توليد نص عربي طبيعي) لا تتم هنا — هذه الشبكة تُخرج: تمثيل/قرار/تصنيف فقط.
  الصياغة النهائية للرد يجب أن تتم عبر LLM API أو قوالب (انظر chat_respond في آخر الملف).
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── أبعاد ثابتة ────────────────────────────────────────────────────────────
IMG_SIDE      = 28
IMG_DIM       = IMG_SIDE * IMG_SIDE      # 784
TEXT_DIM      = 128
AUDIO_DIM     = 128
FUSED_DIM     = 784                       # ← الطبقة الموحّدة المطلوبة: صف واحد × 784 عمود
CONCAT_DIM    = TEXT_DIM + IMG_DIM + AUDIO_DIM   # 1040

HIDDEN_INITIAL_ROWS = 392
HIDDEN_GROW_BY       = 16
HIDDEN_MAX_ROWS      = None

AGENT_ACTIONS_DEFAULT   = 8
ANALYSIS_OUT_DEFAULT    = 4

PLATEAU_WINDOW    = 50
PLATEAU_THRESHOLD = 0.01
PLATEAU_COOLDOWN  = 200

WEIGHTS_DIR = "models/classifiers/multimodal"


# ── دوال مساعدة (مستقلة، لا استيراد خارجي) ──────────────────────────────────
def _he_init(rows: int, cols: int) -> np.ndarray:
    std = math.sqrt(2.0 / max(cols, 1))
    return np.random.normal(0.0, std, size=(rows, cols)).astype(np.float64)


def _xavier_init(rows: int, cols: int) -> np.ndarray:
    limit = math.sqrt(6.0 / (rows + cols))
    return np.random.uniform(-limit, limit, size=(rows, cols)).astype(np.float64)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_deriv(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max()
    exp_x = np.exp(np.clip(shifted, -500, 500))
    total = exp_x.sum()
    return np.ones(len(x)) / len(x) if total == 0 else exp_x / total


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class _Dense:
    """طبقة Dense عامة (نفس فكرة DenseLayer في deep_routing_network.py لكن مستقلة)."""

    def __init__(self, out_dim: int, in_dim: int, activation: str = "relu", name: str = ""):
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.activation = activation
        self.name = name or f"dense_{out_dim}x{in_dim}_{activation}"
        self.weights = _he_init(out_dim, in_dim) if activation == "relu" else _xavier_init(out_dim, in_dim)
        self.biases = np.zeros(out_dim, dtype=np.float64)
        self._last_input = None
        self._last_pre = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_input = x.copy()
        pre = self.weights @ x + self.biases
        self._last_pre = pre.copy()
        if self.activation == "relu":
            return _relu(pre)
        if self.activation == "softmax":
            return _softmax(pre)
        if self.activation == "sigmoid":
            return _sigmoid(pre)
        return pre  # linear

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        if self.activation == "relu":
            grad_pre = grad_out * _relu_deriv(self._last_pre)
        elif self.activation == "sigmoid":
            s = _sigmoid(self._last_pre)
            grad_pre = grad_out * s * (1 - s)
        else:
            grad_pre = grad_out
        grad_w = np.outer(grad_pre, self._last_input)
        grad_x = self.weights.T @ grad_pre
        self.weights -= lr * grad_w
        self.biases -= lr * grad_pre
        self.weights = np.clip(self.weights, -5.0, 5.0)
        return grad_x

    def save(self, prefix: str) -> None:
        np.save(f"{prefix}_w.npy", self.weights)
        np.save(f"{prefix}_b.npy", self.biases)

    def load(self, prefix: str) -> None:
        self.weights = np.load(f"{prefix}_w.npy").astype(np.float64)
        self.biases = np.load(f"{prefix}_b.npy").astype(np.float64)
        self.out_dim, self.in_dim = self.weights.shape


# ── المُشفّرات (Encoders) ────────────────────────────────────────────────

class TextEncoder:
    """يعيد استخدام منطق CKG الموجود لديك إن وُجد، وإلا نسخة محلية مطابقة."""

    @staticmethod
    def encode(query: str, ckg_concepts: dict, dim: int = TEXT_DIM) -> np.ndarray:
        try:
            from ai.deep_routing_network import encode_query_to_ckg_vector
            return encode_query_to_ckg_vector(query, ckg_concepts, dim=dim)
        except Exception:
            return TextEncoder._encode_local(query, ckg_concepts, dim)

    @staticmethod
    def _encode_local(query: str, ckg_concepts: dict, dim: int) -> np.ndarray:
        import re
        def clean(t):
            t = re.sub(r'[ٱ]', 'ا', t)
            t = re.sub(r'[ًٌٍَُِّْٰ]', '', t)
            t = re.sub(r'[^\u0600-\u06FF\s]', ' ', t)
            return t.strip()
        q_words = set(clean(query).split())
        vec = np.zeros(dim, dtype=np.float64)
        for i, (name, meta) in enumerate(list(ckg_concepts.items())[:dim]):
            name_clean = clean(name)
            strength = meta.get("strength", 0.1)
            freq_norm = min(1.0, meta.get("frequency", 1) / 500)
            match = 1.0 if (name_clean in q_words or
                            any(w in name_clean for w in q_words if len(w) >= 3)) else 0.0
            vec[i] = match * strength * (1.0 + freq_norm)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class ImageEncoder:
    """يرسم نص الآية كصورة (للحالات التي لا تتوفر فيها صورة حقيقية)، أو يقرأ صورة من ملف."""

    @staticmethod
    def render_ayah_to_vector(text: str, font_path: Optional[str] = None,
                               side: int = IMG_SIDE) -> np.ndarray:
        from PIL import Image, ImageDraw, ImageFont
        scale = 8
        img = Image.new("L", (side * scale, side * scale), color=255)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(font_path, int(40 * scale / 8)) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        shaped = text
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            shaped = get_display(arabic_reshaper.reshape(text))
        except Exception:
            pass  # سيُرسم بدون تشكيل اتجاه RTL صحيح — كافٍ للتجربة، يُنصح بتثبيت المكتبتين في Colab
        draw.text((4, 4), shaped, fill=0, font=font)
        img = img.resize((side, side), Image.LANCZOS)
        arr = 1.0 - (np.asarray(img, dtype=np.float64) / 255.0)
        return arr.flatten()

    @staticmethod
    def encode_from_file(path: str, side: int = IMG_SIDE) -> np.ndarray:
        from PIL import Image
        img = Image.open(path).convert("L").resize((side, side))
        return (np.asarray(img, dtype=np.float64) / 255.0).flatten()


class AudioEncoder:
    """MFCC مجمّعة (mean+std) إلى متجه ثابت الحجم. يرجع أصفارًا بأمان عند غياب librosa/الملف."""

    @staticmethod
    def encode_from_file(path: str, dim: int = AUDIO_DIM, n_mfcc: int = 20) -> np.ndarray:
        try:
            import librosa
        except ImportError:
            logger.warning("librosa غير مثبت — شغّل: pip install librosa soundfile")
            return np.zeros(dim, dtype=np.float64)
        try:
            y, sr = librosa.load(path, sr=16000, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            pooled = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
            if pooled.shape[0] < dim:
                pooled = np.pad(pooled, (0, dim - pooled.shape[0]))
            else:
                pooled = pooled[:dim]
            norm = np.linalg.norm(pooled)
            return pooled / norm if norm > 0 else pooled
        except Exception as e:
            logger.warning(f"audio encode failed: {e}")
            return np.zeros(dim, dtype=np.float64)


# ── الشبكة الموحّدة الذاتية النمو ─────────────────────────────────────────

class MultimodalRoutingCore:
    """
    الطبقة الموحّدة المطلوبة: دمج [نص+صورة+صوت] (1040) → تمثيل موحّد (1×784)
    ثم طبقة نامية ذاتيًا (أعمدة ثابتة=784، صفوف تنمو) → رأسا Agent وتحليل البيانات.
    """

    def __init__(
        self,
        agent_actions: int = AGENT_ACTIONS_DEFAULT,
        analysis_outputs: int = ANALYSIS_OUT_DEFAULT,
        learning_rate: float = 0.003,
        name: str = "multimodal_core_v1",
    ):
        self.name = name
        self.learning_rate = learning_rate
        self._train_steps = 0
        self._last_loss: Optional[float] = None
        self._loss_history: List[float] = []
        self._steps_since_growth = 0
        self._growth_events: List[dict] = []

        # طبقة الدمج: 1040 → 784 (التمثيل الموحّد، صف واحد × 784 عمود كما طلبت)
        self.fusion = _Dense(FUSED_DIM, CONCAT_DIM, activation="relu",
                              name=f"Fusion_{FUSED_DIM}x{CONCAT_DIM}")

        # الطبقة النامية ذاتيًا: أعمدة ثابتة = 784، صفوف تبدأ 392 وتنمو +16
        self.hidden = _Dense(HIDDEN_INITIAL_ROWS, FUSED_DIM, activation="relu",
                              name=f"Hidden_{HIDDEN_INITIAL_ROWS}x{FUSED_DIM}")

        # رأسا الإخراج
        self.agent_head = _Dense(agent_actions, HIDDEN_INITIAL_ROWS, activation="softmax",
                                  name=f"AgentHead_{agent_actions}x{HIDDEN_INITIAL_ROWS}")
        self.analysis_head = _Dense(analysis_outputs, HIDDEN_INITIAL_ROWS, activation="sigmoid",
                                     name=f"AnalysisHead_{analysis_outputs}x{HIDDEN_INITIAL_ROWS}")

        logger.info(
            f"MultimodalRoutingCore '{self.name}' — "
            f"[text128+image784+audio128]=1040 → Fusion(784) → Hidden({HIDDEN_INITIAL_ROWS}) "
            f"→ Agent({agent_actions}) / Analysis({analysis_outputs}) | "
            f"params: {self._count_params():,}"
        )

    # ── التقاط متعدد الوسائط ────────────────────────────────────────────
    def encode_multimodal(
        self,
        text: Optional[str] = None,
        ckg_concepts: Optional[dict] = None,
        image_path: Optional[str] = None,
        image_render_text: Optional[str] = None,
        audio_path: Optional[str] = None,
        font_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        text_vec = (TextEncoder.encode(text, ckg_concepts or {}, TEXT_DIM)
                    if text and ckg_concepts else np.zeros(TEXT_DIM))

        if image_path:
            image_vec = ImageEncoder.encode_from_file(image_path)
        elif image_render_text:
            image_vec = ImageEncoder.render_ayah_to_vector(image_render_text, font_path)
        else:
            image_vec = np.zeros(IMG_DIM)

        audio_vec = AudioEncoder.encode_from_file(audio_path) if audio_path else np.zeros(AUDIO_DIM)

        concat = np.concatenate([text_vec, image_vec, audio_vec]).astype(np.float64)
        return {"text": text_vec, "image": image_vec, "audio": audio_vec, "concat": concat}

    # ── التمرير الأمامي ──────────────────────────────────────────────────
    def forward(self, concat_vec: np.ndarray) -> Dict[str, np.ndarray]:
        x = np.asarray(concat_vec, dtype=np.float64).ravel()
        if x.shape[0] != CONCAT_DIM:
            if x.shape[0] < CONCAT_DIM:
                x = np.pad(x, (0, CONCAT_DIM - x.shape[0]))
            else:
                x = x[:CONCAT_DIM]
        fused = self.fusion.forward(x)            # التمثيل الموحّد (784)
        hidden_out = self.hidden.forward(fused)    # الطبقة النامية
        agent_probs = self.agent_head.forward(hidden_out)
        analysis_out = self.analysis_head.forward(hidden_out)
        return {"fused": fused, "hidden": hidden_out,
                "agent_probs": agent_probs, "analysis": analysis_out}

    def to_routing_vector(self, fused_vec: np.ndarray, dim: int = 128) -> np.ndarray:
        """يحوّل التمثيل الموحّد (784) إلى متجه 128 متوافق مع DeepRoutingNetwork القديمة
        (دون تعديل تلك الشبكة أبدًا) — تجميع بسيط بطّي المتوسط."""
        fused_vec = np.asarray(fused_vec, dtype=np.float64).ravel()
        factor = len(fused_vec) // dim
        if factor < 1:
            return np.pad(fused_vec, (0, dim - len(fused_vec)))
        trimmed = fused_vec[:factor * dim].reshape(dim, factor)
        return trimmed.mean(axis=1)

    # ── النمو الذاتي (نفس فكرة grow() في deep_routing_network.py) ───────
    def grow(self) -> bool:
        old_rows = self.hidden.out_dim
        added = HIDDEN_GROW_BY
        new_rows = old_rows + added

        new_w = _he_init(added, self.hidden.in_dim)
        self.hidden.weights = np.vstack([self.hidden.weights, new_w])
        self.hidden.biases = np.concatenate([self.hidden.biases, np.zeros(added)])
        self.hidden.out_dim = new_rows
        self.hidden.name = f"Hidden_{new_rows}x{self.hidden.in_dim}"

        for head in (self.agent_head, self.analysis_head):
            new_cols = _he_init(head.out_dim, added)
            head.weights = np.hstack([head.weights, new_cols])
            head.in_dim = new_rows

        logger.info(f"grow(): Hidden {old_rows}→{new_rows} | params: {self._count_params():,}")
        return True

    def _is_plateauing(self) -> bool:
        hist = self._loss_history
        w = PLATEAU_WINDOW
        if len(hist) < w * 2:
            return False
        window = hist[-w * 2:]
        mean_older = float(np.mean(window[:w]))
        mean_recent = float(np.mean(window[w:]))
        if mean_older == 0.0:
            return False
        return (mean_older - mean_recent) / mean_older < PLATEAU_THRESHOLD

    def evolve_if_plateau(self) -> bool:
        if self._steps_since_growth < PLATEAU_COOLDOWN:
            return False
        if not self._is_plateauing():
            return False
        old_rows = self.hidden.out_dim
        self.grow()
        self._steps_since_growth = 0
        self._growth_events.append({
            "step": self._train_steps, "old_rows": old_rows,
            "new_rows": self.hidden.out_dim, "loss_at_growth": self._last_loss,
        })
        return True

    # ── التدريب (متعدد المهام: agent + analysis) ─────────────────────────
    def train_step(self, concat_vec: np.ndarray,
                    agent_target: Optional[int] = None,
                    analysis_target: Optional[np.ndarray] = None) -> float:
        out = self.forward(concat_vec)
        total_loss = 0.0
        grad_hidden = np.zeros_like(out["hidden"])

        if agent_target is not None:
            target_vec = np.zeros(self.agent_head.out_dim)
            target_vec[agent_target] = 1.0
            error = out["agent_probs"] - target_vec
            total_loss += float(np.mean(error ** 2))
            grad = 2.0 * error / self.agent_head.out_dim
            grad_hidden += self.agent_head.backward(grad, self.learning_rate)

        if analysis_target is not None:
            target_vec = np.asarray(analysis_target, dtype=np.float64).ravel()
            error = out["analysis"] - target_vec
            total_loss += float(np.mean(error ** 2))
            grad = 2.0 * error / self.analysis_head.out_dim
            grad_hidden += self.analysis_head.backward(grad, self.learning_rate)

        grad_fused = self.hidden.backward(grad_hidden, self.learning_rate)
        self.fusion.backward(grad_fused, self.learning_rate)

        self._train_steps += 1
        self._steps_since_growth += 1
        self._last_loss = total_loss
        self._loss_history.append(total_loss)
        if len(self._loss_history) > 1000:
            self._loss_history = self._loss_history[-1000:]
        self.evolve_if_plateau()
        return total_loss

    # ── حفظ/تحميل ─────────────────────────────────────────────────────────
    def save(self, directory: str = WEIGHTS_DIR) -> str:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.fusion.save(str(d / "fusion"))
        self.hidden.save(str(d / "hidden"))
        self.agent_head.save(str(d / "agent_head"))
        self.analysis_head.save(str(d / "analysis_head"))
        np.save(str(d / "state.npy"), np.array(
            [self._train_steps, self._last_loss or 0.0, self.hidden.out_dim], dtype=np.float64))
        return str(d.resolve())

    def load(self, directory: str = WEIGHTS_DIR) -> None:
        d = Path(directory)
        for name, layer in [("fusion", self.fusion), ("hidden", self.hidden),
                             ("agent_head", self.agent_head), ("analysis_head", self.analysis_head)]:
            prefix = str(d / name)
            if os.path.exists(f"{prefix}_w.npy"):
                try:
                    layer.load(prefix)
                except Exception as e:
                    logger.warning(f"{name} load failed: {e}")
        state_path = str(d / "state.npy")
        if os.path.exists(state_path):
            state = np.load(state_path)
            self._train_steps = int(state[0])
            self._last_loss = float(state[1]) if state[1] != 0.0 else None

    def _count_params(self) -> int:
        return sum(l.weights.size + l.biases.size for l in
                   (self.fusion, self.hidden, self.agent_head, self.analysis_head))

    def summary(self) -> dict:
        return {
            "name": self.name,
            "architecture": f"Concat({CONCAT_DIM}) → Fusion({FUSED_DIM}) → "
                             f"Hidden({self.hidden.out_dim}) → Agent({self.agent_head.out_dim})"
                             f"/Analysis({self.analysis_head.out_dim})",
            "total_parameters": self._count_params(),
            "hidden_rows": self.hidden.out_dim,
            "hidden_cols_fixed": FUSED_DIM,
            "train_steps": self._train_steps,
            "last_loss": round(self._last_loss, 8) if self._last_loss else None,
            "growth_events": len(self._growth_events),
        }


# ── طبقة الدردشة (فهم + قرار هنا، الصياغة النهائية عبر LLM/قوالب) ─────────

def chat_respond(
    query: str,
    ckg_concepts: dict,
    multimodal_core: "MultimodalRoutingCore",
    deep_router=None,                 # مرّر هنا get_default_deep_network() الموجودة لديك دون تعديلها
    image_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    llm_call=None,                    # دالة اختيارية: llm_call(prompt) -> str لاستدعاء LLM فعلي
    top_k: int = 5,
) -> dict:
    """
    يُعيد dict فيها: المفاهيم المسترجَعة من CKG + قرار الـAgent + أوزان التوجيه (إن وُجدت)
    + رد نصي مبدئي (قالب) أو رد LLM إن مُرّرت llm_call.
    """
    enc = multimodal_core.encode_multimodal(
        text=query, ckg_concepts=ckg_concepts,
        image_path=image_path, audio_path=audio_path,
    )
    out = multimodal_core.forward(enc["concat"])

    # أفضل المفاهيم المطابقة من CKG (بحث نصي مبسّط، نفس منطق match في TextEncoder)
    import re
    def clean(t):
        t = re.sub(r'[ٱ]', 'ا', t)
        t = re.sub(r'[^\u0600-\u06FF\s]', ' ', t)
        return t.strip()
    q_words = set(clean(query).split())
    scored = []
    for name, meta in ckg_concepts.items():
        nc = clean(name)
        if nc in q_words or any(w in nc for w in q_words if len(w) >= 3):
            scored.append((name, meta.get("strength", 0.1) * (1 + min(1, meta.get("frequency", 1) / 500))))
    scored.sort(key=lambda t: -t[1])
    top_concepts = [n for n, _ in scored[:top_k]]

    routing_weights = None
    if deep_router is not None:
        try:
            routing_vec = multimodal_core.to_routing_vector(out["fused"], dim=128)
            routing_weights = deep_router.predict_routing_weights(routing_vec)
        except Exception as e:
            logger.warning(f"deep_router call failed: {e}")

    agent_action = int(np.argmax(out["agent_probs"]))

    if llm_call is not None:
        prompt = (f"سؤال المستخدم: {query}\n"
                  f"مفاهيم ذات صلة من القرآن: {', '.join(top_concepts) if top_concepts else 'لا يوجد'}\n"
                  f"اكتب ردًا عربيًا واضحًا ومبنيًا على هذه المفاهيم فقط.")
        reply_text = llm_call(prompt)
    else:
        reply_text = ("مفاهيم ذات صلة: " + "، ".join(top_concepts)) if top_concepts else \
                     "لم أجد مفاهيم مرتبطة مباشرة في الرسم المعرفي الحالي."

    return {
        "query": query,
        "top_concepts": top_concepts,
        "agent_action": agent_action,
        "agent_probs": out["agent_probs"].tolist(),
        "analysis": out["analysis"].tolist(),
        "routing_weights": routing_weights,
        "reply": reply_text,
    }
