import os
import math
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Constants (matching NumPy version)
D_MODEL      = 4096
N_HEADS      = 32
D_FF         = 16384
N_LAYERS     = 24
MAX_SEQ_LEN  = 128
VOCAB_SIZE   = 8192

@dataclass
class MultimodalKVCacheTF:
    """نسخة TensorFlow من مخزن الـ KV Cache."""
    layer_past_kv: List[Optional[Tuple[tf.Tensor, tf.Tensor]]] = field(default_factory=list)
    multimodal_kv: Optional[Tuple[tf.Tensor, tf.Tensor]] = None
    metadata: Dict[str, any] = field(default_factory=dict)

    def clear(self):
        self.layer_past_kv = [None] * len(self.layer_past_kv)
        self.multimodal_kv = None
        self.metadata = {}

    def get_seq_len(self) -> int:
        if not self.layer_past_kv or self.layer_past_kv[0] is None:
            return 0
        # Shape: (batch, n_heads, seq_len, dk)
        return tf.shape(self.layer_past_kv[0][0])[2]

class PositionalEncodingTF(tf.keras.layers.Layer):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, seq_len: int, offset: int = 0):
        return self.pe[:, offset : offset + seq_len, :]

class CoreMatrixLayerTF(tf.keras.layers.Layer):
    def __init__(self, d_model: int = D_MODEL, image_dim: int = 512, audio_dim: int = 128, video_dim: int = 512):
        super().__init__()
        self.d_model = d_model
        self.core_dim = 784
        
        # Multimodal Projection Heads
        self.W_img = self.add_weight(name="W_img", shape=(image_dim, d_model), initializer="glorot_uniform")
        self.b_img = self.add_weight(name="b_img", shape=(d_model,), initializer="zeros")
        self.W_aud = self.add_weight(name="W_aud", shape=(audio_dim, d_model), initializer="glorot_uniform")
        self.b_aud = self.add_weight(name="b_aud", shape=(d_model,), initializer="zeros")
        self.W_vid = self.add_weight(name="W_vid", shape=(video_dim, d_model), initializer="glorot_uniform")
        self.b_vid = self.add_weight(name="b_vid", shape=(d_model,), initializer="zeros")

        # Cross-Attention weights
        self.Wq_cross = self.add_weight(name="Wq_cross", shape=(d_model, d_model), initializer="glorot_uniform")
        self.Wk_cross = self.add_weight(name="Wk_cross", shape=(d_model, d_model), initializer="glorot_uniform")
        self.Wv_cross = self.add_weight(name="Wv_cross", shape=(d_model, d_model), initializer="glorot_uniform")

        # Core Matrix weights
        self.W_up = self.add_weight(name="W_up", shape=(d_model, self.core_dim), initializer="glorot_uniform")
        self.W_down = self.add_weight(name="W_down", shape=(self.core_dim, d_model), initializer="glorot_uniform")
        self.b_up = self.add_weight(name="b_up", shape=(self.core_dim,), initializer="zeros")
        self.b_down = self.add_weight(name="b_down", shape=(d_model,), initializer="zeros")
        
        self.W_core = self.add_weight(name="W_core", shape=(self.core_dim, self.core_dim), initializer="glorot_uniform")

    def call(self, X, image_feats=None, audio_feats=None, video_feats=None, multimodal_kv=None):
        # X shape: (batch, seq, d_model)
        multimodal_context = tf.zeros_like(X)
        K_mod, V_mod = None, None

        if multimodal_kv is not None:
            K_mod, V_mod = multimodal_kv
        else:
            modality_embeddings = []
            if image_feats is not None:
                modality_embeddings.append(tf.matmul(image_feats, self.W_img) + self.b_img)
            if audio_feats is not None:
                modality_embeddings.append(tf.matmul(audio_feats, self.W_aud) + self.b_aud)
            if video_feats is not None:
                modality_embeddings.append(tf.matmul(video_feats, self.W_vid) + self.b_vid)
            
            if modality_embeddings:
                M = tf.concat(modality_embeddings, axis=1) # (batch, total_mod_tokens, d_model)
                K_mod = tf.matmul(M, self.Wk_cross)
                V_mod = tf.matmul(M, self.Wv_cross)

        if K_mod is not None:
            Q = tf.matmul(X, self.Wq_cross)
            # Simple Cross-Attention
            scores = tf.matmul(Q, K_mod, transpose_b=True) / math.sqrt(self.d_model)
            attn = tf.nn.softmax(scores, axis=-1)
            multimodal_context = tf.matmul(attn, V_mod)

        X_fused = X + multimodal_context
        up = tf.matmul(X_fused, self.W_up) + self.b_up
        core_out = tf.matmul(up, self.W_core)
        
        # Sign-flip activation
        act = tf.nn.relu(core_out)
        mask_flip = tf.abs(core_out) > 0.15
        act = tf.where(mask_flip, act * -0.5, act)
        
        out = tf.matmul(act, self.W_down) + self.b_down
        return out, (K_mod, V_mod)

class MultiHeadAttentionTF(tf.keras.layers.Layer):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.h = n_heads
        self.dk = d_model // n_heads
        self.d_model = d_model
        
        self.Wq = self.add_weight(name="Wq", shape=(d_model, d_model), initializer="glorot_uniform")
        self.Wk = self.add_weight(name="Wk", shape=(d_model, d_model), initializer="glorot_uniform")
        self.Wv = self.add_weight(name="Wv", shape=(d_model, d_model), initializer="glorot_uniform")
        self.Wo = self.add_weight(name="Wo", shape=(d_model, d_model), initializer="glorot_uniform")

    def call(self, X, mask=None, past_kv=None):
        batch = tf.shape(X)[0]
        seq = tf.shape(X)[1]
        
        Q = tf.matmul(X, self.Wq)
        K = tf.matmul(X, self.Wk)
        V = tf.matmul(X, self.Wv)
        
        Qh = tf.transpose(tf.reshape(Q, (batch, seq, self.h, self.dk)), perm=[0, 2, 1, 3])
        Kh = tf.transpose(tf.reshape(K, (batch, seq, self.h, self.dk)), perm=[0, 2, 1, 3])
        Vh = tf.transpose(tf.reshape(V, (batch, seq, self.h, self.dk)), perm=[0, 2, 1, 3])
        
        if past_kv is not None:
            prev_k, prev_v = past_kv
            Kh = tf.concat([prev_k, Kh], axis=2)
            Vh = tf.concat([prev_v, Vh], axis=2)
        
        current_kv = (Kh, Vh)
        
        scores = tf.matmul(Qh, Kh, transpose_b=True) / math.sqrt(float(self.dk))
        if mask is not None:
            scores += (mask * -1e9)
            
        attn = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(attn, Vh)
        out = tf.reshape(tf.transpose(out, perm=[0, 2, 1, 3]), (batch, seq, self.d_model))
        return tf.matmul(out, self.Wo), current_kv

class TransformerBlockTF(tf.keras.layers.Layer):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.mha = MultiHeadAttentionTF(d_model, n_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation="relu"),
            tf.keras.layers.Dense(d_model)
        ])
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, X, mask=None, past_kv=None):
        attn_out, current_kv = self.mha(self.ln1(X), mask=mask, past_kv=past_kv)
        X = X + attn_out
        X = X + self.ffn(self.ln2(X))
        return X, current_kv

class ArabicTransformerTF(tf.keras.Model):
    def __init__(self, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncodingTF(d_model)
        self.core = CoreMatrixLayerTF(d_model)
        self.blocks = [TransformerBlockTF(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.head = tf.keras.layers.Dense(vocab_size)

    @tf.function
    def call(self, ids, image_feats=None, audio_feats=None, video_feats=None, past_kv=None, use_cache=False):
        batch = tf.shape(ids)[0]
        seq = tf.shape(ids)[1]
        
        pos_offset = 0
        if past_kv is not None:
            pos_offset = past_kv.get_seq_len()
            
        X = self.embedding(ids)
        X += self.pos_enc(seq, offset=pos_offset)
        
        mm_kv_in = None
        if past_kv is not None:
            mm_kv_in = past_kv.multimodal_kv
            
        core_out, mm_kv_out = self.core(X, image_feats, audio_feats, video_feats, multimodal_kv=mm_kv_in)
        X = X + core_out
        
        if past_kv is not None:
            past_kv.multimodal_kv = mm_kv_out
            
        new_layer_kv = []
        # Causal mask
        mask = tf.zeros((1, 1, seq, seq), dtype=tf.float32)
        if seq > 1 and not use_cache:
            mask = tf.linalg.band_part(tf.ones((seq, seq)), 0, -1) - tf.eye(seq)
            mask = mask[tf.newaxis, tf.newaxis, :, :]

        for i, block in enumerate(self.blocks):
            p_kv = None
            if past_kv is not None and len(past_kv.layer_past_kv) > i:
                p_kv = past_kv.layer_past_kv[i]
            X, c_kv = block(X, mask=mask, past_kv=p_kv)
            new_layer_kv.append(c_kv)
            
        if past_kv is not None:
            past_kv.layer_past_kv = new_layer_kv
            
        X = self.ln_f(X)
        logits = self.head(X)
        # Return same signature as NumPy version: (logits, hidden, risk, intent)
        return logits, X, tf.constant(0.0, dtype=tf.float32), tf.constant("neutral", dtype=tf.string)

    def _forward(self, ids, mask=None, image_feats=None, audio_feats=None, video_feats=None, past_kv=None, use_cache=False):
        """توقيع متوافق مع NumPy لاستدعائه من قبل الوكلاء والواجهات."""
        if not isinstance(ids, tf.Tensor):
            ids_t = tf.convert_to_tensor(ids, dtype=tf.int32)
            if len(ids_t.shape) == 1: ids_t = tf.expand_dims(ids_t, 0)
        else:
            ids_t = ids
            
        # تحويل الوسائط إذا كانت NumPy
        img_t = tf.convert_to_tensor(image_feats, dtype=tf.float32) if image_feats is not None else None
        aud_t = tf.convert_to_tensor(audio_feats, dtype=tf.float32) if audio_feats is not None else None
        vid_t = tf.convert_to_tensor(video_feats, dtype=tf.float32) if video_feats is not None else None
        
        logits, hidden, risk, intent = self.call(ids_t, img_t, aud_t, vid_t, past_kv, use_cache)
        
        # تحويل النتائج إلى NumPy لضمان عدم كسر call sites الحالية التي تتوقع مصفوفات NumPy
        return logits[0].numpy(), hidden[0].numpy(), float(risk.numpy()), str(intent.numpy())

    def generate_ids(self, text: str, tokenizer, max_new=20, **kwargs) -> np.ndarray:
        """نسخة TensorFlow من generate_ids مع دعم الـ Caching."""
        from ai.arabic_transformer import ArabicTransformer
        # محاكاة لنموذج NumPy لاستخدام منطق التوليد الخاص به مع محرك TF
        class TFProxyModel:
            def __init__(self, tf_model, tokenizer):
                self.tf_model = tf_model
                self.tokenizer = tokenizer
                self.max_seq = MAX_SEQ_LEN
            def _forward(self, *args, **kwargs):
                return self.tf_model._forward(*args, **kwargs)
            def generate_ids(self, text, **gen_kwargs):
                return ArabicTransformer.generate_ids(self, text, **gen_kwargs)
        
        proxy = TFProxyModel(self, tokenizer)
        return proxy.generate_ids(text, max_new=max_new, **kwargs)

    def load_numpy_weights(self, weights_dir: str):
        """تحميل الأوزان من ملفات .npy الخاصة بـ NumPy إلى TensorFlow."""
        logger.info(f"🔄 Loading weights from {weights_dir}...")
        
        # 1. Embedding
        emb_path = os.path.join(weights_dir, "embedding.npy")
        if os.path.exists(emb_path):
            self.embedding.set_weights([np.load(emb_path)])
            
        # 2. CoreMatrix
        core_prefix = os.path.join(weights_dir, "core")
        core_map = {
            "W_up": "Wu", "W_down": "Wd", "b_up": "bu", "b_down": "bd",
            "W_img": "Wimg", "b_img": "bimg", "W_aud": "Waud", "b_aud": "baud",
            "W_vid": "Wvid", "b_vid": "bvid",
            "Wq_cross": "Wq_cross", "Wk_cross": "Wk_cross", "Wv_cross": "Wv_cross",
            "W_core": "core"
        }
        for tf_attr, np_name in core_map.items():
            p = f"{core_prefix}_{np_name}.npy"
            if os.path.exists(p):
                val = np.load(p)
                # TensorFlow Dense layers might need transpose if they were trained differently
                # But here we use manual matmul, so we match NumPy shapes
                getattr(self.core, tf_attr).assign(val)

        # 3. Transformer Blocks
        for i in range(self.n_layers):
            b_prefix = os.path.join(weights_dir, f"layer_{i}")
            block = self.blocks[i]
            
            # MHA
            mha_prefix = f"{b_prefix}_mha"
            for n in ["q", "k", "v", "o"]:
                p = f"{mha_prefix}_W{n}.npy"
                if os.path.exists(p):
                    getattr(block.mha, f"W{n}").assign(np.load(p))
            
            # FFN
            ffn_prefix = f"{b_prefix}_ffn"
            # W1: (d_model, d_ff) -> Dense(d_ff) kernel is (d_model, d_ff)
            # NumPy W1 is (d_ff, d_model), so we transpose for TF Dense kernel
            p_w1 = f"{ffn_prefix}_W1.npy"
            p_b1 = f"{ffn_prefix}_b1.npy"
            if os.path.exists(p_w1) and os.path.exists(p_b1):
                block.ffn.layers[0].set_weights([np.load(p_w1).T, np.load(p_b1)])
            
            p_w2 = f"{ffn_prefix}_W2.npy"
            p_b2 = f"{ffn_prefix}_b2.npy"
            if os.path.exists(p_w2) and os.path.exists(p_b2):
                block.ffn.layers[1].set_weights([np.load(p_w2).T, np.load(p_b2)])

        # 4. Final Norm
        ln_prefix = os.path.join(weights_dir, "final_ln")
        g_p, b_p = f"{ln_prefix}_g.npy", f"{ln_prefix}_b.npy"
        if os.path.exists(g_p) and os.path.exists(b_p):
            self.ln_f.set_weights([np.load(g_p), np.load(b_p)])
            
        logger.info("✅ Weights migration complete.")

