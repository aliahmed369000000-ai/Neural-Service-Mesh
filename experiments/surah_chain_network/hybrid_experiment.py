"""
تجربة معمارية: شبكة 114 طبقة (من ملف الإكسل، أبعادها = عدد آيات السور
المتتالية) تمثّل الطبقات الوسطى المخفية فقط، وتُستخدم كـ"طبقة وسطى
مخفية" بين embedding و output head من نفس معمارية arabic_transformer.py.

بنية الشبكة الفرعية الكاملة = 116 طبقة:
  - طبقة 1  (دخول، W_down)         : عشوائية التهيئة، ليست من ملف السور.
  - طبقات 2–115 (114 طبقة وسطى مخفية): من ملف شبكهه_114-1.xlsx.
  - طبقة 116 (خروج، W_up)          : عشوائية التهيئة، ليست من ملف السور.

هذا اختبار محلي بحت — لا رفع لأي شيء.
"""
import sys, json, time
sys.path.insert(0, '/home/claude/nsm_exp')
import numpy as np
from ai.arabic_transformer import TokenEmbedding, OutputHead, HashTokenizer

with open('/home/claude/nsm_exp/surah_layer_dims.json') as f:
    LAYER_DIMS = json.load(f)  # [(in,out), ...] × 114، يبدأ ب7 وينتهي بـ7

D_MODEL = LAYER_DIMS[0][0]   # 7 — لازم embedding يخرّج بنفس القدر
VOCAB_SIZE = 8192


class LayerNorm1D:
    """LayerNorm بسيط لتثبيت مقياس الإشارة بين الطبقات."""
    def __init__(self, dim, eps=1e-5):
        self.g = np.ones(dim)
        self.b = np.zeros(dim)
        self.eps = eps
        self._cache = None

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        xhat = (x - mu) / np.sqrt(var + self.eps)
        self._cache = (xhat,)
        return xhat * self.g + self.b

    def backward(self, grad, lr):
        (xhat,) = self._cache
        gg = (grad * xhat).sum(axis=0)
        gb = grad.sum(axis=0)
        self.g -= lr * gg
        self.b -= lr * gb
        return grad * self.g  # تبسيط: نتجاهل تدرّج mean/var (شائع في تطبيقات مبسّطة)


class SurahChainLayer:
    """
    طبقة FC واحدة من السلسلة + LayerNorm + Residual connection.
    لو d_in != d_out، بيتضاف إسقاط صغير (شبيه بـ projection shortcut في
    ResNet) عشان الـresidual يفضل شغّال حتى لما الأبعاد تتغيّر — بدون
    كده الـresidual مستحيل رياضياً (مينفعش تجمع متجهين أبعادهم مختلفة).
    """
    def __init__(self, d_in, d_out, seed=None):
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (d_in + d_out))
        self.W = rng.uniform(-limit, limit, (d_out, d_in)).astype(np.float64)
        self.b = np.zeros(d_out, dtype=np.float64)
        self.ln = LayerNorm1D(d_out)
        self._x = None
        self._pre_res = None

        self.has_shortcut_proj = (d_in != d_out)
        if self.has_shortcut_proj:
            lim_s = np.sqrt(6.0 / (d_in + d_out))
            self.W_shortcut = rng.uniform(-lim_s, lim_s, (d_out, d_in)).astype(np.float64)

    def forward(self, x):
        self._x = x
        pre = x @ self.W.T + self.b
        act = np.tanh(pre)
        normed = self.ln.forward(act)
        shortcut = (x @ self.W_shortcut.T) if self.has_shortcut_proj else x
        self._pre_res = (pre, act)
        return normed + shortcut  # residual connection

    def backward(self, grad_out, lr):
        pre, act = self._pre_res
        # فرع الـresidual (shortcut) بيرجع تدرّج مباشر بدون اضمحلال
        if self.has_shortcut_proj:
            g_shortcut_x = grad_out @ self.W_shortcut
            gW_s = grad_out.T @ self._x
            np.clip(gW_s, -5, 5, out=gW_s)
            self.W_shortcut -= lr * gW_s
        else:
            g_shortcut_x = grad_out

        # فرع الطبقة الرئيسية (FC -> tanh -> LayerNorm)
        g_normed = self.ln.backward(grad_out, lr)
        d_pre = g_normed * (1 - act ** 2)
        gW = d_pre.T @ self._x
        gb = d_pre.sum(axis=0)
        gx_main = d_pre @ self.W
        np.clip(gW, -5, 5, out=gW)
        np.clip(gb, -5, 5, out=gb)
        self.W -= lr * gW
        self.b -= lr * gb

        return gx_main + g_shortcut_x  # جمع مساري التدرّج (نفس منطق ResNet)


class SurahChainNetwork:
    """
    الطبقات الوسطى المخفية فقط (114 طبقة، أبعادها من ملف الإكسل
    بالضبط) — تمثّل الطبقات 2–115 من الشبكة الفرعية الكاملة المكوّنة
    من 116 طبقة (مع طبقتي الدخول W_down والخروج W_up العشوائيتين،
    غير المشتقتين من ملف السور).
    """
    def __init__(self, layer_dims):
        self.layers = [SurahChainLayer(d_in, d_out, seed=i)
                       for i, (d_in, d_out) in enumerate(layer_dims)]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
        return grad

    def param_count(self):
        return sum(l.W.size + l.b.size for l in self.layers)


class HybridExperimentModel:
    """
    embedding (من نفس مشروعك)
    -> طبقة 1: إسقاط دخول عشوائي W_down لـ7 أبعاد (ليست من ملف السور)
    -> طبقات 2–115: سلسلة الـ114 طبقة الوسطى المخفية (من ملف السور)
    -> طبقة 116: إسقاط خروج عشوائي W_up لأبعاد المشروع (ليست من ملف السور)
    -> output head (من نفس مشروعك)

    إجمالي الشبكة الفرعية الكاملة: 116 طبقة.
    """
    def __init__(self, project_d_model=256, vocab_size=VOCAB_SIZE, lr=1e-3):
        self.tokenizer = HashTokenizer(vocab_size)
        self.embedding = TokenEmbedding(vocab_size, project_d_model)
        self.head = OutputHead(project_d_model, vocab_size)
        self.chain = SurahChainNetwork(LAYER_DIMS)
        self.lr = lr

        # W_down = الطبقة 1 (طبقة الإدخال) من الشبكة الفرعية الكاملة —
        # عشوائية التهيئة، ليست من ملف السور.
        # W_up   = الطبقة 116 (طبقة الإخراج) من نفس الشبكة الفرعية —
        # عشوائية التهيئة أيضاً، ليست من ملف السور.
        # بينهما (الطبقات 2–115) سلسلة الـ114 طبقة القادمة من الإكسل
        # (SurahChainNetwork)، أبعادها ثابتة (7) مالهاش علاقة بحجم
        # d_model الأصلي للمشروع.
        rng = np.random.default_rng(42)
        lim_down = np.sqrt(6.0 / (project_d_model + D_MODEL))
        self.W_down = rng.uniform(-lim_down, lim_down, (D_MODEL, project_d_model))
        lim_up = np.sqrt(6.0 / (D_MODEL + project_d_model))
        self.W_up = rng.uniform(-lim_up, lim_up, (project_d_model, D_MODEL))

    def train_step(self, text):
        ids = self.tokenizer.encode(text, 12)
        if len(ids) < 2:
            return None
        inp, tgt = ids[:-1], ids[1:]

        X = self.embedding.forward(inp)          # (S, project_d_model)
        X7 = X @ self.W_down.T                    # (S, 7) — دخول السلسلة
        X7_out = self.chain.forward(X7)            # (S, 7) — خروج بعد 114 طبقة
        Xback = X7_out @ self.W_up.T                # (S, project_d_model)
        probs = self.head.forward(Xback)

        n = len(tgt)
        loss = -np.log(np.clip(probs[np.arange(n), tgt], 1e-10, 1)).mean()
        g = probs.copy(); g[np.arange(n), tgt] -= 1; g /= n

        gXback = self.head.backward(g, self.lr)
        gX7_out = gXback @ self.W_up
        self.W_up -= self.lr * (gXback.T @ X7_out)
        gX7 = self.chain.backward(gX7_out, self.lr)
        gX = gX7 @ self.W_down
        self.W_down -= self.lr * (gX7.T @ X)
        self.embedding.backward(gX, self.lr)

        return float(loss)


if __name__ == '__main__':
    m = HybridExperimentModel(project_d_model=256, lr=2e-3)
    print(f"باراميترات سلسلة الـ114 طبقة: {m.chain.param_count():,}")
    print(f"باراميترات embedding: {m.embedding.W.size:,}")
    print(f"باراميترات output head: {m.head.W.size:,}")

    # فحص انتشار الإشارة بعد إضافة LayerNorm+Residual
    ids = m.tokenizer.encode('الرحمة صفة من صفات الله', 12)
    X = m.embedding.forward(ids[:-1])
    x = X @ m.W_down.T
    print("\nفحص انتشار الإشارة عبر الطبقات بعد الإصلاح:")
    for i, layer in enumerate(m.chain.layers):
        x = layer.forward(x)
        if i % 20 == 0 or i == 113:
            print(f"  بعد طبقة {i}: متوسط={np.abs(x).mean():.4f}")

    texts = [
        "الرحمة صفة من صفات الله",
        "الصبر مفتاح الفرج والنجاح",
        "العلم نور يهدي صاحبه",
        "الإيمان بالله أساس كل خير",
        "التوبة باب مفتوح لكل مذنب",
    ] * 40  # تدريب أطول (200 خطوة بدل 30)

    t0 = time.time()
    losses = []
    for i, t in enumerate(texts):
        loss = m.train_step(t)
        if loss is not None:
            losses.append(loss)
    dt = time.time() - t0
    print(f"\n{len(losses)} خطوة تدريب في {dt:.1f}s")
    print("أول 10 loss:", [round(x, 3) for x in losses[:10]])
    print("آخر 10 loss:", [round(x, 3) for x in losses[-10:]])
    print("الاتجاه العام نازل؟", losses[-1] < losses[0])
    print(f"التحسّن: {losses[0]:.3f} -> {losses[-1]:.3f}")
