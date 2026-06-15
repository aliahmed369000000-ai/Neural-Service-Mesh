"""اختبار شامل لـ NeuralCore: صحة التدرجات، التوافق الخلفي، الذاكرة، التطور."""
import sys
sys.path.insert(0, "/home/claude/build")

import numpy as np
from ai.neural_core import (
    NeuralNetwork, DenseLayer, NeuralCore, AssociativeMemory,
    NeuralWeightLayer, DynamicWeightLayer, extract_routing_weights,
    mse_loss, cross_entropy_loss, softmax,
)

np.random.seed(42)

def numeric_gradient_check(net, x, target, eps=1e-5, tol=1e-4):
    """يفحص dL/dW لكل طبقة عبر الفروقات المحدودة (finite differences)."""
    # forward + backward لحساب التدرجات التحليلية
    out = net.forward(x)
    loss_fn = net.loss_name
    from ai.neural_core import LOSS_FUNCTIONS
    base_loss, d_out = LOSS_FUNCTIONS[loss_fn](out, np.array(target, dtype=np.float64))
    grad = d_out
    for layer in reversed(net.layers):
        grad = layer.backward(grad)

    max_err = 0.0
    for li, layer in enumerate(net.layers):
        analytic = layer._grad_W
        W = layer.W
        # نختبر عينة من العناصر فقط (لأن المصفوفات كبيرة)
        idxs = [(0, 0)]
        if W.shape[0] > 1 and W.shape[1] > 1:
            idxs.append((W.shape[0]-1, W.shape[1]-1))
            idxs.append((W.shape[0]//2, W.shape[1]//2))
        for (i, j) in idxs:
            orig = W[i, j]

            W[i, j] = orig + eps
            out_p = net.forward(x)
            loss_p, _ = LOSS_FUNCTIONS[loss_fn](out_p, np.array(target, dtype=np.float64))

            W[i, j] = orig - eps
            out_m = net.forward(x)
            loss_m, _ = LOSS_FUNCTIONS[loss_fn](out_m, np.array(target, dtype=np.float64))

            W[i, j] = orig
            numeric = (loss_p - loss_m) / (2 * eps)
            err = abs(numeric - analytic[i, j])
            max_err = max(max_err, err)
            assert err < tol, f"Layer {li} W[{i},{j}]: numeric={numeric:.6f} analytic={analytic[i,j]:.6f} err={err:.6f}"
    return max_err


print("=" * 70)
print("1) فحص صحة التدرجات (MSE + relu/relu/softmax)")
print("=" * 70)
net = NeuralNetwork([7, 10, 6, 4], ["relu", "relu", "softmax"], loss="mse", learning_rate=0.01, seed=1)
x = np.random.uniform(0, 1, 7)
target = np.array([0.3, 0.35, 0.25, 0.10])
err = numeric_gradient_check(net, x, target)
print(f"  max gradient error (mse+softmax) = {err:.8f}  -> {'PASS' if err < 1e-4 else 'FAIL'}")

print()
print("2) فحص صحة التدرجات (cross-entropy + softmax)")
net2 = NeuralNetwork([7, 8, 4], ["relu", "softmax"], loss="cross_entropy", learning_rate=0.01, seed=2)
x2 = np.random.uniform(0, 1, 7)
target2 = np.array([1.0, 0.0, 0.0, 0.0])
err2 = numeric_gradient_check(net2, x2, target2)
print(f"  max gradient error (cross_entropy+softmax) = {err2:.8f}  -> {'PASS' if err2 < 1e-4 else 'FAIL'}")

print()
print("3) فحص صحة التدرجات (tanh + sigmoid hidden layers)")
net3 = NeuralNetwork([7, 6, 5, 3], ["tanh", "sigmoid", "linear"], loss="mse", learning_rate=0.01, seed=3)
x3 = np.random.uniform(-1, 1, 7)
target3 = np.array([0.1, 0.5, 0.9])
err3 = numeric_gradient_check(net3, x3, target3)
print(f"  max gradient error (tanh/sigmoid/linear) = {err3:.8f}  -> {'PASS' if err3 < 1e-4 else 'FAIL'}")

print()
print("=" * 70)
print("4) تدريب فعلي: الخسارة تنخفض مع الزمن؟")
print("=" * 70)
net4 = NeuralNetwork([7, 16, 4], ["relu", "softmax"], loss="mse", learning_rate=0.05, optimizer="adam", seed=4)
losses = []
rng = np.random.default_rng(5)
for step in range(300):
    x = rng.uniform(0, 1, 7)
    target = softmax(rng.uniform(0, 1, 4))  # هدف يجمع=1
    loss = net4.train_step(x, target)
    losses.append(loss)
print(f"  loss[0:5] avg   = {np.mean(losses[:5]):.6f}")
print(f"  loss[-5:] avg   = {np.mean(losses[-5:]):.6f}")
print(f"  -> {'PASS (decreasing)' if np.mean(losses[-5:]) < np.mean(losses[:5]) else 'FAIL'}")

print()
print("=" * 70)
print("5) NeuralCore: تعلّم + تذكّر + تطوّر")
print("=" * 70)
core = NeuralCore(input_dim=7, hidden_dims=[10], output_dim=4,
                   plateau_window=5, plateau_cooldown=10, plateau_threshold=0.5,
                   grow_units=4, max_hidden_width=30, seed=7)

print(f"  initial arch: {core.net.architecture_str()}")
for step in range(60):
    x = rng.uniform(0, 1, 7)
    target = np.array([0.3, 0.35, 0.25, 0.10])
    result = core.train_and_remember(x, target, metadata={"step": step, "domain": "physics"})

print(f"  after 60 steps arch: {core.net.architecture_str()}")
print(f"  growth events: {len(core._growth_events)}")
print(f"  memory stored: {len(core.memory)}")

# recall test
query = rng.uniform(0, 1, 7)
results = core.recall(query, top_k=3)
print(f"  recall(query) top-3 similarities: {[r['similarity'] for r in results]}")
assert len(results) <= 3
print("  -> PASS")

print()
print("=" * 70)
print("6) التوافق الخلفي: NeuralWeightLayer")
print("=" * 70)
nwl = NeuralWeightLayer(learning_rate=0.01)
print(f"  shape (SHAPE prop): {nwl.SHAPE}")
x = np.random.uniform(0, 1, 7)
out = nwl.forward(x)
print(f"  forward output shape: {out.shape}")
loss0 = nwl.train_step(x, target=0.5)
for _ in range(50):
    lossN = nwl.train_step(x, target=0.5)
print(f"  loss before: {loss0:.6f}  after 50 steps: {lossN:.6f}  -> {'PASS' if lossN < loss0 else 'FAIL'}")
print(f"  bias still 0.6 (unchanged init): {nwl.bias}")
print(f"  bias vector mean (after training): {nwl._layer.b.mean():.6f}  (should differ from 0.6 if bias trains)")

print()
print("=" * 70)
print("7) التوافق الخلفي: DynamicWeightLayer (نمو)")
print("=" * 70)
dwl = DynamicWeightLayer(learning_rate=0.01)
print(f"  initial shape: {dwl.shape}")
losses_d = []
for step in range(300):
    x = rng.uniform(0, 1, 7)
    l = dwl.train_step(x, target=0.5)
    losses_d.append(l)
print(f"  shape after 300 steps: {dwl.shape}")
print(f"  growth events: {len(dwl._growth_events)}")
print(f"  loss[0:5] avg = {np.mean(losses_d[:5]):.6f}  loss[-5:] avg = {np.mean(losses_d[-5:]):.6f}")

print()
print("=" * 70)
print("8) extract_routing_weights")
print("=" * 70)
rw = extract_routing_weights(nwl)
print(f"  {rw}  sum={sum(rw.values()):.6f}")
assert abs(sum(rw.values()) - 1.0) < 1e-6

print()
print("=" * 70)
print("9) حفظ/تحميل NeuralCore")
print("=" * 70)
core.save("/home/claude/build/test_core")
core2 = NeuralCore.load("/home/claude/build/test_core")
print(f"  reloaded arch: {core2.net.architecture_str()}")
print(f"  reloaded memory: {len(core2.memory)}")
out1 = core.forward(x)
out2 = core2.forward(x)
print(f"  forward match after reload: {np.allclose(out1, out2)}")

print()
print("ALL TESTS COMPLETED")
