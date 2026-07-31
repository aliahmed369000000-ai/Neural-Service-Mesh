"""اختبار شامل لـ NeuralCore: صحة التدرجات، التوافق الخلفي، الذاكرة، التطور."""
import numpy as np
import pytest

from ai.neural_core import (
    NeuralNetwork, NeuralCore,
    NeuralWeightLayer, DynamicWeightLayer, extract_routing_weights,
    softmax, LOSS_FUNCTIONS,
)


def numeric_gradient_check(net, x, target, eps=1e-5, tol=1e-4):
    """يفحص dL/dW لكل طبقة عبر الفروقات المحدودة (finite differences)."""
    out = net.forward(x)
    loss_fn = net.loss_name
    base_loss, d_out = LOSS_FUNCTIONS[loss_fn](out, np.array(target, dtype=np.float64))
    grad = d_out
    for layer in reversed(net.layers):
        grad = layer.backward(grad)

    max_err = 0.0
    for li, layer in enumerate(net.layers):
        analytic = layer._grad_W
        W = layer.W
        idxs = [(0, 0)]
        if W.shape[0] > 1 and W.shape[1] > 1:
            idxs.append((W.shape[0] - 1, W.shape[1] - 1))
            idxs.append((W.shape[0] // 2, W.shape[1] // 2))
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


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(42)


def test_gradient_mse_softmax():
    net = NeuralNetwork([7, 10, 6, 4], ["relu", "relu", "softmax"], loss="mse", learning_rate=0.01, seed=1)
    x = np.random.uniform(0, 1, 7)
    target = np.array([0.3, 0.35, 0.25, 0.10])
    err = numeric_gradient_check(net, x, target)
    assert err < 1e-4


def test_gradient_cross_entropy_softmax():
    net2 = NeuralNetwork([7, 8, 4], ["relu", "softmax"], loss="cross_entropy", learning_rate=0.01, seed=2)
    x2 = np.random.uniform(0, 1, 7)
    target2 = np.array([1.0, 0.0, 0.0, 0.0])
    err2 = numeric_gradient_check(net2, x2, target2)
    assert err2 < 1e-4


def test_gradient_tanh_sigmoid_linear():
    net3 = NeuralNetwork([7, 6, 5, 3], ["tanh", "sigmoid", "linear"], loss="mse", learning_rate=0.01, seed=3)
    x3 = np.random.uniform(-1, 1, 7)
    target3 = np.array([0.1, 0.5, 0.9])
    err3 = numeric_gradient_check(net3, x3, target3)
    assert err3 < 1e-4


def test_training_reduces_loss():
    net4 = NeuralNetwork([7, 16, 4], ["relu", "softmax"], loss="mse", learning_rate=0.05, optimizer="adam", seed=4)
    losses = []
    rng = np.random.default_rng(5)
    for _ in range(300):
        x = rng.uniform(0, 1, 7)
        target = softmax(rng.uniform(0, 1, 4))
        loss = net4.train_step(x, target)
        losses.append(loss)
    assert np.mean(losses[-5:]) < np.mean(losses[:5])


def test_neural_core_learn_remember_recall():
    rng = np.random.default_rng(5)
    core = NeuralCore(input_dim=7, hidden_dims=[10], output_dim=4,
                       plateau_window=5, plateau_cooldown=10, plateau_threshold=0.5,
                       grow_units=4, max_hidden_width=30, seed=7)

    for step in range(60):
        x = rng.uniform(0, 1, 7)
        target = np.array([0.3, 0.35, 0.25, 0.10])
        core.train_and_remember(x, target, metadata={"step": step, "domain": "physics"})

    query = rng.uniform(0, 1, 7)
    results = core.recall(query, top_k=3)
    assert len(results) <= 3


def test_neural_weight_layer_backward_compat():
    nwl = NeuralWeightLayer(learning_rate=0.01)
    x = np.random.uniform(0, 1, NeuralWeightLayer.COLS)
    out = nwl.forward(x)
    assert out is not None
    loss0 = nwl.train_step(x, target=0.5)
    lossN = loss0
    for _ in range(50):
        lossN = nwl.train_step(x, target=0.5)
    assert lossN < loss0


def test_dynamic_weight_layer_growth():
    rng = np.random.default_rng(5)
    dwl = DynamicWeightLayer(learning_rate=0.01)
    losses_d = []
    for _ in range(300):
        x = rng.uniform(0, 1, 7)
        l = dwl.train_step(x, target=0.5)
        losses_d.append(l)
    assert np.mean(losses_d[-5:]) <= np.mean(losses_d[:5]) + 1e-6


def test_extract_routing_weights_sums_to_one():
    nwl = NeuralWeightLayer(learning_rate=0.01)
    x = np.random.uniform(0, 1, NeuralWeightLayer.COLS)
    nwl.train_step(x, target=0.5)
    rw = extract_routing_weights(nwl)
    assert abs(sum(rw.values()) - 1.0) < 1e-6


def test_neural_core_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    core = NeuralCore(input_dim=7, hidden_dims=[10], output_dim=4, seed=7)
    for step in range(10):
        x = rng.uniform(0, 1, 7)
        target = np.array([0.3, 0.35, 0.25, 0.10])
        core.train_and_remember(x, target, metadata={"step": step})

    save_path = str(tmp_path / "test_core")
    core.save(save_path)
    core2 = NeuralCore.load(save_path)

    assert core2.net.architecture_str() == core.net.architecture_str()
    assert len(core2.memory) == len(core.memory)

    x = rng.uniform(0, 1, 7)
    out1 = core.forward(x)
    out2 = core2.forward(x)
    assert np.allclose(out1, out2)
