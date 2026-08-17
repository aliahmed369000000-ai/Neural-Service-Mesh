# -*- coding: utf-8 -*-
"""
اختبارات خط فحص الجودة الإلزامي — أهم 10 وحدات حرجة
=====================================================
يتكون هذا الخط من شقين:
  1) اختبار استاتيكي: py_compile لجميع الوحدات الحرجة (SyntaxError يُكشف
     فورًا قبل أي تشغيل).
  2) اختبارات وظيفية: دوال نقية قابلة للاختبار بدون مفاتيح API أو أجهزة GPU.

الوحدات المشمولة (الأضخم والأكثر استخدامًا في المشروع):
  ai/neural_core.py              (3015 سطرًا — قلب الرياضيات)
  ai/model_training_agent.py     (3582 سطرًا — وكيل التدريب)
  ai/nsm_agent_core.py           (2282 سطرًا — نواة الوكيل)
  ai/video_engine.py             (1962 سطرًا — محرك الفيديو)
  ai/arabic_transformer.py       (1951 سطرًا — محول العربية)
  ai/kaggle_provider.py          (1707 سطرًا — مزوّد Kaggle)
  ai/pre_action_reasoning.py     (1661 سطرًا — التفكير ما قبل الفعل)
  ai/notebook_engine.py          (1561 سطرًا — محرك الـNotebook)
  ai/arabic_nlp.py               (1403 سطرًا — معالجة العربية)
  ai/agent_categories.py         (1300 سطرًا — تصنيف الوكلاء)
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ═══════════════════════════════════════════════════════════════════════════
# 1) الفحص الاستاتيكي — py_compile لكل الوحدات الحرجة
# ═══════════════════════════════════════════════════════════════════════════

CRITICAL_MODULES = [
    "ai.neural_core",
    "ai.model_training_agent",
    "ai.nsm_agent_core",
    "ai.video_engine",
    "ai.arabic_transformer",
    "ai.kaggle_provider",
    "ai.pre_action_reasoning",
    "ai.notebook_engine",
    "ai.arabic_nlp",
    "ai.agent_categories",
]


def test_compile_critical_modules():
    """الوحدات الحرجة العشرة يجب أن تُجمَّع بدون SyntaxError."""
    import py_compile
    for mod in CRITICAL_MODULES:
        src = Path(ROOT, *mod.split("."))
        py_compile.compile(str(src) + ".py", doraise=True)


def test_import_critical_modules():
    """استيراد الوحدات الحرجة يجب أن لا يرفع استثناءات عند التحميل."""
    for mod in CRITICAL_MODULES:
        # إعادة تحميل نظيف لكل وحدة
        importlib.import_module(mod)

# ═══════════════════════════════════════════════════════════════════════════
# 2) اختبارات وظيفية — دوال نقية بدون مفاتيح API أو GPU
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np


# ── ai/neural_core.py — الرياضيات الأساسية ───────────────────────────────

def test_neural_core_activations_shape_and_range():
    """دوال التفعيل تحافظ على الأبعاد وتعيد قيمًا في النطاق الصحيح."""
    from ai.neural_core import (
        relu, relu_grad, sigmoid, sigmoid_grad_from_output, tanh,
        tanh_grad_from_output, softmax, linear,
    )
    z = np.array([-2.0, 0.0, 2.0])
    z2 = np.array([0.5, -0.5, 3.0])
    out = relu(z)
    assert out.shape == z.shape and np.all(out >= 0)
    assert np.array_equal(relu_grad(z), (z > 0.0).astype(np.float64))
    sig = sigmoid(z)
    assert np.all((sig > 0) & (sig < 1)) and sig.shape == z.shape
    sg = sigmoid_grad_from_output(sig)
    assert np.all(sg > 0) and sg.shape == z.shape
    th = tanh(z)
    assert np.all((th >= -1) & (th <= 1))
    assert np.allclose(tanh_grad_from_output(th), 1.0 - th ** 2)
    sm = softmax(z)
    assert abs(sm.sum() - 1.0) < 1e-9 and np.all(sm > 0)
    assert np.array_equal(linear(z), z)


def test_neural_core_softmax_stability():
    """softmax لا ينفجر بقيم كبيرة (الانزياح max trick)."""
    from ai.neural_core import softmax
    huge = softmax(np.array([1e3, 1e3 + 1, 1e3 + 2]))
    assert np.isfinite(huge).all() and abs(huge.sum() - 1.0) < 1e-9
    assert huge[2] > huge[1] > huge[0]


def test_neural_core_dense_layer_forward_backward():
    """DenseLayer: forward/backward بأبعاد صحيحة وتدرّج لا يتلاشى للأصفار."""
    from ai.neural_core import DenseLayer, mse_loss
    layer = DenseLayer(4, 3, activation="relu", seed=7)
    x = np.random.default_rng(3).standard_normal(3)
    out = layer.forward(x)
    assert out.shape == (4,)
    d_upstream = np.ones(4)
    dx = layer.backward(d_upstream)
    assert dx.shape == (3,)
    assert hasattr(layer, "_grad_W") and hasattr(layer, "_grad_b")
    assert layer._grad_W.shape == layer.W.shape and layer._grad_b.shape == (4,)
    # MSE loss: التدرّج يجب أن يكون غير صفري مع بيانات مختلفة
    t = np.zeros(4)
    loss, dout = mse_loss(out, t)
    assert np.isfinite(loss) and loss >= 0
    assert not np.allclose(dout, 0)


def test_neural_core_associative_memory_retrieval():
    """الذاكرة الترابطية تخزّن وتعيد المتجه الأكثر تشابهًا (remember/recall)."""
    from ai.neural_core import AssociativeMemory
    mem = AssociativeMemory(dim=8, capacity=100, name="unittest")
    key = np.zeros(8)
    idx = mem.remember(key, {"payload": "test"})
    assert isinstance(idx, int)
    top = mem.recall(key, top_k=1)
    # recall قد يعيد قائمة tuples أو قائمة dicts حسب التنفيذ — نكتفي
    # بأن النتيجة غير فارغة وأنها تشير إلى فهرس الذكرى المخزّنة
    assert top, "recall عاد فارغًا لذكرى موجودة"
    assert top[0].get("index") == idx and "metadata" in top[0]


# ── ai/model_training_agent.py — دوال نقية ───────────────────────────────

def test_mta_kernel_source_generation():
    """_torch_kernel_source يولّد كود PyTorch مكتمل بأرقام صحيحة."""
    from ai.model_training_agent import _torch_kernel_source
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 1, 0, 1])
    src = _torch_kernel_source(X, y, "classification", epochs=10,
                               batch_size=2, run_id="r1",
                               checkpoint_dir="/tmp/ck", best_ckpt=None)
    assert "torch" in src and "nn.Linear" in src
    assert "epochs': ep + 1" in src or "'epochs': ep + 1" in src
    assert "n_out" in src
    # مصفوفة الإدخال يجب أن تظهر كاملة داخل الكود
    assert "[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]" in src


def test_mta_collect_text_streams():
    """_kern_collect_text يجمع stdout ويصف الأخطاء."""
    from ai.model_training_agent import _kern_collect_text
    res = {
        "outputs": [
            {"type": "stream", "name": "stdout", "text": "epoch 1 loss=0.5\n"},
            {"type": "stream", "name": "stderr", "text": "warn1"},
        ],
        "error": "",
    }
    out = _kern_collect_text(res)
    assert "epoch 1 loss=0.5" in out
    assert "warn1" in out and "⚠ kernel:" in out
    # kernel error ظاهر
    out2 = _kern_collect_text({"outputs": [], "error": "OOM killed"})
    assert "OOM killed" in out2


def test_mta_oom_backoff_halves_batch():
    """درس OOM يقلّص batch المقترح إلى النصف ويسجّله."""
    from ai.model_training_agent import (
        _read_mta_lessons, _write_mta_lessons,
        _record_kernel_oom_lesson, ARTIFACTS,
    )
    bak = None
    lessons_path = ARTIFACTS / "mta_lessons.json"
    if lessons_path.is_file():
        bak = open(lessons_path, encoding="utf-8").read()
    try:
        _write_mta_lessons({"preferred_batch": None, "lessons": [], "oom_count": 0})
        lessons = _record_kernel_oom_lesson(context="unittest", current_batch=64)
        assert lessons["preferred_batch"] == 32
        assert lessons["oom_count"] == 1
        lessons2 = _record_kernel_oom_lesson(context="unittest")
        assert lessons2["preferred_batch"] == 16
        assert lessons2["oom_count"] == 2
    finally:
        if bak is not None:
            open(lessons_path, "w", encoding="utf-8").write(bak)
        elif lessons_path.is_file():
            lessons_path.unlink()


# ── ai/kaggle_provider.py — دوال نقية ─────────────────────────────────────

def test_kaggle_safe_slug():
    """_safe_slug يطهّر الأسماء إلى slug ASCII آمن بطول ≤ 48."""
    from ai.kaggle_provider import _safe_slug
    s = _safe_slug("hello world/unsafe@chars_تجربة")
    assert len(s) <= 48
    # لا رموز خطرة في slug الناتج (فقط حروف/أرقام/شرطات/شرطات سفلية)
    assert all(c.isalnum() or c in "-_" for c in s)
    assert s
    assert _safe_slug("") == "nsm"
    assert _safe_slug("x" * 100) == "x" * 48


def test_kaggle_credentials_status_no_secrets_leak():
    """credentials_status يفحص الحالة دون كشف الأسرار نفسها."""
    import os
    from ai.kaggle_provider import credentials_status
    status = credentials_status()
    assert isinstance(status, dict)
    assert "ready" in status and "hint" in status
    for secret_key in ("KAGGLE_KEY", "KAGGLE_USERNAME"):
        assert os.environ.get(secret_key, "") not in str(status) or not os.environ.get(secret_key)


# ── ai/notebook_engine.py — خلايا ودفاتر ──────────────────────────────────

def test_notebook_engine_cell_lifecycle():
    """إنشاء دفتر + خلية + حذف الخلية ينجح ويعيد حالات سليمة."""
    from ai.notebook_engine import create_notebook, add_cell, delete_cell
    nb = create_notebook("فحص-فحص", template="training")
    assert nb.name == "فحص-فحص" and nb.cells
    cell = add_cell(nb, cell_type="code", source="print(1)")
    assert cell.source == "print(1)"
    cid = cell.id
    assert delete_cell(nb, cid)
    assert cid not in {c.id for c in nb.cells}


def test_notebook_engine_truncate():
    """_truncate يتعامل مع النصوص دون أن يفشل (سلوكه داخلي)."""
    from ai.notebook_engine import _truncate
    long_text = "أ" * 2000
    out = _truncate(long_text)
    assert isinstance(out, str)
    assert len(out) <= 2000
    assert "أ" in out


# ── ai/arabic_nlp.py — معالجة النص العربي ─────────────────────────────────

def test_arabic_nlp_fnv1a_deterministic():
    """FNV-1a حتمية: نفس النص يعطي نفس الهاش دائمًا."""
    from ai.arabic_nlp import _fnv1a_hash
    h1 = _fnv1a_hash("بسم الله الرحمن الرحيم")
    h2 = _fnv1a_hash("بسم الله الرحمن الرحيم")
    assert h1 == h2 and isinstance(h1, int)
    # نصان مختلفان يُتوقع اختلاف هاشيهما
    assert _fnv1a_hash("النص الأول") != _fnv1a_hash("النص الثاني")


def test_arabic_nlp_hamza_normalization():
    """_normalize_hamza يوحّد جميع أشكال الهمزة إلى ألف."""
    from ai.arabic_nlp import _normalize_hamza
    assert _normalize_hamza("أئمة") == "اامة"
    assert _normalize_hamza("مؤمن") == "مامن"
    assert _normalize_hamza("إسلام") == "اسلام"
    assert _normalize_hamza("بلا همزة") == "بلا همزة"
    # التأكد من أن كل أشكال الهمزة الست اختفت من الناتج
    assert not any(ch in _normalize_hamza("أ إ آ ٱ ؤ ئ") for ch in "أإآٱؤئ")
