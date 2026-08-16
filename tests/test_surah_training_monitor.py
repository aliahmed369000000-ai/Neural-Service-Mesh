"""اختبارات وحدة مراقبة التدريب الحي (surah_training_monitor) — بدون مفاتيح حقيقية."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def test_import_clean():
    """الاستيراد لا يُطلق استثناءات ولا يستدعي network عند التحميل."""
    import importlib
    mod = importlib.import_module("ai.surah_training_monitor")
    assert hasattr(mod, "fetch_live_state")
    assert hasattr(mod, "render_live_training_dashboard")
    assert hasattr(mod, "live_training_state_text")


def test_fetch_latest_progress_shape():
    """جلب التقدم — النتيجة None أو dict بمفاتيح معروفة (بدون مفاتيح Kaggle)."""
    from ai.surah_training_monitor import fetch_latest_progress
    p = fetch_latest_progress()
    if p is not None:
        assert isinstance(p, dict)
        assert "epoch" in p
        assert "loss" in p
        assert "updated_at" in p


def test_fetch_live_state_keys():
    """الحالة الحيّة تعيد dict بالمفاتيح الأساسية حتى لو تعذّر الجلب."""
    from ai.surah_training_monitor import fetch_live_state
    live = fetch_live_state("aliahmedmo/nsm-surahchain-xlarge-12h-fix2")
    assert isinstance(live, dict)
    for key in ("kernel_slug", "kernel_status", "progress", "checkpoints", "fetched_at"):
        assert key in live


def test_fallback_status_from_progress():
    """عند غياب Kaggle API + وجود تقدم حديث → حالة RUNNING استدلالية."""
    from ai import surah_training_monitor as mon
    recent = {"epoch": 5, "end_epoch": 999, "loss": 3.2, "updated_at": __import__("time").time(), "started_at": __import__("time").time() - 600}
    with mock.patch.object(mon, "fetch_latest_progress", return_value=recent), \
         mock.patch.object(mon, "_try_kaggle_status", return_value=None):
        status = mon.kaggle_kernel_status("aliahmedmo/nsm-surahchain-xlarge-12h-fix2")
        assert "RUNNING" in status


def test_state_text_agent_format():
    """live_training_state_text تعيد نصًا عربيًا صالحًا للوكلاء."""
    from ai import surah_training_monitor as mon
    prog = {"epoch": 12, "end_epoch": 999, "loss": 2.9, "best_loss": 2.95, "elapsed": 1200}
    with mock.patch.object(mon, "fetch_live_state", return_value={
        "kernel_slug": "aliahmedmo/x", "kernel_status": "RUNNING", "progress": prog
    }):
        txt = mon.live_training_state_text("aliahmedmo/x")
        assert "العصر: 12" in txt
        assert "الخسارة: 2.9" in txt


def test_py_compile():
    """فحص صياغي للملفين الجديدين."""
    import py_compile
    for f in ("ai/surah_training_monitor.py", "ui_pages/training_monitor.py"):
        py_compile.compile(str(HERE / f), doraise=True)
