"""تصعيد WARNING → CRITICAL لتنبيهات دورة التشخيص (run_nodes_diagnose_cycle)
عندما تكون نسبة/عدد العُقد المتأثرة كبيرة، + إنفاذ AlertManager.alert_levels
لقنوات الإشعار الخارجية (Telegram/Email) دون التأثير على السجل المحلي."""
from __future__ import annotations

import shutil
import tempfile

import pytest

from core.mesh_bundle import (
    MeshBundle,
    _diagnose_alert_level,
    DIAGNOSE_CRITICAL_RATIO,
    DIAGNOSE_CRITICAL_ABS_COUNT,
    DIAGNOSE_CRITICAL_MIN_SCANNED,
)


@pytest.fixture()
def bundle():
    tmp = tempfile.mkdtemp(prefix="nsm_diag_escalation_")
    try:
        yield MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── اختبارات الدالة المساعِدة البحتة ──────────────────────────────────────

def test_diagnose_alert_level_small_mesh_stays_warning():
    """شبكة صغيرة (أقل من DIAGNOSE_CRITICAL_MIN_SCANNED) لا تُصعَّد بالنسبة."""
    assert _diagnose_alert_level(1, 2) == "WARNING"


def test_diagnose_alert_level_majority_ratio_escalates():
    """نصف العُقد الممسوحة فأكثر (وعددها كافٍ) → CRITICAL."""
    assert _diagnose_alert_level(5, 10) == "CRITICAL"
    assert _diagnose_alert_level(1, 10) == "WARNING"


def test_diagnose_alert_level_absolute_count_escalates_regardless_of_ratio():
    """عدد مطلق كبير يصعّد حتى لو كانت النسبة صغيرة (شبكة كبيرة جداً)."""
    assert _diagnose_alert_level(DIAGNOSE_CRITICAL_ABS_COUNT, 1000) == "CRITICAL"
    assert _diagnose_alert_level(DIAGNOSE_CRITICAL_ABS_COUNT - 1, 1000) == "WARNING"


# ── اختبار تكاملي: run_nodes_diagnose_cycle فعلياً يرسل CRITICAL ─────────

def test_run_nodes_diagnose_cycle_sends_critical_when_majority_low_rep(bundle, monkeypatch):
    role_ids = list(bundle.role_node_ids.values())
    assert len(role_ids) >= DIAGNOSE_CRITICAL_MIN_SCANNED

    # اجعل أغلب العُقد (6 من 10) بسمعة شبه معدومة، والبقية بسمعة مرتفعة
    # نسبياً — حتى يرتفع المتوسط بما يكفي لإسقاط الأغلبية المتعثرة تحت
    # العتبة الديناميكية الفعلية، بدل الاعتماد على penalise() المحدود
    # عملياً بحد أقصى -10 نقطة (غير كافٍ لإنتاج حالة "سمعة منخفضة" حقيقية).
    low_ids = set(role_ids[:6])
    scores = {nid: (0.01 if nid in low_ids else 50.0) for nid in role_ids}

    def fake_get_score(node_id):
        return scores.get(node_id, 50.0)

    monkeypatch.setattr(bundle.reputation_engine, "get_score", fake_get_score)

    captured = []
    import ai.alert_manager as alert_mod

    def fake_send_alert(level, message, details=None, throttle_sec=60):
        captured.append({"level": level, "message": message, "details": details})

    monkeypatch.setattr(alert_mod.alert_manager, "send_alert", fake_send_alert)

    out = bundle.run_nodes_diagnose_cycle()
    assert out["low_reputation"] >= 6

    low_rep_alerts = [c for c in captured if "سمعة منخفضة" in c["message"]]
    assert low_rep_alerts, "expected a low-reputation alert to be sent"
    assert low_rep_alerts[-1]["level"] == "CRITICAL"


def test_run_nodes_diagnose_cycle_single_low_node_stays_warning(bundle, monkeypatch):
    """عقدة واحدة متعثرة من بين عشرة لا تستحق CRITICAL — تبقى WARNING كالسابق."""
    role_ids = list(bundle.role_node_ids.values())
    low_ids = {role_ids[0]}
    scores = {nid: (0.01 if nid in low_ids else 50.0) for nid in role_ids}

    def fake_get_score(node_id):
        return scores.get(node_id, 50.0)

    monkeypatch.setattr(bundle.reputation_engine, "get_score", fake_get_score)

    captured = []
    import ai.alert_manager as alert_mod

    def fake_send_alert(level, message, details=None, throttle_sec=60):
        captured.append({"level": level, "message": message})

    monkeypatch.setattr(alert_mod.alert_manager, "send_alert", fake_send_alert)

    out = bundle.run_nodes_diagnose_cycle()
    assert out["low_reputation"] == 1

    low_rep_alerts = [c for c in captured if "سمعة منخفضة" in c["message"]]
    assert low_rep_alerts
    assert low_rep_alerts[-1]["level"] == "WARNING"


# ── اختبار AlertManager.alert_levels: يُنفَّذ فعلياً على قنوات الإشعار ───

def test_alert_manager_respects_alert_levels_for_external_channels(tmp_path, monkeypatch):
    monkeypatch.setenv("NSM_ALERT_CONFIG_DIR", str(tmp_path))
    import importlib
    import ai.alert_manager as alert_mod
    importlib.reload(alert_mod)

    mgr = alert_mod.AlertManager()
    mgr._init_history_db()

    sent = {"telegram": [], "email": []}
    monkeypatch.setattr(mgr, "_send_telegram", lambda text: sent["telegram"].append(text))
    monkeypatch.setattr(mgr, "_send_email", lambda subject, body: sent["email"].append((subject, body)))
    mgr.config["telegram"]["enabled"] = True
    mgr.config["email"]["enabled"] = True
    # الإعداد الافتراضي: alert_levels = ["CRITICAL", "SECURITY"]

    mgr.send_alert("WARNING", "warn 1", throttle_sec=0)
    mgr.send_alert("INFO", "info 1", throttle_sec=0)
    mgr.send_alert("CRITICAL", "critical 1", throttle_sec=0)

    assert len(sent["telegram"]) == 1
    assert len(sent["email"]) == 1
    assert "critical 1" in sent["telegram"][0]

    # السجل المحلي يحتفظ بكل المستويات رغم الفلترة الخارجية
    hist_levels = sorted(a["level"] for a in mgr.get_recent_alerts(limit=10))
    assert hist_levels == ["CRITICAL", "INFO", "WARNING"]
