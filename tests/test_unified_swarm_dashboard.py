# -*- coding: utf-8 -*-
"""اختبارات لوحة السرب الموحدة (Unified Swarm Dashboard).

لا تعتمد على أي مفاتيح API — أحداث محاكاة في الذاكرة فقط.
الهدف: التأكد من عمل طبقة التجميع الجديدة قبل رفعها إلى الإنتاج،
وأن فشل أي مصدر بيانات (ناقل الأحداث/السرب/المهام طويلة الأمد)
لا يكسر اللوحة بأكملها.
"""
from __future__ import annotations

import platform
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# مسارات البيانات المحلية للوحة — نستخدم نسخة عزل لكل عملية pytest
DATA_DIR = ROOT / ".nsm_data" / "swarm_dashboard"
ALERT_RULES = DATA_DIR / "alert_rules.json"
AUTO_ACTIONS = DATA_DIR / "auto_actions.json"


@pytest.fixture(autouse=True)
def _isolate_dashboard_state(tmp_path, monkeypatch):
    """عزل كامل لحالة اللوحة أثناء الاختبارات (لا نلوّث ملف المشروع).

    نبدأ كل اختبار بحالة افتراضية نظيفة عبر حذف أي ملفات محفوظة من
    اختبار سابق في نفس العملية.
    """
    import ai.unified_swarm_dashboard as mod
    state_dir = tmp_path / "nsm_data" / "swarm_dashboard"
    state_dir.mkdir(parents=True)
    monkeypatch.setattr(mod, "_NSM_DATA", tmp_path / "nsm_data")
    monkeypatch.setattr(mod, "_ROOT", tmp_path)
    yield


def _compile_ok(path: Path) -> bool:
    return py_compile(path)


def py_compile(path: Path) -> bool:
    """py_compile صريح قبل كل فحص (قاعدة رفع إلزامية في NSM)."""
    import py_compile
    py_compile.compile(str(path), doraise=True)
    return True


class TestCompilation:
    def test_module_compiles(self):
        assert _compile_ok(ROOT / "ai" / "unified_swarm_dashboard.py")

    def test_ui_page_compiles(self):
        assert _compile_ok(ROOT / "ui_pages" / "unified_swarm_dashboard.py")


class TestAlertRules:
    def test_default_rules_exist(self):
        from ai.unified_swarm_dashboard import list_alert_rules
        rules = list_alert_rules()
        ids = {r["id"] for r in rules}
        assert {"slow_agent", "stale_agent", "burst_errors"} <= ids
        by_id = {r["id"]: r for r in rules}
        assert by_id["slow_agent"]["kind"] == "slow_threshold_ms"
        assert by_id["slow_agent"]["value"] == 12000
        assert by_id["stale_agent"]["kind"] == "stale_threshold_s"
        assert by_id["burst_errors"]["kind"] == "error_ratio"
        assert by_id["burst_errors"]["value"] == pytest.approx(0.2)

    def test_update_rule_persists(self):
        from ai.unified_swarm_dashboard import list_alert_rules, update_alert_rule
        updated = update_alert_rule("slow_agent", value=7500, enabled=False)
        assert updated["value"] == 7500
        assert updated["enabled"] is False
        # القراءة التالية يجب أن تعود بالقيمة المحفوظة
        by_id = {r["id"]: r for r in list_alert_rules()}
        assert by_id["slow_agent"]["value"] == 7500
        assert by_id["slow_agent"]["enabled"] is False

    def test_update_rule_invalid_id(self):
        from ai.unified_swarm_dashboard import update_alert_rule
        assert update_alert_rule("nonexistent_rule") is None

    def test_new_rules_merged_with_defaults(self):
        from ai.unified_swarm_dashboard import list_alert_rules, update_alert_rule
        update_alert_rule("slow_agent", value=7500)
        # قاعدة مضافة لاحقًا يجب أن تُدمج مع المخصص
        mod = __import__("ai.unified_swarm_dashboard", fromlist=["_DEFAULT_ALERT_RULES"])
        original = mod._DEFAULT_ALERT_RULES[:]
        try:
            mod._DEFAULT_ALERT_RULES.append({
                "id": "future_rule", "enabled": True, "label": "قاعدة مستقبلية",
                "description": "", "kind": "x", "value": 1, "severity": "info",
                "auto_action": None})
            by_id = {r["id"]: r for r in list_alert_rules()}
            assert by_id["future_rule"]["label"] == "قاعدة مستقبلية"
            # التعديل المخصص يجب أن يبقى
            assert by_id["slow_agent"]["value"] == 7500
        finally:
            mod._DEFAULT_ALERT_RULES = original


class TestAutoActions:
    def test_default_actions_exist(self):
        from ai.unified_swarm_dashboard import list_auto_actions
        acts = list_auto_actions()
        ids = {a["id"] for a in acts}
        assert {"restart_role", "freeze_swarm", "notify_discord"} <= ids
        by_id = {a["id"]: a for a in acts}
        assert by_id["restart_role"]["enabled"] is True
        assert by_id["freeze_swarm"]["enabled"] is False
        assert by_id["notify_discord"]["enabled"] is False

    def test_toggle_persists(self):
        from ai.unified_swarm_dashboard import list_auto_actions, toggle_auto_action
        toggled = toggle_auto_action("freeze_swarm", True)
        assert toggled["enabled"] is True
        by_id = {a["id"]: a for a in list_auto_actions()}
        assert by_id["freeze_swarm"]["enabled"] is True
        toggle_auto_action("freeze_swarm", False)
        assert list_auto_actions()[0] and {a["id"]: a for a in list_auto_actions()}[
            "freeze_swarm"]["enabled"] is False

    def test_toggle_invalid_id(self):
        from ai.unified_swarm_dashboard import toggle_auto_action
        assert toggle_auto_action("nonexistent_action", True) is None


class TestAgentsOverview:
    def test_empty_events(self):
        from ai.unified_swarm_dashboard import agents_overview
        result = agents_overview(events=[])
        counts = result["counts"]
        assert counts == {"alive": 0, "done": 0, "failed": 0, "slow": 0, "stale": 0}
        assert result["agents"] == {}

    def test_classification(self):
        from ai.unified_swarm_dashboard import agents_overview
        events = [
            {"agent_id": "a1", "event_type": "started", "status": "running"},
            {"agent_id": "a2", "event_type": "done", "status": "done",
             "duration_ms": 5000},
            {"agent_id": "a3", "event_type": "error", "status": "error"},
        ]
        result = agents_overview(events=events)
        assert result["counts"]["alive"] == 1
        assert result["counts"]["done"] == 1
        assert result["counts"]["failed"] == 1
        assert set(result["agents"]) == {"a1", "a2", "a3"}

    def test_slow_detection(self):
        from ai.unified_swarm_dashboard import agents_overview
        events = [
            {"agent_id": "slow1", "event_type": "started", "status": "running",
             "duration_ms": 15000},
            {"agent_id": "fast1", "event_type": "done", "status": "done",
             "duration_ms": 3000},
        ]
        result = agents_overview(events=events)
        assert result["counts"]["slow"] == 1
        assert result["agents"]["slow1"].get("is_slow") is True
        assert "is_slow" not in result["agents"]["fast1"]


class TestEvaluateAlerts:
    def test_empty_events_no_alerts(self):
        from ai.unified_swarm_dashboard import evaluate_alerts
        assert evaluate_alerts(events=[]) == []

    def test_error_ratio_alert(self):
        from ai.unified_swarm_dashboard import evaluate_alerts
        events = [
            {"status": "error"} for _ in range(3)
        ] + [{"status": "done"} for _ in range(7)]  # 30% أخطاء > عتبة 20%
        alerts = evaluate_alerts(events=events)
        ids = [a.get("id") for a in alerts]
        assert "burst_errors" in ids
        burst = next(a for a in alerts if a.get("id") == "burst_errors")
        assert burst["severity"] == "critical"
        assert burst["action_triggered"] == "freeze_swarm"

    def test_slow_and_stale_via_fallback(self):
        from ai.unified_swarm_dashboard import evaluate_alerts
        import time
        events = [
            {"agent_id": "slow_a", "event_type": "started", "status": "running",
             "duration_ms": 20000,
             "timestamp": time.time() - 60},  # بطيء (20s > 12s) وعتيق (60s > 45s)
            {"agent_id": "ok_b", "event_type": "done", "status": "done",
             "duration_ms": 2000, "timestamp": time.time()},
        ]
        alerts = evaluate_alerts(events=events)
        texts = " | ".join(str(a.get("title")) for a in alerts)
        # يجب أن تُلتقط حالة البطء عبر analyze_alerts
        assert any("slow_a" in str(a.get("title", "")) for a in alerts)

    def test_disabled_rules_not_evaluated(self):
        from ai.unified_swarm_dashboard import evaluate_alerts, update_alert_rule
        update_alert_rule("burst_errors", enabled=False)
        events = [{"status": "error"}]  # نسبة أخطاء 100%
        alerts = evaluate_alerts(events=events)
        assert "burst_errors" not in {a.get("id") for a in alerts}
        # إعادة التفعيل كي لا تتأثر بقية الاختبارات في نفس العملية
        update_alert_rule("burst_errors", enabled=True)


class TestApplyAutoActions:
    def test_empty_alerts(self):
        from ai.unified_swarm_dashboard import apply_auto_actions
        assert apply_auto_actions([]) == []

    def test_disabled_action_not_applied(self):
        from ai.unified_swarm_dashboard import apply_auto_actions
        # freeze_swarm موقوف افتراضيًا
        applied = apply_auto_actions([{
            "title": "دفقة أخطاء", "severity": "critical",
            "action_triggered": "freeze_swarm"}])
        assert applied == []

    def test_enabled_action_applied_once(self):
        from ai.unified_swarm_dashboard import apply_auto_actions, toggle_auto_action
        alerts_two = [
            {"title": "دفقة أخطاء", "severity": "critical",
             "action_triggered": "freeze_swarm"},
            {"title": "دفقة أخطاء 2", "severity": "critical",
             "action_triggered": "freeze_swarm"},  # نفس الإجراء مرة ثانية
        ]
        try:
            toggle_auto_action("freeze_swarm", True)
            applied = apply_auto_actions(alerts_two)
        finally:
            toggle_auto_action("freeze_swarm", False)
        # الإجراء لا يُطبَّق مرتين على تنبيهين متطابقين
        assert len(applied) == 1
        assert applied[0]["id"] == "freeze_swarm"
        assert "حدث تجميد السرب" in applied[0]["detail"]


class TestSnapshot:
    def test_snapshot_keys(self):
        from ai.unified_swarm_dashboard import unified_dashboard_snapshot
        snap = unified_dashboard_snapshot()
        assert set(snap) >= {
            "generated_at", "agents", "swarm", "long_horizon",
            "alerts", "alert_rules", "auto_actions"}
        assert isinstance(snap["agents"].get("counts"), dict)
        assert isinstance(snap["swarm"].get("history"), list)
        assert isinstance(snap["long_horizon"].get("tasks"), list)
        assert isinstance(snap["alerts"], list)
        assert len(snap["alert_rules"]) >= 3
        assert len(snap["auto_actions"]) >= 3

    def test_snapshot_defensive_on_empty_state(self, monkeypatch):
        from ai.unified_swarm_dashboard import unified_dashboard_snapshot
        # عزل ناقل الأحداث المشترك (خارج session_state) من أي أحداث
        # متبقية من اختبارات أخرى في نفس العملية — مثل أحداث tester-agent
        # التي يطلقها اختبار REST API.
        from ai import agent_event_bus as bus
        monkeypatch.setattr(bus, "get_events", lambda limit=80: [])
        # لا أحداث ولا سجلات — اللقطة يجب ألا تكسر
        snap = unified_dashboard_snapshot()
        assert snap["agents"]["counts"] == {
            "alive": 0, "done": 0, "failed": 0, "slow": 0, "stale": 0}
        assert snap["swarm"]["total"] == 0


class TestPerformance:
    """مؤشرات أداء النظام ووقت الاستجابة — دون اعتماديات خارجية."""

    def test_system_performance_keys(self):
        from ai.unified_swarm_dashboard import system_performance
        perf = system_performance()
        required = {"memory_used_mb", "memory_total_mb", "memory_percent",
                    "load_1m", "peak_rss_mb"}
        assert required.issubset(set(perf))
        # على Linux في بيئة الاختبار تُقرأ القيم فعليًا من /proc
        if platform.system().lower() == "linux":
            assert perf["memory_used_mb"] is not None
            assert perf["memory_used_mb"] > 0
            assert perf["memory_total_mb"] is not None
            assert perf["memory_percent"] is not None
            assert perf["memory_percent"] <= 100.0
            assert perf["load_1m"] is not None
            assert perf["peak_rss_mb"] is not None and perf["peak_rss_mb"] > 0

    def test_response_times_default(self):
        from ai.unified_swarm_dashboard import response_times
        rt = response_times(events=[])
        assert rt["count"] == 0
        assert rt["slow_count"] == 0
        assert rt["avg_ms"] == 0.0
        assert rt["max_ms"] == 0.0
        assert rt["slow_ms_threshold"] == 12000.0

    def test_response_times_with_slow_events(self):
        from ai.unified_swarm_dashboard import response_times
        now = 1700000000.0
        events = [
            {"duration_ms": 500.0, "ts": now, "status": "done"},
            {"duration_ms": 25000.0, "ts": now + 1, "status": "done"},  # بطيء
        ]
        rt = response_times(events=events)
        assert rt["count"] == 2
        assert rt["slow_count"] == 1
        assert rt["max_ms"] == 25000.0
        assert rt["avg_ms"] == 12750.0

    def test_response_times_threshold_follows_rule(self):
        from ai.unified_swarm_dashboard import (
            response_times, update_alert_rule)
        try:
            update_alert_rule("slow_agent", value=1000.0)
            events = [{"duration_ms": 2000.0, "ts": 1.0, "status": "done"}]
            rt = response_times(events=events)
            assert rt["slow_ms_threshold"] == 1000.0
            assert rt["slow_count"] == 1
        finally:
            update_alert_rule("slow_agent", value=12000.0)

    def test_agent_last_response_ms(self):
        from ai.unified_swarm_dashboard import agents_overview
        events = [
            {"duration_ms": 3500.0, "ts": 1.0, "status": "done",
             "agent_id": "writer"},
        ]
        overview = agents_overview(events=events)
        assert overview["agents"]["writer"]["last_response_ms"] == 3500.0
        overview_empty = agents_overview(events=[])
        assert overview_empty["agents"] == {}

    def test_snapshot_includes_performance(self):
        from ai.unified_swarm_dashboard import unified_dashboard_snapshot
        snap = unified_dashboard_snapshot()
        perf = snap["performance"]
        assert isinstance(perf, dict)
        sys_perf = perf["system"]
        rt = perf["response_times"]
        required_sys = {"memory_used_mb", "memory_total_mb",
                        "memory_percent", "load_1m", "peak_rss_mb"}
        assert required_sys.issubset(set(sys_perf))
        assert isinstance(rt, dict)
        assert {"count", "avg_ms", "max_ms", "last_ms", "slow_count",
                "slow_ms_threshold"}.issubset(set(rt))

class TestPerformanceTimeline:
    """السلسلة الزمنية للرسوم البيانية التفاعلية (ذاكرة/استجابة)."""

    def test_timeline_rows_and_keys(self, monkeypatch):
        from ai.unified_swarm_dashboard import (
            performance_timeline,
            reset_performance_timeline,
        )
        reset_performance_timeline()
        # cache TTL 5 ثوانٍ — صف واحد في كل استدعاء لنفس الإطار الزمني
        rows1 = performance_timeline(limit=60)
        assert rows1  # صف واحد على الأقل (عينة حية)
        row = rows1[-1]
        required = {"ts", "memory_mb", "memory_percent", "peak_rss_mb",
                    "load_1m", "avg_ms", "last_ms", "max_ms", "slow_count",
                    "event_count"}
        assert required.issubset(set(row))
        # العينة نفسها لا تتكرر أثناء إطار الـ 5 ثوانٍ
        rows2 = performance_timeline(limit=60)
        assert len(rows2) == 1
        # تجاوز صلاحية الكاش يدويًا يعيد التقاط عينة جديدة (reset يجعل
        # expires_at=0.0 فيلتقط عينة تالية)
        import ai.unified_swarm_dashboard as mod
        reset_performance_timeline()
        rows3 = performance_timeline(limit=60)
        assert len(rows3) == 1
        rows4 = performance_timeline(limit=60)
        assert len(rows4) == 1
        reset_performance_timeline()

    def test_timeline_limit(self, monkeypatch):
        from ai.unified_swarm_dashboard import (
            performance_timeline,
            reset_performance_timeline,
        )
        reset_performance_timeline()
        import ai.unified_swarm_dashboard as mod
        mod._TIMELINE_CACHE["expires_at"] = 0.0
        # تعبئة كذا صفًا ثم فحص أن الحد لا يتجاوز _TIMELINE_LIMIT
        for _ in range(5):
            mod._TIMELINE_CACHE["expires_at"] = 0.0
            performance_timeline(limit=60)
        assert len(mod._TIMELINE_CACHE["rows"]) <= mod._TIMELINE_LIMIT
        reset_performance_timeline()

    def test_timeline_empty_values_safe(self):
        from ai.unified_swarm_dashboard import performance_timeline
        # قيم None مسموحة (ذاكرة غير مقروءة) لكن الدالة لا ترمي أبدًا
        rows = performance_timeline(limit=10)
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    def test_timeline_epoch_float_recorded(self):
        from ai.unified_swarm_dashboard import (
            performance_timeline,
            reset_performance_timeline,
        )
        reset_performance_timeline()
        rows = performance_timeline(limit=60)
        assert rows and isinstance(rows[-1].get("epoch_float"), float)
        assert rows[-1]["epoch_float"] > 0
        reset_performance_timeline()


class TestTimelineRangeFiltering:
    """تصفية النطاق الزمني للرسوم البيانية (النطاقات المسماة + المخصصة)."""

    def test_filter_named_range(self, monkeypatch):
        from ai import unified_swarm_dashboard as mod
        now = 1700000000.0
        monkeypatch.setattr(mod.time, "time", lambda: now)
        rows = [
            {"ts": "00:00:01", "epoch_float": now - 3600},
            {"ts": "00:00:02", "epoch_float": now - 600},   # ضمن 15m
            {"ts": "00:00:03", "epoch_float": now - 60},    # ضمن 15m
            {"ts": "00:00:04", "epoch_float": now + 10},    # في المستقبل القريب
        ]
        kept = mod.filter_timeline(rows, range_name="15m")
        epochs = [r["epoch_float"] for r in kept]
        # صف الآن+10 خارج النطاق (نهايته الآن+1) فلا يظهر
        assert epochs == [now - 600, now - 60]

    def test_filter_custom_range(self, monkeypatch):
        from ai import unified_swarm_dashboard as mod
        from datetime import datetime, timezone
        now = 1700000000.0
        monkeypatch.setattr(mod.time, "time", lambda: now)
        rows = [
            {"ts": "00:00:01", "epoch_float": now - 7200},
            {"ts": "00:00:02", "epoch_float": now - 1800},
            {"ts": "00:00:03", "epoch_float": now - 300},
        ]
        # من (الآن - 2500 ثانية) إلى (الآن - 500 ثانية)
        frm = datetime.utcfromtimestamp(now - 2500).strftime(
            "%Y-%m-%dT%H:%M:%S")
        to = datetime.utcfromtimestamp(now - 500).strftime("%Y-%m-%dT%H:%M:%S")
        kept = mod.filter_timeline(rows, from_iso=frm, to_iso=to)
        assert [r["epoch_float"] for r in kept] == [now - 1800]

    def test_filter_fallback_to_last_row(self, monkeypatch):
        from ai import unified_swarm_dashboard as mod
        now = 1700000000.0
        monkeypatch.setattr(mod.time, "time", lambda: now)
        rows = [{"ts": "00:00:01", "epoch_float": 100.0}]
        # نطاق قديم جدًا لا يحتوي أي صف — يعود بآخر صف وحيد
        kept = mod.filter_timeline(rows, from_iso="1970-01-02T00:00:00",
                                   to_iso="1970-01-02T00:01:00")
        assert kept == rows
        # صفوف بلا طابع زمني صالح لا تدخل القائمة النشطة، لكن
        # fallback الدفاعي يعيد الصف الأخير الوحيد (السلوك المقصود)
        fallback = mod.filter_timeline([{"ts": "??"}], range_name="5m")
        assert fallback == [{"ts": "??"}]

    def test_filter_invalid_range_passthrough(self, monkeypatch):
        from ai import unified_swarm_dashboard as mod
        now = 1700000000.0
        monkeypatch.setattr(mod.time, "time", lambda: now)
        rows = [{"ts": "00:00:01", "epoch_float": 1.0}]
        # نطاق غير معروف + قيمة من غير صالحة = بدون تصفية
        kept = mod.filter_timeline(rows, range_name="99z",
                                   from_iso="not-a-date")
        assert kept == rows
