# -*- coding: utf-8 -*-
"""لوحة السرب الموحدة (Unified Swarm Dashboard).

طبقة تجميع واحدة تعرض حالة السرب كاملاً:
- الوكلاء: أحياء / منتهون / فاشلون (من ناقل أحداث الوكلاء agent_event_bus)
- مهام السرب: تاريخ التنفيذ المحفوظ في SQLite عبر mesh bundle
- المهام طويلة الأمد: LongHorizonTaskManager
- التنبيهات: من ناقل الأحداث مع عتبات قابلة للتخصيص تُحفظ في JSON

كل دالة هنا صافية قدر الإمكان (بدون مفاتيح API) كي تبقى قابلة للاختبار.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── مسارات البيانات ──────────────────────────────────────────────
_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
_NSM_DATA = _ROOT / ".nsm_data"

def _dashboard_data_dir() -> Path:
    data_dir = _NSM_DATA / "swarm_dashboard"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

_ALERT_RULES_PATH = lambda: _dashboard_data_dir() / "alert_rules.json"
_AUTO_ACTIONS_PATH = lambda: _dashboard_data_dir() / "auto_actions.json"


def _copy_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    """نسخة سطحية لقاعدة تنبيه مع تحويل القيم غير القابلة للتسلسل."""
    return dict(rule)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# ── قواعد التنبيهات القابلة للتخصيص ─────────────────────────────
_DEFAULT_ALERT_RULES = [
    {
        "id": "slow_agent",
        "enabled": True,
        "label": "وكيل بطيء",
        "description": "عندما يتجاوز زمن تنفيذ وكيل العتبة (مللي ثانية)",
        "kind": "slow_threshold_ms",
        "value": 12000,
        "severity": "warning",
        "auto_action": "restart_role",
    },
    {
        "id": "stale_agent",
        "enabled": True,
        "label": "وكيل مختنق",
        "description": "عندما يتوقف وكيل عن الاستجابة بعد عتبة (ثانية)",
        "kind": "stale_threshold_s",
        "value": 45,
        "severity": "warning",
        "auto_action": "restart_role",
    },
    {
        "id": "burst_errors",
        "enabled": True,
        "label": "دفقة أخطاء",
        "description": "عندما تتجاوز نسبة الأخطاء في آخر الأحداث العتبة",
        "kind": "error_ratio",
        "value": 0.2,
        "severity": "critical",
        "auto_action": "freeze_swarm",
    },
]

_DEFAULT_AUTO_ACTIONS = [
    {
        "id": "restart_role",
        "enabled": True,
        "label": "إعادة تشغيل دور الوكيل",
        "description": "يفرّغ ذاكرة الدور ويرسل إشارة إعادة تهيئة (تسجيل حدث recovery)",
    },
    {
        "id": "freeze_swarm",
        "enabled": False,
        "label": "تجميد السرب مؤقتاً",
        "description": "يجمّد تنفيذ السرب حتى إزالة التجميد يدوياً (تسجيل حدث freeze)",
    },
    {
        "id": "notify_discord",
        "enabled": False,
        "label": "إشعار ديسكورد",
        "description": "يرسل ملخص التنبيه إلى قناة ديسكورد إذا كانت مهيأة",
    },
]


def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except (json.JSONDecodeError, OSError):
        pass
    return default


def _save_json(path: Path, value: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass


def list_alert_rules() -> List[Dict[str, Any]]:
    """قواعد التنبيهات القابلة للتخصيص (مع القيم الافتراضية).

    يعيد نسخة جديدة دائمًا كي لا يعدّل المستدعي الافتراضيات في الموقع.
    """
    saved = _load_json(_ALERT_RULES_PATH(), None)
    rules = [_copy_rule(r) for r in (saved if saved is not None
                                     else _DEFAULT_ALERT_RULES)]
    ids = {r["id"] for r in rules}
    for rule in _DEFAULT_ALERT_RULES:
        if rule["id"] not in ids:
            rules.append(_copy_rule(rule))
    return rules


def update_alert_rule(rule_id: str, **changes: Any) -> Optional[Dict[str, Any]]:
    """تحديث قاعدة تنبيه واحدة (enabled/value/label/auto_action)."""
    rules = list_alert_rules()
    for rule in rules:
        if rule["id"] == rule_id:
            rule.update({k: v for k, v in changes.items() if k in {
                "enabled", "value", "label", "description", "severity",
                "auto_action", "kind"}})
            _save_json(_ALERT_RULES_PATH(), rules)
            return rule
    return None


def list_auto_actions() -> List[Dict[str, Any]]:
    """الإجراءات التلقائية المتاحة مع حالة تفعيلها.

    يعيد نسخة جديدة دائمًا كي لا يعدّل المستدعي الافتراضيات في الموقع.
    """
    saved = _load_json(_AUTO_ACTIONS_PATH(), None)
    actions = [dict(a) for a in (saved if saved is not None
                                 else _DEFAULT_AUTO_ACTIONS)]
    ids = {a["id"] for a in actions}
    for action in _DEFAULT_AUTO_ACTIONS:
        if action["id"] not in ids:
            actions.append(dict(action))
    return actions


def toggle_auto_action(action_id: str, enabled: bool) -> Optional[Dict[str, Any]]:
    """تفعيل/إيقاف إجراء تلقائي."""
    actions = list_auto_actions()
    for action in actions:
        if action["id"] == action_id:
            action["enabled"] = bool(enabled)
            _save_json(_AUTO_ACTIONS_PATH(), actions)
            return action
    return None


# ── مؤشرات الأداء ──────────────────────────────────────────────
def system_performance() -> Dict[str, Any]:
    """مؤشرات أداء النظام دون اعتماديات خارجية (stdlib فقط).

    يقرأ الذاكرة من /proc/self/status (VmRSS الفعلية) و/proc/meminfo
    (الإجمالي/المتاح)، والحمل من os.getloadavg، وأقصى RSS من
    resource.getrusage. كل قراءة تتسامح مع أي فشل فردي.
    """
    result: Dict[str, Any] = {
        "memory_used_mb": None, "memory_total_mb": None,
        "memory_percent": None, "load_1m": None, "peak_rss_mb": None,
    }
    # VmRSS الحالية (KB في /proc/self/status)
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    result["memory_used_mb"] = round(
                        int(line.split()[1]) / 1024.0, 1)
                    break
    except (OSError, IndexError, ValueError):
        pass
    # الذاكرة الإجمالية والمتاحة (kB في /proc/meminfo)
    try:
        info: Dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2 and parts[-1] == "kB":
                    info[parts[0].rstrip(":")] = int(parts[1])
        total_kb = info.get("MemTotal")
        available_kb = info.get("MemAvailable")
        if total_kb:
            result["memory_total_mb"] = round(total_kb / 1024.0, 1)
            used_mb = result.get("memory_used_mb")
            if available_kb is not None:
                result["memory_percent"] = round(
                    100.0 * (total_kb - available_kb) / total_kb, 1)
            elif used_mb is not None:
                result["memory_percent"] = round(
                    100.0 * (used_mb * 1024.0) / total_kb, 1)
    except (OSError, IndexError, ValueError):
        pass
    # الحمل والنسبة القصوى
    try:
        result["load_1m"] = round(os.getloadavg()[0], 2)
    except OSError:
        pass
    try:
        import resource as _res
        result["peak_rss_mb"] = round(
            _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:  # pragma: no cover — macOS فقط قد تختلف الوحدة
        pass
    return result


def response_times(events: Optional[List[Dict[str, Any]]] = None,
                   limit: int = 80) -> Dict[str, Any]:
    """مؤشرات وقت الاستجابة للوكلاء من ناقل الأحداث.

    يجمع average/max/last من performance_summary، ويحسب نسبة الوكلاء
    البطيئين وفق عتبة البطء في قواعد التنبيهات، وعدد الأحداث الكلي.
    """
    result: Dict[str, Any] = {"count": 0, "avg_ms": None, "max_ms": None,
                              "last_ms": None, "slow_count": 0,
                              "slow_ms_threshold": 12000.0}
    try:
        from ai.agent_event_bus import get_events, performance_summary
        if events is None:
            events = get_events(limit)
        perf = performance_summary(events) or {}
        result["count"] = int(perf.get("count", 0))
        result["avg_ms"] = perf.get("avg_ms")
        result["max_ms"] = perf.get("max_ms")
        result["last_ms"] = perf.get("last_ms")
        slow_rule = next((r for r in list_alert_rules()
                          if r.get("enabled")
                          and r.get("kind") == "slow_threshold_ms"), None)
        result["slow_ms_threshold"] = float(
            slow_rule["value"] if slow_rule else 12000.0)
        for event in events or []:
            if event.get("duration_ms") is not None:
                try:
                    if float(event["duration_ms"]) >= result["slow_ms_threshold"]:
                        result["slow_count"] += 1
                except (TypeError, ValueError):
                    pass
    except ImportError:  # pragma: no cover
        pass
    return result


# ── الحالة الموحدة ──────────────────────────────────────────────
def _safe(func, *args, **kwargs):  # noqa: N802 — اسم قصير خاص
    """تنفيذ آمن: أي استيراد أو تنفيذ فاشل لا يكسر اللوحة."""
    try:
        return func(*args, **kwargs)
    except Exception:  # pragma: no cover - حماية دفاعية للوحة
        return None


def agents_overview(events: Optional[List[Dict[str, Any]]] = None,
                    limit: int = 80) -> Dict[str, Any]:
    """نظرة عامة على الوكلاء: أحياء / منتهون / فاشلون / بطيئون."""
    result: Dict[str, Any] = {"agents": {}, "counts": {
        "alive": 0, "done": 0, "failed": 0, "slow": 0, "stale": 0}}
    try:
        from ai.agent_event_bus import (
            current_agent_states, get_events, performance_summary)
        if events is None:
            events = get_events(limit)
        states = current_agent_states(events) or {}
        perf = performance_summary(events) or {}
        slow_ms = float(perf.get("slow_threshold_ms", 12000))
        for agent_id, row in states.items():
            status = str(row.get("status") or "waiting")
            row_view = dict(row)
            if status == "running":
                result["counts"]["alive"] += 1
            elif status == "error":
                result["counts"]["failed"] += 1
            elif status == "done":
                result["counts"]["done"] += 1
            if row.get("duration_ms") is not None:
                try:
                    if float(row["duration_ms"]) >= slow_ms:
                        result["counts"]["slow"] += 1
                        row_view["is_slow"] = True
                except (TypeError, ValueError):
                    pass
            # زمن آخر استجابة لكل وكيل (ms أو None)
            row_view["last_response_ms"] = row.get("duration_ms")
            result["agents"][agent_id] = row_view
    except Exception:  # pragma: no cover
        pass
    return result


def swarm_status() -> Dict[str, Any]:
    """حالة السرب: تاريخ التنفيذ المحفوظ في SQLite عبر mesh bundle."""
    result: Dict[str, Any] = {"history": [], "total": 0, "successful": 0,
                              "failed": 0, "mesh": None}
    try:
        from core.mesh_bundle import get_mesh_bundle
        bundle = get_mesh_bundle()
        history = _safe(bundle.history, 20) or []
        result["history"] = list(history)
        result["total"] = len(history)
        result["successful"] = sum(
            1 for h in history if (h.get("success") if isinstance(h, dict)
                                   else False))
        result["failed"] = sum(
            1 for h in history if (not h.get("success") if isinstance(h, dict)
                                   else False))
        result["mesh"] = _safe(bundle.summary)
    except Exception:  # pragma: no cover
        pass
    return result


def long_horizon_status(limit: int = 10) -> Dict[str, Any]:
    """المهام طويلة الأمد الحالية (قيد التنفيذ / مكتملة / ملغاة)."""
    result: Dict[str, Any] = {"tasks": [], "counts": {
        "pending": 0, "running": 0, "done": 0, "cancelled": 0}}
    try:
        from ai.long_horizon_tasks import get_long_horizon_manager
        manager = get_long_horizon_manager()
        tasks = manager.list_tasks(limit=limit)
        for task in tasks or []:
            status = str(task.get("status") if isinstance(task, dict) else task)
            if status in result["counts"]:
                result["counts"][status] += 1
            result["tasks"].append(task)
    except Exception:  # pragma: no cover
        pass
    return result


# ── التنبيهات مع القواعد المخصصة ────────────────────────────────
def evaluate_alerts(events: Optional[List[Dict[str, Any]]] = None,
                    limit: int = 80) -> List[Dict[str, Any]]:
    """تقييم التنبيهات وفق قواعد التنبيهات القابلة للتخصيص.

    يعيد قائمة تنبيهات (dict: id/title/severity/detail/action_triggered).
    إذا لم تتحقق أي قاعدة مخصصة يسقط إلى analyze_alerts الافتراضية.
    """
    try:
        from ai.agent_event_bus import analyze_alerts, get_events
    except ImportError:  # pragma: no cover
        return []
    if events is None:
        events = get_events(limit)
    rules = [r for r in list_alert_rules() if r.get("enabled")]
    thresholds = {r["kind"]: float(r["value"]) for r in rules
                  if r["kind"] in {"slow_threshold_ms", "stale_threshold_s"}}
    fallback = analyze_alerts(
        events,
        slow_threshold_ms=thresholds.get("slow_threshold_ms", 12000),
        stale_threshold_s=thresholds.get("stale_threshold_s", 45),
    ) if rules else []
    alerts: List[Dict[str, Any]] = [
        dict(alert) if isinstance(alert, dict) else {"title": str(alert)}
        for alert in list(fallback)]
    for alert in alerts:
        if "title" not in alert and alert.get("label"):
            alert["title"] = alert["label"]
    # تنبيه دفقة الأخطاء المخصص
    error_rule = next((r for r in rules if r["kind"] == "error_ratio"), None)
    if error_rule and events:
        errors = sum(1 for e in events if str(e.get("status")) == "error")
        ratio = errors / max(len(events), 1)
        if ratio >= float(error_rule["value"]):
            alerts.append({
                "id": "burst_errors",
                "title": "دفقة أخطاء",
                "severity": error_rule.get("severity", "critical"),
                "detail": (f"نسبة الأخطاء {ratio:.1%} تجاوزت العتبة "
                           f"{float(error_rule['value']):.0%} "
                           f"({errors} من {len(events)})"),
                "action_triggered": error_rule.get("auto_action"),
            })
    return alerts


def apply_auto_actions(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """تنفيذ الإجراءات التلقائية للتفعيل المطابقة للتنبيهات.

    يعيد سجل الإجراءات المنفذة (id/label/detail) مع تسجيل حدث recovery/
    freeze في ناقل الأحداث عند توفره.
    """
    applied: List[Dict[str, Any]] = []
    enabled_actions = {a["id"]: a for a in list_auto_actions()
                       if a.get("enabled")}
    action_ids = set()
    for alert in alerts:
        act_id = alert.get("action_triggered")
        if not act_id or act_id not in enabled_actions:
            continue
        if act_id in action_ids:
            continue
        action_ids.add(act_id)
        detail = _execute_action(act_id)
        applied.append({"id": act_id, "label": enabled_actions[act_id]["label"],
                        "detail": detail})
    return applied


def _execute_action(action_id: str) -> str:
    """تنفيذ إجراء تلقائي محلي: تسجيل حدث في ناقل الأحداث."""
    try:
        from ai.agent_event_bus import emit_event
        if action_id == "restart_role":
            emit_event(
                agent_id="swarm_dashboard",
                event_type="recovery",
                title="إعادة تشغيل دور الوكيل",
                detail="إجراء تلقائي من لوحة السرب الموحدة",
                status="info",
            )
            return "سُجّل حدث إعادة تهيئة الدور في ناقل الأحداث"
        if action_id == "freeze_swarm":
            emit_event(
                agent_id="swarm_dashboard",
                event_type="swarm_freeze",
                title="تجميد السرب",
                detail="إجراء تلقائي من لوحة السرب الموحدة",
                status="info",
            )
            return "سُجّل حدث تجميد السرب في ناقل الأحداث"
        if action_id == "notify_discord":
            # قناة ديسكورد اختيارية — إن وُجدت وحدة الإشعار تستخدمها،
            # وإلا تسجّل الحدث محلياً فقط دون تعطيل اللوحة.
            emit_event(
                agent_id="swarm_dashboard",
                event_type="discord_notify_attempt",
                title="محاولة إشعار ديسكورد",
                detail="إجراء تلقائي من لوحة السرب الموحدة — يُرسل إن كان "
                       "توكن الديسكورد مهيأً في أسرار التطبيق",
                status="info",
            )
            return "سُجّلت محاولة إشعار (تحتاج توكن ديسكورد مهيأً)"
    except ImportError:  # pragma: no cover
        pass
    return "لم ينفَّذ أي إجراء (الإجراء غير متوفر)"


# ── اللقطة الموحدة الكاملة ──────────────────────────────────────
def unified_dashboard_snapshot() -> Dict[str, Any]:
    """لقطة واحدة لكل حالة السرب — نقطة الدخول الرئيسية للوحة UI."""
    try:
        from ai.agent_event_bus import get_events
        events = get_events(80)
    except ImportError:  # pragma: no cover
        events = []
    return {
        "generated_at": _now_iso(),
        "agents": agents_overview(events),
        "swarm": swarm_status(),
        "long_horizon": long_horizon_status(),
        "performance": {
            "system": _safe(system_performance) or {"memory_used_mb": None,
                                                    "memory_total_mb": None,
                                                    "memory_percent": None,
                                                    "load_1m": None,
                                                    "peak_rss_mb": None},
            "response_times": response_times(events),
        },
        "alerts": evaluate_alerts(events),
        "alert_rules": list_alert_rules(),
        "auto_actions": list_auto_actions(),
    }
