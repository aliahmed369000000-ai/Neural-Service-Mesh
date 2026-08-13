# -*- coding: utf-8 -*-
"""نظام التنبيهات القابل للتخصيص (Custom Alert Rules).

يوحّد تحليل تنبيهات الشبكة في طبقة قواعد قابلة للتعديل من المستخدم أو من
ملف التكوين config/alert_rules.json، مع حماية من التكرار عبر فترة تبريد
(cooldown) لكل قاعدة، وسجل تنبيهات مركزي يمكن للوحة المراقبة عرضه.

القواعد الافتراضية:
- failure_direct: أي حدث فشل من وكيل يتحول لتنبيه حرج.
- slow_response: استجابة أبطأ من عتبة مللي ثانية (حرجة عند ضعف العتبة).
- repeated_errors: عدد أخطاء متتالية لوكيل واحد يتجاوز حدًا.
- swarm_failure_rate: نسبة فشل السرب في آخر نافذة مهام تتجاوز حدًا.
- agent_degraded: أداء وكيل حرج (نسبة فشل عالية مع سجل مهام كافٍ).
- swarm_excluded: وكيل استُبعد مؤقتًا من السرب.
- congestion: عدد وكلاء متزامنين قيد التشغيل يتجاوز حدًا.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── ثوابت ──────────────────────────────────────────────────────────────────
_CONFIG_PATH = Path("config") / "alert_rules.json"
_MAX_ALERT_LOG = 400
_RULE_LOCK = threading.Lock()

DEFAULT_RULES: Dict[str, Dict[str, Any]] = {
    "failure_direct": {
        "enabled": True,
        "severity": "critical",
        "cooldown_minutes": 5,
        "message": "فشل مباشر لدى الوكيل: {agent}",
        "detail_tpl": "{detail}",
    },
    "slow_response": {
        "enabled": True,
        "severity": "warning",
        "cooldown_minutes": 15,
        "threshold_ms": 12000.0,
        "critical_multiplier": 2.0,
        "message": "استجابة بطيئة من الوكيل: {agent} ({ms:.0f} مللي ثانية)",
        "detail_tpl": "زمن التنفيذ {ms:.0f} مللي ثانية (العتبة {threshold:.0f})",
    },
    "repeated_errors": {
        "enabled": True,
        "severity": "critical",
        "cooldown_minutes": 30,
        "min_errors": 2,
        "message": "تكرار أخطاء لدى الوكيل: {agent} ({count} خطأ)",
        "detail_tpl": "سُجلت {count} أخطاء متتالية في السجل الحالي",
    },
    "swarm_failure_rate": {
        "enabled": True,
        "severity": "critical",
        "cooldown_minutes": 20,
        "failure_rate_threshold": 0.5,
        "min_tasks": 3,
        "message": "نسبة فشل السرب مرتفعة ({rate:.0%})",
        "detail_tpl": "{failures} فشل من {tasks} مهمة في آخر {window} حدثًا",
    },
    "agent_degraded": {
        "enabled": True,
        "severity": "warning",
        "cooldown_minutes": 60,
        "failure_rate_threshold": 0.75,
        "min_tasks": 2,
        "message": "أداء وكيل حرج: {agent} (نسبة فشل {rate:.0%})",
        "detail_tpl": "فشل في {failures} من {tasks} مهمة — مرشح للاستبعاد",
    },
    "swarm_excluded": {
        "enabled": True,
        "severity": "info",
        "cooldown_minutes": 60,
        "message": "استبعاد مؤقت للوكيل: {agent}",
        "detail_tpl": "استُبعد من التوجيه بسبب تكرار الفشل",
    },
    "congestion": {
        "enabled": True,
        "severity": "warning",
        "cooldown_minutes": 10,
        "max_concurrent": 3,
        "message": "ازدحام في التنفيذ ({count} وكلاء متزامنين)",
        "detail_tpl": "{count} وكلاء في حالة تشغيل متزامنة",
    },
}


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    """يقرأ ملف JSON بأمان — يرجع None عند أي خلل (ملف مشوه أو مفقود)."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _load_rules() -> Dict[str, Dict[str, Any]]:
    """يقرأ قواعد التنبيه من config/alert_rules.json مع fallback للقيم الافتراضية."""
    data = _safe_load_json(_CONFIG_PATH)
    rules: Dict[str, Dict[str, Any]] = {}
    for name, default in DEFAULT_RULES.items():
        entry = dict(default)
        custom = data.get(name) if data else None
        if isinstance(custom, dict):
            for key, value in custom.items():
                default_value = default.get(key)
                if isinstance(default_value, (int, float)) and isinstance(
                    value, (int, float)
                ):
                    entry[key] = type(default_value)(value)
                elif isinstance(default_value, bool) and isinstance(value, bool):
                    entry[key] = value
                elif isinstance(default_value, str) and isinstance(value, str):
                    entry[key] = value
        rules[name] = entry
    return rules


_rules_cache: Optional[Dict[str, Dict[str, Any]]] = None
_rules_cache_time: float = 0.0


def get_rules(force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
    """يعيد قواعد التنبيه مع تخزين مؤقت بسيط (يُحدَّث كل 10 ثوانٍ)."""
    import time

    now = time.time()
    global _rules_cache, _rules_cache_time
    if _rules_cache is None or force_reload or now - _rules_cache_time >= 10:
        _rules_cache = _load_rules()
        _rules_cache_time = now
    return _rules_cache


def reset_rules_cache() -> None:
    """يلغي التخزين المؤقت — مفيد للاختبار ولحفظ تخصيص جديد من الواجهة."""
    global _rules_cache, _rules_cache_time
    _rules_cache = None
    _rules_cache_time = 0.0


def save_custom_rules(custom: Dict[str, Any]) -> bool:
    """يحفظ تخصيص المستخدم في config/alert_rules.json ثم يُعيد تحميل القواعد."""
    try:
        merged = dict(DEFAULT_RULES)
        for name in merged:
            entry = dict(merged[name])
            override = custom.get(name)
            if isinstance(override, dict):
                for key, value in override.items():
                    if key in entry and (
                        isinstance(value, type(entry[key]))
                        or (
                            isinstance(value, (int, float))
                            and isinstance(entry[key], (int, float))
                        )
                    ):
                        entry[key] = value
            merged[name] = entry
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _CONFIG_PATH.open("w", encoding="utf-8") as handle:
            json.dump(
                merged,
                handle,
                ensure_ascii=False,
                indent=2,
            )
        reset_rules_cache()
        return True
    except Exception:
        return False


# ── فترة التبريد والسجل ────────────────────────────────────────────────────
_fired: Dict[str, float] = {}


def _now_seconds() -> float:
    import time

    return time.time()


def check_cooldown(rule_name: str, rule: Dict[str, Any], now: float) -> bool:
    """يرجع True إذا كان مسموحًا بإطلاق القاعدة (تجاوز التبريد أو غير مُطلقة سابقًا)."""
    cooldown = float(rule.get("cooldown_minutes", 0))
    if cooldown <= 0:
        return True
    last = _fired.get(rule_name)
    return last is None or now - last >= cooldown * 60


def mark_fired(rule_name: str) -> None:
    _fired[rule_name] = _now_seconds()


def reset_cooldowns() -> None:
    _fired.clear()


# ── سجل التنبيهات ──────────────────────────────────────────────────────────
_alert_log: List[Dict[str, Any]] = []
_log_lock = threading.Lock()


def add_alert(alert: Dict[str, Any]) -> Dict[str, Any]:
    """يضيف تنبيهًا إلى السجل المركزي ويرجعه مع معرف متسلسل."""
    with _log_lock:
        alert = dict(alert)
        alert["log_id"] = len(_alert_log) + 1
        alert.setdefault("timestamp", _format_now())
        _alert_log.append(alert)
        if len(_alert_log) > _MAX_ALERT_LOG:
            del _alert_log[: len(_alert_log) - _MAX_ALERT_LOG]
    return alert


def get_alert_log(limit: int = 50) -> List[Dict[str, Any]]:
    return list(_alert_log[-limit:])


def clear_alert_log() -> None:
    with _log_lock:
        _alert_log.clear()


def _format_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── المحلل الرئيسي ─────────────────────────────────────────────────────────

def _swarm_stats(events: List[Dict[str, Any]], window: int = 200) -> Dict[str, Any]:
    """يحصي مهام السرب من أحداث البدء والانتهاء في آخر نافذة أحداث."""
    from ai.adaptive_swarm import _FAILURE_EVENTS, _SUCCESS_EVENTS

    rows = events[-window:] if window < len(events) else list(events)
    tasks = 0
    failures = 0
    for row in rows:
        if row.get("event_type") in _SUCCESS_EVENTS:
            tasks += 1
        elif row.get("event_type") in _FAILURE_EVENTS:
            tasks += 1
            failures += 1
    return {"tasks": tasks, "failures": failures}


def check_alert_rules(
    events: Optional[List[Dict[str, Any]]] = None,
    *,
    rules: Optional[Dict[str, Dict[str, Any]]] = None,
    excluded_agents: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """يحلل الأحداث وفق القواعد المخصصة ويرجع تنبيهات جديدة فقط (مع التبريد).

    ينشر كل تنبيه في السجل المركزي عبر add_alert، ويستبعد التنبيهات المكررة
    داخل فترة التبريد الخاصة بكل قاعدة.
    """
    from ai.adaptive_swarm import _FAILURE_EVENTS

    rules = rules if rules is not None else get_rules()
    rows = list(events if events is not None else [])
    if not rows and not excluded_agents:
        return []
    now = _now_seconds()
    alerts: List[Dict[str, Any]] = []
    failures_by_agent: Dict[str, int] = {}
    for row in rows:
        if row.get("status") == "error":
            _aid = row.get("agent_id") or "orchestrator"
            failures_by_agent[_aid] = failures_by_agent.get(_aid, 0) + 1
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        latest[row.get("agent_id") or "orchestrator"] = row

    with _RULE_LOCK:
        # ── failure_direct ──
        rule = rules.get("failure_direct", DEFAULT_RULES["failure_direct"])
        if rule.get("enabled"):
            for row in rows:
                if row.get("status") == "error" and check_cooldown(
                    "failure_direct", rule, now
                ):
                    agent = row.get("title") or row.get("agent_id") or "المدير"
                    alert = {
                        "severity": rule.get("severity", "critical"),
                        "rule": "failure_direct",
                        "title": rule["message"].format(agent=agent),
                        "detail": rule["detail_tpl"].format(
                            detail=row.get("detail") or "حدث خطأ غير موصوف"
                        ),
                        "event": row,
                    }
                    mark_fired("failure_direct")
                    alerts.append(add_alert(alert))

        # ── slow_response ──
        rule = rules.get("slow_response", DEFAULT_RULES["slow_response"])
        if rule.get("enabled"):
            threshold = float(rule.get("threshold_ms", 12000))
            multiplier = float(rule.get("critical_multiplier", 2.0))
            for row in rows:
                duration = row.get("duration_ms")
                if duration is None:
                    continue
                duration_f = float(duration)
                if duration_f < threshold:
                    continue
                if not check_cooldown("slow_response", rule, now):
                    continue
                agent = row.get("title") or row.get("agent_id") or "المدير"
                severity = (
                    "critical" if duration_f >= threshold * multiplier
                    else rule.get("severity", "warning")
                )
                alert = {
                    "severity": severity,
                    "rule": "slow_response",
                    "title": rule["message"].format(agent=agent, ms=duration_f),
                    "detail": rule["detail_tpl"].format(
                        ms=duration_f, threshold=threshold
                    ),
                    "event": row,
                }
                mark_fired("slow_response")
                alerts.append(add_alert(alert))

        # ── repeated_errors ──
        rule = rules.get("repeated_errors", DEFAULT_RULES["repeated_errors"])
        if rule.get("enabled"):
            for agent_id, count in failures_by_agent.items():
                min_errors = int(rule.get("min_errors", 2))
                if count < min_errors or not check_cooldown(
                    "repeated_errors", rule, now
                ):
                    continue
                label = next(
                    (r.get("title") for r in rows if r.get("agent_id") == agent_id),
                    agent_id,
                )
                alert = {
                    "severity": rule.get("severity", "critical"),
                    "rule": "repeated_errors",
                    "title": rule["message"].format(agent=label, count=count),
                    "detail": rule["detail_tpl"].format(count=count),
                    "event": {"agent_id": agent_id},
                }
                mark_fired("repeated_errors")
                alerts.append(add_alert(alert))

        # ── swarm_failure_rate ──
        rule = rules.get("swarm_failure_rate", DEFAULT_RULES["swarm_failure_rate"])
        if rule.get("enabled"):
            stats = _swarm_stats(list(events) if events is not None else rows)
            tasks = stats["tasks"]
            failures = stats["failures"]
            min_tasks = int(rule.get("min_tasks", 3))
            threshold = float(rule.get("failure_rate_threshold", 0.5))
            window = min(int(stats.get("window", 200)), len(rows)) if "window" in stats else len(rows)
            if (
                tasks >= min_tasks
                and failures / tasks >= threshold
                and check_cooldown("swarm_failure_rate", rule, now)
            ):
                alert = {
                    "severity": rule.get("severity", "critical"),
                    "rule": "swarm_failure_rate",
                    "title": rule["message"].format(rate=failures / tasks),
                    "detail": rule["detail_tpl"].format(
                        failures=failures, tasks=tasks, window=window
                    ),
                    "event": {"window": window, "tasks": tasks, "failures": failures},
                }
                mark_fired("swarm_failure_rate")
                alerts.append(add_alert(alert))

        # ── agent_degraded ──
        rule = rules.get("agent_degraded", DEFAULT_RULES["agent_degraded"])
        if rule.get("enabled"):
            for agent_id, count in failures_by_agent.items():
                min_tasks = int(rule.get("min_tasks", 2))
                if count < min_tasks:
                    continue
                total = sum(
                    1 for r in rows
                    if (r.get("agent_id") or "orchestrator") == agent_id
                    and r.get("event_type") in _FAILURE_EVENTS
                ) or count
                rate = count / max(total, 1)
                threshold = float(rule.get("failure_rate_threshold", 0.75))
                if rate < threshold or not check_cooldown(
                    "agent_degraded", rule, now
                ):
                    continue
                label = next(
                    (r.get("title") for r in rows if r.get("agent_id") == agent_id),
                    agent_id,
                )
                alert = {
                    "severity": rule.get("severity", "warning"),
                    "rule": "agent_degraded",
                    "title": rule["message"].format(agent=label, rate=rate),
                    "detail": rule["detail_tpl"].format(
                        failures=count, tasks=total
                    ),
                    "event": {"agent_id": agent_id, "failure_rate": rate},
                }
                mark_fired("agent_degraded")
                alerts.append(add_alert(alert))

        # ── swarm_excluded ──
        rule = rules.get("swarm_excluded", DEFAULT_RULES["swarm_excluded"])
        if rule.get("enabled") and excluded_agents:
            for _exc in excluded_agents:
                agent = str(_exc[0]) if isinstance(_exc, (list, tuple)) else str(_exc)
                if not check_cooldown(
                    f"swarm_excluded:{agent}",
                    {**rule, "cooldown_minutes": rule.get("cooldown_minutes", 60)},
                    now,
                ):
                    continue
                alert = {
                    "severity": rule.get("severity", "info"),
                    "rule": "swarm_excluded",
                    "title": rule["message"].format(agent=agent),
                    "detail": rule["detail_tpl"].format(),
                    "event": {"agent_id": agent},
                }
                mark_fired(f"swarm_excluded:{agent}")
                alerts.append(add_alert(alert))

        # ── congestion ──
        rule = rules.get("congestion", DEFAULT_RULES["congestion"])
        if rule.get("enabled"):
            running = [r for r in latest.values() if r.get("status") == "running"]
            max_concurrent = int(rule.get("max_concurrent", 3))
            if (
                len(running) >= max_concurrent
                and check_cooldown("congestion", rule, now)
            ):
                alert = {
                    "severity": rule.get("severity", "warning"),
                    "rule": "congestion",
                    "title": rule["message"].format(count=len(running)),
                    "detail": rule["detail_tpl"].format(count=len(running)),
                    "event": {},
                }
                mark_fired("congestion")
                alerts.append(add_alert(alert))

    return alerts
