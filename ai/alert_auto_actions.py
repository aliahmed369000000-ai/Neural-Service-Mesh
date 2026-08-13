# -*- coding: utf-8 -*-
"""الإجراءات التلقائية للتنبيهات (Alert Auto-Actions) وتشخيص الوكلاء الفاشلين.

طبقة استجابة تلقائية فوق نظام التنبيهات القابل للتخصيص (ai/alert_config):
عند إطلاق تنبيه تنفَّذ إجراءات آلية محدَّدة حسب القاعدة المُطلِقة:

| القاعدة المُطلِقة      | الإجراء التلقائي الافتراضي          |
|------------------------|-------------------------------------|
| agent_degraded         | تشخيص نمطي + تسجيل درس في الذاكرة الجماعية |
| repeated_errors        | استبعاد تلقائي مؤقت + تشخيص          |
| failure_direct         | تشخيص نمطي (بدون مفتاح API)          |
| swarm_failure_rate     | تسجيل درس جماعي + تصعيد              |
| slow_response          | تسجيل درس أداء                      |
| swarm_excluded         | إشعار متابعة الاستبعاد              |
| congestion             | تسجيل حدث تخفيف                      |

كل إجراء قابل للتفعيل/التعطيل وتحديد فترة تبريد خاصة به من ملف
config/auto_alert_actions.json أو من لوحة المراقبة مباشرة.

التشخيص النمطي مبني على مصنفات أنماط الأخطاء (نفس منهجية
agent_reflection) ويعمل محليًا بالكامل بدون أي مفتاح API.
"""

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── ثوابت ──────────────────────────────────────────────────────────────────
_ACTIONS_CONFIG_PATH = Path("config") / "auto_alert_actions.json"
_MAX_ACTION_LOG = 400
_DEFAULT_COOLDIAGN_MINUTES = 15

# ── أنماط تشخيص الأخطاء (نمطي، بدون مفتاح API) ─────────────────────────────
_DIAGNOSIS_PATTERNS: List[Tuple[str, str, str, str]] = [
    # (النمط، الفئة، التشخيص، الاقتراح)
    (r"rate.?limit|429|overloaded|busy", "rate_limit",
     "تشبع حدود الاستخدام (429) أو ازدحام المزوّد",
     "تفعيل سياسة إعادة المحاولة بتراجع (backoff) أو التبديل لمزوّد بديل"),
    (r"timeout|timed out|تجاوز مهلة", "timeout",
     "انتهت مهلة الاتصال أو الاستجابة",
     "توسيع المهلة أو التحقق من سلامة الشبكة والمزوّد"),
    (r"no provider|unavailable|unreachable|لا يوجد مزوّد",
     "provider_unavailable", "المزوّد غير متاح أو غير مُعدّ",
     "إضافة مزوّد احتياطي أو إعادة إعداد مفاتيح API"),
    (r"refused|connection (error|reset)|[45]\d\d|HTTP", "transient_http",
     "خطأ اتصال أو استجابة HTTP فاشلة",
     "إعادة المحاولة تلقائيًا؛ إن تكرر الخطأ يُراجع سجل الخادم"),
    (r"(?:empty|فارغة|لا يوجد|لا يوجد نص|لم يتم|لم يُنتج)", "empty_response",
     "استجابة فارغة أو ناقصة من الوكيل",
     "مراجعة صياغة المهمة أو تفعيل سياسة إعادة المحاولة"),
]

DIAGNOSIS_UNKNOWN = ("unknown", "نمط غير معروف", "إعادة محاولة عامة مع مراقبة")

# ── تعريفات الإجراءات الافتراضية ──────────────────────────────────────────
DEFAULT_ACTIONS: Dict[str, Dict[str, Any]] = {
    "agent_degraded": {
        "enabled": True,
        "action_type": "diagnose_and_lesson",
        "cooldown_minutes": 60,
        "description": "تشخيص نمطي للوكيل المتدهور وتسجيل درس في الذاكرة الجماعية",
    },
    "repeated_errors": {
        "enabled": True,
        "action_type": "auto_exclude_and_diagnose",
        "cooldown_minutes": 30,
        "description": "استبعاد تلقائي مؤقت للوكيل الفاشل المتكرر مع تشخيص وسبب",
    },
    "failure_direct": {
        "enabled": True,
        "action_type": "diagnose",
        "cooldown_minutes": 10,
        "description": "تشخيص نمطي فوري لخطأ الوكيل المباشر",
    },
    "swarm_failure_rate": {
        "enabled": True,
        "action_type": "lesson_and_escalate",
        "cooldown_minutes": 45,
        "description": "تسجيل درس جماعي عن فشل السرب وتصعيد للإشراف",
    },
    "slow_response": {
        "enabled": True,
        "action_type": "performance_lesson",
        "cooldown_minutes": 30,
        "description": "تسجيل درس أداء لبطء الاستجابة لتفاديه مستقبلاً",
    },
    "swarm_excluded": {
        "enabled": False,
        "action_type": "exclude_followup",
        "cooldown_minutes": 60,
        "description": "إشعار متابعة بعد استبعاد وكيل من السرب",
    },
    "congestion": {
        "enabled": False,
        "action_type": "throttle_hint",
        "cooldown_minutes": 10,
        "description": "اقتراح تخفيف الازدحام عند تجاوز الحد المتزامن",
    },
}

# الوكلاء المستبعدون تلقائيًا عبر الإجراءات (ذاكرة داخلية)
_auto_excluded: Dict[str, Dict[str, Any]] = {}
_auto_excluded_lock = threading.Lock()

# سجل الإجراءات المركزي
_action_log: List[Dict[str, Any]] = []
_action_log_lock = threading.Lock()

# توقيت الإجراءات (تبريد)
_action_fired: Dict[str, float] = {}
_action_lock = threading.Lock()

_rules_lock = threading.Lock()

# ── ملف التهيئة ────────────────────────────────────────────────────────────

def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _load_actions_config() -> Dict[str, Dict[str, Any]]:
    data = _safe_load_json(_ACTIONS_CONFIG_PATH)
    merged: Dict[str, Dict[str, Any]] = {}
    for name, default in DEFAULT_ACTIONS.items():
        entry = dict(default)
        custom = data.get(name) if data else None
        if isinstance(custom, dict):
            for key, value in custom.items():
                dv = default.get(key)
                if isinstance(dv, (int, float)) and isinstance(value, (int, float)):
                    entry[key] = type(dv)(value)
                elif isinstance(dv, bool) and isinstance(value, bool):
                    entry[key] = value
                elif isinstance(dv, str) and isinstance(value, str):
                    entry[key] = value
        merged[name] = entry
    return merged


def load_actions_config() -> Dict[str, Dict[str, Any]]:
    """يعيد إعدادات الإجراءات مع تخزين مؤقت بسيط."""
    return _load_actions_config()


def save_actions_config(custom: Dict[str, Any]) -> bool:
    """يحفظ تخصيص الإجراءات في config/auto_alert_actions.json."""
    try:
        merged: Dict[str, Dict[str, Any]] = {}
        for name, default in DEFAULT_ACTIONS.items():
            entry = dict(default)
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
        _ACTIONS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _ACTIONS_CONFIG_PATH.open("w", encoding="utf-8") as handle:
            json.dump(merged, handle, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


# ── التشخيص النمطي ──────────────────────────────────────────────────────────

def diagnose_error(error_message: str, is_empty: bool = False) -> Dict[str, Any]:
    """يشخّص رسالة خطأ نمطيًا — يعمل محليًا بالكامل بدون مفتاح API."""
    text = f"{error_message}".lower()
    if is_empty or (not error_message and is_empty):
        category, diagnosis, suggestion = _DIAGNOSIS_PATTERNS[4][1], \
            _DIAGNOSIS_PATTERNS[4][2], _DIAGNOSIS_PATTERNS[4][3]
    else:
        category, diagnosis, suggestion = DIAGNOSIS_UNKNOWN
        for pattern, cat, diag, sug in _DIAGNOSIS_PATTERNS:
            if re.search(pattern, text):
                category, diagnosis, suggestion = cat, diag, sug
                break
    return {
        "category": category,
        "diagnosis": diagnosis,
        "suggestion": suggestion,
    }


def _diagnose_from_alert(alert: Dict[str, Any]) -> Dict[str, Any]:
    event = alert.get("event") or {}
    detail = alert.get("detail") or ""
    error_text = (
        event.get("detail") or detail or event.get("error") or ""
    )
    is_empty = "فارغة" in error_text or "empty" in error_text.lower()
    return diagnose_error(error_text, is_empty=is_empty)


# ── سجل الإجراءات ────────────────────────────────────────────────────────────

def _format_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def add_action_record(record: Dict[str, Any]) -> Dict[str, Any]:
    with _action_log_lock:
        record = dict(record)
        record["action_id"] = len(_action_log) + 1
        record.setdefault("timestamp", _format_now())
        _action_log.append(record)
        if len(_action_log) > _MAX_ACTION_LOG:
            del _action_log[: len(_action_log) - _MAX_ACTION_LOG]
    return record


def get_action_log(limit: int = 50) -> List[Dict[str, Any]]:
    return list(_action_log[-limit:])


def clear_action_log() -> None:
    with _action_log_lock:
        _action_log.clear()


def reset_action_cooldowns() -> None:
    _action_fired.clear()


# ── الاستبعاد التلقائي ──────────────────────────────────────────────────────

def auto_excluded_agents() -> Dict[str, Dict[str, Any]]:
    """يرجع الوكلاء المستبعدين تلقائيًا عبر الإجراءات التلقائية."""
    with _auto_excluded_lock:
        return dict(_auto_excluded)


def auto_unexclude(agent_id: str) -> bool:
    with _auto_excluded_lock:
        removed = agent_id in _auto_excluded
        _auto_excluded.pop(agent_id, None)
    return removed


def _auto_exclude_agent(
    agent_id: str, rule_name: str, diagnosis: Dict[str, Any]
) -> Dict[str, Any]:
    """يستبعد وكيلًا مؤقتًا عبر الإجراءات التلقائية ويسجل القرار."""
    now = time.time()
    record = {
        "action_type": "auto_exclude",
        "rule": rule_name,
        "agent_id": agent_id,
        "status": "excluded",
        "diagnosis": diagnosis,
        "duration_minutes": 15,
        "reason": f"استبعاد تلقائي بسبب {rule_name}",
    }
    with _auto_excluded_lock:
        _auto_excluded[agent_id] = {
            "excluded_at": _format_now(),
            "until_ts": now + 15 * 60,
            "rule": rule_name,
            "diagnosis": diagnosis,
        }
    record = add_action_record(record)
    try:
        from ai.agent_event_bus import emit_event as _emit_event
        _emit_event(
            "agent_auto_excluded",
            agent_id=agent_id,
            title="الإجراءات التلقائية",
            status="done",
            detail=record["reason"],
            metadata={
                "rule": rule_name,
                "diagnosis_category": diagnosis.get("category"),
                "until": record.get("timestamp"),
            },
        )
    except Exception:
        pass
    return record


# ── محرك الإجراءات ───────────────────────────────────────────────────────────

def _action_allowed(action_name: str, cooldown_minutes: float, now: float) -> bool:
    cooldown = float(cooldown_minutes or _DEFAULT_COOLDIAGN_MINUTES)
    if cooldown <= 0:
        return True
    last = _action_fired.get(action_name)
    return last is None or now - last >= cooldown * 60


def _mark_action(action_name: str, now: float) -> None:
    _action_fired[action_name] = now


def execute_auto_actions(
    alerts: List[Dict[str, Any]],
    events: Optional[List[Dict[str, Any]]] = None,
    *,
    actions_config: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """ينفّذ الإجراءات التلقائية على قائمة تنبيهات جديدة ويرجع سجل التنفيذ.

    كل قاعدة تنبيه لها إجراء مخصص: التشخيص النمطي، تسجيل درس في الذاكرة
    الجماعية، استبعاد تلقائي مؤقت، أو تصعيد. تحترم فترة تبريد خاصة بكل إجراء.
    """
    from ai.alert_config import DEFAULT_RULES as _DEFAULT_ALERT_RULES
    from ai.collective_memory import get_collective_memory

    config = actions_config if actions_config is not None else load_actions_config()
    events = list(events) if events is not None else []
    now = time.time()
    executed: List[Dict[str, Any]] = []

    with _rules_lock:
        for alert in alerts:
            rule_name = alert.get("rule") or ""
            action_cfg = config.get(rule_name)
            if not action_cfg or not action_cfg.get("enabled"):
                continue
            action_name = f"{rule_name}::action"
            cooldown = float(
                action_cfg.get("cooldown_minutes", _DEFAULT_COOLDIAGN_MINUTES)
            )
            if not _action_allowed(action_name, cooldown, now):
                continue
            action_type = action_cfg.get("action_type", "diagnose")
            diagnosis = _diagnose_from_alert(alert)
            agent_id = (
                alert.get("event", {}).get("agent_id")
                or alert.get("agent_id")
                or "orchestrator"
            )
            records: List[Dict[str, Any]] = []

            # ── استبعاد تلقائي (repeated_errors) ──
            if action_type == "auto_exclude_and_diagnose":
                rec = _auto_exclude_agent(agent_id, rule_name, diagnosis)
                records.append(rec)
                records.append(
                    add_action_record({
                        "action_type": "diagnose",
                        "rule": rule_name,
                        "agent_id": agent_id,
                        "status": "diagnosed",
                        "diagnosis": diagnosis,
                        "title": alert.get("title", ""),
                    })
                )
            # ── تشخيص وتسجيل درس (agent_degraded) ──
            elif action_type == "diagnose_and_lesson":
                records.append(
                    add_action_record({
                        "action_type": "diagnose",
                        "rule": rule_name,
                        "agent_id": agent_id,
                        "status": "diagnosed",
                        "diagnosis": diagnosis,
                        "title": alert.get("title", ""),
                    })
                )
                memory = get_collective_memory()
                memory.record_lesson(
                    domain="أداء الوكلاء",
                    question_hint=(
                        f"تدهور أداء الوكيل {agent_id}"
                    ),
                    lesson=(
                        f"الوكيل {agent_id} تدهور أداؤه: "
                        f"{diagnosis['diagnosis']} — {diagnosis['suggestion']}"
                    ),
                    evidence=alert.get("detail") or alert.get("title", ""),
                    source_agent="alert_auto_actions",
                    quality=0.6,
                )
            # ── تشخيص فوري (failure_direct) ──
            elif action_type == "diagnose":
                records.append(
                    add_action_record({
                        "action_type": "diagnose",
                        "rule": rule_name,
                        "agent_id": agent_id,
                        "status": "diagnosed",
                        "diagnosis": diagnosis,
                        "title": alert.get("title", ""),
                    })
                )
            # ── درس جماعي وتصعيد (swarm_failure_rate) ──
            elif action_type == "lesson_and_escalate":
                stats = _swarm_stats_for_lesson(events)
                memory = get_collective_memory()
                memory.record_lesson(
                    domain="أداء السرب",
                    question_hint="فشل متكرر في مهام السرب",
                    lesson=(
                        "نسبة فشل السرب ارتفعت "
                        f"({stats['failures']}/{stats['tasks']} مهام) — "
                        f"مراجعة التوجيه والترتيب الأولي"
                    ),
                    evidence=alert.get("detail") or alert.get("title", ""),
                    source_agent="alert_auto_actions",
                    quality=0.7,
                )
                records.append(
                    add_action_record({
                        "action_type": "escalate",
                        "rule": rule_name,
                        "agent_id": "swarm",
                        "status": "escalated",
                        "lesson_saved": True,
                        "stats": stats,
                        "title": alert.get("title", ""),
                    })
                )
            # ── درس أداء (slow_response) ──
            elif action_type == "performance_lesson":
                memory = get_collective_memory()
                memory.record_lesson(
                    domain="أداء الوكلاء",
                    question_hint="بطء استجابة الوكلاء",
                    lesson=(
                        f"استجابة بطيئة من {agent_id}: "
                        f"{alert.get('detail', '')} — "
                        f"{diagnosis['suggestion']}"
                    ),
                    evidence=alert.get("detail") or alert.get("title", ""),
                    source_agent="alert_auto_actions",
                    quality=0.5,
                )
                records.append(
                    add_action_record({
                        "action_type": "performance_lesson",
                        "rule": rule_name,
                        "agent_id": agent_id,
                        "status": "lesson_saved",
                        "title": alert.get("title", ""),
                    })
                )
            # ── متابعة استبعاد (swarm_excluded) ──
            elif action_type == "exclude_followup":
                records.append(
                    add_action_record({
                        "action_type": "exclude_followup",
                        "rule": rule_name,
                        "agent_id": agent_id,
                        "status": "followed_up",
                        "diagnosis": diagnosis,
                        "title": alert.get("title", ""),
                    })
                )
            # ── تلميح تخفيف (congestion) ──
            elif action_type == "throttle_hint":
                records.append(
                    add_action_record({
                        "action_type": "throttle_hint",
                        "rule": rule_name,
                        "agent_id": agent_id,
                        "status": "hinted",
                        "hint": diagnosis["suggestion"],
                        "title": alert.get("title", ""),
                    })
                )

            if records:
                _mark_action(action_name, now)
                executed.extend(records)
    return executed


def _swarm_stats_for_lesson(
    events: List[Dict[str, Any]], window: int = 200
) -> Dict[str, Any]:
    try:
        from ai.adaptive_swarm import _FAILURE_EVENTS, _SUCCESS_EVENTS
    except Exception:
        return {"tasks": 0, "failures": 0}
    rows = events[-window:] if window < len(events) else list(events)
    tasks = failures = 0
    for row in rows:
        et = row.get("event_type")
        if et in _SUCCESS_EVENTS:
            tasks += 1
        elif et in _FAILURE_EVENTS:
            tasks += 1
            failures += 1
    return {"tasks": tasks, "failures": failures}


def is_agent_auto_excluded(agent_id: str) -> bool:
    with _auto_excluded_lock:
        entry = _auto_excluded.get(agent_id)
    if entry is None:
        return False
    if entry.get("until_ts") and time.time() > float(entry["until_ts"]):
        _auto_excluded.pop(agent_id, None)
        return False
    return True
