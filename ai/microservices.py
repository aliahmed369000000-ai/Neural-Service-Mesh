# -*- coding: utf-8 -*-
"""Microservices Layer — طبقة الخدمات المصغرة بنمط طلب/استجابة ثابت.

تسمح لأجزاء النظام (الوكلاء، الواجهة، MCP) بالتحدث مع بعضها كخدمات
مستقلة بعقد ثابت:
    الطلب (request):  {"service", "action", "payload", "request_id",
                       "requested_by", "timeout_ms"}
    الاستجابة (response): {"ok", "service", "action", "request_id",
                          "result", "error", "latency_ms",
                          "schema_version": "nsm-ms/1.0"}

الخدمات المسجّلة:
- meta: list_services / describe_service
- harm: classify (تصنيف أذى نص — يعتمد ai.harm_guard إن وُجد)
- ckg: search (بحث في قاعدة المعرفة)
- dashboard: snapshot (لقطة لوحة السرب)
- connectors: list / describe / call (الموصلات الخارجية)
- backend: kv / agents / tasks / memories / messages (مركز البيانات)

يُنفَّذ الاستدعاء بنمطين:
1) مباشر (call_service) — مناسب للواجهة والاختبارات
2) عبر ناقل أحداث agent_event_bus (call_service_async) — يُسجَّل طلبًا
   وحدث استجابة لاحقًا، فترى الوكيل والواجهة الاستدعاء في السجل الحي.

لا مفاتيح API هنا — كل شيء استدعاءات داخلية أو محاكاة موثوقة.
"""
from __future__ import annotations

import random
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

SERVICE_REGISTRY: Dict[str, Dict[str, Any]] = {}
_REGISTRY_LOCK = threading.Lock()
SCHEMA_VERSION = "nsm-ms/1.0"
DEFAULT_TIMEOUT_MS = 5000

# ── مؤشرات أداء الخدمات (KPIs: ذاكرة + وقت الاستجابة) ─────────
_METRICS: Dict[str, Dict[str, Any]] = {}  # {service: {calls, ok, failed,
#            latencies: [ms], last_latency_ms, max_latency_ms, slow_count}}
_METRICS_LOCK = threading.Lock()
_TIMELINE: List[Dict[str, Any]] = []  # [(ts, service, latency_ms, ok)]
_TIMELINE_LOCK = threading.Lock()
_TIMELINE_LIMIT = 500
_DEFAULT_SLOW_THRESHOLD_MS = 2000.0  # عتبة بطء الخدمة (ms)


def set_service_slow_threshold(threshold_ms: float) -> None:
    """تغيير عتبة البطء القياسية لمؤشرات الخدمات (مستقلة عن عتبة الوكلاء)."""
    global _DEFAULT_SLOW_THRESHOLD_MS
    _DEFAULT_SLOW_THRESHOLD_MS = float(threshold_ms)


def get_service_slow_threshold() -> float:
    return float(_DEFAULT_SLOW_THRESHOLD_MS)


def _make_response(ok: bool, service: str, action: str,
                   request_id: str, result: Any = None,
                   error: Optional[str] = None, latency_ms: float = 0.0
                   ) -> Dict[str, Any]:
    return {"ok": ok, "service": service, "action": action,
            "request_id": request_id, "result": result, "error": error,
            "latency_ms": round(latency_ms, 2),
            "schema_version": SCHEMA_VERSION}


# ------------------------------------------------------------------ meta
def _meta_handler(action: str, payload: Dict[str, Any]
                  ) -> Optional[Dict[str, Any]]:
    with _REGISTRY_LOCK:
        names = sorted(SERVICE_REGISTRY.keys())
    if action == "list_services":
        return {"services": names}
    if action == "describe_service":
        svc = str(payload.get("service", ""))
        if svc not in SERVICE_REGISTRY:
            return {"ok": False, "error": f"خدمة غير مسجلة: {svc}"}
        info = SERVICE_REGISTRY[svc]
        return {"ok": True, "service": svc,
                "doc": info.get("doc", ""),
                "actions": info.get("actions", [])}
    return None


def _harm_handler(action: str, payload: Dict[str, Any]
                  ) -> Optional[Dict[str, Any]]:
    if action == "classify":
        text = str(payload.get("text", ""))
        try:
            from ai.harm_guard import analyze_harm
            analysis = analyze_harm(text)
            return {"ok": True, "text": text, "analysis": analysis}
        except Exception:
            # مسار احتياطي إن وُجدت دالة أخرى في harm_guard
            try:
                from ai.harm_guard import check_text
                return {"ok": True, "text": text,
                        "analysis": {"check_text": check_text(text)}}
            except Exception:
                return {"ok": True, "text": text,
                        "analysis": {"available": False,
                                     "note": "harm_guard غير قابل للاستيراد"}}
    return None


def _ckg_handler(action: str, payload: Dict[str, Any]
                 ) -> Optional[Dict[str, Any]]:
    if action == "search":
        query = str(payload.get("query", ""))
        limit = int(payload.get("limit", 5))
        try:
            from ai.collective_knowledge import search as ckg_search
            hits = ckg_search(query, limit=limit)
            return {"ok": True, "query": query, "hits": hits}
        except Exception:
            return {"ok": True, "query": query, "hits": [],
                    "note": "قاعدة المعرفة غير متاحة حاليًا"}
    return None


def _dashboard_handler(action: str, payload: Dict[str, Any]
                       ) -> Optional[Dict[str, Any]]:
    if action == "snapshot":
        try:
            from ai.unified_swarm_dashboard import unified_dashboard_snapshot
            return {"ok": True, "snapshot": unified_dashboard_snapshot(
                limit=int(payload.get("limit", 80)))}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return None


def _connectors_handler(action: str, payload: Dict[str, Any]
                        ) -> Optional[Dict[str, Any]]:
    from connectors.external_services import (
        call_connector, describe_connector, list_connectors)
    if action == "list":
        return {"ok": True, "connectors": list_connectors()}
    if action == "describe":
        return describe_connector(str(payload.get("service", "")))
    if action == "call":
        return call_connector(str(payload.get("service", "")),
                              str(payload.get("action", "")),
                              payload.get("payload"))
    return None


def _backend_handler(action: str, payload: Dict[str, Any]
                     ) -> Optional[Dict[str, Any]]:
    from ai import backend_layer as bl
    if action == "counts":
        return {"ok": True, "counts": bl.backend_counts()}
    if action == "kv_get":
        return {"ok": True, "value": bl.kv_get(
            str(payload.get("key", "")),
            str(payload.get("domain", "general")))}
    if action == "kv_set":
        return bl.kv_set(str(payload.get("key", "")), payload.get("value"),
                         str(payload.get("domain", "general")))
    if action == "task_create":
        return bl.task_create(str(payload.get("title", "")),
                              str(payload.get("type", "general")),
                              payload.get("payload"))
    if action == "task_update":
        return bl.task_update(str(payload.get("task_id", "")),
                              payload.get("updates"))
    if action == "memory_add":
        return bl.memory_add(str(payload.get("subject", "")),
                             str(payload.get("content", "")),
                             payload.get("tags"),
                             float(payload.get("importance", 0.5)))
    if action == "memory_search":
        return {"ok": True, "memories": bl.memory_search(
            str(payload.get("query", "")),
            limit=int(payload.get("limit", 25)))}
    if action == "message_send":
        return bl.message_send(str(payload.get("sender", "")),
                               str(payload.get("receiver", "")),
                               str(payload.get("subject", "")),
                               str(payload.get("body", "")),
                               payload.get("headers"))
    if action == "message_inbox":
        return {"ok": True, "messages": bl.message_inbox(
            str(payload.get("receiver", "")),
            limit=int(payload.get("limit", 50)),
            unread_only=bool(payload.get("unread_only")))}
    return None


def _register_defaults() -> None:
    with _REGISTRY_LOCK:
        if SERVICE_REGISTRY:
            return
        SERVICE_REGISTRY["meta"] = {
            "doc": "اكتشاف الخدمات: list_services / describe_service",
            "actions": ["list_services", "describe_service"],
            "handler": _meta_handler}
        SERVICE_REGISTRY["harm"] = {
            "doc": "تصنيف أذى النصوص (harm_guard)",
            "actions": ["classify"], "handler": _harm_handler}
        SERVICE_REGISTRY["ckg"] = {
            "doc": "البحث في قاعدة المعرفة الجماعية",
            "actions": ["search"], "handler": _ckg_handler}
        SERVICE_REGISTRY["dashboard"] = {
            "doc": "لقطة لوحة السرب الموحدة",
            "actions": ["snapshot"], "handler": _dashboard_handler}
        SERVICE_REGISTRY["connectors"] = {
            "doc": "الموصلات الخارجية: list / describe / call",
            "actions": ["list", "describe", "call"],
            "handler": _connectors_handler}
        SERVICE_REGISTRY["backend"] = {
            "doc": "مركز البيانات: counts/kv_*/task_*/memory_*/message_*",
            "actions": ["counts", "kv_get", "kv_set", "task_create",
                        "task_update", "memory_add", "memory_search",
                        "message_send", "message_inbox"],
            "handler": _backend_handler}


def register_service(name: str, doc: str, actions: List[str],
                     handler: Callable[[str, Dict[str, Any]],
                                       Optional[Dict[str, Any]]]
                     ) -> Dict[str, Any]:
    with _REGISTRY_LOCK:
        SERVICE_REGISTRY[str(name)] = {"doc": doc, "actions": actions,
                                       "handler": handler}
    return {"ok": True, "service": name}


def list_services() -> List[str]:
    _register_defaults()
    with _REGISTRY_LOCK:
        return sorted(SERVICE_REGISTRY.keys())


def call_service(service: str, action: str,
                 payload: Optional[Dict[str, Any]] = None,
                 request_id: Optional[str] = None,
                 requested_by: str = "",
                 timeout_ms: int = DEFAULT_TIMEOUT_MS
                 ) -> Dict[str, Any]:
    """استدعاء خدمة بنمط request/response ثابت (مباشر)."""
    _register_defaults()
    start = time.time()
    with _REGISTRY_LOCK:
        info = SERVICE_REGISTRY.get(str(service).lower())
    if info is None:
        latency_ms = (time.time() - start) * 1000
        resp = _make_response(False, service, action,
                              request_id or _make_request_id(),
                              error=f"خدمة غير مسجلة: {service}",
                              latency_ms=latency_ms)
        _record_metric(service, latency_ms, False,
                       _DEFAULT_SLOW_THRESHOLD_MS)
        return resp
    handler = info["handler"]
    result = handler(action, payload or {})
    if result is None:
        latency_ms = (time.time() - start) * 1000
        resp = _make_response(False, service, action,
                              request_id or _make_request_id(),
                              error=f"إجراء غير معروف: {action}",
                              latency_ms=latency_ms)
        _record_metric(service, latency_ms, False,
                       _DEFAULT_SLOW_THRESHOLD_MS)
        return resp
    latency = (time.time() - start) * 1000
    if timeout_ms and latency > timeout_ms:
        resp = _make_response(False, service, action,
                              request_id or _make_request_id(),
                              result=result,
                              error=f"تجاوز المهلة: {timeout_ms}ms",
                              latency_ms=latency)
        _record_metric(service, latency, False,
                       _DEFAULT_SLOW_THRESHOLD_MS)
        return resp
    resp = _make_response(result.get("ok", True), service, action,
                          request_id or _make_request_id(),
                          result=result, latency_ms=latency)
    _record_metric(service, latency,
                   resp.get("ok", False),
                   _DEFAULT_SLOW_THRESHOLD_MS)
    _emit_service_event(service, action, resp, requested_by)
    return resp


def _make_request_id() -> str:
    return f"ms_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


# ── مؤشرات الأداء ──────────────────────────────────────────────
def _record_metric(service: str, latency_ms: float, ok: bool,
                   slow_threshold_ms: float) -> None:
    """تسجيل قياس أداء واحد للخدمة (يُستدعى بعد كل استدعاء ناجح)."""
    with _METRICS_LOCK:
        row = _METRICS.setdefault(str(service), {
            "service": str(service), "calls": 0, "ok": 0, "failed": 0,
            "latencies": [], "last_latency_ms": None, "max_latency_ms": None,
            "avg_latency_ms": None, "slow_count": 0,
            "slow_threshold_ms": slow_threshold_ms})
        row["calls"] += 1
        if ok:
            row["ok"] += 1
        else:
            row["failed"] += 1
        row["latencies"].append(latency_ms)
        if len(row["latencies"]) > 500:
            row["latencies"] = row["latencies"][-250:]
    # السلسلة الزمنية للرسوم البيانية التفاعلية (لكل استدعاء خدمة)
    with _TIMELINE_LOCK:
        _TIMELINE.append({
            "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "service": str(service), "latency_ms": round(latency_ms, 2),
            "ok": bool(ok)})
        if len(_TIMELINE) > _TIMELINE_LIMIT:
            del _TIMELINE[:len(_TIMELINE) - _TIMELINE_LIMIT]
        row["last_latency_ms"] = round(latency_ms, 2)
        prev_max = row["max_latency_ms"]
        if prev_max is None or latency_ms > prev_max:
            row["max_latency_ms"] = round(latency_ms, 2)
        n = len(row["latencies"])
        row["avg_latency_ms"] = round(sum(row["latencies"]) / n, 2)
        if latency_ms >= slow_threshold_ms:
            row["slow_count"] += 1


def service_metrics(threshold_ms: Optional[float] = None,
                    limit: int = 60) -> Dict[str, Any]:
    """مؤشرات وقت الاستجابة الجماعية للخدمات المصغرة.

    يعيد: count / avg_ms / max_ms / last_ms / slow_count /
    slow_ms_threshold — من آخر `limit` استدعاء مسجّل.
    """
    out: Dict[str, Any] = {"count": 0, "avg_ms": None, "max_ms": None,
                           "last_ms": None, "slow_count": 0}
    thr = float(threshold_ms) if threshold_ms is not None \
        else _DEFAULT_SLOW_THRESHOLD_MS
    out["slow_ms_threshold"] = thr
    with _METRICS_LOCK:
        rows = list(_METRICS.values())
    flat: List[float] = []
    for row in rows:
        flat.extend(row["latencies"][-1000:])
    flat = flat[-limit:]
    out["count"] = len(flat)
    if flat:
        out["avg_ms"] = round(sum(flat) / len(flat), 2)
        out["max_ms"] = round(max(flat), 2)
        out["last_ms"] = round(flat[-1], 2)
        out["slow_count"] = sum(1 for v in flat if v >= thr)
    return out


def service_usage(service: str) -> Optional[Dict[str, Any]]:
    """مؤشرات خدمة واحدة: عدد الاستدعاءات/النجاح/الفشل/الاستجابة/الذروة."""
    with _METRICS_LOCK:
        row = _METRICS.get(str(service))
    if row is None:
        return None
    return {"service": row["service"], "calls": row["calls"],
            "ok": row["ok"], "failed": row["failed"],
            "avg_latency_ms": row["avg_latency_ms"],
            "max_latency_ms": row["max_latency_ms"],
            "last_latency_ms": row["last_latency_ms"],
            "slow_count": row["slow_count"],
            "slow_threshold_ms": row["slow_threshold_ms"],
            "health": "healthy" if not row["slow_count"]
            else ("degraded" if row["slow_count"] < row["calls"] * 0.5
                  else "critical")}


def all_service_usage() -> Dict[str, Any]:
    """مؤشرات كل الخدمات المسجّلة مرتبة بالاستخدام."""
    with _METRICS_LOCK:
        rows = sorted(_METRICS.values(), key=lambda r: r["calls"],
                      reverse=True)
    return {"services": [{k: v for k, v in row.items()}
                          for row in rows],
            "total_services": len(rows),
            "total_calls": sum(r["calls"] for r in rows),
            "total_slow": sum(r["slow_count"] for r in rows),
            "slow_ms_threshold": _DEFAULT_SLOW_THRESHOLD_MS}


def reset_service_metrics() -> None:
    """إعادة ضبط كل مؤشرات الأداء (للاختبارات والصيانة)."""
    with _METRICS_LOCK:
        _METRICS.clear()
    with _TIMELINE_LOCK:
        _TIMELINE.clear()


def service_timeline(limit: int = 60) -> List[Dict[str, Any]]:
    """سلسلة زمنية لاستجابة الخدمات المصغرة بمرور الوقت.

    كل استدعاء خدمة يسجّل صفًا: {ts (HH:MM:SS)، service، latency_ms، ok}.
    السجل بحد أقصى `_TIMELINE_LIMIT` صفًا (أحدثها أولًا للعرض).
    بدون مفاتيح API، ويُعاد ضبطه عبر reset_service_metrics().
    """
    with _TIMELINE_LOCK:
        rows = _TIMELINE[-max(1, int(limit)):]
    return list(rows)


def system_memory() -> Dict[str, Any]:
    """مؤشرات استخدام الذاكرة دون اعتماديات خارجية (stdlib فقط).

    VmRSS الفعلية من /proc/self/status، الإجمالي/المتاح من /proc/meminfo،
    وأقصى RSS من resource.getrusage. كل قراءة تتسامح مع فشل فردي.
    """
    result: Dict[str, Any] = {"memory_used_mb": None,
                              "memory_total_mb": None,
                              "memory_percent": None,
                              "peak_rss_mb": None}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    result["memory_used_mb"] = round(
                        int(line.split()[1]) / 1024.0, 1)
                    break
    except (OSError, IndexError, ValueError):
        pass
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
    try:
        import resource as _res
        result["peak_rss_mb"] = round(
            _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:
        pass
    return result


def _emit_service_event(service: str, action: str,
                        response: Dict[str, Any], requested_by: str) -> None:
    try:
        from ai.agent_event_bus import emit_event
        ok = response.get("ok")
        emit_event("microservice.call",
                   agent_id=requested_by or "microservices",
                   title=f"service:{service}/{action}",
                   status="done" if ok else "failed",
                   detail=f"ok={ok} schema={SCHEMA_VERSION}",
                   metadata={"service": service, "action": action,
                             "response": response})
    except Exception:
        pass  # ناقل الأحداث غير متاح — لا نكسر مسار الخدمة


def call_service_async(service: str, action: str,
                       payload: Optional[Dict[str, Any]] = None,
                       requested_by: str = "",
                       timeout_ms: int = DEFAULT_TIMEOUT_MS
                       ) -> Dict[str, Any]:
    """نفس call_service لكن الاستجابة تُنفَّذ بخيط خلفي وتُسجَّل في ناقل
    الأحداث (نمط request/reply عبر bus)."""
    request_id = _make_request_id()
    try:
        from ai.agent_event_bus import emit_event
        emit_event("microservice.request",
                   agent_id=requested_by or "microservices",
                   title=f"service:{service}/{action}",
                   status="running",
                   detail="تم إرسال الطلب عبر الخدمات المصغرة",
                   metadata={"service": service, "action": action,
                             "payload": payload or {},
                             "request_id": request_id})
    except Exception:
        pass

    def _run() -> None:
        resp = call_service(service, action, payload, request_id,
                            requested_by, timeout_ms)
        try:
            from ai.agent_event_bus import emit_event
            emit_event("microservice.reply",
                       agent_id=requested_by or "microservices",
                       title=f"service:{service}/{action}",
                       status="done" if resp.get("ok") else "failed",
                       detail=f"ok={resp.get('ok')} latency={resp.get('latency_ms')}ms",
                       metadata=resp)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "request_id": request_id, "status": "queued"}
