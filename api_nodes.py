# -*- coding: utf-8 -*-
"""واجهة REST لإدارة العقد الحقيقية (core.mesh_bundle.get_mesh_bundle).

تكمّل /process و/plan في api_server.py: تلك تشغّل مسارات، وهذه تعرض العقد
وتديرها:

  GET  /nodes                      قائمة العقد (?state= &tag=) بحالتها وسمعتها
  GET  /nodes/neural/ranking       ترتيب العقد بتوقّع الشبكة العصبية (?limit=)
  GET  /nodes/runs/{run_id}        نتيجة تشغيل محفوظ
  GET  /nodes/{node_id}            تفاصيل عقدة (schemas + سمعة + توقّع الشبكة)
  GET  /nodes/{node_id}/runs       آخر تشغيلات مرّت بالعقدة (?limit=)
  POST /nodes/{node_id}/execute    تشغيل عقدة واحدة {"data": {...}}
  POST /nodes/{node_id}/pause      إيقاف يدوي {"reason": "manual"}
  POST /nodes/{node_id}/resume     استئناف

كل المسارات محمية بـ X-API-Key مقابل NSM_API_KEY/NSM_ADMIN_KEY — fail-closed:
إن لم يُضبط أي مفتاح بالبيئة تُرفض كل الطلبات (403). التشغيل يمرّ بنفس
ExecutionEngine التي يستخدمها /process (fallback + سمعة + حجر + الشبكة العصبية).
"""
from __future__ import annotations

import hmac
import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/nodes", tags=["nodes"])


# ── المصادقة (fail-closed) ───────────────────────────────────────────────
def require_api_key(request: Request) -> None:
    provided = (request.headers.get("x-api-key") or "").strip().encode("utf-8")
    if provided:
        for name in ("NSM_API_KEY", "NSM_ADMIN_KEY"):
            configured = os.environ.get(name, "").strip().encode("utf-8")
            # bytes وليس str: hmac.compare_digest ترفع TypeError لنص غير ASCII
            if configured and len(provided) == len(configured) and hmac.compare_digest(provided, configured):
                return
    raise HTTPException(status_code=403, detail="X-API-Key غير مطابق أو NSM_API_KEY/NSM_ADMIN_KEY غير مضبوط")


def get_bundle():
    try:
        from core.mesh_bundle import get_mesh_bundle
        return get_mesh_bundle()
    except Exception as e:  # noqa: BLE001
        logger.error("api_nodes: الحزمة غير متاحة: %s", e)
        raise HTTPException(status_code=503, detail="شبكة العقد غير متاحة")


# ── نماذج الطلب ──────────────────────────────────────────────────────────
class ExecuteBody(BaseModel):
    data: Dict[str, Any] = Field(default_factory=dict)


class PauseBody(BaseModel):
    reason: str = Field(default="manual", min_length=1, max_length=64)


# ── مساعدات ──────────────────────────────────────────────────────────────
def _node_or_404(bundle, node_id: str):
    node = bundle.registry.get(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="العقدة غير موجودة")
    return node


def _rep_info(bundle, node_id: str) -> Dict[str, Any]:
    rep = bundle.reputation_engine.get_reputation(node_id)
    return {
        "reputation_score": bundle.reputation_engine.get_score(node_id),
        "is_quarantined": bool(rep and rep.is_quarantined),
    }


def _neural_info(bundle, node) -> Dict[str, Any]:
    adv = bundle.neural_advisor
    rep_norm = bundle.reputation_engine.get_score(node.node_id) / 100.0
    return {
        "predicted_success": round(adv.predict(node.node_id, node.state, rep_norm), 4),
        "smoothed_success_rate": round(adv.success_rate(node.node_id), 4),
    }


def _summary(bundle, node) -> Dict[str, Any]:
    d = node.to_dict()
    out = {k: d.get(k) for k in (
        "node_id", "name", "node_type", "state", "pause_reason", "tags",
        "execution_count", "last_executed",
    )}
    out.update(_rep_info(bundle, node.node_id))
    return out


def _engine(bundle):
    from core.engine import ExecutionEngine
    return ExecutionEngine(bundle.registry, bundle.graph, bundle.storage,
                           db=bundle.exec_log, ai=bundle.ai_decision)


# ── المسارات (الحرفية أولاً قبل {node_id}) ───────────────────────────────
@router.get("", dependencies=[Depends(require_api_key)])
def list_nodes(state: Optional[str] = None, tag: Optional[str] = None, bundle=Depends(get_bundle)):
    nodes = bundle.registry.list_all()
    if state:
        nodes = [n for n in nodes if n.state == state]
    if tag:
        nodes = [n for n in nodes if tag in (n.tags or [])]
    items = sorted((_summary(bundle, n) for n in nodes), key=lambda x: str(x.get("name")))
    return {"count": len(items), "nodes": items}


@router.get("/neural/ranking", dependencies=[Depends(require_api_key)])
def neural_ranking(limit: int = Query(20, ge=1, le=200), bundle=Depends(get_bundle)):
    rows = []
    for n in bundle.registry.list_all():
        row = {"node_id": n.node_id, "name": n.name, "state": n.state}
        row.update(_neural_info(bundle, n))
        rows.append(row)
    rows.sort(key=lambda r: -r["predicted_success"])
    return {"advisor": bundle.neural_advisor.summary(), "ranking": rows[:limit]}


@router.get("/runs/{run_id}", dependencies=[Depends(require_api_key)])
def get_run(run_id: str, bundle=Depends(get_bundle)):
    run = _engine(bundle).get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="التشغيل غير موجود")
    return run


@router.get("/{node_id}", dependencies=[Depends(require_api_key)])
def get_node(node_id: str, bundle=Depends(get_bundle)):
    node = _node_or_404(bundle, node_id)
    out = node.to_dict()
    out.update(_rep_info(bundle, node_id))
    rep = bundle.reputation_engine.get_reputation(node_id)
    out["reputation"] = rep.to_dict() if rep else None
    out["neural"] = _neural_info(bundle, node)
    return out


@router.get("/{node_id}/runs", dependencies=[Depends(require_api_key)])
def node_runs(node_id: str, limit: int = Query(20, ge=1, le=200), bundle=Depends(get_bundle)):
    _node_or_404(bundle, node_id)
    recent = _engine(bundle).get_history(500)
    hits = [r for r in recent if node_id in (r.get("path") or [])]
    return {"node_id": node_id, "count": min(len(hits), limit), "runs": hits[:limit]}


@router.post("/{node_id}/execute", dependencies=[Depends(require_api_key)])
def execute_node(node_id: str, body: ExecuteBody, bundle=Depends(get_bundle)):
    _node_or_404(bundle, node_id)
    engine = _engine(bundle)
    prev_states = bundle.snapshot_node_states()
    result = engine.run_path([node_id], body.data)
    try:  # نفس ربط /process بالسمعة/الحجر؛ فشله لا يُسقط الاستجابة
        bundle.record_direct_execution(result, prev_states)
    except Exception as e:  # noqa: BLE001
        logger.warning("api_nodes: record_direct_execution failed: %s", e)
    return {"status": result.status, "result": result.to_dict()}


@router.post("/{node_id}/pause", dependencies=[Depends(require_api_key)])
def pause_node(node_id: str, body: PauseBody, bundle=Depends(get_bundle)):
    node = _node_or_404(bundle, node_id)
    node.pause(reason=body.reason)
    bundle.registry.refresh_meta(node_id)
    return {"node_id": node_id, "state": node.state, "pause_reason": node.pause_reason}


@router.post("/{node_id}/resume", dependencies=[Depends(require_api_key)])
def resume_node(node_id: str, bundle=Depends(get_bundle)):
    node = _node_or_404(bundle, node_id)
    resumed = bool(node.resume())
    bundle.registry.refresh_meta(node_id)
    info = _rep_info(bundle, node_id)
    return {
        "node_id": node_id, "resumed": resumed, "state": node.state,
        # الاستئناف اليدوي لا يرفع الحجر في طبقة السمعة؛ يبقى التوجيه يتجنّبها
        # حتى يرفعه التعافي التلقائي — نُبلّغ العميل بدل إخفاء ذلك.
        "reputation_quarantined": info["is_quarantined"],
    }
