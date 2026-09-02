# -*- coding: utf-8 -*-
"""
NSM Node Health & Verifiable Task Layer
=======================================
طبقة موحّدة فوق LivingMeshNode:
  - صحة العقدة والمسارات (health / routes)
  - مهام قابلة للتحقق عبر إيصال موقّع + digest النتيجة
  - اختيار المسار (مباشر / relay) حسب RTT والسمعة
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("NodeHealthLayer")

TASK_LOG_MAX = 2000  # أقصى عدد إدخالات في سجل المهام؛ الأقدم يُحذف أولاً


class NodeHealthLayer:
    def __init__(self, mesh_node):
        self.node = mesh_node
        self._route_cache: Dict[str, Dict[str, Any]] = {}
        self._task_log: List[Dict[str, Any]] = []

    def _log_task(self, entry: Dict[str, Any]) -> None:
        """إضافة إدخال لسجل المهام مع حد أقصى لمنع تسرّب الذاكرة على عقدة طويلة التشغيل."""
        self._task_log.append(entry)
        if len(self._task_log) > TASK_LOG_MAX:
            del self._task_log[: len(self._task_log) - TASK_LOG_MAX]

    # ------------------------------------------------------------------
    # صحة
    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        snap = self.node.network_health_snapshot()
        rep = self.node.get_reputation(self.node.node_id)
        return {
            "status": "ok",
            "layer": "nsm-health-v1",
            "node_id": snap.get("node_id"),
            "online_peers": snap.get("online_peers"),
            "known_nodes": snap.get("known_nodes"),
            "reputation": rep.get("score", 0),
            "receipts": snap.get("receipts"),
            "content_objects": snap.get("content_objects"),
            "identity_fp": snap.get("identity_pub_fingerprint"),
            "routes_cached": len(self._route_cache),
            "tasks_logged": len(self._task_log),
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    def routes_table(self) -> Dict[str, Any]:
        """جدول مسارات معروف: أقران + آخر RTT + سمعة."""
        peers = self.node._get_active_peers_list()
        rep_all = self.node.get_reputation()
        rows = []
        for p in peers:
            pid = p.get("id")
            if pid == self.node.node_id:
                continue
            cached = self._route_cache.get(pid) or {}
            rows.append({
                "peer_id": pid,
                "host": p.get("host"),
                "port": p.get("port"),
                "capabilities": p.get("capabilities") or [],
                "last_rtt_ms": cached.get("rtt_ms", p.get("last_rtt_ms")),
                "reachable": cached.get("ok"),
                "reputation": (rep_all.get(pid) or {}).get("score", 0),
                "path": cached.get("path", "direct"),
            })
        rows.sort(key=lambda r: (r.get("last_rtt_ms") is None, r.get("last_rtt_ms") or 1e9))
        return {"node_id": self.node.node_id, "routes": rows, "count": len(rows)}

    async def probe_routes(self, timeout: float = 4.0) -> Dict[str, Any]:
        """يفحص كل الأقران ويحدّث كاش المسارات."""
        results = await self.node.measure_peers_health(timeout=timeout)
        for r in results:
            pid = r.get("peer_id") or f"{r.get('host')}:{r.get('port')}"
            self._route_cache[pid] = {
                "ok": bool(r.get("ok")),
                "rtt_ms": r.get("rtt_ms"),
                "path": "direct" if r.get("ok") else "down",
                "probed_at": time.time(),
                "error": r.get("error"),
            }
        return {"probed": len(results), "reachable": sum(1 for r in results if r.get("ok")), "details": results}

    def best_route(self, require_capabilities=None) -> Optional[Dict[str, Any]]:
        """يختار أفضل مسار: reachable + أقل RTT + سمعة أعلى."""
        table = self.routes_table()["routes"]
        need = None
        if require_capabilities:
            need = {require_capabilities} if isinstance(require_capabilities, str) else set(require_capabilities)
        candidates = []
        for r in table:
            if need and not need.issubset(set(r.get("capabilities") or [])):
                continue
            if r.get("reachable") is False:
                continue
            candidates.append(r)
        if not candidates:
            # لا كاش بعد — خذ من قائمة الأقران النشطين
            peers = self.node._get_active_peers_list(require_capabilities=require_capabilities)
            for p in peers:
                if p.get("id") == self.node.node_id:
                    continue
                if p.get("host") and p.get("port") is not None:
                    return {"peer_id": p.get("id"), "host": p["host"], "port": p["port"], "path": "direct"}
            return None
        candidates.sort(
            key=lambda r: (
                r.get("last_rtt_ms") is None,
                r.get("last_rtt_ms") or 1e9,
                -int(r.get("reputation") or 0),
            )
        )
        return candidates[0]

    # ------------------------------------------------------------------
    # مهام قابلة للتحقق
    # ------------------------------------------------------------------
    @staticmethod
    def result_digest(result: Dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(result, sort_keys=True, default=str).encode()
        ).hexdigest()

    def verify_receipt(self, receipt: Dict[str, Any], result: Dict[str, Any] = None) -> Dict[str, Any]:
        """يتحقق من إيصال موقّع (+ تطابق digest إن وُجدت النتيجة)."""
        if not receipt:
            return {"ok": False, "error": "empty_receipt"}
        body = {k: receipt[k] for k in receipt if k != "signature"}
        canonical = json.dumps(body, sort_keys=True)
        node_id = receipt.get("node_id")
        sig = receipt.get("signature")
        if not node_id or not sig:
            return {"ok": False, "error": "missing_fields"}
        key_path = self.node.keys_dir / f"{node_id}.pub"
        if node_id == self.node.node_id:
            pub = self.node._pub_pem().encode()
        elif key_path.exists():
            pub = key_path.read_bytes()
        else:
            return {"ok": False, "error": "unknown_signer"}
        sig_ok = self.node.verify_signature(pub, canonical, sig)
        digest_ok = True
        if result is not None and receipt.get("result_digest"):
            digest_ok = self.result_digest(result) == receipt["result_digest"]
        return {
            "ok": bool(sig_ok and digest_ok),
            "signature_valid": bool(sig_ok),
            "digest_valid": bool(digest_ok),
            "node_id": node_id,
            "task_id": receipt.get("task_id"),
        }

    async def submit_verifiable_task(
        self,
        kind: str,
        payload: Dict[str, Any],
        host: str = None,
        port: int = None,
        local: bool = False,
    ) -> Dict[str, Any]:
        """
        يرسل/ينفّذ مهمة ويعيد نتيجة + إيصال قابل للتحقق.
        local=True: تنفيذ محلي فوري عبر mesh_task_protocol.
        """
        from ai import mesh_task_protocol as mt

        task_id = payload.get("task_id") or f"vt_{int(time.time()*1000)}_{kind[:8]}"
        payload = dict(payload)
        payload["task_id"] = task_id
        t0 = time.time()

        if local or host is None:
            result = mt.dispatch_task(kind, payload)
            if result is None:
                return {"ok": False, "error": f"unknown_kind:{kind}", "task_id": task_id}
            receipt = self.node.issue_execution_receipt(task_id, kind, result)
            entry = {
                "task_id": task_id,
                "kind": kind,
                "mode": "local",
                "ok": bool(result.get("ok")),
                "elapsed_ms": round((time.time() - t0) * 1000, 2),
                "receipt": receipt,
                "result": result,
            }
            self._log_task(entry)
            return entry

        disp = await self.node.dispatch_mesh_task(host, int(port), kind, payload)
        entry = {
            "task_id": task_id,
            "kind": kind,
            "mode": "remote",
            "dispatch": disp,
            "elapsed_ms": round((time.time() - t0) * 1000, 2),
            "ok": bool(disp.get("ok")),
        }
        self._log_task(entry)
        return entry

    def recent_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._task_log[-limit:]

    # ------------------------------------------------------------------
    # #6 اختيار أفضل عامل حسب الصحة والكمون والسمعة
    # ------------------------------------------------------------------
    def rank_workers(
        self,
        require_capabilities=None,
        max_workers: int = 10,
    ) -> List[Dict[str, Any]]:
        """يرتب العمال: reachable أولاً، ثم أقل RTT، ثم أعلى سمعة."""
        table = self.routes_table()["routes"]
        need = None
        if require_capabilities:
            need = {require_capabilities} if isinstance(require_capabilities, str) else set(require_capabilities)

        ranked = []
        for r in table:
            if need and not need.issubset(set(r.get("capabilities") or [])):
                continue
            score = 0.0
            if r.get("reachable") is True:
                score += 1000.0
            elif r.get("reachable") is False:
                score -= 500.0
            rtt = r.get("last_rtt_ms")
            if rtt is not None:
                score += max(0.0, 200.0 - float(rtt))  # أسرع = أعلى
            score += float(r.get("reputation") or 0) * 5.0
            ranked.append({**r, "selection_score": round(score, 2)})

        # إن لم يوجد كاش، استخدم الأقران النشطين
        if not ranked:
            peers = self.node._get_active_peers_list(require_capabilities=require_capabilities)
            for p in peers:
                if p.get("id") == self.node.node_id:
                    continue
                if not p.get("host") or p.get("port") is None:
                    continue
                ranked.append({
                    "peer_id": p.get("id"),
                    "host": p.get("host"),
                    "port": p.get("port"),
                    "capabilities": p.get("capabilities") or [],
                    "last_rtt_ms": p.get("last_rtt_ms"),
                    "reachable": None,
                    "reputation": 0,
                    "path": "direct",
                    "selection_score": 100.0,
                })

        ranked.sort(key=lambda x: -x.get("selection_score", 0))
        return ranked[: max(1, max_workers)]

    def select_best_worker(self, require_capabilities=None) -> Optional[Dict[str, Any]]:
        """#6 يختار أفضل عامل واحد."""
        ranked = self.rank_workers(require_capabilities=require_capabilities, max_workers=1)
        return ranked[0] if ranked else self.best_route(require_capabilities=require_capabilities)

    # ------------------------------------------------------------------
    # #7 استمرار المهام عند سقوط عامل (failover)
    # ------------------------------------------------------------------
    async def submit_with_failover(
        self,
        kind: str,
        payload: Dict[str, Any],
        require_capabilities=None,
        max_attempts: int = 3,
        prefer_local_fallback: bool = True,
    ) -> Dict[str, Any]:
        """
        يكلّف أفضل عامل؛ عند الفشل ينتقل للتالي.
        يحفظ حالة المهمة في task journal لاستئناف لاحق.
        """
        from ai import mesh_task_protocol as mt

        task_id = payload.get("task_id") or f"fo_{int(time.time()*1000)}_{kind[:8]}"
        payload = dict(payload)
        payload["task_id"] = task_id
        journal = {
            "task_id": task_id,
            "kind": kind,
            "payload": payload,
            "status": "pending",
            "attempts": [],
            "created_at": time.time(),
        }
        self._log_task({"task_id": task_id, "kind": kind, "mode": "failover_start"})

        workers = self.rank_workers(require_capabilities=require_capabilities, max_workers=max_attempts)
        t0 = time.time()

        for i, w in enumerate(workers[:max_attempts]):
            host, port = w.get("host"), w.get("port")
            if not host or port is None:
                continue
            attempt = {
                "worker": w.get("peer_id"),
                "host": host,
                "port": port,
                "selection_score": w.get("selection_score"),
                "attempt": i + 1,
            }
            try:
                disp = await self.node.dispatch_mesh_task(
                    host, int(port), kind, payload, target_id=w.get("peer_id"), use_relay=True
                )
                attempt["dispatch"] = disp
                if disp.get("ok"):
                    journal["status"] = "dispatched"
                    journal["assigned_worker"] = w.get("peer_id")
                    journal["attempts"].append(attempt)
                    entry = {
                        "task_id": task_id,
                        "kind": kind,
                        "mode": "failover_remote",
                        "ok": True,
                        "worker": w.get("peer_id"),
                        "attempts": i + 1,
                        "elapsed_ms": round((time.time() - t0) * 1000, 2),
                        "dispatch": disp,
                        "journal": journal,
                    }
                    self._log_task(entry)
                    self.node.update_reputation(w.get("peer_id") or "unknown", delta=1, reason="failover_success")
                    return entry
                attempt["error"] = "dispatch_not_ok"
            except Exception as e:
                attempt["error"] = str(e)
            journal["attempts"].append(attempt)
            # عامل ساقط — خفّض سمعته قليلاً
            if w.get("peer_id"):
                self.node.update_reputation(w["peer_id"], delta=-1, reason="failover_worker_down")

        # كل العمال فشلوا — تنفيذ محلي كاستمرارية
        if prefer_local_fallback:
            result = mt.dispatch_task(kind, payload)
            if result is not None:
                receipt = self.node.issue_execution_receipt(task_id, kind, result)
                journal["status"] = "completed_local_fallback"
                entry = {
                    "task_id": task_id,
                    "kind": kind,
                    "mode": "failover_local",
                    "ok": bool(result.get("ok")),
                    "attempts": len(journal["attempts"]),
                    "elapsed_ms": round((time.time() - t0) * 1000, 2),
                    "result": result,
                    "receipt": receipt,
                    "journal": journal,
                }
                self._log_task(entry)
                return entry

        journal["status"] = "failed"
        entry = {
            "task_id": task_id,
            "kind": kind,
            "mode": "failover_exhausted",
            "ok": False,
            "attempts": len(journal["attempts"]),
            "elapsed_ms": round((time.time() - t0) * 1000, 2),
            "journal": journal,
        }
        self._log_task(entry)
        return entry

    def resume_pending_tasks(self) -> List[Dict[str, Any]]:
        """يعيد المهام التي بقيت pending/failed من السجل (للاستئناف بعد إعادة التشغيل)."""
        pending = [
            t for t in self._task_log
            if (t.get("journal") or {}).get("status") in ("pending", "failed", "dispatched")
            or t.get("mode") == "failover_exhausted"
        ]
        return pending


    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """حالة مهمة من سجل LivingMeshNode المحلي."""
        return self.node.get_task_status(task_id)

    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """إلغاء مهمة محلية عبر السجل."""
        return self.node.cancel_local_task(task_id)

    def list_tasks(self, status: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        return self.node.list_tasks(status=status, limit=limit)

    def orchestrator(self):
        """مدير مهام متعدد العمال (مرحلة A)."""
        from ai.mesh_job_orchestrator import MeshJobOrchestrator
        if not hasattr(self, "_orchestrator") or self._orchestrator is None:
            self._orchestrator = MeshJobOrchestrator(self.node, health_layer=self)
        return self._orchestrator

    async def submit_job(self, kind: str, payload: dict = None, **kwargs):
        return await self.orchestrator().submit_job(kind, payload, **kwargs)

    def cognitive_net(self, quorum: int = 2, require_independent: bool = True):
        """واجهة مختصرة لشبكة التنفيذ المعرفي القابلة للتحقق."""
        from ai.verifiable_cognitive_net import VerifiableCognitiveNet
        return VerifiableCognitiveNet(self.node, quorum=quorum, require_independent=require_independent)
