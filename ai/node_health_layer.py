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


class NodeHealthLayer:
    def __init__(self, mesh_node):
        self.node = mesh_node
        self._route_cache: Dict[str, Dict[str, Any]] = {}
        self._task_log: List[Dict[str, Any]] = []

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
            self._task_log.append(entry)
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
        self._task_log.append(entry)
        return entry

    def recent_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._task_log[-limit:]
