# -*- coding: utf-8 -*-
"""
Mesh Job Orchestrator — طبقة إدارة مهام خفيفة فوق Living Mesh RPC
================================================================
المرحلة A:
  - إسناد مهمة إلى N عمال (حسب القدرات/الصحة/السمعة)
  - جمع النتائج عبر request_from_peer
  - اختيار فائز: first_success | majority | best_reputation
  - إعادة محاولة محدودة عند فشل عامل (لا عند duplicate_rejected)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MeshJobOrchestrator")

STRATEGY_FIRST = "first_success"
STRATEGY_MAJORITY = "majority"
STRATEGY_REPUTATION = "best_reputation"
ALL_STRATEGIES = {STRATEGY_FIRST, STRATEGY_MAJORITY, STRATEGY_REPUTATION}

# أخطاء لا تُعاد محاولتها على نفس task_id
NON_RETRYABLE_ERRORS = frozenset({
    "duplicate_rejected",
    "task_cancelled",
    "missing_capabilities",
})


# حقول دلالية فقط — تُستبعد المعرّفات الفريدة لكل عامل (task_id/worker/receipt/…)
_SEMANTIC_KEYS = (
    "ok", "output", "error", "modality", "model_hint", "prompt_preview",
    "counts", "mean_loss", "accuracy", "loss", "found", "value",
    "layers_count", "text_out", "summary",
)


def semantic_payload(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """يستخرج الجسم الدلالي للمقارنة بين عمال مختلفين."""
    if not result or not isinstance(result, dict):
        return {}
    body = {}
    for k in _SEMANTIC_KEYS:
        if k in result:
            body[k] = result.get(k)
    # إن لم يوجد أي مفتاح معروف: خذ كل شيء عدا الحقول الفريدة
    if not body:
        skip = {
            "task_id", "worker_node", "receipt", "node_id", "elapsed_ms",
            "job_id", "retry_of", "from", "signature",
        }
        body = {k: v for k, v in result.items() if k not in skip}
    return body


def _result_digest(result: Optional[Dict[str, Any]]) -> Optional[str]:
    """هاش دلالي للاتفاق — لا يعتمد على task_id/worker_node/receipt."""
    if not result or not isinstance(result, dict):
        return None
    body = semantic_payload(result)
    if not body:
        return None
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str, ensure_ascii=False).encode()
    ).hexdigest()


class MeshJobOrchestrator:
    def __init__(self, mesh_node, health_layer=None):
        self.node = mesh_node
        self.health = health_layer
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def _rank_workers(
        self,
        require_capabilities=None,
        max_workers: int = 5,
    ) -> List[Dict[str, Any]]:
        if self.health is not None and hasattr(self.health, "rank_workers"):
            return self.health.rank_workers(
                require_capabilities=require_capabilities,
                max_workers=max_workers,
            )
        peers = self.node._get_active_peers_list(require_capabilities=require_capabilities)
        out = []
        for p in peers:
            if p.get("id") == self.node.node_id:
                continue
            if not p.get("host") or p.get("port") is None:
                continue
            out.append({
                "peer_id": p.get("id"),
                "host": p.get("host"),
                "port": p.get("port"),
                "capabilities": p.get("capabilities") or [],
                "reputation": 0,
                "selection_score": 100.0,
            })
        return out[: max(1, max_workers)]

    def _reputation(self, peer_id: str) -> int:
        try:
            rep = self.node.get_reputation(peer_id) or {}
            return int(rep.get("score") or 0)
        except Exception:
            return 0

    async def _dispatch_one(
        self,
        worker: Dict[str, Any],
        kind: str,
        payload: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        host, port = worker.get("host"), worker.get("port")
        peer_id = worker.get("peer_id") or worker.get("id")
        task_id = payload.get("task_id") or f"jobtask_{uuid.uuid4().hex[:10]}"
        data = dict(payload)
        data["task_id"] = task_id
        t0 = time.time()
        try:
            res = await self.node.request_from_peer(
                host, int(port), kind, data, timeout=timeout
            )
        except Exception as e:
            return {
                "ok": False,
                "worker": peer_id,
                "host": host,
                "port": port,
                "task_id": task_id,
                "error": f"{type(e).__name__}: {e}",
                "rtt_ms": round((time.time() - t0) * 1000, 2),
                "result": None,
                "digest": None,
            }
        result = res.get("result") if isinstance(res, dict) else None
        err = None
        if isinstance(result, dict) and result.get("error"):
            err = result.get("error")
        elif isinstance(res, dict) and res.get("error"):
            err = res.get("error")
        ok = bool(res.get("ok") and result is not None and (result.get("ok", True)))
        if err in NON_RETRYABLE_ERRORS:
            ok = False
        return {
            "ok": ok,
            "worker": peer_id,
            "host": host,
            "port": port,
            "task_id": task_id,
            "acked": bool(res.get("acked")) if isinstance(res, dict) else False,
            "error": err,
            "rtt_ms": round((time.time() - t0) * 1000, 2),
            "result": result,
            "digest": _result_digest(result) if result else None,
            "rpc": {
                "mode": res.get("mode") if isinstance(res, dict) else None,
                "from": res.get("from") if isinstance(res, dict) else None,
            },
        }

    def _select_winner(
        self,
        attempts: List[Dict[str, Any]],
        strategy: str,
    ) -> Dict[str, Any]:
        successes = [a for a in attempts if a.get("ok") and a.get("result") is not None]
        if not successes:
            return {
                "winner": None,
                "agreement": 0.0,
                "strategy": strategy,
                "reason": "no_successful_results",
            }

        if strategy == STRATEGY_FIRST:
            # أسرع نجاح
            best = min(successes, key=lambda a: a.get("rtt_ms") if a.get("rtt_ms") is not None else 1e9)
            return {
                "winner": best,
                "agreement": 1.0 / max(1, len(attempts)),
                "strategy": strategy,
                "reason": "first_success_lowest_rtt",
            }

        if strategy == STRATEGY_REPUTATION:
            best = max(
                successes,
                key=lambda a: (
                    self._reputation(a.get("worker") or ""),
                    -1 * (a.get("rtt_ms") or 0),
                ),
            )
            return {
                "winner": best,
                "agreement": 1.0,
                "strategy": strategy,
                "reason": "best_reputation",
                "reputation": self._reputation(best.get("worker") or ""),
            }

        # majority by digest
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for a in successes:
            d = a.get("digest") or f"unique_{a.get('task_id')}"
            buckets.setdefault(d, []).append(a)
        top_digest, top_list = max(buckets.items(), key=lambda kv: len(kv[1]))
        # ضمن المجموعة: أعلى سمعة ثم أقل RTT
        winner = max(
            top_list,
            key=lambda a: (
                self._reputation(a.get("worker") or ""),
                -1 * (a.get("rtt_ms") or 0),
            ),
        )
        agreement = len(top_list) / max(1, len(successes))
        return {
            "winner": winner,
            "agreement": round(agreement, 3),
            "strategy": STRATEGY_MAJORITY,
            "reason": "majority_digest",
            "digest": top_digest,
            "cluster_size": len(top_list),
            "success_count": len(successes),
        }

    async def submit_job(
        self,
        kind: str,
        payload: Dict[str, Any] = None,
        *,
        n_workers: int = 3,
        strategy: str = STRATEGY_MAJORITY,
        require_capabilities=None,
        timeout_per_task: float = 12.0,
        retry_failed: int = 1,
        worker_list: Optional[List[Dict[str, Any]]] = None,
        quorum: int = 2,
    ) -> Dict[str, Any]:
        """
        يرسل المهمة إلى عدة عمال ويجمع النتائج ويختار فائزاً.
        كل عامل يحصل على task_id فريد مرتبط بـ job_id.
        """
        strategy = strategy if strategy in ALL_STRATEGIES else STRATEGY_MAJORITY
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        payload = dict(payload or {})
        payload.setdefault("job_id", job_id)

        workers = worker_list or self._rank_workers(
            require_capabilities=require_capabilities,
            max_workers=max(1, n_workers),
        )
        workers = workers[: max(1, n_workers)]
        if not workers:
            report = {
                "ok": False,
                "job_id": job_id,
                "error": "no_workers_available",
                "strategy": strategy,
                "attempts": [],
                "winner": None,
            }
            self._jobs[job_id] = report
            return report

        attempts: List[Dict[str, Any]] = []
        used_peers = set()
        t0 = time.time()

        for i, w in enumerate(workers):
            peer = w.get("peer_id") or w.get("id")
            used_peers.add(peer)
            task_payload = dict(payload)
            task_payload["task_id"] = f"{job_id}_w{i}_{uuid.uuid4().hex[:6]}"
            att = await self._dispatch_one(w, kind, task_payload, timeout_per_task)
            attempts.append(att)

        # إعادة محاولة للفشل القابل للإعادة على عمال لم يُستخدموا
        failed = [
            a for a in attempts
            if not a.get("ok") and (a.get("error") or "unknown") not in NON_RETRYABLE_ERRORS
        ]
        retries_done = 0
        if retry_failed > 0 and failed:
            extra = self._rank_workers(
                require_capabilities=require_capabilities,
                max_workers=n_workers + retry_failed + 2,
            )
            alternates = [
                w for w in extra
                if (w.get("peer_id") or w.get("id")) not in used_peers
            ]
            for a in failed[:retry_failed]:
                if not alternates:
                    break
                w = alternates.pop(0)
                peer = w.get("peer_id") or w.get("id")
                used_peers.add(peer)
                task_payload = dict(payload)
                task_payload["task_id"] = f"{job_id}_retry_{uuid.uuid4().hex[:6]}"
                task_payload["retry_of"] = a.get("task_id")
                att = await self._dispatch_one(w, kind, task_payload, timeout_per_task)
                att["retry_of"] = a.get("task_id")
                attempts.append(att)
                retries_done += 1

        selection = self._select_winner(attempts, strategy)
        winner = selection.get("winner")

        # --- المرحلة B: نصاب الاتفاق + سمعة أوثق ---
        quorum_n = max(1, int(quorum))
        cluster_size = int(selection.get("cluster_size") or (1 if winner else 0))
        success_count = sum(1 for a in attempts if a.get("ok"))
        quorum_met = False
        if winner and strategy == STRATEGY_MAJORITY:
            quorum_met = cluster_size >= quorum_n
        elif winner and strategy in (STRATEGY_FIRST, STRATEGY_REPUTATION):
            # نجاح واحد كافٍ لهذه الاستراتيجيات، لكن نسجّل إن وصل النصاب
            quorum_met = success_count >= quorum_n
        elif winner:
            quorum_met = True

        winner_digest = (winner or {}).get("digest")
        for a in attempts:
            pid = a.get("worker")
            if not pid:
                continue
            try:
                if not a.get("ok"):
                    if a.get("error") not in NON_RETRYABLE_ERRORS:
                        self.node.update_reputation(pid, delta=-1, reason=f"job_fail:{job_id}")
                    continue
                # نجاح ضمن كتلة الاتفاق
                if winner_digest and a.get("digest") == winner_digest and quorum_met:
                    self.node.update_reputation(pid, delta=2, reason=f"job_quorum_agree:{job_id}")
                elif a.get("ok") and quorum_met and a.get("digest") != winner_digest:
                    # نتيجة ناجحة لكنها خارجة عن الأغلبية
                    self.node.update_reputation(pid, delta=0, reason=f"job_dissent:{job_id}")
                elif a.get("ok"):
                    self.node.update_reputation(pid, delta=1, reason=f"job_ok:{job_id}")
            except Exception:
                pass

        report = {
            "ok": winner is not None and (quorum_met if strategy == STRATEGY_MAJORITY else True),
            "job_id": job_id,
            "kind": kind,
            "strategy": strategy,
            "n_workers_targeted": len(workers),
            "attempts_count": len(attempts),
            "success_count": success_count,
            "retries_done": retries_done,
            "elapsed_ms": round((time.time() - t0) * 1000, 2),
            "quorum_required": quorum_n,
            "quorum_met": quorum_met,
            "selection": {k: v for k, v in selection.items() if k != "winner"},
            "winner": {
                "worker": winner.get("worker"),
                "task_id": winner.get("task_id"),
                "digest": winner.get("digest"),
                "rtt_ms": winner.get("rtt_ms"),
                "result": winner.get("result"),
            } if winner else None,
            "all_results": [
                {
                    "worker": a.get("worker"),
                    "task_id": a.get("task_id"),
                    "ok": a.get("ok"),
                    "error": a.get("error"),
                    "rtt_ms": a.get("rtt_ms"),
                    "digest": a.get("digest"),
                    "acked": a.get("acked"),
                    "retry_of": a.get("retry_of"),
                }
                for a in attempts
            ],
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if strategy == STRATEGY_MAJORITY and winner and not quorum_met:
            report["ok"] = False
            report["error"] = "quorum_not_met"
            report["reason"] = (
                f"cluster_size={cluster_size} < quorum_required={quorum_n}"
            )

        # سجل قرار قابل للتدقيق على حالة العقدة المنظمة
        try:
            self._persist_job_decision(report)
        except Exception as e:
            logger.warning(f"⚠️ persist job decision failed: {e}")

        self._jobs[job_id] = report
        logger.info(
            f"📦 job {job_id} ok={report['ok']} success={report['success_count']}/"
            f"{report['attempts_count']} quorum_met={quorum_met} strategy={strategy}"
        )
        return report

    def _persist_job_decision(self, report: Dict[str, Any]) -> None:
        """يحفظ قرار المهمة في network_state للعقدة المنظمة (provenance خفيف)."""
        if not hasattr(self.node, "_load_state"):
            return
        state = self.node._load_state()
        ledger = state.setdefault("job_decisions", {})
        ledger[report["job_id"]] = {
            "job_id": report.get("job_id"),
            "kind": report.get("kind"),
            "ok": report.get("ok"),
            "strategy": report.get("strategy"),
            "quorum_required": report.get("quorum_required"),
            "quorum_met": report.get("quorum_met"),
            "agreement": (report.get("selection") or {}).get("agreement"),
            "winner_worker": (report.get("winner") or {}).get("worker"),
            "winner_digest": (report.get("winner") or {}).get("digest"),
            "success_count": report.get("success_count"),
            "attempts_count": report.get("attempts_count"),
            "workers": [a.get("worker") for a in (report.get("all_results") or [])],
            "ts": report.get("ts"),
            "error": report.get("error"),
        }
        # احتفظ بآخر 200 قرار
        if len(ledger) > 200:
            for k in sorted(ledger.keys())[: len(ledger) - 200]:
                ledger.pop(k, None)
        self.node._save_state(state)


    async def submit_collective_summary(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        *,
        n_workers: int = 3,
        redundancy: int = 1,
        strategy: str = STRATEGY_MAJORITY,
        quorum: int = 2,
        timeout_per_task: float = 12.0,
        require_capabilities=None,
        worker_list: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        تلخيص جماعي لمصادر محلية مع إثبات:
          - كل مصدر له source_id + source_hash
          - يُوزَّع على العمال (مع redundancy اختيارية)
          - يُجمع الملخص والفائز عبر submit_job / أو دمج ملخصات المصادر
        sources: [{"source_id": str, "text": str}, ...]
        لا يصل للشبكة الخارجية — آمن افتراضياً.
        """
        from ai import mesh_task_protocol as mt

        query = (query or "").strip()
        sources = [s for s in (sources or []) if (s.get("text") or "").strip()]
        if not sources:
            return {"ok": False, "error": "no_sources", "query": query}

        workers = worker_list or self._rank_workers(
            require_capabilities=require_capabilities or ["text"],
            max_workers=max(n_workers, len(sources) * max(1, redundancy)),
        )
        job_id = f"csum_{uuid.uuid4().hex[:12]}"
        # بدون عمال: تنفيذ محلي على العقدة المنظمة (مناسب لتجربة المنتج)
        if not workers:
            per_source = []
            t0 = time.time()
            for si, src in enumerate(sources):
                sid = src.get("source_id") or f"src_{si}"
                text = (src.get("text") or "").strip()
                source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                local = mt.execute_summarize_chunk({
                    "source_id": sid, "text": text, "query": query,
                    "task_id": f"{job_id}_local_{si}",
                })
                per_source.append({
                    "source_id": sid,
                    "source_hash": source_hash,
                    "ok": bool(local.get("ok")),
                    "quorum_met": True,
                    "agreement": 1.0,
                    "summary": local.get("summary") or local.get("output"),
                    "winner_worker": getattr(self.node, "node_id", "local"),
                    "winner_digest": _result_digest(local),
                    "job_id": job_id,
                    "attempts": [],
                })
            ok_sources = [s for s in per_source if s.get("ok") and s.get("summary")]
            combined = " | ".join(f"[{s['source_id']}] {s['summary']}" for s in ok_sources)
            out = {
                "ok": len(ok_sources) > 0,
                "job_id": job_id,
                "kind": "collective_summary",
                "query": query,
                "sources_total": len(sources),
                "sources_ok": len(ok_sources),
                "combined_summary": combined,
                "per_source": per_source,
                "provenance": [
                    {"source_id": s["source_id"], "source_hash": s["source_hash"],
                     "worker": s.get("winner_worker"), "digest": s.get("winner_digest"),
                     "quorum_met": True}
                    for s in per_source
                ],
                "elapsed_ms": round((time.time() - t0) * 1000, 2),
                "ts": datetime.now(timezone.utc).isoformat(),
                "mode": "local_fallback",
            }
            self._jobs[job_id] = out
            try:
                self._persist_job_decision({
                    "job_id": job_id, "kind": "collective_summary", "ok": out["ok"],
                    "strategy": strategy, "quorum_required": quorum, "quorum_met": True,
                    "selection": {}, "winner": {}, "success_count": len(ok_sources),
                    "attempts_count": len(sources), "all_results": [], "ts": out["ts"],
                    "error": None if out["ok"] else "all_sources_failed",
                })
            except Exception:
                pass
            return out

        per_source = []
        t0 = time.time()

        # لكل مصدر: redundancy عمال (أو 1)
        red = max(1, min(int(redundancy), 3))
        for si, src in enumerate(sources):
            sid = src.get("source_id") or f"src_{si}"
            text = (src.get("text") or "").strip()
            source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            # اختر عمالاً لهذا المصدر
            picks = []
            for r in range(red):
                w = workers[(si + r) % len(workers)]
                picks.append(w)
            # فريدون
            seen = set()
            uniq = []
            for w in picks:
                pid = w.get("peer_id") or w.get("id")
                if pid in seen:
                    continue
                seen.add(pid)
                uniq.append(w)
            if not uniq:
                uniq = workers[:1]

            report = await self.submit_job(
                mt.KIND_SUMMARIZE,
                {
                    "job_id": job_id,
                    "source_id": sid,
                    "text": text,
                    "query": query,
                    "max_chars": int(src.get("max_chars") or 240),
                },
                n_workers=len(uniq),
                strategy=strategy,
                quorum=min(quorum, len(uniq)),
                timeout_per_task=timeout_per_task,
                retry_failed=1,
                worker_list=uniq,
                require_capabilities=require_capabilities or ["text"],
            )
            winner_res = ((report.get("winner") or {}).get("result")) or {}
            per_source.append({
                "source_id": sid,
                "source_hash": source_hash,
                "ok": report.get("ok"),
                "quorum_met": report.get("quorum_met"),
                "agreement": (report.get("selection") or {}).get("agreement"),
                "summary": winner_res.get("summary") or winner_res.get("output"),
                "winner_worker": (report.get("winner") or {}).get("worker"),
                "winner_digest": (report.get("winner") or {}).get("digest"),
                "job_id": report.get("job_id"),
                "attempts": report.get("all_results"),
            })

        ok_sources = [s for s in per_source if s.get("ok") and s.get("summary")]
        combined = " | ".join(
            f"[{s['source_id']}] {s['summary']}" for s in ok_sources
        )
        provenance = [
            {
                "source_id": s["source_id"],
                "source_hash": s["source_hash"],
                "worker": s.get("winner_worker"),
                "digest": s.get("winner_digest"),
                "quorum_met": s.get("quorum_met"),
            }
            for s in per_source
        ]
        out = {
            "ok": len(ok_sources) > 0,
            "job_id": job_id,
            "kind": "collective_summary",
            "query": query,
            "sources_total": len(sources),
            "sources_ok": len(ok_sources),
            "combined_summary": combined,
            "per_source": per_source,
            "provenance": provenance,
            "elapsed_ms": round((time.time() - t0) * 1000, 2),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._persist_job_decision({
                "job_id": job_id,
                "kind": "collective_summary",
                "ok": out["ok"],
                "strategy": strategy,
                "quorum_required": quorum,
                "quorum_met": all(s.get("quorum_met") for s in ok_sources) if ok_sources else False,
                "selection": {"agreement": None},
                "winner": {"worker": None, "digest": None},
                "success_count": len(ok_sources),
                "attempts_count": len(sources),
                "all_results": [{"worker": p.get("worker")} for p in provenance],
                "ts": out["ts"],
                "error": None if out["ok"] else "all_sources_failed",
            })
        except Exception:
            pass
        self._jobs[job_id] = out
        logger.info(
            f"📚 collective_summary {job_id} ok={out['ok']} "
            f"sources={out['sources_ok']}/{out['sources_total']}"
        )
        return out

    def list_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """قرارات محفوظة في حالة العقدة (دفتر خفيف)."""
        if not hasattr(self.node, "_load_state"):
            return []
        state = self.node._load_state()
        ledger = state.get("job_decisions") or {}
        items = list(ledger.values())
        items.sort(key=lambda x: x.get("ts") or "", reverse=True)
        return items[: max(1, limit)]


    async def submit_collective_search(
        self,
        query: str,
        corpus: List[Dict[str, Any]],
        *,
        top_k: int = 5,
        n_workers: int = 3,
        strategy: str = STRATEGY_MAJORITY,
        quorum: int = 1,
        timeout_per_task: float = 12.0,
        then_summarize: bool = False,
        require_capabilities=None,
        worker_list: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        بحث معرفي متعدد المصادر — corpus مُمرَّر فقط (لا HTTP خارجي / لا SSRF).
        يوزّع الوثائق على العمال، يدمج النتائج حسب score، ويعيد hashes + snippets.
        إن then_summarize=True يمرّر أفضل المصادر إلى submit_collective_summary.
        """
        from ai import mesh_task_protocol as mt

        query = (query or "").strip()
        corpus = [d for d in (corpus or []) if (d.get("text") or d.get("content") or "").strip()]
        if not query:
            return {"ok": False, "error": "empty_query"}
        if not corpus:
            return {"ok": False, "error": "empty_corpus"}

        workers = worker_list or self._rank_workers(
            require_capabilities=require_capabilities or ["text"],
            max_workers=max(1, n_workers),
        )
        workers = workers[: max(1, n_workers)]
        job_id = f"csearch_{uuid.uuid4().hex[:12]}"
        t0 = time.time()

        # توزيع الوثائق على العمال (شرائح)
        if workers:
            shards: List[List[Dict[str, Any]]] = [[] for _ in workers]
            for i, doc in enumerate(corpus):
                shards[i % len(workers)].append(doc)
            attempts = []
            for i, w in enumerate(workers):
                docs = shards[i]
                if not docs:
                    continue
                payload = {
                    "job_id": job_id,
                    "query": query,
                    "documents": [
                        {
                            "source_id": d.get("source_id") or d.get("id") or f"doc_{i}_{j}",
                            "text": d.get("text") or d.get("content") or "",
                        }
                        for j, d in enumerate(docs)
                    ],
                    "top_k": top_k,
                }
                att = await self._dispatch_one(w, mt.KIND_SEARCH, payload, timeout_per_task)
                attempts.append(att)
        else:
            # محلي على العقدة المنظمة
            local = mt.execute_search_chunk({
                "query": query,
                "documents": corpus,
                "top_k": top_k,
                "task_id": f"{job_id}_local",
            })
            attempts = [{
                "ok": bool(local.get("ok")),
                "worker": getattr(self.node, "node_id", "local"),
                "result": local,
                "error": local.get("error"),
                "rtt_ms": local.get("elapsed_ms"),
                "digest": _result_digest(local),
            }]

        # دمج الإصابات
        merged: Dict[str, Dict[str, Any]] = {}
        for att in attempts:
            if not att.get("ok"):
                continue
            hits = ((att.get("result") or {}).get("hits") or [])
            for h in hits:
                sid = h.get("source_id")
                if not sid:
                    continue
                prev = merged.get(sid)
                if prev is None or float(h.get("score") or 0) > float(prev.get("score") or 0):
                    entry = dict(h)
                    entry["worker"] = att.get("worker")
                    merged[sid] = entry
        ranked = sorted(merged.values(), key=lambda x: float(x.get("score") or 0), reverse=True)[: max(1, top_k)]

        out = {
            "ok": len(ranked) > 0,
            "job_id": job_id,
            "kind": "collective_search",
            "query": query,
            "corpus_size": len(corpus),
            "hit_count": len(ranked),
            "hits": ranked,
            "provenance": [
                {
                    "source_id": h.get("source_id"),
                    "source_hash": h.get("source_hash"),
                    "score": h.get("score"),
                    "worker": h.get("worker"),
                }
                for h in ranked
            ],
            "workers_used": [a.get("worker") for a in attempts],
            "attempts_ok": sum(1 for a in attempts if a.get("ok")),
            "elapsed_ms": round((time.time() - t0) * 1000, 2),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if not out["ok"]:
            out["error"] = "no_hits"

        # تلخيص اختياري لأفضل المصادر
        if then_summarize and ranked:
            # استرجع النص الكامل من corpus
            by_id = {
                (d.get("source_id") or d.get("id")): (d.get("text") or d.get("content") or "")
                for d in corpus
            }
            sources = []
            for h in ranked:
                sid = h.get("source_id")
                text = by_id.get(sid) or ""
                if text:
                    sources.append({"source_id": sid, "text": text})
            if sources:
                summary = await self.submit_collective_summary(
                    query,
                    sources,
                    n_workers=max(1, len(workers) or 1),
                    redundancy=1,
                    strategy=strategy,
                    quorum=min(quorum, max(1, len(workers) or 1)),
                    timeout_per_task=timeout_per_task,
                    worker_list=workers or None,
                    require_capabilities=require_capabilities or ["text"],
                )
                out["summary"] = {
                    "ok": summary.get("ok"),
                    "combined_summary": summary.get("combined_summary"),
                    "provenance": summary.get("provenance"),
                    "job_id": summary.get("job_id"),
                }

        try:
            self._persist_job_decision({
                "job_id": job_id,
                "kind": "collective_search",
                "ok": out["ok"],
                "strategy": strategy,
                "quorum_required": quorum,
                "quorum_met": out.get("attempts_ok", 0) >= max(1, min(quorum, len(workers) or 1)),
                "selection": {"agreement": None},
                "winner": {"worker": None, "digest": None},
                "success_count": out.get("hit_count"),
                "attempts_count": len(attempts),
                "all_results": [{"worker": a.get("worker")} for a in attempts],
                "ts": out["ts"],
                "error": out.get("error"),
            })
        except Exception:
            pass
        self._jobs[job_id] = out
        logger.info(
            f"🔎 collective_search {job_id} ok={out['ok']} hits={out.get('hit_count')} "
            f"corpus={len(corpus)}"
        )
        return out


    async def submit_web_task(
        self,
        url: str,
        *,
        n_workers: int = 1,
        strategy: str = STRATEGY_FIRST,
        quorum: int = 1,
        max_chars: int = 6000,
        timeout_per_task: float = 20.0,
        allowlist: Optional[List[str]] = None,
        worker_list: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """تكليف عامل/عمال بجلب صفحة HTTPS عامة (محمي من SSRF)."""
        from ai import mesh_task_protocol as mt
        url = (url or "").strip()
        if not url:
            return {"ok": False, "error": "empty_url"}
        payload = {
            "url": url,
            "max_chars": max_chars,
            "timeout": timeout_per_task,
        }
        if allowlist:
            payload["allowlist"] = list(allowlist)
        report = await self.submit_job(
            mt.KIND_WEB_FETCH,
            payload,
            n_workers=max(1, n_workers),
            strategy=strategy,
            quorum=max(1, quorum),
            timeout_per_task=timeout_per_task,
            retry_failed=1,
            worker_list=worker_list,
            require_capabilities=["web"],
        )
        report["task_type"] = "web_fetch"
        report["url"] = url
        return report

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        items = list(self._jobs.values())
        items.sort(key=lambda x: x.get("ts") or "", reverse=True)
        return items[: max(1, limit)]
