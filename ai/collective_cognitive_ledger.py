# -*- coding: utf-8 -*-
"""
Collective Cognitive Ledger (CCL)
=================================
يحوّل VCEN من التحقق من نتائج منفردة إلى:
  - ذاكرة جماعية قابلة للتدقيق (append-only)
  - نماذج جماعية تُدمج فقط من مطالبات model_update المقبولة
  - قرارات جماعية بتوقيع + hash + quorum

كل إدخال في السجل مرتبط بـ claim_id و result_hash ومسار القبول.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai.verifiable_cognitive_net import VerifiableCognitiveNet, canonical_hash, POLICY_VERSION

logger = logging.getLogger("CCL")


class CollectiveCognitiveLedger:
    def __init__(
        self,
        mesh_node,
        vcen: VerifiableCognitiveNet = None,
        quorum: int = 2,
        require_independent: bool = True,
        storage_dir: Path = None,
    ):
        self.node = mesh_node
        self.vcen = vcen or VerifiableCognitiveNet(
            mesh_node, quorum=quorum, require_independent=require_independent
        )
        self.quorum = self.vcen.quorum
        self.guard = None  # يُربط اختيارياً بـ ByzantineDecisionGuard
        root = storage_dir or (Path(getattr(mesh_node, "keys_dir", Path("."))).parent / "collective")
        self.storage_dir = Path(root)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.storage_dir / "collective_memory.json"
        self.model_path = self.storage_dir / "collective_model.json"
        self.decisions_path = self.storage_dir / "collective_decisions.json"
        self.audit_path = self.storage_dir / "audit_trail.jsonl"

        self.memory: List[Dict[str, Any]] = self._load_json(self.memory_path, [])
        self.model: Dict[str, Any] = self._load_json(
            self.model_path,
            {
                "version": 0,
                "weights": {},
                "history": [],
                "updated_at": None,
            },
        )
        self.decisions: Dict[str, Dict[str, Any]] = self._load_json(self.decisions_path, {})

    @staticmethod
    def _load_json(path: Path, default):
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                return default
        return default

    def _save_json(self, path: Path, data):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))

    def _audit(self, event: str, payload: Dict[str, Any]):
        line = {
            "event": event,
            "ts": datetime.now(timezone.utc).isoformat(),
            "node_id": self.node.node_id,
            **payload,
        }
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")

    # ------------------------------------------------------------------
    # ذاكرة جماعية
    # ------------------------------------------------------------------
    def ingest_accepted_claim(self, claim: Dict[str, Any], verdict: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        يُدخل مطالبة مقبولة فقط إلى الذاكرة الجماعية.
        إن لم يُمرَّر verdict يُعاد التقييم عبر VCEN.
        """
        verdict = verdict or self.vcen.evaluate_acceptance(claim)
        if not verdict.get("accepted"):
            self._audit("memory_reject", {"claim_id": claim.get("claim_id"), "verdict": verdict})
            return {"ok": False, "error": "claim_not_accepted", "verdict": verdict}

        entry = {
            "entry_id": f"mem_{uuid.uuid4().hex[:12]}",
            "claim_id": claim.get("claim_id"),
            "claim_type": claim.get("claim_type"),
            "kind": claim.get("kind"),
            "result_hash": claim.get("result_hash"),
            "executor_id": claim.get("executor_id"),
            "result": claim.get("result"),
            "verdict_summary": {
                "valid_attestations": verdict.get("valid_attestations"),
                "independent_attestations": verdict.get("independent_attestations"),
                "quorum_required": verdict.get("quorum_required"),
            },
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "ingested_by": self.node.node_id,
        }
        entry["entry_hash"] = canonical_hash({
            "entry_id": entry["entry_id"],
            "claim_id": entry["claim_id"],
            "result_hash": entry["result_hash"],
            "executor_id": entry["executor_id"],
        })
        self.memory.append(entry)
        self._save_json(self.memory_path, self.memory)
        self._audit("memory_ingest", {"entry_id": entry["entry_id"], "claim_id": entry["claim_id"]})
        return {"ok": True, "entry": entry}

    def memory_snapshot(self, limit: int = 50) -> Dict[str, Any]:
        items = self.memory[-limit:]
        return {
            "count": len(self.memory),
            "items": items,
            "chain_tip": items[-1]["entry_hash"] if items else None,
            "policy": POLICY_VERSION,
        }

    def verify_memory_integrity(self) -> Dict[str, Any]:
        bad = []
        for e in self.memory:
            expected = canonical_hash({
                "entry_id": e["entry_id"],
                "claim_id": e["claim_id"],
                "result_hash": e["result_hash"],
                "executor_id": e["executor_id"],
            })
            if e.get("entry_hash") != expected:
                bad.append(e.get("entry_id"))
        return {"ok": len(bad) == 0, "checked": len(self.memory), "corrupt": bad}

    # ------------------------------------------------------------------
    # نماذج جماعية
    # ------------------------------------------------------------------
    def apply_model_update_claim(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        يدمج تحديث نموذج فقط إذا قبلته VCEN كـ model_update.
        """
        if claim.get("claim_type") != "model_update":
            claim = dict(claim)
            claim["claim_type"] = "model_update"
        verdict = self.vcen.accept_model_update(claim)
        if not verdict.get("accepted"):
            self._audit("model_reject", {"claim_id": claim.get("claim_id"), "verdict": verdict})
            return {"ok": False, "error": "model_update_not_accepted", "verdict": verdict}

        result = claim.get("result") or {}
        weights = result.get("partial_weights") or result.get("weights") or {}
        if isinstance(weights, list):
            weights = {str(i): w for i, w in enumerate(weights)}

        # متوسط بسيط مع الحالة الحالية
        merged = dict(self.model.get("weights") or {})
        if not merged:
            merged = {k: float(v) for k, v in weights.items()}
        else:
            for k, v in weights.items():
                if k in merged:
                    merged[k] = (float(merged[k]) + float(v)) / 2.0
                else:
                    merged[k] = float(v)

        self.model["version"] = int(self.model.get("version") or 0) + 1
        self.model["weights"] = merged
        self.model["updated_at"] = datetime.now(timezone.utc).isoformat()
        hist = {
            "version": self.model["version"],
            "claim_id": claim.get("claim_id"),
            "result_hash": claim.get("result_hash"),
            "executor_id": claim.get("executor_id"),
            "loss": result.get("final_loss") or result.get("mean_loss"),
        }
        self.model.setdefault("history", []).append(hist)
        self.model["history"] = self.model["history"][-100:]
        self.model["model_hash"] = canonical_hash({
            "version": self.model["version"],
            "weights": merged,
        })
        self._save_json(self.model_path, self.model)
        # الذاكرة أيضاً
        self.ingest_accepted_claim(claim, verdict=verdict)
        self._audit("model_apply", {"version": self.model["version"], "claim_id": claim.get("claim_id")})
        return {"ok": True, "model": self.model_snapshot(), "verdict": verdict}

    def model_snapshot(self) -> Dict[str, Any]:
        return {
            "version": self.model.get("version"),
            "model_hash": self.model.get("model_hash"),
            "weights_keys": list((self.model.get("weights") or {}).keys())[:32],
            "history_len": len(self.model.get("history") or []),
            "updated_at": self.model.get("updated_at"),
        }

    # ------------------------------------------------------------------
    # قرارات جماعية
    # ------------------------------------------------------------------
    def propose_decision(
        self,
        title: str,
        payload: Dict[str, Any],
        threshold: int = None,
    ) -> Dict[str, Any]:
        """يطرح قراراً جماعياً — يحتاج عتبة أصوات موقّعة."""
        threshold = max(1, int(threshold or self.quorum))
        body = {
            "decision_id": f"dec_{uuid.uuid4().hex[:12]}",
            "title": title,
            "payload": payload,
            "payload_hash": canonical_hash(payload),
            "proposer_id": self.node.node_id,
            "threshold": threshold,
            "votes": [],
            "status": "open",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        body["proposer_signature"] = self.node.sign_message(json.dumps({
            "decision_id": body["decision_id"],
            "title": title,
            "payload_hash": body["payload_hash"],
            "proposer_id": body["proposer_id"],
        }, sort_keys=True))
        self.decisions[body["decision_id"]] = body
        self._save_json(self.decisions_path, self.decisions)
        self._audit("decision_propose", {"decision_id": body["decision_id"]})
        return body

    def vote_decision(self, decision_id: str, approve: bool = True) -> Dict[str, Any]:
        dec = self.decisions.get(decision_id)
        if not dec:
            return {"ok": False, "error": "unknown_decision"}
        if dec.get("status") != "open":
            return {"ok": False, "error": f"decision_{dec.get('status')}"}

        vote = {
            "voter_id": self.node.node_id,
            "approve": bool(approve),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        vote["signature"] = self.node.sign_message(json.dumps({
            "decision_id": decision_id,
            "voter_id": vote["voter_id"],
            "approve": vote["approve"],
            "payload_hash": dec.get("payload_hash"),
        }, sort_keys=True))

        # منع تكرار التصويت
        dec["votes"] = [v for v in dec.get("votes") or [] if v.get("voter_id") != self.node.node_id]
        dec["votes"].append(vote)
        self._save_json(self.decisions_path, self.decisions)
        self._audit("decision_vote", {"decision_id": decision_id, "approve": approve})
        return {"ok": True, "votes": len(dec["votes"]), "decision_id": decision_id}

    def finalize_decision(
        self,
        decision_id: str,
        voter_keys: Dict[str, bytes] = None,
        observed_leaders: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        يغلق القرار إن بلغ النصاب من الأصوات الموقّعة الصالحة.
        مع ByzantineDecisionGuard: نصاب الاتحاد الكامل + منع split-brain.
        """
        dec = self.decisions.get(decision_id)
        if not dec:
            return {"ok": False, "error": "unknown_decision"}

        valid_yes = []
        for v in dec.get("votes") or []:
            if not v.get("approve"):
                continue
            vid = v.get("voter_id")
            sig = v.get("signature")
            if not vid or not sig:
                continue
            if vid == self.node.node_id:
                pub = self.node._pub_pem().encode()
            elif voter_keys and vid in voter_keys:
                pub = voter_keys[vid]
            else:
                key_path = self.node.keys_dir / f"{vid}.pub"
                if not key_path.exists():
                    continue
                pub = key_path.read_bytes()
            payload = json.dumps({
                "decision_id": decision_id,
                "voter_id": vid,
                "approve": True,
                "payload_hash": dec.get("payload_hash"),
            }, sort_keys=True)
            if self.node.verify_signature(pub, payload, sig):
                valid_yes.append(v)

        threshold = int(dec.get("threshold") or self.quorum)
        if len(valid_yes) < threshold:
            return {
                "ok": False,
                "error": "quorum_not_met",
                "valid_yes": len(valid_yes),
                "threshold": threshold,
                "status": dec.get("status"),
            }

        # حماية بيزنطية / انقسام شبكة
        if self.guard is not None:
            gate = self.guard.safe_finalize_gate(
                dec,
                verified_yes_voters=[v.get("voter_id") for v in valid_yes],
                observed_leaders=observed_leaders,
            )
            if not gate.get("ok"):
                return {"ok": False, "error": gate.get("reason"), "guard": gate}

        dec["status"] = "accepted"
        dec["finalized_at"] = datetime.now(timezone.utc).isoformat()
        dec["valid_yes"] = len(valid_yes)
        dec["decision_hash"] = canonical_hash({
            "decision_id": decision_id,
            "payload_hash": dec["payload_hash"],
            "valid_yes": len(valid_yes),
            "status": "accepted",
        })
        self._save_json(self.decisions_path, self.decisions)

        # إدخال في الذاكرة الجماعية كقرار مدقّق
        mem_claim = {
            "claim_id": f"claim_dec_{decision_id}",
            "claim_type": "collective_decision",
            "kind": "decision",
            "result_hash": dec["decision_hash"],
            "executor_id": dec.get("proposer_id"),
            "executor_signature": dec.get("proposer_signature"),
            "result": {"decision_id": decision_id, "title": dec.get("title"), "payload": dec.get("payload")},
            "attestations": [],
        }
        # سجل مباشر في الذاكرة مع وسم قرار (تجاوز VCEN الضيق للقرارات الموقّعة جماعياً)
        entry = {
            "entry_id": f"mem_{uuid.uuid4().hex[:12]}",
            "claim_id": mem_claim["claim_id"],
            "claim_type": "collective_decision",
            "kind": "decision",
            "result_hash": dec["decision_hash"],
            "executor_id": dec.get("proposer_id"),
            "result": mem_claim["result"],
            "verdict_summary": {"valid_votes": len(valid_yes), "threshold": threshold},
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "ingested_by": self.node.node_id,
        }
        entry["entry_hash"] = canonical_hash({
            "entry_id": entry["entry_id"],
            "claim_id": entry["claim_id"],
            "result_hash": entry["result_hash"],
            "executor_id": entry["executor_id"],
        })
        self.memory.append(entry)
        self._save_json(self.memory_path, self.memory)
        self._audit("decision_finalize", {"decision_id": decision_id, "valid_yes": len(valid_yes)})
        return {"ok": True, "decision": dec, "memory_entry_id": entry["entry_id"]}

    # ------------------------------------------------------------------
    # مسار جماعي كامل: تنفيذ → تحقق VCEN → ذاكرة/نموذج
    # ------------------------------------------------------------------
    def collective_execute(
        self,
        kind: str,
        payload: Dict[str, Any],
        verifier_vcens: List[VerifiableCognitiveNet],
        as_model_update: bool = False,
    ) -> Dict[str, Any]:
        claim_type = "model_update" if as_model_update else "task_result"
        path = self.vcen.simulate_quorum_path(
            kind, payload, verifier_nodes=verifier_vcens, claim_type=claim_type
        )
        if not path.get("ok"):
            return {"ok": False, "stage": "vcen", "path": path}

        claim = self.vcen._claims.get(path["claim_id"]) or {}
        # استرجاع المطالبة من accepted
        if not claim and path["claim_id"] in self.vcen._accepted:
            acc = self.vcen._accepted[path["claim_id"]]
            claim = {
                "claim_id": path["claim_id"],
                "claim_type": claim_type,
                "kind": kind,
                "result_hash": path.get("result_hash"),
                "executor_id": self.node.node_id,
                "result": acc.get("result"),
                "executor_signature": "from_accepted",
                "attestations": [],
            }
            # للذاكرة نستخدم verdict الجاهز
            if as_model_update:
                # أعد بناء claim كامل من execute
                pass

        # أعد التنفيذ للحصول على claim كامل إن لزم
        if not claim.get("executor_signature") or claim.get("executor_signature") == "from_accepted":
            full = self.vcen.execute_and_claim(kind, payload, claim_type=claim_type)
            claim = full["claim"]
            for v in verifier_vcens:
                key_path = v.node.keys_dir / f"{self.node.node_id}.pub"
                key_path.write_text(self.node._pub_pem())
                self.node.keys_dir.joinpath(f"{v.node.node_id}.pub").write_text(v.node._pub_pem())
                att = v.attest_as_verifier(claim)
                claim.setdefault("attestations", []).append(att)

        if as_model_update:
            return self.apply_model_update_claim(claim)
        mem = self.ingest_accepted_claim(claim)
        return {"ok": mem.get("ok"), "stage": "memory", "memory": mem, "vcen": path}

    def full_audit_export(self) -> Dict[str, Any]:
        audit_lines = []
        if self.audit_path.exists():
            for line in self.audit_path.read_text(encoding="utf-8").splitlines():
                try:
                    audit_lines.append(json.loads(line))
                except Exception:
                    pass
        return {
            "memory": self.memory_snapshot(),
            "model": self.model_snapshot(),
            "decisions_open": sum(1 for d in self.decisions.values() if d.get("status") == "open"),
            "decisions_accepted": sum(1 for d in self.decisions.values() if d.get("status") == "accepted"),
            "audit_events": len(audit_lines),
            "audit_tail": audit_lines[-20:],
            "memory_integrity": self.verify_memory_integrity(),
        }
