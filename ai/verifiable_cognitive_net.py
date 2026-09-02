# -*- coding: utf-8 -*-
"""
Verifiable Cognitive Execution Network (VCEN)
=============================================
لا تُقبل أي نتيجة تنفيذ أو تحديث نموذج إلا إذا توفّر:
  1) توقيع المنفّذ (signature)
  2) Hash للنتيجة (result_hash)
  3) مصادقة مدقق مستقل واحد على الأقل (independent attestation)
  4) Quorum واضح من المدققين (threshold)

الرفض صريح عند غياب أي شرط.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("VCEN")

POLICY_VERSION = "vcen-v1"


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


class VerifiableCognitiveNet:
    def __init__(
        self,
        mesh_node,
        quorum: int = 2,
        require_independent: bool = True,
    ):
        """
        quorum: الحد الأدنى لعدد المصادقات المقبولة (بما فيها أو بدون المنفّذ حسب السياسة).
        require_independent: يجب أن يكون أحد المدققين ≠ المنفّذ.
        """
        self.node = mesh_node
        self.quorum = max(1, int(quorum))
        self.require_independent = bool(require_independent)
        self._claims: Dict[str, Dict[str, Any]] = {}
        self._accepted: Dict[str, Dict[str, Any]] = {}
        self._rejected: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # بناء مطالبة قابلة للتحقق (من جهة المنفّذ)
    # ------------------------------------------------------------------
    def build_claim(
        self,
        kind: str,
        result: Dict[str, Any],
        claim_type: str = "task_result",
        meta: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        claim_type: task_result | model_update
        """
        result_hash = canonical_hash(result)
        body = {
            "claim_id": f"cl_{uuid.uuid4().hex[:12]}",
            "policy": POLICY_VERSION,
            "claim_type": claim_type,
            "kind": kind,
            "result_hash": result_hash,
            "executor_id": self.node.node_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "meta": meta or {},
        }
        # لا نُضمّن النتيجة الكاملة في التوقيع إن كانت ضخمة — الهاش يكفي للربط
        sig_payload = {
            "claim_id": body["claim_id"],
            "policy": body["policy"],
            "claim_type": body["claim_type"],
            "kind": body["kind"],
            "result_hash": body["result_hash"],
            "executor_id": body["executor_id"],
            "ts": body["ts"],
        }
        body["executor_signature"] = self.node.sign_message(json.dumps(sig_payload, sort_keys=True))
        body["result"] = result
        body["attestations"] = []
        self._claims[body["claim_id"]] = body
        return body

    def _executor_sig_payload(self, claim: Dict[str, Any]) -> str:
        return json.dumps({
            "claim_id": claim["claim_id"],
            "policy": claim.get("policy"),
            "claim_type": claim.get("claim_type"),
            "kind": claim.get("kind"),
            "result_hash": claim["result_hash"],
            "executor_id": claim["executor_id"],
            "ts": claim.get("ts"),
        }, sort_keys=True)

    # ------------------------------------------------------------------
    # مدقق مستقل
    # ------------------------------------------------------------------
    def attest_as_verifier(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        يعيد مصادقة مدقق: يتحقق من الهاش والتوقيع ثم يوقّع attestation.
        يرفض إن كان المدقق هو نفس المنفّذ (عند تفعيل الاستقلالية في مرحلة القبول).
        """
        errors = []
        result = claim.get("result")
        if result is None:
            errors.append("missing_result")
        else:
            actual = canonical_hash(result)
            if actual != claim.get("result_hash"):
                errors.append("hash_mismatch")

        executor_id = claim.get("executor_id")
        sig = claim.get("executor_signature")
        if not executor_id or not sig:
            errors.append("missing_executor_signature")
        else:
            key_path = self.node.keys_dir / f"{executor_id}.pub"
            if executor_id == self.node.node_id:
                pub = self.node._pub_pem().encode()
            elif key_path.exists():
                pub = key_path.read_bytes()
            else:
                # لا مفتاح — لا يمكن التحقق المستقل
                errors.append("unknown_executor_key")
                pub = None
            if pub is not None:
                if not self.node.verify_signature(pub, self._executor_sig_payload(claim), sig):
                    errors.append("invalid_executor_signature")

        if errors:
            att = {
                "attestation_id": f"at_{uuid.uuid4().hex[:10]}",
                "claim_id": claim.get("claim_id"),
                "verifier_id": self.node.node_id,
                "accepted": False,
                "errors": errors,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            att["verifier_signature"] = self.node.sign_message(json.dumps({
                "attestation_id": att["attestation_id"],
                "claim_id": att["claim_id"],
                "verifier_id": att["verifier_id"],
                "accepted": False,
                "errors": errors,
            }, sort_keys=True))
            return att

        att_body = {
            "attestation_id": f"at_{uuid.uuid4().hex[:10]}",
            "claim_id": claim["claim_id"],
            "verifier_id": self.node.node_id,
            "accepted": True,
            "result_hash": claim["result_hash"],
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        att_body["verifier_signature"] = self.node.sign_message(json.dumps({
            "attestation_id": att_body["attestation_id"],
            "claim_id": att_body["claim_id"],
            "verifier_id": att_body["verifier_id"],
            "accepted": True,
            "result_hash": att_body["result_hash"],
        }, sort_keys=True))
        return att_body

    def _verify_attestation(self, claim: Dict[str, Any], att: Dict[str, Any]) -> Tuple[bool, str]:
        if not att.get("accepted"):
            return False, "attestation_rejected"
        if att.get("claim_id") != claim.get("claim_id"):
            return False, "claim_id_mismatch"
        if att.get("result_hash") and att.get("result_hash") != claim.get("result_hash"):
            return False, "attestation_hash_mismatch"
        vid = att.get("verifier_id")
        sig = att.get("verifier_signature")
        if not vid or not sig:
            return False, "missing_verifier_signature"
        key_path = self.node.keys_dir / f"{vid}.pub"
        if vid == self.node.node_id:
            pub = self.node._pub_pem().encode()
        elif key_path.exists():
            pub = key_path.read_bytes()
        else:
            return False, "unknown_verifier_key"
        payload = json.dumps({
            "attestation_id": att.get("attestation_id"),
            "claim_id": att.get("claim_id"),
            "verifier_id": vid,
            "accepted": True,
            "result_hash": att.get("result_hash"),
        }, sort_keys=True)
        if not self.node.verify_signature(pub, payload, sig):
            return False, "invalid_verifier_signature"
        return True, "ok"

    # ------------------------------------------------------------------
    # قبول / رفض حسب السياسة
    # ------------------------------------------------------------------
    def evaluate_acceptance(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        يقيّم هل المطالبة مستوفية: توقيع + hash + مدقق مستقل + quorum.
        لا يغيّر الحالة — للفحص فقط.
        """
        reasons_reject = []
        reasons_ok = []

        # 1) توقيع المنفّذ
        if not claim.get("executor_signature"):
            reasons_reject.append("no_executor_signature")
        else:
            executor_id = claim.get("executor_id")
            key_path = self.node.keys_dir / f"{executor_id}.pub"
            if executor_id == self.node.node_id:
                pub = self.node._pub_pem().encode()
            elif key_path.exists():
                pub = key_path.read_bytes()
            else:
                pub = None
                reasons_reject.append("unknown_executor_key")
            if pub is not None:
                if self.node.verify_signature(pub, self._executor_sig_payload(claim), claim["executor_signature"]):
                    reasons_ok.append("executor_signature_valid")
                else:
                    reasons_reject.append("executor_signature_invalid")

        # 2) Hash
        if not claim.get("result_hash"):
            reasons_reject.append("no_result_hash")
        elif claim.get("result") is not None:
            if canonical_hash(claim["result"]) == claim["result_hash"]:
                reasons_ok.append("result_hash_valid")
            else:
                reasons_reject.append("result_hash_mismatch")
        else:
            # بدون نتيجة مرفقة نثق بالهاش فقط إن وُجدت مصادقات
            reasons_ok.append("result_hash_present_unverified_body")

        # 3) مصادقات
        valid_atts = []
        independent = []
        for att in claim.get("attestations") or []:
            ok, why = self._verify_attestation(claim, att)
            if ok:
                valid_atts.append(att)
                if att.get("verifier_id") != claim.get("executor_id"):
                    independent.append(att)
            else:
                reasons_reject.append(f"bad_attestation:{why}")

        if len(valid_atts) >= self.quorum:
            reasons_ok.append(f"quorum_met:{len(valid_atts)}>={self.quorum}")
        else:
            reasons_reject.append(f"quorum_not_met:{len(valid_atts)}<{self.quorum}")

        if self.require_independent:
            if independent:
                reasons_ok.append("independent_verifier_present")
            else:
                reasons_reject.append("no_independent_verifier")

        accepted = len(reasons_reject) == 0
        return {
            "accepted": accepted,
            "claim_id": claim.get("claim_id"),
            "claim_type": claim.get("claim_type"),
            "valid_attestations": len(valid_atts),
            "independent_attestations": len(independent),
            "quorum_required": self.quorum,
            "reasons_ok": reasons_ok,
            "reasons_reject": reasons_reject,
        }

    def accept_or_reject(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """قبول نهائي أو رفض صريح مع تسجيل."""
        verdict = self.evaluate_acceptance(claim)
        record = {
            **verdict,
            "ts": datetime.now(timezone.utc).isoformat(),
            "executor_id": claim.get("executor_id"),
            "result_hash": claim.get("result_hash"),
            "kind": claim.get("kind"),
        }
        if verdict["accepted"]:
            self._accepted[claim["claim_id"]] = {
                "claim": {
                    "claim_id": claim["claim_id"],
                    "claim_type": claim.get("claim_type"),
                    "kind": claim.get("kind"),
                    "result_hash": claim.get("result_hash"),
                    "executor_id": claim.get("executor_id"),
                },
                "verdict": verdict,
                "result": claim.get("result"),
            }
            # تحديث سجل الشبكة
            try:
                state = self.node._load_state()
                state.setdefault("vcen_accepted", {})[claim["claim_id"]] = record
                self.node._save_state(state)
            except Exception:
                pass
            logger.info(f"✅ VCEN accepted claim {claim.get('claim_id')}")
        else:
            self._rejected.append(record)
            logger.warning(f"❌ VCEN rejected claim {claim.get('claim_id')}: {verdict['reasons_reject']}")
        return record

    def accept_model_update(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """مسار مخصص لتحديثات النموذج — نفس السياسة مع claim_type=model_update."""
        if claim.get("claim_type") != "model_update":
            claim = dict(claim)
            claim["claim_type"] = "model_update"
        return self.accept_or_reject(claim)

    def add_attestation(self, claim_id: str, attestation: Dict[str, Any]) -> Dict[str, Any]:
        claim = self._claims.get(claim_id)
        if not claim:
            return {"ok": False, "error": "unknown_claim"}
        claim.setdefault("attestations", []).append(attestation)
        return {"ok": True, "attestations": len(claim["attestations"])}

    # ------------------------------------------------------------------
    # مسار تنفيذي موحّد: نفّذ → claim → (مدققون) → قبول/رفض
    # ------------------------------------------------------------------
    def execute_and_claim(self, kind: str, payload: Dict[str, Any], claim_type: str = "task_result") -> Dict[str, Any]:
        from ai import mesh_task_protocol as mt
        result = mt.dispatch_task(kind, payload or {})
        if result is None:
            return {"ok": False, "error": f"unknown_kind:{kind}"}
        claim = self.build_claim(kind, result, claim_type=claim_type)
        return {"ok": True, "claim": claim, "result": result}

    def simulate_quorum_path(
        self,
        kind: str,
        payload: Dict[str, Any],
        verifier_nodes: List[Any],
        claim_type: str = "task_result",
    ) -> Dict[str, Any]:
        """
        مسار كامل للاختبار: منفّذ واحد + قائمة مدققين (كائنات VerifiableCognitiveNet أخرى).
        """
        exec_out = self.execute_and_claim(kind, payload, claim_type=claim_type)
        if not exec_out.get("ok"):
            return exec_out
        claim = exec_out["claim"]
        # شارك مفاتيح المنفّذ مع المدققين (محاكاة اكتشاف)
        exec_pub = self.node._pub_pem()
        for v in verifier_nodes:
            key_path = v.node.keys_dir / f"{self.node.node_id}.pub"
            key_path.write_text(exec_pub)
            # ومفتاح المدقق للمنفّذ
            self.node.keys_dir.joinpath(f"{v.node.node_id}.pub").write_text(v.node._pub_pem())

        for v in verifier_nodes:
            att = v.attest_as_verifier(claim)
            claim.setdefault("attestations", []).append(att)
            # مدقق يرفض لا يُحتسب
            if att.get("accepted"):
                # تبادل مفتاح المدقق للتحقق لاحقاً
                self.node.keys_dir.joinpath(f"{v.node.node_id}.pub").write_text(v.node._pub_pem())

        verdict = self.accept_or_reject(claim)
        return {
            "ok": verdict.get("accepted"),
            "claim_id": claim.get("claim_id"),
            "verdict": verdict,
            "attestations": len(claim.get("attestations") or []),
            "result_hash": claim.get("result_hash"),
        }
