# -*- coding: utf-8 -*-
"""
Private Federated Learning (PFL)
================================
حماية البيانات الخاصة أثناء التعلم الجماعي:
  - ممنوع إرسال العينات الخام (raw samples) خارج العقدة
  - يُشارك فقط: تحديثات أوزان/تدرجات بعد قصّ (clip) + ضوضاء اختيارية
  - تجميع آمن مبسّط: مساهمات مقنّعة تُلغى عند الجمع (mask summation)
  - لا يُقبل التحديث الجماعي إلا عبر مسار VCEN/CCL عند التفعيل

هذا ليس بديلاً كاملاً عن DP-SGD أو SMPC الإنتاجي، لكنه يفرض سياسة
«لا بيانات خام على السلك» داخل NSM.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("PrivateFL")

POLICY_VERSION = "pfl-v1"


class PrivacyViolation(Exception):
    pass


def _hash_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


class PrivateFederatedLearning:
    def __init__(
        self,
        mesh_node,
        clip_norm: float = 1.0,
        noise_multiplier: float = 0.05,
        max_payload_fields: int = 64,
        allow_raw_data: bool = False,
    ):
        self.node = mesh_node
        self.clip_norm = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.max_payload_fields = int(max_payload_fields)
        self.allow_raw_data = bool(allow_raw_data)
        self._round_masks: Dict[str, List[float]] = {}
        self._local_stats: Dict[str, Any] = {"rounds": 0, "violations": 0}

    # ------------------------------------------------------------------
    # سياسة: رفض البيانات الخام
    # ------------------------------------------------------------------
    FORBIDDEN_KEYS = {
        "raw_data", "samples", "dataset", "x_train", "y_train",
        "images", "texts", "records", "rows", "examples", "batch_raw",
    }

    def sanitize_outgoing(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """يزيل أي حقول بيانات خام ويمنع إرسالها."""
        if self.allow_raw_data:
            return dict(payload)
        def is_forbidden(k: str) -> bool:
            kl = k.lower()
            if kl in self.FORBIDDEN_KEYS:
                return True
            # raw_* ما عدا أعلام السياسة مثل raw_data_included
            if kl.startswith("raw_") and kl not in ("raw_data_included",):
                return True
            return False

        bad = [k for k in payload.keys() if is_forbidden(k)]
        if bad:
            self._local_stats["violations"] = int(self._local_stats.get("violations") or 0) + 1
            raise PrivacyViolation(f"forbidden_raw_fields:{bad}")
        clean = {k: v for k, v in payload.items() if not is_forbidden(k)}
        return clean

    def assert_no_raw(self, payload: Dict[str, Any]) -> None:
        self.sanitize_outgoing(payload)

    # ------------------------------------------------------------------
    # قصّ وضوضاء على التدرجات/الأوزان
    # ------------------------------------------------------------------
    def clip_vector(self, values: List[float]) -> List[float]:
        if not values:
            return []
        norm = math.sqrt(sum(float(v) ** 2 for v in values)) or 1.0
        scale = min(1.0, self.clip_norm / norm)
        return [float(v) * scale for v in values]

    def add_gaussian_noise(self, values: List[float], seed: int = None) -> List[float]:
        rng = random.Random(seed)
        sigma = self.noise_multiplier * self.clip_norm
        return [float(v) + rng.gauss(0.0, sigma) for v in values]

    def protect_update(
        self,
        weights_or_grads: List[float],
        add_noise: bool = True,
        seed: int = None,
    ) -> Dict[str, Any]:
        clipped = self.clip_vector(list(weights_or_grads)[: self.max_payload_fields])
        protected = self.add_gaussian_noise(clipped, seed=seed) if add_noise else clipped
        return {
            "update": [round(v, 8) for v in protected],
            "dim": len(protected),
            "clip_norm": self.clip_norm,
            "noise_multiplier": self.noise_multiplier if add_noise else 0.0,
            "policy": POLICY_VERSION,
            "raw_data_included": False,
        }

    # ------------------------------------------------------------------
    # تجميع آمن مبسّط (masks تُلغي عند الجمع)
    # ------------------------------------------------------------------
    def generate_pairwise_masks(
        self,
        round_id: str,
        my_id: str,
        peer_ids: List[str],
        dim: int,
        seed_key: str = "",
    ) -> List[float]:
        """
        لكل زوج (me, peer): قناع عشوائي مشترك من seed مشتق.
        المساهمة النهائية = update + sum(masks) بحيث تُلغى الأقنعة عند جمع كل العقد.
        """
        masks = [0.0] * dim
        for peer in peer_ids:
            if peer == my_id:
                continue
            pair = tuple(sorted([my_id, peer]))
            seed_material = f"{round_id}|{pair[0]}|{pair[1]}|{seed_key}"
            seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
            rng = random.Random(seed)
            for i in range(dim):
                val = rng.uniform(-1.0, 1.0)
                # الطرف الأصغر lex يضيف، الأكبر يطرح → المجموع صفر
                if my_id == pair[0]:
                    masks[i] += val
                else:
                    masks[i] -= val
        self._round_masks[round_id] = masks
        return masks

    def mask_update(self, update: List[float], mask: List[float]) -> List[float]:
        n = min(len(update), len(mask))
        return [update[i] + mask[i] for i in range(n)]

    def secure_aggregate(self, masked_updates: List[List[float]]) -> List[float]:
        """جمع التحديثات المقنّعة — الأقنعة تُلغى إن شارك كل الأطراف."""
        if not masked_updates:
            return []
        dim = min(len(u) for u in masked_updates)
        total = [0.0] * dim
        for u in masked_updates:
            for i in range(dim):
                total[i] += float(u[i])
        n = len(masked_updates)
        return [v / n for v in total]

    # ------------------------------------------------------------------
    # جولة تعلم خاصة محلية
    # ------------------------------------------------------------------
    def local_private_train_step(
        self,
        seed_weights: List[float] = None,
        steps: int = 3,
        # البيانات تبقى محلية — لا تُمرَّر للخارج
        local_batch_size: int = 8,
    ) -> Dict[str, Any]:
        """
        يحاكي خطوة تدريب محلية على بيانات لا تغادر العقدة.
        يُرجع فقط تحديثاً محمياً (clip + noise) بدون عينات.
        """
        dim = len(seed_weights) if seed_weights else 8
        w = [float(x) for x in (seed_weights or [0.1] * dim)][: self.max_payload_fields]
        # «تدريب» محلي وهمي يعتمد على seed العقدة فقط — لا batch خام
        rng = random.Random(hashlib.sha256(f"{self.node.node_id}:{time.time()}".encode()).hexdigest())
        for _ in range(max(1, steps)):
            grad = [rng.uniform(-0.1, 0.1) for _ in w]
            # لا نستخدم local_batch_size إلا كمعامل إحصائي معلن
            for i in range(len(w)):
                w[i] -= 0.05 * grad[i]
        protected = self.protect_update(w, add_noise=True)
        out = {
            "ok": True,
            "node_id": self.node.node_id,
            "partial_weights": protected["update"],
            "update_meta": {
                "clip_norm": protected["clip_norm"],
                "noise_multiplier": protected["noise_multiplier"],
                "policy": POLICY_VERSION,
                "local_batch_size_declared": local_batch_size,
                "raw_data_included": False,
            },
            "final_loss": round(abs(sum(protected["update"])) / max(len(protected["update"]), 1) * 0.1, 6),
            "task_id": f"pfl_{uuid.uuid4().hex[:10]}",
        }
        # تحقق ذاتي
        self.assert_no_raw(out)
        self._local_stats["rounds"] = int(self._local_stats.get("rounds") or 0) + 1
        return out

    def build_private_share(
        self,
        round_id: str,
        peer_ids: List[str],
        seed_weights: List[float] = None,
        steps: int = 3,
    ) -> Dict[str, Any]:
        """خطوة محلية + قناع زوجي للمساهمة في التجميع الآمن."""
        local = self.local_private_train_step(seed_weights=seed_weights, steps=steps)
        update = local["partial_weights"]
        mask = self.generate_pairwise_masks(round_id, self.node.node_id, peer_ids, dim=len(update))
        masked = self.mask_update(update, mask)
        share = {
            "round_id": round_id,
            "node_id": self.node.node_id,
            "masked_update": [round(v, 8) for v in masked],
            "dim": len(masked),
            "policy": POLICY_VERSION,
            "raw_data_included": False,
            "update_meta": local["update_meta"],
            "final_loss": local["final_loss"],
            "task_id": local["task_id"],
        }
        self.assert_no_raw(share)
        return share

    def aggregate_shares(self, shares: List[Dict[str, Any]]) -> Dict[str, Any]:
        """يجمع المساهمات المقنّعة ويرفض أي سهم يحتوي بيانات خام."""
        clean = []
        for s in shares:
            try:
                self.assert_no_raw(s)
            except PrivacyViolation as e:
                logger.warning(f"drop share: {e}")
                continue
            if s.get("raw_data_included"):
                continue
            mu = s.get("masked_update")
            if isinstance(mu, list) and mu:
                clean.append(mu)
        if not clean:
            return {"ok": False, "error": "no_valid_shares"}
        agg = self.secure_aggregate(clean)
        result = {
            "ok": True,
            "partial_weights": [round(v, 8) for v in agg],
            "n_shares": len(clean),
            "policy": POLICY_VERSION,
            "raw_data_included": False,
            "mean_loss": None,
        }
        losses = [s.get("final_loss") for s in shares if s.get("final_loss") is not None]
        if losses:
            result["mean_loss"] = round(sum(losses) / len(losses), 6)
        return result

    def private_round_to_vcen_claim(
        self,
        vcen,
        shares: List[Dict[str, Any]],
        verifier_vcens: List = None,
    ) -> Dict[str, Any]:
        """تجميع خاص ثم مطالبة VCEN من نوع model_update."""
        agg = self.aggregate_shares(shares)
        if not agg.get("ok"):
            return agg
        claim = vcen.build_claim(
            "submodel_train",
            {
                "ok": True,
                "partial_weights": agg["partial_weights"],
                "final_loss": agg.get("mean_loss"),
                "n_shares": agg["n_shares"],
                "policy": POLICY_VERSION,
                "raw_data_included": False,
            },
            claim_type="model_update",
            meta={"privacy_policy": POLICY_VERSION},
        )
        if verifier_vcens:
            for v in verifier_vcens:
                key_path = v.node.keys_dir / f"{self.node.node_id}.pub"
                key_path.write_text(self.node._pub_pem())
                self.node.keys_dir.joinpath(f"{v.node.node_id}.pub").write_text(v.node._pub_pem())
                att = v.attest_as_verifier(claim)
                claim.setdefault("attestations", []).append(att)
            verdict = vcen.accept_model_update(claim)
            return {"ok": verdict.get("accepted"), "claim": claim, "verdict": verdict, "aggregate": agg}
        return {"ok": True, "claim": claim, "aggregate": agg}

    def stats(self) -> Dict[str, Any]:
        return {
            "policy": POLICY_VERSION,
            "clip_norm": self.clip_norm,
            "noise_multiplier": self.noise_multiplier,
            "allow_raw_data": self.allow_raw_data,
            **self._local_stats,
        }
