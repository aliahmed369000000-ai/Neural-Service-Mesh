# -*- coding: utf-8 -*-
"""
Byzantine / Partition-Safe Decision Guard
=========================================
يحمي القرارات الجماعية وانتخاب القائد من:
  - قائد مزدوج (split-brain)
  - أصوات من عقد غير معروفة أو مكررة
  - term متلاعب / stale
  - قبول قرار من أقلية في قسم صغير من الشبكة

القواعد:
  1) العضوية الصريحة (roster) — لا صوت خارج القائمة
  2) majority = floor(n/2)+1 من حجم الاتحاد المعروف (ليس القسم المحلي فقط)
  3) رفض term أقدم من last_seen_term
  4) قائد واحد لكل term؛ تعارض القادة → rejected_conflict
  5) القرار يُقبل فقط إذا valid_yes >= majority من roster الكامل
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("ByzantineGuard")


class ByzantineDecisionGuard:
    def __init__(self, mesh_node, storage_dir: Path = None):
        self.node = mesh_node
        root = storage_dir or (Path(getattr(mesh_node, "keys_dir", Path("."))).parent / "guard")
        self.storage_dir = Path(root)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.storage_dir / "guard_state.json"
        self.state = self._load({
            "roster": [],              # قائمة أعضاء الاتحاد الموثوقين
            "last_seen_term": 0,
            "accepted_leaders": {},    # term -> leader_id
            "rejected_events": [],
            "partition_epoch": 0,
        })

    def _load(self, default):
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                return default
        return default

    def _save(self):
        self.path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False, default=str))

    def _reject(self, reason: str, detail: Dict[str, Any] = None) -> Dict[str, Any]:
        ev = {
            "reason": reason,
            "detail": detail or {},
            "ts": datetime.now(timezone.utc).isoformat(),
            "node_id": self.node.node_id,
        }
        self.state.setdefault("rejected_events", []).append(ev)
        self.state["rejected_events"] = self.state["rejected_events"][-100:]
        self._save()
        logger.warning(f"🛡️ Guard reject: {reason}")
        return {"ok": False, "accepted": False, "reason": reason, "detail": detail or {}}

    # ------------------------------------------------------------------
    # العضوية
    # ------------------------------------------------------------------
    def set_roster(self, member_ids: List[str]) -> Dict[str, Any]:
        roster = sorted(set(member_ids))
        if self.node.node_id not in roster:
            roster.append(self.node.node_id)
            roster = sorted(set(roster))
        self.state["roster"] = roster
        self._save()
        return {"ok": True, "roster": roster, "majority": self.majority()}

    def add_member(self, node_id: str) -> Dict[str, Any]:
        r = set(self.state.get("roster") or [])
        r.add(node_id)
        return self.set_roster(list(r))

    def roster(self) -> List[str]:
        return list(self.state.get("roster") or [self.node.node_id])

    def majority(self) -> int:
        n = max(1, len(self.roster()))
        return n // 2 + 1

    def is_member(self, node_id: str) -> bool:
        return node_id in set(self.roster())

    # ------------------------------------------------------------------
    # حماية التصويت / القيادة
    # ------------------------------------------------------------------
    def validate_vote(
        self,
        voter_id: str,
        term: int,
        candidate_id: str,
        signature_ok: bool,
    ) -> Dict[str, Any]:
        if not signature_ok:
            return self._reject("invalid_vote_signature", {"voter_id": voter_id})
        if not self.is_member(voter_id):
            return self._reject("voter_not_in_roster", {"voter_id": voter_id})
        if candidate_id and not self.is_member(candidate_id):
            return self._reject("candidate_not_in_roster", {"candidate_id": candidate_id})
        last = int(self.state.get("last_seen_term") or 0)
        if term < last:
            return self._reject("stale_term", {"term": term, "last_seen_term": last})
        if term > last:
            self.state["last_seen_term"] = term
            self._save()
        return {"ok": True, "accepted": True, "majority": self.majority()}

    def validate_leader_claim(
        self,
        term: int,
        leader_id: str,
        vote_count: int,
        signature_ok: bool = True,
    ) -> Dict[str, Any]:
        if not signature_ok:
            return self._reject("invalid_leader_signature", {"leader_id": leader_id})
        if not self.is_member(leader_id):
            return self._reject("leader_not_in_roster", {"leader_id": leader_id})
        last = int(self.state.get("last_seen_term") or 0)
        if term < last:
            return self._reject("stale_leader_term", {"term": term, "last": last})

        maj = self.majority()
        if vote_count < maj:
            return self._reject("leader_quorum_not_met", {
                "vote_count": vote_count, "majority": maj, "roster_size": len(self.roster()),
            })

        accepted = self.state.setdefault("accepted_leaders", {})
        prev = accepted.get(str(term))
        if prev and prev != leader_id:
            return self._reject("split_brain_conflict", {
                "term": term, "existing_leader": prev, "claimant": leader_id,
            })

        accepted[str(term)] = leader_id
        self.state["last_seen_term"] = max(last, term)
        self._save()
        return {
            "ok": True,
            "accepted": True,
            "term": term,
            "leader_id": leader_id,
            "majority": maj,
        }

    def detect_split_brain(
        self,
        observations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        observations: [{term, leader_id, from_partition?}, ...]
        يكشف قائدين مختلفين لنفس term.
        """
        by_term: Dict[int, Set[str]] = {}
        for obs in observations:
            t = int(obs.get("term") or 0)
            lid = obs.get("leader_id")
            if not lid:
                continue
            by_term.setdefault(t, set()).add(lid)
        conflicts = {t: list(leaders) for t, leaders in by_term.items() if len(leaders) > 1}
        if conflicts:
            self.state["partition_epoch"] = int(self.state.get("partition_epoch") or 0) + 1
            self._save()
            return {
                "split_brain": True,
                "conflicts": conflicts,
                "partition_epoch": self.state["partition_epoch"],
                "action": "reject_all_conflicting_terms",
            }
        return {"split_brain": False, "conflicts": {}, "partition_epoch": self.state.get("partition_epoch")}

    # ------------------------------------------------------------------
    # حماية القرارات الجماعية
    # ------------------------------------------------------------------
    def validate_decision_votes(
        self,
        decision: Dict[str, Any],
        verified_yes_voters: List[str],
    ) -> Dict[str, Any]:
        """
        verified_yes_voters: معرفات الأصوات التي تم التحقق من توقيعها مسبقاً.
        يُقبل القرار فقط إذا عدد المصوّتين الموثوقين من الـ roster >= majority.
        """
        roster = set(self.roster())
        unique = []
        seen = set()
        for vid in verified_yes_voters:
            if vid in seen:
                continue
            seen.add(vid)
            if vid not in roster:
                return self._reject("decision_voter_not_in_roster", {"voter_id": vid})
            unique.append(vid)

        maj = self.majority()
        if len(unique) < maj:
            return self._reject("decision_quorum_not_met", {
                "valid_yes": len(unique),
                "majority": maj,
                "roster_size": len(roster),
                "partition_risk": True,
            })

        # منع إغلاق قرار بعتبة محلية أصغر من majority الاتحاد
        threshold = int(decision.get("threshold") or 0)
        if threshold and threshold < maj:
            return self._reject("threshold_below_federation_majority", {
                "threshold": threshold, "majority": maj,
            })

        return {
            "ok": True,
            "accepted": True,
            "valid_yes": len(unique),
            "majority": maj,
            "voters": unique,
        }

    def safe_finalize_gate(
        self,
        decision: Dict[str, Any],
        verified_yes_voters: List[str],
        observed_leaders: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """بوابة واحدة قبل finalize_decision: انقسام + نصاب الاتحاد."""
        if observed_leaders:
            sb = self.detect_split_brain(observed_leaders)
            if sb.get("split_brain"):
                return self._reject("split_brain_blocks_decision", sb)
        return self.validate_decision_votes(decision, verified_yes_voters)

    def status(self) -> Dict[str, Any]:
        return {
            "roster": self.roster(),
            "roster_size": len(self.roster()),
            "majority": self.majority(),
            "last_seen_term": self.state.get("last_seen_term"),
            "accepted_leaders": self.state.get("accepted_leaders"),
            "partition_epoch": self.state.get("partition_epoch"),
            "rejected_count": len(self.state.get("rejected_events") or []),
        }
