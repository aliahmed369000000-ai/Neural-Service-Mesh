# -*- coding: utf-8 -*-
"""
Leader Election & Round Continuity (بدون منسّق ثابت)
====================================================
- انتخاب قائد مؤقت حسب term + أصوات موقّعة
- تسليم القيادة (graceful handoff)
- كشف فشل القائد عبر انتهاء lease / غياب heartbeat
- استكمال الجولة (round journal) بعد فشل القائد بقائد جديد

الهدف: اتحاد عقد يدير نفسه دون اعتماد على coordinator ثابت.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("LeaderElection")


class LeaderElection:
    def __init__(
        self,
        mesh_node,
        lease_seconds: float = 15.0,
        storage_dir: Path = None,
    ):
        self.node = mesh_node
        self.lease_seconds = float(lease_seconds)
        root = storage_dir or (Path(getattr(mesh_node, "keys_dir", Path("."))).parent / "leader")
        self.storage_dir = Path(root)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.storage_dir / "leader_state.json"
        self.rounds_path = self.storage_dir / "round_journal.json"
        self.state = self._load(self.state_path, {
            "term": 0,
            "leader_id": None,
            "voted_for": None,
            "lease_until": 0.0,
            "last_heartbeat": 0.0,
            "role": "follower",  # follower | candidate | leader
        })
        self.rounds: Dict[str, Dict[str, Any]] = self._load(self.rounds_path, {})
        self.guard = None  # ByzantineDecisionGuard اختياري

    @staticmethod
    def _load(path: Path, default):
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                return default
        return default

    def _save_state(self):
        self.state_path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False))

    def _save_rounds(self):
        self.rounds_path.write_text(json.dumps(self.rounds, indent=2, ensure_ascii=False, default=str))

    def now(self) -> float:
        return time.time()

    # ------------------------------------------------------------------
    # انتخاب
    # ------------------------------------------------------------------
    def is_leader_alive(self) -> bool:
        if not self.state.get("leader_id"):
            return False
        return self.now() < float(self.state.get("lease_until") or 0)

    def current_leader(self) -> Optional[str]:
        if self.is_leader_alive():
            return self.state.get("leader_id")
        return None

    def start_election(self, known_peers: List[str] = None) -> Dict[str, Any]:
        """
        يبدأ term جديد ويصوّت لنفسه. known_peers: معرفات للمعلومات فقط في الوضع المحلي.
        في الإنتاج تُرسل VoteRequest عبر الشبكة؛ هنا نبني الطلب الموقّع.
        """
        self.state["term"] = int(self.state.get("term") or 0) + 1
        self.state["role"] = "candidate"
        self.state["voted_for"] = self.node.node_id
        self.state["leader_id"] = None
        self.state["lease_until"] = 0.0
        self._save_state()

        req = {
            "type": "vote_request",
            "term": self.state["term"],
            "candidate_id": self.node.node_id,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        req["signature"] = self.node.sign_message(json.dumps({
            "type": "vote_request",
            "term": req["term"],
            "candidate_id": req["candidate_id"],
        }, sort_keys=True))

        # صوت ذاتي
        votes = [{
            "voter_id": self.node.node_id,
            "term": self.state["term"],
            "grant": True,
            "signature": self.node.sign_message(json.dumps({
                "type": "vote_response",
                "term": self.state["term"],
                "voter_id": self.node.node_id,
                "grant": True,
                "candidate_id": self.node.node_id,
            }, sort_keys=True)),
        }]

        return {
            "ok": True,
            "term": self.state["term"],
            "vote_request": req,
            "votes": votes,
            "peers_hint": known_peers or [],
            "role": self.state["role"],
        }

    def handle_vote_request(self, req: Dict[str, Any], candidate_pub: bytes = None) -> Dict[str, Any]:
        """يعالج طلب تصويت من مرشح — يمنح صوتاً واحداً لكل term."""
        term = int(req.get("term") or 0)
        candidate = req.get("candidate_id")
        if term < int(self.state.get("term") or 0):
            return self._vote_response(term, candidate, False, reason="stale_term")

        # تحقق توقيع المرشح إن توفر المفتاح
        if candidate_pub and req.get("signature"):
            payload = json.dumps({
                "type": "vote_request",
                "term": term,
                "candidate_id": candidate,
            }, sort_keys=True)
            if not self.node.verify_signature(candidate_pub, payload, req["signature"]):
                return self._vote_response(term, candidate, False, reason="bad_signature")

        if term > int(self.state.get("term") or 0):
            self.state["term"] = term
            self.state["voted_for"] = None
            self.state["role"] = "follower"
            self.state["leader_id"] = None

        voted_for = self.state.get("voted_for")
        grant = voted_for is None or voted_for == candidate
        if grant:
            self.state["voted_for"] = candidate
            self.state["term"] = term
            self._save_state()
        return self._vote_response(term, candidate, grant, reason="ok" if grant else "already_voted")

    def _vote_response(self, term: int, candidate_id: str, grant: bool, reason: str = "") -> Dict[str, Any]:
        resp = {
            "type": "vote_response",
            "term": term,
            "voter_id": self.node.node_id,
            "candidate_id": candidate_id,
            "grant": bool(grant),
            "reason": reason,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        resp["signature"] = self.node.sign_message(json.dumps({
            "type": "vote_response",
            "term": term,
            "voter_id": self.node.node_id,
            "grant": bool(grant),
            "candidate_id": candidate_id,
        }, sort_keys=True))
        return resp

    def tally_votes(
        self,
        term: int,
        votes: List[Dict[str, Any]],
        cluster_size: int,
        voter_pubs: Dict[str, bytes] = None,
    ) -> Dict[str, Any]:
        """يحسب الأصوات الموقّعة؛ الفوز يحتاج majority = floor(n/2)+1."""
        majority = max(1, cluster_size // 2 + 1)
        valid = []
        for v in votes:
            if int(v.get("term") or -1) != term or not v.get("grant"):
                continue
            vid = v.get("voter_id")
            if not vid or not v.get("signature"):
                continue
            if voter_pubs and vid in voter_pubs:
                pub = voter_pubs[vid]
            elif vid == self.node.node_id:
                pub = self.node._pub_pem().encode()
            else:
                key_path = self.node.keys_dir / f"{vid}.pub"
                if not key_path.exists():
                    continue
                pub = key_path.read_bytes()
            payload = json.dumps({
                "type": "vote_response",
                "term": term,
                "voter_id": vid,
                "grant": True,
                "candidate_id": v.get("candidate_id") or self.node.node_id,
            }, sort_keys=True)
            # candidate_id في التوقيع يجب أن يطابق
            if not self.node.verify_signature(pub, payload, v["signature"]):
                # جرّب بدون اشتراط candidate في حال صوّت لغيره — نتخطى
                continue
            valid.append(vid)

        won = len(set(valid)) >= majority
        guard_result = None
        if won and self.guard is not None:
            guard_result = self.guard.validate_leader_claim(
                term=term,
                leader_id=self.node.node_id,
                vote_count=len(set(valid)),
            )
            if not guard_result.get("ok"):
                return {
                    "won": False,
                    "valid_votes": list(set(valid)),
                    "count": len(set(valid)),
                    "majority": majority,
                    "term": term,
                    "leader_id": None,
                    "guard": guard_result,
                }
            # استخدم majority الاتحاد إن كان أكبر
            majority = max(majority, self.guard.majority())
            won = len(set(valid)) >= majority
        if won and term == int(self.state.get("term") or 0):
            self.become_leader(term)
        return {
            "won": won,
            "valid_votes": list(set(valid)),
            "count": len(set(valid)),
            "majority": majority,
            "term": term,
            "leader_id": self.state.get("leader_id") if won else None,
            "guard": guard_result,
        }

    def become_leader(self, term: int = None) -> Dict[str, Any]:
        term = int(term if term is not None else self.state.get("term") or 1)
        self.state["term"] = term
        self.state["role"] = "leader"
        self.state["leader_id"] = self.node.node_id
        self.state["lease_until"] = self.now() + self.lease_seconds
        self.state["last_heartbeat"] = self.now()
        self._save_state()
        logger.info(f"👑 Became leader term={term} id={self.node.node_id}")
        return self.heartbeat()

    def heartbeat(self) -> Dict[str, Any]:
        """يجدد الـ lease إن كنا قائداً."""
        if self.state.get("role") != "leader" or self.state.get("leader_id") != self.node.node_id:
            return {"ok": False, "error": "not_leader"}
        self.state["last_heartbeat"] = self.now()
        self.state["lease_until"] = self.now() + self.lease_seconds
        self._save_state()
        msg = {
            "type": "leader_heartbeat",
            "term": self.state["term"],
            "leader_id": self.node.node_id,
            "lease_until": self.state["lease_until"],
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        msg["signature"] = self.node.sign_message(json.dumps({
            "type": "leader_heartbeat",
            "term": msg["term"],
            "leader_id": msg["leader_id"],
            "lease_until": msg["lease_until"],
        }, sort_keys=True))
        return msg

    def accept_heartbeat(self, msg: Dict[str, Any], leader_pub: bytes = None) -> Dict[str, Any]:
        term = int(msg.get("term") or 0)
        if term < int(self.state.get("term") or 0):
            return {"ok": False, "error": "stale_term"}
        if leader_pub and msg.get("signature"):
            payload = json.dumps({
                "type": "leader_heartbeat",
                "term": term,
                "leader_id": msg.get("leader_id"),
                "lease_until": msg.get("lease_until"),
            }, sort_keys=True)
            if not self.node.verify_signature(leader_pub, payload, msg["signature"]):
                return {"ok": False, "error": "bad_signature"}
        self.state["term"] = max(int(self.state.get("term") or 0), term)
        self.state["leader_id"] = msg.get("leader_id")
        self.state["lease_until"] = float(msg.get("lease_until") or (self.now() + self.lease_seconds))
        self.state["role"] = "follower" if msg.get("leader_id") != self.node.node_id else "leader"
        self.state["last_heartbeat"] = self.now()
        self._save_state()
        return {"ok": True, "leader_id": self.state["leader_id"], "term": self.state["term"]}

    # ------------------------------------------------------------------
    # تسليم القيادة
    # ------------------------------------------------------------------
    def handoff(self, successor_id: str) -> Dict[str, Any]:
        """تسليم طوعي للقيادة — ينهي الـ lease ويمنح صوته للخليفة في term جديد."""
        if self.state.get("role") != "leader" or self.state.get("leader_id") != self.node.node_id:
            return {"ok": False, "error": "not_leader"}
        old_term = int(self.state.get("term") or 0)
        self.state["role"] = "follower"
        self.state["leader_id"] = None
        self.state["lease_until"] = 0.0
        self.state["voted_for"] = successor_id
        self.state["term"] = old_term + 1
        self._save_state()
        msg = {
            "type": "leader_handoff",
            "from_id": self.node.node_id,
            "successor_id": successor_id,
            "old_term": old_term,
            "new_term": self.state["term"],
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        msg["signature"] = self.node.sign_message(json.dumps({
            "type": "leader_handoff",
            "from_id": msg["from_id"],
            "successor_id": successor_id,
            "old_term": old_term,
            "new_term": msg["new_term"],
        }, sort_keys=True))
        logger.info(f"🔁 Handoff from {self.node.node_id} → {successor_id} term {old_term}→{self.state['term']}")
        return {"ok": True, "handoff": msg}

    def accept_handoff(self, msg: Dict[str, Any], from_pub: bytes = None) -> Dict[str, Any]:
        if msg.get("successor_id") != self.node.node_id:
            # تابع: حدّث التوقعات فقط
            self.state["term"] = max(int(self.state.get("term") or 0), int(msg.get("new_term") or 0))
            self.state["leader_id"] = None
            self.state["lease_until"] = 0.0
            self.state["role"] = "follower"
            self._save_state()
            return {"ok": True, "role": "follower", "awaiting_election": True}
        if from_pub and msg.get("signature"):
            payload = json.dumps({
                "type": "leader_handoff",
                "from_id": msg.get("from_id"),
                "successor_id": msg.get("successor_id"),
                "old_term": msg.get("old_term"),
                "new_term": msg.get("new_term"),
            }, sort_keys=True)
            if not self.node.verify_signature(from_pub, payload, msg["signature"]):
                return {"ok": False, "error": "bad_signature"}
        self.state["term"] = int(msg.get("new_term") or (int(self.state.get("term") or 0) + 1))
        return self.become_leader(self.state["term"])

    def detect_leader_failure_and_elect(
        self,
        cluster_size: int,
        peer_vote_fn=None,
    ) -> Dict[str, Any]:
        """
        إن انتهت الـ lease يبدأ انتخاباً محلياً.
        peer_vote_fn: اختياري Callable(vote_request) -> vote_response لجمع أصوات حقيقية.
        """
        if self.is_leader_alive():
            return {"ok": True, "action": "none", "leader_id": self.state.get("leader_id")}
        election = self.start_election()
        votes = list(election["votes"])
        if peer_vote_fn:
            for _ in range(max(0, cluster_size - 1)):
                try:
                    vr = peer_vote_fn(election["vote_request"])
                    if vr:
                        votes.append(vr)
                except Exception:
                    pass
        tally = self.tally_votes(election["term"], votes, cluster_size=cluster_size)
        return {
            "ok": True,
            "action": "election",
            "election": election,
            "tally": tally,
            "leader_id": self.current_leader(),
        }

    # ------------------------------------------------------------------
    # استكمال الجولة بعد فشل القائد
    # ------------------------------------------------------------------
    def open_round(self, round_type: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """يفتح جولة عمل تحت القائد الحالي — تُحفظ في journal للاستكمال."""
        if self.state.get("role") != "leader" or not self.is_leader_alive():
            return {"ok": False, "error": "not_active_leader"}
        rid = f"rnd_{uuid.uuid4().hex[:12]}"
        journal = {
            "round_id": rid,
            "round_type": round_type,
            "plan": plan,
            "term": self.state["term"],
            "leader_id": self.node.node_id,
            "status": "open",
            "completed_shards": [],
            "pending_shards": list(plan.get("shards") or plan.get("workers") or []),
            "results": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.rounds[rid] = journal
        self._save_rounds()
        return {"ok": True, "round": journal}

    def report_shard_result(self, round_id: str, shard_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        rnd = self.rounds.get(round_id)
        if not rnd:
            return {"ok": False, "error": "unknown_round"}
        rnd.setdefault("results", {})[shard_id] = result
        if shard_id in rnd.get("pending_shards", []):
            rnd["pending_shards"] = [s for s in rnd["pending_shards"] if s != shard_id]
        if shard_id not in rnd.get("completed_shards", []):
            rnd.setdefault("completed_shards", []).append(shard_id)
        rnd["updated_at"] = datetime.now(timezone.utc).isoformat()
        if not rnd["pending_shards"]:
            rnd["status"] = "completed"
        self._save_rounds()
        return {"ok": True, "status": rnd["status"], "pending": len(rnd["pending_shards"])}

    def mark_leader_failed_on_round(self, round_id: str) -> Dict[str, Any]:
        rnd = self.rounds.get(round_id)
        if not rnd:
            return {"ok": False, "error": "unknown_round"}
        rnd["status"] = "leader_failed"
        rnd["failed_leader"] = rnd.get("leader_id")
        rnd["failed_at"] = datetime.now(timezone.utc).isoformat()
        self._save_rounds()
        # إسقاط الـ lease محلياً
        if self.state.get("leader_id") == rnd.get("leader_id"):
            self.state["lease_until"] = 0.0
            self.state["leader_id"] = None
            if self.state.get("role") == "leader":
                self.state["role"] = "follower"
            self._save_state()
        return {"ok": True, "round": rnd}

    def resume_round_as_leader(self, round_id: str) -> Dict[str, Any]:
        """
        قائد جديد يستكمل الجولة من journal: يعيد جدولة الـ pending فقط.
        """
        if self.state.get("role") != "leader" or not self.is_leader_alive():
            return {"ok": False, "error": "not_active_leader"}
        rnd = self.rounds.get(round_id)
        if not rnd:
            return {"ok": False, "error": "unknown_round"}
        rnd["status"] = "resumed"
        rnd["leader_id"] = self.node.node_id
        rnd["term"] = self.state["term"]
        rnd["resumed_at"] = datetime.now(timezone.utc).isoformat()
        rnd["updated_at"] = rnd["resumed_at"]
        pending = list(rnd.get("pending_shards") or [])
        self._save_rounds()
        return {
            "ok": True,
            "round_id": round_id,
            "resume_shards": pending,
            "completed": list(rnd.get("completed_shards") or []),
            "status": rnd["status"],
        }

    def continue_after_leader_failure(
        self,
        round_id: str,
        cluster_size: int,
        peer_vote_fn=None,
    ) -> Dict[str, Any]:
        """مسار كامل: فشل قائد → انتخاب → استكمال الجولة."""
        self.mark_leader_failed_on_round(round_id)
        elect = self.detect_leader_failure_and_elect(cluster_size, peer_vote_fn=peer_vote_fn)
        if not elect.get("tally", {}).get("won"):
            return {"ok": False, "stage": "election", "elect": elect}
        resumed = self.resume_round_as_leader(round_id)
        return {"ok": resumed.get("ok"), "stage": "resumed", "elect": elect, "resume": resumed}

    def status(self) -> Dict[str, Any]:
        return {
            "node_id": self.node.node_id,
            "term": self.state.get("term"),
            "role": self.state.get("role"),
            "leader_id": self.current_leader(),
            "lease_alive": self.is_leader_alive(),
            "lease_until": self.state.get("lease_until"),
            "open_rounds": sum(1 for r in self.rounds.values() if r.get("status") in ("open", "resumed", "leader_failed")),
            "completed_rounds": sum(1 for r in self.rounds.values() if r.get("status") == "completed"),
        }
