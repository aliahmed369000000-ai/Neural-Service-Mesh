"""
NodeChannel — قناة تواصل حقيقية بين عُقد الـmesh
=====================================================
قبل هذا الملف، كان "التواصل" بين المكوّنات يقتصر على:
  - ai/agent_event_bus.py: سجلّ أحداث مؤقت في session_state (لعرض الواجهة
    فقط، يُمسَح عند انتهاء الجلسة، ولا تقرأه أي عقدة أخرى).
  - SwarmCoordinator يستدعي AgentFactory.run_task مباشرة (استدعاء دوال،
    وليس تراسلاً بين هويات مسجّلة في الـregistry).

لا توجد آلية تسمح لعقدة مسجّلة في core/registry.py أن "ترسل" شيئاً فعلياً
لعقدة أخرى بحيث تبقى الرسالة موجودة، قابلة للقراءة لاحقاً، ومرتبطة
بهويتَي المرسل والمستقبل الحقيقيتين (node_id).

NodeChannel يسدّ هذه الفجوة: صندوق بريد دائم لكل node_id، محفوظ عبر نفس
FileStorage الذي تستخدمه NodeRegistry، بحيث:
  - أي عقدة (دور وكيل، أداة MCP، عقدة مُولَّدة ذاتياً عبر EvolutionEngine)
    يمكنها إرسال رسالة موجّهة (send) أو بث جماعي لجيرانها في ServiceGraph
    (broadcast).
  - الرسائل تُقرأ من inbox حقيقي لكل node_id، وتبقى محفوظة بين جلسات
    Streamlit المختلفة (نفس آلية REGISTRY_FILE في core/registry.py).
  - يُستخدم فعلياً من core/mesh_bundle.py في نقطتين حقيقيتين:
      1) عند انضمام عقدة جديدة عبر دورة تطوّر ذاتي (EvolutionEngine)،
         تُرسِل إعلان قدرات لجيرانها في الرسم البياني.
      2) عند تنفيذ سرب حقيقي (record_swarm_result)، تُرسِل عقدة التنسيق
         الجذرية نتيجة كل مهمة فرعية للعقدة التي نفّذتها فعلاً.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from storage.file_storage import FileStorage

logger = logging.getLogger(__name__)

CHANNEL_FILE = "node_channel.json"
MAX_INBOX_PER_NODE = 100
MAX_LOG = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NodeChannel:
    """صندوق بريد دائم ومشترك بين كل عُقد الـmesh المسجّلة."""

    def __init__(self, storage: FileStorage):
        self._storage = storage
        self._inboxes: Dict[str, List[dict]] = {}
        self._log: List[dict] = []
        self._load()
        logger.info("NodeChannel initialised")

    # ── إرسال ──────────────────────────────────────────────────────────────

    def send(
        self,
        from_id: str,
        to_id: str,
        topic: str,
        payload: Optional[dict] = None,
    ) -> dict:
        """يرسل رسالة موجّهة من عقدة إلى عقدة أخرى محدّدة بهويّتها الحقيقية."""
        message = {
            "message_id": str(uuid.uuid4()),
            "from_id": from_id,
            "to_id": to_id,
            "topic": topic,
            "payload": payload or {},
            "sent_at": _now(),
            "read": False,
        }
        inbox = self._inboxes.setdefault(to_id, [])
        inbox.append(message)
        if len(inbox) > MAX_INBOX_PER_NODE:
            self._inboxes[to_id] = inbox[-MAX_INBOX_PER_NODE:]
        self._log.append(message)
        if len(self._log) > MAX_LOG:
            self._log = self._log[-MAX_LOG:]
        self._save()
        return message

    def broadcast(
        self,
        from_id: str,
        to_ids: List[str],
        topic: str,
        payload: Optional[dict] = None,
    ) -> List[dict]:
        """يرسل نفس الرسالة لعدّة عُقد دفعة واحدة (مثلاً: كل جيران عقدة في الرسم البياني)."""
        sent = []
        for to_id in to_ids:
            if to_id and to_id != from_id:
                sent.append(self.send(from_id, to_id, topic, payload))
        return sent

    # ── قراءة ──────────────────────────────────────────────────────────────

    def inbox(self, node_id: str, unread_only: bool = False, limit: int = 50) -> List[dict]:
        messages = self._inboxes.get(node_id, [])
        if unread_only:
            messages = [m for m in messages if not m["read"]]
        return messages[-limit:]

    def mark_read(self, node_id: str, message_id: str) -> bool:
        for m in self._inboxes.get(node_id, []):
            if m["message_id"] == message_id:
                m["read"] = True
                self._save()
                return True
        return False

    def unread_count(self, node_id: str) -> int:
        return sum(1 for m in self._inboxes.get(node_id, []) if not m["read"])

    def recent(self, limit: int = 50) -> List[dict]:
        """آخر الرسائل عبر كل القناة (للمراقبة/لوحة التحكم)."""
        return self._log[-limit:]

    def stats(self) -> dict:
        return {
            "total_messages": len(self._log),
            "nodes_with_inbox": len(self._inboxes),
            "unread_total": sum(
                1 for msgs in self._inboxes.values() for m in msgs if not m["read"]
            ),
        }

    # ── تخزين ──────────────────────────────────────────────────────────────

    def _save(self):
        self._storage.save(CHANNEL_FILE, {
            "saved_at": _now(),
            "inboxes": self._inboxes,
            "log": self._log,
        })

    def _load(self):
        data = self._storage.load(CHANNEL_FILE)
        if data:
            self._inboxes = data.get("inboxes", {}) or {}
            self._log = data.get("log", []) or []
            logger.info(f"NodeChannel loaded {len(self._log)} message(s)")

    def __repr__(self):
        s = self.stats()
        return f"<NodeChannel messages={s['total_messages']} nodes={s['nodes_with_inbox']}>"
