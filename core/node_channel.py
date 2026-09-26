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
import threading
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
    """صندوق بريد دائم ومشترك بين كل عُقد الـmesh المسجّلة.

    تزامن الخيوط: SwarmCoordinator.execute ينفّذ المهام الفرعية فعلياً
    عبر ThreadPoolExecutor (ai/swarm_coordinator.py)، وMeshBundle يحمي
    استدعاءاته لهذه القناة بقفل RLock خاص به — لكن NodeChannel نفسها
    كانت بلا أي حماية تزامن داخلية، رغم أنها مصمَّمة كقناة عامة يمكن أن
    "ترسل" منها أي عقدة مسجّلة (وليس فقط عبر MeshBundle). بلا قفل هنا،
    استدعاءان متزامنان من خيطين مختلفين لنفس صندوق بريد (send/broadcast)
    قد يتسابقان على self._inboxes/self._log (قراءة-تعديل-كتابة غير ذرّية)
    فتُفقَد إحدى الرسالتين. self._lock يجعل القناة آمنة بذاتها بغض النظر
    عن انضباط الطرف المستدعي، بنفس نمط القفل المستخدم فعلياً في بقية
    مكوّنات الحالة المشتركة بالمشروع (SwarmCoordinator، TaskManager،
    CollectiveMemory، AgentCollaboration، ...)."""

    def __init__(self, storage: FileStorage):
        self._storage = storage
        self._inboxes: Dict[str, List[dict]] = {}
        self._log: List[dict] = []
        self._lock = threading.RLock()
        self._load()
        logger.info("NodeChannel initialised")

    # ── إرسال ──────────────────────────────────────────────────────────────

    def send(
        self,
        from_id: str,
        to_id: str,
        topic: str,
        payload: Optional[dict] = None,
        reply_to: Optional[str] = None,
    ) -> dict:
        """يرسل رسالة موجّهة من عقدة إلى عقدة أخرى محدّدة بهويّتها الحقيقية.

        reply_to اختياري: message_id لرسالة سابقة يُربَط بها هذا الرد —
        يسمح لطبقة تعتمد على نمط طلب/استجابة حقيقي (وليس بثاً بلا سياق
        فقط) أن تُطابق الرد بالطلب الأصلي عبر القناة نفسها.

        يرفض إرسال رسالة بلا from_id/to_id حقيقيين (بدل حفظها بصمت في
        صندوق بريد بمفتاح فارغ لا تقرأه أي عقدة أبداً) — خطأ إعداد يجب
        أن يظهر فوراً للمُرسِل بدل أن يختفي في القناة."""
        if not from_id or not to_id:
            raise ValueError(
                f"NodeChannel.send: from_id و to_id يجب أن يكونا هويتي عقدة "
                f"حقيقيتين (from_id={from_id!r}, to_id={to_id!r})"
            )
        message = {
            "message_id": str(uuid.uuid4()),
            "from_id": from_id,
            "to_id": to_id,
            "topic": topic,
            "payload": payload or {},
            "reply_to": reply_to,
            "sent_at": _now(),
            "read": False,
        }
        with self._lock:
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
        reply_to: Optional[str] = None,
    ) -> List[dict]:
        """يرسل نفس الرسالة لعدّة عُقد دفعة واحدة (مثلاً: كل جيران عقدة في الرسم البياني).

        كل عملية إرسال ضمن هذا البث تجري تحت self._lock (عبر send())،
        لكن البث كاملاً هنا ليس ذرّياً عبر كل المستقبِلين معاً — وهذا
        مقصود: مستقبِل واحد بهوية غير صالحة لا يجب أن يُسقِط بقية البث
        الفعلي الصالح لبقية الجيران (لذلك نتخطّاه بصمت بدل رفع
        ValueError من send)."""
        sent = []
        for to_id in to_ids:
            if to_id and to_id != from_id:
                sent.append(self.send(from_id, to_id, topic, payload, reply_to=reply_to))
        return sent

    # ── قراءة ──────────────────────────────────────────────────────────────

    def inbox(self, node_id: str, unread_only: bool = False, limit: int = 50) -> List[dict]:
        with self._lock:
            messages = list(self._inboxes.get(node_id, []))
        if unread_only:
            messages = [m for m in messages if not m["read"]]
        return messages[-limit:]

    def mark_read(self, node_id: str, message_id: str) -> bool:
        with self._lock:
            for m in self._inboxes.get(node_id, []):
                if m["message_id"] == message_id:
                    m["read"] = True
                    self._save()
                    return True
        return False

    def unread_count(self, node_id: str) -> int:
        with self._lock:
            return sum(1 for m in self._inboxes.get(node_id, []) if not m["read"])

    def recent(self, limit: int = 50) -> List[dict]:
        """آخر الرسائل عبر كل القناة (للمراقبة/لوحة التحكم)."""
        with self._lock:
            return list(self._log[-limit:])

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_messages": len(self._log),
                "nodes_with_inbox": len(self._inboxes),
                "unread_total": sum(
                    1 for msgs in self._inboxes.values() for m in msgs if not m["read"]
                ),
            }

    # ── تخزين ──────────────────────────────────────────────────────────────
    # ملاحظة: _save()/_load() تُستدعيان دائماً من داخل self._lock بالفعل
    # (من send/mark_read أعلاه أو من __init__)، فلا تُقفِلان هنا بأنفسهما
    # لتفادي إعادة دخول لا حاجة له.

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
