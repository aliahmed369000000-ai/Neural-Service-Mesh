from __future__ import annotations
import logging
import threading
from typing import Dict, List, Optional
from datetime import datetime, timezone

from core.node import BaseNode, NodeState
from storage.file_storage import FileStorage

logger = logging.getLogger(__name__)
REGISTRY_FILE = "nodes.json"


class NodeRegistry:
    def __init__(self, storage: FileStorage):
        self._storage = storage
        self._nodes: Dict[str, BaseNode] = {}
        self._meta_cache: Dict[str, dict] = {}
        # RLock: register/unregister/refresh_meta/_save تُستدعى من خيوط
        # متعددة (تنفيذ السرب المتوازي)؛ بدونه list(self._meta_cache.values())
        # داخل _save قد ترفع 'dictionary changed size during iteration'.
        # 🆕 القفل كان يحمي الكتابة فقط (register/unregister/refresh_meta/
        # _save) — القراءات (get/get_by_name/get_by_tag/get_by_state/
        # list_all/list_metadata/orphaned_metadata/count/exists/
        # get_interrupted) بقيت بلا قفل رغم أنها تُبنى بتكرار مباشر على
        # self._nodes/self._meta_cache. أي list_all()/list_metadata() تُنفَّذ
        # في خيط بينما register() يعمل في خيط آخر على نفس النسخة (كما يحدث
        # فعلياً: dev_console يعرض هذه القوائم بينما EvolutionEngine/AutoRuntime
        # قد يسجّل عقدة self_evolved في الخلفية) يرفع فعلياً RuntimeError:
        # dictionary changed size during iteration — تحقّقتُ من هذا بتكرار
        # فعلي قبل هذا التعديل (2000 تسجيل متزامن مع قارئ في خيط منفصل: أكثر
        # من 1400 استثناء). كل دالة قراءة الآن تحت نفس self._lock أيضاً.
        self._lock = threading.RLock()
        self._load()
        logger.info("NodeRegistry initialized")

    def register(self, node: BaseNode, overwrite: bool = False) -> str:
        with self._lock:
            if node.node_id in self._nodes and not overwrite:
                raise ValueError(f"Node '{node.node_id}' already registered")
            self._nodes[node.node_id] = node
            self._meta_cache[node.node_id] = node.to_dict()
            self._save()
        logger.info(f"Registered: {node.name} [{node.node_id[:8]}]")
        return node.node_id

    def unregister(self, node_id: str) -> bool:
        with self._lock:
            if node_id not in self._nodes:
                return False
            del self._nodes[node_id]
            self._meta_cache.pop(node_id, None)
            self._save()
            return True

    def get(self, node_id: str) -> Optional[BaseNode]:
        with self._lock:
            return self._nodes.get(node_id)

    def get_by_name(self, name: str) -> Optional[BaseNode]:
        with self._lock:
            return next((n for n in self._nodes.values() if n.name == name), None)

    def get_meta_by_name(self, name: str) -> Optional[dict]:
        """بحث في meta_cache المحفوظ (وليس العُقد الحيّة فقط) عن آخر سجل
        بنفس الاسم. يُستخدم عند إعادة بناء عقدة بعد إعادة تشغيل العملية
        (get_by_name يرجع None لأن _nodes يبدأ فارغاً كل تشغيل) حتى تقدر
        الطبقة الأعلى تسترجع نفس node_id والتاريخ بدل البدء من الصفر."""
        with self._lock:
            return next(
                (m for m in self._meta_cache.values() if m.get("name") == name),
                None,
            )

    def get_by_tag(self, tag: str) -> List[BaseNode]:
        with self._lock:
            return [n for n in self._nodes.values() if tag in n.tags]

    def get_by_state(self, state: str) -> List[BaseNode]:
        """إرجاع كل العُقد الحيّة التي حالتها الحالية تطابق state
        (مثل NodeState.ACTIVE / PAUSED / FAILED / CREATED)."""
        with self._lock:
            return [n for n in self._nodes.values() if n.state == state]

    def refresh_meta(self, node_id: str, persist: bool = True) -> Optional[dict]:
        """إعادة مزامنة meta_cache من حالة العقدة الحيّة (state، execution_count،
        last_executed...) بعد أي تنفيذ، لأن الكاش كان يُحفظ فقط لحظة register()
        ولا يتحدّث تلقائياً بعد ذلك. تُستدعى من ExecutionEngine بعد كل خطوة."""
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return None
            snapshot = node.to_dict()
            self._meta_cache[node_id] = snapshot
            if persist:
                self._save()
            return snapshot

    def get_interrupted(self) -> List[dict]:
        """سجلات meta محفوظة بحالة 'running' — أي عُقد كانت وسط process()
        لحظة توقف العملية فجأة (انهيار/kill) في جلسة سابقة، ولم تصل لا
        لنجاح ولا لفشل معروف. تُستخدم عند الإقلاع لمعرفة أي عمل معلَّق
        يستحق إعادة المحاولة (انظر ExecutionEngine.resume_interrupted)."""
        with self._lock:
            return [m for m in self._meta_cache.values() if m.get("state") == NodeState.RUNNING]

    def list_all(self) -> List[BaseNode]:
        with self._lock:
            return list(self._nodes.values())

    def list_metadata(self) -> List[dict]:
        with self._lock:
            return list(self._meta_cache.values())

    def orphaned_metadata(self) -> List[dict]:
        """سجلّات meta_cache المحفوظة (من تشغيلات سابقة) التي لا يقابلها
        كائن عقدة حيّ حالياً في self._nodes — أي عقدة كانت مسجَّلة فعلاً
        (self.register() نجح واستدعى self._save()) لكن لم يُعِد أي كود
        بناءها كـ Python object بعد إعادة تشغيل العملية الحالية.

        _load() يملأ meta_cache فقط (بيانات وصفية)، لا self._nodes (كائنات
        حيّة)؛ فقط الأدوار/أدوات MCP من الكتالوج الثابت تُعاد بناؤها فعلياً
        عند الإقلاع (MeshBundle._register_roles/_register_mcp_tools). أي
        عقدة أخرى — كعُقد self_evolved التي يُنشئها EvolutionEngine أثناء
        دورة تطوّر ذاتي — تبقى 'شبح': موجودة في التخزين، غائبة تماماً عن
        list_all()/get_by_state()/الرسم البياني/الحجر بالسمعة، حتى تُستعاد
        صراحة. راجع MeshBundle._restore_dynamic_nodes() للاستخدام."""
        with self._lock:
            return [m for nid, m in self._meta_cache.items() if nid not in self._nodes]

    def count(self) -> int:
        with self._lock:
            return len(self._nodes)

    def exists(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self._nodes

    def _save(self):
        with self._lock:
            self._storage.save(REGISTRY_FILE, {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "count": len(self._meta_cache),
                "nodes": list(self._meta_cache.values()),
            })

    def _load(self):
        data = self._storage.load(REGISTRY_FILE)
        if data:
            for nm in data.get("nodes", []):
                nid = nm.get("node_id")
                if nid:
                    self._meta_cache[nid] = nm
            logger.info(f"Registry loaded {len(self._meta_cache)} records")

    def __repr__(self):
        return f"<NodeRegistry count={self.count()}>"
