"""
Mesh Bundle — التوصيل الفعلي بين كل مكوّنات الـmesh
=====================================================
قبل هذا الملف، كانت core/registry.py و ai/memory_engine.py و
ai/agent_factory.py و ai/system_dna.py و ai/swarm_coordinator.py و
ai/reputation_engine.py و ai/scoring_engine.py موجودة ومكتوبة لكن غير
مستوردة من streamlit_app.py أبداً — كل واحد "جزيرة" منفصلة.

هذا الملف ينشئ "mesh bundle" حقيقياً واحداً:
  - NodeRegistry (core/registry.py) مبني على FileStorage مشتركة
  - MemoryEngine + ScoringEngine (SQLite واحد مشترك: data/mesh.db)
  - NodeReputationEngine مربوط بـ MemoryEngine
  - AgentFactory + SwarmCoordinator
  - SystemDNA يلتقط صوراً دورية من الحالة الفعلية (registry + scoring + memory)
  - SQLiteStorage (storage/db.py) كسجلّ تدقيق حقيقي (execution_logs +
    connections) — كان مكتوباً بالكامل ولم يُستخدم في أي مكان بالمشروع

الـ singleton محفوظ على مستوى العملية (process-level, عبر @lru_cache) وليس
فقط session_state — بذلك يبقى حياً ومشتركاً بين كل جلسات Streamlit التي
تخدمها نفس العملية، بدل أن يُعاد إنشاؤه فارغاً في كل rerun.

الاستخدام:
    from core.mesh_bundle import get_mesh_bundle
    bundle = get_mesh_bundle()
    result = bundle.coordinator.execute("هدف ما", data={...})
    bundle.record_swarm_result(result)
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from core.node import BaseNode, NodeSchema
from core.registry import NodeRegistry
from storage.file_storage import FileStorage
from storage.db import SQLiteStorage

from ai.memory_engine import MemoryEngine
from ai.scoring_engine import ScoringEngine
from ai.reputation_engine import NodeReputationEngine
from ai.system_dna import SystemDNA
from ai.agent_factory import AgentFactory, AGENT_CATALOGUE
from ai.swarm_coordinator import SwarmCoordinator

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def datetime_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentRoleNode(BaseNode):
    """
    عقدة (BaseNode) حقيقية تمثّل دوراً واحداً من AGENT_CATALOGUE داخل
    NodeRegistry. هذا هو الرابط الفعلي بين "الوكلاء" (ai/agent_factory)
    و"شبكة العُقد" (core/registry) — كانا نظامين منفصلين تماماً قبل ذلك.
    """

    def __init__(self, role: str, spec: dict):
        super().__init__(
            name=role,
            description=spec.get("description", ""),
            tags=list(spec.get("tags", [])) + ["agent_role"],
        )
        self._role = role
        self._capabilities = spec.get("capabilities", [])

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(
            fields={"task": "str"},
            required=["task"],
            description=f"مهمة نصية يُنفّذها دور {self._role}",
        )

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(
            fields={"result": "str"},
            required=[],
            description="نتيجة تنفيذ المهمة",
        )

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # التنفيذ الفعلي يمر عبر AgentFactory.run_task (محرك NSMAgent الحقيقي)،
        # وليس عبر هذه العقدة مباشرة. هذه العقدة تمثّل "الهوية" المسجّلة
        # للدور داخل الـregistry حتى يشارك في التسجيل/السمعة/الذاكرة/الـDNA.
        return {"result": ""}


class MCPToolNode(BaseNode):
    """
    عقدة (BaseNode) تمثّل أداة MCP حقيقية موجودة فعلاً في
    mcp_server/server.py (quran_lookup، classify_harm، ask_nsm، ...).
    process() تستدعي الدالة الحقيقية مباشرة (نفس الدالة التي يستخدمها أي
    عميل MCP خارجي) — وليس محاكاة أو بيانات وهمية.
    """

    def __init__(self, tool_name: str, fn, description: str):
        super().__init__(
            name=tool_name,
            description=description.strip().splitlines()[0] if description else "",
            tags=["mcp_tool"],
        )
        self._fn = fn

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={"kwargs": "dict"}, required=[],
                           description="وسائط الأداة (تُمرَّر كما هي للدالة الحقيقية)")

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={"result": "str"}, required=[],
                           description="نص JSON من نفس دالة أداة MCP")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = data.get("kwargs", {}) if isinstance(data, dict) else {}
        return {"result": self._fn(**kwargs)}


class MeshBundle:
    """الحزمة الحيّة الواحدة التي تربط كل مكوّنات الـmesh ببعضها فعلياً."""

    def __init__(self, storage_dir: Optional[str] = None, db_path: Optional[str] = None):
        storage_dir = storage_dir or str(DATA_DIR)
        db_path = db_path or str(DATA_DIR / "mesh.db")

        self.storage = FileStorage(storage_dir=storage_dir)
        self.registry = NodeRegistry(self.storage)

        self.memory_engine = MemoryEngine(db_path=db_path)
        self.scoring_engine = ScoringEngine(db_path=db_path)
        self.reputation_engine = NodeReputationEngine(memory_engine=self.memory_engine)
        self.dna = SystemDNA()

        # storage/db.py::SQLiteStorage كان مكتوباً بالكامل (جداول nodes/
        # connections/execution_logs) لكن لم يُبنَ (instantiate) في أي مكان
        # بالمشروع — يُستخدم هنا كسجلّ تدقيق (audit log) حقيقي لتنفيذات
        # السرب، بنفس ملف data/mesh.db المشترك (لا تضارب أسماء جداول مع
        # MemoryEngine/ScoringEngine — تحقّقت من ذلك).
        self.exec_log = SQLiteStorage(db_path=db_path)

        self.agent_factory = AgentFactory()
        self.coordinator = SwarmCoordinator(self.agent_factory, max_agents=20)

        self._lock = threading.Lock()
        self.role_node_ids: Dict[str, str] = {}
        self.mcp_tool_node_ids: Dict[str, str] = {}
        self._root_node_id = self._register_roles()
        self._register_mcp_tools()
        self._sync_nodes_to_exec_log()

        logger.info(
            "MeshBundle initialised: %d nodes registered, db=%s",
            self.registry.count(), db_path,
        )

    # ── أسماء بديلة (aliases) تطابق ما يتوقعه ai/validator.py::Phase6Validator
    # (self.mesh.registry / .memory / .agent_factory / .scoring / .system_dna /
    # .swarm / .reputation) — بدل تعديل الفاليديتور لأسماء صفاتنا الداخلية ──
    @property
    def memory(self):
        return self.memory_engine

    @property
    def scoring(self):
        return self.scoring_engine

    @property
    def system_dna(self):
        return self.dna

    @property
    def swarm(self):
        return self.coordinator

    @property
    def reputation(self):
        return self.reputation_engine

    # ── مزامنة عُقد الـregistry إلى SQLiteStorage (storage/db.py) ────────────
    def _sync_nodes_to_exec_log(self) -> None:
        for node in self.registry.list_all():
            self.exec_log.upsert_node(node.to_dict())

    # ── تسجيل كل الأدوار الموجودة في الكتالوج كعُقد حقيقية داخل الـregistry ──
    def _register_roles(self) -> str:
        root_id = None
        for role, spec in AGENT_CATALOGUE.items():
            existing = self.registry.get_by_name(role)
            if existing:
                self.role_node_ids[role] = existing.node_id
                continue
            node = AgentRoleNode(role, spec)
            node_id = self.registry.register(node)
            self.role_node_ids[role] = node_id
            if root_id is None:
                root_id = node_id
        # عقدة جذر رمزية يُبنى منها "المسار" (path) عند تغذية الـScoringEngine/
        # MemoryEngine — تمثّل SwarmCoordinator نفسه كنقطة انطلاق كل المهام.
        return root_id or "swarm_coordinator_root"

    # ── تسجيل أدوات MCP الحقيقية (mcp_server/server.py) كعُقد داخل الـregistry ──
    def _register_mcp_tools(self) -> None:
        """
        يستورد mcp_server/server.py (نفس السيرفر الذي يستخدمه أي عميل MCP
        خارجي فعلياً) ويسجّل كل أداة موجودة فيه كعقدة MCPToolNode حقيقية —
        بذلك يظهر عدد نودات الـregistry فعلاً في check_project_health بدل
        أن يبقى 0 دائماً. الاستيراد داخل الدالة (lazy) لتفادي أي دورة
        استيراد مع mcp_server/server.py الذي يستورد get_mesh_bundle بدوره.
        """
        try:
            import mcp_server.server as _mcp_srv
        except Exception as e:
            logger.warning("MeshBundle: تعذّر تحميل mcp_server.server: %s", e)
            return

        tool_names = [
            "quran_lookup", "quran_search", "classify_harm", "ask_nsm",
            "search_ckg", "sensor_hub_status", "check_project_health",
        ]
        for tool_name in tool_names:
            fn = getattr(_mcp_srv, tool_name, None)
            if fn is None or not callable(fn):
                continue
            existing = self.registry.get_by_name(tool_name)
            if existing:
                self.mcp_tool_node_ids[tool_name] = existing.node_id
                continue
            node = MCPToolNode(tool_name, fn, fn.__doc__ or "")
            node_id = self.registry.register(node)
            self.mcp_tool_node_ids[tool_name] = node_id

    # ── تغذية نتيجة تنفيذ سرب حقيقية إلى Scoring + Memory + Reputation ──────
    def record_swarm_result(self, swarm_result) -> None:
        """
        يأخذ SwarmResult حقيقياً (من SwarmCoordinator.execute) ويغذّي كل
        مهمة فرعية فيه إلى ScoringEngine و MemoryEngine و
        NodeReputationEngine — وهذا هو الرابط الذي كان مفقوداً: نتائج
        السرب كانت تُعرض في الواجهة فقط ولا تصل أبداً لمحركات التقييم/
        الذاكرة/السمعة.
        """
        with self._lock:
            for task in getattr(swarm_result, "tasks", []):
                agent_id = getattr(task, "assigned_agent_id", None)
                agent = self.agent_factory._agents.get(agent_id) if agent_id else None
                # نحصل على اسم الدور الحقيقي من الوكيل المُسنَد فعلياً (agent.role)
                # لا من required_capability (وهو اسم قدرة مثل "search"، وليس اسم
                # دور مثل "ResearchAgent" — الاثنان مختلفان في هذا الكتالوج).
                role = agent.role if agent else None
                node_id = self.role_node_ids.get(role)
                if not node_id:
                    continue
                success = task.status == "done"
                latency = float(task.duration_ms or 0.0)

                self.reputation_engine.record_execution(
                    node_id, role, success, latency
                )

                started = getattr(task, "started_at", None) or datetime_now_iso()
                finished = getattr(task, "finished_at", None) or started
                self.exec_log.upsert_connection(
                    self._root_node_id, node_id, weight=1.0, label=role or ""
                )
                self.exec_log.save_run({
                    "run_id": getattr(task, "task_id", "") or f"task_{id(task)}",
                    "status": "success" if success else "failed",
                    "path": [self._root_node_id, node_id],
                    "started_at": started,
                    "finished_at": finished,
                    "total_duration_ms": latency,
                    "final_output": (task.result or {}).get("result_text") if task.result else None,
                    "steps": [{
                        "node_id": node_id, "node_name": role,
                        "duration_ms": latency,
                        "status": "success" if success else "failed",
                    }],
                })

                run_result = {
                    "run_id": getattr(task, "task_id", ""),
                    "status": "success" if success else "failed",
                    "total_duration_ms": latency,
                    "path": [self._root_node_id, node_id],
                    "steps": [{
                        "node_id": node_id,
                        "node_name": role,
                        "duration_ms": latency,
                        "status": "success" if success else "failed",
                    }],
                }
                self.scoring_engine.record_run(run_result)
                self.memory_engine.learn_from_run(run_result)

            try:
                self.dna.snapshot(
                    registry=self.registry,
                    scoring_engine=self.scoring_engine,
                    memory_engine=self.memory_engine,
                    notes=f"swarm:{getattr(swarm_result, 'goal', '')[:60]}",
                )
            except Exception as e:
                logger.warning("MeshBundle: DNA snapshot failed: %s", e)

    def summary(self) -> dict:
        return {
            "nodes": self.registry.count(),
            "scoring": self.scoring_engine.summary(),
            "memory": self.memory_engine.summary(),
            "reputation": self.reputation_engine.summary(),
            "dna_versions": len(self.dna.history(limit=1000)),
            "exec_log": self.exec_log.db_stats(),
        }


@lru_cache(maxsize=1)
def get_mesh_bundle() -> MeshBundle:
    """Singleton حقيقي على مستوى العملية — يبقى حياً بين كل sessions/reruns."""
    return MeshBundle()
