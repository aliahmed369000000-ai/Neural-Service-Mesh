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

from core.node import BaseNode, NodeSchema, NodeState
from core.registry import NodeRegistry
from core.graph import ServiceGraph
from core.node_channel import NodeChannel
from storage.file_storage import FileStorage
from storage.db import SQLiteStorage

from ai.memory_engine import MemoryEngine
from ai.scoring_engine import ScoringEngine
from ai.reputation_engine import NodeReputationEngine
from ai.system_dna import SystemDNA
from ai.agent_factory import AgentFactory, AGENT_CATALOGUE
from ai.swarm_coordinator import SwarmCoordinator
from ai.gap_detector import GapDetectionEngine
from ai.service_generator import ServiceGeneratorEngine
from ai.governor import AIGovernanceLayer
from ai.capability_marketplace import CapabilityMarketplace
from ai.evolution_engine import EvolutionEngine
from ai.decision import AIDecisionLayer

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# كل كم نتيجة سرب حقيقية (record_swarm_result) تُشغَّل دورة تطوّر ذاتي كاملة
# تلقائياً (انظر التعليق داخل record_swarm_result أدناه).
EVOLUTION_CYCLE_INTERVAL = 5


def datetime_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentRoleNode(BaseNode):
    """
    عقدة (BaseNode) حقيقية تمثّل دوراً واحداً من AGENT_CATALOGUE داخل
    NodeRegistry. هذا هو الرابط الفعلي بين "الوكلاء" (ai/agent_factory)
    و"شبكة العُقد" (core/registry) — كانا نظامين منفصلين تماماً قبل ذلك.
    """

    def __init__(self, role: str, spec: dict, node_id: Optional[str] = None):
        super().__init__(
            name=role,
            description=spec.get("description", ""),
            tags=list(spec.get("tags", [])) + ["agent_role"],
            node_id=node_id,
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

    def __init__(self, tool_name: str, fn, description: str, node_id: Optional[str] = None):
        super().__init__(
            name=tool_name,
            description=description.strip().splitlines()[0] if description else "",
            tags=["mcp_tool"],
            node_id=node_id,
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

        # ── التواصل الحقيقي بين العُقد + رسم بياني حيّ للطوبولوجيا ──────────
        # (core/node_channel.py) قناة رسائل دائمة بين node_id حقيقية، و
        # (core/graph.py) رسم بياني يُستخدم فعلياً من GapDetectionEngine
        # لاكتشاف الفجوات ومن AIGovernanceLayer لفحص المسارات — كلاهما كان
        # موجوداً ومكتوباً بالكامل لكن بلا رسم بياني حيّ يُغذّيه.
        self.channel = NodeChannel(self.storage)
        self.graph = ServiceGraph()

        # ── طبقة القرار الذكي (ai/decision.py::AIDecisionLayer) ─────────────
        # كانت مكتوبة بالكامل (اختيار مسار بالتقييم، ترتيب مسارات بديلة،
        # اقتراح بديل عند فشل/حجر عقدة، تعلّم بسيط من تاريخ التنفيذ) لكن لا
        # يوجد أي مكان في المشروع يبنيها فعلياً (تحقّقت بالبحث عن
        # "AIDecisionLayer(" في كل الملفات) — api_server.py كان يبني
        # core.engine.ExecutionEngine بلا `ai=` إطلاقاً، فيبقى self._ai=None
        # هناك دائماً. النتيجة العملية: كل منطق fallback في
        # ExecutionEngine.run_path (لعقدة غير موجودة، ولعقدة محجورة بعد آخر
        # تعديل) لم يكن يعمل أبداً في أي طلب حقيقي عبر /process، وrun_between
        # كان يستخدم BFS البسيط دائماً بدل اختيار مسار مُقيَّم. نسخة واحدة
        # هنا على مستوى MeshBundle (singleton للعملية) تُشارَك بين كل طلبات
        # /process المتتالية حتى تتراكم إحصاءات learn_from_run فعلياً بدل
        # إعادة بناء طبقة فارغة الذاكرة في كل طلب.
        self.ai_decision = AIDecisionLayer(graph=self.graph, db=self.exec_log)

        # RLock وليس Lock عادياً: record_swarm_result يستدعي الآن
        # _apply_reputation_feedback/_apply_reputation_recovery مباشرة (بعد أن
        # كانتا تُستدعيان فقط من run_evolution_cycle — انظر التعليق هناك)، وكل
        # عدة نتائج سرب حقيقية تستدعي run_evolution_cycle نفسها أيضاً، وهي
        # تُعيد طلب self._lock من جديد. Lock عادي كان سيتعطّل (deadlock) على
        # أول استدعاء متداخل من نفس الخيط؛ RLock يسمح بإعادة الدخول من نفس
        # الخيط بأمان دون تغيير أي سلوك تزامن فعلي بين خيوط مختلفة.
        self._lock = threading.RLock()
        self._swarm_results_since_evolution = 0
        self.role_node_ids: Dict[str, str] = {}
        self.mcp_tool_node_ids: Dict[str, str] = {}
        self._root_node_id = self._register_roles()
        self._register_mcp_tools()
        self._sync_nodes_to_exec_log()
        self._sync_nodes_to_graph()

        # ── التطوّر الذاتي الحقيقي (Phase 5/7): GapDetector → ServiceGenerator
        # → AIGovernanceLayer → تسجيل عقدة جديدة فعلياً في الـregistry نفسه ──
        # كانت هذه المحركات الأربعة مكتوبة بالكامل (ai/gap_detector.py،
        # ai/service_generator.py، ai/governor.py، ai/capability_marketplace.py،
        # ai/evolution_engine.py) لكن EvolutionEngine._mesh كان None دائماً —
        # لا تُستدعى أبداً من streamlit_app.py، ولا رسم بياني حقيقياً تعمل عليه.
        self.governance = AIGovernanceLayer(
            graph=self.graph, reputation_engine=self.reputation_engine,
        )
        self.gap_detector = GapDetectionEngine(
            graph=self.graph, memory_engine=self.memory_engine,
            scoring_engine=self.scoring_engine,
        )
        self.service_generator = ServiceGeneratorEngine(governance=self.governance)
        self.marketplace = CapabilityMarketplace()
        self.evolution = EvolutionEngine(
            mesh=self,
            gap_detector=self.gap_detector,
            service_generator=self.service_generator,
            governance=self.governance,
            capability_marketplace=self.marketplace,
        )

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

    # ── مزامنة عُقد الـregistry إلى ServiceGraph (core/graph.py) ────────────
    # يبني طوبولوجيا أولية حقيقية (كل دور/أداة متصل بعقدة SwarmCoordinator
    # الجذرية) حتى يكون لدى GapDetectionEngine رسم بياني فعلي يفحصه بدل
    # رسم فارغ — بدون هذا، pass الفجوات الهيكلية (routing gaps) لا يجد شيئاً
    # لأن ServiceGraph فارغ دائماً.
    def _sync_nodes_to_graph(self) -> None:
        for node in self.registry.list_all():
            self.graph.add_node(node.node_id, node.to_dict())
        if self._root_node_id and self.graph.has_node(self._root_node_id):
            for node_id in list(self.role_node_ids.values()) + list(self.mcp_tool_node_ids.values()):
                if node_id != self._root_node_id and self.graph.has_node(node_id):
                    self.graph.add_edge(self._root_node_id, node_id, label="mesh_member")

    # ── واجهة التسجيل التي يتوقّعها EvolutionEngine (mesh.register_node) ────
    # EvolutionEngine._mesh.register_node(node, connect_to=...) هي نقطة
    # الوصل الوحيدة التي كان ينقصها — بدونها EvolutionEngine._mesh=None دائماً
    # ولا تُسجَّل أي عقدة مُولَّدة ذاتياً في الـregistry الحقيقي.
    def register_node(self, node: BaseNode, connect_to: Optional[str] = None) -> str:
        node_id = self.registry.register(node)
        self.graph.add_node(node_id, node.to_dict())
        self.exec_log.upsert_node(node.to_dict())
        source = connect_to if (connect_to and self.graph.has_node(connect_to)) else self._root_node_id
        if source and self.graph.has_node(source) and source != node_id:
            self.graph.add_edge(source, node_id, label="self_evolved")
            self.exec_log.upsert_connection(source, node_id, weight=1.0, label="self_evolved")
        # تواصل فعلي: العقدة الجديدة تُعلن نفسها لجيرانها في الرسم البياني —
        # هذا هو "التواصل بين العُقد" الحقيقي، وليس سجلّ عرض واجهة فقط.
        neighbors = [source] + self.graph.get_neighbors(source) if source else []
        self.channel.broadcast(
            from_id=node_id,
            to_ids=[n for n in neighbors if n and n != node_id],
            topic="node_joined",
            payload={
                "name": node.name,
                "description": node.description,
                "tags": node.tags,
            },
        )
        return node_id

    # ── تسجيل كل الأدوار الموجودة في الكتالوج كعُقد حقيقية داخل الـregistry ──
    def _register_roles(self) -> str:
        root_id = None
        for role, spec in AGENT_CATALOGUE.items():
            existing = self.registry.get_by_name(role)
            if existing:
                self.role_node_ids[role] = existing.node_id
                continue
            persisted = self.registry.get_meta_by_name(role)
            node = AgentRoleNode(
                role, spec,
                node_id=persisted.get("node_id") if persisted else None,
            )
            if persisted:
                node.restore_state(persisted)
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
            persisted = self.registry.get_meta_by_name(tool_name)
            node = MCPToolNode(
                tool_name, fn, fn.__doc__ or "",
                node_id=persisted.get("node_id") if persisted else None,
            )
            if persisted:
                node.restore_state(persisted)
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

                # ── تحديث دورة حياة العقدة نفسها من نتيجة السرب الحقيقية ──────
                # كانت execution_count/state تبقى مجمّدة على 'created' للأبد
                # لأن تنفيذ السرب يمر عبر AgentFactory.run_task مباشرة وليس
                # عبر node.execute() (process() في AgentRoleNode مجرد هوية
                # فارغة). record_execution() تحدّث الحالة من النتيجة الحقيقية
                # المعروفة سلفاً (success/task.error) بدل استدعاء process().
                node = self.registry.get(node_id)
                prev_state = node.state if node else None
                if node:
                    node.record_execution(success, error=None if success else task.error)
                    self.registry.refresh_meta(node_id)

                started = getattr(task, "started_at", None) or datetime_now_iso()
                finished = getattr(task, "finished_at", None) or started
                self.exec_log.upsert_connection(
                    self._root_node_id, node_id, weight=1.0, label=role or ""
                )
                if self.graph.has_node(self._root_node_id) and self.graph.has_node(node_id):
                    self.graph.add_edge(self._root_node_id, node_id, label=role or "")

                # ── تواصل: إشعار جيران العقدة في الرسم البياني عند انتقال
                # حقيقي للحالة (فشل جديد، أو تعافٍ بعد فشل) — وليس عند كل
                # نجاح روتيني حتى لا تُغرَق القناة برسائل بلا قيمة تشغيلية.
                if node and node.state != prev_state:
                    state_topic = None
                    if node.state == NodeState.FAILED:
                        state_topic = "node_failed"
                    elif prev_state == NodeState.FAILED and node.state == NodeState.ACTIVE:
                        state_topic = "node_recovered"
                    if state_topic and self.graph.has_node(node_id):
                        neighbors = self.graph.get_neighbors(node_id)
                        self.channel.broadcast(
                            from_id=node_id,
                            to_ids=[n for n in neighbors if n and n != node_id],
                            topic=state_topic,
                            payload={
                                "role": role,
                                "previous_state": prev_state,
                                "current_state": node.state,
                                "error": task.error if not success else None,
                            },
                        )
                # تواصل حقيقي مرتبط بالتنفيذ الفعلي: عقدة التنسيق الجذرية
                # تُرسل للعقدة التي نفّذت المهمة فعلاً نتيجة تنفيذها — رسالة
                # حقيقية محفوظة في صندوق بريد node_id، وليست سطراً في سجلّ عرض.
                self.channel.send(
                    from_id=self._root_node_id,
                    to_id=node_id,
                    topic="task_result",
                    payload={
                        "success": success,
                        "latency_ms": latency,
                        "role": role,
                    },
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
                # ── الذاكرة الجماعية: درس دائم من كل مهمة فرعية ──────────
                try:
                    from ai.collective_memory import get_collective_memory
                    get_collective_memory().record_task_result(
                        task=(task.result or {}).get("task") if task.result else "",
                        success=success,
                        duration_ms=latency,
                        agent_id=agent_id or "",
                        run_id=getattr(task, "task_id", "") or "",
                        output_hint=(task.result or {}).get("result_text")
                        if task.result else "",
                    )
                except Exception as _cm_err:
                    logger.warning("MeshBundle: collective memory failed: %s", _cm_err)

            try:
                self.dna.snapshot(
                    registry=self.registry,
                    scoring_engine=self.scoring_engine,
                    memory_engine=self.memory_engine,
                    notes=f"swarm:{getattr(swarm_result, 'goal', '')[:60]}",
                )
            except Exception as e:
                logger.warning("MeshBundle: DNA snapshot failed: %s", e)

            # ── ربط فعلي: هذه النتيجة الحقيقية تُحدِّث سمعة/حالة العُقد الآن،
            # لا فقط عند اكتمال دورة تطوّر ذاتي. قبل هذا التعديل كانت
            # _apply_reputation_feedback/_apply_reputation_recovery (حجر
            # العقدة ضعيفة السمعة ورفع الحجر عن المتعافية — آخر كوميتين)
            # قابلتين للاستدعاء فقط من run_evolution_cycle()، ولم يكن أي
            # مكان في المشروع كله يستدعي run_evolution_cycle() فعلياً (تحقّقت
            # بالبحث عبر الكود) — أي أن الحجر/رفع الحجر لم يكونا يعملان في
            # التشغيل الفعلي إطلاقاً رغم أنهما مكتوبان ومختبَران. الآن كل
            # نتيجة سرب حقيقية (وليس فقط دورة تطوّر يدوية) تُشغّلهما مباشرة.
            try:
                self._apply_reputation_feedback()
                self._apply_reputation_recovery()
            except Exception as e:
                logger.warning("MeshBundle: reputation feedback/recovery after swarm result failed: %s", e)

            # ── تشغيل دوري فعلي لدورة التطوّر الذاتي الكاملة (اكتشاف فجوات +
            # توليد/اعتماد عُقد جديدة) كل EVOLUTION_CYCLE_INTERVAL نتيجة سرب
            # حقيقية — كانت run_evolution_cycle() نفسها معرَّفة بالكامل
            # ومختبَرة لكن بلا أي نقطة تشغيل تلقائية في كامل المشروع (لا UI،
            # لا مجدوَل خلفي)، فتبقى "cycles_run" في summary() صفراً للأبد.
            # لا تكلفة LLM هنا (GapDetector/ServiceGenerator/Governance كلها
            # فحص رسم بياني وقواعد محلية) فالتشغيل الدوري آمن التكلفة.
            self._swarm_results_since_evolution += 1
            if self._swarm_results_since_evolution >= EVOLUTION_CYCLE_INTERVAL:
                self._swarm_results_since_evolution = 0
                try:
                    self.run_evolution_cycle()
                except Exception as e:
                    logger.warning("MeshBundle: periodic run_evolution_cycle after swarm result failed: %s", e)

    # ── دورة تطوّر ذاتي حقيقية (تُستدعى يدوياً، أو تلقائياً كل
    # EVOLUTION_CYCLE_INTERVAL نتيجة سرب من record_swarm_result أعلاه) ──────
    # تشغّل EvolutionEngine.run_cycle() الحقيقي: يفحص الرسم البياني الفعلي
    # (self.graph) بحثاً عن فجوات، يولّد عقداً جديدة، يمرّرها على
    # AIGovernanceLayer (حدود صارمة: لا حلقات، حد أقصى للتوليد، سمعة دنيا)،
    # ثم يسجّل المعتمَد منها فعلياً في self.registry عبر register_node أعلاه.
    def run_evolution_cycle(self) -> dict:
        with self._lock:
            cycle = self.evolution.run_cycle(auto_register=True, verbose=False)
            self._apply_reputation_feedback()
            self._apply_reputation_recovery()
            try:
                self.dna.snapshot(
                    registry=self.registry,
                    scoring_engine=self.scoring_engine,
                    memory_engine=self.memory_engine,
                    notes=f"evolution_cycle:{cycle.cycle_number}",
                )
            except Exception as e:
                logger.warning("MeshBundle: DNA snapshot after evolution failed: %s", e)
            return cycle.to_dict() if hasattr(cycle, "to_dict") else cycle.summary

    # ── تغذية السمعة إلى قرارات حيّة (حجر/رفع حجر) + إبلاغ الجيران ──────────
    # جزء من "التطوّر الذاتي": عقدة سمعتها منخفضة باستمرار تُحجَر (quarantine)
    # فعلياً في NodeReputationEngine (فتُستبعد من التوجيه)، وتُبلَّغ جيرانها
    # في الرسم البياني بذلك عبر NodeChannel — إشعار حقيقي، وليس مجرد رقم
    # في لوحة تحكم لا يقرأه أحد.
    def _apply_reputation_feedback(self, low_threshold: float = 25.0) -> None:
        for rep in self.reputation_engine.low_reputation_nodes(threshold=low_threshold):
            node_id = rep.get("node_id")
            if not node_id or rep.get("is_quarantined"):
                continue
            if rep.get("total_runs", 0) < 5:
                continue  # بيانات غير كافية بعد لاتخاذ قرار
            self.reputation_engine.quarantine(node_id)
            # حجر فعلي على مستوى العقدة نفسها (BaseNode.state=paused) لا
            # فقط في طبقة السمعة/التوجيه — بدون هذا كان أي استدعاء مباشر
            # لـ node.execute() (خارج مسار التوجيه القائم على السمعة) ينفّذ
            # عقدة محجورة دون أن يعرف.
            node = self.registry.get(node_id)
            if node:
                node.pause(reason="quarantine")
                self.registry.refresh_meta(node_id)
            if self.graph.has_node(node_id):
                neighbors = self.graph.get_neighbors(node_id)
                self.channel.broadcast(
                    from_id=node_id,
                    to_ids=neighbors,
                    topic="node_quarantined",
                    payload={"reason": "low_reputation", "score": rep.get("reputation_score")},
                )
            logger.info("MeshBundle: quarantined low-reputation node %s", node_id[:8])

    # ── رفع الحجر تلقائياً عن عقدة تعافت فعلياً + استئنافها الحقيقي ─────────
    # quarantine() كان طريقاً باتجاه واحد: unquarantine() موجودة في
    # NodeReputationEngine منذ البداية لكن لا شيء كان يستدعيها أبداً في كل
    # المستودع — أي عقدة تُحجَر تبقى محجورة للأبد حتى لو تحسّن أداؤها
    # الفعلي لاحقاً. هذه الدالة تفحص العُقد المحجورة، وتفكّ الحجر فعلياً
    # (في السمعة وحالة العقدة معاً عبر node.resume()) عن أي عقدة راكمت
    # تنفيذاً جديداً كافياً وتجاوزت درجتها الحقيقية (غير المُصفَّرة بسبب
    # الحجر) عتبة التعافي، ثم تُبلغ جيرانها.
    def _apply_reputation_recovery(self, recovery_threshold: float = 60.0,
                                    min_new_runs: int = 5) -> None:
        for rep in self.reputation_engine.release_eligible_nodes(
            recovery_threshold=recovery_threshold, min_new_runs=min_new_runs
        ):
            node_id = rep.get("node_id")
            if not node_id:
                continue
            self.reputation_engine.unquarantine(node_id)
            node = self.registry.get(node_id)
            if node:
                # expected_reason="quarantine": إن أوقف إنسان هذه العقدة
                # يدوياً بسبب آخر بعد حجرها (node.pause_reason != "quarantine"
                # الآن)، لا نستأنفها رغماً عنه فقط لأن سمعتها تعافت.
                resumed = node.resume(expected_reason="quarantine")
                if not resumed:
                    logger.info(
                        "MeshBundle: skipped auto-resume for %s — "
                        "currently paused for a different reason (%s)",
                        node_id[:8], node.pause_reason,
                    )
                    continue
                self.registry.refresh_meta(node_id)
            updated = self.reputation_engine.get_reputation(node_id)
            score = updated.reputation_score if updated else None
            if self.graph.has_node(node_id):
                neighbors = self.graph.get_neighbors(node_id)
                self.channel.broadcast(
                    from_id=node_id,
                    to_ids=[n for n in neighbors if n and n != node_id],
                    topic="node_unquarantined",
                    payload={"reason": "reputation_recovered", "score": score},
                )
            logger.info("MeshBundle: released quarantine for recovered node %s", node_id[:8])

    def summary(self) -> dict:
        _cm_summary = {}
        try:
            from ai.collective_memory import get_collective_memory
            _cm_summary = get_collective_memory().summary()
        except Exception:
            pass
        return {
            "nodes": self.registry.count(),
            "scoring": self.scoring_engine.summary(),
            "memory": self.memory_engine.summary(),
            "collective_memory": _cm_summary,
            "reputation": self.reputation_engine.summary(),
            "dna_versions": len(self.dna.history(limit=1000)),
            "exec_log": self.exec_log.db_stats(),
            "graph": self.graph.stats(),
            "channel": self.channel.stats(),
            "evolution": {
                "cycles_run": len(self.evolution._history),
                "generated": self.service_generator.summary(),
                "governance": self.governance.summary(),
            },
        }


@lru_cache(maxsize=1)
def get_mesh_bundle() -> MeshBundle:
    """Singleton حقيقي على مستوى العملية — يبقى حياً بين كل sessions/reruns."""
    return MeshBundle()
