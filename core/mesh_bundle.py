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
from core.engine import ExecutionEngine
from core.node_hands import NodeHands, LEFT, RIGHT
from services.dynamic_node import PassThroughNode
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
from ai.multi_goal_planner import MultiGoalPlanner
from ai.decision import AIDecisionLayer
from knowledge.knowledge_store import KnowledgeStore
from ai.discovery_engine import DiscoveryEngine
from ai.optimization_engine import OptimizationEngine

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# كل كم نتيجة سرب حقيقية (record_swarm_result) تُشغَّل دورة تطوّر ذاتي كاملة
# تلقائياً (انظر التعليق داخل record_swarm_result أدناه).
EVOLUTION_CYCLE_INTERVAL = 5
DIAGNOSE_LOW_REP_THRESHOLD = 0.15
DIAGNOSE_HIGH_UNREAD_THRESHOLD = 20
DIAGNOSE_INTERVAL_S = 120  # ثانية بين دورات التشخيص الخلفية

# اليد اليمنى للعقدة: أفعال مسموحة فقط (كل ما عداها مرفوض) + قيود صارمة.
HAND_RIGHT_ALLOWED = ("send_message", "request_evolution", "request_peer_ping")
HAND_EVOLUTION_COOLDOWN_S = 300.0   # أقل فاصل بين طلبَي تطوّر صادرَين من أي عُقد
HAND_MESSAGE_MAX_CHARS = 4000
HAND_TOPIC_MAX_CHARS = 64


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
        self.reputation_engine = NodeReputationEngine(
            memory_engine=self.memory_engine, storage=self.storage
        )
        self.dna = SystemDNA()

        # ai/*_engine.py (discovery/memory/optimization/routing/gap_detector...)
        # كلها مكتوبة بالكامل ومصمَّمة صراحة لتُحقَن بـKnowledgeStore واحد
        # مشترك (set_knowledge_store()/knowledge_store=... — التعليق نفسه
        # "Phase 3 knowledge layer" مكرَّر في كل واحدة)، لكن KnowledgeStore
        # لم تكن تُبنى (instantiate) في أي مكان بالمشروع كله (تحقّقت بالبحث
        # عن "KnowledgeStore(" في كل الملفات: صفر نتائج خارج تعريف الكلاس
        # نفسه ومثال docstring). النتيجة العملية على الأقل لـGapDetectionEngine:
        # self._knowledge=None دائماً فيصبح _persist_gaps() لا-عملية (no-op)
        # تماماً — أي فجوة تُكتشف تختفي فوراً عند إعادة التشغيل، وmark_resolved()
        # لا معنى له عملياً لأنه لا يوجد شيء محفوظ يُعاد تحميله أصلاً. هذا
        # السلك هنا مقصور على GapDetectionEngine (أوضح استهلاك جاهز) —
        # بقية المحركات (discovery/memory/optimization/routing) لا تزال
        # تنتظر نفس السلك في مهمة لاحقة.
        self.knowledge_store = KnowledgeStore(knowledge_dir=str(Path(storage_dir) / "knowledge"))
        # MemoryEngine نفسها لديها تخزينها الأساسي عبر SQLite (self._load()
        # أعلاه) وتعمل صحيحة بدونه، لكن set_knowledge_store() هنا يُفعِّل
        # تصديرها التكميلي الحقيقي (upsert_route/append_route_execution/
        # update_node_execution_stats/promote_route/demote_route — كلها
        # مكتوبة بالكامل في ai/memory_engine.py) إلى knowledge/route_memory.json
        # وknowledge/node_profiles.json. بلا هذا السطر: self._knowledge=None
        # في MemoryEngine (تحقّقت — لا مكان آخر يستدعي set_knowledge_store
        # لها)، فتبقى هذه الملفات بمخططها الافتراضي الفارغ للأبد رغم أن
        # ai/routing_engine.py مصمَّم صراحة ليقرأ منها فعلياً (بحسب
        # docstring الملف: "Reads best routes from knowledge/route_memory.json
        # via KnowledgeStore") — طبقة اكتشاف المسارات الجاهزة تبقى بلا أي
        # بيانات حقيقية تقرأها رغم أن MemoryEngine تراكم مسارات فعلية طوال
        # الوقت في SQLite.
        self.memory_engine.set_knowledge_store(self.knowledge_store)

        # ai/discovery_engine.py::DiscoveryEngine كانت مكتوبة بالكامل (إعلان
        # العُقد لنفسها بمخطط كامل + حفظ SQLite + كتابة profile دلالي إلى
        # knowledge/node_profiles.json) لكنها لا تُبنى في أي مكان بالمشروع
        # (تحقّقت بالبحث عن "DiscoveryEngine(": لا شيء خارج تعريفها). وهذا
        # هو السبب الفعلي لأن KnowledgeStore.update_node_execution_stats()
        # التي تستدعيها MemoryEngine بعد كل تنفيذ كانت no-op دائماً: تتطلب
        # profile مسجَّلاً مسبقاً للعقدة (upsert_node_profile) ولا أحد كان
        # يسجّله. هنا: نسخة واحدة مربوطة بنفس KnowledgeStore، وكل عقدة
        # مسجَّلة تُعلَن (انظر _announce_registered_nodes وregister_node).
        self.discovery_engine = DiscoveryEngine(db_path=db_path)
        self.discovery_engine.set_knowledge_store(self.knowledge_store)

        # storage/db.py::SQLiteStorage كان مكتوباً بالكامل (جداول nodes/
        # connections/execution_logs) لكن لم يُبنَ (instantiate) في أي مكان
        # بالمشروع — يُستخدم هنا كسجلّ تدقيق (audit log) حقيقي لتنفيذات
        # السرب، بنفس ملف data/mesh.db المشترك (لا تضارب أسماء جداول مع
        # MemoryEngine/ScoringEngine — تحقّقت من ذلك).
        self.exec_log = SQLiteStorage(db_path=db_path)

        self.agent_factory = AgentFactory()
        self.coordinator = SwarmCoordinator(
            self.agent_factory, max_agents=20,
            is_role_quarantined=self._is_role_quarantined,
            role_hands=self._hands_for_role,
            role_reputation=self._role_reputation,
            role_penalty_factor=self._role_penalty_factor_for_role,
        )

        # ── التواصل الحقيقي بين العُقد + رسم بياني حيّ للطوبولوجيا ──────────
        # (core/node_channel.py) قناة رسائل دائمة بين node_id حقيقية، و
        # (core/graph.py) رسم بياني يُستخدم فعلياً من GapDetectionEngine
        # لاكتشاف الفجوات ومن AIGovernanceLayer لفحص المسارات — كلاهما كان
        # موجوداً ومكتوباً بالكامل لكن بلا رسم بياني حيّ يُغذّيه.
        self.channel = NodeChannel(self.storage)
        self.graph = ServiceGraph()

        # ai/optimization_engine.py::OptimizationEngine كانت مكتوبة بالكامل
        # (analyze(): يحلّل self.graph/self.scoring_engine/self.memory_engine
        # الحقيقية فعلاً — يقترح حذف وصلات فاشلة باستمرار، ترقية وصلات
        # ناجحة، عُقداً غير مُستخدَمة إطلاقاً، وتحديث أوزان الحواف حسب الأداء
        # الفعلي؛ ثم يكتب المقاييس إلى knowledge/graph_metrics.json عبر
        # KnowledgeStore.record_optimization_run/update_node_rankings/
        # append_health_snapshot — كلها مكتوبة ومُختبَرة في
        # knowledge/knowledge_store.py) لكنها، تماماً كـDiscoveryEngine أعلاه
        # قبل إصلاحها، لم تُبنَ في أي مكان بالمشروع (تحقّقت بالبحث عن
        # "OptimizationEngine(": صفر نتائج خارج تعريف الكلاس نفسه). النتيجة:
        # knowledge/graph_metrics.json يبقى بمخططه الفارغ للأبد، ولا أحد
        # يكتشف وصلة فاشلة باستمرار أو عقدة لم تُنفَّذ قط إلا يدوياً عبر
        # dev_console. هنا: نسخة واحدة مربوطة بنفس self.graph/scoring_engine/
        # memory_engine/knowledge_store المشتركة (بعد بناء self.graph مباشرة،
        # لأن OptimizationEngine.__init__ يقبل graph=None لكن analyze()
        # يحتاجه فعلياً لكل خطوة تحليل). semantic_matcher عمداً غير مربوط
        # الآن (يبقى _suggest_new_connections لا-عملية بأمان بدونه) — نفس
        # نمط "سلك واحد مقصود، والبقية تنتظر مهمة لاحقة" الموثّق أعلاه لبقية
        # محركات discovery/memory/optimization/routing.
        #
        # 🆕 analyze() فقط تُستدعى تلقائياً أدناه (run_evolution_cycle) — لا
        # apply_report(). التصميم نفسه في OptimizationEngine يقول صراحة:
        # "Actions are generated but NOT auto-applied — the mesh or user
        # decides"؛ احترمت هذا القصد عمداً بدل تفعيل حذف حواف/عُقد تلقائياً
        # دون تقييم أثره على مسارات /process الحيّة. آخر تقرير متاح عبر
        # self.optimization_engine.last_report() لأي طبقة لاحقة (dev_console
        # أو تطبيق تلقائي مستقبلي) تقرّر استخدامه.
        self.optimization_engine = OptimizationEngine(
            graph=self.graph, scoring_engine=self.scoring_engine,
            memory_engine=self.memory_engine,
        )
        self.optimization_engine.set_knowledge_store(self.knowledge_store)

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
        # قبل الإحياء: جاهزية حقول الأيدي حتى يستطيع _restore استدعاء _equip_hands
        self._hand_evolution_last = 0.0
        # بيانات تشغيل حيّة لكل عقدة (آخر فحص طرفية، تشخيص...) — لا تُحفظ كحالة
        # دائمة للعقدة؛ تُعرض عبر node_status/self_diagnose فقط.
        self._node_runtime_meta: Dict[str, dict] = {}
        self._restore_dynamic_nodes()
        self._sync_nodes_to_exec_log()
        self._sync_nodes_to_graph()
        self._announce_registered_nodes()

        # ── «اليدان»: كل عقدة حيّة تحصل على يد يسرى (قراءة) ويمنى (أفعال مقيَّدة) ──
        # الدفاع الثاني: حتى لو فوّت _restore عقدة، هذه الحلقة تغطي كل registry
        for _n in self.registry.list_all():
            self._equip_hands(_n)

        # ── استئناف تلقائي للأسرِبة المتوقفة عند إعادة تشغيل العملية ────────
        # ai/swarm_coordinator.py::resume()/list_resumable() كانا يتطلّبان
        # نداءً يدوياً من المستخدم (زر في swarm_studio.py). هنا يُستأنف كل
        # سرب متوقف تلقائياً فور إقلاع MeshBundle (بلا انتظار زيارة المستخدم
        # للواجهة)، وتُغذّى نتيجته إلى دورة حياة العقد (record_swarm_result)
        # تماماً كأي تنفيذ طبيعي — بذلك تنعكس execution_count/state/السمعة
        # على العمل الذي اكتمل فعلاً بعد التوقف، لا أن تبقى مجمّدة. لا يجوز
        # لهذا أن يُعطّل إقلاع الحزمة أبداً مهما حدث.
        # في خيط خلفي daemon: resume() ينفّذ مهام LLM فعلية قد تستغرق دقائق
        # لكل سرب، وتشغيلها هنا كان يحجز إقلاع الحزمة (وأول تحميل للواجهة)
        # حتى ينتهي كل سرب متوقف. الخيط daemon فلا يمنع إغلاق العملية.
        threading.Thread(
            target=self._auto_resume_swarms, name="nsm-swarm-auto-resume",
            daemon=True,
        ).start()

        # ── استئناف تلقائي لعُقد ExecutionEngine + مهام المحتوى/الفيديو المتوقفة
        # قسراً — في نفس خيط _auto_resume_swarms الخلفي أدناه، وليس هنا في
        # __init__ نفسه. قبل هذا كانت الثلاثة تُستدعى متزامنة هنا مباشرة:
        # ExecutionEngine.resume_interrupted() ومهام محتوى/فيديو متوقفة قد
        # تشمل استدعاءات LLM أو ffmpeg طويلة فعلياً (دقائق لكل عنصر) — نفس
        # علّة "استئناف الأسرِبة يحجز إقلاع الحزمة" التي أُصلحت سابقاً
        # بنقلها لخيط daemon، لكن هذه الثلاثة أُضيفت لاحقاً بنفس التعليق
        # ("بنفس نمط استئناف الأسرِبة") دون أن تُنقَل فعلياً إلى الخيط —
        # التعليق وصف النية، والتنفيذ بقي متزامناً في __init__.
        threading.Thread(
            target=self._auto_resume_engine_and_jobs, name="nsm-jobs-auto-resume",
            daemon=True,
        ).start()

        # ── تشخيص دوري مستقل عن نتائج السرب (كل DIAGNOSE_INTERVAL_S) ──────
        self._diagnose_stop = threading.Event()
        threading.Thread(
            target=self._auto_nodes_diagnose_loop,
            name="nsm-nodes-diagnose",
            daemon=True,
        ).start()

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
            scoring_engine=self.scoring_engine, knowledge_store=self.knowledge_store,
        )
        # ai/service_generator.py::ServiceGeneratorEngine كانت تُبنى بلا
        # knowledge_store أيضاً (بحث منفصل عن الإصلاح أعلاه لـ
        # CapabilityMarketplace) — _persist_spec() مكتوبة بالكامل لكنها كانت
        # no-op دائماً، وبلا أي دالة استعادة مقابلة: كل GeneratedServiceSpec
        # (بما فيها الحالة proposed/approved/rejected وسياق الفجوة) تختفي
        # عند إعادة التشغيل رغم أن العُقد الحية نفسها تُستعاد فعلياً (إصلاح
        # سابق). أضفت _load_generated() في service_generator.py نفسها.
        self.service_generator = ServiceGeneratorEngine(
            governance=self.governance, knowledge_store=self.knowledge_store,
        )

        # ── ai/capability_marketplace.py::CapabilityMarketplace كانت تُبنى
        # بلا knowledge_store إطلاقاً (_persist() تصبح no-op دائماً رغم أنها
        # مكتوبة)، ولم يكن أي شيء يستدعي restore() المقابلة (أضيفت الآن) —
        # فيبدأ المتجر فارغاً بعد كل إعادة تشغيل. والأهم: advertise_from_node()
        # كانت تُستدعى فقط من ai/evolution_engine.py عند اعتماد عقدة
        # self_evolved جديدة، فتبقى الأدوار السبعة الثابتة (AGENT_CATALOGUE)
        # وأدوات MCP السبعة (_register_mcp_tools أعلاه) — أي كل ما هو موجود
        # فعلياً منذ إقلاع أول للمشروع — بلا أي إعلان قدرات في المتجر أبداً،
        # فيبقى find_providers()/best_provider() يرجعان فارغَين لأي طلب
        # حقيقي حتى تتطور أول عقدة ذاتياً. هنا: knowledge_store حقيقي،
        # استعادة أي إعلانات محفوظة من جلسة سابقة (تحافظ على درجات الجودة
        # المتراكمة)، ثم إعلان أي عقدة مسجَّلة حالياً لم تُعلَن بعد (تخطّي
        # ما استُعيد فعلاً كي لا تُصفَّر درجاته بإعادة advertise() افتراضية).
        self.marketplace = CapabilityMarketplace(knowledge_store=self.knowledge_store)
        try:
            self.marketplace.restore()
        except Exception as e:
            logger.warning("MeshBundle: تعذّرت استعادة CapabilityMarketplace: %s", e)
        for _node in self.registry.list_all():
            try:
                if not self.marketplace.capabilities_for_node(_node.node_id):
                    self.marketplace.advertise_from_node(_node)
            except Exception as e:
                logger.warning(
                    "MeshBundle: تعذّر إعلان قدرات العقدة %s في المتجر: %s",
                    getattr(_node, "node_id", "?")[:8], e,
                )

        self.evolution = EvolutionEngine(
            mesh=self,
            gap_detector=self.gap_detector,
            service_generator=self.service_generator,
            governance=self.governance,
            capability_marketplace=self.marketplace,
        )

        # ── ai/multi_goal_planner.py::MultiGoalPlanner كانت مكتوبة بالكامل
        # (تفكيك هدف مركّب إلى أهداف فرعية مرتَّبة، حلّ كل هدف فرعي إلى عقدة
        # فعلية عبر CapabilityMarketplace بدل اسم عقدة ثابت، ثم تنفيذ المسار
        # المركَّب بالكامل عبر ExecutionEngine) لكن لا يوجد أي مكان في
        # المشروع كله يبنيها فعلياً (تحقّقت بالبحث عن "MultiGoalPlanner(":
        # صفر نتائج خارج تعريف الكلاس نفسه) — أي أن المهام متعددة الخطوات
        # (تنظيف→ترجمة→تحليل مشاعر→تقرير، إلخ) لم يكن لها أي مسار تنفيذ
        # حقيقي إطلاقاً رغم اكتمال الكود. مثيل واحد هنا على مستوى الحزمة،
        # مربوط بنفس CapabilityMarketplace المشترك أعلاه، مع plan_and_execute_goal()
        # كنقطة استخدام فعلية (انظر أسفل).
        self.multi_goal_planner = MultiGoalPlanner(
            capability_marketplace=self.marketplace,
            memory_engine=self.memory_engine,
            knowledge_store=self.knowledge_store,
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
        # ── استعادة الروابط الديناميكية المحفوظة (self_evolved + نتائج سرب
        # سابقة) من exec_log.list_connections() (storage/db.py) ────────────
        # register_node() وrecord_swarm_result() كانا يستدعيان
        # exec_log.upsert_connection() ويحفظانها فعلياً في SQLite، لكن
        # ServiceGraph نفسها بُنيت هنا فقط من "mesh_member" (طوبولوجيا
        # الكتالوج الثابتة) — list_connections() كانت مكتوبة بالكامل
        # (storage/db.py) ومُستدعاة فقط لعدّ الروابط في stats()، بلا أي
        # مكان يعيد بناء الرسم البياني منها فعلياً. النتيجة العملية: بعد
        # أي إعادة تشغيل للعملية (كما يحدث فعلياً على Streamlit Cloud)،
        # كل رابط self_evolved أو رابط نتيجة سرب حقيقية يختفي من
        # ServiceGraph فوراً — GapDetectionEngine وAIGovernanceLayer
        # يفحصان طوبولوجيا أفقر من الواقع المحفوظ فعلياً، وأي بث
        # channel.broadcast يعتمد على graph.get_neighbors() يفوّت جيراناً
        # حقيقيين اكتسبتهم العقدة قبل إعادة التشغيل مباشرة.
        try:
            for conn in self.exec_log.list_connections():
                src, tgt = conn.get("source_id"), conn.get("target_id")
                if src and tgt and self.graph.has_node(src) and self.graph.has_node(tgt):
                    self.graph.add_edge(
                        src, tgt,
                        weight=conn.get("weight", 1.0),
                        label=conn.get("label", ""),
                    )
        except Exception as e:
            logger.warning("MeshBundle: تعذّرت استعادة روابط الرسم البياني من exec_log: %s", e)

    # ── واجهة التسجيل التي يتوقّعها EvolutionEngine (mesh.register_node) ────
    # EvolutionEngine._mesh.register_node(node, connect_to=...) هي نقطة
    # الوصل الوحيدة التي كان ينقصها — بدونها EvolutionEngine._mesh=None دائماً
    # ولا تُسجَّل أي عقدة مُولَّدة ذاتياً في الـregistry الحقيقي.
    # ── «اليدان» ────────────────────────────────────────────────────────────
    # (core/node_hands.py) اليسرى: قراءة فقط. اليمنى: فعلان فقط
    # (send_message / request_evolution) وتحت سياسة تمنعها عن أي عقدة
    # موقوفة/محجورة/فاشلة. لا shell ولا git push ولا fetch_url ولا توكنات.
    def pump_inboxes(self, max_per_node: int = 10) -> dict:
        """تشغّل بروتوكول نبض بسيط بين العُقد باستخدام يديها فقط: كل عقدة تقرأ
        رسائل "ping" في صندوقها (يد يسرى) وتردّ "pong" بـreply_to (يد يمنى).
        - لا ردّ على pong أبداً (لا حلقات).
        - مرسل غير مسجَّل أو ping من العقدة لنفسها: تُستهلك بلا ردّ.
        - يد محجوبة/محدودة المعدّل: تبقى الرسالة غير مقروءة وتُعاد المحاولة في
          الجولة التالية (لا فقدان صامت).
        - الرسائل غير ping (node_joined/node_failed...) تبقى كما هي لواجهة
          المراقبة ولا تُستهلك هنا.

        🆕 ترتيب حقيقي حسب السمعة: العُقد الأعلى سمعة تُعالَج أولاً، وداخل
        صندوق كل عقدة تُفضَّل رسائل المرسلين الأعلى سمعة — تحت حد المعدّل
        لليد اليمنى هذا يقلّل تأجيل نبض العُقد الموثوقة."""
        def _rep(nid: str) -> float:
            try:
                return float(self.reputation_engine.get_score(nid) or 0.0)
            except Exception:
                return 0.0

        stats = {
            "nodes": 0, "pings_answered": 0, "pings_dropped": 0, "deferred": 0,
            "order": "reputation_desc",
        }
        nodes = [
            n for n in self.registry.list_all()
            if getattr(n, "hands", None) is not None
        ]
        nodes.sort(key=lambda n: -_rep(getattr(n, "node_id", "") or ""))

        for node in nodes:
            r = node.use_hand(LEFT, "read_inbox", unread_only=True, topic="ping", limit=max_per_node)
            if not r.ok:
                continue
            stats["nodes"] += 1
            messages = list(r.output or [])
            messages.sort(
                key=lambda m: -_rep(str(m.get("from_id") or "")),
            )
            for m in messages:
                sender = m.get("from_id")
                if sender == node.node_id or not self.registry.exists(sender):
                    self.channel.mark_read(node.node_id, m["message_id"])
                    stats["pings_dropped"] += 1
                    continue
                seq = (m.get("payload") or {}).get("seq")
                payload = {"echo": seq} if isinstance(seq, (int, str)) and len(str(seq)) <= 64 else {}
                rep = node.use_hand(RIGHT, "send_message", to_id=sender, topic="pong",
                                    payload=payload, reply_to=m["message_id"])
                if rep.ok:
                    self.channel.mark_read(node.node_id, m["message_id"])
                    stats["pings_answered"] += 1
                else:
                    stats["deferred"] += 1
                    break
        return stats

    def _hands_policy(self, node: BaseNode, hand: str, tool: str, kwargs: dict):
        if hand == LEFT:
            return True, "read-only hand"
        if tool not in HAND_RIGHT_ALLOWED:
            return False, f"action {tool!r} is not in the right-hand allow-list"
        if node.state in (NodeState.PAUSED, NodeState.FAILED):
            return False, f"node state {node.state!r} cannot act; resume it first"
        return True, "allow-listed action"

    def _left_hand_tools(self, node: BaseNode):
        """أدوات قراءة محلية فقط، مضيَّقة: امتدادات نصية محددة، وبلا git remote
        (قد يكشف توكناً داخل الرابط).

        🆕 الأداوت الأساسية (peers + read_inbox) تُربَط دائماً حتى لو فشل
        استيراد ai.agent_tools — سابقاً كان استثناء الاستيراد يُسقط *كل*
        أدوات اليد اليسرى بما فيها peers/read_inbox (المستخدمتان في
        pump_inboxes والتواصل بين العقد)."""
        def peers():
            return [
                {"node_id": m.get("node_id"), "name": m.get("name"), "state": m.get("state")}
                for m in self.registry.list_metadata()
            ]

        def read_inbox(unread_only: bool = True, topic: Optional[str] = None, limit: int = 10):
            """يقرأ صندوق بريد هذه العقدة فقط (لا معامل node_id عمداً)، الأقدم
            أولاً، ويعيد نسخاً لا مراجع حيّة لرسائل القناة."""
            limit = max(1, min(int(limit), 50))
            msgs = self.channel.inbox(node.node_id, unread_only=bool(unread_only), limit=100)
            if topic is not None:
                msgs = [m for m in msgs if m.get("topic") == topic]
            return [dict(m) for m in msgs[:limit]]

        def node_status():
            """حالة هذه العقدة من السجل + السمعة + آخر فحص طرفية (قراءة فقط)."""
            meta = None
            for m in self.registry.list_metadata():
                if m.get("node_id") == node.node_id:
                    meta = m
                    break
            score = 0.0
            try:
                score = float(self.reputation_engine.get_score(node.node_id) or 0.0)
            except Exception:
                pass
            runtime = {}
            try:
                runtime = dict(self._node_runtime_meta.get(node.node_id) or {})
            except Exception:
                runtime = {}
            return {
                "node_id": node.node_id,
                "name": getattr(node, "name", None),
                "state": str(getattr(node, "state", None)),
                "execution_count": getattr(node, "execution_count", None),
                "last_executed": getattr(node, "last_executed", None) or (
                    meta.get("last_executed") if meta else None
                ),
                "node_type": type(node).__name__,
                "reputation_score": score,
                "last_terminal_check": runtime.get("last_terminal_check"),
                "hands_bound": getattr(node, "hands", None) is not None,
                "meta": {
                    k: meta.get(k) for k in ("tags", "pause_reason", "description")
                    if meta and k in meta
                } if meta else {},
            }

        def mesh_health():
            """ملخص صحة الشبكة من منظور MeshBundle (عدد العقد/الحالات/السمعة)."""
            metas = self.registry.list_metadata()
            by_state = {}
            for m in metas:
                st = str(m.get("state") or "unknown")
                by_state[st] = by_state.get(st, 0) + 1
            scores = []
            for m in metas:
                nid = m.get("node_id")
                if not nid:
                    continue
                try:
                    scores.append(float(self.reputation_engine.get_score(nid) or 0.0))
                except Exception:
                    pass
            avg_rep = (sum(scores) / len(scores)) if scores else 0.0
            return {
                "layer": "mesh-bundle-health-v1",
                "total_nodes": len(metas),
                "by_state": by_state,
                "avg_reputation": round(avg_rep, 4),
                "roles": len(getattr(self, "role_node_ids", {}) or {}),
                "viewer_node_id": node.node_id,
            }

        def routes():
            """جدول مسارات مبسّط: أقران + حالة + سمعة (قراءة فقط، شبيه
            NodeHealthLayer.routes_table لكن من منظور MeshBundle)."""
            rows = []
            for m in self.registry.list_metadata():
                nid = m.get("node_id")
                if not nid or nid == node.node_id:
                    continue
                score = 0.0
                try:
                    score = float(self.reputation_engine.get_score(nid) or 0.0)
                except Exception:
                    pass
                rows.append({
                    "peer_id": nid,
                    "name": m.get("name"),
                    "state": m.get("state"),
                    "node_type": m.get("node_type"),
                    "reputation": score,
                    "reachable": str(m.get("state") or "").lower() not in ("paused", "failed", "offline"),
                })
            rows.sort(key=lambda r: (-float(r.get("reputation") or 0), str(r.get("name") or "")))
            return {"routes": rows, "source": "mesh_bundle_registry", "viewer": node.node_id}

        def capabilities():
            """قدرات هذه العقدة: النوع، الوسوم، مخطط الإدخال/الإخراج."""
            try:
                in_s = node.input_schema
                out_s = node.output_schema
                in_fields = getattr(in_s, "fields", {}) or {}
                out_fields = getattr(out_s, "fields", {}) or {}
                in_req = list(getattr(in_s, "required", None) or [])
                out_req = list(getattr(out_s, "required", None) or [])
            except Exception:
                in_fields, out_fields, in_req, out_req = {}, {}, [], []
            return {
                "node_id": node.node_id,
                "name": getattr(node, "name", None),
                "node_type": type(node).__name__,
                "tags": list(getattr(node, "tags", None) or []),
                "description": (getattr(node, "description", None) or "")[:500],
                "input_fields": list(in_fields.keys()) if isinstance(in_fields, dict) else [],
                "input_required": in_req,
                "output_fields": list(out_fields.keys()) if isinstance(out_fields, dict) else [],
                "output_required": out_req,
                "hands": {
                    "left": [t["name"] for t in (node.hands.tools().get("left") or [])]
                    if getattr(node, "hands", None) else [],
                    "right": [t["name"] for t in (node.hands.tools().get("right") or [])]
                    if getattr(node, "hands", None) else [],
                },
            }

        def neighbors():
            """جيران العقدة في الرسم البياني للخدمات (أسلاف + أخلاف)."""
            nid = node.node_id
            try:
                succ = list(self.graph.get_neighbors(nid) or [])
            except Exception:
                succ = []
            try:
                pred = list(self.graph.get_predecessors(nid) or [])
            except Exception:
                pred = []
            def _brief(i):
                meta = None
                for m in self.registry.list_metadata():
                    if m.get("node_id") == i:
                        meta = m
                        break
                return {
                    "node_id": i,
                    "name": (meta or {}).get("name"),
                    "state": (meta or {}).get("state"),
                    "node_type": (meta or {}).get("node_type"),
                }
            return {
                "node_id": nid,
                "successors": [_brief(i) for i in succ],
                "predecessors": [_brief(i) for i in pred],
                "degree_out": len(succ),
                "degree_in": len(pred),
            }

        def reputation_detail():
            """تفاصيل سمعة هذه العقدة من محرك السمعة (إن وُجدت)."""
            try:
                rep = self.reputation_engine.get_reputation(node.node_id)
            except Exception as e:
                return {"node_id": node.node_id, "error": str(e)}
            if rep is None:
                return {
                    "node_id": node.node_id,
                    "score": float(self.reputation_engine.get_score(node.node_id) or 0.0),
                    "detail": None,
                }
            if hasattr(rep, "to_dict"):
                data = rep.to_dict()
            elif isinstance(rep, dict):
                data = dict(rep)
            else:
                data = {
                    "score": getattr(rep, "score", None),
                    "successes": getattr(rep, "successes", None),
                    "failures": getattr(rep, "failures", None),
                    "is_quarantined": getattr(rep, "is_quarantined", None),
                }
            data["node_id"] = node.node_id
            return data

        def graph_stats():
            """إحصاءات الرسم البياني للخدمات (قراءة فقط)."""
            try:
                return dict(self.graph.stats() or {})
            except Exception as e:
                return {"error": str(e)}

        tools = [
            ("read_inbox", read_inbox, "قراءة صندوق بريد هذه العقدة فقط (الأقدم أولاً)"),
            ("peers", peers, "قائمة العُقد المعروفة وحالاتها"),
            ("node_status", node_status, "حالة هذه العقدة + سمعتها من السجل"),
            ("mesh_health", mesh_health, "ملخص صحة الشبكة (عدد/حالات/سمعة متوسطة)"),
            ("routes", routes, "جدول مسارات الأقران مع الحالة والسمعة"),
            ("capabilities", capabilities, "نوع العقدة والوسوم ومخطط الإدخال/الإخراج واليدين"),
            ("neighbors", neighbors, "جيران العقدة في رسم الخدمات (أسلاف/أخلاف)"),
            ("reputation_detail", reputation_detail, "تفاصيل سمعة هذه العقدة من المحرك"),
            ("graph_stats", graph_stats, "إحصاءات الرسم البياني للخدمات"),
        ]

        # ── طرفية آمنة (قراءة/فحوص فقط عبر allowlist) ──────────────────────
        def terminal_policy() -> dict:
            """ماذا يُسمح تلقائياً من أوامر الطرفية للعقدة."""
            try:
                from ai.terminal_auto_policy import explain_policy, list_allowed_examples
                policy_text = explain_policy()
                examples = list_allowed_examples()
            except Exception as e:
                policy_text = f"policy unavailable: {e}"
                examples = []
            return {
                "mode": "safe-allowlist",
                "shell": False,
                "operators_banned": [";", "&&", "||", "|", ">", ">>", "<"],
                "policy": policy_text,
                "examples": examples,
                "node_id": node.node_id,
            }

        def terminal_run_safe(cmd: str, timeout: int = 30) -> dict:
            """تشغيل أمر طرفية مسموح فقط (git status/diff، pytest، py_compile...).
            بلا shell وبلا كتابة/حذف/شبكة. أي أمر خارج القائمة → مرفوض.
            يحدّث last_terminal_check في بيانات تشغيل العقدة."""
            if not isinstance(cmd, str) or not cmd.strip():
                raise ValueError("cmd must be a non-empty string")
            if len(cmd) > 500:
                raise ValueError("cmd too long (max 500 chars)")
            timeout = max(1, min(int(timeout), 60))
            try:
                from ai.agent_tools import run_safe_cmd
                result = run_safe_cmd(cmd.strip(), timeout=timeout)
            except Exception:
                from ai.terminal_auto_policy import decide, run_auto
                from pathlib import Path as _P
                decision = decide(cmd.strip())
                if not decision.allowed:
                    result = {
                        "ok": False,
                        "cmd": cmd.strip(),
                        "msg": decision.reason,
                        "automatic": False,
                        "requires_approval": True,
                    }
                else:
                    root = _P(__file__).resolve().parent.parent
                    output = run_auto(cmd.strip(), cwd=str(root), timeout=timeout)
                    result = {
                        "ok": output.startswith("exit=0"),
                        "cmd": list(decision.command) if decision.command else cmd.strip(),
                        "output": output,
                        "automatic": True,
                    }
            try:
                from datetime import datetime, timezone as _tz
                entry = {
                    "at": datetime.now(_tz.utc).isoformat(),
                    "cmd": cmd.strip()[:120],
                    "ok": bool(result.get("ok")) if isinstance(result, dict) else False,
                    "requires_approval": bool(
                        (result or {}).get("requires_approval")
                    ) if isinstance(result, dict) else False,
                }
                with self._lock:
                    meta = dict(self._node_runtime_meta.get(node.node_id) or {})
                    meta["last_terminal_check"] = entry
                    hist = list(meta.get("terminal_checks") or [])
                    hist.append(entry)
                    meta["terminal_checks"] = hist[-20:]
                    self._node_runtime_meta[node.node_id] = meta
            except Exception:
                pass
            return result

        def terminal_history(limit: int = 20) -> list:
            """سجل أوامر طرفية هذا الوكيل/العقدة إن وُجدت (قراءة فقط)."""
            limit = max(1, min(int(limit), 50))
            key = getattr(node, "name", None) or node.node_id
            # أولاً: سجل فحوص terminal_run_safe المحلية على هذه العقدة
            local = []
            try:
                local = list(
                    (self._node_runtime_meta.get(node.node_id) or {}).get("terminal_checks")
                    or []
                )
            except Exception:
                local = []
            remote = []
            try:
                from ai.agent_terminals import get_agent_terminals
                remote = get_agent_terminals().agent_history(str(key), limit=limit)
            except Exception as e:
                remote = [{"error": str(e), "node": key}]
            return {
                "node_checks": list(reversed(local))[:limit],
                "agent_terminal": remote[:limit] if isinstance(remote, list) else remote,
            }

        def inbox_summary() -> dict:
            """ملخص صندوق البريد: عدد غير المقروء حسب الموضوع + إشارة mesh_diagnose."""
            try:
                msgs = self.channel.inbox(node.node_id, unread_only=True, limit=100)
            except Exception as e:
                return {"error": str(e), "node_id": node.node_id}
            by_topic: dict = {}
            for m in msgs or []:
                t = str(m.get("topic") or "unknown")
                by_topic[t] = by_topic.get(t, 0) + 1
            # أحدث إشارات تشخيص جماعية (مقروءة أو لا) كمؤشر صحة الشبكة
            collective = None
            try:
                all_diag = self.channel.inbox(
                    node.node_id, unread_only=False, topic="mesh_diagnose", limit=5,
                )
                if all_diag:
                    latest = all_diag[-1] if isinstance(all_diag, list) else None
                    if latest and isinstance(latest.get("payload"), dict):
                        collective = {
                            "from_id": latest.get("from_id"),
                            "message_id": latest.get("message_id"),
                            "payload": latest.get("payload"),
                        }
            except Exception:
                collective = None
            return {
                "node_id": node.node_id,
                "unread_total": len(msgs or []),
                "by_topic": by_topic,
                "mesh_diagnose_unread": int(by_topic.get("mesh_diagnose") or 0),
                "collective_diagnose": collective,
            }

        def self_diagnose() -> dict:
            """تشخيص موحّد: حالة + صحة الشبكة + جيران + وارد + صحة جماعية."""
            st = node_status()
            try:
                mh = mesh_health()
            except Exception as e:
                mh = {"error": str(e)}
            try:
                nb = neighbors()
            except Exception as e:
                nb = {"error": str(e)}
            try:
                ib = inbox_summary()
            except Exception as e:
                ib = {"error": str(e)}
            try:
                rt = routes()
                top_routes = (rt.get("routes") or [])[:5]
            except Exception:
                top_routes = []
            try:
                cycle = self.get_nodes_diagnose_summary() or {}
            except Exception:
                cycle = {}
            collective = (ib or {}).get("collective_diagnose") if isinstance(ib, dict) else None
            collective_health = {
                "has_signal": collective is not None,
                "from_cycle_local": {
                    "scanned": cycle.get("scanned", 0),
                    "low_reputation": len(cycle.get("low_reputation") or []),
                    "high_unread": len(cycle.get("high_unread") or []),
                    "ts": cycle.get("ts"),
                },
            }
            if collective and isinstance(collective.get("payload"), dict):
                pl = collective["payload"]
                collective_health["from_peer_broadcast"] = {
                    "ts": pl.get("ts"),
                    "scanned": pl.get("scanned"),
                    "low_reputation_count": pl.get("low_reputation_count"),
                    "high_unread_count": pl.get("high_unread_count"),
                    "from_id": collective.get("from_id"),
                }
            return {
                "layer": "node-self-diagnose-v2",
                "status": st,
                "mesh_health": {
                    k: mh.get(k) for k in ("total_nodes", "by_state", "avg_reputation")
                    if isinstance(mh, dict)
                },
                "neighbors": nb,
                "inbox": ib,
                "top_routes_by_reputation": top_routes,
                "collective_health": collective_health,
            }

        def mesh_diagnose_summary() -> dict:
            """آخر ملخص دورة التشخيص الدورية لكل الأدوار (قراءة فقط)."""
            return self.get_nodes_diagnose_summary() or {
                "scanned": 0,
                "note": "لم تُشغَّل دورة تشخيص بعد",
            }

        def peer_compare() -> dict:
            """مقارنة سمعة هذه العقدة بمتوسط الأقران وترتيبها."""
            try:
                my = float(self.reputation_engine.get_score(node.node_id) or 0.0)
            except Exception:
                my = 0.0
            peers = []
            for m in self.registry.list_metadata():
                nid = m.get("node_id")
                if not nid or nid == node.node_id:
                    continue
                try:
                    sc = float(self.reputation_engine.get_score(nid) or 0.0)
                except Exception:
                    sc = 0.0
                peers.append({
                    "node_id": nid,
                    "name": m.get("name"),
                    "state": m.get("state"),
                    "reputation": sc,
                })
            peers.sort(key=lambda r: -float(r.get("reputation") or 0))
            avg = (sum(p["reputation"] for p in peers) / len(peers)) if peers else 0.0
            rank = 1 + sum(1 for p in peers if p["reputation"] > my)
            return {
                "node_id": node.node_id,
                "my_reputation": round(my, 4),
                "peer_avg_reputation": round(avg, 4),
                "rank_among_peers": rank,
                "peers_total": len(peers),
                "delta_vs_avg": round(my - avg, 4),
                "top_peers": peers[:5],
            }

        tools.extend([
            ("terminal_policy", terminal_policy, "سياسة الطرفية الآمنة المسموحة للعقدة"),
            ("terminal_run_safe", terminal_run_safe,
             "تشغيل أمر طرفية من القائمة الآمنة فقط (git status/pytest/py_compile...)"),
            ("terminal_history", terminal_history, "سجل فحوص الطرفية المحلية + طرفية الوكيل"),
            ("inbox_summary", inbox_summary, "ملخص الرسائل غير المقروءة حسب الموضوع"),
            ("self_diagnose", self_diagnose,
             "تشخيص موحّد: حالة + صحة الشبكة + جيران + وارد + مسارات"),
            ("mesh_diagnose_summary", mesh_diagnose_summary,
             "ملخص آخر دورة تشخيص دورية لكل عُقد الأدوار"),
            ("peer_compare", peer_compare,
             "مقارنة سمعة هذه العقدة بمتوسط الأقران وترتيبها"),
        ])

        try:
            from ai import agent_tools as at
        except Exception as e:
            logger.warning("MeshBundle: ai.agent_tools غير متاح — أدوات القراءة الإضافية معطّلة: %s", e)
            return tools

        def search_code(pattern: str, path: str = ".", glob: str = "*.py", max_matches: int = 20):
            if glob not in ("*.py", "*.md"):
                raise ValueError("glob must be '*.py' or '*.md'")
            return at.search_code(pattern, path=path, glob=glob, max_matches=min(int(max_matches), 40))

        def find_files(name_glob: str = "*.py", path: str = ".", limit: int = 50):
            return at.find_files(name_glob, path=path, limit=min(int(limit), 80))

        def git_info(what: str = "status"):
            if what not in ("status", "log", "diff", "branch"):
                raise ValueError("what must be one of: status, log, diff, branch")
            return at.git_info(what)

        tools.extend([
            ("search_code", search_code, "بحث نصي/regex في ملفات py/md للمشروع"),
            ("find_files", find_files, "بحث عن ملفات بالاسم/الامتداد"),
            ("git_info", git_info, "status/log/diff/branch (قراءة فقط)"),
            ("py_compile_check", at.py_compile_check, "فحص بناء جملة ملف Python"),
            ("system_info", at.system_info, "لمحة عن البيئة بلا أسرار"),
        ])
        return tools

    def _equip_hands(self, node: BaseNode) -> None:
        if getattr(node, "hands", None) is not None:
            return
        hands = NodeHands(
            node,
            policy=self._hands_policy,
            audit_path=Path(self.storage.storage_dir) / "node_hands_audit.jsonl",
        )
        try:
            for name, fn, desc in self._left_hand_tools(node):
                hands.bind(LEFT, name, fn, desc)
        except Exception as e:
            logger.warning("MeshBundle: تعذّر ربط أدوات اليد اليسرى لـ %s: %s", node.name, e)

        def send_message(to_id: str, topic: str, payload: Optional[dict] = None,
                         reply_to: Optional[str] = None):
            if not isinstance(topic, str) or not topic or len(topic) > HAND_TOPIC_MAX_CHARS:
                raise ValueError(f"topic must be a non-empty string up to {HAND_TOPIC_MAX_CHARS} chars")
            if not self.registry.exists(to_id):
                raise ValueError("recipient is not a registered node")
            import json as _json
            if len(_json.dumps(payload or {}, ensure_ascii=False, default=str)) > HAND_MESSAGE_MAX_CHARS:
                raise ValueError(f"payload exceeds {HAND_MESSAGE_MAX_CHARS} chars")
            if reply_to is not None and (not isinstance(reply_to, str) or len(reply_to) > HAND_TOPIC_MAX_CHARS):
                raise ValueError("reply_to must be a message_id string")
            msg = self.channel.send(node.node_id, to_id, topic, payload or {}, reply_to=reply_to)
            return {"message_id": msg["message_id"], "to_id": to_id}

        def request_evolution():
            import time as _time
            now = _time.time()
            with self._lock:
                wait = HAND_EVOLUTION_COOLDOWN_S - (now - self._hand_evolution_last)
                if wait > 0:
                    raise RuntimeError(f"evolution cooldown: retry in {int(wait)}s")
                self._hand_evolution_last = now
            return self.run_evolution_cycle()

        def request_peer_ping(to_id: Optional[str] = None, seq: Optional[str] = None):
            """نبض ping موجّه: إن حُدّد to_id يُستخدم (إن كان مسجّلاً وليس الذات)،
            وإلا يُختار أعلى الأقران سمعةً من غير الذات. لا حلقات — topic=ping فقط."""
            target = to_id
            if target is not None:
                if not isinstance(target, str) or not target:
                    raise ValueError("to_id must be a non-empty string")
                if target == node.node_id:
                    raise ValueError("cannot ping self")
                if not self.registry.exists(target):
                    raise ValueError("recipient is not a registered node")
            else:
                best_id, best_score = None, float("-inf")
                for m in self.registry.list_metadata():
                    nid = m.get("node_id")
                    if not nid or nid == node.node_id:
                        continue
                    st = str(m.get("state") or "").lower()
                    if st in ("paused", "failed", "offline"):
                        continue
                    try:
                        sc = float(self.reputation_engine.get_score(nid) or 0.0)
                    except Exception:
                        sc = 0.0
                    if sc > best_score:
                        best_score, best_id = sc, nid
                if not best_id:
                    raise RuntimeError("no reachable peer to ping")
                target = best_id
            if seq is None:
                import uuid as _uuid
                seq = _uuid.uuid4().hex[:8]
            elif not isinstance(seq, (str, int)) or len(str(seq)) > 64:
                raise ValueError("seq must be str/int up to 64 chars")
            msg = self.channel.send(
                node.node_id, target, "ping", {"seq": str(seq)}, reply_to=None,
            )
            return {
                "message_id": msg["message_id"],
                "to_id": target,
                "topic": "ping",
                "seq": str(seq),
            }

        hands.bind(RIGHT, "send_message", send_message, "إرسال رسالة لعقدة مسجَّلة أخرى")
        hands.bind(RIGHT, "request_evolution", request_evolution,
                   "طلب دورة تطوّر ذاتي (تمرّ بالحوكمة وتبريد 5 دقائق)")
        hands.bind(RIGHT, "request_peer_ping", request_peer_ping,
                   "إرسال نبض ping لأعلى الأقران سمعة (أو to_id محدد)")
        node.attach_hands(hands)

    def register_node(self, node: BaseNode, connect_to: Optional[str] = None) -> str:
        node_id = self.registry.register(node)
        self.graph.add_node(node_id, node.to_dict())
        self.exec_log.upsert_node(node.to_dict())
        self._announce_node(node)
        self._equip_hands(node)
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

    # ── إعلان العُقد في DiscoveryEngine (يملأ node_profiles.json) ───────────
    def _announce_node(self, node: BaseNode) -> None:
        try:
            self.discovery_engine.announce(node)
        except Exception as e:
            logger.warning(
                "MeshBundle: تعذّر إعلان العقدة %s في DiscoveryEngine: %s",
                getattr(node, "node_id", "?")[:8], e,
            )

    def _announce_registered_nodes(self) -> None:
        """يعلن كل عقدة مسجَّلة لم تُعلَن بعد (تخطّي المُعلَنة سابقاً يحفظ
        announced_at الأصلي؛ إعادة مزامنتها إلى knowledge تتم أصلاً داخل
        DiscoveryEngine.set_knowledge_store)."""
        for node in self.registry.list_all():
            if self.discovery_engine.get_announcement(node.node_id) is None:
                self._announce_node(node)

    # ── تسجيل كل الأدوار الموجودة في الكتالوج كعُقد حقيقية داخل الـregistry ──
    def _auto_resume_engine_and_jobs(self) -> None:
        """يستأنف عُقد ExecutionEngine المتوقفة قسراً ومهام المحتوى/الفيديو
        الخلفية — في خيط daemon (انظر التعليق في __init__)، بنفس منطق
        الكتلة الأصلية حرفياً، فقط بلا حجب إقلاع الحزمة. لا يرفع استثناءً
        أبداً؛ كل قسم معزول بـtry/except خاص كما كان."""
        try:
            engine = ExecutionEngine(
                self.registry, self.graph, self.storage,
                db=self.exec_log, ai=self.ai_decision,
            )
            resumed_results = engine.resume_interrupted()
            for r in resumed_results:
                logger.info(
                    "MeshBundle: استؤنفت عقدة توقفت قسراً — run_id=%s status=%s",
                    r.run_id, r.status,
                )
        except Exception as exc:
            logger.warning("MeshBundle: تعذّر فحص العُقد المتوقفة قسراً: %s", exc)

        try:
            from ai.content_job_manager import get_content_job_manager
            resumed_jobs = get_content_job_manager().resume_interrupted()
            for jid in resumed_jobs:
                logger.info("MeshBundle: استُؤنفت مهمة محتوى متوقفة #%s", jid)
        except Exception as exc:
            logger.warning("MeshBundle: تعذّر فحص مهام المحتوى المتوقفة: %s", exc)

        try:
            from ai.video_job_manager import get_video_job_manager
            resumed_video_jobs = get_video_job_manager().resume_interrupted()
            for jid in resumed_video_jobs:
                logger.info("MeshBundle: استُؤنفت مهمة فيديو متوقفة #%s", jid)
        except Exception as exc:
            logger.warning("MeshBundle: تعذّر فحص مهام الفيديو المتوقفة: %s", exc)

    def _auto_resume_swarms(self) -> None:
        """يستأنف الأسرِبة المتوقفة (يُشغَّل في خيط خلفي عند الإقلاع). لا يرفع
        استثناءً أبداً؛ list_resumable/resume يتجاوزان أي سرب حي حالياً."""
        try:
            for cp in self.coordinator.list_resumable():
                swarm_id = cp.get("swarm_id")
                if not swarm_id:
                    continue
                try:
                    resumed = self.coordinator.resume(swarm_id)
                    if resumed:
                        self.record_swarm_result(resumed)
                        logger.info(
                            "MeshBundle: استؤنف تلقائياً سرب متوقف %s "
                            "(%d/%d مهمة ناجحة)",
                            swarm_id, resumed.success_count, len(resumed.tasks),
                        )
                except Exception as exc:
                    logger.warning(
                        "MeshBundle: تعذّر استئناف السرب %s تلقائياً: %s",
                        swarm_id, exc,
                    )
        except Exception as exc:
            logger.warning("MeshBundle: تعذّر فحص الأسرِبة المتوقفة: %s", exc)

    def _routing_penalty_factor(self, node_id: str, raw_score: float, diag: Optional[dict] = None) -> float:
        """عامل تخفيض توجيه السرب: 0.1 كامل / 0.5 تعافٍ / 1.0 بلا عقوبة.

        يُرفع تدريجياً عندما تتجاوز السمعة الخام العتبة الديناميكية حتى لو
        بقيت العقدة في قائمة low_reputation لدورة سابقة."""
        if not node_id:
            return 1.0
        try:
            diag = diag if diag is not None else (self.get_nodes_diagnose_summary() or {})
            low_ids = {
                x.get("node_id")
                for x in (diag.get("low_reputation") or [])
                if isinstance(x, dict)
            }
            if node_id not in low_ids:
                return 1.0
            thr = float(
                diag.get("effective_low_rep_threshold")
                or DIAGNOSE_LOW_REP_THRESHOLD
            )
            if raw_score >= thr * 1.2:
                return 1.0  # تعافٍ كامل
            if raw_score >= thr:
                return 0.5  # تعافٍ جزئي
            return 0.1  # عقوبة كاملة
        except Exception:
            return 1.0

    def get_routing_penalties(self) -> dict:
        """الأدوار المعاقَبة حالياً في توجيه السرب — للواجهة والتدقيق."""
        diag = self.get_nodes_diagnose_summary() or {}
        thr = float(diag.get("effective_low_rep_threshold") or DIAGNOSE_LOW_REP_THRESHOLD)
        id_to_role = {v: k for k, v in (getattr(self, "role_node_ids", None) or {}).items()}
        roles = []
        for x in diag.get("low_reputation") or []:
            if not isinstance(x, dict):
                continue
            nid = x.get("node_id")
            if not nid:
                continue
            try:
                raw = float(self.reputation_engine.get_score(nid) or 0.0)
            except Exception:
                raw = float(x.get("reputation") or 0.0)
            factor = self._routing_penalty_factor(nid, raw, diag)
            role = id_to_role.get(nid) or x.get("name")
            roles.append({
                "role": role,
                "node_id": nid,
                "raw_reputation": round(raw, 4),
                "effective_reputation": round(raw * factor, 4),
                "penalty_factor": factor,
                "threshold": round(thr, 4),
                "status": (
                    "recovered" if factor >= 1.0
                    else ("recovering" if factor >= 0.5 else "penalized")
                ),
            })
        return {
            "threshold": round(thr, 4),
            "penalized_count": sum(1 for r in roles if r["penalty_factor"] < 1.0),
            "roles": roles,
        }

    def _role_reputation(self, role) -> float:
        """درجة سمعة عقدة الدور في الـregistry — تُمرَّر لـSwarmCoordinator
        لتفضيل الأدوار الأعلى سمعة عند توزيع المهام. 0.0 إن لم تُوجد عقدة.

        عقوبة تشخيص low_reputation مع رفع تدريجي بعد تجاوز العتبة الديناميكية."""
        if not role:
            return 0.0
        node_id = (getattr(self, "role_node_ids", None) or {}).get(role)
        if not node_id:
            return 0.0
        try:
            score = float(self.reputation_engine.get_score(node_id) or 0.0)
        except Exception:
            score = 0.0
        try:
            diag = self.get_nodes_diagnose_summary() or {}
            factor = self._routing_penalty_factor(node_id, score, diag)
            score *= factor
        except Exception:
            pass
        return score

    def _role_penalty_factor_for_role(self, role) -> float:
        """عامل العقوبة لدور (1.0 / 0.5 / 0.1) — لتدقيق _pick_agent."""
        if not role:
            return 1.0
        node_id = (getattr(self, "role_node_ids", None) or {}).get(role)
        if not node_id:
            return 1.0
        try:
            raw = float(self.reputation_engine.get_score(node_id) or 0.0)
        except Exception:
            raw = 0.0
        return float(self._routing_penalty_factor(node_id, raw))

    def _hands_for_role(self, role) -> Optional[NodeHands]:
        """«اليد اليسرى» الحقيقية لعقدة هذا الدور في الـregistry، إن
        وُجدت — تُمرَّر لـSwarmCoordinator._run_task فيستخدمها الوكيل عبر
        NSMAgent.run(hands=...) «إن كان مناسباً» للمهمة بالفعل (قراءة
        كود/ملفات/حالة أقران فقط؛ اليد اليمنى مخصّصة لبروتوكول pump_inboxes
        بين العُقد ولا تُعرَض هنا). بدون عقدة مسجَّلة لهذا الدور، أو بدون
        يدين مُرفَقتين بعد: None — فيتصرّف NSMAgent.run كأن hands لم
        تُمرَّر أصلاً (سلوك ما قبل هذه الميزة تماماً)."""
        if not role:
            return None
        node_id = getattr(self, "role_node_ids", {}).get(role)
        if not node_id:
            return None
        node = self.registry.get(node_id)
        return getattr(node, "hands", None) if node else None

    def _is_role_quarantined(self, role) -> bool:
        """هل الدور محجور حالياً بسبب سمعة منخفضة؟ تُمرَّر لـSwarmCoordinator
        كدالة فحص حتى لا تُوجَّه مهام جديدة لدور محجور. تقرأ role_node_ids
        وreputation_engine وقت الاستدعاء (لا وقت البناء)، فترتيب التهيئة
        داخل __init__ لا يهم."""
        if not role:
            return False
        node_id = getattr(self, "role_node_ids", {}).get(role)
        if not node_id:
            return False
        rep = self.reputation_engine.get_reputation(node_id)
        return bool(rep and rep.is_quarantined)

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

    # ── إحياء عُقد ديناميكية (self_evolved) بعد إعادة تشغيل العملية ─────────
    def _restore_dynamic_nodes(self) -> None:
        """يعيد بناء كائن Python حيّ لكل عقدة محفوظة في التخزين لكن ليست
        جزءاً من الكتالوج الثابت (أدوار/أدوات MCP يُعاد بناؤها أعلاه بنوعها
        الصحيح دائماً). قبل هذه الدالة كانت عُقد self_evolved (التي يُنشئها
        ServiceGeneratorEngine أثناء دورة تطوّر ذاتي حقيقية عبر
        register_node()) 'أشباحاً' بعد أي إعادة تشغيل: موجودة في
        registry.list_metadata()/exec_log لكن غائبة تماماً عن
        registry.list_all()/get_by_state()، فتصبح غير مرئية لـ
        GapDetectionEngine وServiceGraph، وأي محاولة حجر/رفع حجر بالسمعة
        عليها (self.registry.get(node_id)) ترجع None بصمت.

        منذ إضافة services/generated_service_nodes.py، عُقد self_evolved
        الحديثة (النوع الحقيقي — NormalizerNode/ValidatorNode/...، وليس
        PassThroughNode دائماً) تحمل نوعها الفعلي في
        meta["node_type"] = BaseNode.metadata.node_type = self.__class__.__name__،
        محفوظ فعلياً في meta_cache منذ لحظة تسجيلها لا مُخمَّناً هنا. نستخدم
        NODE_CLASS_BY_NAME لإعادة بنائها بنفس صنفها الحقيقي — لا نُنزلها إلى
        PassThroughNode عام إلا لعقدة قديمة/غير معروفة النوع (سجلّ من قبل
        هذه الخريطة، أو PassThroughNode أصلاً)، وrestore_state() يستعيد
        حالتها/تاريخها الحقيقي (state, pause_reason, execution_count) بنفس
        node_id تماماً في الحالتين."""
        from services.generated_service_nodes import NODE_CLASS_BY_NAME
        for meta in self.registry.orphaned_metadata():
            node_id = meta.get("node_id")
            if not node_id:
                continue
            shell_cls = NODE_CLASS_BY_NAME.get(meta.get("node_type"), PassThroughNode)
            shell = shell_cls(
                name=meta.get("name", node_id[:8]),
                description=meta.get("description", ""),
                tags=meta.get("tags") or ["dynamic", "passthrough", "restored"],
                node_id=node_id,
            )
            shell.restore_state(meta)
            try:
                self.registry.register(shell)
                # 🆕 دفاع صريح: تجهيز اليدين فور الإحياء — لا نعتمد فقط على
                # حلقة list_all اللاحقة في __init__ (قد تتغيّر ترتيباً أو تُتخطى).
                self._equip_hands(shell)
                logger.info(
                    "MeshBundle: أُحييت عقدة ديناميكية بعد إعادة التشغيل: %s [%s] (hands=%s)",
                    shell.name, node_id[:8],
                    "yes" if getattr(shell, "hands", None) else "no",
                )
            except ValueError:
                pass  # سبقتنا خطوة أخرى لتسجيلها بنفس node_id — لا مشكلة

    # ── تغذية نتيجة تنفيذ سرب حقيقية إلى Scoring + Memory + Reputation ──────
    def record_swarm_result(self, swarm_result) -> None:
        """
        يأخذ SwarmResult حقيقياً (من SwarmCoordinator.execute) ويغذّي كل
        مهمة فرعية فيه إلى ScoringEngine و MemoryEngine و
        NodeReputationEngine — وهذا هو الرابط الذي كان مفقوداً: نتائج
        السرب كانت تُعرض في الواجهة فقط ولا تصل أبداً لمحركات التقييم/
        الذاكرة/السمعة.

        🆕 يلتقط أيضاً آخر قرارات _pick_audit ويربطها بنتيجة السرب
        (شفافية توزيع المهام حسب السمعة) دون تغيير منطق السمعة نفسه.
        """
        # لقطة قرارات التوزيع قبل القفل الطويل
        pick_snapshot = []
        try:
            pick_snapshot = list(getattr(self.coordinator, "get_pick_audit", lambda **k: [])(limit=20) or [])
        except Exception:
            pick_snapshot = []
        if pick_snapshot and not hasattr(swarm_result, "_pick_audit_snapshot"):
            try:
                setattr(swarm_result, "_pick_audit_snapshot", pick_snapshot)
            except Exception:
                pass

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
                        # 🆕 إصلاح: "task" لم يكن مفتاحاً موجوداً إطلاقاً في
                        # قاموس task.result (مفاتيحه الحقيقية: sub_goal/
                        # result_text/... — انظر SwarmCoordinator._run_task)،
                        # فكان .get("task") يُرجع None دائماً في المسار
                        # الشائع (أي مهمة نُفّذت فعلاً ولها نتيجة)، فتنكسر
                        # _extract_domain(None).lower() صامتاً في كل مرة.
                        # sub_goal الحقيقي متاح مباشرة على task نفسه.
                        task=getattr(task, "sub_goal", "") or "",
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
                self._apply_node_retirement()
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
                    self.pump_inboxes()
                except Exception as e:
                    logger.warning("MeshBundle: periodic pump_inboxes failed: %s", e)
                try:
                    self.run_nodes_diagnose_cycle()
                except Exception as e:
                    logger.warning("MeshBundle: periodic nodes diagnose cycle failed: %s", e)
                try:
                    self.run_evolution_cycle()
                except Exception as e:
                    logger.warning("MeshBundle: periodic run_evolution_cycle after swarm result failed: %s", e)

    def snapshot_node_states(self) -> Dict[str, str]:
        """لقطة {node_id: state} لكل عُقدة مسجَّلة حالياً — يستدعيها المتصل
        (api_server.py::/process) قبل تنفيذ فعلي عبر core.engine.ExecutionEngine
        مباشرة، ليمرّرها لاحقاً إلى record_direct_execution أدناه لكشف انتقال
        حالة حقيقي (لا فقط: هل الخطوة فشلت الآن) — بدون هذه اللقطة، عقدة
        محجورة/فاشلة مسبقاً تعيد الفشل مرة أخرى ستُبَث كـ"node_failed" في
        كل مرة (تُغرِق القناة)، لأن الحالة النهائية بعد execute() تكون مطابقة
        لما قبله (FAILED→FAILED) وليست انتقالاً جديداً فعلياً."""
        return {n.node_id: n.state for n in self.registry.list_all()}

    def record_direct_execution(self, result, prev_states: Optional[Dict[str, str]] = None) -> None:
        """يأخذ core.engine.ExecutionResult حقيقياً من مسار /process المباشر
        (run_path/run_between/run_full_graph عبر ExecutionEngine) ويغذّي كل
        خطوة تنفيذ فعلية فيه إلى NodeReputationEngine + NodeChannel، تماماً
        كما تفعل record_swarm_result أعلاه لكل مهمة سرب.

        المشكلة التي يحلّها: بعد إصلاح /process (ربطها بـregistry/graph
        الحقيقية) ثم ربط AIDecisionLayer، صار /process ينفّذ عُقداً حقيقية
        فعلاً — node.execute() نفسها تُحدِّث BaseNode.state (مباشرة، لا عبر
        هذه الدالة). لكن reputation_engine (وبالتالي كل نظام الحجر/رفع
        الحجر التلقائي في _apply_reputation_feedback/_apply_reputation_recovery،
        وبثّ node_failed/node_recovered للجيران) لا يعرف عن أي طلب /process
        شيئاً على الإطلاق، لأن التغذية الوحيدة الموجودة (أعلاه في
        record_swarm_result) مصدرها حصرياً AgentFactory.run_task عبر مسار
        السرب — مسار منفصل تماماً عن core.engine.ExecutionEngine المستخدَم
        هنا. عملياً: عقدة تُستدعى مباشرة عبر /process وتفشل مئات المرات لن
        تُحجَر أبداً، وعقدة تتعافى بعد حجر لن يُرفَع عنها الحجر تلقائياً إن
        كان تعافيها ظاهراً فقط عبر /process لا عبر السرب.

        prev_states: لقطة من snapshot_node_states() قبل التنفيذ — لتمييز
        انتقال حالة حقيقي عن تكرار نفس الحالة (راجع توثيق تلك الدالة).

        نفس الملاحظة تنطبق أيضاً على ScoringEngine.record_run (يحدّث درجة
        كل حافة src→tgt في المسار الفعلي) وMemoryEngine.learn_from_run
        (ذاكرة مسار كامل + ذاكرة كل عقدة) — كلاهما مصمَّم أصلاً ليأخذ
        run_result بنفس الشكل الذي يُنتجه ExecutionResult.to_dict() بالضبط
        (نفس الحقول: path/steps/status/total_duration_ms)، وكانا يُستدعَيان
        فقط من record_swarm_result (سطر run_result اليدوي المُصنَّع هناك)،
        فلا يعرفان عن أي نتيجة /process شيئاً. لا أُقحِم هنا collective_memory
        (تتوقّع نص "task" وagent_id لا معنى مباشراً لهما خارج AgentFactory)
        ولا dna.snapshot (لقطة كاملة لكل الـregistry/scoring/memory — مكلفة
        لتشغيلها على كل طلب /process بلا داعٍ حقيقي على هذا المستوى من
        التفصيل) — كلاهما يستحق تقييماً منفصلاً إن لزم لاحقاً."""
        prev_states = prev_states or {}
        with self._lock:
            run_dict = result.to_dict() if hasattr(result, "to_dict") else None
            if run_dict:
                try:
                    self.scoring_engine.record_run(run_dict)
                    self.memory_engine.learn_from_run(run_dict)
                except Exception as e:
                    logger.warning(
                        "MeshBundle: scoring/memory learning after direct execution failed: %s", e
                    )

            for step in getattr(result, "steps", []):
                node_id = getattr(step, "node_id", None)
                status = getattr(step, "status", None)
                if not node_id or status not in ("success", "error"):
                    continue
                node = self.registry.get(node_id)
                if not node:
                    continue  # عقدة غير موجودة أصلاً — لا سمعة لها لتُسجَّل

                success = status == "success"
                name = getattr(step, "node_name", None) or node.name
                latency = float(getattr(step, "duration_ms", None) or 0.0)

                self.reputation_engine.record_execution(node_id, name, success, latency)

                prev_state = prev_states.get(node_id)
                if prev_state is not None and node.state != prev_state:
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
                                "role": name,
                                "previous_state": prev_state,
                                "current_state": node.state,
                                "error": getattr(step, "error", None) if not success else None,
                            },
                        )

            try:
                self._apply_reputation_feedback()
                self._apply_reputation_recovery()
                self._apply_node_retirement()
            except Exception as e:
                logger.warning(
                    "MeshBundle: reputation feedback/recovery after direct execution failed: %s", e
                )

    # ── دورة تطوّر ذاتي حقيقية (تُستدعى يدوياً، أو تلقائياً كل
    # EVOLUTION_CYCLE_INTERVAL نتيجة سرب من record_swarm_result أعلاه) ──────
    # تشغّل EvolutionEngine.run_cycle() الحقيقي: يفحص الرسم البياني الفعلي
    # (self.graph) بحثاً عن فجوات، يولّد عقداً جديدة، يمرّرها على
    # AIGovernanceLayer (حدود صارمة: لا حلقات، حد أقصى للتوليد، سمعة دنيا)،
    # ثم يسجّل المعتمَد منها فعلياً في self.registry عبر register_node أعلاه.
    def run_nodes_diagnose_cycle(self) -> dict:
        """تشخيص دوري خفيف لكل عُقد الأدوار: سمعة + وارد + حالة.
        يخزّن الملخص في _node_runtime_meta ويُطلق تنبيهات عند العتبات.
        لا يستدعي LLM — قراءة محلية فقط عبر اليد اليسرى."""
        from core.node_hands import LEFT
        from datetime import datetime, timezone as _tz

        now = datetime.now(_tz.utc).isoformat()
        nodes_out = []
        low_rep = []
        high_unread = []
        errors = 0

        role_ids = list((getattr(self, "role_node_ids", None) or {}).values())
        for nid in role_ids:
            node = self.registry.get(nid)
            if node is None or getattr(node, "hands", None) is None:
                continue
            entry = {"node_id": nid, "name": getattr(node, "name", None)}
            try:
                st = node.use_hand(LEFT, "node_status")
                if st.ok and isinstance(st.output, dict):
                    entry["state"] = st.output.get("state")
                    entry["reputation_score"] = float(st.output.get("reputation_score") or 0.0)
                    entry["last_terminal_check"] = st.output.get("last_terminal_check")
                else:
                    entry["status_error"] = getattr(st, "error", "status failed")
                    errors += 1
            except Exception as e:
                entry["status_error"] = str(e)
                errors += 1
            try:
                ib = node.use_hand(LEFT, "inbox_summary")
                if ib.ok and isinstance(ib.output, dict):
                    entry["unread_total"] = int(ib.output.get("unread_total") or 0)
                    entry["by_topic"] = ib.output.get("by_topic") or {}
                else:
                    entry["unread_total"] = 0
            except Exception:
                entry["unread_total"] = 0

            rep = float(entry.get("reputation_score") or 0.0)
            unread = int(entry.get("unread_total") or 0)
            # عتبة ديناميكية تُحسب بعد جمع كل الدرجات — تُطبَّق في ممر ثانٍ
            nodes_out.append(entry)

        scores = [float(e.get("reputation_score") or 0.0) for e in nodes_out]
        avg_rep = (sum(scores) / len(scores)) if scores else 0.0
        # نصف متوسط الأقران، بين 0.05 و 2× العتبة الثابتة
        dynamic_thr = avg_rep * 0.5 if scores else DIAGNOSE_LOW_REP_THRESHOLD
        effective_thr = max(0.05, min(DIAGNOSE_LOW_REP_THRESHOLD * 2.0, dynamic_thr))
        if not scores or avg_rep <= 0:
            effective_thr = DIAGNOSE_LOW_REP_THRESHOLD

        for entry in nodes_out:
            rep = float(entry.get("reputation_score") or 0.0)
            unread = int(entry.get("unread_total") or 0)
            entry["effective_rep_threshold"] = round(effective_thr, 4)
            if rep < effective_thr:
                low_rep.append({
                    "node_id": entry.get("node_id"),
                    "name": entry.get("name"),
                    "reputation": rep,
                    "threshold": round(effective_thr, 4),
                })
            if unread >= DIAGNOSE_HIGH_UNREAD_THRESHOLD:
                high_unread.append({
                    "node_id": entry.get("node_id"),
                    "name": entry.get("name"),
                    "unread": unread,
                })

        # إزالة المعافين: من كانوا في low_reputation السابق وتعافوا الآن
        prev = {}
        try:
            with self._lock:
                prev = dict(self._node_runtime_meta.get("__mesh_diagnose_summary__") or {})
        except Exception:
            prev = {}
        prev_low = {
            x.get("node_id")
            for x in (prev.get("low_reputation") or [])
            if isinstance(x, dict) and x.get("node_id")
        }
        current_low_ids = {x.get("node_id") for x in low_rep if x.get("node_id")}
        recovered = []
        for entry in nodes_out:
            nid = entry.get("node_id")
            if not nid or nid not in prev_low or nid in current_low_ids:
                continue
            rep = float(entry.get("reputation_score") or 0.0)
            recovered.append({
                "node_id": nid,
                "name": entry.get("name"),
                "reputation": rep,
                "threshold": round(effective_thr, 4),
                "status": "recovered",
            })
        # low_reputation الحالية لا تتضمن المعافين (أُعيد بناؤها من الصفر أعلاه)

        summary = {
            "ts": now,
            "layer": "mesh-nodes-diagnose-cycle-v2",
            "scanned": len(nodes_out),
            "errors": errors,
            "avg_reputation": round(avg_rep, 4),
            "effective_low_rep_threshold": round(effective_thr, 4),
            "static_low_rep_threshold": DIAGNOSE_LOW_REP_THRESHOLD,
            "low_reputation": low_rep,
            "recovered": recovered,
            "high_unread": high_unread,
            "nodes": nodes_out,
        }
        with self._lock:
            self._node_runtime_meta["__mesh_diagnose_summary__"] = summary

        # 🆕 سجل تاريخي SQLite (penalized / recovered عبر الزمن)
        try:
            from ai.nodes_diagnose_store import NodesDiagnoseStore
            db = Path(self.storage.storage_dir) / "nodes_diagnose_history.db"
            store = NodesDiagnoseStore(db_path=db)
            store.log_cycle(summary)
            try:
                store.prune_old(keep_last=500)
            except Exception:
                pass
            # تنبيه إن ارتفع low_rep فوق المتوسط التاريخي
            try:
                low_n = len(summary.get("low_reputation") or [])
                spike = store.spike_vs_average(low_n, lookback=20)
                summary["low_rep_spike"] = spike
                with self._lock:
                    self._node_runtime_meta["__mesh_diagnose_summary__"] = summary
                if spike.get("spike"):
                    from ai.alert_manager import alert_manager
                    alert_manager.send_alert(
                        "WARNING",
                        f"ارتفاع low_rep_count={spike.get('current')} فوق المتوسط {spike.get('avg')}",
                        details=spike,
                        throttle_sec=600,
                    )
            except Exception as e:
                logger.debug("low_rep spike check skipped: %s", e)
        except Exception as e:
            logger.debug("nodes diagnose history log skipped: %s", e)

        # 🆕 بث ملخص التشخيص للأقران عبر القناة (topic=mesh_diagnose)
        try:
            payload = {
                "ts": now,
                "scanned": len(nodes_out),
                "low_reputation_count": len(low_rep),
                "high_unread_count": len(high_unread),
                "errors": errors,
                "low_reputation": low_rep[:5],
                "high_unread": high_unread[:5],
            }
            to_ids = []
            for nid in role_ids:
                if self.registry.exists(nid):
                    to_ids.append(nid)
            # من جذر الشبكة أو أول عقدة دور كمرسل منطقي
            from_id = getattr(self, "_root_node_id", None) or (to_ids[0] if to_ids else None)
            if from_id and to_ids:
                self.channel.broadcast(
                    from_id=from_id,
                    to_ids=[i for i in to_ids if i != from_id],
                    topic="mesh_diagnose",
                    payload=payload,
                )
        except Exception as e:
            logger.debug("mesh_diagnose broadcast skipped: %s", e)

        # تنبيهات (مع كبح داخل AlertManager)
        try:
            from ai.alert_manager import alert_manager
            if low_rep:
                alert_manager.send_alert(
                    "WARNING",
                    f"عُقد بسمعة منخفضة ({len(low_rep)})",
                    details={"threshold": DIAGNOSE_LOW_REP_THRESHOLD, "nodes": low_rep[:10]},
                    throttle_sec=300,
                )
            if high_unread:
                alert_manager.send_alert(
                    "WARNING",
                    f"عُقد بوارد مرتفع غير مقروء ({len(high_unread)})",
                    details={"threshold": DIAGNOSE_HIGH_UNREAD_THRESHOLD, "nodes": high_unread[:10]},
                    throttle_sec=300,
                )
        except Exception as e:
            logger.debug("diagnose alerts skipped: %s", e)

        return {
            "scanned": len(nodes_out),
            "low_reputation": len(low_rep),
            "high_unread": len(high_unread),
            "errors": errors,
            "ts": now,
        }

    def get_nodes_diagnose_summary(self) -> dict:
        """آخر ملخص تشخيص دوري للعُقد (إن وُجد)."""
        with self._lock:
            return dict(self._node_runtime_meta.get("__mesh_diagnose_summary__") or {})

    def get_nodes_diagnose_history(self, limit: int = 30) -> dict:
        """اتجاه تاريخي لعدد المعاقَبين/المعافين من SQLite."""
        try:
            from ai.nodes_diagnose_store import NodesDiagnoseStore
            db = Path(self.storage.storage_dir) / "nodes_diagnose_history.db"
            store = NodesDiagnoseStore(db_path=db)
            return {
                "trend": store.trend(limit=limit),
                "summary": store.summary(),
                "recent": store.get_recent(limit=min(limit, 20)),
                "csv": store.export_csv(limit=max(limit, 50)),
            }
        except Exception as e:
            return {"trend": {"points": 0}, "summary": {}, "recent": [], "csv": "", "error": str(e)}

    def _auto_nodes_diagnose_loop(self) -> None:
        """خيط daemon: تشخيص العُقد كل DIAGNOSE_INTERVAL_S بلا اعتماد على السرب."""
        # تأخير أولي قصير حتى يكتمل الإقلاع
        stop = getattr(self, "_diagnose_stop", None)
        if stop is not None and stop.wait(15):
            return
        while True:
            try:
                self.run_nodes_diagnose_cycle()
            except Exception as e:
                logger.warning("MeshBundle: background nodes diagnose failed: %s", e)
            try:
                self._maybe_send_diagnose_weekly_summary()
            except Exception as e:
                logger.debug("weekly diagnose summary skipped: %s", e)
            if stop is not None and stop.wait(max(30, int(DIAGNOSE_INTERVAL_S))):
                return

    def _maybe_send_diagnose_weekly_summary(self) -> None:
        """يرسل ملخصاً أسبوعياً عبر alert_manager مرة كل ~7 أيام."""
        import time as _time
        key = "__diagnose_weekly_last_ts__"
        now = _time.time()
        with self._lock:
            last = float(self._node_runtime_meta.get(key) or 0.0)
            if now - last < 7 * 24 * 3600:
                return
            self._node_runtime_meta[key] = now
        try:
            from ai.nodes_diagnose_store import NodesDiagnoseStore
            db = Path(self.storage.storage_dir) / "nodes_diagnose_history.db"
            report = NodesDiagnoseStore(db_path=db).weekly_report(days=7)
        except Exception as e:
            logger.debug("weekly_report failed: %s", e)
            return
        if not report.get("cycles"):
            return
        try:
            from ai.alert_manager import alert_manager
            alert_manager.send_alert(
                "INFO",
                (
                    f"ملخص تشخيص أسبوعي: cycles={report.get('cycles')} "
                    f"avg_low_rep={report.get('avg_low_rep')} "
                    f"max_low_rep={report.get('max_low_rep')} "
                    f"recovered={report.get('total_recovered')} "
                    f"spikes≈{report.get('approx_spike_events')}"
                ),
                details=report,
                throttle_sec=6 * 24 * 3600,
            )
        except Exception as e:
            logger.debug("weekly alert failed: %s", e)

    def run_evolution_cycle(self) -> dict:
        with self._lock:
            cycle = self.evolution.run_cycle(auto_register=True, verbose=False)
            self._apply_reputation_feedback()
            self._apply_reputation_recovery()
            self._apply_node_retirement()
            try:
                # تحليل فقط (analyze())، لا تطبيق (apply_report()) — انظر
                # تعليق تسليك OptimizationEngine في __init__ أعلاه للسبب.
                opt_report = self.optimization_engine.analyze()
                if opt_report.actions:
                    logger.info(
                        "MeshBundle: OptimizationEngine اقترح %d إجراء (%s)",
                        len(opt_report.actions), opt_report.summary.get("action_counts"),
                    )
            except Exception as e:
                logger.warning("MeshBundle: تحليل OptimizationEngine فشل: %s", e)
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

    # ── تقاعد دائم لعُقد ذاتية التوليد فشلت بشكل مزمن ─────────────────────
    # الحجر/رفع الحجر (أعلاه) يغطيان حالتين: عقدة سيئة مؤقتاً (تُحجَر ثم
    # تتعافى) — لكن عقدة ai-generated (service_generator.py) قد تبقى
    # محجورة للأبد دون أي تعافٍ حقيقي (خطأ بنيوي في القالب المولَّد، ليس
    # عطلاً عابراً) وتبقى في الـregistry/الرسم البياني إلى ما لا نهاية،
    # مجرّد بيانات ميتة لا تُنفَّذ أبداً (routing يستبعدها أصلاً بسبب
    # الحجر). هذه الدالة تتقاعد فعلياً (unregister + إزالة من الرسم
    # البياني + حذف من exec_log الدائم) أي عقدة محجورة منذ فترة طويلة
    # (quarantine_checks) دون تعافٍ — بشرط أساسي وغير قابل للتنازل: لا
    # تُتقاعَد أبداً عقدة لا تحمل وسم 'ai-generated' (أي عُقد الكتالوج
    # الأساسية agent_role/mcp_tool تبقى مهما ساءت سمعتها، لأنها ستُعاد
    # تسجيلها من AGENT_CATALOGUE عند إعادة التشغيل على أي حال، وتقاعدها
    # يعني فقط تعطيلها المؤقت حتى إعادة تشغيل تالية بلا أي فائدة حقيقية).
    def _apply_node_retirement(self, min_quarantine_checks: int = 15) -> None:
        self.reputation_engine.tick_quarantine_checks()
        for rep in self.reputation_engine.retirement_eligible_nodes(min_quarantine_checks):
            node_id = rep.get("node_id")
            if not node_id:
                continue
            node = self.registry.get(node_id)
            if node is None:
                continue
            if "ai-generated" not in (node.tags or []):
                # حماية: لا تُطبَّق آلية التقاعد إلا على عُقد ai-generated —
                # عُقد الكتالوج الأساسية تبقى محجورة (لا تُنفَّذ) لكن لا تُحذف.
                continue

            predecessors = self.graph.get_predecessors(node_id) if self.graph.has_node(node_id) else []
            neighbors = (
                predecessors + self.graph.get_neighbors(node_id)
                if self.graph.has_node(node_id) else []
            )
            self.channel.broadcast(
                from_id=node_id,
                to_ids=[n for n in set(neighbors) if n and n != node_id],
                topic="node_retired",
                payload={
                    "reason": "chronic_quarantine",
                    "quarantine_checks": rep.get("quarantine_checks"),
                },
            )

            self.registry.unregister(node_id)
            if self.graph.has_node(node_id):
                self.graph.remove_node(node_id)
            try:
                self.exec_log.delete_node(node_id)
                for pred in predecessors:
                    self.exec_log.delete_connection(pred, node_id)
            except Exception as e:
                logger.warning("MeshBundle: تعذّر تنظيف exec_log بعد تقاعد العقدة %s: %s",
                                node_id[:8], e)

            logger.info(
                "MeshBundle: تقاعد العقدة %s '%s' نهائياً بعد %s فحص سمعة محجورة دون تعافٍ",
                node_id[:8], node.name, rep.get("quarantine_checks"),
            )

    # ── تخطيط وتنفيذ هدف مركّب متعدد الخطوات عبر MultiGoalPlanner (Phase 5) ──
    # نقطة الاستخدام الفعلية الوحيدة لـself.multi_goal_planner أعلاه: تفكّك
    # الهدف إلى أهداف فرعية، تحلّ كل واحد إلى عقدة فعلية عبر المتجر، ثم
    # تنفّذ المسار المركَّب بالكامل عبر ExecutionEngine حقيقي (نفس registry/
    # graph/exec_log/ai_decision المشتركة). كل خطوة نُفِّذت فعلياً تُغذّي
    # نتيجتها (نجاح/فشل + الكمون الحقيقي) مرة أخرى إلى
    # CapabilityMarketplace.record_execution — بذلك تتحسن قرارات
    # best_provider() المستقبلية فعلياً من تجربة حقيقية، لا أن تبقى ثابتة
    # عند القيمة الابتدائية 0.8 للأبد.
    def plan_and_execute_goal(self, goal: str, data: Optional[Dict[str, Any]] = None) -> dict:
        with self._lock:
            plan = self.multi_goal_planner.plan(goal)
            if not plan.resolved_path:
                return {
                    "status": "failed",
                    "error": "لا توجد عقدة تفي بأي من القدرات المطلوبة لهذا الهدف",
                    "plan": plan.to_dict(),
                }

            engine = ExecutionEngine(
                self.registry, self.graph, self.storage,
                db=self.exec_log, ai=self.ai_decision,
            )

            if len(plan.resolved_path) >= 2:
                result_dict = self.multi_goal_planner.execute_plan(plan, engine, data or {})
            else:
                # execute_plan() ترفض مسار عقدة واحدة فقط (تتطلب >=2)، لكن
                # هدف بسيط يُحلّ إلى عقدة وحيدة يظل قابلاً للتنفيذ الحقيقي.
                exec_result = engine.run_path(plan.resolved_path, data or {})
                result_dict = exec_result.to_dict()
                plan.status = result_dict.get("status", "completed")
                # نفس إصلاح المرجع الدائري في MultiGoalPlanner.execute_plan
                # أعلاه بالضبط: التقط النسخة قبل تعيين plan.result كي لا
                # تحتوي result_dict["multi_goal_plan"]["result"] على
                # result_dict نفسه.
                plan_snapshot = plan.to_dict()
                plan.result = result_dict
                result_dict["multi_goal_plan"] = plan_snapshot

            sg_capability_by_node = {
                sg.resolved_node_id: sg.capability
                for sg in plan.sub_goals if sg.resolved_node_id
            }
            for step in result_dict.get("steps", []):
                capability = sg_capability_by_node.get(step.get("node_id"))
                if not capability:
                    continue
                try:
                    self.marketplace.record_execution(
                        node_id=step["node_id"],
                        capability=capability,
                        success=(step.get("status") == "success"),
                        latency_ms=step.get("duration_ms") or 0.0,
                    )
                except Exception as e:
                    logger.warning("MeshBundle: تعذّر تحديث المتجر بعد التنفيذ: %s", e)

            return result_dict

    def summary(self) -> dict:
        _cm_summary = {}
        try:
            from ai.collective_memory import get_collective_memory
            _cm_summary = get_collective_memory().summary()
        except Exception:
            pass
        diag = self.get_nodes_diagnose_summary()
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
            "marketplace": self.marketplace.summary(),
            "multi_goal_planner": self.multi_goal_planner.summary(),
            "nodes_diagnose": {
                "ts": diag.get("ts"),
                "scanned": diag.get("scanned", 0),
                "low_reputation": len(diag.get("low_reputation") or []),
                "high_unread": len(diag.get("high_unread") or []),
                "errors": diag.get("errors", 0),
            } if diag else {},
        }


@lru_cache(maxsize=1)
def get_mesh_bundle() -> MeshBundle:
    """Singleton حقيقي على مستوى العملية — يبقى حياً بين كل sessions/reruns."""
    return MeshBundle()
