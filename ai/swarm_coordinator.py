"""
Phase 6 – Swarm Coordinator
=============================
Orchestrates a swarm of autonomous agents to execute tasks in parallel.

Workflow:
  1. Receive a high-level task
  2. Decompose into sub-tasks
  3. Assign each sub-task to the best available agent
  4. Monitor execution
  5. Merge and validate results

Example:
  coordinator = SwarmCoordinator(factory, max_agents=20)
  result = coordinator.execute("translate and review document", data={...})
"""
from __future__ import annotations

import json
import re
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable

from ai.agent_factory import AgentFactory, AgentInstance, AGENT_CATALOGUE

logger = logging.getLogger(__name__)


class SwarmTask:
    """A single unit of work assigned to one agent."""

    def __init__(
        self,
        task_id: str,
        sub_goal: str,
        required_capability: str,
        data: dict,
        priority: int = 5,
    ):
        self.task_id = task_id
        self.sub_goal = sub_goal
        self.required_capability = required_capability
        self.data = data
        self.priority = priority          # 1 (highest) … 10 (lowest)
        self.assigned_agent_id: Optional[str] = None
        self.status = "pending"           # pending / running / done / failed
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.duration_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "sub_goal": self.sub_goal,
            "required_capability": self.required_capability,
            "priority": self.priority,
            "assigned_agent_id": self.assigned_agent_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
        }


class SwarmResult:
    """Aggregated result from a swarm execution run."""

    def __init__(self, swarm_id: str, goal: str):
        self.swarm_id = swarm_id
        self.goal = goal
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.finished_at: Optional[str] = None
        self.tasks: List[SwarmTask] = []
        self.merged_output: Optional[dict] = None
        self.status = "running"

    @property
    def success_count(self):
        return sum(1 for t in self.tasks if t.status == "done")

    @property
    def failed_count(self):
        return sum(1 for t in self.tasks if t.status == "failed")

    def complete(self, merged: dict):
        self.merged_output = merged
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.status = "done" if self.failed_count == 0 else "partial"

    def to_dict(self) -> dict:
        return {
            "swarm_id": self.swarm_id,
            "goal": self.goal,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_tasks": len(self.tasks),
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "merged_output": self.merged_output,
            "tasks": [t.to_dict() for t in self.tasks],
        }


class SwarmCoordinator:
    """
    Phase 6: Multi-Agent Swarm Coordinator.

    Decomposes goals into parallel tasks, assigns agents, and merges results.
    Supports up to `max_agents` concurrent workers.
    """

    # Default task decomposition rules: goal keyword → list of (sub_goal, capability)
    DECOMPOSITION_RULES: Dict[str, List[tuple]] = {
        "translate": [
            ("translate content", "translate"),
            ("validate translation", "validate"),
        ],
        "research": [
            ("search for information", "search"),
            ("summarize findings", "summarize"),
            ("fact-check results", "fact_check"),
        ],
        "review": [
            ("audit artifact", "audit"),
            ("score quality", "score"),
        ],
        "plan": [
            ("decompose goal into tasks", "plan"),
            ("schedule task execution", "schedule"),
        ],
        "optimize": [
            ("benchmark current performance", "benchmark"),
            ("identify improvements", "optimize"),
            ("apply tuning", "tune"),
        ],
        "monitor": [
            ("collect health metrics", "monitor"),
            ("generate alerts if needed", "alert"),
        ],
    }

    # 🆕 مرادفات عربية لكل مفتاح تفكيك — القواعد أعلاه (DECOMPOSITION_RULES)
    # كانت تُطابَق بكلماتها الإنجليزية فقط (translate/research/review/...)
    # رغم أن NSM تطبيق عربي بالكامل، فكانت هذه الخطة الاحتياطية عملياً
    # غير قابلة للتفعيل من مستخدم يكتب هدفه بالعربية (المسار الوحيد الذي
    # يعمل فعلياً هو _decompose_via_planner، وإن فشل — كان يسقط مباشرة
    # للمهمة العامة الواحدة). الآن تُطابَق الكلمتان معاً.
    DECOMPOSITION_KEYWORDS: Dict[str, List[str]] = {
        "translate": ["translate", "ترجم", "ترجمة", "ترجمها"],
        "research":  ["research", "ابحث", "بحث", "البحث", "استقص", "تقصّي"],
        "review":    ["review", "راجع", "مراجعة", "دقق", "تدقيق", "تحقق"],
        "plan":      ["plan", "خطط", "خطة", "تخطيط", "فكّك", "فكك", "جدول"],
        "optimize":  ["optimize", "حسّن", "حسن", "تحسين", "سرّع", "تسريع"],
        "monitor":   ["monitor", "راقب", "مراقبة", "رصد", "تتبع", "تتبّع"],
    }

    def __init__(
        self,
        factory: AgentFactory,
        max_agents: int = 20,
        knowledge_store=None,
    ):
        self._factory = factory
        self._max_agents = max_agents
        self._knowledge = knowledge_store
        self._history: List[SwarmResult] = []
        self._lock = threading.Lock()
        # 🆕 تخزين دائم لنتائج السرب (SQLite) — self._history وحدها كانت
        # في الذاكرة فقط وتُمسح بإعادة تشغيل الحاوية. انظر ai/swarm_history_store.py
        try:
            from ai.swarm_history_store import get_default_swarm_store
            self._store = get_default_swarm_store()
        except Exception as exc:
            logger.warning(f"تعذّر تفعيل تخزين تاريخ السرب الدائم: {exc}")
            self._store = None
        logger.info(f"SwarmCoordinator initialised (max_agents={max_agents})")

    # ── Main execution ────────────────────────────────────────────────────

    def execute(
        self,
        goal: str,
        data: dict,
        custom_tasks: Optional[List[dict]] = None,
        use_planner: bool = True,
        retry_failed: bool = True,
        synthesize: bool = False,
    ) -> SwarmResult:
        """
        Execute a goal using the swarm.

        Args:
            goal:         High-level goal string.
            data:         Input data passed to each sub-task.
            custom_tasks: Optional manual task list
                          [{"sub_goal": ..., "capability": ..., "priority": ...}]
            use_planner:  🆕 إذا True (الافتراضي)، يُستخدم PlanningAgent حقيقي
                          لتفكيك الهدف ديناميكياً قبل اللجوء لقواعد الكلمات
                          المفتاحية الثابتة (DECOMPOSITION_RULES) كخطة احتياطية.
            retry_failed: 🆕 إذا True (الافتراضي)، تُعاد محاولة كل مهمة فرعية
                          فشلت مرة واحدة إضافية عبر وكيل جديد يُنشأ خصيصاً لنفس
                          القدرة المطلوبة (بدل الاكتفاء بفشل الوكيل الأول الذي
                          قد يكون تعطّل لسبب عابر — مزوّد LLM بطيء، تحميل أول
                          مرة، إلخ). لا تُعاد محاولة مهمة نجحت من أول مرة.
            synthesize:   🆕 إذا True، يُولَّف كل نتائج المهام الناجحة في
                          إجابة نهائية واحدة موحّدة عبر LLM (بنفس أسلوب
                          "🤝 منسّق الوكلاء")، وتُخزَّن في
                          result.merged_output["synthesis"]. False افتراضياً
                          حفاظاً على التوافق الخلفي (سلوك سابق دون توليف).
        """
        swarm_id = f"swarm_{str(uuid.uuid4())[:8]}"
        result = SwarmResult(swarm_id, goal)

        # 1. Decompose goal into tasks
        tasks = self._decompose(goal, data, custom_tasks, use_planner=use_planner)
        result.tasks = tasks

        if not tasks:
            result.merged_output = {"error": "Could not decompose goal into tasks"}
            result.status = "failed"
            result.finished_at = datetime.now(timezone.utc).isoformat()
            self._history.append(result)
            self._persist_result(result)
            return result

        # 2. Sort by priority
        tasks.sort(key=lambda t: t.priority)

        # 3. Execute in parallel (bounded by max_agents)
        workers = min(len(tasks), self._max_agents)
        task_outputs: Dict[str, dict] = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_task: Dict[Future, SwarmTask] = {}
            for task in tasks:
                agent = self._factory.best_agent_for(task.required_capability)
                if agent is None:
                    # Auto-spawn an agent for this capability
                    agent = self._auto_spawn_for_capability(task.required_capability)
                if agent:
                    task.assigned_agent_id = agent.agent_id
                    fut = pool.submit(self._run_task, task, agent)
                    future_to_task[fut] = task
                else:
                    task.status = "failed"
                    task.error = f"No agent available for capability '{task.required_capability}'"

            for fut in as_completed(future_to_task):
                task = future_to_task[fut]
                try:
                    output = fut.result()
                    task.result = output
                    # 🆕 الحالة تُبنى على success الفعلي من agent.execute()،
                    # وليس فقط على عدم وجود استثناء (لأن فشل التنفيذ —
                    # مثل غياب مفتاح API — يعود كنتيجة عادية بدون استثناء).
                    if output.get("success", True):
                        task.status = "done"
                    else:
                        task.status = "failed"
                        task.error = output.get("result_text") or "فشل تنفيذ المهمة"
                    task_outputs[task.task_id] = output
                except Exception as exc:
                    task.status = "failed"
                    task.error = str(exc)
                    logger.error(f"Task {task.task_id} failed: {exc}")

        # 4. 🆕 إعادة محاولة المهام الفاشلة مرة واحدة عبر وكيل جديد لنفس
        #    القدرة، قبل التوليف والإنهاء (فشل عابر لا يعني أن المهمة غير
        #    قابلة للتنفيذ إطلاقاً).
        if retry_failed:
            self._retry_failed_tasks(tasks, task_outputs)

        # 5. Merge results (بعد إعادة المحاولة، حتى تعكس الحالة النهائية)
        merged = self._merge_results(goal, tasks, task_outputs)

        # 6. 🆕 توليف اختياري لكل النتائج الناجحة في إجابة واحدة موحّدة
        if synthesize:
            merged["synthesis"] = self._synthesize(goal, tasks, task_outputs)

        result.complete(merged)

        with self._lock:
            self._history.append(result)
        self._persist_result(result)

        logger.info(
            f"Swarm {swarm_id} finished: "
            f"{result.success_count}/{len(tasks)} tasks succeeded"
        )
        return result

    # ── Internals ─────────────────────────────────────────────────────────

    def _decompose(
        self,
        goal: str,
        data: dict,
        custom_tasks: Optional[List[dict]],
        use_planner: bool = True,
    ) -> List[SwarmTask]:
        if custom_tasks:
            return [
                SwarmTask(
                    task_id=f"task_{i}_{str(uuid.uuid4())[:6]}",
                    sub_goal=t["sub_goal"],
                    required_capability=t.get("capability", "search"),
                    data=data,
                    priority=t.get("priority", 5),
                )
                for i, t in enumerate(custom_tasks)
            ]

        # 🆕 المحاولة الأولى: تفكيك ديناميكي حقيقي عبر PlanningAgent، بدل
        # الاعتماد فوراً على قواعد الكلمات المفتاحية الثابتة. يعمل مع أي
        # هدف مهما كانت صياغته، وليس فقط الأهداف التي تحتوي كلمة مفتاحية
        # معروفة مسبقاً (translate/research/review/...).
        if use_planner:
            planned = self._decompose_via_planner(goal, data)
            if planned:
                logger.info(f"PlanningAgent decomposed goal into {len(planned)} tasks")
                return planned

        goal_lower = goal.lower()
        for canonical_key, synonyms in self.DECOMPOSITION_KEYWORDS.items():
            if any(kw in goal_lower for kw in synonyms):
                sub_specs = self.DECOMPOSITION_RULES[canonical_key]
                return [
                    SwarmTask(
                        task_id=f"task_{i}_{str(uuid.uuid4())[:6]}",
                        sub_goal=sub_goal,
                        required_capability=cap,
                        data=data,
                        priority=i + 1,
                    )
                    for i, (sub_goal, cap) in enumerate(sub_specs)
                ]

        # Fallback: single generic task
        return [
            SwarmTask(
                task_id=f"task_0_{str(uuid.uuid4())[:6]}",
                sub_goal=goal,
                required_capability="search",
                data=data,
                priority=5,
            )
        ]

    def _decompose_via_planner(self, goal: str, data: dict) -> Optional[List[SwarmTask]]:
        """
        🆕 يستخدم PlanningAgent (محرك NSMAgent حقيقي) لتفكيك هدف معقد إلى
        مهام فرعية ديناميكياً حسب محتوى الهدف الفعلي، بدل قواعد الكلمات
        المفتاحية الثابتة. يُعيد None إذا فشل التخطيط (مثلاً: لا مفتاح
        API) أو تعذّر تحليل الرد كـ JSON صالح — عندها _decompose() يرجع
        تلقائياً لقواعد الكلمات المفتاحية كخطة احتياطية آمنة.
        """
        known_capabilities = sorted({
            cap for spec in AGENT_CATALOGUE.values() for cap in spec.get("capabilities", [])
        })
        try:
            planner = self._factory.spawn("PlanningAgent")
        except Exception as exc:
            logger.warning(f"تعذّر إنشاء PlanningAgent: {exc}")
            return None

        prompt = (
            f'فكّك هذا الهدف إلى مهام فرعية منفذة قابلة للتوزيع على وكلاء متخصصين: "{goal}"\n\n'
            f"القدرات المتاحة فقط (اختر لكل مهمة قدرة واحدة من هذه القائمة حصراً): "
            f"{', '.join(known_capabilities)}\n\n"
            "أجب بصيغة JSON فقط، مصفوفة من كائنات بهذا الشكل بالضبط، بدون أي "
            "نص خارج المصفوفة:\n"
            '[{"sub_goal": "وصف مختصر للمهمة الفرعية", "capability": "إحدى القدرات أعلاه", "priority": 1}]'
        )

        try:
            exec_result = planner.execute(prompt)
        except Exception as exc:
            logger.warning(f"PlanningAgent decomposition raised: {exc}")
            return None

        if not exec_result.get("success"):
            return None

        raw = exec_result.get("result", "")
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return None

        tasks: List[SwarmTask] = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue
            sub_goal = str(item.get("sub_goal") or "").strip()
            capability = str(item.get("capability") or "").strip()
            if not sub_goal or capability not in known_capabilities:
                continue
            try:
                priority = int(item.get("priority", i + 1))
            except (TypeError, ValueError):
                priority = i + 1
            tasks.append(SwarmTask(
                task_id=f"task_{i}_{str(uuid.uuid4())[:6]}",
                sub_goal=sub_goal,
                required_capability=capability,
                data=data,
                priority=priority,
            ))

        return tasks or None

    def _run_task(self, task: SwarmTask, agent: AgentInstance) -> dict:
        import time
        task.status = "running"
        task.started_at = datetime.now(timezone.utc).isoformat()
        t0 = time.time()

        try:
            # 🆕 تنفيذ حقيقي عبر محرك الوكيل (NSMAgent)، بدل المحاكاة القديمة.
            task_text = self._build_task_text(task)
            exec_result = agent.execute(task_text)
            output = {
                "task_id": task.task_id,
                "sub_goal": task.sub_goal,
                "agent_id": agent.agent_id,
                "agent_role": agent.role,
                "capability_used": task.required_capability,
                "status": "completed" if exec_result.get("success") else "failed",
                "success": exec_result.get("success", False),
                "result_text": exec_result.get("result", ""),
                "data_keys_processed": list(task.data.keys()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            # 🆕 ملاحظة: agent.execute() يسجّل نجاح/فشل المهمة داخلياً بالفعل
            # (agent.record_task)، فلا نكرر التسجيل هنا لتفادي الازدواجية.
            return output
        except (NotImplementedError, RuntimeError) as exc:
            # دور بلا محرك تنفيذ حقيقي، أو وكيل غير نشط — لم يُسجَّل داخل
            # execute() لأنه رُفع قبل استدعاء المحرك، فنسجّله هنا يدوياً.
            agent.record_task(success=False)
            raise exc
        finally:
            task.finished_at = datetime.now(timezone.utc).isoformat()
            task.duration_ms = round((time.time() - t0) * 1000, 2)

    def _build_task_text(self, task: SwarmTask) -> str:
        """
        🆕 يبني نص المهمة الفعلي المُرسل لمحرك الوكيل، بدمج الهدف الفرعي
        مع أي بيانات نصية ذات صلة موجودة في task.data (مثل content/text/
        query/goal/target)، حتى لا يستقبل المحرك جملة عامة قصيرة فقط
        (مثل "translate content") بدون السياق الفعلي المطلوب تنفيذه عليه.
        """
        parts = [task.sub_goal]
        if isinstance(task.data, dict):
            for key in ("content", "text", "query", "goal", "artifact", "target", "path"):
                val = task.data.get(key)
                if val:
                    parts.append(f"{key}: {val}")
        return "\n".join(str(p) for p in parts)

    def _auto_spawn_for_capability(self, capability: str) -> Optional[AgentInstance]:
        """Find a role that has the required capability and spawn an agent."""
        from ai.agent_factory import AGENT_CATALOGUE
        for role, spec in AGENT_CATALOGUE.items():
            if capability in spec.get("capabilities", []):
                try:
                    agent = self._factory.spawn(role)
                    logger.info(f"Auto-spawned {role} for capability '{capability}'")
                    return agent
                except Exception:
                    pass
        return None

    def _retry_failed_tasks(self, tasks: List[SwarmTask], task_outputs: Dict[str, dict]) -> None:
        """🆕 يعيد محاولة كل مهمة بحالة 'failed' مرة واحدة فقط، عبر وكيل
        جديد يُنشأ خصيصاً لنفس القدرة المطلوبة (وليس نفس الوكيل الذي فشل).
        لا يرفع أي استثناء للخارج — أي فشل في إعادة المحاولة نفسها يُسجَّل
        في task.error ويُترَك status = 'failed' كما هو."""
        for task in tasks:
            if task.status != "failed":
                continue
            agent = self._auto_spawn_for_capability(task.required_capability)
            if agent is None:
                continue
            task.assigned_agent_id = agent.agent_id
            try:
                output = self._run_task(task, agent)
                task.result = output
                task_outputs[task.task_id] = output
                if output.get("success", True):
                    task.status = "done"
                    task.error = None
                else:
                    task.status = "failed"
                    task.error = (
                        output.get("result_text") or "فشلت المهمة بعد إعادة المحاولة أيضاً"
                    )
            except Exception as exc:
                task.status = "failed"
                task.error = f"فشلت إعادة المحاولة أيضاً: {exc}"
                logger.error(f"Retry for task {task.task_id} failed: {exc}")

    def _synthesize(
        self, goal: str, tasks: List[SwarmTask], task_outputs: Dict[str, dict]
    ) -> Optional[str]:
        """🆕 يولّف نتائج المهام الناجحة في إجابة نهائية واحدة موحّدة، بنفس
        أسلوب توليف "🤝 منسّق الوكلاء" (COORDINATOR_SYSTEM_PROMPT من
        ai.godmode). يعيد None بأمان عند عدم وجود نتائج ناجحة أو عند فشل
        استدعاء LLM — لا يُسقِط تنفيذ السرب بالكامل."""
        successful = [o for o in task_outputs.values() if o.get("success")]
        if not successful:
            return None
        combined = "\n\n".join(
            f"[{o.get('sub_goal', '')}]\n{o.get('result_text', '')}" for o in successful
        )
        try:
            from ai.llm_fallback import LLMFallback
            from ai.godmode import COORDINATOR_SYSTEM_PROMPT
            llm = LLMFallback()
            res = llm.generate(
                query=f"الهدف الأصلي: {goal}\n\nنتائج المهام الفرعية:\n{combined}",
                system_prompt=COORDINATOR_SYSTEM_PROMPT,
            )
            return res.text
        except Exception as exc:
            logger.warning(f"تعذّر توليف نتائج السرب: {exc}")
            return None

    def _merge_results(
        self,
        goal: str,
        tasks: List[SwarmTask],
        outputs: Dict[str, dict],
    ) -> dict:
        return {
            "goal": goal,
            "tasks_completed": len(outputs),
            "tasks_failed": sum(1 for t in tasks if t.status == "failed"),
            "outputs": list(outputs.values()),
            "merged_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Public helpers ────────────────────────────────────────────────────

    def _persist_result(self, result: "SwarmResult") -> None:
        """يحفظ نتيجة swarm في SQLite (memory/swarm_history.db) كي تبقى
        بعد إعادة تشغيل الحاوية. لا يرفع استثناء أبداً — التخزين الدائم
        لا يجب أن يُعطّل تنفيذ السرب."""
        if self._store is None:
            return
        try:
            self._store.log_result(result.to_dict())
        except Exception as exc:
            logger.warning(f"تعذّر حفظ نتيجة السرب {result.swarm_id} بشكل دائم: {exc}")

    def history(self, limit: int = 20) -> List[dict]:
        """يُفضّل التاريخ الدائم (SQLite) إن كان متاحاً — يشمل عمليات
        السرب من قبل إعادة تشغيل الحاوية الأخيرة، لا فقط الجلسة الحالية."""
        if self._store is not None:
            try:
                persisted = self._store.get_recent(limit)
                if persisted:
                    return persisted
            except Exception as exc:
                logger.warning(f"تعذّر قراءة تاريخ السرب الدائم: {exc}")
        return [r.to_dict() for r in self._history[-limit:]]

    def summary(self) -> dict:
        if self._store is not None:
            try:
                base = self._store.summary()
                return {
                    "total_swarms": base["total_swarms"],
                    "done": base["done"],
                    "partial": base["partial"],
                    "failed": base["failed"],
                    "max_agents": self._max_agents,
                    "active_agents": len(self._factory.list_active()),
                }
            except Exception as exc:
                logger.warning(f"تعذّر قراءة ملخص السرب الدائم: {exc}")

        total = len(self._history)
        done = sum(1 for r in self._history if r.status == "done")
        partial = sum(1 for r in self._history if r.status == "partial")
        failed = sum(1 for r in self._history if r.status == "failed")
        return {
            "total_swarms": total,
            "done": done,
            "partial": partial,
            "failed": failed,
            "max_agents": self._max_agents,
            "active_agents": len(self._factory.list_active()),
        }
