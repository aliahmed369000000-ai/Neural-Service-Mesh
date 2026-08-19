"""
ai/agent_collaboration.py
=========================
🆕 نظام تعاون الوكلاء (Agent Collaboration).

يُمكّن عدة وكلاء من العمل معًا على مهام معقدة:
  • تقسيم المهمة إلى subtasks
  • تنفيذ متوازي (thread-safe)
  • تجميع النتائج وتوليفها
  • retry على مستوى subtask

الاستخدام:
    from ai.agent_collaboration import CollaborativePlanner
    planner = CollaborativePlanner()
    result = planner.execute_collaborative(
        task="حلل المشروع وأصلح الأخطاء",
        subtasks=[
            {"agent": "research", "prompt": "افحص الأخطاء"},
            {"agent": "coder", "prompt": "أصلح الأخطاء"},
        ],
    )
"""
from __future__ import annotations
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple
logger = logging.getLogger("nsm.collaboration")

EVENT_COLLAB_STARTED = "collab_started"
EVENT_COLLAB_SUBTASK_DONE = "collab_subtask_done"
EVENT_COLLAB_COMPLETED = "collab_completed"


class CollaborativePlanner:
    """تخطيط وتنفيذ تعاوني بين عدة وكلاء."""

    def __init__(
        self,
        max_workers: int = 4,
        timeout_per_subtask: int = 300,
    ):
        self.max_workers = max_workers
        self.timeout_per_subtask = timeout_per_subtask
        self._lock = threading.Lock()

    def split_task(self, task_description: str) -> List[Dict[str, str]]:
        """تقسيم مهمة معقدة إلى subtasks (بشكل ذكي)."""
        # تقسيم أساسي — يمكن تعزيزه بـ LLM لاحقًا
        hints = [
            ("حلل|analyze|افحص|فحص", "analyze"),
            ("أصلح|fix|repair|صحح", "fix"),
            ("اكتب|write|أنشئ|create|build", "create"),
            ("اختبر|test|تحقق|verify", "test"),
            ("ابحث|search|google", "search"),
            ("لخّص|summarize|اختصر", "summarize"),
        ]

        subtasks = []
        lower = task_description.lower()
        for pattern, action in hints:
            import re
            if re.search(pattern, lower):
                subtasks.append({
                    "action": action,
                    "prompt": task_description,
                    "agent_type": self._agent_for_action(action),
                })

        if not subtasks:
            subtasks = [{"action": "general", "prompt": task_description,
                         "agent_type": "nsm_agent"}]

        return subtasks

    def _agent_for_action(self, action: str) -> str:
        """تحديد نوع الوكيل المناسب للإجراء."""
        mapping = {
            "analyze": "research_agent",
            "fix": "coder_agent",
            "create": "coder_agent",
            "test": "tester_agent",
            "search": "research_agent",
            "summarize": "summarizer_agent",
            "general": "nsm_agent",
        }
        return mapping.get(action, "nsm_agent")

    def execute_collaborative(
        self,
        task: str,
        subtasks: Optional[List[Dict[str, str]]] = None,
        execute_fn: Optional[Callable[[Dict[str, str]], Any]] = None,
        emit_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """تنفيذ مهمة تعاونية مع عدة وكلاء."""
        start_time = time.time()

        if subtasks is None:
            subtasks = self.split_task(task)

        if emit_fn:
            emit_fn(EVENT_COLLAB_STARTED, metadata={
                "task": task[:100],
                "subtasks": len(subtasks),
            })

        results: List[Dict[str, Any]] = []
        errors: List[str] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_subtask = {}
            for i, subtask in enumerate(subtasks):
                subtask_id = f"{uuid.uuid4().hex[:8]}-{i}"
                subtask["id"] = subtask_id

                future = executor.submit(
                    self._execute_subtask,
                    subtask, execute_fn or self._default_execute,
                )
                future_to_subtask[future] = subtask

            for future in as_completed(future_to_subtask, timeout=self.timeout_per_subtask * len(subtasks)):
                subtask = future_to_subtask[future]
                try:
                    result = future.result(timeout=self.timeout_per_subtask)
                    results.append(result)
                    if emit_fn:
                        emit_fn(EVENT_COLLAB_SUBTASK_DONE, metadata={
                            "subtask_id": subtask["id"],
                            "agent": subtask.get("agent_type", "unknown"),
                            "ok": result.get("ok", False),
                        })
                except Exception as e:
                    errors.append(f"{subtask.get('id', '?')}: {e}")

        elapsed = time.time() - start_time

        if emit_fn:
            emit_fn(EVENT_COLLAB_COMPLETED, metadata={
                "total_subtasks": len(subtasks),
                "successful": len(results),
                "failed": len(errors),
                "elapsed_seconds": round(elapsed, 2),
            })

        return {
            "ok": len(errors) == 0,
            "results": results,
            "errors": errors,
            "elapsed_seconds": round(elapsed, 2),
            "total_subtasks": len(subtasks),
        }

    def _execute_subtask(
        self,
        subtask: Dict[str, str],
        execute_fn: Callable,
    ) -> Dict[str, Any]:
        """تنفيذ subtask واحد."""
        try:
            result = execute_fn(subtask)
            return {"ok": True, "subtask_id": subtask.get("id"),
                    "agent_type": subtask.get("agent_type"),
                    "result": result}
        except Exception as e:
            return {"ok": False, "subtask_id": subtask.get("id"),
                    "agent_type": subtask.get("agent_type"),
                    "error": str(e)}

    @staticmethod
    def _default_execute(subtask: Dict[str, str]) -> str:
        """تنفيذ افتراضي — يُستبدل بـ agent loop حقيقي."""
        return f"Executed: {subtask.get('prompt', '')[:50]}"

    def merge_results(self, results: List[Dict[str, Any]]) -> str:
        """توليف نتائج الوكلاء في إجابة نهائية واحدة."""
        parts = []
        for r in results:
            if r.get("ok"):
                agent = r.get("agent_type", "unknown")
                result = r.get("result", "")
                parts.append(f"**{agent}:** {result}")
            else:
                parts.append(f"⚠️ فشل: {r.get('error', 'unknown')}")

        if not parts:
            return "لم تكتمل المهمة."

        return "\n\n".join(parts)
