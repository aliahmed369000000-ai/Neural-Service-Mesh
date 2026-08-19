"""
ai/agent_auto_heal.py
=====================
🆕 نظام الإصلاح التلقائي (Auto-Healing) للوكلاء.

عند فشل أداة أو أمر، يُشخّص السبب تلقائيًا ويُعيد المحاولة باستراتيجية
مختلفة (تعديل الأمر، استخدام بديل، تبسيط المدخلات).

الاستخدام:
    from ai.agent_auto_heal import AutoHeal
    healer = AutoHeal(max_rounds=3)
    result = healer.execute_with_healing(
        tool_fn=shell_execute,
        tool_args={"cmd": "pip install ..."},
        context={"prev_error": None},
    )

الأحداث: heal_started, heal_diagnosis, heal_retry, heal_resolved, heal_gave_up
"""
from __future__ import annotations
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
logger = logging.getLogger("nsm.auto_heal")

EVENT_HEAL_STARTED = "heal_started"
EVENT_HEAL_DIAGNOSIS = "heal_diagnosis"
EVENT_HEAL_RETRY = "heal_retry"
EVENT_HEAL_RESOLVED = "heal_resolved"
EVENT_HEAL_GAVE_UP = "heal_gave_up"


class AutoHeal:
    """نظام إصلاح تلقائي للوكلاء — يشخّص الخطأ ويُعيد المحاولة."""

    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds

    # ═══ أنماط الأخطاء الشائعة وإصلاحاتها ═══
    HEAL_STRATEGIES: Dict[str, Dict[str, Any]] = {
        "timeout": {
            "patterns": [r"timeout", r"timed out", r"توقف", r"انتهت مهلة"],
            "fix": "increase_timeout",
            "desc": "زيادة مهلة التنفيذ 3×",
        },
        "pip_missing": {
            "patterns": [r"ModuleNotFoundError", r"No module named", r"ImportError"],
            "fix": "install_missing",
            "desc": "تثبيت المكتبة المفقودة",
        },
        "permission": {
            "patterns": [r"PermissionError", r"permission denied", r"لا إذن"],
            "fix": "retry_with_sudo",
            "desc": "إعادة المحاولة بصلاحيات مختلفة",
        },
        "file_not_found": {
            "patterns": [r"FileNotFoundError", r"No such file", r"ملف غير موجود"],
            "fix": "create_or_locate",
            "desc": "إنشاء الملف أو البحث عن بديل",
        },
        "rate_limit": {
            "patterns": [r"429", r"rate.limit", r"too many"],
            "fix": "backoff_retry",
            "desc": "انتظار ثم إعادة المحاولة",
        },
        "memory": {
            "patterns": [r"MemoryError", r"OOM", r"killed", r"out of memory"],
            "fix": "reduce_batch",
            "desc": "تقليل حجم الدفعة وإعادة المحاولة",
        },
        "network": {
            "patterns": [r"ConnectionError", r"ConnectionRefused", r"DNS", r"Unreachable"],
            "fix": "retry_network",
            "desc": "إعادة المحاولة مع انتظار قصير",
        },
    }

    def diagnose(self, error_output: str) -> Optional[Dict[str, Any]]:
        """تشخيص الخطأ وإرجاع الاستراتيجية المناسبة."""
        if not error_output:
            return None
        lower = error_output.lower()
        for key, strategy in self.HEAL_STRATEGIES.items():
            for pattern in strategy["patterns"]:
                if re.search(pattern, lower, re.I):
                    return {"key": key, "strategy": strategy["fix"],
                            "desc": strategy["desc"]}
        return None

    def execute_with_healing(
        self,
        tool_fn: Callable[..., Any],
        tool_args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        emit_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """تنفيذ الأداة مع إصلاح تلقائي عند الفشل."""
        attempt = 0
        last_error = ""

        # emit heal_started
        if emit_fn:
            emit_fn(EVENT_HEAL_STARTED, metadata={
                "tool": tool_fn.__name__ if hasattr(tool_fn, '__name__') else str(tool_fn),
                "max_rounds": self.max_rounds,
            })

        while attempt < self.max_rounds:
            attempt += 1
            try:
                # تعديل args بناءً على الاستراتيجية
                if attempt > 1 and last_error:
                    diag = self.diagnose(last_error)
                    if diag:
                        tool_args = self._apply_fix(diag, tool_args)
                        if emit_fn:
                            emit_fn(EVENT_HEAL_DIAGNOSIS, metadata={
                                "attempt": attempt,
                                "diagnosis": diag,
                            })
                            emit_fn(EVENT_HEAL_RETRY, metadata={
                                "attempt": attempt,
                                "strategy": diag["strategy"],
                            })

                result = tool_fn(**tool_args)

                # فحص هل النتيجة تحتوي خطأ
                if self._is_success(result):
                    if emit_fn and attempt > 1:
                        emit_fn(EVENT_HEAL_RESOLVED, metadata={
                            "attempts": attempt,
                        })
                    return {"ok": True, "result": result, "attempts": attempt,
                            "healed": attempt > 1}

                last_error = self._extract_error(result)

            except Exception as e:
                last_error = str(e)
                diag = self.diagnose(last_error)
                if diag and emit_fn:
                    emit_fn(EVENT_HEAL_DIAGNOSIS, metadata={
                        "attempt": attempt,
                        "diagnosis": diag,
                    })
                if attempt >= self.max_rounds:
                    if emit_fn:
                        emit_fn(EVENT_HEAL_GAVE_UP, metadata={
                            "attempts": attempt,
                            "error": last_error,
                        })
                    return {"ok": False, "error": last_error,
                            "attempts": attempt, "healed": False}

            # انتظار قصير قبل إعادة المحاولة
            if attempt < self.max_rounds:
                time.sleep(0.5 * attempt)

        return {"ok": False, "error": last_error,
                "attempts": attempt, "healed": False}

    def _apply_fix(self, diagnosis: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
        """تطبيق الإصلاح على args."""
        fix = diagnosis["strategy"]
        args = dict(args)  # copy

        if fix == "increase_timeout":
            current = args.get("timeout", 90)
            args["timeout"] = min(current * 3, 600)
        elif fix == "install_missing":
            # استخراج اسم المكتبة من الخطأ
            m = re.search(r"No module named ['\"](\w+)", diagnosis.get("desc", ""))
            if m:
                args.setdefault("_pre_commands", [])
                args["_pre_commands"].append(f"pip install {m.group(1)} -q")
        elif fix == "backoff_retry":
            time.sleep(2 * (self.max_rounds - self.max_rounds + 1))
        elif fix == "reduce_batch":
            if "batch_size" in args:
                args["batch_size"] = max(1, args["batch_size"] // 2)
            if "batch" in args:
                args["batch"] = max(1, args["batch"] // 2)

        return args

    @staticmethod
    def _is_success(result: Any) -> bool:
        """فحص هل النتيجة نجاح."""
        if isinstance(result, dict):
            return result.get("ok", result.get("success", False))
        if isinstance(result, tuple) and len(result) >= 2:
            return result[0] == 0
        return result is not None

    @staticmethod
    def _extract_error(result: Any) -> str:
        """استخراج رسالة الخطأ من النتيجة."""
        if isinstance(result, dict):
            return result.get("error", result.get("msg", str(result)))
        if isinstance(result, tuple) and len(result) >= 2:
            return str(result[1])
        return str(result)
