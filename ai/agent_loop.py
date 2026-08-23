"""
ai/agent_loop.py
================
🆕 حلقة التنفيذ متعددة الجولات (Multi-turn Agent Loop) — جوهر الوكيل الذاتي.

تحول الوكيل من "مولّد قائمة خطوات من جولة واحدة" إلى "عامل يلاحظ ثم يتصرف":

    [user] → LLM → (plan) → {act → observe → decide} ×N → (final answer)
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger("NeuralServiceMesh.AgentLoop")

ROOT = Path(__file__).resolve().parent.parent

# ═════════════════════════ حدود الأمان (ثوابت) ═════════════════════
_MAX_ROUNDS = 10                # أقصى جولات LLM في الحلقة الواحدة
_MAX_TOOLS_PER_ROUND = 8        # أقصى أدوات في الجولة الواحدة
_MAX_TOTAL_TOOLS = 40           # ميزانية الأدوات الإجمالية
_DEFAULT_TIMEOUT = 90           # ثوانٍ لتنفيذ الأمر الواحد
_LONG_CMD_MARKERS = ("git clone", "git pull --rebase", "kaggle kernels output",
                     "kaggle datasets download", "pip install -r", "pytest")

def _default_timeout_for(cmd: str) -> int:
    c = (cmd or "").lower()
    return 600 if any(m in c for m in _LONG_CMD_MARKERS) else _DEFAULT_TIMEOUT

_MAX_OUTPUT_CHARS = 4000
_CMD_BLOCKLIST = (
    "sudo ", "sudo\t", "rm -rf /", "mkfs", ":(){ :|:& };:",
    ">:() { :|:& };:", ">:(){ :|:& };:", "chmod 777 /",
    "crontab -r", ">/dev/sd", "pkill", "killall", "kill -9 -",
    "dd if=/dev/zero", "format C:", "del /f /s /q \\",
    "shutdown", "reboot", "init 0", "init 6",
)
_PATH_TRAVERSAL_RE = re.compile(r"(^|/)((\.\.(\/|\\|$))+)")

_RUN_LOCK = threading.Lock()

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

def _safe_tool_path(raw: str) -> Optional[Path]:
    if not raw or not str(raw).strip():
        return None
    try:
        candidate = (ROOT / str(raw).strip()).resolve()
        candidate.relative_to(ROOT.resolve())
    except (ValueError, OSError, TypeError):
        return None
    return candidate

def _cmd_safe(cmd: str) -> Tuple[bool, str]:
    c = (cmd or "").strip()
    if not c:
        return False, "أمر فارغ"
    if any(c.startswith(b) for b in _CMD_BLOCKLIST):
        return False, "أمر محظور لأسباب أمنية"
    if re.search(r"(^|\|)\s*(su|pkexec|nohup)\b", c):
        return False, "استدعاء صلاحيات محظور"
    return True, ""

# ═════════════════════════ سجل المراجعة ════════════════════════════
_AUDIT_DIR = ROOT / "artifacts" / "agent_loop" / "audit"
_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

def _audit(loop_id: str, entry: Dict[str, Any]) -> None:
    try:
        p = _AUDIT_DIR / f"{loop_id}.jsonl"
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _now(), **entry}, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("audit failed: %s", e)

# ═════════════════════════ Registry الأدوات ════════════════════════
class ToolSpec:
    def __init__(self, name: str, description: str, params_schema: Dict[str, Any],
                 executor: Callable[[Dict[str, Any]], str], dangerous: bool = False):
        self.name = name
        self.description = description
        self.params_schema = params_schema
        self.executor = executor
        self.dangerous = dangerous

TOOL_REGISTRY: Dict[str, ToolSpec] = {}
_TOOL_ORDER: List[str] = []

def register_tool(spec: ToolSpec) -> ToolSpec:
    TOOL_REGISTRY[spec.name] = spec
    if spec.name not in _TOOL_ORDER:
        _TOOL_ORDER.append(spec.name)
    return spec

def _truncate_obs(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if not text or len(text) <= limit:
        return text
    return text[:limit - 200] + f"\n\n... [مقطوع — الأصل {len(text)} حرف]"

# ── الأدوات الأساسية ──────────────────────────────────────────────
def _exec_shell(params: Dict[str, Any]) -> str:
    cmd = str(params.get("cmd", "")).strip()
    ok, why = _cmd_safe(cmd)
    if not ok: return f"❌ shell: {why}"
    try:
        from ai.nsm_terminal import get_terminal
        r = get_terminal().run_agent("agent_loop", cmd, timeout=int(params.get("timeout", _default_timeout_for(cmd))))
        out = []
        if r.stdout: out.append(r.stdout[:_MAX_OUTPUT_CHARS])
        if r.stderr: out.append("[stderr]\n" + r.stderr[:_MAX_OUTPUT_CHARS])
        return ("\n".join(out) or "تم التنفيذ")[:_MAX_OUTPUT_CHARS]
    except Exception as e: return f"❌ shell: {e}"

register_tool(ToolSpec("shell", "تنفيذ أمر shell", {"type": "object", "properties": {"cmd": {"type": "string"}}}, _exec_shell, dangerous=True))

def _tool_read(params: Dict[str, Any]) -> str:
    path = _safe_tool_path(str(params.get("path", "")))
    if not path or not path.exists(): return "❌ read_file: مسار غير صالح"
    try: return path.read_text(encoding="utf-8")[:_MAX_OUTPUT_CHARS]
    except Exception as e: return f"❌ read_file: {e}"

register_tool(ToolSpec("read_file", "قراءة ملف", {"type": "object", "properties": {"path": {"type": "string"}}}, _tool_read))

def _tool_write(params: Dict[str, Any]) -> str:
    path = _safe_tool_path(str(params.get("path", "")))
    if not path: return "❌ write_file: مسار غير صالح"
    content = str(params.get("content", ""))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"✅ كُتب {params.get('path')}"
    except Exception as e: return f"❌ write_file: {e}"

register_tool(ToolSpec("write_file", "كتابة ملف", {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}, _tool_write, dangerous=True))

def _tool_sleep(params: Dict[str, Any]) -> str:
    agent_id = str(params.get("agent_id", "default"))
    return f"SIGNAL_SLEEP:{agent_id}"

register_tool(ToolSpec("sleep", "دخول وضع النوم", {"type": "object", "properties": {"agent_id": {"type": "string"}, "reason": {"type": "string"}, "wake_up_after": {"type": "integer"}}}, _tool_sleep))

def _tool_wake_up(params: Dict[str, Any]) -> str:
    agent_id = str(params.get("agent_id", ""))
    lazy = bool(params.get("lazy", False))
    try:
        from ai.agent_hibernation import wake_up_agent
        state = wake_up_agent(agent_id, lazy=lazy)
        if state: return f"🌅 الوكيل {agent_id} استيقظ ({'تدريجي' if lazy else 'كامل'})."
        return f"ℹ️ لا توجد حالة للوكيل {agent_id}."
    except Exception as e: return f"❌ wake_up: {e}"

register_tool(ToolSpec("wake_up", "إيقاظ وكيل", {"type": "object", "properties": {"agent_id": {"type": "string"}, "lazy": {"type": "boolean"}}}, _tool_wake_up))

# ═════════════════════════ محرك الحلقة ═════════════════════════════
_SYSTEM_PROMPT = """أنت الوكيل التنفيذي لـ NSM. رد JSON فقط:
{"thinking": "...", "tools": [{"tool": "...", "params": {...}}], "finish": "...", "end": true/false}"""

def _parse_tool_call(raw: str) -> Optional[Dict[str, Any]]:
    try: return json.loads(re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.S))
    except: return None

def _invoke_llm(llm_fn: Callable, system: str, history: List[Dict[str, Any]]) -> str:
    resp = llm_fn(system, history)
    if not resp: raise RuntimeError("رد LLM فارغ")
    return str(resp)

def _build_tools_prompt() -> str:
    return "الأدوات: " + ", ".join(_TOOL_ORDER)

class LoopState:
    def __init__(self, loop_id: str, user_input: str):
        self.loop_id, self.user_input = loop_id, user_input
        self.round = 0
        self.steps = []
        self.status = "pending"
        self.started_at = _now()
        self.tools_used = 0

    def record(self, event: Dict[str, Any]): self.steps.append(event)

def run_agent_loop(user_input: str, *, llm_fn: Optional[Callable] = None, max_rounds: int = 10) -> Generator[Dict[str, Any], None, None]:
    loop_id = f"loop_{uuid.uuid4().hex[:8]}"
    state = LoopState(loop_id, user_input)
    
    def _emit(event: Dict[str, Any]):
        state.record(event)
        _audit(loop_id, event)
        yield_queue.append(event)

    yield_queue = []
    def _flush():
        while yield_queue: yield yield_queue.pop(0)

    try:
        with _RUN_LOCK:
            state.status = "running"
            _emit({"type": "status", "loop_id": loop_id, "status": "running"})
            yield from _flush()

            fn = llm_fn or (lambda s, h: "JSON logic here") # Placeholder
            system = _SYSTEM_PROMPT + "\n" + _build_tools_prompt()
            
            # استعادة الحالة (الاستيقاظ التدريجي)
            from ai.agent_hibernation import wake_up_agent
            recovered = wake_up_agent(user_input)
            if recovered:
                history = recovered.context
                history.append({"role": "user", "content": "🌅 استيقظت. لخص أين توقفت."})
                _emit({"type": "info", "text": "🌅 تم استعادة الحالة (Mental Warm-up)..."})
            else:
                history = [{"role": "user", "content": user_input}]

            from ai.workload_monitor import WorkloadMonitor
            monitor = WorkloadMonitor()
            
            total_tools, done = 0, False
            while state.round < max_rounds and not done:
                state.round += 1
                _emit({"type": "status", "round": state.round})
                
                # Auto-Save
                if state.round % 5 == 0:
                    from ai.agent_hibernation import hibernate_agent
                    hibernate_agent(f"{loop_id}_autosave", history, {"steps": state.steps})
                    _emit({"type": "info", "text": "💾 حفظ تلقائي."})
                
                target_agent_id = f"agent_{loop_id}"
                try:
                    monitor.record_activity()
                    raw = _invoke_llm(fn, system, history)
                except Exception as e:
                    _emit({"type": "answer", "text": f"❌ خطأ LLM: {e}"})
                    break
                
                history.append({"role": "assistant", "content": raw})
                parsed = _parse_tool_call(raw)
                if not parsed: continue

                tools = parsed.get("tools") or []
                obs_round = []
                sleep_requested = False
                
                for t_req in tools:
                    tname = t_req.get("tool")
                    params = t_req.get("params", {})
                    spec = TOOL_REGISTRY.get(tname)
                    total_tools += 1
                    state.tools_used += 1
                    _emit({"type": "tool", "tool": tname, "params": params})
                    
                    if not spec: obs = f"❌ أداة غير معروفة: {tname}"
                    else:
                        obs = _truncate_obs(spec.executor(params))
                        if str(obs).startswith("SIGNAL_SLEEP:"):
                            sleep_requested = True
                            target_agent_id = str(obs).split(":")[1]
                    
                    obs_round.append(f"[{tname}] {obs}")
                    _emit({"type": "result", "tool": tname, "output": obs})
                    if sleep_requested: break

                if sleep_requested:
                    from ai.agent_hibernation import hibernate_agent
                    sleep_reason = "Manual"
                    wake_after = 0
                    for t in tools:
                        if t.get("tool") == "sleep":
                            sleep_reason = t.get("params", {}).get("reason", sleep_reason)
                            wake_after = int(t.get("params", {}).get("wake_up_after", 0))
                    
                    if hibernate_agent(target_agent_id, history, {"steps": state.steps}):
                        if wake_after > 0:
                            from ai.agent_hibernation import schedule_wake_up
                            schedule_wake_up(target_agent_id, wake_after)
                        _emit({"type": "answer", "text": f"💤 نام الوكيل {target_agent_id}."})
                        done = True
                        break
                else:
                    sleep_est = monitor.estimate_sleep_need(len(state.steps))
                    if sleep_est["should_sleep"]:
                        _emit({"type": "info", "text": "💡 توصية بالنوم."})

                if obs_round:
                    history.append({"role": "user", "content": "\n".join(obs_round)})
                
                if parsed.get("end"):
                    _emit({"type": "answer", "text": parsed.get("finish", "تم")})
                    done = True
            
            yield from _flush()
    finally:
        state.status = "done"
        yield from _flush()
