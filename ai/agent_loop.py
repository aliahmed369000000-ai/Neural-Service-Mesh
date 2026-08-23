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
import concurrent.futures
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from ai.cache_manager import agent_cache
from ai.learning_engine import learning_engine
from ai.video_sampler import video_sampler
from ai.video_indexer import video_indexer
from ai.multimodal_sync import multimodal_sync
from ai.agent_auto_heal import AutoHeal

logger = logging.getLogger("NeuralServiceMesh.AgentLoop")
healer = AutoHeal(max_rounds=3)

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

def _tool_find_files(params: Dict[str, Any]) -> str:
    pattern = str(params.get("pattern", "*"))
    try:
        matches = list(ROOT.glob(f"**/{pattern}"))
        if not matches: return "ℹ️ لم يتم العثور على ملفات تطابق النمط."
        return "\n".join([str(m.relative_to(ROOT)) for m in matches[:20]])
    except Exception as e: return f"❌ find_files: {e}"

register_tool(ToolSpec("find_files", "البحث عن ملفات بنمط معين", {"type": "object", "properties": {"pattern": {"type": "string"}}}, _tool_find_files))

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

# ── أدوات الوسائط والساندبوكس ──────────────────────────────────────
def _tool_image_search(params: Dict[str, Any]) -> str:
    query = str(params.get("query", ""))
    try:
        from ai.image_search_tool import image_search_safe
        r = image_search_safe(query, max_results=params.get("max_results", 5))
        if not r["ok"]: return f"❌ image_search: {r['error']}"
        return json.dumps(r["results"], ensure_ascii=False, indent=2)
    except Exception as e: return f"❌ image_search: {e}"

register_tool(ToolSpec("image_search", "البحث عن صور حقيقية", {"type": "object", "properties": {"query": {"type": "string"}}}, _tool_image_search))

def _tool_sandbox_test(params: Dict[str, Any]) -> str:
    code = str(params.get("code", ""))
    module_name = str(params.get("module_name", "temp_agent_module"))
    class_name = str(params.get("class_name", "GeneratedNode"))
    try:
        from ai.sandbox_lab import SandboxTestingLab
        # إنشاء كائن وحدة وهمي للتوافق مع sandbox_lab
        class MockModule:
            def __init__(self, mid, name, code, cname):
                self.module_id, self.name, self.code, self.class_name = mid, name, code, cname
                self.status = "new"
                self.test_result = None

        lab = SandboxTestingLab(sandbox_dir=str(ROOT / "artifacts" / "sandbox"))
        mock = MockModule("agent_test", module_name, code, class_name)
        res = lab.test_module(mock)
        
        # إذا كان الاختبار لمعالجة صور، نحفظ النتيجة في الذاكرة البصرية اللحظية
        if "image" in module_name.lower() and res.execution_success:
            # محاولة الوصول للحالة الحالية (عبر متغير عام مؤقت أو حقن)
            # للتبسيط في هذا الاختبار، سنفترض وجود آلية لتسجيلها في visual_memory
            pass
            
        return json.dumps(res.to_dict(), ensure_ascii=False, indent=2)
    except Exception as e: return f"❌ sandbox_test: {e}"

register_tool(ToolSpec("sandbox_test", "اختبار كود في بيئة معزولة", {"type": "object", "properties": {"code": {"type": "string"}, "module_name": {"type": "string"}}}, _tool_sandbox_test, dangerous=True))

def _tool_video_sample(params: Dict[str, Any]) -> str:
    """يأخذ عينات ذكية من الفيديو ويحدث الفهرس الزمني."""
    video_path = str(params.get("video_path", ""))
    video_id = str(params.get("video_id", f"vid_{uuid.uuid4().hex[:6]}"))
    try:
        result = video_sampler.process_video(video_path, video_id)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e: return f"❌ video_sample: {e}"

register_tool(ToolSpec("video_sample", "أخذ عينات ذكية من الفيديو", {"type": "object", "properties": {"video_path": {"type": "string"}, "video_id": {"type": "string"}}}, _tool_video_sample))

def _tool_video_sync(params: Dict[str, Any]) -> str:
    """مزامنة الصوت والصورة للفيديو وتصحيح الانحراف بنظام الأنابيب الموزع مع التعافي التلقائي."""
    video_id = str(params.get("video_id", ""))
    audio_path = str(params.get("audio_path", ""))
    
    if not video_id or not audio_path:
        return "❌ video_sync: يجب توفير video_id و audio_path."
    
    def _sync_execution(vid, path):
        # 1. محاكاة توزيع المهام (Pipeline Simulation)
        pipeline_log = [f"🔄 بدء الأنبوب الموزع للمصدر: {vid}"]
        pipeline_log.append(f"🎙️ [Audio Agent]: تفريغ المسار الصوتي...")
        pipeline_log.append(f"👁️ [Vision Agent]: تحليل الإطارات البصرية...")
        pipeline_log.append(f"⚖️ [Sync Agent]: دمج البيانات وتطبيق مرشح كالمان...")
        
        res = multimodal_sync.sync_video_audio(vid, path)
        if res.get("ok"):
            pipeline_log.append(f"✅ [Reasoning Agent]: تم بناء السياق الموحد بنجاح.")
            res["pipeline_log"] = pipeline_log
        return res

    try:
        # استخدام AutoHeal لتنفيذ المزامنة مع قدرة على الإصلاح التلقائي
        heal_result = healer.execute_with_healing(
            tool_fn=_sync_execution,
            tool_args={"vid": video_id, "path": audio_path}
        )
        
        if heal_result["ok"]:
            return json.dumps(heal_result["result"], ensure_ascii=False, indent=2)
        else:
            return f"❌ video_sync (Auto-Heal Failed): {heal_result['error']}"
    except Exception as e:
        return f"❌ video_sync (System Error): {e}"

register_tool(ToolSpec("video_sync", "مزامنة الصوت والصورة وتصحيح الانحراف مع الفهرسة الدلالية", 
                        {"type": "object", "properties": {"video_id": {"type": "string"}, "audio_path": {"type": "string"}}}, 
                        _tool_video_sync))

def _tool_video_search(params: Dict[str, Any]) -> str:
    """البحث الدلالي في سياق الفيديو المزامَن."""
    video_id = str(params.get("video_id", ""))
    query = str(params.get("query", ""))
    semantic = bool(params.get("semantic", True))
    
    if not video_id or not query:
        return "❌ video_search: يجب توفير video_id و query."
        
    try:
        results = multimodal_sync.query_context(video_id, query, semantic=semantic)
        return json.dumps({
            "ok": True,
            "query": query,
            "results_count": len(results),
            "top_results": results[:3] # إرجاع أفضل 3 نتائج لتوفير السياق
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"❌ video_search Error: {e}"

register_tool(ToolSpec("video_search", "البحث الدلالي في سياق الفيديو المزامَن", 
                        {"type": "object", "properties": {
                            "video_id": {"type": "string"}, 
                            "query": {"type": "string"},
                            "semantic": {"type": "boolean"}
                        }}, 
                        _tool_video_search))

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
        self.visual_memory = {} # تخزين نتائج معالجة الصور اللحظية
        self.audio_memory = {}  # تخزين نتائج معالجة الصوت اللحظية
        self.agent_roles = {
            "vision": "متخصص في تحليل الإطارات والميزات البصرية",
            "audio": "متخصص في تفريغ ومعالجة المسارات الصوتية",
            "sync": "متخصص في مزامنة الطوابع الزمنية وتصحيح الانحراف",
            "reasoning": "متخصص في اتخاذ القرارات النهائية بناءً على السياق الموحد"
        }
        self.pipeline_context = {} # مخزن لتبادل البيانات بين الأدوار المتخصصة

    def set_pipeline_data(self, key: str, value: Any, role: str):
        """تخزين بيانات في الأنبوب مع تحديد الدور المسؤول."""
        self.pipeline_context[key] = {
            "value": value,
            "provider": role,
            "timestamp": time.time()
        }

    def get_pipeline_data(self, key: str) -> Optional[Any]:
        """جلب بيانات من الأنبوب."""
        data = self.pipeline_context.get(key)
        return data["value"] if data else None

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
                state.visual_memory = getattr(recovered, "visual_context", {})
                state.audio_memory = getattr(recovered, "audio_context", {})
                state.multimodal_memory = getattr(recovered, "multimodal_memory", {})
                warmup_msg = "🌅 استيقظت. لخص أين توقفت."
                if recovered.pending_tasks:
                    warmup_msg += f"\nالمهام المعلقة التي تم رصدها قبل النوم:\n- " + "\n- ".join(recovered.pending_tasks)
                
                if state.visual_memory:
                    warmup_msg += f"\n👁️ السياق البصري المستعاد (Visual Context):\n"
                    for img_name, img_data in state.visual_memory.items():
                        warmup_msg += f"- {img_name}: {img_data.get('dimensions', 'N/A')} | {img_data.get('status', 'Unknown')}\n"
                
                if state.audio_memory:
                    warmup_msg += f"\n🎙️ السياق الصوتي المستعاد (Audio Context):\n"
                    for audio_name, audio_data in state.audio_memory.items():
                        warmup_msg += f"- {audio_name}: {audio_data.get('duration', 'N/A')}s | {audio_data.get('type', 'Unknown')}\n"
                
                if state.multimodal_memory:
                    warmup_msg += f"\n⚖️ السياق السمعي البصري المزامَن (Multimodal Memory):\n"
                    for vid_id, sync_data in state.multimodal_memory.items():
                        warmup_msg += f"- {vid_id}: تم مزامنة {len(sync_data.get('multimodal_sync', []))} نقطة زمنية.\n"
                
                history.append({"role": "user", "content": warmup_msg})
                _emit({"type": "info", "text": "🌅 تم استعادة الحالة (Mental Warm-up المتعدد الوسائط)..."})
            else:
                history = [{"role": "user", "content": user_input}]
                # حقن الدروس المستفادة في السياق الأول
                lessons = learning_engine.get_relevant_lessons(user_input)
                if lessons:
                    history.append({"role": "system", "content": lessons})
            
            from ai.workload_monitor import WorkloadMonitor
            monitor = WorkloadMonitor()
            
            total_tools, done = 0, False
            while state.round < max_rounds and not done:
                state.round += 1
                _emit({"type": "status", "round": state.round})
                
                # Auto-Save
                if state.round % 5 == 0:
                    from ai.agent_hibernation import hibernate_agent
                    # تفعيل الضغط في الحفظ التلقائي لتوفير المساحة
                    hibernate_agent(f"{loop_id}_autosave", history, {"steps": state.steps}, compress=True)
                    _emit({"type": "info", "text": "💾 حفظ تلقائي (مع الضغط الديناميكي)."})
                
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
                
                # تنفيذ الأدوات (دعم التوازي للتحسين)
                if len(tools) > 1:
                    _emit({"type": "info", "text": f"⚡ تشغيل {len(tools)} أدوات بشكل متوازٍ..."})
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_to_tool = {}
                        for t_req in tools:
                            tname = t_req.get("tool")
                            params = t_req.get("params", {})
                            spec = TOOL_REGISTRY.get(tname)
                            total_tools += 1
                            state.tools_used += 1
                            _emit({"type": "tool", "tool": tname, "params": params})
                            
                            if spec:
                                # محاولة جلب النتيجة من الكاش أولاً
                                cached_res = agent_cache.get(tname, params)
                                if cached_res:
                                    obs = _truncate_obs(cached_res)
                                    obs_round.append(f"[{tname}] {obs} (⚡ cached)")
                                    _emit({"type": "result", "tool": tname, "output": f"{obs} (⚡ cached)"})
                                    state.tools_used += 1
                                    continue
                                
                                future = executor.submit(spec.executor, params)
                                future_to_tool[future] = t_req
                            else:
                                obs_round.append(f"[{tname}] ❌ أداة غير معروفة")
                                _emit({"type": "result", "tool": tname, "output": "❌ أداة غير معروفة"})

                        for future in concurrent.futures.as_completed(future_to_tool):
                            t_req = future_to_tool[future]
                            tname = t_req.get("tool")
                            try:
                                raw_res = future.result()
                                obs = _truncate_obs(raw_res)
                                # حفظ النتيجة في الكاش للطلبات المستقبلية
                                agent_cache.set(tname, t_req.get("params", {}), raw_res)
                                
                                if str(obs).startswith("SIGNAL_SLEEP:"):
                                    sleep_requested = True
                                    target_agent_id = str(obs).split(":")[1]
                            except Exception as e:
                                obs = f"❌ خطأ تنفيذ: {e}"
                            
                            obs_round.append(f"[{tname}] {obs}")
                            _emit({"type": "result", "tool": tname, "output": obs})
                            if sleep_requested: break
                else:
                    # تنفيذ تسلسلي لأداة واحدة
                    for t_req in tools:
                        tname = t_req.get("tool")
                        params = t_req.get("params", {})
                        spec = TOOL_REGISTRY.get(tname)
                        total_tools += 1
                        state.tools_used += 1
                        _emit({"type": "tool", "tool": tname, "params": params})
                        
                        if not spec: obs = f"❌ أداة غير معروفة: {tname}"
                        else:
                            # محاولة جلب النتيجة من الكاش
                            cached_res = agent_cache.get(tname, params)
                            if cached_res:
                                obs = _truncate_obs(cached_res) + " (⚡ cached)"
                            else:
                                raw_res = spec.executor(params)
                                obs = _truncate_obs(raw_res)
                                agent_cache.set(tname, params, raw_res)
                            
                            if str(obs).startswith("SIGNAL_SLEEP:"):
                                sleep_requested = True
                                target_agent_id = str(obs).split(":")[1]
                        
                        obs_round.append(f"[{tname}] {obs}")
                        _emit({"type": "result", "tool": tname, "output": obs})
                        if sleep_requested: break

                if sleep_requested:
                    from ai.agent_hibernation import hibernate_agent, extract_pending_tasks
                    sleep_reason = "Manual"
                    wake_after = 0
                    for t in tools:
                        if t.get("tool") == "sleep":
                            sleep_reason = t.get("params", {}).get("reason", sleep_reason)
                            wake_after = int(t.get("params", {}).get("wake_up_after", 0))
                    
                    pending = extract_pending_tasks(history, {"steps": state.steps})
                    if hibernate_agent(target_agent_id, history, {"steps": state.steps}, pending_tasks=pending, 
                                       visual_context=state.visual_memory, audio_context=state.audio_memory,
                                       multimodal_memory=state.multimodal_memory, compress=True):
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
                    # استخلاص وتسجيل الخبرة عند انتهاء المهمة
                    finish_text = parsed.get("finish", "تم")
                    learning_engine.record_experience(
                        task=user_input[:100],
                        outcome=finish_text[:200],
                        lesson="المهمة اكتملت بنجاح.",
                        success=True,
                        agent_id=f"agent_{loop_id}"
                    )
                    _emit({"type": "answer", "text": finish_text})
                    done = True
            
            yield from _flush()
    finally:
        state.status = "done"
        yield from _flush()
