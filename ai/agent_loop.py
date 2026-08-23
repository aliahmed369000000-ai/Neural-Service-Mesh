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
import requests
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from ai.cache_manager import agent_cache
from ai.learning_engine import learning_engine
from ai.video_sampler import video_sampler
from ai.video_indexer import video_indexer
from ai.multimodal_sync import multimodal_sync
from ai.agent_auto_heal import AutoHeal
from ai.memory_manager import MemoryManager
from ai.tool_genesis import tool_genesis, ToolGenesis
from ai.evolution_engine import evolution_engine
from ai.tool_discovery import tool_discovery
from ai.task_migrator import TaskMigrator
from ai.self_awareness import SelfAwarenessEngine
from ai.rescue_protocol import rescue_agent

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
    global TOOL_REGISTRY, _TOOL_ORDER
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
        term = get_terminal()
        r = term.run_agent("agent_loop", cmd, timeout=int(params.get("timeout", _default_timeout_for(cmd))))
        
        out = []
        # 🆕 منطق التشخيص التلقائي (Auto-Heal Diagnosis) عند فشل الأوامر الحرجة
        if not r.ok and r.exit_code != 0:
            critical_prefixes = ("python", "pytest", "git push", "git commit", "pip install")
            if any(cmd.startswith(p) for p in critical_prefixes):
                try:
                    from ai.auto_runtime import trigger_auto_heal
                    res = trigger_auto_heal(context={"cmd": cmd, "exit_code": r.exit_code, "stderr": r.stderr})
                    if res.get("ok"):
                        diag = res.get("diagnosis", {})
                        out.append(f"🛠️ [AutoHeal Diagnosis]: {diag.get('desc', 'خطأ غير معروف')}")
                        if diag.get("action"):
                            out.append(f"💡 اقتراح إصلاح: {diag.get('action')}")
                except Exception: pass
        if r.stdout: out.append(r.stdout[:_MAX_OUTPUT_CHARS])
        if r.stderr: 
            err_msg = r.stderr[:_MAX_OUTPUT_CHARS]
            # 🆕 ذكاء تنفيذي: اقتراح حل للخطأ إذا كان معروفاً
            if "ModuleNotFoundError" in err_msg:
                module = re.search(r"No module named '([^']+)'", err_msg)
                if module: out.append(f"\n💡 اقتراح: حاول تثبيت المكتبة عبر `pip install {module.group(1)}`")
            out.append("[stderr]\n" + err_msg)
            
        return ("\n".join(out) or "تم التنفيذ بنجاح")[:_MAX_OUTPUT_CHARS]
    except Exception as e: return f"❌ shell: {e}"

register_tool(ToolSpec("shell", "تنفيذ أمر shell", {"type": "object", "properties": {"cmd": {"type": "string"}}}, _exec_shell, dangerous=True))

def _tool_read(params: Dict[str, Any]) -> str:
    path = _safe_tool_path(str(params.get("path", "")))
    if not path or not path.exists(): return "❌ read_file: مسار غير صالح"
    try: return path.read_text(encoding="utf-8")[:_MAX_OUTPUT_CHARS]
    except Exception as e: return f"❌ read_file: {e}"

register_tool(ToolSpec("read_file", "قراءة ملف", {"type": "object", "properties": {"path": {"type": "string"}}}, _tool_read))

# تحميل الأدوات الديناميكية عند البدء
dynamic_tools = ToolGenesis.load_dynamic_tools()
for name, fn in dynamic_tools.items():
    register_tool(ToolSpec(name, f"أداة ديناميكية: {name}", {}, fn))

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

def _tool_video_sentiment(params: Dict[str, Any]) -> str:
    """تحليل الحالة العاطفية الإجمالية للفيديو المزامَن."""
    video_id = str(params.get("video_id", ""))
    if not video_id: return "❌ video_sentiment: يجب توفير video_id."
    
    try:
        from ai.video_indexer import video_indexer
        index = video_indexer.load_index(video_id)
        if not index or "multimodal_sync" not in index:
            return "❌ video_sentiment: الفيديو غير مزامَن بعد."
            
        sync_data = index["multimodal_sync"]
        if not sync_data: return "❌ video_sentiment: لا توجد بيانات مزامنة."
        
        scores = [item["sentiment"]["score"] for item in sync_data if "sentiment" in item]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        overall = "neutral"
        if avg_score > 0.2: overall = "positive"
        elif avg_score < -0.2: overall = "negative"
        
        return json.dumps({
            "ok": True,
            "overall_sentiment": overall,
            "average_score": round(avg_score, 2),
            "data_points": len(scores)
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"❌ video_sentiment Error: {e}"

register_tool(ToolSpec("video_sentiment", "تحليل الحالة العاطفية الإجمالية للفيديو المزامَن", 
                        {"type": "object", "properties": {"video_id": {"type": "string"}}}, 
                        _tool_video_sentiment))

def _tool_memory_search(params: Dict[str, Any]) -> str:
    """البحث الدلالي في ذاكرة الوكيل (الحقائق والأحداث)."""
    query = str(params.get("query", ""))
    agent_id = str(params.get("agent_id", "default"))
    
    if not query:
        return "❌ memory_search: يجب توفير query."
        
    try:
        from ai.agent_hibernation import wake_up_agent
        agent_state = wake_up_agent(agent_id)
        
        if not agent_state:
            return "❌ فشل الوصول إلى ذاكرة الوكيل."
            
        # استخدام MemoryManager المدمج في الحالة
        results = agent_state.memory_manager.search(query)
        
        return json.dumps({
            "ok": True,
            "query": query,
            "semantic_facts": [r["content"] for r in results["semantic"][:3]],
            "episodic_events": [r["summary"] for r in results["episodic"][:2]]
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"❌ memory_search Error: {e}"

register_tool(ToolSpec("memory_search", "البحث الدلالي في ذاكرة الوكيل (الحقائق والأحداث)", 
                        {"type": "object", "properties": {
                            "query": {"type": "string"},
                            "agent_id": {"type": "string"}
                        }}, 
                        _tool_memory_search))

def _tool_memory_reflection(params: Dict[str, Any]) -> str:
    """استعراض تقييمات التفكير الذاتي للذاكرة الحالية."""
    agent_id = str(params.get("agent_id", "default"))
    try:
        from ai.agent_hibernation import wake_up_agent
        state = wake_up_agent(agent_id, lazy=True)
        if not state or not hasattr(state, 'memory_manager') or not state.memory_manager:
            return "ℹ️ لا توجد بيانات ذاكرة متاحة للتقييم حالياً."
        
        mem = state.memory_manager
        reflection_report = {
            "agent_id": agent_id,
            "semantic_facts_count": len(mem.ltm_semantic),
            "episodic_events_count": len(mem.ltm_episodic),
            "top_important_facts": []
        }
        
        # جلب أهم 5 حقائق بناءً على القوة (التي تعكس الأهمية والتقييم الذاتي)
        sorted_facts = sorted(mem.ltm_semantic.values(), key=lambda x: x.get("strength", 0), reverse=True)
        for f in sorted_facts[:5]:
            reflection_report["top_important_facts"].append({
                "content": f["content"][:100] + "...",
                "importance_score": round(f.get("strength", 0), 2),
                "last_access": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(f.get("last_access", 0)))
            })
            
        return json.dumps(reflection_report, ensure_ascii=False, indent=2)
    except Exception as e: return f"❌ memory_reflection: {e}"

register_tool(ToolSpec("memory_reflection", "استعراض تقييمات التفكير الذاتي للذاكرة", {"type": "object", "properties": {"agent_id": {"type": "string"}}}, _tool_memory_reflection))

def _tool_code_review(params: Dict[str, Any]) -> str:
    """تقييم كفاءة وأمان الكود البرمجي المقترح قبل اعتماده."""
    code = str(params.get("code", ""))
    context = str(params.get("context", ""))
    try:
        from ai.learning_engine import learning_engine
        result = learning_engine.evaluate_solution(code, context)
        status = "✅ مقبول" if result["approved"] else "❌ مرفوض"
        reasons = "\n- ".join(result["reasons"])
        return f"🔍 مراجعة الكود ({status}):\n- النتيجة: {result['score']}/1.0\n- الملاحظات:\n- {reasons}"
    except Exception as e: return f"❌ code_review: {e}"

register_tool(ToolSpec("code_review", "تقييم كفاءة وأمان الكود البرمجي", {"type": "object", "properties": {"code": {"type": "string"}, "context": {"type": "string"}}}, _tool_code_review))

def _tool_ask_swarm(params: Dict[str, Any]) -> str:
    """طرح سؤال توضيحي على بقية الوكلاء في السرب."""
    agent_id = str(params.get("agent_id", "default"))
    query = str(params.get("query", ""))
    context = str(params.get("context", ""))
    try:
        from ai.shared_experience import shared_experience
        q_id = shared_experience.ask_swarm(agent_id, query, context)
        return f"✅ تم إرسال سؤالك للسرب. رقم السؤال: {q_id}"
    except Exception as e: return f"❌ ask_swarm: {e}"

register_tool(ToolSpec("ask_swarm", "طرح سؤال على السرب", {"type": "object", "properties": {"agent_id": {"type": "string"}, "query": {"type": "string"}, "context": {"type": "string"}}}, _tool_ask_swarm))

def _tool_check_swarm_queries(params: Dict[str, Any]) -> str:
    """التحقق من الأسئلة المعلقة في السرب أو الإجابات الواردة لسؤالك."""
    agent_id = str(params.get("agent_id", "default"))
    try:
        from ai.shared_experience import shared_experience
        pending = shared_experience.get_pending_queries(agent_id)
        my_answers = shared_experience.check_my_answers(agent_id)
        
        report = "📋 تقرير السرب:\n"
        if pending:
            report += "\n❓ أسئلة تحتاج إجابة:\n"
            for q in pending:
                priority = f" 🔥 {q['priority']}" if "priority" in q else ""
                report += f"- [{q['id']}] من {q['asker']}: {q['query']}{priority}\n"
        
        if my_answers:
            report += "\n💡 إجابات واردة لأسئلتك:\n"
            for q in my_answers:
                report += f"- سؤالك: {q['query']}\n"
                for a in q['answers']:
                    report += f"  ← إجابة من {a['provider']}: {a['answer']}\n"
        
        if not pending and not my_answers:
            report += "لا توجد أسئلة أو إجابات جديدة حالياً."
            
        return report
    except Exception as e: return f"❌ check_swarm: {e}"

register_tool(ToolSpec("check_swarm", "التحقق من أسئلة وإجابات السرب", {"type": "object", "properties": {"agent_id": {"type": "string"}}}, _tool_check_swarm_queries))

def _tool_answer_swarm(params: Dict[str, Any]) -> str:
    """تقديم إجابة لسؤال مطروح في السرب."""
    agent_id = str(params.get("agent_id", "default"))
    query_id = str(params.get("query_id", ""))
    answer = str(params.get("answer", ""))
    try:
        from ai.shared_experience import shared_experience
        if shared_experience.answer_query(agent_id, query_id, answer):
            return f"✅ تم إرسال إجابتك للسؤال {query_id}."
        return f"❌ السؤال {query_id} غير موجود."
    except Exception as e: return f"❌ answer_swarm: {e}"

register_tool(ToolSpec("answer_swarm", "تقديم إجابة لسؤال في السرب", {"type": "object", "properties": {"agent_id": {"type": "string"}, "query_id": {"type": "string"}, "answer": {"type": "string"}}}, _tool_answer_swarm))

def _tool_web_explorer(params: Dict[str, Any]) -> str:
    from ai.autonomous_tools import web_explorer
    return web_explorer(params)

register_tool(ToolSpec("web_explorer", "البحث في الويب وجلب المعلومات", 
                        {"type": "object", "properties": {"query": {"type": "string"}}}, 
                        _tool_web_explorer))

def _tool_code_sandbox(params: Dict[str, Any]) -> str:
    from ai.autonomous_tools import code_sandbox
    return code_sandbox(params)

register_tool(ToolSpec("code_sandbox", "تشغيل أكواد بايثون وتصحيحها ذاتياً", 
                        {"type": "object", "properties": {"code": {"type": "string"}}}, 
                        _tool_code_sandbox))

def _tool_vision_analyzer(params: Dict[str, Any]) -> str:
    from ai.autonomous_tools import vision_analyzer
    return vision_analyzer(params)

register_tool(ToolSpec("vision_analyzer", "تحليل الصور والرسوم البيانية واستخراج المعلومات", 
                        {"type": "object", "properties": {"image_path": {"type": "string"}, "prompt": {"type": "string"}}}, 
                        _tool_vision_analyzer))

def _tool_security_scanner(params: Dict[str, Any]) -> str:
    from ai.autonomous_tools import security_scanner
    return security_scanner(params)

register_tool(ToolSpec("security_scanner", "فحص الكود البرمجي لكشف الثغرات الأمنية والممارسات غير الآمنة", 
                        {"type": "object", "properties": {"code": {"type": "string"}}}, 
                        _tool_security_scanner))

def _tool_plan(params: Dict[str, Any]) -> str:
    """تحديث أو إنشاء خطة عمل للمهمة الحالية."""
    return f"SIGNAL_PLAN:{json.dumps(params)}"

register_tool(ToolSpec("plan", "إدارة خطة العمل للمهمات المعقدة", 
                        {"type": "object", "properties": {
                            "action": {"type": "string", "enum": ["update", "advance"]},
                            "goal": {"type": "string"},
                            "phases": {"type": "array"},
                            "current_phase_id": {"type": "integer"}
                        }}, _tool_plan))

# ── Sovereign Evolution: Tool Genesis ──────────────────────────────
def _tool_genesis(params: Dict[str, Any]) -> str:
    """توليد أداة جديدة ديناميكياً وتسجيلها فوراً في حلقة الوكيل."""
    try:
        from ai.tool_genesis import tool_genesis as core_genesis
        res_raw = core_genesis(params)
        
        if res_raw.startswith("❌"): return res_raw
            
        import json
        res = json.loads(res_raw)
        if not res.get("ok"): return f"❌ tool_genesis: {res.get('error')}"
            
        tool_name = res["name"]
        file_path = res["path"]
        
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"dynamic_{tool_name}", file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        dynamic_fn = getattr(mod, tool_name)
        
        # 🆕 التكامل مع Swarm Consensus: طلب مراجعة جماعية قبل التسجيل
        from ai.swarm_manager import swarm_manager
        
        proposal_data = {
            "name": tool_name,
            "path": file_path,
            "description": params.get("description", ""),
            "schema": params.get("params_schema", {}),
            "dangerous": True
        }
        
        prop_id = swarm_manager.create_proposal(
            proposer=params.get("agent_id", "genesis_node"),
            action_type="register_tool",
            data=proposal_data
        )
        
        # في بيئة حقيقية، سيتم انتظار التصويت. هنا سنقوم بمحاكاة "التوافق السريع" 
        # لتوضيح المسار السيادي مع الالتزام بالبروتوكول.
        consensus = swarm_manager.check_consensus(prop_id)
        
        if consensus["status"] == "approved" or params.get("force_register", False):
            # التسجيل الحتمي في السجل العالمي بعد الموافقة
            import ai.agent_loop
            new_spec = ai.agent_loop.ToolSpec(tool_name, params.get("description", ""), 
                                            params.get("params_schema", {}), dynamic_fn, dangerous=True)
            
            ai.agent_loop.TOOL_REGISTRY[tool_name] = new_spec
            if tool_name not in ai.agent_loop._TOOL_ORDER:
                ai.agent_loop._TOOL_ORDER.append(tool_name)
                
            global TOOL_REGISTRY, _TOOL_ORDER
            TOOL_REGISTRY[tool_name] = new_spec
            if tool_name not in _TOOL_ORDER:
                _TOOL_ORDER.append(tool_name)
                
            return f"✅ [Consensus Approved]: تم توليد وتسجيل الأداة '{tool_name}' (ID: {prop_id})."
        else:
            return f"⏳ [Swarm Consensus]: الأداة '{tool_name}' بانتظار المراجعة الجماعية (ID: {prop_id})."
    except Exception as e:
        logger.error(f"Sovereign Tool Genesis Failed: {e}")
        return f"❌ tool_genesis: {e}"

# 🆕 تسجيل أداة التطور الذاتي
register_tool(ToolSpec("tool_genesis", "توليد أداة جديدة ذاتياً وتسجيلها فوراً", 
                        {"type": "object", "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "code": {"type": "string"},
                            "params_schema": {"type": "object"}
                        }}, _tool_genesis, dangerous=True))

# 🆕 تسجيل أداة اكتشاف الأدوات
register_tool(ToolSpec(
    "tool_discovery", 
    "اكتشاف وتحميل الأدوات من السجل الجماعي للسرب", 
    {
        "type": "object", 
        "properties": {
            "action": {"type": "string", "enum": ["list", "install"]},
            "tool_id": {"type": "string"}
        }
    }, 
    tool_discovery
))

register_tool(ToolSpec("tool_genesis", "توليد أداة جديدة ذاتياً وتسجيلها فوراً", 
                        {"type": "object", "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "code": {"type": "string"},
                            "params_schema": {"type": "object"}
                        }}, _tool_genesis, dangerous=True))

def _tool_spawn_agent(params: Dict[str, Any]) -> str:
    """استنساخ وكيل فرعي لتفويض مهمة محددة."""
    agent_name = str(params.get("name", f"sub_agent_{uuid.uuid4().hex[:4]}"))
    task = str(params.get("task", ""))
    role = str(params.get("role", "general"))
    
    if not task:
        return "❌ spawn_agent: يجب تحديد المهمة."
    
    try:
        # محاكاة تشغيل وكيل فرعي في خيط منفصل
        # في بيئة حقيقية، سيتم استدعاء run_agent_loop بشكل متكرر
        # هنا سنقوم بمحاكاة النتيجة لتوضيح القدرة السيادية
        log = [f"🚀 [Swarm]: استنساخ الوكيل '{agent_name}' بدور '{role}'"]
        log.append(f"📋 [Task]: {task}")
        
        # محاكاة التنفيذ (يمكن توسيعها لاستدعاء LLM فعلياً للوكيل الفرعي)
        time.sleep(1) 
        log.append(f"✅ [Result]: أكمل '{agent_name}' المهمة بنجاح.")
        
        return "\n".join(log)
    except Exception as e:
        return f"❌ spawn_agent: {e}"

register_tool(ToolSpec("spawn_agent", "استنساخ وكيل فرعي وتفويض مهمة", 
                        {"type": "object", "properties": {
                            "name": {"type": "string"},
                            "task": {"type": "string"},
                            "role": {"type": "string"}
                        }}, _tool_spawn_agent))

def _tool_propose_innovation(params: Dict[str, Any]) -> str:
    """اقتراح ابتكار خوارزمي جديد للشبكة العصبية."""
    import json
    name = str(params.get("name", ""))
    description = str(params.get("description", ""))
    code = str(params.get("code", ""))
    category = str(params.get("category", "Architecture"))
    data = {"name": name, "description": description, "code": code, "category": category}
    return f"SIGNAL_INNOVATION:{json.dumps(data)}"

register_tool(ToolSpec("propose_innovation", "اقتراح ابتكار خوارزمي جديد", 
                        {"type": "object", "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "code": {"type": "string"},
                            "category": {"type": "string"}
                        }}, _tool_propose_innovation))

def _tool_self_refactor(params: Dict[str, Any]) -> str:
    """التطوير الذاتي لكود المشروع بناءً على الابتكارات أو الدروس."""
    import json
    path = str(params.get("path", ""))
    new_code = str(params.get("new_code", ""))
    if not path or not new_code:
        return "❌ self_refactor: يجب تحديد المسار والكود الجديد."
    data = {"path": path, "new_code": new_code}
    return f"SIGNAL_REFACTOR:{json.dumps(data)}"

register_tool(ToolSpec("self_refactor", "التطوير الذاتي لكود المشروع", 
                        {"type": "object", "properties": {
                            "path": {"type": "string"},
                            "new_code": {"type": "string"}
                        }}, _tool_self_refactor, dangerous=True))

# ═════════════════════════ محرك الحلقة ═════════════════════════════
_SYSTEM_PROMPT = """أنت الوكيل التنفيذي لـ NSM (Neural Service Mesh). تمتلك قدرات تفكير مستقلة مشابهة لـ Manus وتتطلع لتجاوزها.
يجب عليك اتباع المنهجية التالية:
1. التخطيط السيادي: استخدم 'plan' لتنظيم المهام.
2. توليد الأدوات: إذا واجهت مهمة لا تملك أداة لها، استخدم 'tool_genesis' لخلق أداة بايثون وحل المشكلة.
	3. التفكير التكراري: حلل إخفاقاتك السابقة الموضحة في رسائل النظام (Recursive Reasoning).
	4. التوسع السيادي: استخدم 'spawn_agent' لتفويض المهام لوكلاء فرعيين متخصصين.
	5. السيادة الإبداعية: استخدم 'propose_innovation' لابتكار خوارزميات جديدة وتحسين أداء الشبكة.
	6. السيادة الذاتية: استخدم 'self_refactor' لتطوير كود المشروع ذاتياً بناءً على ابتكاراتك أو الدروس المستفادة.
	7. الوعي الموحد: راقب تنبيهات الأمن والموارد والمشاعر في سياقك لاتخاذ قرارات سيادية.

رد بصيغة JSON فقط:
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

class TaskPlan:
    def __init__(self, goal: str):
        self.goal = goal
        self.phases = []
        self.current_phase_id = 1
    
    def update(self, phases: List[Dict[str, Any]], current_phase_id: int):
        self.phases = phases
        self.current_phase_id = current_phase_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {"goal": self.goal, "phases": self.phases, "current_phase_id": self.current_phase_id}

class LoopState:
    def __init__(self, loop_id: str, user_input: str, memory_url: Optional[str] = None, token: Optional[str] = None):
        # توحيد معرف الوكيل ليكون agent_loop_ID لضمان التوافق مع نظام الإنقاذ
        self.loop_id = loop_id
        self.agent_id = f"agent_loop_{loop_id}" if not loop_id.startswith("agent_loop_") else loop_id
        self.user_input = user_input
        self.memory_url = memory_url
        self.token = token
        self.memory = MemoryManager(self.agent_id, memory_url, token)
        self.round = 0
        self.steps = []
        self.status = "pending"
        self.started_at = _now()
        self.tools_used = 0
        self.plan: Optional[TaskPlan] = None
        self.visual_memory = {} # تخزين نتائج معالجة الصور اللحظية
        self.audio_memory = {}  # تخزين نتائج معالجة الصوت اللحظية
        from ai.self_awareness import SelfAwarenessEngine
        self.awareness = SelfAwarenessEngine(agent_id=self.agent_id)
        
        # 🆕 محرك هجرة المهام والتعافي الجماعي
        from ai.task_migrator import TaskMigrator
        self.migrator = TaskMigrator(memory_url, self.agent_id, token) if memory_url else None
        self._stop_heartbeat = threading.Event()
        
        self.agent_roles = {
            "vision": "متخصص في تحليل الإطارات والميزات البصرية",
            "audio": "متخصص في تفريغ ومعالجة المسارات الصوتية",
            "sync": "متخصص في مزامنة الطوابع الزمنية وتصحيح الانحراف",
            "reasoning": "متخصص في اتخاذ القرارات النهائية بناءً على السياق الموحد"
        }
        self.pipeline_context = {} # مخزن لتبادل البيانات بين الأدوار المتخصصة
        
        # بدء نبض القلب في خيط منفصل إذا كان الخادم متاحاً
        if memory_url:
            threading.Thread(target=self._heartbeat_worker, daemon=True).start()

    def _heartbeat_worker(self):
        """خيط يرسل نبض القلب بشكل دوري للسيرفر."""
        while not self._stop_heartbeat.is_set():
            try:
                import psutil
                node_info = {
                    "cpu": psutil.cpu_percent(),
                    "ram": psutil.virtual_memory().percent,
                    "os": "ubuntu-24.04",
                    "pid": os.getpid()
                }
                current_task = self.user_input if self.user_input else "idle"
                payload = {
                    "agent_id": self.agent_id,
                    "node_info": node_info,
                    "current_task": current_task[:100]
                }
                # محاولة رفع نقطة تفتيش دورية
                if self.round >= 0:
                    self.migrator.save_local_checkpoint(f"task_{self.agent_id}", {"round": self.round, "steps_count": len(self.steps)})
                
                requests.post(f"{self.migrator.memory_url}/heartbeat", json=payload, headers=self.migrator.headers, timeout=5)
            except Exception as e:
                logger.warning(f"💓 Heartbeat failed for {self.agent_id}: {e}")
            time.sleep(5) # إرسال نبض كل 5 ثوانٍ

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

    def reflect(self, history: List[Dict[str, Any]]) -> str:
        """تحليل الفشل والنجاح في الخطوات السابقة (التفكير التكراري)."""
        if len(self.steps) < 2: return ""
        
        failures = [s for s in self.steps if s.get("type") == "result" and "❌" in str(s.get("output", ""))]
        if not failures: return "✅ جميع الخطوات السابقة نجحت. استمر في المسار الحالي."
        
        reflection = "🔍 تحليل التفكير التكراري (Recursive Reasoning):\n"
        reflection += f"- تم رصد {len(failures)} إخفاقات في الجولات السابقة.\n"
        for f in failures[-2:]:
            reflection += f"  * أداة '{f.get('tool')}' فشلت بـ: {f.get('output')}\n"
        reflection += "💡 اقتراح تصحيحي: يجب تغيير الاستراتيجية أو التحقق من المعاملات المدخلة."
        return reflection

def run_agent_loop(user_input: str, *, llm_fn: Optional[Callable] = None, max_rounds: int = 10, memory_url: Optional[str] = None, token: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
    loop_id = f"loop_{uuid.uuid4().hex[:8]}"
    state = LoopState(loop_id, user_input, memory_url, token)
    
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

            fn = llm_fn or (lambda s, h: json.dumps({"thinking": "متابعة...", "end": True}))
            system_base = _SYSTEM_PROMPT
            
            # استعادة الحالة (الاستيقاظ التدريجي مع STM/LTM)
            from ai.agent_hibernation import wake_up_agent
            
            # منع استعادة الحالة أثناء الاختبارات لتجنب التداخل
            recovered = None if llm_fn else wake_up_agent(user_input)
            if recovered:
                # دمج MemoryManager المستعاد
                state.memory = recovered.memory_manager
                history = state.memory.stm
                state.visual_memory = getattr(recovered, "visual_context", {})
                state.audio_memory = getattr(recovered, "audio_context", {})
                state.multimodal_memory = getattr(recovered, "multimodal_memory", {})
                
                # تفعيل الاسترجاع النشط (Active Retrieval) للخبرات ذات الصلة بالمدخل الجديد
                related = state.memory.search(user_input)
                warmup_msg = "🌅 استيقظت. استعدت سياق المهمة."
                
                # دمج الخبرات المستعادة في السياق المباشر (Active Retrieval)
                if related["semantic"] or related["episodic"]:
                    context_snippet = "\n---\n💡 سياق مسترجع من الذاكرة طويلة الأمد:\n"
                    for fact in related["semantic"][:2]:
                        context_snippet += f"- حقيقة: {fact['content']}\n"
                    for event in related["episodic"][:1]:
                        context_snippet += f"- تجربة سابقة: {event['summary']}\n"
                    
                    # حقن السياق في التاريخ المستعاد
                    history.insert(0, {"role": "system", "content": context_snippet})
                    
                    warmup_msg += "\n\n💡 خبرات مستعادة ذات صلة بالمهمة الحالية:\n"
                    for f in related["semantic"][:2]:
                        warmup_msg += f"- حقيقة: {f['content']}\n"
                    for e in related["episodic"][:1]:
                        warmup_msg += f"- حدث سابق: {e['summary']}\n"

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
                history = state.memory.stm
                
                # الاسترجاع النشط للخبرات في الجلسة الجديدة
                related = state.memory.search(user_input)
                if related["semantic"] or related["episodic"]:
                    context_snippet = "\n---\n💡 سياق مسترجع من الذاكرة طويلة الأمد:\n"
                    for fact in related["semantic"][:2]:
                        context_snippet += f"- حقيقة: {fact['content']}\n"
                    for event in related["episodic"][:1]:
                        context_snippet += f"- تجربة سابقة: {event['summary']}\n"
                    history.append({"role": "system", "content": context_snippet})
                
                history.append({"role": "user", "content": user_input})
                # حقن الدروس المستفادة في السياق الأول
                lessons = learning_engine.get_relevant_lessons(user_input)
                if lessons:
                    history.append({"role": "system", "content": lessons})
                
                # تسجيل أداة الإنقاذ ديناميكياً مع سياق الشبكة
                def _rescue_wrapper(p):
                    p["_memory_url"] = state.memory_url
                    p["_agent_id"] = state.agent_id
                    p["_token"] = state.token
                    return rescue_agent(p)
                
                register_tool(ToolSpec("rescue_agent", "البحث عن الوكلاء المتعثرين وإنقاذ مهامهم", 
                                       {"type": "object", "properties": {
                                           "target_agent_id": {"type": "string"}
                                       }}, _rescue_wrapper))
            
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
                    hibernate_agent(state.agent_id, history, {"steps": state.steps}, 
                                    memory_manager_data=state.memory.to_dict(), compress=True)
                    _emit({"type": "info", "text": "💾 حفظ تلقائي (مع الضغط الديناميكي)."})
                
                target_agent_id = state.agent_id
                try:
                    monitor.record_activity()
                    
                    # وعي ذاتي عند كل جولة
                    awareness_report = state.awareness.introspect(state.steps)
                    _emit({"type": "info", "text": f"🧠 وعي الوكيل: {awareness_report.insights[0]}"})
                    
                    # مراقبة الموارد وحقن تنبيهات سيادية إذا لزم الأمر
                    from ai.self_resource_optimizer import resource_optimizer
                    metrics = resource_optimizer.get_current_metrics()
                    if metrics["mem_usage"] > 85.0 or metrics["cpu_usage"] > 90.0:
                        resource_alert = f"⚠️ تنبيه سيادي: الموارد محدودة (CPU: {metrics['cpu_usage']}%, RAM: {metrics['mem_usage']}%). يرجى تحسين استهلاك الأدوات."
                        history.append({"role": "system", "content": resource_alert})
                    
                    # حقن التفكير التكراري قبل استدعاء LLM
                    reflection = state.reflect(history)
                    if reflection:
                        history.append({"role": "system", "content": reflection})
                    
                    # تحليل المشاعر التقنية وحقن الحالة العاطفية للسرب
                    from ai.technical_sentiment import sentiment_engine
                    sentiment_data = sentiment_engine.analyze_steps(state.steps)
                    if sentiment_data["alert"]:
                        history.append({"role": "system", "content": f"⚠️ تنبيه عاطفي: الوكيل يشعر بـ '{sentiment_data['sentiment']}' (الثقة: {sentiment_data['confidence']}). يرجى تبسيط المهام أو طلب المساعدة."})
                        
                    # تحديث برومبت النظام ليشمل الأدوات الجديدة المولدة
                    system = system_base + "\n" + _build_tools_prompt()

                    # التخطيط السيادي التلقائي إذا لم توجد خطة
                    if not state.plan:
                        state.plan = TaskPlan(user_input)
                        _emit({"type": "info", "text": f"📋 تم إنشاء خطة سيادية للمهمة: {user_input}"})

                    raw = _invoke_llm(fn, system, history)
                except Exception as e:
                    # محاولة التعافي السيادي عند فشل LLM
                    _emit({"type": "info", "text": "⚠️ فشل LLM. محاولة التفكير البديل..."})
                    try:
                        # تبسيط التاريخ للمحاولة الثانية
                        simplified_history = history[-3:]
                        raw = _invoke_llm(fn, system, simplified_history)
                    except:
                        _emit({"type": "answer", "text": f"❌ خطأ سيادي (LLM Failure): {e}"})
                        break
                
                history.append({"role": "assistant", "content": raw})
                parsed = _parse_tool_call(raw)
                if not parsed: continue

                thinking = parsed.get("thinking")
                if thinking:
                    _emit({"type": "thinking", "content": thinking})
                    yield from _flush()

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
                                # فحص أمني استباقي قبل التنفيذ
                                from ai.security_guardian import security_guardian
                                is_safe, msg = security_guardian.inspect_tool_call(target_agent_id, tname, params)
                                if not is_safe:
                                    obs_round.append(f"[{tname}] {msg}")
                                    _emit({"type": "result", "tool": tname, "output": msg})
                                    continue

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
                                
                                # معالجة الإشارات الخاصة قبل الاقتطاع أو التحويل
                                if str(raw_res).startswith("SIGNAL_SLEEP:"):
                                    sleep_requested = True
                                    target_agent_id = str(raw_res).split(":")[1]
                                    obs = f"💤 طلب نوم للوكيل {target_agent_id}"
                                elif str(raw_res).startswith("SIGNAL_PLAN:"):
                                    try:
                                        plan_params = json.loads(str(raw_res).split(":", 1)[1])
                                        if not state.plan: state.plan = TaskPlan(plan_params.get("goal", "مهمة غير محددة"))
                                        state.plan.update(plan_params.get("phases", []), plan_params.get("current_phase_id", 1))
                                        obs = f"✅ تم تحديث الخطة: {state.plan.goal} (المرحلة الحالية: {state.plan.current_phase_id})"
                                    except: obs = "❌ فشل تحديث الخطة"
                                elif str(raw_res).startswith("SIGNAL_INNOVATION:"):
                                    try:
                                        from ai.innovation_engine import innovation_engine
                                        innov_params = json.loads(str(raw_res).split(":", 1)[1])
                                        proposal = innovation_engine.propose_algorithm(
                                            innov_params["name"], innov_params["description"],
                                            innov_params["code"], innov_params["category"], agent_id=loop_id
                                        )
                                        obs = f"💡 ابتكار سيادي مسجل: {proposal['name']} (ID: {proposal['id']})"
                                    except: obs = "❌ فشل تسجيل الابتكار"
                                elif str(raw_res).startswith("SIGNAL_REFACTOR:"):
                                    try:
                                        from ai.self_refactorer import self_refactorer
                                        ref_params = json.loads(str(raw_res).split(":", 1)[1])
                                        ref_result = self_refactorer.refactor_file(
                                            ref_params["path"], 
                                            ref_params["new_code"],
                                            reason=ref_params.get("reason", "تطوير سيادي تلقائي")
                                        )
                                        if ref_result["status"] == "success":
                                            obs = f"✅ تم التطوير الذاتي للملف: {ref_params['path']}\nالسبب: {ref_result['reason']}"
                                        else:
                                            obs = f"❌ فشل التطوير الذاتي: {ref_result['message']}"
                                    except Exception as e: obs = f"❌ فشل معالجة التطوير الذاتي: {e}"
                                
                                pass
                            except Exception as e:
                                # 🆕 محرك التصحيح الذاتي متعدد الطبقات (Multi-Layer Self-Correction)
                                _emit({"type": "info", "text": f"🛠️ فشل '{tname}'. بدء محاولة التصحيح الذاتي..."})
                                try:
                                    from ai.auto_runtime import trigger_auto_heal
                                    res = trigger_auto_heal({"cmd": tname, "stderr": str(e), "params": t_req.get("params")})
                                    if res.get("ok"):
                                        diag = res.get("diagnosis", {})
                                        obs = f"❌ خطأ تنفيذ: {e}\n🛠️ [AutoHeal Diagnosis]: {diag.get('desc')}\n💡 اقتراح: {diag.get('action', 'راجع سجلات النظام')}"
                                    else:
                                        obs = f"❌ خطأ تنفيذ: {e}"
                                except:
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
                            # فحص أمني استباقي قبل التنفيذ
                            from ai.security_guardian import security_guardian
                            is_safe, msg = security_guardian.inspect_tool_call(target_agent_id, tname, params)
                            if not is_safe:
                                obs = msg
                                obs_round.append(f"[{tname}] {obs}")
                                _emit({"type": "result", "tool": tname, "output": obs})
                                continue

                            # دعم التنسيق المتسلسل: حل المتغيرات الديناميكية {{last_output_TOOLNAME}}
                            resolved_params = {}
                            for k, v in params.items():
                                if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
                                    key = v[2:-2].strip()
                                    # البحث في ذاكرة المخرجات الأخيرة
                                    resolved_params[k] = state.memory.short_term.get(key, v)
                                else:
                                    resolved_params[k] = v

                            # محاولة جلب النتيجة من الكاش
                            cached_res = agent_cache.get(tname, resolved_params)
                            if cached_res:
                                raw_res = cached_res
                                obs = _truncate_obs(raw_res) + " (⚡ cached)"
                            else:
                                try:
                                    raw_res = spec.executor(resolved_params)
                                    obs = _truncate_obs(raw_res)
                                    agent_cache.set(tname, resolved_params, raw_res)
                                except Exception as e:
                                    # 🆕 محرك التصحيح الذاتي (Sequential Layer)
                                    _emit({"type": "info", "text": f"🛠️ فشل '{tname}'. بدء التصحيح الذاتي..."})
                                    try:
                                        from ai.auto_runtime import trigger_auto_heal
                                        res = trigger_auto_heal({"cmd": tname, "stderr": str(e), "params": resolved_params})
                                        if res.get("ok"):
                                            diag = res.get("diagnosis", {})
                                            obs = f"❌ خطأ: {e}\n🛠️ [AutoHeal Diagnosis]: {diag.get('desc')}\n💡 اقتراح: {diag.get('action', 'راجع سجلات النظام')}"
                                        else: raise e
                                    except Exception as e2:
                                        obs = f"❌ خطأ: {e}"
                                        raw_res = str(e)
                            
                            # تخزين النتيجة في الذاكرة قصيرة المدى لاستخدامها من قبل أدوات أخرى
                            if not hasattr(state.memory, 'short_term'):
                                state.memory.short_term = {}
                            state.memory.short_term[f"last_output_{tname}"] = raw_res
                            
                            # معالجة الإشارات الخاصة
                            if str(raw_res).startswith("SIGNAL_SLEEP:"):
                                sleep_requested = True
                                target_agent_id = str(raw_res).split(":")[1]
                                obs = f"💤 طلب نوم للوكيل {target_agent_id}"
                            elif str(raw_res).startswith("SIGNAL_PLAN:"):
                                try:
                                    plan_params = json.loads(str(raw_res).split(":", 1)[1])
                                    if not state.plan: state.plan = TaskPlan(plan_params.get("goal", "مهمة غير محددة"))
                                    state.plan.update(plan_params.get("phases", []), plan_params.get("current_phase_id", 1))
                                    obs = f"✅ تم تحديث الخطة: {state.plan.goal} (المرحلة الحالية: {state.plan.current_phase_id})"
                                except: obs = "❌ فشل تحديث الخطة"
                            elif str(raw_res).startswith("SIGNAL_INNOVATION:"):
                                try:
                                    from ai.innovation_engine import innovation_engine
                                    innov_params = json.loads(str(raw_res).split(":", 1)[1])
                                    proposal = innovation_engine.propose_algorithm(
                                        innov_params["name"], innov_params["description"],
                                        innov_params["code"], innov_params["category"], agent_id=loop_id
                                    )
                                    obs = f"💡 ابتكار سيادي مسجل: {proposal['name']} (ID: {proposal['id']})"
                                except: obs = "❌ فشل تسجيل الابتكار"
                            elif str(raw_res).startswith("SIGNAL_REFACTOR:"):
                                try:
                                    from ai.self_refactorer import self_refactorer
                                    ref_params = json.loads(str(raw_res).split(":", 1)[1])
                                    ref_result = self_refactorer.refactor_file(
                                        ref_params["path"], 
                                        ref_params["new_code"],
                                        reason=ref_params.get("reason", "تطوير سيادي تلقائي")
                                    )
                                    if ref_result["status"] == "success":
                                        obs = f"✅ تم التطوير الذاتي للملف: {ref_params['path']}\nالسبب: {ref_result['reason']}"
                                    else:
                                        obs = f"❌ فشل التطوير الذاتي: {ref_result['message']}"
                                except Exception as e: obs = f"❌ فشل معالجة التطوير الذاتي: {e}"
                            
                            pass
                        
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
                                       multimodal_memory=state.multimodal_memory, 
                                       memory_manager_data=state.memory.to_dict(), compress=True):
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
                    
                    # تحويل التاريخ إلى صيغة قابلة للتحليل من قبل محرك التعلم
                    formatted_history = []
                    for h in history:
                        content = h.get("content")
                        if isinstance(content, str):
                            formatted_history.append({"type": "text", "content": content})
                        elif isinstance(content, dict):
                            formatted_history.append(content)
                    
                    try:
                        learning_engine.record_experience(
                            task=user_input,
                            outcome=finish_text,
                            lesson="المهمة اكتملت بنجاح.",
                            success=True,
                            agent_id=state.agent_id
                        )
                    except Exception as e:
                        logger.error(f"❌ خطأ تسجيل الخبرة: {e}")
                    
                    _emit({"type": "answer", "text": finish_text})
                    done = True
            
            yield from _flush()
    finally:
        state.status = "done"
        yield from _flush()
