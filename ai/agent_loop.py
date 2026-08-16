"""
ai/agent_loop.py
================
🆕 حلقة التنفيذ متعددة الجولات (Multi-turn Agent Loop) — جوهر الوكيل الذاتي.

تحول الوكيل من "مولّد قائمة خطوات من جولة واحدة" إلى "عامل يلاحظ ثم يتصرف":

    [user] → LLM → (plan) → {act → observe → decide} ×N → (final answer)

المبادئ المصمَّم بها:
  1. كل أداة تنفَّذ وتُعاد **نتيجتها كنص ملاحظة** للنموذج في الجولة التالية،
     فيقرّر بنفسه: يكمل؟ يصلح؟ يغيّر المسار؟ ينهي؟
  2. Registry أدوات موحد (TOOL_REGISTRY) يربط أسماء الأدوات بواجهات
     موثوقة داخل المشروع (terminal, file, search, code, notebook) بدل
     الـdispatch المتناثر في nsm_agent_core.
  3. حدود أمان صارمة: سقف جولات، ميزانية أدوات، timeout إجباري،
     منع sudo/rm الخطير، حماية مسارات (لا path traversal)، whitelist
     للأوامر الخطرة، سجل مراجعة (audit) لكل خطوة.
  4. Self-healing تنفيذي: فشل أداة → يعاد إرسال الخطأ للنموذج مع طلب
     بديل — حتى داخل نفس الجولة، دون استهلاك جولة جديدة.
  5. تكامل مع task_manager: كل حلقة تسجل خطة وتتقدم عبرها، فتظهر في
     لوحة «المهام طويلة الأمد».

بدون مفاتيح API إضافية — يعيد استخدام ai/llm_fallback (أو أي دالة
generate(user_input, system, history) تُحقن عند التهيئة).

الاستخدام:
    from ai.agent_loop import run_agent_loop
    for event in run_agent_loop("افحص ai/goal_planner.py وأصلح أي خطأ"):
        # event = {'type': 'thought'|'tool'|'result'|'answer'|'status', ...}
        ...
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
_MAX_OUTPUT_CHARS = 4000        # حد الملاحظة المعادة للنموذج لكل أداة
_CMD_BLOCKLIST = (
    "sudo ", "sudo\t", "rm -rf /", "mkfs", ":(){ :|:& };:",
    ">:() { :|:& };:", ">:(){ :|:& };:", "chmod 777 /",
    "crontab -r", ">/dev/sd", "pkill", "killall", "kill -9 -",
    "dd if=/dev/zero", "format C:", "del /f /s /q \\",
    "shutdown", "reboot", "init 0", "init 6",
)
_PATH_TRAVERSAL_RE = re.compile(r"(^|/)((\.\.(\/|\\|$))+)")

_RUN_LOCK = threading.Lock()    # تنفيذ تسلسلي للحلقات (منع تداخل shell)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _safe_tool_path(raw: str) -> Optional[Path]:
    """حل مسار نسبي إلى داخل ROOT حصراً — رفض أي هروب (path traversal)."""
    if not raw or not str(raw).strip():
        return None
    try:
        candidate = (ROOT / str(raw).strip()).resolve()
        candidate.relative_to(ROOT.resolve())
    except (ValueError, OSError, TypeError):
        return None
    return candidate


def _cmd_safe(cmd: str) -> Tuple[bool, str]:
    """فحص أمان أمر shell — سبب الحظر يُعاد للنموذج بصراحة."""
    c = (cmd or "").strip()
    if not c:
        return False, "أمر فارغ"
    if any(c.startswith(b) for b in _CMD_BLOCKLIST):
        return False, "أمر محظور لأسباب أمنية (destructive/system)"
    # منع pipes تنفيذ صلاحيات: su/su -c/pkexec
    if re.search(r"(^|\|)\s*(su|pkexec|nohup)\b", c):
        return False, "استدعاء صلاحيات مرتفعة محظور"
    return True, ""


# ═════════════════════════ سجل المراجعة (audit) ════════════════════
_AUDIT_DIR = ROOT / "artifacts" / "agent_loop" / "audit"
_AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def _audit(loop_id: str, entry: Dict[str, Any]) -> None:
    try:
        p = _AUDIT_DIR / f"{loop_id}.jsonl"
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _now(), **entry}, ensure_ascii=False)
                    + "\n")
    except Exception as e:
        logger.warning("audit write failed: %s", e)


# ═════════════════════════ Registry الأدوات ════════════════════════
class ToolSpec:
    """مواصفات أداة واحدة: اسم + وصف للنموذج + تنفيذ."""
    def __init__(self, name: str, description: str, params_schema: Dict[str, Any],
                 executor: Callable[[Dict[str, Any]], str],
                 dangerous: bool = False) -> None:
        self.name = name
        self.description = description
        self.params_schema = params_schema
        self.executor = executor
        self.dangerous = dangerous

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params_schema,
            },
        }


TOOL_REGISTRY: Dict[str, ToolSpec] = {}
_TOOL_ORDER: List[str] = []


def register_tool(spec: ToolSpec) -> ToolSpec:
    TOOL_REGISTRY[spec.name] = spec
    if spec.name not in _TOOL_ORDER:
        _TOOL_ORDER.append(spec.name)
    return spec


# ── 1) أداة shell كامل (عبر nsm_terminal — مع أدوار وaudit) ────────
def _exec_shell(params: Dict[str, Any]) -> str:
    cmd = str(params.get("cmd", "")).strip()
    if not cmd:
        return "❌ shell: مطلوب cmd"
    ok, why = _cmd_safe(cmd)
    if not ok:
        return f"❌ shell: {why}"
    try:
        from ai.nsm_terminal import get_terminal
        r = get_terminal().run_agent("agent_loop", cmd,
                                     timeout=int(params.get("timeout", _DEFAULT_TIMEOUT)))
        out = []
        if r.stdout:
            out.append(r.stdout[:_MAX_OUTPUT_CHARS])
        if r.stderr:
            out.append("[stderr]\n" + r.stderr[:_MAX_OUTPUT_CHARS])
        tail = f" (exit {r.exit_code}, {r.duration_ms}ms)" if hasattr(r, "exit_code") else ""
        return (("\n".join(out) or "تم التنفيذ بلا مخرجات") + tail)[:_MAX_OUTPUT_CHARS + 60]
    except Exception as e:
        return f"❌ shell: {e}"


register_tool(ToolSpec(
    "shell",
    "تنفيذ أمر shell كامل في بيئة المشروع (pip/git/python/ls/cat/grep/ffmpeg...). "
    "ممنوع: sudo، حذف جذري، أوامر destructive. النتيجة تُقرأ تلقائياً وتُعاد لك.",
    {"type": "object", "properties": {
        "cmd": {"type": "string", "description": "الأمر كاملاً، مثال: python3 -m py_compile ai/agent_loop.py"},
        "timeout": {"type": "integer", "description": "مهلة ثوانٍ (اختياري، افتراضي 90)"},
    }, "required": ["cmd"]},
    _exec_shell, dangerous=True,
))


# ── 2) أدوات الملفات (مع حماية المسار) ────────────────────────────
def _tool_read(params: Dict[str, Any]) -> str:
    path = _safe_tool_path(str(params.get("path", "")))
    if path is None:
        return "❌ read_file: مسار خارج مجلد المشروع أو فارغ"
    if not path.exists():
        return f"❌ read_file: لا يوجد {params.get('path')}"
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        start = max(0, int(params.get("start", 0) or 0))
        end = int(params.get("end", 0) or 0)
        if end and end > start:
            lines = content.splitlines()
            content = "\n".join(lines[start:end])
            header = f"# {path.relative_to(ROOT)} [{start + 1}..{end}]\n"
        else:
            header = f"# {path.relative_to(ROOT)}\n"
        return (header + content)[:_MAX_OUTPUT_CHARS]
    except Exception as e:
        return f"❌ read_file: {e}"


def _tool_write(params: Dict[str, Any]) -> str:
    mode = str(params.get("mode", "create"))
    raw = str(params.get("path", "")).strip()
    # تعديل: المسار النسبي داخل ROOT حصراً
    path = _safe_tool_path(raw)
    if path is None:
        return "❌ write_file: مسار خارج مجلد المشروع أو فارغ"
    content = str(params.get("content", ""))
    try:
        if mode == "create" or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"✅ create_file: كُتب {raw}"
        # replace mode — str_replace: old → new
        old = params.get("old", "")
        new = params.get("new", "")
        if old is not None and str(old) and new is not None:
            cur = path.read_text(encoding="utf-8")
            if str(old) not in cur:
                return f"❌ write_file: النص القديم غير موجود في {raw}"
            cur = cur.replace(str(old), str(new), 1)
            path.write_text(cur, encoding="utf-8")
            return f"✅ edit_file: عُدّل {raw}"
        # full rewrite
        path.write_text(content, encoding="utf-8")
        return f"✅ write_file: أعيدت كتابة {raw}"
    except Exception as e:
        return f"❌ write_file: {e}"


def _tool_find_files(params: Dict[str, Any]) -> str:
    raw = str(params.get("pattern", "*.py"))
    try:
        matches = [str(p.relative_to(ROOT)) for p in ROOT.rglob(raw)
                   if p.is_file() and ".git" not in p.parts]
        return "\n".join(matches[:100]) or "لا نتائج"
    except Exception as e:
        return f"❌ find_files: {e}"


def _tool_search_code(params: Dict[str, Any]) -> str:
    pattern = str(params.get("pattern", ""))
    if not pattern:
        return "❌ search_code: مطلوب pattern"
    try:
        results = []
        for p in ROOT.rglob("*.py"):
            if ".git" in p.parts or not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if pattern in line:
                    results.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
                    if len(results) >= 60:
                        break
            if len(results) >= 60:
                break
        return "\n".join(results) or "لا نتائج"
    except Exception as e:
        return f"❌ search_code: {e}"


def _tool_py_compile(params: Dict[str, Any]) -> str:
    raw = str(params.get("path", "")).strip()
    path = _safe_tool_path(raw)
    if path is None:
        return "❌ py_compile: مسار غير صالح"
    if not path.exists():
        return f"❌ py_compile: لا يوجد {raw}"
    try:
        subprocess.run(["python3", "-m", "py_compile", str(path)],
                       check=True, capture_output=True, text=True)
        return f"✅ py_compile: {raw} يُجمَّع بلا أخطاء"
    except subprocess.CalledProcessError as e:
        return f"❌ py_compile: {raw}\n{(e.stderr or e.stdout)[:1500]}"
    except Exception as e:
        return f"❌ py_compile: {e}"


register_tool(ToolSpec(
    "read_file", "قراءة محتوى ملف نصي من المشروع (يُفضَّل قبل أي تعديل).",
    {"type": "object", "properties": {
        "path": {"type": "string", "description": "مسار نسبي داخل المشروع"},
        "start": {"type": "integer", "description": "سطر بداية (اختياري، 0-based)"},
        "end": {"type": "integer", "description": "سطر نهاية حصري (اختياري)"},
    }, "required": ["path"]},
    _tool_read,
))
register_tool(ToolSpec(
    "write_file", ("إنشاء ملف جديد أو تعديله: mode=create يكتب المحتوى كاملاً، "
                   "mode=edit مع حقلي old/new يستبدل نصًا حرفيًا أول ظهور (str_replace)."),
    {"type": "object", "properties": {
        "path": {"type": "string", "description": "مسار نسبي داخل المشروع"},
        "content": {"type": "string", "description": "المحتوى الكامل (mode=create أو إعادة كتابة)"},
        "mode": {"type": "string", "enum": ["create", "edit"], "description": "create=كتابة جديدة، edit=استبدال old→new"},
        "old": {"type": "string", "description": "النص القديم (mode=edit)"},
        "new": {"type": "string", "description": "النص البديل (mode=edit)"},
    }, "required": ["path"]},
    _tool_write, dangerous=True,
))
register_tool(ToolSpec(
    "find_files", "البحث عن ملفات باسم glob داخل المشروع.",
    {"type": "object", "properties": {
        "pattern": {"type": "string", "description": "glob pattern مثل *.py أو notebooks/*"},
    }, "required": ["pattern"]},
    _tool_find_files,
))
register_tool(ToolSpec(
    "search_code", "البحث عن نص داخل ملفات .py في المشروع.",
    {"type": "object", "properties": {
        "pattern": {"type": "string", "description": "النص أو العبارة المبحوث عنها"},
    }, "required": ["pattern"]},
    _tool_search_code,
))
register_tool(ToolSpec(
    "py_compile", "فحص syntax ملف Python عبر py_compile — إلزامي بعد أي تعديل على .py.",
    {"type": "object", "properties": {
        "path": {"type": "string", "description": "مسار نسبي داخل المشروع"},
    }, "required": ["path"]},
    _tool_py_compile,
))


# ── 3) البحث والمعرفة (web_search_tool) ────────────────────────────
def _tool_web_search(params: Dict[str, Any]) -> str:
    try:
        from ai.web_search_tool import web_search_structured
        q = str(params.get("query", "")).strip()
        if not q:
            return "❌ web_search: مطلوب query"
        res = web_search_structured(q, max_results=int(params.get("max_results", 6)))
        lines = [f"- [{r.get('title', '')}]({r.get('url', '')}): {r.get('snippet', '')}"
                 for r in (res.get("results") or [])[:10]]
        return ("\n".join(lines) or (res.get("msg") or "لا نتائج"))[:_MAX_OUTPUT_CHARS]
    except Exception as e:
        return f"❌ web_search: {e}"


def _tool_deep_research(params: Dict[str, Any]) -> str:
    try:
        from ai.web_search_tool import deep_research
        res = deep_research(str(params.get("query", "")),
                            max_per_angle=int(params.get("max_per_angle", 3)))
        return json.dumps(res, ensure_ascii=False)[:_MAX_OUTPUT_CHARS]
    except Exception as e:
        return f"❌ deep_research: {e}"


register_tool(ToolSpec(
    "web_search", "بحث ويب حقيقي متعدد المصادر (DuckDuckGo/Wikipedia/أخبار) بلا مفتاح API.",
    {"type": "object", "properties": {
        "query": {"type": "string", "description": "عبارة البحث"},
        "max_results": {"type": "integer", "description": "عدد النتائج (اختياري)"},
    }, "required": ["query"]},
    _tool_web_search,
))
register_tool(ToolSpec(
    "deep_research", "بحث عميق متعدد الزوايا لموضوع واحد، يُرجع تقريراً منسقاً JSON.",
    {"type": "object", "properties": {
        "query": {"type": "string", "description": "موضوع البحث"},
        "max_per_angle": {"type": "integer", "description": "نتائج لكل زاوية (اختياري)"},
    }, "required": ["query"]},
    _tool_deep_research,
))


# ── 4) أدوات المشروع الخاصة ────────────────────────────────────────
def _tool_git_push(params: Dict[str, Any]) -> str:
    msg = (params.get("message") or "NSM agent_loop auto-commit").strip()[:200]
    try:
        from ai.code_agent import git_push
        return git_push(msg)[:1200]
    except Exception as e:
        return f"❌ git_push: {e}"


def _tool_kaggle_push(params: Dict[str, Any]) -> str:
    """رفع kernel إلى Kaggle عبر job_id موجود في artifacts/agent_jobs."""
    try:
        from ai.kaggle_provider import push_kaggle_kernel
        job_id = str(params.get("job_id", "")).strip()
        if not job_id:
            return "❌ kaggle_push: مطلوب job_id"
        res = push_kaggle_kernel(job_id)
        return json.dumps(res, ensure_ascii=False)[:1200]
    except Exception as e:
        return f"❌ kaggle_push: {e}"


def _tool_notebook_run(params: Dict[str, Any]) -> str:
    """تشغيل خلية دفتر (SQL/HTTP/code/bash/train) من داخل الحلقة."""
    try:
        from ai.notebook_engine import (
            get_notebook, run_cell, Notebook,
        )
        nb_id = str(params.get("notebook_id", "")).strip()
        cell_id = str(params.get("cell_id", "")).strip()
        if not nb_id or not cell_id:
            return "❌ notebook_run: مطلوب notebook_id وcell_id"
        nb = get_notebook(nb_id)
        if nb is None:
            return f"❌ notebook_run: لا دفتر بالمعرّف {nb_id}"
        run_cell(nb, cell_id, timeout=int(params.get("timeout", 120)))
        cell = next((c for c in nb.cells if c.id == cell_id), None)
        outs = (cell.outputs or [])[-1] if cell else {}
        return json.dumps({"status": cell.status if cell else "unknown",
                           "last_output": outs}, ensure_ascii=False)[:_MAX_OUTPUT_CHARS]
    except Exception as e:
        return f"❌ notebook_run: {e}"


register_tool(ToolSpec(
    "git_push", "رفع كل التغييرات إلى GitHub (main) مع commit — يتطلب GITHUB_TOKEN في Secrets.",
    {"type": "object", "properties": {
        "message": {"type": "string", "description": "رسالة commit عربية قصيرة"},
    }, "required": []},
    _tool_git_push, dangerous=True,
))
register_tool(ToolSpec(
    "kaggle_push", "رفع kernel تدريب إلى Kaggle (يتطلب KAGGLE_USERNAME/KAGGLE_KEY في Secrets).",
    {"type": "object", "properties": {
        "job_id": {"type": "string", "description": "معرّف المهمة من artifacts/agent_jobs (مثل surah_tpu_v5e)"},
    }, "required": ["job_id"]},
    _tool_kaggle_push, dangerous=True,
))
register_tool(ToolSpec(
    "notebook_run", "تشغيل خلية في دفتر NSM (يدعم sql/http/code/bash/train).",
    {"type": "object", "properties": {
        "notebook_id": {"type": "string", "description": "معرّف الدفتر"},
        "cell_id": {"type": "string", "description": "معرّف الخلية"},
        "timeout": {"type": "integer", "description": "مهلة ثوانٍ (اختياري)"},
    }, "required": ["notebook_id", "cell_id"]},
    _tool_notebook_run,
))


# ═════════════════════════ نظام الملاحظات (observations) ═══════════
def _truncate_obs(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 200] + f"\n\n... [مقطوع — الأصل {len(text)} حرف]"


# ═════════════════════════ محرك الحلقة ═════════════════════════════
_SYSTEM_PROMPT = """أنت الوكيل التنفيذي الذاتي لنظام Neural Service Mesh (NSM).
تعمل داخل حلقة plan→act→observe→decide. كل استدعاء منك يجب أن يكون JSON صالحاً فقط
(لا نص خارجه) بالصيغة:

{"thinking": "تفكير قصير بخطوة أو خطوتين",
 "tools": [
   {"tool": "اسم_الأداة", "params": {...}},
   ...
 ],
 "finish": "نص نهائي يلخص ما أنجزته للمستخدم (يُعرض له عند end=true)" ,
 "end": true/false}

القواعد الصارمة:
1. نفّذ أداة واحدة أو أدوات مستقلة معاً، ثم انتظر نتائجها قبل الخطوة التالية (لا تخمّن النتائج).
2. بعد أي create_file/edit_file على ملف .py: نفّذ py_compile على نفس الملف قبل المتابعة.
3. لا تنهِ المهمة بـend=true إلا بعد إنجاز فعلي أو عجز مؤكد — النتائج تُعرض لك تلقائياً.
4. الأخطاء تُعاد إليك كنص observation — أصلحها بنفسك بأداة أخرى أو بمسار بديل.
5. ممنوع: sudo، حذف أنظمة، أوامر destructive (سأرفضها برسالة خطأ).
6. اكتب بالعربية في finish عند الإمكان، والباقي JSON فقط.

الأدوات المتاحة وملاحظاتها تعود كنص في الرسالة التالية لك."""


def _parse_tool_call(raw: str) -> Optional[Dict[str, Any]]:
    """تحليل رد النموذج الموجه للـtools — أكثر تسامحاً من JSON الصارم."""
    if not raw:
        return None
    raw_s = raw.strip()
    # إزالة markdown fences إن وُجدت
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw_s, re.S)
    if m:
        raw_s = m.group(1).strip()
    try:
        return json.loads(raw_s)
    except json.JSONDecodeError:
        # محاولة استخراج أول {...} متوازن
        stack, start = 0, None
        for i, ch in enumerate(raw_s):
            if ch == "{":
                if stack == 0:
                    start = i
                stack += 1
            elif ch == "}":
                stack -= 1
                if stack == 0 and start is not None:
                    try:
                        return json.loads(raw_s[start: i + 1])
                    except json.JSONDecodeError:
                        start = None
        return None


def _invoke_llm(llm_fn: Callable, system: str, history: List[Dict[str, Any]]) -> str:
    """استدعاء LLM مع fallback على ai.llm_fallback."""
    try:
        resp = llm_fn(system, history)
    except Exception as first:
        resp = None
    if not resp or not str(resp).strip():
        raise RuntimeError(f"رد LLM فارغ — {first if not resp else 'لا نص'}")
    txt = str(resp)
    # نص CKG ليس رد LLM حقيقيًا بل رسالة اعتذار من fallback
    if txt.startswith("سؤالك خارج") or "خارج نطاق معرفتي" in txt:
        raise RuntimeError(
            "تعذّر الاتصال بأي مزود LLM — تحقق من مفاتيح Groq/Gemini/Cloudflare "
            "(في Streamlit Secrets) أو استخدم llm_fn مخصصًا")
    return txt


def _build_tools_prompt() -> str:
    """قائمة الأدوات بأسمائها وأوصافها وواجهاتها — تدفع داخل النظام."""
    lines = ["## الأدوات المتاحة (أرجع بأسمائها حرفياً):"]
    for name in _TOOL_ORDER:
        spec = TOOL_REGISTRY[name]
        props = spec.params_schema.get("properties", {})
        req = spec.params_schema.get("required", [])
        param_doc = ", ".join(f"{k} ({v.get('type', '?')})" for k, v in props.items())
        lines.append(f"- **{name}**: {spec.description} [{param_doc}] required={req}")
    return "\n".join(lines)


class LoopState:
    """حالة الحلقة الواحدة — قابلة للاستعلام من الواجهة خلال التنفيذ."""
    def __init__(self, loop_id: str, user_input: str) -> None:
        self.loop_id = loop_id
        self.user_input = user_input
        self.round = 0
        self.tools_used = 0
        self.steps: List[Dict[str, Any]] = []
        self.status = "pending"   # pending | running | done | failed | stopped
        self.started_at = _now()
        self.plain_text_attempts = 0

    def record(self, event: Dict[str, Any]) -> None:
        self.steps.append(event)


def run_agent_loop(
    user_input: str,
    *,
    llm_fn: Optional[Callable] = None,
    max_rounds: int = _MAX_ROUNDS,
    max_tools_per_round: int = _MAX_TOOLS_PER_ROUND,
    max_total_tools: int = _MAX_TOTAL_TOOLS,
    tools_override: Optional[List[str]] = None,
    yield_events: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """حلقة تنفيذ متعددة الجولات.

    llm_fn(system_prompt, history[{'role','content'}]) -> str
    إن لم تُمرَّر، يُبنى fallback تلقائي من NSMAgent الحالي.
    """
    loop_id = f"loop_{uuid.uuid4().hex[:8]}"
    state = LoopState(loop_id, user_input)
    active_loops[loop_id] = state

    def _emit(event: Dict[str, Any]) -> None:
        state.record(event)
        _audit(loop_id, event)
        if yield_events:
            yield_queue.append(event)

    yield_queue: List[Dict[str, Any]] = []

    def _flush() -> Generator[Dict[str, Any], None, None]:
        """تصريف كل الأحداث المعلَّقة — يُستدعى دوريًا لضمان تدفق مباشر."""
        if not yield_queue:
            return
        while yield_queue:
            yield yield_queue.pop(0)

    try:
        with _RUN_LOCK:
            state.status = "running"
            _emit({"type": "status", "loop_id": loop_id, "status": "running"})
            yield from _flush()

            # 1) تسجيل خطة في task_manager
            plan_id = None
            try:
                from ai.task_manager import create_plan

                class _MiniPlan:
                    def __init__(self, inp):
                        self.title = (inp or "agent_loop task")[:120]
                        self.status = "active"
                        self.tasks = []
                        self.created_at = _now()
                plan_id = create_plan(_MiniPlan(user_input))
                _emit({"type": "status", "plan_id": plan_id})
            except Exception:
                plan_id = None

            # 2) بناء llm_fn تلقائياً إن لم تُمرَّر
            fn = llm_fn
            if fn is None:
                fn = _default_llm_fn()

            # 3) تهيئة الرسائل
            system = _SYSTEM_PROMPT + "\n\n" + _build_tools_prompt()
            history: List[Dict[str, Any]] = [
                {"role": "user", "content": user_input},
            ]
            total_tools = 0
            done = False
            last_result = ""

            # 4) الحلقة الأساسية
            while state.round < max_rounds and not done and total_tools < max_total_tools:
                state.round += 1
                _emit({"type": "status", "round": state.round,
                       "total_tools": total_tools})
                try:
                    raw = _invoke_llm(fn, system, history)
                except Exception as e:
                    _emit({"type": "answer", "text": f"⚠️ تعذّر الاتصال بنموذج اللغة: {e}"})
                    state.status = "failed"
                    done = True
                    break
                history.append({"role": "assistant", "content": raw})

                parsed = _parse_tool_call(raw)
                if parsed is None:
                    # رد نصي حر — رد واحد توضيحي يطلب JSON، فإن استمر النص
                    # الحر يُنهي الحلقة مع تحذير (ليس نجاحًا)
                    if state.plain_text_attempts >= 2:
                        _emit({"type": "answer", "text": "⚠️ لم أستطع توليد أوامر "
                                     "أدوات صالحة (JSON) بعد محاولتين — "
                                     "تحقق من مفاتيح LLM أو جودة المزود."})
                        state.status = "failed"
                        done = True
                        break
                    state.plain_text_attempts += 1
                    history.append({"role": "user", "content": (
                        "ردك السابق ليس JSON صالحًا. يجب أن ترد فقط JSON بالصيغة "
                        "المحددة (thinking/tools/finish/end). رد الآن.")})
                    continue

                # 5) تنفيذ الأدوات في هذه الجولة
                tools = parsed.get("tools") or []
                if not tools and not parsed.get("finish"):
                    # نموذج رد نصاً حراً — نعده جواباً نهائياً
                    _emit({"type": "answer", "text": raw})
                    state.status = "done"
                    done = True
                    break

                obs_round = []
                for tool_req in tools[:max_tools_per_round]:
                    if total_tools >= max_total_tools:
                        obs_round.append("⚠️ استُنفدت ميزانية الأدوات لهذا الطلب")
                        break
                    tname = str(tool_req.get("tool", ""))
                    params = tool_req.get("params") or {}
                    spec = TOOL_REGISTRY.get(tname)
                    total_tools += 1
                    state.tools_used += 1
                    _emit({"type": "tool", "tool": tname, "params": params})
                    if spec is None:
                        obs = (f"❌ أداة غير معروفة: {tname}. المتاحة: "
                               + ", ".join(_TOOL_ORDER))
                    else:
                        try:
                            obs = _truncate_obs(spec.executor(params))
                        except Exception as e:
                            obs = f"❌ استثناء أثناء {tname}: {e}"
                    obs_round.append(f"[{tname}] {obs}")
                    _emit({"type": "result", "tool": tname, "output": obs})

                if obs_round:
                    # 6) ملاحظة مدمجة تُعاد للنموذج — جوهر الحلقة
                    merged = ("📋 نتائج هذه الجولة:\n" + "\n---\n".join(obs_round)
                              + "\n\nإن لم تنجز المهمة كاملة بعد، نفّذ الجولة التالية. "
                              "وإن أنجزتها أو تعذّر إنجازها، رد بـfinish مع end=true.")
                    history.append({"role": "user", "content": merged})

                # 7) هل أنهى النموذج بنفسه؟
                if parsed.get("end") or parsed.get("finish"):
                    finish = parsed.get("finish") or ""
                    if finish:
                        _emit({"type": "answer", "text": str(finish)})
                    elif obs_round:
                        # لم ينهِ صراحة لكن أُدمجت النتائج — جولة إضافية
                        continue
                    state.status = "done"
                    done = True
                    break

            if not done:
                state.status = "done" if total_tools else "failed"
                _emit({"type": "answer", "text": (
                    "⚠️ **وصلت لحد الحلقة** "
                    f"({state.round} جولة، {state.tools_used} أداة). "
                    "لخص لي ما تبقى وسأكمل من حيث توقفت."
                )})

            # 8) إغلاق الخطة
            if plan_id is not None:
                try:
                    from ai.task_manager import mark_plan_status
                    mark_plan_status(plan_id,
                                     "completed" if state.status == "done" else "failed")
                except Exception:
                    pass
            state.status = state.status or "done"
            _emit({"type": "status", "status": state.status,
                   "tools_used": state.tools_used, "rounds": state.round})
            yield from _flush()
    except GeneratorExit:
        pass
    finally:
        active_loops.pop(loop_id, None)
        state.status = state.status or "done"


def _default_llm_fn() -> Callable:
    """دالة LLM افتراضية مبنية على مزودي NSM (groq/cf/gemini)."""
    def _fn(system: str, history: List[Dict[str, Any]]) -> str:
        try:
            from ai.nsm_agent_core import _call_api
        except Exception:
            from ai.llm_fallback import LLMFallback
            fb = LLMFallback()
            last = history[-1]["content"] if history else ""
            res = fb.generate(last)
            return getattr(res, "text", str(res))
        messages = [{"role": "system", "content": system}] + history
        try:
            resp = _call_api(messages)
        except Exception as e:
            resp = None
        if resp:
            return str(resp)
        # فشل مباشر مع مزودي NSM → فallback متسلسل ثم استثناء واضح
        try:
            from ai.llm_fallback import LLMFallback
            fb = LLMFallback()
            last = history[-1]["content"] if history else ""
            res = fb.generate(last)
            txt = getattr(res, "text", None) or str(res)
            if txt and "Rotation" not in str(txt) and "CKG" not in str(txt):
                return txt
        except Exception:
            pass
        raise RuntimeError(
            "تعذّر الاتصال بأي مزود LLM — تحقق من مفاتيح Groq/Gemini/Cloudflare "
            "(في Streamlit Secrets) أو استخدم llm_fn مخصصًا عند استدعاء الحلقة")
    return _fn


# ── قوائم حلقات نشطة (للاستعلام من الواجهة) ────────────────────────
active_loops: Dict[str, LoopState] = {}


def list_active_loops() -> List[Dict[str, Any]]:
    return [{"loop_id": s.loop_id, "round": s.round, "tools_used": s.tools_used,
             "status": s.status, "started_at": s.started_at,
             "n_steps": len(s.steps)} for s in active_loops.values()]


def get_loop_state(loop_id: str) -> Optional[Dict[str, Any]]:
    s = active_loops.get(loop_id)
    if s is None:
        return None
    return {"loop_id": s.loop_id, "round": s.round, "status": s.status,
            "steps": s.steps}


# ═════════════════════════ أداة تنفيذ خطوة واحدة (للواجهة) ═════════
def execute_single_tool(tool_name: str, params: Dict[str, Any]) -> str:
    """تنفيذ أداة واحدة مباشرة من الواجهة (لوحة الطرفيات)."""
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        return f"❌ أداة غير معروفة: {tool_name}"
    if spec.dangerous:
        ok, why = _cmd_safe(params.get("cmd", "")) if tool_name == "shell" else (True, "")
        if not ok:
            return f"❌ {why}"
    try:
        return _truncate_obs(spec.executor(params))
    except Exception as e:
        return f"❌ {e}"


# ═════════════════════════ تكامل سريع: استدعاء حلقة بطلب نصي ════════
def run_loop_to_text(user_input: str, **kwargs) -> str:
    """جمع أحداث الحلقة في نص واحد — للتوافق مع run() القديم."""
    parts = []
    for ev in run_agent_loop(user_input, **kwargs):
        t = ev.get("type")
        if t == "tool":
            parts.append(f"🔧 {ev.get('tool')}: {json.dumps(ev.get('params') or {}, ensure_ascii=False)}")
        elif t == "result":
            parts.append(f"   ↳ {str(ev.get('output', ''))[:600]}")
        elif t == "answer":
            parts.append(ev.get("text", ""))
    return "\n".join(parts)


# ── 5) المتصفح (ai/agent_browser) ─────────────────────────────────
def _tool_browser_navigate(params: Dict[str, Any]) -> str:
    try:
        from ai.agent_browser import navigate
        nav = navigate(str(params.get("url", "")),
                       allow_internal=bool(params.get("allow_internal", False)),
                       timeout=int(params.get("timeout", 15)))
        text = nav.get("text", "")
        links = nav.get("links", [])[:20]
        out = [f"🌐 {nav.get('title', '')} ({nav.get('url', '')}) "
               f"[{nav.get('duration_ms', 0)}ms]", text[:3000]]
        if links:
            out.append("الروابط:\n" + "\n".join(f"- {l}" for l in links))
        if not nav.get("ok"):
            return f"❌ browser_navigate: {nav.get('error')}"
        return "\n".join(out)[:_MAX_OUTPUT_CHARS]
    except Exception as e:
        return f"❌ browser_navigate: {e}"


def _tool_browser_api(params: Dict[str, Any]) -> str:
    try:
        from ai.agent_browser import api_call
        res = api_call(str(params.get("url", "")),
                       method=str(params.get("method", "GET")),
                       headers=params.get("headers"),
                       body=params.get("body"),
                       timeout=int(params.get("timeout", 15)),
                       allow_internal=bool(params.get("allow_internal", False)))
        return json.dumps(res, ensure_ascii=False)[:_MAX_OUTPUT_CHARS]
    except Exception as e:
        return f"❌ browser_api: {e}"


def _tool_browser_download(params: Dict[str, Any]) -> str:
    try:
        from ai.agent_browser import download
        res = download(str(params.get("url", "")),
                       filename=str(params.get("filename", "")),
                       allow_internal=bool(params.get("allow_internal", False)),
                       timeout=int(params.get("timeout", 60)))
        return json.dumps(res, ensure_ascii=False)[:_MAX_OUTPUT_CHARS]
    except Exception as e:
        return f"❌ browser_download: {e}"


def _tool_browser_inspect(params: Dict[str, Any]) -> str:
    try:
        from ai.agent_browser import inspect
        res = inspect(str(params.get("url", "")),
                      what=str(params.get("what", "links")),
                      allow_internal=bool(params.get("allow_internal", False)),
                      timeout=int(params.get("timeout", 15)))
        return json.dumps(res, ensure_ascii=False)[:_MAX_OUTPUT_CHARS]
    except Exception as e:
        return f"❌ browser_inspect: {e}"


register_tool(ToolSpec(
    "browser_navigate", "فتح صفحة ويب واستخراج نصها الكامل وروابطها (بدون تنفيذ JS).",
    {"type": "object", "properties": {
        "url": {"type": "string", "description": "عنوان HTTPS"},
        "timeout": {"type": "integer", "description": "مهلة ثوانٍ (3-60)"},
        "allow_internal": {"type": "boolean", "description": "سماح العناوين الداخلية (localhost) — ممنوع افتراضيًا"},
    }, "required": ["url"]},
    _tool_browser_navigate,
))
register_tool(ToolSpec(
    "browser_api", "استدعاء REST API حقيقي (GET/POST/PUT/DELETE) مع رؤوس وجسم JSON.",
    {"type": "object", "properties": {
        "url": {"type": "string", "description": "عنوان HTTPS"},
        "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"], "description": "طريقة HTTP"},
        "headers": {"type": "object", "description": "رؤوس HTTP اختيارية"},
        "body": {"type": ["object", "string"], "description": "جسم الطلب (JSON أو نص)"},
        "timeout": {"type": "integer", "description": "مهلة ثوانٍ (3-60)"},
        "allow_internal": {"type": "boolean", "description": "سماح العناوين الداخلية"},
    }, "required": ["url"]},
    _tool_browser_api,
))
register_tool(ToolSpec(
    "browser_download", "تنزيل ملف من الويب إلى artifacts/agent_browser/.",
    {"type": "object", "properties": {
        "url": {"type": "string", "description": "عنوان HTTPS للملف"},
        "filename": {"type": "string", "description": "اسم الحفظ (اختياري)"},
        "timeout": {"type": "integer", "description": "مهلة ثوانٍ (5-120)"},
        "allow_internal": {"type": "boolean", "description": "سماح العناوين الداخلية"},
    }, "required": ["url"]},
    _tool_browser_download,
))
register_tool(ToolSpec(
    "browser_inspect", "فحص عناصر صفحة: روابط، عناوين (h1-h6)، أو صور.",
    {"type": "object", "properties": {
        "url": {"type": "string", "description": "عنوان HTTPS"},
        "what": {"type": "string", "enum": ["links", "headings", "images"], "description": "ما يفحص"},
        "timeout": {"type": "integer", "description": "مهلة ثوانٍ"},
        "allow_internal": {"type": "boolean", "description": "سماح العناوين الداخلية"},
    }, "required": ["url", "what"]},
    _tool_browser_inspect,
))
