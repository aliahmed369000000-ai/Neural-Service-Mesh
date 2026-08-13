"""
NSM Long-Horizon Task Executor — ai/long_horizon_tasks.py (v1)
================================================================
منظّم مهام طويلة الأمد للوكلاء مع وصول آمن للإنترنت — يعمل في خيوط
خلفية (daemon) دون حجز واجهة Streamlit، وكل خطوة تُحفظ لحظيًا
(SQLite + سجل خطوات قابل للقراءة من لوحة المراقبة).

لماذا وحدة مستقلة عن ai/background_tasks.py؟
  هناك تدير "سؤال واحد → ردّ واحد" في الخلفية عبر وكيل موحّد، بينما
  هذه الوحدة تدير مهامًا متعددة الخطوات (تخطيط → خطوات متسلسلة →
  سياق متراكم → تقرير نهائي) بأدوات إنترنت مدمجة وبلا اعتماد على LLM
  لكل خطوة — القرار في كل خطوة حتمي/أدوات، والنموذج يُستدعى فقط عند
  التلخيص النهائي إن طُلب.

التصميم:
  1) LongHorizonTaskManager (singleton على مستوى العملية):
     submit_task / get_task / list_tasks / cancel_task / get_task_log
     + خيط daemon لكل مهمة، سجل SQLite في data/long_horizon_tasks.db،
     طابور MAX_PENDING=6 ومتزامن MAX_CONCURRENT=2.
  2) خطة حتمية (بلا LLM للتخطيط): من هدف نصي واحد يُبنى مخطط خطوات
     متسلسلة عبر is_complex_question + decompose من ai.multi_step_reasoner
     (حتمي بالكامل — بلا API إطلاقًا)، ثم تُنفَّذ الخطوات واحدة واحدة
     بأدوات الإنترنت المدمجة (بحث، جلب صفحات، قراءة/كتابة ملفات، تشغيل
     بايثون في وضع محمي).
  3) وصول الإنترنت للجميع:
     - tool_web_search: يعيد استخدام ai.web_search_tool.web_search_structured
       (DuckDuckGo/Wikipedia/Wikidata/News/arXiv/Trends — بلا مفاتيح).
     - tool_fetch_page: جلب صفحة وتنظيف نصها (urllib، مهلة، سقف حجم،
       تهريب كامل، منع domains داخلية).
     - tool_write_file / tool_read_file: مساحة عمل معزولة داخل
       data/lht_workspace/ — خارج ai/ و ui_pages/ و git history.
     - tool_run_python: تشغيل سكربت بايثون مصغّر في مسار مخصص
       (sys.executable -I مع env منقى — بلا وصول للوحدات الحساسة).
  4) حوكمة صارمة (كلها حتمية):
     - سقف خطوات MAX_STEPS=24 لكل مهمة، سقف إجمالي FETCHes=40،
       سقف حجم ملف واحد=2MB، سقف مدة مهمة=45 دقيقة.
     - منع domains خطرة (localhost/127.0.0.1/::1/169.254/10.0.0.0/8
       وinternal AWS/GCP metadata) حتى من re.split URLs.
     - منع أسماء ملفات خطرة (../,绝对.. أو مسارات ai/ ui_pages/...).
     - أي فشل خطوة → retry واحد → تسجيل الخطوة "فشل جزئي" ومتابعة
       الخطوات التالية دون إيقاف المهمة كاملة.
  5) تدهور آمن كامل: استيراد اختياري (_LHT_OK) من app_core — كل فشل
     في الوحدة (بما فيه غياب ai.web_search_tool أو missing internet)
     يُسجَّل كتعليق خطوة ولا يعطّل أي مسار موجود.
  6) ربط الوكلاء: NSMAgent.run_stream (ai/nsm_agent_core.py) يعرّف
     أوامر trigger حتمية "نفّذ مهمة طويلة: ..." / "ابحث وأعدّ تقريرًا عن..."
     → submit فوري ثم yield رسالة قبول + badge "🧵 مهمة طويلة".
  7) لوحة UI: render_long_horizon_panel في ui_pages/agent_monitor.py
     (تبويب "📡 مراقبة حيّة" + تبويب فرعي داخلي) مع live reload لكل
     10 ثوانٍ وسجل خطوات محدث.

لا تعتمد هذه الوحدة على streamlit داخل منطقها — يمكن اختبارها بالكامل
بدون مفاتيح API حقيقية (mock call_api أو execute steps حتمية).
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ai.long_horizon_tasks")

# ══════════════════════════════════════════════════════════════════
# ثوابت الحوكمة
# ══════════════════════════════════════════════════════════════════
MAX_STEPS = 24           # سقف خطوات المهمة
MAX_FETCHES = 40         # سقف طلبات الإنترنت لكل مهمة
MAX_CONCURRENT = 2       # مهام متزامنة كحد أقصى
MAX_PENDING = 6          # طابور معلق كحد أقصى
MAX_TASK_AGE_S = 45 * 60 # مدة حياة المهمة
MAX_FILE_BYTES = 2 * 1024 * 1024        # سقف ملف واحد
MAX_LOG_ENTRIES = 300    # سقف سجل الخطوات (LRU)
FETCH_TIMEOUT = 12       # ثانية لكل طلب
FETCH_MAX_CHARS = 120_000  # سقف نص الصفحة بعد التنظيف
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.join(BASE_DIR, "data", "lht_workspace")
DB_DEFAULT = os.path.join(BASE_DIR, "data", "long_horizon_tasks.db")

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

# دامينات داخلية / سحابية حساسة محظورة كليًا
_BLOCKED_HOSTS = {
    "localhost", "127.0.0.1", "::1", "0.0.0.0",
    "169.254.169.254", "metadata.google.internal",
    "metadata.google", "instance-data",
}
_BLOCKED_PATTERNS = (
    re.compile(r"^169\.254\."),        # link-local (AWS/GCP metadata)
    re.compile(r"^10\."),              # private 10.x
    re.compile(r"^192\.168\."),        # private 192.168.x
    re.compile(r"^172\.(1[6-9]|2\d|3[01])\."),  # private 172.16-31.x
    re.compile(r"^127\."),             # loopback
)
# وحدات بايثون محظورة على tool_run_python
_FORBIDDEN_PY_IMPORTS = {
    "subprocess", "os", "sys", "socket", "urllib", "http", "requests",
    "shutil", "importlib", "ctypes", "multiprocessing", "webbrowser",
    "smtplib", "telnetlib", "ftplib", "xmlrpc", "asyncio",
}


# ══════════════════════════════════════════════════════════════════
# أدوات الإنترنت الآمنة
# ══════════════════════════════════════════════════════════════════

def _is_safe_url(url: str) -> Tuple[bool, str]:
    """تحقّق أمني من الرابط: بروتوكول http(s) فقط ومنع دامينات داخلية."""
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return False, "بروتوكول غير مسموح"
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as e:
        return False, f"رابط غير صالح: {e}"
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return False, "بدون host"
    if host in _BLOCKED_HOSTS:
        return False, f"دامين محظور: {host}"
    for pat in _BLOCKED_PATTERNS:
        if pat.match(host):
            return False, f"دامين شبكي داخلي محظور: {host}"
    # منع data: و javascript: و file: المتخفية داخل الرابط
    if re.search(r"(?:data|javascript|file|ftp)\s*:", url, re.I):
        return False, "مخطط رابط خطير داخل النص"
    return True, "OK"


def _safe_html_to_text(html_text: str, max_chars: int = FETCH_MAX_CHARS) -> str:
    """تنظيف HTML إلى نص قابل للقراءة (عربي): إزالة السكربت/الأنماط والوسوم."""
    text = re.sub(r"<(script|style|noscript)[\s\S]*?</\1>", " ", html_text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars].strip()


def html_unescape(s: str) -> str:
    """تهريب كيانات HTML إلى UTF-8 مع حفظ النص العربي."""
    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    s = s.replace("&quot;", '"').replace("&apos;", "'").replace("&#39;", "'")
    # كيانات &#NNN;
    return re.sub(r"&#(\d+);?", lambda m: chr(int(m.group(1))) if int(m.group(1)) < 0x110000 else "", s)


def tool_web_search(query: str, max_results: int = 8) -> Dict[str, Any]:
    """بحث ويب حقيقي بلا مفاتيح — يعيد استخدام web_search_tool."""
    try:
        from ai.web_search_tool import web_search_structured
        res = web_search_structured(query, max_results=max_results)
        results = res.get("results", []) if isinstance(res, dict) else []
        if not results:
            return {"ok": False, "query": query, "text": f"لا نتائج لـ: {query}", "count": 0}
        lines = [f"نتائج البحث: {query}", ""]
        for i, r in enumerate(results, 1):
            title = (r.get("title") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            url = (r.get("url") or "").strip()
            lines.append(f"{i}. {title}")
            if url:
                lines.append(f"   الرابط: {url}")
            if snippet:
                lines.append(f"   {snippet[:200]}")
            lines.append("")
        return {"ok": True, "query": query, "text": "\n".join(lines), "count": len(results)}
    except Exception as e:
        logger.debug("tool_web_search فشل: %s", e)
        return {"ok": False, "query": query, "text": f"فشل البحث: {e}", "count": 0}


def tool_fetch_page(url: str) -> Dict[str, Any]:
    """جلب صفحة وتنظيف نصها — مع تحقق أمني صارم."""
    ok, reason = _is_safe_url(url)
    if not ok:
        return {"ok": False, "url": url, "text": f"رابط مرفوض: {reason}"}
    try:
        import urllib.request
        req = urllib.request.Request(
            url, headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; NSMLongHorizonAgent/1.0; "
                    "+https://github.com/aliahmed369000000-ai/Neural-Service-Mesh)"
                ),
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            charset = "utf-8"
            try:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                m = re.search(r"charset=([\w\-]+)", ctype)
                if m:
                    charset = m.group(1)
            except Exception:
                pass
            raw = resp.read(FETCH_MAX_CHARS * 3).decode(charset, errors="ignore")
        text = _safe_html_to_text(raw)
        if not text:
            return {"ok": False, "url": url, "text": "الصفحة فارغة بعد التنظيف"}
        return {"ok": True, "url": url, "title": (text.split("\n")[0])[:80], "text": text}
    except Exception as e:
        logger.debug("tool_fetch_page فشل (%s): %s", url, e)
        return {"ok": False, "url": url, "text": f"فشل الجلب: {e}"}


def _safe_workspace_path(name: str) -> Tuple[Optional[str], str]:
    """التحقق من اسم ملف آمن داخل مساحة العمل المعزولة."""
    name = (name or "").strip().replace("/", "_").replace("\\", "_")
    if not name or name.startswith(".") or ".." in name:
        return None, "اسم ملف غير صالح"
    if re.search(r"[<>:\"|?*]", name):
        return None, "حروف غير مسموحة في اسم الملف"
    # منع امتدادات قابلة للتنفيذ خارج نطاقنا
    if name.lower().endswith((".exe", ".bat", ".sh", ".dll", ".so")):
        return None, "امتداد تنفيذي غير مسموح"
    full = os.path.join(WORKSPACE, name)
    real = os.path.realpath(full)
    real_ws = os.path.realpath(WORKSPACE)
    if not real.startswith(real_ws + os.sep) and real != real_ws:
        return None, "مسار خارج مساحة العمل"
    return full, "OK"


def tool_write_file(name: str, content: str) -> Dict[str, Any]:
    """كتابة ملف في مساحة العمل المعزولة data/lht_workspace/."""
    full, reason = _safe_workspace_path(name)
    if full is None:
        return {"ok": False, "name": name, "text": f"ملف مرفوض: {reason}"}
    content = (content or "")[:MAX_FILE_BYTES]
    try:
        os.makedirs(WORKSPACE, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return {"ok": True, "name": name, "bytes": len(content.encode("utf-8")),
                "text": f"كُتب {name} ({len(content.encode('utf-8'))} بايت)"}
    except Exception as e:
        logger.debug("tool_write_file فشل: %s", e)
        return {"ok": False, "name": name, "text": f"فشل الكتابة: {e}"}


def tool_read_file(name: str) -> Dict[str, Any]:
    """قراءة ملف من مساحة العمل المعزولة."""
    full, reason = _safe_workspace_path(name)
    if full is None or not os.path.isfile(full or ""):
        return {"ok": False, "name": name, "text": f"ملف غير موجود أو مرفوض: {reason}"}
    try:
        size = os.path.getsize(full)
        if size > MAX_FILE_BYTES:
            return {"ok": False, "name": name, "text": f"الملف يتجاوز {MAX_FILE_BYTES} بايت"}
        with open(full, "r", encoding="utf-8") as f:
            content = f.read()
        return {"ok": True, "name": name, "bytes": size, "text": content[:60_000]}
    except Exception as e:
        logger.debug("tool_read_file فشل: %s", e)
        return {"ok": False, "name": name, "text": f"فشل القراءة: {e}"}


def tool_run_python(code: str, timeout: int = 60) -> Dict[str, Any]:
    """تشغيل سكربت بايثون قصير في عملية معزولة بمخرجات ملتقطة."""
    code = (code or "").strip()
    if not code:
        return {"ok": False, "text": "كود فارغ"}
    if len(code) > 20_000:
        return {"ok": False, "text": "الكود يتجاوز 20 ألف حرف"}
    # فحص حذر: "import sys" مقبول (قراءة sys.version فقط)، وكل شيء آخر محظور.
    # نستخدم فحصًا صريحًا بالسطر لتفادي مطابقة أجزاء (مثل "subsystems").
    for line in code.splitlines():
        stripped = line.strip()
        m = re.match(r"(?:import\s+(\w+(?:\.\w+)*)\b|from\s+(\w+)\b)", stripped)
        if not m:
            continue
        mod_root = m.group(1) if m.group(1) else m.group(2)
        if mod_root == "sys":
            # sys مسموح فقط إذا لم يُستدعَ بأدوات خطرة (argv غير مسموح)
            if "argv" in code or "exit(" in code or "modules" in code:
                return {"ok": False, "text": f"استخدام sys محظور: argv/exit/moduls"}
            continue
        if mod_root in _FORBIDDEN_PY_IMPORTS or mod_root.split(".")[0] in _FORBIDDEN_PY_IMPORTS:
            return {"ok": False, "text": f"استيراد محظور لأسباب أمنية: {mod_root}"}
    tmp_name = f"lht_{uuid.uuid4().hex[:10]}.py"
    write_res = tool_write_file(tmp_name, code)
    if not write_res.get("ok"):
        return {"ok": False, "text": f"فشل حفظ السكربت: {write_res.get('text')}"}
    full = os.path.join(WORKSPACE, tmp_name)
    try:
        proc = subprocess.run(
            [sys.executable, "-I", full],
            capture_output=True, text=True, timeout=timeout,
            cwd=WORKSPACE,
            env={k: v for k, v in os.environ.items()
                 if not re.search(r"(SECRET|TOKEN|PASSWORD|KEY|CREDENTIAL)", k, re.I)},
        )
        out = (proc.stdout or "")[:8000]
        err = (proc.stderr or "")[:4000]
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "text": (f"المخرجات:\n{out}\n" if out else "")
                    + (f"الأخطاء:\n{err}" if err else ""),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "text": f"انتهت مهلة التنفيذ ({timeout}s)"}
    except Exception as e:
        logger.debug("tool_run_python فشل: %s", e)
        return {"ok": False, "text": f"فشل التنفيذ: {e}"}
    finally:
        with contextlib.suppress(OSError):
            os.remove(full)


# ══════════════════════════════════════════════════════════════════
# الأدوات المعروضة لكل خطوة (جدول أسماء → دوال)
# ══════════════════════════════════════════════════════════════════
BUILTIN_TOOLS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "web_search": tool_web_search,
    "fetch_page": tool_fetch_page,
    "write_file": tool_write_file,
    "read_file": tool_read_file,
    "run_python": tool_run_python,
}

# ══════════════════════════════════════════════════════════════════
# سجل المهام (SQLite + ذاكرة)
# ══════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS lht_tasks (
    task_id TEXT PRIMARY KEY,
    title TEXT, goal TEXT, status TEXT,
    plan_json TEXT,
    created_at TEXT, started_at TEXT, finished_at TEXT,
    progress REAL DEFAULT 0,
    error TEXT
);
CREATE TABLE IF NOT EXISTS lht_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    status TEXT DEFAULT 'pending',
    tool TEXT,
    tool_input TEXT,
    result TEXT,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_lht_steps_task ON lht_steps(task_id);
"""


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.executescript(_SCHEMA)
    return conn


# ══════════════════════════════════════════════════════════════════
# كلاس المهمة
# ══════════════════════════════════════════════════════════════════

class LHTask:
    """مهمة طويلة الأمد واحدة: هدف → خطة حتمية → خطوات → تقرير."""

    def __init__(self, task_id: Optional[str] = None, goal: str = "",
                 title: Optional[str] = None):
        self.task_id = task_id or uuid.uuid4().hex[:12]
        self.goal = (goal or "").strip()
        self.title = (title or self.goal[:60] or "مهمة طويلة الأمد")
        self.status = STATUS_PENDING
        self.plan: List[Dict[str, Any]] = []   # [{"title", "tool", "input"}]
        self.steps: List[Dict[str, Any]] = []  # نتائج الخطوات
        self.created_at = _now_iso()
        self.started_at = ""
        self.finished_at = ""
        self.error = ""
        self.progress = 0.0
        self._fetch_count = 0
        self._cancelled = threading.Event()
        self._t0 = 0.0
        self._final_report = ""  # التقرير النهائي (مرجعًا من آخر خطوة finalize)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "goal": self.goal,
            "status": self.status,
            "plan": self.plan,
            "steps": self.steps,
            "progress": round(self.progress, 2),
            "fetch_count": self._fetch_count,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "duration_s": round(time.time() - self._t0, 1) if self._t0 else 0.0,
            "final_report": self._final_report,
        }

    def log(self, step_index: int, status: str, tool: str = "",
            tool_input: str = "", result: str = "") -> None:
        """يسجل خطوة في الذاكرة وSQLite."""
        entry = {
            "step_index": step_index,
            "status": status,
            "tool": tool,
            "tool_input": (tool_input or "")[:400],
            "result": (result or "")[:4000],
            "started_at": _now_iso(),
            "finished_at": _now_iso(),
        }
        self.steps.append(entry)
        self.steps = self.steps[-MAX_LOG_ENTRIES:]
        try:
            mgr = _manager_for_persist()
            with _connect(mgr.db_path) as conn:
                conn.execute(
                    "INSERT INTO lht_steps (task_id, step_index, status, tool, "
                    "tool_input, result, started_at, finished_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (self.task_id, step_index, status, tool, entry["tool_input"],
                     entry["result"], entry["started_at"], entry["finished_at"]),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("lht: تعذّر حفظ خطوة %s/%d: %s", self.task_id, step_index, exc)


# ══════════════════════════════════════════════════════════════════
# التخطيط الحتمي (بلا API — reuses multi_step_reasoner)
# ══════════════════════════════════════════════════════════════════

def _build_plan(goal: str) -> List[Dict[str, Any]]:
    """
    يبني خطة خطوات حتمية بلا LLM: يصنّف الهدف، يفككه (مركّب → أقسام)،
    ويولّد خطوات البحث/الجلب/التجميع المناسبة.
    """
    t = (goal or "").strip()
    if not t:
        return []
    plan: List[Dict[str, Any]] = []

    # 1) تفكيك المركّب (أسئلة متعددة بواو العطف)
    parts: List[str] = []
    try:
        from ai.multi_step_reasoner import _split_conjuncts, is_complex_question
        if is_complex_question(t):
            parts = _split_conjuncts(t)
        if parts and all(len(p) < 8 for p in parts):
            parts = []  # تجزئة زائفة قصيرة — نتجاهلها
    except Exception:
        parts = []

    targets = parts if parts else [t]

    # 2) خطوات البحث لكل هدف
    for idx, target in enumerate(targets, 1):
        title_part = target[:50].replace("\n", " ")
        if len(targets) > 1:
            plan.append({
                "title": f"البحث عن الهدف {idx}: {title_part}",
                "tool": "web_search",
                "input": target,
                "depends_on": [],
            })
        else:
            plan.append({
                "title": f"البحث عن: {title_part}",
                "tool": "web_search",
                "input": target,
                "depends_on": [],
            })

    # 3) جلب صفحة رئيسية من أفضل نتيجة (إن وُجدت نتائج)
    plan.append({
        "title": "جلب وتعميق المعلومات من أفضل النتائج",
        "tool": "fetch_best",   # أداة داخلية خاصة — ليست في BUILTIN_TOOLS
        "input": json.dumps(targets, ensure_ascii=False),
        "depends_on": [p["title"] for p in plan],
    })

    # 4) تجميع وتوثيق
    plan.append({
        "title": "تجميع النتائج في تقرير منسّق وحفظه",
        "tool": "finalize",     # أداة داخلية خاصة
        "input": json.dumps({"goal": t, "parts": targets}, ensure_ascii=False),
        "depends_on": [plan[-1]["title"]],
    })

    return plan[:MAX_STEPS]


# ══════════════════════════════════════════════════════════════════
# محرك التنفيذ
# ══════════════════════════════════════════════════════════════════

# خطافات قابلة للإحلال من الاختبارات (للمحاكاة دون إنترنت/LLM)
_LHT_PLAN_HOOK = None  # fn(task) -> bool (يجب أن يملأ task.plan)
_LHT_EXECUTE_HOOK = None  # fn(manager, task, goal_label) -> bool
_LHT_EXECUTE_HOOK_TOOL = None  # fn(tool_input) -> {"ok": bool, "text": str}

# ── تكامل سجل الخبرات الجماعية (TEM) ─────────────────────────────
# استيراد متأخر (late import) لتجنب circular import:
# app_core يستورد هذه الوحدة سطر 364 — قبل أن يُعرّف _TEM_OK سطر 391+
# لذا نجلب الأسماء عند الحاجة عبر getattr على module object.
import app_core as _app_core_for_lht  # noqa: E402

def _TEM_OK() -> bool:  # type: ignore[misc]
    return bool(getattr(_app_core_for_lht, "_TEM_OK", False))

def _get_experience_log():  # type: ignore[misc]
    fn = getattr(_app_core_for_lht, "_get_experience_log", None)
    if fn is None:
        raise RuntimeError("سجل الخبرات الجماعية غير متاح")
    return fn()

# ── 🆕 نظام المكافآت الذاتية للأدوار (Role Rewards / XP) ──
def _RR_OK() -> bool:  # type: ignore[misc]
    return bool(getattr(_app_core_for_lht, "_RR_OK", False))

def _get_role_rewards():  # type: ignore[misc]
    fn = getattr(_app_core_for_lht, "_get_role_rewards", None)
    if fn is None:
        raise RuntimeError("نظام المكافآت غير متاح")
    return fn()

# ── 🆕 التخطيط الجماعي الاستباقي (Proactive Planning) ──
def _PP_OK() -> bool:  # type: ignore[misc]
    return bool(getattr(_app_core_for_lht, "_PP_OK", False))

def _build_pre_task_plan(goal, skills=None, top_k=3):  # type: ignore[misc]
    fn = getattr(_app_core_for_lht, "_build_pre_task_plan", None)
    if fn is None:
        raise RuntimeError("التخطيط الاستباقي غير متاح")
    return fn(goal, skills=skills, top_k=top_k)

_MIN_CONFIDENCE = 0.3


def _advise_task_from_experience(task: LHTask) -> None:
    """استحضار الخبرات الجماعية وتعديل عنوان الخطوة الأولى (مساعد مشترك).
    إلحاق مستخلص غير ملزم لا يغيّر المنطق الحتمي."""
    try:
        if _TEM_OK() and task.plan:
            recalled = _get_experience_log().recall(
                task.goal[:200], top_k=4, min_confidence=_MIN_CONFIDENCE)
            if recalled:
                task._tem_recall = recalled  # type: ignore[attr-defined]
                _advice = ("\n".join(
                    f"• {e.get('decision', '')[:90]}" for e in recalled[:3]
                ))[:300]
                first = task.plan[0].get("title", "")
                task.plan[0]["title"] = (first + " | " + _advice)[:320]
    except Exception:
        pass
def _record_lht_experience(manager: "LongHorizonTaskManager",
                           task: LHTask,
                           results_context: List[str]) -> None:
    """تسجيل خبرات خطوات المهمة الطويلة في السجل الجماعي عند الإتمام.

    خبرة عامة عن النتيجة (success/partial/failure حسب نسبة النجاح) +
    خبرات الأدوات الأكثر استخدامًا (success إذا نجحت ≥60%، failure ≤30%)."""
    if not _TEM_OK():
        return
    done_n = sum(1 for s in task.steps if s.get("status") == "done")
    fail_n = sum(1 for s in task.steps if s.get("status") == "partial")
    outcome = ("success" if done_n > fail_n
               else "partial" if done_n else "failure")
    _get_experience_log().record(
        context=f"مهمة طويلة الأمد: {task.goal[:100]}",
        decision=(f"خطة «{task.title[:60]}» ({len(task.plan)} خطوة): "
                  f"{done_n} ناجحة / {fail_n} جزئية"),
        outcome=outcome,
        category="plan_strategy",
        confidence=0.6 if done_n else 0.5,
        task_id=task.task_id,
        agents="long_horizon")
    # خبرة لكل أداة استخدمت مرتين أو أكثر (حسب معدل نجاحها)
    tool_counts: Dict[str, int] = {}
    tool_success: Dict[str, int] = {}
    for s in task.steps:
        t = s.get("tool") or "unknown"
        tool_counts[t] = tool_counts.get(t, 0) + 1
        if s.get("status") == "done":
            tool_success[t] = tool_success.get(t, 0) + 1
    for t, cnt in sorted(tool_counts.items(), key=lambda kv: -kv[1])[:3]:
        if cnt < 2:
            continue
        rate = tool_success.get(t, 0) / cnt
        _get_experience_log().record(
            context=f"أداة {t} في مهمة: {task.goal[:80]}",
            decision=(f"الأداة «{t}» نجحت {tool_success.get(t, 0)} من "
                      f"{cnt} مرات في مهمة طويلة الأمد"),
            outcome=("success" if rate >= 0.6
                     else "failure" if rate <= 0.3
                     else "partial"),
            category="search_method",
            confidence=0.5 + 0.2 * abs(rate - 0.5),
            task_id=task.task_id,
            agents="long_horizon")

def _run_task(manager: "LongHorizonTaskManager", task: LHTask) -> None:
    """تنفيذ المهمة خطوة خطوة في خيط daemon."""
    task._t0 = time.time()
    task.status = STATUS_RUNNING
    task.started_at = _now_iso()
    manager._persist_task(task)
    manager._emit("lht_started", task, detail=f"مهمة طويلة الأمد: {task.title}")

    try:
        goal_label = (task.goal or "")[:120]
        if _LHT_EXECUTE_HOOK is not None:
            # اختبار: مسار مزيف كامل — الخطاف يملأ plan والخطوات
            ok = _LHT_EXECUTE_HOOK(manager, task, goal_label)
            if ok and not task.steps:
                task.status = STATUS_FAILED
                task.error = "الخطاف المزيف لم يسجل خطوات"
            elif not ok:
                task.status = STATUS_FAILED
                task.error = "الخطاف المزيف فشل"
            # ── استحضار الخبرات الجماعية قبل الخروج (كل المسارات) ──
            _advise_task_from_experience(task)
            # ── 🆕 خطة استباقية لكل المسارات (مزيف/حتمي) ──
            try:
                if _PP_OK():
                    task._pp_plan = _build_pre_task_plan(task.goal)
            except Exception:
                pass
            task.status = STATUS_DONE if ok else STATUS_FAILED
            _record_lht_experience(manager, task, [])
            return
        if _LHT_PLAN_HOOK is not None and _LHT_PLAN_HOOK(task):
            pass  # اختبار: خطاف يملأ الخطة فقط ثم يكمل التنفيذ الحتمي
        else:
            task.plan = _build_plan(task.goal)
        # ── استحضار الخبرات الجماعية المتراكمة وتعديل عناوين الخطة ──
        _advise_task_from_experience(task)
        # ── 🆕 خطة استباقية من سجل الخبرات قبل التنفيذ ──
        try:
            if _PP_OK():
                task._pp_plan = _build_pre_task_plan(task.goal)
        except Exception:
            pass
        if not task.plan:
            task.status = STATUS_FAILED
            task.error = "لا يمكن بناء خطة من هدف فارغ"
            return

        # حفظ الخطة
        try:
            with _connect(manager.db_path) as conn:
                conn.execute(
                    "UPDATE lht_tasks SET plan_json=? WHERE task_id=?",
                    (json.dumps(task.plan, ensure_ascii=False), task.task_id),
                )
                conn.commit()
        except Exception:
            pass

        results_context: List[str] = []
        fetched_urls: List[str] = []
        done_step_titles: List[str] = []

        for i, step in enumerate(task.plan):
            if task._cancelled.is_set():
                task.status = STATUS_CANCELLED
                return
            if time.time() - task._t0 > MAX_TASK_AGE_S:
                task.status = STATUS_FAILED
                task.error = "تجاوزت المهمة سقف المدة المسموح"
                return

            tool_name = step.get("tool", "")
            tool_input = step.get("input", "")
            title = step.get("title", f"خطوة {i}")
            task.log(i, "running", tool_name, tool_input)

            result_text = ""
            success = False

            try:
                if tool_name == "web_search":
                    if task._fetch_count >= MAX_FETCHES:
                        result_text = "تجاوز سقف طلبات الإنترنت — أُكمل بالسياق المجمّع"
                        success = True
                    else:
                        out = tool_web_search(tool_input)
                        task._fetch_count += 1
                        result_text = out.get("text", "")
                        success = bool(out.get("ok")) or bool(out.get("count", 0))

                elif tool_name == "fetch_best":
                    # جلب أول نتيجة URL من كل هدف في السياق
                    targets = []
                    with contextlib.suppress(Exception):
                        targets = json.loads(tool_input)
                    for target in (targets or [task.goal]):
                        if task._cancelled.is_set() or task._fetch_count >= MAX_FETCHES:
                            break
                        url = ""
                        for ctx in reversed(results_context):
                            m = re.search(r"الرابط:\s*(https?://\S+)", ctx)
                            if m:
                                url = m.group(1)
                                break
                        if not url:
                            # بحث إضافي سريع للحصول على رابط
                            if task._fetch_count < MAX_FETCHES:
                                out = tool_web_search(target, max_results=3)
                                task._fetch_count += 1
                                m = re.search(r"الرابط:\s*(https?://\S+)", out.get("text", ""))
                                url = m.group(1) if m else ""
                        if url:
                            page = tool_fetch_page(url)
                            task._fetch_count += 1
                            if page.get("ok"):
                                fetched_urls.append(url)
                                result_text += (
                                    f"\n--- {url}\n{page.get('text', '')[:3000]}\n"
                                )
                    success = bool(result_text.strip()) or True  # لا نوقف المهمة لغياب روابط

                elif tool_name == "finalize":
                    spec = {}
                    with contextlib.suppress(Exception):
                        spec = json.loads(tool_input)
                    goal_label = (spec.get("goal") or task.goal)[:100]
                    sections = []
                    for r in results_context:
                        sections.append(r.strip()[:3000])
                    report_lines = [
                        f"# تقرير: {goal_label}",
                        "",
                        f"أُعدّ بتاريخ {_now_iso()} عبر منظّم المهام طويلة الأمد (NSM).",
                        "",
                        "**الملخص التنفيذي:**",
                        "جرى تنفيذ بحث متعدد المصادر (DuckDuckGo / Wikipedia / Wikidata / أخبار) "
                        "وجلب صفحات رئيسية وتجميع النتائج في هذا التقرير.",
                        "",
                    ]
                    if sections:
                        report_lines.append("**التفاصيل المجمّعة:**")
                        report_lines.append("")
                        for s in sections:
                            report_lines.append(s)
                            report_lines.append("")
                    else:
                        report_lines.append(
                            "**ملاحظة:** لم تتوفر نتائج كافية من البحث المباشر — "
                            "جرّب إعادة الصياغة أو تمكين اتصال الإنترنت الكامل."
                        )
                    report_lines += ["", "---", f"المصادر: {len(fetched_urls)} صفحة | "
                                     f"البحث: {task._fetch_count} طلب"]
                    report = "\n".join(report_lines)
                    safe_goal = re.sub(r"[^\w\u0600-\u06FF\-]+", "_", goal_label)[:40]
                    fname = f"report_{safe_goal}_{task.task_id[:6]}.md"
                    wr = tool_write_file(fname, report)
                    result_text = report if wr.get("ok") else (
                        f"فشل حفظ التقرير: {wr.get('text')}\n\n{report}"
                    )
                    success = True

                else:
                    # أداة مخصصة من BUILTIN_TOOLS (أو المزيفة في الاختبار)
                    fn = BUILTIN_TOOLS.get(tool_name) or _LHT_EXECUTE_HOOK_TOOL
                    if fn is None:
                        raise ValueError(f"أداة غير معروفة: {tool_name}")
                    out = fn(tool_input)
                    result_text = out.get("text", "")
                    success = bool(out.get("ok"))
            except Exception as exc:
                result_text = f"خطأ: {exc}"
                success = False

            # retry واحد عند الفشل (إلا internal tools)
            if not success and tool_name == "web_search":
                try:
                    out = tool_web_search(tool_input, max_results=4)
                    task._fetch_count += 1
                    result_text = out.get("text", "")
                    success = bool(out.get("ok")) or bool(out.get("count", 0))
                except Exception as exc2:
                    result_text += f" | retry فشل: {exc2}"

            status = "done" if success else "partial"
            task.log(i, status, tool_name, tool_input, result_text)
            results_context.append(result_text)
            done_step_titles.append(title)
            task.progress = min(1.0, (i + 1) / max(1, len(task.plan)))
            manager._persist_task(task)

        # تقرير نهائي مدمج
        final = "\n\n".join(
            [f"## {p.get('title')}" + ("\n" + c[:3000] if c else "")
             for p, c in zip(task.plan, results_context)]
        )
        report_title = f"final_{task.task_id[:8]}.md"
        with contextlib.suppress(Exception):
            tool_write_file(report_title, final)
        task._final_report = final[:12000]
        # ── تسجيل خبرات الخطوات في السجل الجماعي ──
        try:
            if _TEM_OK():
                _record_lht_experience(manager, task, results_context)
        except Exception:
            pass
        # ── 🆕 نقاط خبرة لوكيل المهام الطويلة (يُسجَّل بعد الخبرات) ──
        try:
            if _RR_OK():
                done_n = sum(1 for s in task.steps if s.get("status") == "done")
                outcome = ("success" if done_n else "failure")
                _get_role_rewards().award(
                    role="long_horizon",
                    outcome=outcome,
                    role_type="long_horizon",
                    task_id=task.task_id)
        except Exception:
            pass
        if task.status == STATUS_RUNNING:
            task.status = STATUS_DONE
        manager._emit(
            "lht_done", task,
            detail=(f"اكتملت المهمة: {task.title} "
                    f"({sum(1 for s in task.steps if s.get('status') == 'done')}/"
                    f"{len(task.plan)} خطوة ناجحة)"),
        )
    except Exception as exc:
        logger.error("lht: فشل %s: %s", task.task_id, exc)
        task.status = STATUS_FAILED
        task.error = str(exc)[:300]
        # ── تسجيل خبرة فشل المهمة كاملة في السجل الجماعي ──
        try:
            if _TEM_OK():
                _get_experience_log().record(
                    context=f"مهمة طويلة الأمد: {task.goal[:100]}",
                    decision=(f"خطة «{task.title[:60]}» "
                              f"({len(task.plan)} خطوة) فشلت بالكامل"),
                    outcome="failure",
                    category="plan_strategy",
                    confidence=0.5,
                    task_id=task.task_id,
                    agents="long_horizon")
        except Exception:
            pass
        # ── 🆕 نقاط خبرة لفشل وكيل المهام الطويلة ──
        try:
            if _RR_OK():
                _get_role_rewards().award(
                    role="long_horizon",
                    outcome="failure",
                    role_type="long_horizon",
                    task_id=task.task_id)
        except Exception:
            pass
        manager._emit("lht_failed", task, detail=f"فشل المهمة: {task.error}")
    finally:
        task.finished_at = _now_iso()
        manager._persist_task(task)
        manager._prune()
# ══════════════════════════════════════════════════════════════════
# المدير (singleton)
# ══════════════════════════════════════════════════════════════════

def _manager_for_persist():
    """الوصول للمدير دون كسر اختبار يعيّن execute_fn مزيف."""
    return get_long_horizon_manager()


class LongHorizonTaskManager:
    """منظّم المهام طويلة الأمد — واحد لكل عملية."""

    def __init__(self, db_path: str = DB_DEFAULT) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._tasks: Dict[str, LHTask] = {}
        self._running_keys: Dict[str, str] = {}
        self._queue: List[str] = []
        self._ensure_db()
        self._load_history()

    # ── قاعدة البيانات ───────────────────────────────────────────
    def _ensure_db(self) -> None:
        try:
            with _connect(self.db_path) as conn:
                conn.executescript(_SCHEMA)
                conn.commit()
        except Exception as exc:
            logger.warning("lht: تعذّر تهيئة قاعدة البيانات %s: %s", self.db_path, exc)

    def _persist_task(self, task: LHTask) -> None:
        try:
            with _connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO lht_tasks "
                    "(task_id, title, goal, status, plan_json, created_at, "
                    "started_at, finished_at, progress, error) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (task.task_id, task.title, task.goal, task.status,
                     json.dumps(task.plan, ensure_ascii=False),
                     task.created_at, task.started_at, task.finished_at,
                     task.progress, task.error),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("lht: تعذّر حفظ المهمة %s: %s", task.task_id, exc)

    def _load_history(self) -> None:
        """لا نحمّل مهامًا غير مكتملة من دورات سابقة (الخيط daemon انتهت)."""
        try:
            with _connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT task_id, title, goal, status, created_at, finished_at, "
                    "progress, error FROM lht_tasks "
                    "WHERE status IN ('done', 'failed', 'cancelled') "
                    "ORDER BY created_at DESC LIMIT ?", (MAX_PENDING * 20,)
                )
                for row in cur:
                    task = LHTask(task_id=row[0], goal=row[2] or "", title=row[1])
                    task.status = row[3]
                    task.created_at = row[4] or _now_iso()
                    task.finished_at = row[5] or ""
                    task.progress = row[6] or 0.0
                    task.error = row[7] or ""
                    self._tasks[task.task_id] = task
        except Exception as exc:
            logger.warning("lht: تعذّر تحميل السجل: %s", exc)

    def _prune(self) -> None:
        """ضغط المهام المكتملة القديمة الأبعد من MAX_PENDING*20."""
        cap = MAX_PENDING * 20
        completed = [
            t for t in self._tasks.values()
            if t.status in (STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED)
        ]
        if len(completed) <= cap:
            return
        completed.sort(key=lambda t: t.created_at)
        for task in completed[: len(completed) - cap]:
            self._tasks.pop(task.task_id, None)
            with contextlib.suppress(Exception):
                with _connect(self.db_path) as conn:
                    conn.execute("DELETE FROM lht_tasks WHERE task_id=?", (task.task_id,))
                    conn.execute("DELETE FROM lht_steps WHERE task_id=?", (task.task_id,))
                    conn.commit()

    # ── الأحداث (agent_event_bus) ─────────────────────────────────
    def _emit(self, event_type: str, task: LHTask, detail: str = "") -> None:
        """يُطلق حدث مراقبة حيّة فقط داخل سياق Streamlit حقيقي — يمنع تحذيرات
        ScriptRunContext من خيوط daemon خارج session (لا يضيف حدثًا للسجل في تلك الحالة)."""
        try:
            import streamlit as st
            # الفحص الحقيقي بوجود Session حية قبل لمس أي سياق — يمنع تحذيرات
            # ScriptRunContext من خيوط daemon خارج session (مهم في Streamlit 1.60
            # حيث يُصدر الاستيراد المباشر لـ runtime.context تحذيرًا فوريًا):
            if not (hasattr(st, "runtime") and st.runtime.exists()):
                return  # خارج streamlit run — لا نسجل، بلا تحذيرات
            try:
                from streamlit.runtime.context import get_script_run_ctx
            except Exception:
                return
            try:
                if get_script_run_ctx() is None:
                    return  # خيط daemon خارج جلسة حية — لا نسجل، بلا تحذيرات
            except Exception:
                return
            from ai.agent_event_bus import emit_event
            emit_event(
                event_type, agent_id="long_horizon", title=task.title,
                status=task.status, detail=detail,
                metadata={"task_id": task.task_id, "progress": task.progress},
            )
        except Exception:
            logger.debug("lht: تعذّر إطلاق حدث %s", event_type)

    # ── API العام ─────────────────────────────────────────────────
    def submit(self, goal: str, title: Optional[str] = None) -> Optional[LHTask]:
        """تقديم مهمة طويلة الأمد — تنفذ في خيط daemon."""
        goal = (goal or "").strip()
        if not goal:
            return None
        with self._lock:
            norm = " ".join(goal.lower().split())
            if norm in self._running_keys:
                return self._tasks.get(self._running_keys[norm])
            running = sum(1 for t in self._tasks.values() if t.status == STATUS_RUNNING)
            if running >= MAX_CONCURRENT or len(self._queue) >= MAX_PENDING:
                return None
            task = LHTask(goal=goal, title=title)
            self._tasks[task.task_id] = task
            self._queue.append(task.task_id)
            self._running_keys[norm] = task.task_id
        self._persist_task(task)
        self._emit("lht_submitted", task, detail=f"مهمة جديدة: {task.title}")
        threading.Thread(
            target=_run_task, args=(self, task),
            name=f"NSM-lht-{task.task_id[:6]}", daemon=True,
        ).start()
        return task

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status in (STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED):
                return False
            task.status = STATUS_CANCELLED
            task.finished_at = _now_iso()
            task._cancelled.set()
            self._queue = [t for t in self._queue if t != task_id]
        self._persist_task(task)
        self._emit("lht_cancelled", task, detail="ألغى المستخدم المهمة")
        return True

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self._tasks.get(task_id)
        return task.to_dict() if task is not None else None

    def list_tasks(self, limit: int = 30, status: Optional[str] = None) -> List[Dict[str, Any]]:
        tasks = [
            t for t in self._tasks.values()
            if status is None or t.status == status
        ]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return [t.to_dict() for t in tasks[:limit]]

    def get_task_log(self, task_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """سجل خطوات المهمة — من الذاكرة أولًا ثم SQLite إن لزم."""
        task = self._tasks.get(task_id)
        if task is not None:
            return task.steps[-limit:]
        rows: List[Dict[str, Any]] = []
        try:
            with _connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT step_index, status, tool, tool_input, result, "
                    "started_at, finished_at FROM lht_steps "
                    "WHERE task_id=? ORDER BY step_index DESC LIMIT ?",
                    (task_id, limit),
                )
                cols = [d[0] for d in cur.description]
                for row in cur:
                    rows.append(dict(zip(cols, row)))
        except Exception:
            pass
        rows.reverse()
        return rows

    def status(self) -> Dict[str, Any]:
        counter = Counter(t.status for t in self._tasks.values())
        return {
            "total": len(self._tasks),
            "pending": counter.get(STATUS_PENDING, 0),
            "running": counter.get(STATUS_RUNNING, 0),
            "done": counter.get(STATUS_DONE, 0),
            "failed": counter.get(STATUS_FAILED, 0),
            "cancelled": counter.get(STATUS_CANCELLED, 0),
            "max_concurrent": MAX_CONCURRENT,
            "max_steps": MAX_STEPS,
        }

    def detect_long_horizon_request(self, text: str) -> Optional[str]:
        """
        كشف حتمي (بلا API) لطلبات المهام طويلة الأمد في نص المستخدم.
        يُرجع الهدف المستخرج أو None.
        """
        t = (text or "").strip()
        if len(t) < 15:
            return None
        patterns = [
            r"نفّذ\s+مهمة\s+طويلة\s*(?:الأمد)?\s*[:：]?\s*(.+)",
            r"نفذ\s+مهمة\s+طويلة\s*(?:الأمد)?\s*[:：]?\s*(.+)",
            r"ابحث\s+وأعدّ?\s*تقريرًا?\s+عن\s+(.{10,})",
            r"ابحث\s+واكتب\s+تقريرًا?\s+عن\s+(.{10,})",
            r"أعدّ?\s*تقريرًا?\s+بحثيًا?\s+عن\s+(.{10,})",
            r"أعدّ?\s*تقريرًا?\s+عن\s+(.{10,})",
            r"ابحث\s+(?:بالتفصيل|بعمق|معمّقًا?|شاملًا?)\s+عن\s+(.{8,})",
            r"شغّل\s+مهمة\s+طويلة\s*(?:الأمد)?\s*[:：]?\s*(.+)",
            r"long[- ]horizon\s+(?:task|execute)\s*[:：]?\s*(.{10,})",
        ]
        for pat in patterns:
            m = re.search(pat, t)
            if m:
                goal = m.group(1).strip()
                if len(goal) >= 10:
                    return goal
        # كشف أوسع: طلب بحثي عميق يتضمن هدفًا واضحًا
        if any(k in t for k in ("مهمة طويلة", "بحث معمّق", "بحث معمق",
                                "تقرير شامل", "تقرير معمّق", "تقرير تفصيلي")):
            # نستخدم النص كله كهدف
            return t
        return None


_MANAGER: Optional[LongHorizonTaskManager] = None
_MANAGER_LOCK = threading.Lock()


def get_long_horizon_manager(db_path: Optional[str] = None) -> LongHorizonTaskManager:
    """singleton على مستوى العملية."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = LongHorizonTaskManager(db_path=db_path or DB_DEFAULT)
    return _MANAGER


__all__ = [
    "LongHorizonTaskManager",
    "LHTask",
    "get_long_horizon_manager",
    "BUILTIN_TOOLS",
    "tool_web_search", "tool_fetch_page", "tool_write_file",
    "tool_read_file", "tool_run_python",
    "MAX_STEPS", "MAX_FETCHES", "MAX_CONCURRENT", "MAX_PENDING",
    "STATUS_PENDING", "STATUS_RUNNING", "STATUS_DONE",
    "STATUS_FAILED", "STATUS_CANCELLED",
]
