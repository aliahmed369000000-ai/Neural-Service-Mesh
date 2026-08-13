# -*- coding: utf-8 -*-
"""تعاون الوكلاء في المهام طويلة الأمد (NSM Collaborative Long-Horizon Tasks).

يتيح للوكلاء التخصصيين العمل معًا على مهمة واحدة مركّبة:
- المفكّك يحسّم المهمة المركّبة إلى أدوار متوازية (باحث/محلّل/مدقق...)
- كل دور وكيلٌ متخصصٌ ينفَّذ في خيط مستقل عبر أدوات الإنترنت الآمنة
  (نفس أدوات long_horizon_tasks: بحث، جلب صفحات، ملفات معزولة، بايثون محمي)
- اجتماع التجميع يدمج نتائج الأدوار في تقرير موحد بمصادر كل دور
- ناقل الأحداث المشترك يسجّل تعاون الأدوار (طلب/بدء/إنجاز/جمع)

التكامل: استيراد اختياري (_COOP_OK) في app_core — أي فشل يعيد السلوك الأصلي.

الحوكمة:
- MAX_COLLAB_ROLES=4 أدوار كحد أقصى لكل مهمة تعاونية
- MAX_ROLE_STEPS=12 خطوة لكل دور
- MAX_ROLE_FETCHES=20 طلب إنترنت لكل دور
- MAX_COLLAB_CONCURRENT=2 مهمة تعاونية متزامنة (ضمن السقف العام)
- سقف زمني MAX_COLLAB_AGE_S=900 ثانية للمهمة التعاونية
- كل دور له مساحة ملفات معزولة data/lht_workspace/collab_{id}/role_{name}/
"""
import contextlib
import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nsm.collab")

# ─────────────────────────────────────────────────────────────────────
# الحوكمة
# ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DEFAULT = os.path.join(BASE_DIR, "data", "collaborative_tasks.db")

MAX_COLLAB_ROLES = 4        # أدوار كحد أقصى لكل مهمة تعاونية
MAX_ROLE_STEPS = 12         # خطوات لكل دور
MAX_ROLE_FETCHES = 20       # طلبات إنترنت لكل دور
MAX_COLLAB_CONCURRENT = 2   # مهام تعاونية متزامنة
MAX_COLLAB_AGE_S = 900      # سقف زمني للمهمة التعاونية

# حالات الدور
ROLE_PENDING = "pending"
ROLE_RUNNING = "running"
ROLE_DONE = "done"
ROLE_FAILED = "failed"

# حالات المهمة التعاونية
COLLAB_PENDING = "pending"
COLLAB_RUNNING = "running"
COLLAB_GATHERING = "gathering"
COLLAB_DONE = "done"
COLLAB_FAILED = "failed"
COLLAB_CANCELLED = "cancelled"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _connect(db_path: str):
    import sqlite3
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _init_db(db_path: str) -> None:
    with _connect(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS collab_tasks (
            task_id TEXT PRIMARY KEY,
            title TEXT, goal TEXT, status TEXT,
            plan_json TEXT, created_at TEXT, started_at TEXT, finished_at TEXT,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS collab_role_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL, role TEXT NOT NULL, step_index INTEGER,
            status TEXT, tool TEXT, tool_input TEXT, result TEXT,
            started_at TEXT, finished_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_collab_steps_task ON collab_role_steps(task_id);
        """)
        conn.commit()


# ─────────────────────────────────────────────────────────────────────
# الأدوات الآمنة (مستعارة من وحدة المهام الطويلة)
# ─────────────────────────────────────────────────────────────────────

def _safe_lht_tools():
    """جلب أدوات الإنترنت الآمنة — يرجع dict فارغًا عند فشل الاستيراد."""
    try:
        from ai.long_horizon_tasks import (  # type: ignore
            tool_web_search, tool_fetch_page,
            tool_write_file as _twf, tool_read_file as _trf,
            tool_run_python as _trp,
        )
        return {
            "web_search": tool_web_search,
            "fetch_page": tool_fetch_page,
            "write_file": _twf,
            "read_file": _trf,
            "run_python": _trp,
        }
    except Exception:
        return {}


ROLE_TOOLS: Dict[str, Any] = {}  # يُملأ عند أول استخدام
ROLE_TOOLS_LOCK = threading.Lock()


def _get_role_tools() -> Dict[str, Any]:
    global ROLE_TOOLS
    if not ROLE_TOOLS:
        with ROLE_TOOLS_LOCK:
            if not ROLE_TOOLS:
                ROLE_TOOLS = _safe_lht_tools()
    return ROLE_TOOLS


def role_web_search(role_ws: str, query: str, max_results: int = 6,
                    fetch_state: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """بحث ويب محكوم داخل دور (يعدّ من سقف طلبات الدور)."""
    tools = _get_role_tools()
    fn = tools.get("web_search")
    if fn is None:
        return {"ok": False, "text": "أدوات البحث غير متاحة حاليًا"}
    if fetch_state is not None and fetch_state.get("count", 0) >= MAX_ROLE_FETCHES:
        return {"ok": False, "text": "تجاوز الدور سقف طلبات الإنترنت"}
    out = fn(query, max_results=min(max_results, 6))
    if fetch_state is not None:
        fetch_state["count"] = fetch_state.get("count", 0) + 1
    return out


def role_fetch_page(role_ws: str, url: str,
                    fetch_state: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """جلب صفحة محكوم داخل دور."""
    tools = _get_role_tools()
    fn = tools.get("fetch_page")
    if fn is None:
        return {"ok": False, "text": "أدوات الجلب غير متاحة حاليًا"}
    if fetch_state is not None and fetch_state.get("count", 0) >= MAX_ROLE_FETCHES:
        return {"ok": False, "text": "تجاوز الدور سقف طلبات الإنترنت"}
    out = fn(url)
    if fetch_state is not None:
        fetch_state["count"] = fetch_state.get("count", 0) + 1
    return out


def role_write_file(role_ws: str, name: str, content: str) -> Dict[str, Any]:
    """كتابة ملف في مساحة الدور المعزولة (داخل lht_workspace/collab_{id}/)."""
    tools = _get_role_tools()
    fn = tools.get("write_file")
    if fn is None:
        return {"ok": False, "text": "أدوات الملفات غير متاحة حاليًا"}
    safe_name = re.sub(r"[^\w\u0600-\u06FF\-\.]+", "_", (name or "file")[:40])
    out = fn(os.path.join("collab_ws", safe_name), content)
    if out.get("ok") and role_ws:
        # إعادة الكتابة في مسار الدور المادي لضمان العزل
        full = os.path.join(role_ws, safe_name)
        try:
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content[:200_000])
        except Exception:
            pass
    return out


def role_run_python(code: str, timeout: int = 30) -> Dict[str, Any]:
    """بايثون محمي داخل الدور (يرث قواعد tool_run_python)."""
    tools = _get_role_tools()
    fn = tools.get("run_python")
    if fn is None:
        return {"ok": False, "text": "أدوات البايثون غير متاحة حاليًا"}
    return fn(code, timeout=timeout)


# ══════════════════════════════════════════════════════════════════
# المفكّك الحتمي (بلا API/LLM)
# ══════════════════════════════════════════════════════════════════

# روابط مركّبة بين هدفين أو أكثر
CONJUNCT_SPLITTERS = [
    r"\s+و\s+",               # و مع مسافات (واثق تمامًا)
    r"\s+ثم\s+",              # ثم (الترتيب)
    r"\s+بالإضافة\s+إلى\s+",   # بالإضافة إلى
    r"\s+أيضًا\s+",            # أيضًا
    r"\s+وكذلك\s+",            # وكذلك
    r"\s+علاوةً?\s+على\s+",    # علاوة على
    r"\s?و\s+",               # و ملتصقة بما قبلها فقط
    r"\s+و\s?",               # و ملتصقة بما بعدها فقط
    r"\s*،\s*و\s+",            # ، و
]
CONJUNCT_MIN_LEN = 5          # طول أدنى لكل جزء حتى يُقبل التقسيم (يمنع كسر الكلمات)


def _split_subgoals(text: str) -> List[str]:
    """تقسيم الهدف المركّب إلى أهداف فرعية عند الروابط العربية.

    يقبل التقسيم فقط إذا كان كل جزء >= CONJUNCT_MIN_LEN حتى لا نكسر
    الكلمات العربية الملصوقة بواو العطف (مثل «ابن الهيثم»)."""
    parts = [text]
    for pat in CONJUNCT_SPLITTERS:
        new_parts: List[str] = []
        for p in parts:
            pieces = re.split(pat, p)
            if len(pieces) > 1 and all(len(x.strip()) >= CONJUNCT_MIN_LEN
                                       for x in pieces):
                new_parts.extend(x.strip() for x in pieces)
            else:
                new_parts.append(p)
        parts = new_parts
    return [p.strip() for p in parts
            if len(p.strip()) >= CONJUNCT_MIN_LEN][:MAX_COLLAB_ROLES]


# كلمات الكشف الحتمي عن مهمة تعاونية مركّبة
COLLAB_KEYWORDS = (
    "مقارنة بين", "الفرق بين", "وأعدّ", "و اكتب", "وثمّ",
    "بالإضافة إلى", "بالإضافة الى",
)


def _collab_patterns():
    return [
        # و ملتصقة بلا مسافات في العربية («الاصطناعي وأثره») — non-greedy أولًا
        r"أعدّ?\s*تقريرًا?[\s،]*(?:شاملًا?|معمّقًا?|مركّبًا?)?\s*عن\s+(.{8,}?\s?و\s?.{5,})",
        r"ابحث[\s،]+وأعدّ?\s*تقريرًا?\s+عن\s+(.{8,}?\s?و\s?.{5,})",
        r"أعدّ?\s*تقريرًا?[\s،]*(?:شاملًا?|معمّقًا?)?\s*عن\s+(.{8,}?)\s+ثم\s+(.{8,})",
        r"قارن\s+(?:بين\s+)?(.{6,}?(?:\s|\s?و\s?).{6,})",
        r"ما\s+الفرق\s+بين\s+(.{5,}?(?:\s|\s?و\s?).{5,})",
        r"مقارنة\s+(?:بين\s+)?(.{6,}?\s?و\s?.{6,})",
        r"أحلّل\s+(?:و|\s*ع)\s*(.{8,}?\s?و\s?.{5,})",
    ]


def detect_collaborative_request(text: str) -> Optional[List[str]]:
    """كشف حتمي (بلا API) للمهام التعاونية المركّبة.

    يرجع قائمة الأهداف الفرعية (2+) أو None.
    """
    t = (text or "").strip()
    if len(t) < 18:
        return None
    for pat in _collab_patterns():
        m = re.search(pat, t)
        if m:
            if m.lastindex and m.lastindex > 1:
                # نمط صريح بجزأين (مثل «...عن X ثم Y»)
                goals = [g.strip() for g in m.groups() if g and len(g.strip()) >= 5]
            else:
                goals = _split_subgoals(m.group(1).strip())
            if len(goals) >= 2:
                return goals[:MAX_COLLAB_ROLES]
    if any(k in t for k in COLLAB_KEYWORDS):
        goals = _split_subgoals(t)
        if len(goals) >= 2:
            return goals[:MAX_COLLAB_ROLES]
    return None


# ══════════════════════════════════════════════════════════════════
# مهمة تعاونية ودورها
# ══════════════════════════════════════════════════════════════════

class CollabRole:
    """دور وكيل متخصص داخل مهمة تعاونية."""

    def __init__(self, task_id: str, name: str, goal: str) -> None:
        self.task_id = task_id
        self.name = name
        self.goal = goal
        self.status = ROLE_PENDING
        self.steps: List[Dict[str, Any]] = []
        self.fetch_state: Dict[str, int] = {"count": 0}
        self.started_at = ""
        self.finished_at = ""
        self.error = ""

    def log(self, step_index: int, status: str, tool: str = "",
            tool_input: str = "", result: str = "") -> None:
        entry = {
            "step_index": step_index,
            "status": status,
            "tool": tool,
            "tool_input": (tool_input or "")[:500],
            "result": (result or "")[:2000],
            "started_at": _now_iso(),
            "finished_at": _now_iso(),
        }
        self.steps.append(entry)
        self._persist_step(entry)

    def _persist_step(self, entry: Dict[str, Any]) -> None:
        try:
            manager = _get_manager_for_persist()
            if manager is not None:
                with _connect(manager.db_path) as conn:
                    conn.execute(
                        "INSERT INTO collab_role_steps (task_id, role, "
                        "step_index, status, tool, tool_input, result, "
                        "started_at, finished_at) VALUES (?,?,?,?,?,?,?,?,?)",
                        (self.task_id, self.name, entry["step_index"],
                         entry["status"], entry["tool"], entry["tool_input"],
                         entry["result"], entry["started_at"],
                         entry["finished_at"]),
                    )
                    conn.commit()
        except Exception as exc:
            logger.debug("collab: تعذّر حفظ خطوة الدور %s: %s", self.name, exc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "goal": self.goal,
            "status": self.status,
            "steps": self.steps,
            "fetch_count": self.fetch_state.get("count", 0),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


class CollaborativeTask:
    """مهمة تعاونية: أهداف فرعية → أدوار متوازية → اجتماع تجميع → تقرير موحد."""

    def __init__(self, task_id: Optional[str] = None, goal: str = "",
                 subgoals: Optional[List[str]] = None,
                 title: Optional[str] = None) -> None:
        self.task_id = task_id or uuid.uuid4().hex[:12]
        self.goal = (goal or "").strip()
        self.subgoals = subgoals or []
        self.title = (title or self.goal[:70] or "مهمة تعاونية")
        self.status = COLLAB_PENDING
        self.roles: List[CollabRole] = []
        self.created_at = _now_iso()
        self.started_at = ""
        self.finished_at = ""
        self.error = ""
        self.progress = 0.0
        self._synthesis = ""
        self._cancelled = threading.Event()
        self._t0 = 0.0
        self._role_dirs: Dict[str, str] = {}  # مسار مساحة كل دور

    def assign_roles(self) -> None:
        """تعيين الأدوار: دور «باحث» لكل هدف فرعي + دور «مدقق النتائج»."""
        for i, sg in enumerate(self.subgoals):
            self.roles.append(CollabRole(
                self.task_id, f"باحث {i + 1}", sg))
        self.roles.append(CollabRole(self.task_id, "مدقق النتائج", ""))
        self.roles = self.roles[:MAX_COLLAB_ROLES]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "goal": self.goal,
            "subgoals": self.subgoals,
            "status": self.status,
            "roles": [r.to_dict() for r in self.roles],
            "progress": round(self.progress, 2),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "duration_s": round(time.time() - self._t0, 1) if self._t0 else 0.0,
            "synthesis": self._synthesis,
        }


# ══════════════════════════════════════════════════════════════════
# خطافات الاختبار (محاكاة دون إنترنت/LLM)
# ══════════════════════════════════════════════════════════════════

_COLLAB_ROLE_HOOK = None  # fn(manager, task, role) -> bool (يملأ role.steps)


def _set_collab_role_hook(fn: Optional[Any]) -> None:
    """إحلال مسار الدور بالاختبار. None يعيد المسار الأصلي."""
    global _COLLAB_ROLE_HOOK
    _COLLAB_ROLE_HOOK = fn


# ══════════════════════════════════════════════════════════════════
# تكامل مع app_core: ناقل المعرفة المشترك (SKB) + سجل الخبرات (TEM)
# globals آمنة مع fallback — أي فشل استيراد يعيد السلوك الأصلي بلا
# NameError صامت (كان الـtry السابق يبتلع NameError دون تشغيل فعلي).
# ══════════════════════════════════════════════════════════════════
# استيراد متأخر (late import) لتجنب circular import:
# app_core يستورد هذه الوحدة سطر 377 — قبل أن يُعرّف _SKB_OK/_TEM_OK
# (سطور 391-412). لذا نجلب الأسماء عند الحاجة داخل دالة واحدة.
import app_core as _app_core_for_tem  # noqa: E402

def _SKB_OK() -> bool:  # type: ignore[misc]
    return bool(getattr(_app_core_for_tem, "_SKB_OK", False))

def _get_skb():  # type: ignore[misc]
    return getattr(_app_core_for_tem, "_get_skb", lambda: (_ for _ in ()).throw(
        RuntimeError("ناقل المعرفة المشترك غير متاح")))()

def _TEM_OK() -> bool:  # type: ignore[misc]
    return bool(getattr(_app_core_for_tem, "_TEM_OK", False))

def _get_experience_log():  # type: ignore[misc]
    fn = getattr(_app_core_for_tem, "_get_experience_log", None)
    if fn is None:
        raise RuntimeError("سجل الخبرات الجماعية غير متاح")
    return fn()

def _share_role_finding(task_id: str, role_name: str, text: str,
                        tool: str, source: str, index: int) -> None:
    """يشارك الدور نتيجته في الناقل المشترك (فشل صامت = لا شيء)."""
    try:
        if _SKB_OK():
            _get_skb().share_finding(task_id, role_name, text, tool,
                                     source, index)
    except Exception:
        pass


def _share_peer_knowledge(role: CollabRole, task_id: str,
                          target: str) -> None:
    """يستحضر ما وجده الزملاء عن الهدف ويثبّته في سجل الدور كخطوة تمهيدية.

    بهذا يجد كل دور معرفة زملائه قبل بحثه — تعاون فعلي لا عمل منعزل."""
    try:
        if not _SKB_OK():
            return
        peers = _get_skb().query_knowledge(target, task_id=task_id, k=4)
        peers = [p for p in peers
                 if p.get("role") != role.name and p.get("text")]
        if peers:
            lines = ["ما وجده الزملاء في الناقل المشترك:"]
            for j, p in enumerate(peers[:4]):
                lines.append(f"[{p.get('role', '?')} — "
                             f"{p.get('tool', '')}]: "
                             f"{(p.get('text') or '')[:300]}")
            role.log(len(role.steps), "done", "skb_peer_lookup",
                     target, "\n".join(lines))
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
# محرك تنفيذ الدور (التعاون الفعلي)
# ══════════════════════════════════════════════════════════════════

def _run_role(manager: "CollaborativeManager", task: CollaborativeTask,
              role: CollabRole) -> None:
    """تنفيذ دور وكيل في خيط مستقل."""
    role.status = ROLE_RUNNING
    role.started_at = _now_iso()
    manager._persist_collab(task)
    manager._emit("collab_role_started", task,
                  detail=f"بدأ الباحث {role.name} هدفه: {role.goal[:60]}")
    role_ws = task._role_dirs.get(role.name, "")
    try:
        if _COLLAB_ROLE_HOOK is not None and _COLLAB_ROLE_HOOK(manager, task, role):
            return  # مسار الاختبار المزيّف
        # الخطة الحتمية للدور: بحث ← جلب ← توثيق
        targets = _extract_role_targets(role.goal)
        for i, target in enumerate(targets):
            if task._cancelled.is_set():
                role.status = ROLE_FAILED
                role.error = "ألغيت المهمة"
                return
            if time.time() - task._t0 > MAX_COLLAB_AGE_S:
                role.status = ROLE_FAILED
                role.error = "تجاوزت المهمة سقف المدة"
                return
            if i >= MAX_ROLE_STEPS:
                break
            found = 0
            # ── استحضار ما شاركه الزملاء في الناقل المشترك ──
            _share_peer_knowledge(role, task.task_id, target)
            out = role_web_search(role_ws, target, max_results=4,
                                  fetch_state=role.fetch_state)
            result_text = out.get("text", "")
            success = bool(out.get("ok")) and bool(out.get("count", 0))
            role.log(i, "done" if success else "partial",
                     "web_search", target, result_text)
            if success:
                # ── مشاركة النتيجة في الناقل المشترك ──
                _share_role_finding(task.task_id, role.name,
                                    result_text, "web_search",
                                    role.steps[-1].get("source", ""), i)
                # ── تسجيل خبرة: أسلوب البحث نجح على هذا الهدف ──
                _record_role_experience(task, role, i,
                                        "web_search", "success",
                                        result_text[:180])
            if success:
                found += 1
            # جلب أهم نتيجة
            if found:
                m = re.search(r"https?://\S+", result_text)
                if m:
                    fp = role_fetch_page(role_ws, m.group(0),
                                         fetch_state=role.fetch_state)
                    role.log(i + 1, "done" if fp.get("ok") else "partial",
                             "fetch_page", m.group(0)[:100],
                             fp.get("text", "")[:600])
                    if fp.get("ok") and fp.get("text"):
                        _share_role_finding(
                            task.task_id, role.name,
                            fp.get("text", "")[:1800], "fetch_page",
                            m.group(0), i + 1)
            # توثيق نتائج الدور في ملفه المعزول
            doc = f"# {role.name}\nالهدف: {role.goal}\n\n{result_text[:2500]}\n"
            role_write_file(role_ws, f"role_{role.name}_{i}.md", doc)
        role.status = ROLE_DONE
    except Exception as exc:
        role.status = ROLE_FAILED
        role.error = str(exc)[:300]
        # ── تسجيل خبرة: الدور فشل — تحذير للمستقبل ──
        with contextlib.suppress(Exception):
            if _TEM_OK():
                _get_experience_log().record(
                    context=(f"مهمة تعاونية: {task.goal[:100]} "
                             f"(دور: {role.name})"),
                    decision=(f"أُضيف دور «{role.name}» وهدفه "
                              f"({role.goal[:80]}) وفشل التنفيذ"),
                    outcome="failure",
                    category="role_assign",
                    confidence=0.5,
                    task_id=task.task_id,
                    agents=role.name)
    finally:
        role.finished_at = _now_iso()
        manager._emit(
            "collab_role_done", task,
            detail=f"أنجز {role.name} ({len([s for s in role.steps if s.get('status') == 'done'])} خطوة)",
        )
        manager._persist_collab(task)


def _advise_roles_from_experience(task: CollaborativeTask) -> None:
    """استحضار الخبرات الجماعية المرتبطة بهدف المهمة قبل التخطيط.

    تُحفظ في task._tem_recall ليستخدمها توليف التقرير، ولا تغيّر
    المنطق الحتمي (إلحاق مستخلص غير مُلزم لأهداف الأدوار اللاحقة)."""
    recalled = _get_experience_log().recall(
        task.goal[:200], top_k=5, min_confidence=_MIN_CONFIDENCE)
    task._tem_recall = recalled  # type: ignore[attr-defined]
    if recalled:
        _summary = "\n".join(
            f"• {e.get('decision', '')[:120]}" for e in recalled[:5])
        # توصية عامة تُوثق في المهمة (لا تُعدّل subgoals الحتمي)
        task._tem_advice = (  # type: ignore[attr-defined]
            "توصيات من الخبرات الجماعية السابقة:\n" + _summary)[:500]


def _record_role_experience(task: CollaborativeTask, role: CollabRole,
                            step_index: int, tool: str,
                            outcome: str, detail: str) -> None:
    """تسجيل خبرة خطوة ناجحة/فاشلة لزيادة رصيد المعرفة الجماعية."""
    try:
        if not _TEM_OK():
            return
        if step_index >= _STEP_MAX:
            return
        _get_experience_log().record(
            context=f"مهمة تعاونية: {task.goal[:100]}",
            decision=(f"دور «{role.name}»: {tool} على الهدف "
                      f"({(role.goal or task.goal)[:80]}) — "
                      f"{detail[:120]}"),
            outcome=outcome,
            category=("search_method" if "search" in tool
                      else "verification" if "verify" in tool
                      else "general"),
            confidence=0.6 if outcome == "success" else 0.5,
            task_id=task.task_id,
            agents=role.name)
    except Exception:
        pass

_MIN_CONFIDENCE = 0.3
_STEP_MAX = 25


def _extract_role_targets(goal: str) -> List[str]:
    """استخراج أهداف بحثية للهدف الفرعي للدور (3 كحد أقصى)."""
    targets = [goal]
    # محاولة إضافة وجهات نظر مشتقة من أفعال البحث العربية
    for m in re.finditer(r"عن\s+(.{8,}?(?:\s|$))", goal):
        extra = m.group(1).strip()
        if len(extra) >= 6 and extra not in targets and len(targets) < 3:
            targets.append(f"تطورات {extra}")
    return targets[:3]


# ══════════════════════════════════════════════════════════════════
# اجتماع التجميع والتوليف الحتمي
# ══════════════════════════════════════════════════════════════════

def _synthesize(task: CollaborativeTask) -> str:
    """توليف حتمي: دمج نتائج الأدوار في تقرير موحد بمصادر كل دور."""
    lines = [
        f"# تقرير تعاوني: {task.title}",
        "",
        "هذا التقرير أعدّه فريق وكلاء متخصصين (باحثون مستقلون + مدقق "
        "النتائج) في مهمة طويلة الأمد تعاونية عبر ناقل أحداث مشترك.",
        "",
    ]
    done_roles = [r for r in task.roles if r.status == ROLE_DONE]
    failed_roles = [r for r in task.roles if r.status == ROLE_FAILED]
    # ── مخرجات الناقل المشترك: ما تبادله الفريق فعليًا ──
    shared_findings: List[Dict[str, Any]] = []
    with contextlib.suppress(Exception):
        if _SKB_OK():
            shared_findings = _get_skb().query_knowledge(
                task.goal[:150], task_id=task.task_id, k=8)
    # ── الخبرات الجماعية التي استُحضرت قبل التخطيط ──
    with contextlib.suppress(Exception):
        if _TEM_OK():
            recalled: List[Dict[str, Any]] = getattr(
                task, "_tem_recall", [])
            if recalled:
                lines.append("## خبرات جماعية مستحضرَة قبل التخطيط")
                lines.append("")
                lines.append("خبرات متراكمة من مهام سابقة ذات صلة "
                             "استُحضرت في مرحلة التخطيط:")
                lines.append("")
                for exp in recalled[:5]:
                    mark = ("✓" if exp.get("outcome") == "success"
                            else "✗" if exp.get("outcome") == "failure"
                            else "~")
                    lines.append(
                        f"- {mark} [{exp.get('category', '')}] "
                        f"{exp.get('decision', '')[:140]} "
                        f"(نتيجة: {exp.get('outcome', '')}، "
                        f"تكرار {exp.get('hits', 0)})")
                lines.append("")
    if shared_findings:
        lines.append("## المعارف المشتركة في ناقل الفريق")
        lines.append("")
        lines.append("نتائج تبادلها الأدوار عبر الناقل المشترك لحظيًا "
                     "(بحث دلالي عربي):")
        lines.append("")
        for f in shared_findings:
            lines.append(
                f"- **{f.get('role', '?')}** ({f.get('tool', '')}): "
                f"{(f.get('text') or '').strip()[:400]}")
        lines.append("")
    for role in done_roles:
        lines.append(f"## {role.name} — {role.goal}")
        lines.append("")
        seen: List[str] = []
        for step in role.steps:
            txt = (step.get("result") or "").strip()
            if txt and txt not in seen:
                seen.append(txt)
                lines.append(txt[:1200])
                lines.append("")
    if not done_roles:
        lines.append("**ملاحظة:** لم تكتمل أي أدوار بحثية — تحقق من "
                     "سجل الخطوات أدناه.")
    lines += ["", "---", "",
              "## سجل أدوار الفريق", "",
              f"| الدور | الحالة | الخطوات | طلبات الإنترنت |",
              "|---|---|---|---|"]
    for role in task.roles:
        label = {"done": "✅ اكتمل", "failed": "❌ فشل",
                 "running": "⚙️ يعمل", "pending": "⏳ معلق"}.get(
            role.status, role.status)
        lines.append(
            f"| {role.name} | {label} | {len(role.steps)} | "
            f"{role.fetch_state.get('count', 0)} |")
    if failed_roles:
        lines.append("")
        lines.append("### أدوار فشلت")
        for role in failed_roles:
            lines.append(f"- **{role.name}**: {role.error[:120]}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# محرك المهمة التعاونية
# ══════════════════════════════════════════════════════════════════

def _run_collaborative_task(manager: "CollaborativeManager",
                            task: CollaborativeTask) -> None:
    """تشغيل المهمة التعاونية: أدوار متوازية ثم اجتماع تجميع."""
    task._t0 = time.time()
    task.status = COLLAB_RUNNING
    task.started_at = _now_iso()
    manager._persist_collab(task)
    manager._emit("collab_started", task,
                  detail=f"مهمة تعاونية: {task.title} "
                         f"({len(task.roles)} أدوار)")
    # ── استحضار الخبرات الجماعية المتراكمة قبل التخطيط ──
    with contextlib.suppress(Exception):
        if _TEM_OK():
            _advise_roles_from_experience(task)
    try:
        task.assign_roles()
        # مساحات معزولة لكل دور
        try:
            ws_root = os.path.join(BASE_DIR, "data", "lht_workspace",
                                   "collab_ws")
            os.makedirs(ws_root, exist_ok=True)
            for role in task.roles:
                d = os.path.join(ws_root,
                                 f"{task.task_id}_{role.name[:20]}")
                os.makedirs(d, exist_ok=True)
                task._role_dirs[role.name] = d
        except Exception as exc:
            logger.debug("collab: تعذّر إنشاء مساحات الأدوار: %s", exc)

        # تشغيل الأدوار البحثية متوازية
        threads = [
            threading.Thread(
                target=_run_role, args=(manager, task, role),
                name=f"NSM-collab-{task.task_id[:6]}-{role.name[:10]}",
                daemon=True,
            )
            for role in task.roles[:-1]  # كل الأدوار عدا المدقق
        ]
        for th in threads:
            th.start()

        # انتظار الأدوار (بحد زمني)
        deadline = time.time() + MAX_COLLAB_AGE_S
        for role in task.roles[:-1]:
            wait = max(0.1, deadline - time.time())
            for _ in range(int(wait * 10)):
                if role.status in (ROLE_DONE, ROLE_FAILED):
                    break
                if task._cancelled.is_set() or \
                        time.time() - task._t0 > MAX_COLLAB_AGE_S:
                    break
                time.sleep(0.1)

        if task._cancelled.is_set():
            task.status = COLLAB_CANCELLED
            return
        if time.time() - task._t0 > MAX_COLLAB_AGE_S:
            task.status = COLLAB_FAILED
            task.error = "تجاوزت المهمة سقف المدة المسموح"
            return

        # المدقق: مراجعة نتائج الأدوار
        verifier = task.roles[-1]
        verifier.status = ROLE_RUNNING
        verifier.started_at = _now_iso()
        verdict_lines = ["## مدقق النتائج — خلاصة المراجعة", ""]
        done_count = sum(1 for r in task.roles[:-1] if r.status == ROLE_DONE)
        total = len(task.roles) - 1
        verdict_lines.append(
            f"اكتمل {done_count} من {total} أدوار بحثية.")
        for role in task.roles[:-1]:
            if role.status == ROLE_DONE:
                found = sum(1 for s in role.steps
                            if s.get("status") == "done")
                verdict_lines.append(
                    f"- {role.name}: موثقة ({found} خطوة ناجحة) — مقبولة")
            else:
                verdict_lines.append(
                    f"- {role.name}: {role.error[:80] or 'لم تكتمل'}")
        verifier_result = "\n".join(verdict_lines)
        verifier.log(0, "done", "verify", "", verifier_result)
        verifier.status = ROLE_DONE
        verifier.finished_at = _now_iso()

        # اجتماع التجميع والتوليف
        task.status = COLLAB_GATHERING
        manager._emit("collab_gathering", task,
                      detail="اجتماع التجميع: دمج نتائج الأدوار")
        task._synthesis = _synthesize(task)

        # حفظ التقرير في مساحة الملف العامة (نفس أدوات LHT)
        tools = _get_role_tools()
        with contextlib.suppress(Exception):
            if "write_file" in tools:
                safe = re.sub(r"[^\w\u0600-\u06FF\-]+", "_",
                              task.title[:30])
                fname = f"collab_{safe}_{task.task_id[:6]}.md"
                tools["write_file"](fname, task._synthesis)
        manager._emit("collab_done", task,
                      detail=(f"اكتملت المهمة التعاونية: {task.title} "
                              f"({done_count}/{total} أدوار ناجحة)"))
        task.status = COLLAB_DONE
    except Exception as exc:
        logger.error("collab: فشل %s: %s", task.task_id, exc)
        task.status = COLLAB_FAILED
        task.error = str(exc)[:300]
        manager._emit("collab_failed", task,
                      detail=f"فشل المهمة التعاونية: {task.error}")
    finally:
        task.progress = 1.0
        task.finished_at = _now_iso()
        manager._persist_collab(task)
        manager._prune()


# ══════════════════════════════════════════════════════════════════
# المدير (singleton)
# ══════════════════════════════════════════════════════════════════

_MANAGER: Optional["CollaborativeManager"] = None
_MANAGER_LOCK = threading.Lock()
_PERSIST_MANAGER: Optional["CollaborativeManager"] = None


def _get_manager_for_persist() -> Optional["CollaborativeManager"]:
    return _PERSIST_MANAGER or _MANAGER


class CollaborativeManager:
    """منظّم المهام التعاونية — واحد لكل عملية."""

    def __init__(self, db_path: str = DB_DEFAULT) -> None:
        self.db_path = db_path
        _init_db(db_path)
        self._lock = threading.Lock()
        self._tasks: Dict[str, CollaborativeTask] = {}
        self._running_keys: Dict[str, str] = {}
        # ربط المدير لحفظ الخطوات من الأدوار
        global _PERSIST_MANAGER
        _PERSIST_MANAGER = self

    # ── أحداث عبر ناقل الأحداث ─────────────────────────────────
    def _emit(self, event_type: str, task: CollaborativeTask,
              detail: str = "") -> None:
        try:
            if not event_type.startswith("collab_"):
                return
            from ai.agent_event_bus import emit_event  # type: ignore
            emit_event(
                event_type, agent_id="collaborative", title=task.title,
                status=task.status, detail=detail,
                metadata={"task_id": task.task_id,
                          "progress": task.progress,
                          "roles": len(task.roles)},
            )
        except Exception:
            logger.debug("collab: تعذّر إطلاق حدث %s", event_type)

    # ── الحفظ في SQLite ─────────────────────────────────────────
    def _persist_collab(self, task: CollaborativeTask) -> None:
        try:
            with _connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO collab_tasks "
                    "(task_id, title, goal, status, plan_json, "
                    "created_at, started_at, finished_at, error) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (task.task_id, task.title, task.goal, task.status,
                     json.dumps([r.to_dict() for r in task.roles],
                                ensure_ascii=False),
                     task.created_at, task.started_at,
                     task.finished_at, task.error[:300]),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("collab: تعذّر حفظ المهمة %s: %s",
                         task.task_id, exc)

    def _prune(self) -> None:
        """إبقاء آخر 50 مهمة في الذاكرة."""
        with self._lock:
            if len(self._tasks) <= 50:
                return
            done = [
                t for t in self._tasks.values()
                if t.status in (COLLAB_DONE, COLLAB_FAILED, COLLAB_CANCELLED)
            ]
            done.sort(key=lambda t: t.finished_at or "0")
            for old in done[: max(1, len(done) - 20)]:
                del self._tasks[old.task_id]

    # ── API العام ─────────────────────────────────────────────────
    def submit(self, goal: str,
               subgoals: Optional[List[str]] = None) -> Optional[CollaborativeTask]:
        """تقديم مهمة تعاونية — أدوارها تتوازى في خيوط daemon."""
        goal = (goal or "").strip()
        if not goal:
            return None
        with self._lock:
            running = sum(
                1 for t in self._tasks.values()
                if t.status in (COLLAB_PENDING, COLLAB_RUNNING,
                                COLLAB_GATHERING))
            if running >= MAX_COLLAB_CONCURRENT:
                return None
            task = CollaborativeTask(goal=goal, subgoals=subgoals)
            self._tasks[task.task_id] = task
        self._persist_collab(task)
        self._emit("collab_submitted", task,
                   detail=f"مهمة تعاونية جديدة: {task.title}")
        threading.Thread(
            target=_run_collaborative_task, args=(self, task),
            name=f"NSM-collab-{task.task_id[:6]}", daemon=True,
        ).start()
        return task

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status in (COLLAB_DONE, COLLAB_FAILED, COLLAB_CANCELLED):
                return False
            task.status = COLLAB_CANCELLED
            task.finished_at = _now_iso()
            task._cancelled.set()
        self._persist_collab(task)
        self._emit("collab_cancelled", task,
                   detail="ألغى المستخدم المهمة التعاونية")
        return True

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self._tasks.get(task_id)
        return task.to_dict() if task is not None else None

    def list_tasks(self, limit: int = 30) -> List[Dict[str, Any]]:
        tasks = sorted(self._tasks.values(),
                       key=lambda t: t.created_at, reverse=True)
        return [t.to_dict() for t in tasks[:limit]]

    def status(self) -> Dict[str, Any]:
        counter: Dict[str, int] = {}
        for t in self._tasks.values():
            counter[t.status] = counter.get(t.status, 0) + 1
        return {
            "total": len(self._tasks),
            "pending": counter.get(COLLAB_PENDING, 0),
            "running": counter.get(COLLAB_RUNNING, 0)
                + counter.get(COLLAB_GATHERING, 0),
            "done": counter.get(COLLAB_DONE, 0),
            "failed": counter.get(COLLAB_FAILED, 0),
            "cancelled": counter.get(COLLAB_CANCELLED, 0),
            "max_concurrent": MAX_COLLAB_CONCURRENT,
            "max_roles": MAX_COLLAB_ROLES,
            "max_role_steps": MAX_ROLE_STEPS,
            "max_role_fetches": MAX_ROLE_FETCHES,
        }


def get_collaborative_manager(
        db_path: Optional[str] = None) -> CollaborativeManager:
    """singleton على مستوى العملية."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = CollaborativeManager(db_path=db_path or DB_DEFAULT)
    return _MANAGER


__all__ = [
    "CollaborativeManager",
    "CollaborativeTask",
    "CollabRole",
    "get_collaborative_manager",
    "detect_collaborative_request",
    "MAX_COLLAB_ROLES",
    "MAX_ROLE_STEPS",
    "MAX_ROLE_FETCHES",
    "MAX_COLLAB_CONCURRENT",
    "_set_collab_role_hook",
]
