"""Agent Terminals — طرفية خاصة بكل وكيل AI في NSM
====================================================================
يربط كل وكيل من وكلاء المشروع التسعة (المعرَّفة في ai/agent_categories.py)
بطرفية دائمة خاصة به:

  • كل وكيل له جلسة طرفية مستقلة (TerminalSession) لا يشاركها مع غيره
  • cwd مقيد لكل وكيل حسب تخصصه (scope) — لا يمكنه الخروج منه
  • دور صلاحيات محدد لكل وكيل + قيود إضافية (regex مسموح / كلمات ممنوعة)
  • سجل تدقيق مستقل لكل وكيل (audit JSONL بفلتر agent)
  • سجل أوامر (history) مستقل لكل وكيل
  • تسجيل تلقائي في role_manager عند أول وصول (lazy registration)

الاستخدام من الواجهة:

    from ai.agent_terminals import get_agent_terminals
    at = get_agent_terminals()
    info = at.agent_info("coding")      # اسم/دور/مجلد/حالة
    r = at.run("coding", "git status")  # تنفيذ أمر باسم الوكيل
    hist = at.agent_history("coding")   # سجل أوامره
    aud = at.agent_audit("coding")      # تدقيقه الخاص
    all_ = at.list_agents()             # كل الوكلاء مع خلاصاتهم

هذا الملف إضافي بالكامل — لا يُعدّل أي سلوك في ai/nsm_terminal.py
أو ai/terminal_roles.py؛ يستخدم run_agent/audit_events الموجودة أصلاً.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from ai.agent_categories import AGENT_CATEGORIES, CATEGORY_ORDER
    _CATEGORIES_OK = True
except Exception:
    AGENT_CATEGORIES = {}
    CATEGORY_ORDER = []
    _CATEGORIES_OK = False

from ai.nsm_terminal import get_terminal, CommandResult

# ─────────────── إعدادات الوكلاء ───────────────

# قيود افتراضية لكل وكيل: (الدور، المجلد المقيد، kaggle مسموح/لا،
#   regex إضافي مسموح، كلمات ممنوعة)
# المجلدات مقيدة داخل جذر المشروع — لا وكيل يخرج من نطاقه.
AGENT_SETUP: Dict[str, Dict[str, Any]] = {
    # وكلاء عامة — نطاق جذر المشروع
    "assistant": {
        "role": "agent", "scope": None, "can_kaggle": False,
        "extra_allowed": r"^(git|ls|cat|head|tail|grep|find)\s+",
        "extra_denied": ["rm -rf", "shutdown", "reboot", "mkfs"],
    },
    "automation": {
        "role": "agent", "scope": None, "can_kaggle": False,
        "extra_allowed": r"^(git|ls|cat|head|tail|grep|find|python3|bash)\s+",
        "extra_denied": ["rm -rf", "shutdown", "reboot", "sudo"],
    },
    # وكلاء تحليل وبحث — قراءة فقط افتراضيًا
    "analytics": {
        "role": "sandbox", "scope": None, "can_kaggle": False,
        "extra_denied": ["rm", "mv", "cp ", "chmod", "pip install", "sudo"],
        "safe_list": ["ls", "cat", "head", "tail", "grep", "find", "wc",
                      "sort", "uniq", "cut", "python3", "python3 -m"],
    },
    "research": {
        "role": "sandbox", "scope": None, "can_kaggle": False,
        "extra_denied": ["rm", "chmod", "pip install", "sudo"],
        "safe_list": ["ls", "cat", "head", "tail", "grep", "find", "wc",
                      "sort", "curl", "wget", "python3", "python3 -m"],
    },
    # وكيل المنطق — قراءة عامة مع تنفيذ سكربتات آمن
    "reasoning": {
        "role": "agent", "scope": None, "can_kaggle": False,
        "extra_allowed": r"^(ls|cat|head|tail|grep|find|python3)\s+",
        "extra_denied": ["rm -rf", "shutdown", "sudo"],
    },
    # وكيل البرمجة — git كامل + بناء
    "coding": {
        "role": "agent", "scope": None, "can_kaggle": False,
        "extra_allowed": r"^(git|python3|npm|pnpm|pip3|make|pytest|ls|cat)\s+",
        "extra_denied": ["rm -rf /", "git push", "git merge origin"],
    },
    # وكيل الصيانة — نطاق config/memory فقط
    "maintenance": {
        "role": "agent", "scope": "config", "can_kaggle": False,
        "extra_allowed": r"^(ls|cat|grep|find|tail)\s+",
        "extra_denied": ["rm", "mv", "chmod", "sudo"],
    },
    # وكيل المحتوى — قراءة assets/notes
    "content": {
        "role": "sandbox", "scope": None, "can_kaggle": False,
        "extra_allowed": r"^(ls|cat|head|tail|grep|find)\s+",
        "extra_denied": ["rm", "mv", "chmod", "pip install", "sudo"],
    },
    # وكيل تدريب النماذج — Kaggle مقيد بـstatus فقط
    "model_trainer": {
        "role": "agent", "scope": None, "can_kaggle": True,
        "extra_allowed": r"^(git|ls|cat|head|tail|grep|find|nvidia-smi)\s+",
        "extra_denied": ["rm -rf", "shutdown", "sudo"],
    },
}

# أسماء عرض عربية للوكلاء مع أدوارهم المسموحة لأوامر Kaggle
AGENT_META: Dict[str, Dict[str, str]] = {
    "assistant":      {"title_ar": "المساعد", "role_ar": "عام"},
    "automation":     {"title_ar": "الأتمتة", "role_ar": "عام + سكربتات"},
    "analytics":      {"title_ar": "التحليلات", "role_ar": "قراءة فقط"},
    "research":       {"title_ar": "البحث", "role_ar": "قراءة + بحث ويب"},
    "reasoning":      {"title_ar": "المنطق", "role_ar": "قراءة + تفسير"},
    "coding":         {"title_ar": "البرمجة", "role_ar": "git + بناء"},
    "maintenance":    {"title_ar": "الصيانة", "role_ar": "config فقط"},
    "content":        {"title_ar": "المحتوى", "role_ar": "قراءة فقط"},
    "model_trainer":  {"title_ar": "مدرب النماذج", "role_ar": "Kaggle status"},
}


@dataclass
class AgentTerminalInfo:
    """خلاصة طرفية وكيل واحدة."""
    key: str
    title: str
    title_ar: str
    role: str
    role_ar: str
    scope: str
    cwd: str
    mode: str
    kaggle_cli: bool
    cmd_count: int
    session_id: str
    last_cmd: str = ""


class AgentTerminalsManager:
    """مدير طرفيات الوكلاء — طرفية دائمة لكل وكيل.

    تصميم lazy: لا تُنشأ جلسات الوكلاء عند الاستيراد بل عند أول وصول
    لطرفية وكيل — حتى لا تستهلك موارد في الخلفية إذا لم يستخدمها أحد.
    جميع العمليات thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._registered: Dict[str, bool] = {}

    # ─── helpers ───
    def _term(self):
        return get_terminal()

    def _ensure(self, key: str) -> None:
        """تسجيل الوكيل في role_manager عند أول وصول (مرة واحدة)."""
        with self._lock:
            if self._registered.get(key):
                return
            self._registered[key] = True
        setup = AGENT_SETUP.get(key) or {
            "role": "agent", "scope": None, "can_kaggle": False}
        rm = self._term().role_manager
        rm.register_agent(
            key,
            role=str(setup.get("role", "agent")),
            scope=setup.get("scope"),
            can_kaggle=bool(setup.get("can_kaggle", False)),
            extra_allowed=setup.get("extra_allowed"),
            extra_denied=list(setup.get("extra_denied", [])),
            safe_list=setup.get("safe_list"),
        )

    def _agent_session(self, key: str):
        """يُعيد (أو ينشئ) جلسة الطرفية الدائمة للوكيل."""
        self._ensure(key)
        term = self._term()
        with term._lock:
            sess = next((s for s in term._sessions.values() if s.agent == key), None)
            if sess is None:
                sess = term.create_session(mode="safe", agent=key)
            sess.last_active_ts = __import__("time").time()
            return sess

    # ─── واجهة عامة ───
    def list_agents(self) -> List[AgentTerminalInfo]:
        """خلاصة طرفيات كل الوكلاء (المرتبة بـCATEGORY_ORDER)."""
        out: List[AgentTerminalInfo] = []
        term = self._term()
        for key in CATEGORY_ORDER:
            self._ensure(key)
            sess = self._agent_session(key)
            meta = AGENT_META.get(key, {"title_ar": key, "role_ar": ""})
            setup = AGENT_SETUP.get(key, {})
            cat = AGENT_CATEGORIES.get(key) if _CATEGORIES_OK else None
            hist = self.agent_history(key, limit=1)
            out.append(AgentTerminalInfo(
                key=key,
                title=cat.title if cat else key,
                title_ar=meta["title_ar"],
                role=str(setup.get("role", "agent")),
                role_ar=meta.get("role_ar", ""),
                scope=str(setup.get("scope")) or "جذر المشروع",
                cwd=sess.cwd if sess else "",
                mode=sess.mode if sess else "safe",
                kaggle_cli=bool(setup.get("can_kaggle", False)),
                cmd_count=len(hist),
                session_id=sess.id if sess else "",
                last_cmd=(hist[0]["cmd"] if hist and hist[0].get("cmd") else ""),
            ))
        return out

    def agent_info(self, key: str) -> Optional[AgentTerminalInfo]:
        if key not in AGENT_SETUP and key not in CATEGORY_ORDER:
            return None
        for a in self.list_agents():
            if a.key == key:
                return a
        return None

    def run(self, key: str, cmd: str, timeout: int = 30) -> Tuple[CommandResult, str]:
        """تنفيذ أمر باسم وكيل — يرجع (النتيجة, اسم الوكيل)."""
        self._ensure(key)
        term = self._term()
        # جلسة الوكيل الدائمة (run_agent يفتحها تلقائياً أيضاً)
        sess = self._agent_session(key)
        r = term.run_agent(key, cmd.strip(), session_id=sess.id, timeout=timeout)
        return r, key

    def agent_history(self, key: str, limit: int = 50) -> List[Dict[str, Any]]:
        """سجل أوامر الوكيل (من history جلسته)."""
        term = self._term()
        sess = next((s for s in term._sessions.values() if s.agent == key), None)
        if sess is None:
            return []
        with term._lock:
            rows = list(sess.history)
        return [dict(h) for h in rows[-limit:]][::-1]

    def agent_audit(self, key: str, limit: int = 50) -> List[Dict[str, Any]]:
        """سجل التدقيق الخاص بالوكيل فقط."""
        from ai.terminal_roles import audit_events
        return audit_events(limit=limit, agent=key)

    def agent_permissions(self, key: str) -> Dict[str, Any]:
        """نسخة صلاحيات الوكيل الحالية (لاستعراضها في الواجهة)."""
        self._ensure(key)
        rm = self._term().role_manager
        return rm.role_of(key)


# Singleton
_AGENT_TERMINALS: Optional[AgentTerminalsManager] = None
_AGENT_TERMINALS_LOCK = threading.Lock()


def get_agent_terminals() -> AgentTerminalsManager:
    """يرجع المدير الوحيد لطرفيات الوكلاء."""
    global _AGENT_TERMINALS
    with _AGENT_TERMINALS_LOCK:
        if _AGENT_TERMINALS is None:
            _AGENT_TERMINALS = AgentTerminalsManager()
        return _AGENT_TERMINALS
