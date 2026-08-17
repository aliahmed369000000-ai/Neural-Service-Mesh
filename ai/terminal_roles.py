"""
NSM Terminal — الصلاحيات الدقيقة وسجل التدقيق (Audit)
====================================================
طبقة صلاحيات أعلى من أوضاع safe/admin الحالية، تتيح:
  • أدوار (Roles): owner / admin / agent / sandbox
  • قيود لكل وكيل (cwd مقيد + قائمة أوامر مسموحة/ممنوعة لكل وكيل)
  • سجل تدقيق منفصل (audit log JSONL) يوثّق كل أمر مع: الوكيل، الدور،
    القرار (سمح/رفض)، السبب، الوقت — دون حذف أو تدوير تلقائي سريع
    (التدوير هنا بطيء لأن التدقيق مهم للنظام).

يُدمج في NSMTerminal عبر role=... في إنشاء الجلسات، ويُستدعى من
الوكلاء (nsm_agent_core، autonomous_will) مع معرف الوكيل.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent


def get_root() -> Path:
    """الجذر الفعلي للمشروع — يُقرأ ديناميكيًا من ai/nsm_terminal إن كان
    مستبدلًا (اختبارات/تكوينات خاصة)، وإلا يعود لجذر هذه الوحدة."""
    try:
        import ai.nsm_terminal as _nt
        if _nt.ROOT is not None and _nt.ROOT != ROOT:
            return Path(_nt.ROOT)
    except Exception:
        pass
    return ROOT

_ROLE_NAMES = ("owner", "admin", "agent", "sandbox")

# ───────────────────────────── سجل التدقيق ─────────────────────────────
_AUDIT_LOG = ROOT / "memory" / "terminal_audit.jsonl"
_MAX_AUDIT_BYTES = 8_000_000  # 8MB — تدوير بطيء جدًا (احتفظ بالنصف الأحدث)
_audit_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def audit_log_path() -> Path:
    return _AUDIT_LOG


def _append_audit(entry: dict) -> None:
    """يكتب دخولًا في سجل التدقيق مع تدوير بطيء جدًا يمنع نموه بلا حد."""
    with _audit_lock:
        try:
            _AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
            if _AUDIT_LOG.exists() and _AUDIT_LOG.stat().st_size > _MAX_AUDIT_BYTES:
                try:
                    lines = _AUDIT_LOG.read_text(encoding="utf-8").splitlines()
                    keep = lines[len(lines) // 2:]
                    _AUDIT_LOG.write_text("\n".join(keep) + "\n", encoding="utf-8")
                except Exception:
                    pass
            with _AUDIT_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass


def audit_events(limit: int = 50, agent: Optional[str] = None) -> List[dict]:
    """يقرأ آخر N أحداث تدقيق، مع فلترة اختيارية حسب معرف الوكيل."""
    if not _AUDIT_LOG.exists():
        return []
    try:
        lines = [ln for ln in _AUDIT_LOG.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return []
    if agent:
        a_low = agent.lower()
        lines = [ln for ln in lines if a_low in ln.lower()]
    out: List[dict] = []
    for ln in lines[-limit * 3:]:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
        if agent and (agent or "").lower() not in json.dumps(out[-1], ensure_ascii=False).lower():
            out.pop()
            continue
    return out[-limit:]


# ───────────────────────────── تعريف الأدوار ─────────────────────────────
@dataclass
class RolePermissions:
    role: str
    default_safe_prefixes_only: bool = True   # True => قائمة safe prefixes فقط
    allow_shell_operators: bool = False       # && ; | || و > < و $()
    allow_cd: bool = True
    allow_export: bool = True
    max_timeout_seconds: int = 45
    max_cwd_depth: int = 0                    # 0 => أي عمق داخل الجذر
    can_use_kaggle_cli: bool = False          # هل يُسمح بأوامر kaggle CLI
    can_run_background: bool = False
    can_override_mode: bool = False           # هل يستطيع ترقية نفسه إلى admin
    can_manage_session: bool = True           # cd/export/unset/snapshot/alias
    allow_regex: Optional[re.Pattern] = None  # أوامر إضافية مسموحة (نمط regex)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        if self.allow_regex:
            d["allow_regex"] = self.allow_regex.pattern
        return d


DEFAULT_ROLES: Dict[str, RolePermissions] = {
    "owner": RolePermissions(
        role="owner", default_safe_prefixes_only=False, allow_shell_operators=True,
        max_timeout_seconds=3600, can_use_kaggle_cli=True, can_run_background=True,
        can_override_mode=True,
    ),
    "admin": RolePermissions(
        role="admin", default_safe_prefixes_only=False, allow_shell_operators=True,
        max_timeout_seconds=3600, can_use_kaggle_cli=True, can_run_background=True,
    ),
    "agent": RolePermissions(
        role="agent", default_safe_prefixes_only=True, allow_shell_operators=False,
        max_timeout_seconds=1800, can_use_kaggle_cli=True, can_run_background=True,
    ),
    "sandbox": RolePermissions(
        role="sandbox", default_safe_prefixes_only=True, allow_shell_operators=False,
        max_timeout_seconds=30, allow_cd=False, allow_export=False,
        can_use_kaggle_cli=False, can_run_background=False, can_manage_session=False,
    ),
}

# أوامر kaggle CLI المسموحة ضمن كل دور لديه can_use_kaggle_cli
KAGGLE_ALLOWED_SUBS = (
    "kernels status", "kernels logs", "kernels list", "kernels files",
    "kernels output", "kernels pull", "kernels topics",
    "datasets list", "datasets files", "datasets status",
    "models list", "models get",
    "config view", "quota", "whoami", "--version",
)
KAGGLE_DENIED_PREFIXES = ("rm", "delete", "--yes", "-y")


# ───────────────────────────── مدير الأدوار والوكلاء ─────────────────────────────
class TerminalRoleManager:
    """مدير أدوار الوكلاء والصلاحيات الدقيقة + سجل التدقيق."""

    def __init__(self):
        self._roles: Dict[str, RolePermissions] = dict(DEFAULT_ROLES)
        # قيود خاصة بوكيل معين (تتجاوز إعدادات الدور الافتراضي)
        self._agent_overrides: Dict[str, Dict[str, Any]] = {}
        # المجلدات المقيدة لكل وكيل (None => جذر المشروع)
        self._agent_scopes: Dict[str, Optional[Path]] = {}
        self._lock = threading.Lock()

    # ───────── التسجيل والقيود ─────────
    def register_agent(
        self,
        agent_id: str,
        role: str = "agent",
        scope: Optional[str] = None,
        extra_allowed: Optional[str] = None,
        extra_denied: Optional[List[str]] = None,
        can_kaggle: Optional[bool] = None,
        safe_list: Optional[List[str]] = None,
    ) -> None:
        """يسجل وكيلًا بدور وقيود خاصة:
        scope  => مجلد مقيد يتحرك داخله فقط (نسبي لجذر المشروع)
        extra_allowed => نمط regex لأوامر إضافية مسموحة
        extra_denied => كلمات محظورة لهذا الوكيل تحديدًا
        safe_list => قائمة أوامر مسموحة حصرية للوكيل (تتخطى قائمة safe العامة).
        إن حددت فهي القائمة الكاملة المسموحة (بدون union مع القائمة العامة).
        """
        if role not in _ROLE_NAMES:
            role = "agent"
        overrides: Dict[str, Any] = {"role": role}
        if extra_allowed:
            try:
                overrides["allow_regex"] = re.compile(extra_allowed)
            except re.error:
                overrides["allow_regex"] = None
        if extra_denied:
            overrides["extra_denied"] = [str(d) for d in extra_denied]
        if can_kaggle is not None:
            overrides["can_use_kaggle_cli"] = bool(can_kaggle)
        if safe_list is not None:
            try:
                overrides["safe_list"] = [str(s).lower() for s in safe_list]
            except Exception:
                overrides["safe_list"] = None
        with self._lock:
            self._agent_overrides[agent_id] = overrides
            if scope:
                try:
                    _root = get_root()
                    p = (_root / scope).resolve()
                    p.relative_to(_root)
                    self._agent_scopes[agent_id] = p
                except Exception:
                    self._agent_scopes[agent_id] = None
            else:
                self._agent_scopes.pop(agent_id, None)

    def unregister_agent(self, agent_id: str) -> None:
        with self._lock:
            self._agent_overrides.pop(agent_id, None)
            self._agent_scopes.pop(agent_id, None)

    # ───────── الفحص ─────────
    def role_of(self, agent_id: str) -> RolePermissions:
        with self._lock:
            ov = self._agent_overrides.get(agent_id, {})
            role = ov.get("role", "agent")
        return dict(self._roles.get(role, DEFAULT_ROLES["agent"]).__dict__)  # snapshot

    def scoped_cwd(self, agent_id: str, requested: Path) -> Tuple[Path, str]:
        """يقيّد cwd بالوكيل: إن كان له scope وcwd المطلوب خارجه يُعاد إلى	scope.
        تعيد (المسار النهائي الفعلي, سبب التعديل أو "")."""
        with self._lock:
            scope = self._agent_scopes.get(agent_id)
        if scope is None:
            return requested, ""
        try:
            requested.relative_to(scope)
            return requested, ""
        except Exception:
            return scope, f"cwd مقيد للوكيل داخل {scope}"

    def check(
        self, agent_id: str, cmd: str, is_kaggle_cli: bool = False
    ) -> Tuple[bool, str]:
        """يفحص الأمر حسب دور الوكيل ويرجع (مسموح, السبب).
        ملاحظة: الفحص هنا تكميلي لفحص _is_safe_command الأصلي — يُستدعى
        قبله للأوامر الخاصة (kaggle) أو فوقه لإضافة قيود الوكيل."""
        c = cmd.strip()
        low = c.lower()
        with self._lock:
            ov = dict(self._agent_overrides.get(agent_id, {}))
        role_name = str(ov.get("role", "agent"))
        perms = self._roles.get(role_name, DEFAULT_ROLES["agent"])

        # ── كلمات محظورة خاصة بالوكيل ──
        for d in ov.get("extra_denied", []):
            if d.lower() in low:
                return False, f"ممنوع للوكيل {agent_id}: {d}"

        # ── أوامر kaggle CLI ──
        if is_kaggle_cli:
            if not perms.can_use_kaggle_cli:
                return False, f"أوامر Kaggle CLI غير مسموحة لدور {role_name}"
            sub = re.sub(r"\s+", " ", low)
            # استخراج subcommand بعد kaggle
            m = re.match(r"^kaggle\s+(\S+\s+\S+|\S+)", sub)
            if m:
                subcmd = m.group(1).rstrip()
                if any(subcmd.startswith(a) for a in KAGGLE_ALLOWED_SUBS):
                    # فحص المحظور على نص الأمر بعد subcommand مباشرة
                    rest = re.sub(r"^kaggle\s+\S+", "", sub).strip()
                    for d in KAGGLE_DENIED_PREFIXES:
                        if f" {d}" in (" " + rest):
                            return False, f"أمر Kaggle خطير محظور: {d}"
                    return True, ""
                return False, f"أمر Kaggle غير مسموح: kaggle {subcmd}"
            return False, "صيغة kaggle غير معروفة"

        # ── ترقية الدور ──
        if not perms.can_override_mode and re.search(r"(mode=|mode:)\s*admin", low):
            return False, f"دور {role_name} لا يستطيع ترقية نفسه إلى admin"
        # ── أوامر إدارة فقط (cd/export/unset/snapshot/alias) ──
        if not perms.can_manage_session and re.match(
                r"^(cd|export|unset|snapshot|alias)(\s|$)", low):
            return False, f"الأوامر الإدارية ({low.split()[0]}) محظورة لدور {role_name}"

        # ── قائمة الوكيل الحصرية (safe_list) ──
        sl = ov.get("safe_list")
        if sl is not None:
            first = low.split()[0] if low.split() else ""
            _hit = any(
                low == s.strip() or low.startswith(s.strip()) or first == s.strip()
                for s in sl)
            if not _hit:
                return False, (f"الأمر محظور على الوكيل {agent_id} — "
                               f"مسموح فقط: {', '.join(sl[:10])}...")

        # ── نمط regex إضافي مسموح ──
        rx = ov.get("allow_regex")
        if rx and isinstance(rx, re.Pattern) and rx.search(c):
            return True, ""

        # ── القائمة الآمنة الافتراضية (default_safe_prefixes_only) ──
        # الأدوار المقيّدة (agent/sandbox) تُحصَر في قائمة الأوامر الآمنة
        # الافتراضية؛ الأدوار الحرة (owner/admin) تتخطاها:
        if perms.default_safe_prefixes_only:
            from ai.nsm_terminal import _SAFE_PREFIXES as _safe_prefixes
            _in = False
            for p in _safe_prefixes:
                p = p.strip()
                if low == p or low.startswith(p):
                    _in = True
                    break
            if not _in:
                return False, (f"الأمر خارج قائمة الوكيل الآمنة الافتراضية "
                               f"— دور {role_name} مقيد بالأوامر الأساسية")

        # ── مشغلات shell ──
        if perms.allow_shell_operators:
            return True, ""
        masked = c
        if "$(" in masked or "`" in masked or ">" in masked or "<" in masked:
            return False, "إعادة توجيه/استبدال أوامر غير مسموح لدورك — اطلب من المالك"
        if "&" in masked.replace("&&", ""):
            return False, "التشغيل بالخلفية & غير مسموح لدورك"
        if re.search(r"&&|\|\||\||;", masked):
            return False, "تسلسل الأوامر (&&/;/|) غير مسموح لدورك"

        return True, ""

    # ───────── التدقيق ─────────
    def record(
        self, agent_id: str, cmd: str, allowed: bool, reason: str,
        role: str = "", extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "ts": _now(),
            "agent": agent_id,
            "role": role,
            "cmd": cmd,
            "allowed": bool(allowed),
            "reason": reason,
        }
        if extra:
            entry.update(extra)
        _append_audit(entry)

    def roles_snapshot(self) -> Dict[str, dict]:
        return {k: v.to_dict() for k, v in self._roles.items()}


# ───────────────────────────── singleton ─────────────────────────────
_role_singleton: Optional[TerminalRoleManager] = None
_role_lock = threading.Lock()


def get_role_manager() -> TerminalRoleManager:
    global _role_singleton
    with _role_lock:
        if _role_singleton is None:
            _role_singleton = TerminalRoleManager()
        return _role_singleton


def register_default_agents() -> None:
    """تسجيل الوكلاء المعروفة في المشروع بقيود مناسبة افتراضيًا.
    يُستدعى مرة عند بدء التشغيل من الوحدة المستوردة (ui أو agent core)."""
    mgr = get_role_manager()
    defaults = {
        "nsm_agent": {"role": "agent", "can_kaggle": True,
                      "extra_allowed": r"^git\s+"},
        "autonomous_will": {"role": "agent", "can_kaggle": True},
        "auto_runtime": {"role": "agent", "can_kaggle": False},
        "sandbox": {"role": "sandbox"},
    }
    for agent_id, kwargs in defaults.items():
        mgr.register_agent(agent_id, **kwargs)
