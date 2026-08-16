"""
NSM Terminal Engine — أفضل طرفية للوكلاء والواجهة
=================================================
محرك أوامر مشترك:
  • جلسات متعددة مع cwd مستقل
  • سجل أوامر + مخرجات (JSONL)
  • أوضاع: safe (افتراضي للوكلاء) / admin (وضع المالك)
  • كشف أسرار في المخرجات
  • API للوكلاء: run(), run_safe(), session_status()

يُستخدم من:
  - ui_pages/nsm_terminal.py
  - nsm_agent_core (action: terminal)
  - agent_tools / nsm_chat
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
_LOG = ROOT / "memory" / "terminal_sessions.jsonl"
_DEFAULT_TIMEOUT = 45
_MAX_OUTPUT = 24_000
_MAX_HISTORY = 200

_SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+"),
]

_SAFE_PREFIXES = (
    "python ", "python3 ", "pytest", "pip show", "pip list", "pip freeze",
    "git status", "git log", "git diff", "git branch", "git remote", "git lfs",
    "git show", "git rev-parse", "git describe",
    "ls", "find ", "wc ", "head ", "tail ", "cat ", "sed -n", "grep ", "rg ",
    "echo ", "uname", "date", "pwd", "whoami", "df ", "du ", "file ",
    "which ", "type ", "env", "printenv", "id", "stat ",
    "python -m py_compile", "python3 -m py_compile",
    "python -m pytest", "python3 -m pytest",
    "python -m compileall", "python3 -m compileall",
)

_BLOCKED = (
    "rm -rf /", "mkfs", "dd if=", ":(){", "shutdown", "reboot",
    "sudo ", "su ", "chmod 777 /", "chown -R",
)


_MAX_LOG_BYTES = 2_000_000  # 2MB — تدوير تلقائي لمنع نمو غير محدود
_MAX_SESSIONS = 40  # حد أقصى للجلسات المتزامنة — إخلاء الأقدم نشاطاً عند التجاوز
_SESSION_TTL_SECONDS = 6 * 3600  # جلسات خاملة أكثر من 6 ساعات تُخلى تلقائياً

# 🆕 aliases قابلة للتخصيص + حفظ/استعادة جلسات

def _cfg_dir() -> Path:
    return ROOT / "config"

def _aliases_path() -> Path:
    return _cfg_dir() / "terminal_aliases.json"

def _sessions_snapshot_path() -> Path:
    return ROOT / "memory" / "terminal_sessions_snapshot.json"
_DEFAULT_ALIASES: Dict[str, str] = {
    "gs": "git status --short --branch",
    "gl": "git log -12 --oneline --decorate",
    "gd": "git diff --stat HEAD",
    "py": "python3",
    "pi": "python3 -m pip",
    "tc": "python3 -m py_compile",
    "pt": "python3 -m pytest -q --tb=line",
    "kstat": "kaggle kernels status",
    "klog": "kaggle kernels logs",
    "kls": "kaggle kernels list",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact(text: str) -> str:
    if not text:
        return text
    out = text
    for pat in _SECRET_PATTERNS:
        out = pat.sub("***", out)
    # env secrets
    for key in ("GITHUB_TOKEN", "GITHUB_PERSONAL_ACCESS_TOKEN", "OPENAI_API_KEY",
                "GOOGLE_API_KEY", "GROQ_API_KEY", "NSM_ADMIN_KEY"):
        val = os.environ.get(key, "")
        if val and len(val) > 6 and val in out:
            out = out.replace(val, "***")
    return out


def _append_log(entry: dict) -> None:
    try:
        _LOG.parent.mkdir(parents=True, exist_ok=True)
        import json
        # 🆕 تدوير السجل: إن تجاوز الحجم الأقصى، أبقِ فقط النصف الأحدث لتفادي
        # نمو غير محدود لملف terminal_sessions.jsonl مع كثرة الاستخدام.
        if _LOG.exists() and _LOG.stat().st_size > _MAX_LOG_BYTES:
            try:
                lines = _LOG.read_text(encoding="utf-8").splitlines()
                keep = lines[len(lines) // 2:]
                _LOG.write_text("\n".join(keep) + "\n", encoding="utf-8")
            except Exception:
                pass
        with _LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


@dataclass
class CommandResult:
    ok: bool
    cmd: str
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    mode: str
    ts: str = field(default_factory=_now)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def formatted(self) -> str:
        lines = [
            f"$ {self.cmd}",
            f"# cwd={self.cwd}  exit={self.exit_code}  {self.duration_ms}ms  mode={self.mode}",
        ]
        if self.stdout:
            lines.append(self.stdout.rstrip())
        if self.stderr:
            lines.append(self.stderr.rstrip())
        if self.error:
            lines.append(f"[error] {self.error}")
        return "\n".join(lines)


@dataclass
class TerminalSession:
    id: str
    cwd: str
    mode: str = "safe"  # safe | admin
    history: List[dict] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    last_active_ts: float = field(default_factory=time.time)
    env: Dict[str, str] = field(default_factory=dict)  # 🆕 متغيرات export مستمرة داخل الجلسة
    agent: str = ""  # 🆕 معرف الوكيل المالك للجلسة (فارغ => مالك بشري)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "cwd": self.cwd,
            "mode": self.mode,
            "history_len": len(self.history),
            "created_at": self.created_at,
            "env": dict(self.env),
            "agent": self.agent,
            "last": self.history[-1] if self.history else None,
        }


@dataclass
class BackgroundJob:
    id: str
    cmd: str
    session_id: str
    cwd: str
    mode: str
    status: str = "running"  # running | done | killed | error | timed_out
    started_at: str = field(default_factory=_now)
    finished_at: str = ""
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    live_lines: List[str] = field(default_factory=list)  # 🆕 مخرجات حية خطًا بخط (stdout+stderr ممزوجان بوسم)
    timeout: int = 300  # 🆕 مهلة خاصة بالمهمة (صفر = لا مهلة)
    live_lock: threading.Lock = field(default_factory=threading.Lock)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "cmd": self.cmd, "session_id": self.session_id, "cwd": self.cwd,
            "mode": self.mode, "status": self.status, "started_at": self.started_at,
            "finished_at": self.finished_at, "exit_code": self.exit_code,
            "stdout": self.stdout, "stderr": self.stderr, "error": self.error,
            "timeout": self.timeout, "lines": len(self.live_lines),
        }

    # 🆕 واجهات آمنة الخيط للبث الحي
    def append_line(self, line: str) -> None:
        with self.live_lock:
            self.live_lines.append(line)
            if len(self.live_lines) > _MAX_HISTORY:
                self.live_lines = self.live_lines[-_MAX_HISTORY:]

    def tail(self, n: int = 40) -> List[str]:
        with self.live_lock:
            return self.live_lines[-n:]

    def duration_s(self) -> float:
        t0 = datetime.fromisoformat(self.started_at).timestamp()
        return max(0.0, time.time() - t0)


def _mask_quotes(s: str) -> str:
    """يستبدل محتوى أي مقطع مقتبس (' أو ") برمز محايد 'Q' بنفس الطول، حتى لا
    تُكتشف عوامل السلسلة/التوجيه بالخطأ داخل نصوص حرفية مثل:
    python3 -c "import time; time.sleep(1)"  — الفاصلة المنقوطة هنا جزء من
    الكود الممرَّر لبايثون، وليست عامل تسلسل shell."""
    out = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch in ("'", '"'):
            q = ch
            out.append("Q")
            i += 1
            while i < n and s[i] != q:
                out.append("Q")
                i += 1
            if i < n:
                out.append("Q")
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _split_shell_segments(cmd: str) -> List[str]:
    """يقسّم الأمر عند &&, ||, ;, | خارج أي نص مقتبس فقط."""
    masked = _mask_quotes(cmd)
    pattern = re.compile(r"\|\||&&|;|\|")
    segments, last = [], 0
    for m in pattern.finditer(masked):
        segments.append(cmd[last:m.start()])
        last = m.end()
    segments.append(cmd[last:])
    return [seg.strip() for seg in segments if seg.strip()]


class NSMTerminal:
    """طرفية مشتركة — جلسة واحدة افتراضية + جلسات مسماة."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or ROOT).resolve()
        self._sessions: Dict[str, TerminalSession] = {}
        self._lock = threading.RLock()  # reentrant — تسمح بالتداخل (create_session يُستدعى داخل lock من طرفيات الوكلاء)
        self._jobs: Dict[str, BackgroundJob] = {}
        self._procs: Dict[str, subprocess.Popen] = {}
        self._jobs_lock = threading.Lock()
        self._default_id: Optional[str] = None  # يُضبط بعد إنشاء الجلسة الافتراضية أدناه
        self._default_id = self.create_session(mode="safe").id
        # 🆕 aliases قابلة للتخصيص (تُحمّل من config/terminal_aliases.json إن وُجدت)
        self.aliases: Dict[str, str] = dict(_DEFAULT_ALIASES)
        self._load_aliases()
        # 🆕 طبقة الصلاحيات الدقيقة وسجل التدقيق
        from ai.terminal_roles import get_role_manager, register_default_agents
        self.role_manager = get_role_manager()
        try:
            register_default_agents()
        except Exception:
            pass

    # ─────────────── 🆕 aliases + حفظ/استعادة ───────────────
    def _load_aliases(self) -> None:
        try:
            if _aliases_path().exists():
                custom = json.loads(_aliases_path().read_text(encoding="utf-8"))
                if isinstance(custom, dict):
                    self.aliases.update({k: v for k, v in custom.items() if isinstance(v, str)})
        except Exception:
            pass

    def save_aliases(self) -> bool:
        """يحفظ aliases الحالية إلى config/terminal_aliases.json."""
        try:
            _aliases_path().parent.mkdir(parents=True, exist_ok=True)
            _aliases_path().write_text(json.dumps(self.aliases, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
            return True
        except Exception:
            return False

    def set_alias(self, name: str, cmd: str) -> Tuple[bool, str]:
        name = name.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_-]*$", name):
            return False, f"اسم غير صالح: {name}"
        if not cmd.strip():
            return False, "الأمر فارغ"
        self.aliases[name] = cmd.strip()
        self.save_aliases()
        return True, f"alias {name}={cmd.strip()}"

    def del_alias(self, name: str) -> Tuple[bool, str]:
        if name in self.aliases:
            del self.aliases[name]
            self.save_aliases()
            return True, f"alias {name} محذوف"
        return False, f"لا يوجد alias: {name}"

    def _expand_alias(self, cmd: str) -> str:
        first, _, rest = cmd.partition(" ")
        if first in self.aliases:
            return self.aliases[first] + (" " + rest if rest else "")
        return cmd

    def save_sessions_snapshot(self) -> bool:
        """يحفظ حالة الجلسات الحالية (بدون السجل الكامل) لاستعادتها عند إعادة التشغيل."""
        try:
            data = [{"id": s.id, "cwd": s.cwd, "mode": s.mode, "agent": s.agent,
                     "env": dict(s.env), "created_at": s.created_at}
                    for s in self._sessions.values()]
            _sessions_snapshot_path().parent.mkdir(parents=True, exist_ok=True)
            _sessions_snapshot_path().write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
            return True
        except Exception:
            return False

    def restore_sessions_snapshot(self) -> int:
        """يستعيد الجلسات من آخر snapshot — يعيد العدد فقط (السجل يبقى في JSONL)."""
        restored = 0
        try:
            if not _sessions_snapshot_path().exists():
                return 0
            data = json.loads(_sessions_snapshot_path().read_text(encoding="utf-8"))
            for item in data:
                if not isinstance(item, dict):
                    continue
                sid = str(item.get("id", ""))
                if sid == self._default_id or sid in self._sessions:
                    continue
                s = TerminalSession(
                    id=sid, cwd=str(item.get("cwd", self.root)),
                    mode=item.get("mode", "safe"), agent=str(item.get("agent", "")),
                    env={k: v for k, v in item.get("env", {}).items() if isinstance(v, str)},
                    created_at=str(item.get("created_at", _now())),
                )
                with self._lock:
                    self._sessions[sid] = s
                restored += 1
        except Exception:
            pass
        return restored

    # ─────────────── 🆕 أوامر وكيل (agent run) ───────────────
    def run_agent(self, agent_id: str, cmd: str, session_id: Optional[str] = None,
                  timeout: int = _DEFAULT_TIMEOUT) -> CommandResult:
        """تنفيذ أمر باسم وكيل — يفحص صلاحيات الدور، يسجل في audit، ويلتزم بـscope.
        الأوامر تبدأ بـkaggle (مسافات) تعالج كمجموعة Kaggle CLI."""
        cmd = (cmd or "").strip()
        is_kaggle_cli = cmd.lower().startswith("kaggle ") or cmd == "kaggle"
        # افتح أو أعد استخدام جلسة الوكيل (لكل وكيل جلسة باسمه إن لم توجد)
        with self._lock:
            agent_sess = next((s for s in self._sessions.values()
                               if s.agent == agent_id), None)
        if agent_sess is None:
            agent_sess = self.create_session(mode="safe", agent=agent_id)
            # 🆕 قيّد cwd بالوكيل (scope) — قد لا يكون الدور مسجلاً وقت إنشاء الجلسة
            # لأن التسجيل يحدث متأخرًا في دورة الطرفيات، فنقيّد هنا أيضًا:
            start, why = self.role_manager.scoped_cwd(agent_id,
                                                      Path(agent_sess.cwd).resolve())
            if str(start) != agent_sess.cwd:
                agent_sess.cwd = str(start)
        if session_id is None:
            session_id = agent_sess.id
        allowed, reason = self.role_manager.check(agent_id, cmd, is_kaggle_cli=is_kaggle_cli)
        self.role_manager.record(agent_id, cmd, allowed, reason, extra={
            "is_kaggle_cli": is_kaggle_cli, "cwd": agent_sess.cwd,
        })
        if not allowed:
            r = CommandResult(ok=False, cmd=cmd, cwd=agent_sess.cwd,
                              exit_code=126, stdout="", stderr=reason, duration_ms=0,
                              mode="safe", error=reason)
            # 🆕 الرفض لا يُدوَّن في سجل الجلسة (audit يملكه منفصلًا)
            return r
        r = self.run(cmd, session_id=session_id, mode="safe", timeout=timeout)
        return r

    def create_session(self, mode: str = "safe", cwd: Optional[str] = None,
                       agent: str = "") -> TerminalSession:
        sid = uuid.uuid4().hex[:10]
        start = Path(cwd).resolve() if cwd else self.root
        try:
            start.relative_to(self.root)
        except Exception:
            start = self.root
        # 🆕 قيد cwd بالوكيل عبر TerminalRoleManager
        if agent:
            start, _ = self.role_manager.scoped_cwd(agent, start)
        sess = TerminalSession(id=sid, cwd=str(start), mode=mode if mode in ("safe", "admin") else "safe",
                               agent=agent)
        with self._lock:
            self._sessions[sid] = sess
            self._evict_stale_locked()
        return sess

    def get_session(self, session_id: Optional[str] = None) -> TerminalSession:
        with self._lock:
            sid = session_id or self._default_id
            if sid not in self._sessions:
                sess = TerminalSession(id=sid, cwd=str(self.root), mode="safe")
                self._sessions[sid] = sess
            self._sessions[sid].last_active_ts = time.time()
            return self._sessions[sid]

    def list_sessions(self) -> List[dict]:
        with self._lock:
            return [s.to_dict() for s in self._sessions.values()]

    def close_session(self, session_id: str) -> bool:
        """يُخلي جلسة يدوياً (لا يمكن إخلاء الجلسة الافتراضية)."""
        with self._lock:
            if session_id == self._default_id or session_id not in self._sessions:
                return False
            del self._sessions[session_id]
            return True

    def _evict_stale_locked(self) -> None:
        """🆕 يمنع تسرّب الذاكرة: يخلي الجلسات الخاملة أكثر من TTL، وإذا تجاوز
        عدد الجلسات الحد الأقصى يُخلي الأقدم نشاطاً (باستثناء الجلسة الافتراضية).
        يُستدعى دوماً داخل self._lock — لا يُستدعى مباشرة من الخارج.
        """
        now = time.time()
        stale = [
            sid for sid, s in self._sessions.items()
            if sid != self._default_id and (now - s.last_active_ts) > _SESSION_TTL_SECONDS
        ]
        for sid in stale:
            del self._sessions[sid]
        if len(self._sessions) > _MAX_SESSIONS:
            ordered = sorted(
                (s for sid, s in self._sessions.items() if sid != self._default_id),
                key=lambda s: s.last_active_ts,
            )
            overflow = len(self._sessions) - _MAX_SESSIONS
            for s in ordered[:overflow]:
                self._sessions.pop(s.id, None)

    def _is_safe_command(self, cmd: str) -> Tuple[bool, str]:
        """يتحقق من أمان الأمر بالكامل، بما فيه أي أجزاء مسلسلة (&&, ;, |, ||).

        🔒 إصلاح أمني: الفحص القديم كان يتحقق فقط من بادئة السلسلة الكاملة،
        فأمر مثل "git status && rm -rf ai" أو "echo hi | bash" كان يمرّ لأن
        بادئته "git status"/"echo " مسموحة — بينما ينفَّذ عبر shell=True بكامل
        السلسلة. الآن نمنع رموز الاستبدال/التوجيه/الخلفية كلياً في الوضع الآمن،
        ونقسّم أي تسلسل (&&, ||, ;, |) ونتحقق من كل جزء منه بشكل مستقل.
        """
        c = cmd.strip()
        for bad in _BLOCKED:
            if bad in c.lower():
                return False, f"أمر محظور: {bad.strip()}"
        masked = _mask_quotes(c)
        if "$(" in masked or "`" in masked:
            return False, "استبدال أوامر ($() أو `) غير مسموح في الوضع الآمن — استخدم mode=admin"
        if ">" in masked or "<" in masked:
            return False, "إعادة توجيه (> أو <) غير مسموحة في الوضع الآمن — استخدم mode=admin"
        if "&" in masked.replace("&&", ""):
            return False, "تشغيل بالخلفية (&) غير مسموح في الوضع الآمن — استخدم mode=admin"
        segments = _split_shell_segments(c)
        if len(segments) > 1:
            for seg in segments:
                ok, reason = self._is_single_command_safe(seg)
                if not ok:
                    return False, f"جزء غير آمن ضمن السلسلة: '{seg}' — {reason}"
            return True, ""
        return self._is_single_command_safe(c)

    def _is_single_command_safe(self, c: str) -> Tuple[bool, str]:
        low = c.lower()
        for bad in _BLOCKED:
            if bad in low:
                return False, f"أمر محظور: {bad.strip()}"
        if low == "cd" or low.startswith("cd "):
            return True, ""
        if low.startswith("python -m ") or low.startswith("python3 -m "):
            return True, ""
        for p in _SAFE_PREFIXES:
            if low == p.strip() or low.startswith(p.strip() + " ") or low.startswith(p):
                return True, ""
        # allow simple relative scripts
        if re.match(r"^(python3?|pytest)\s+\S+\.py(\s|$)", low):
            return True, ""
        return False, "الأمر خارج القائمة الآمنة — استخدم mode=admin من وضع المالك"

    def _handle_export(self, sess: TerminalSession, cmd: str) -> CommandResult:
        """export KEY=VALUE [KEY2=VALUE2 ...] — يحفظ المتغيرات في بيئة الجلسة
        فتصبح متاحة لكل الأوامر التالية ضمن نفس الجلسة (سلوك طرفية حقيقية،
        بخلاف subprocess عادي حيث تُفقد متغيرات export بعد كل أمر)."""
        try:
            tokens = shlex.split(cmd)[1:]
        except ValueError as e:
            return CommandResult(ok=False, cmd=cmd, cwd=sess.cwd, exit_code=1,
                                  stdout="", stderr="", duration_ms=0, mode=sess.mode, error=str(e))
        if not tokens:
            lines = [f"{k}={v}" for k, v in sess.env.items()]
            return CommandResult(ok=True, cmd=cmd, cwd=sess.cwd, exit_code=0,
                                  stdout="\n".join(lines), stderr="", duration_ms=0, mode=sess.mode)
        set_names = []
        for tok in tokens:
            if "=" not in tok:
                return CommandResult(ok=False, cmd=cmd, cwd=sess.cwd, exit_code=1,
                                      stdout="", stderr="", duration_ms=0, mode=sess.mode,
                                      error=f"صيغة غير صحيحة: {tok} — استخدم KEY=VALUE")
            key, _, val = tok.partition("=")
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                return CommandResult(ok=False, cmd=cmd, cwd=sess.cwd, exit_code=1,
                                      stdout="", stderr="", duration_ms=0, mode=sess.mode,
                                      error=f"اسم متغير غير صالح: {key}")
            sess.env[key] = val
            set_names.append(key)
        return CommandResult(ok=True, cmd=cmd, cwd=sess.cwd, exit_code=0,
                              stdout=f"exported: {', '.join(set_names)}", stderr="",
                              duration_ms=0, mode=sess.mode)

    def _handle_unset(self, sess: TerminalSession, cmd: str) -> CommandResult:
        tokens = shlex.split(cmd)[1:]
        removed = [k for k in tokens if sess.env.pop(k, None) is not None]
        return CommandResult(ok=True, cmd=cmd, cwd=sess.cwd, exit_code=0,
                              stdout=f"unset: {', '.join(removed) or '(none)'}", stderr="",
                              duration_ms=0, mode=sess.mode)

    def _handle_cd(self, sess: TerminalSession, cmd: str) -> CommandResult:
        parts = shlex.split(cmd)
        target = parts[1] if len(parts) > 1 else str(self.root)
        if target == "-":
            target = str(self.root)
        path = Path(target)
        if not path.is_absolute():
            path = (Path(sess.cwd) / path).resolve()
        else:
            path = path.resolve()
        try:
            path.relative_to(self.root)
        except Exception:
            return CommandResult(
                ok=False, cmd=cmd, cwd=sess.cwd, exit_code=1,
                stdout="", stderr="", duration_ms=0, mode=sess.mode,
                error="مسار خارج جذر المشروع",
            )
        if not path.is_dir():
            return CommandResult(
                ok=False, cmd=cmd, cwd=sess.cwd, exit_code=1,
                stdout="", stderr="", duration_ms=0, mode=sess.mode,
                error=f"ليس مجلداً: {path}",
            )
        sess.cwd = str(path)
        return CommandResult(
            ok=True, cmd=cmd, cwd=sess.cwd, exit_code=0,
            stdout=sess.cwd, stderr="", duration_ms=0, mode=sess.mode,
        )

    def run(
        self,
        cmd: str,
        session_id: Optional[str] = None,
        mode: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT,
        env_extra: Optional[Dict[str, str]] = None,
    ) -> CommandResult:
        cmd = (cmd or "").strip()
        # 🆕 توسيع aliases (gs => git status ...) — لا ينطبق على الأوامر المدمجة الخاصة
        if not cmd.startswith(("cd", "export", "unset", "pwd", "clear")):
            cmd = self._expand_alias(cmd)
        sess = self.get_session(session_id)
        sess.last_active_ts = time.time()
        if mode in ("safe", "admin"):
            sess.mode = mode
        if not cmd:
            return CommandResult(
                ok=False, cmd="", cwd=sess.cwd, exit_code=1,
                stdout="", stderr="", duration_ms=0, mode=sess.mode, error="أمر فارغ",
            )

        # builtins
        if cmd == "pwd":
            r = CommandResult(ok=True, cmd=cmd, cwd=sess.cwd, exit_code=0,
                              stdout=sess.cwd, stderr="", duration_ms=0, mode=sess.mode)
            self._push_history(sess, r)
            return r
        if cmd == "clear":
            sess.history.clear()
            r = CommandResult(ok=True, cmd=cmd, cwd=sess.cwd, exit_code=0,
                              stdout="", stderr="", duration_ms=0, mode=sess.mode)
            return r
        if cmd.startswith("cd"):
            r = self._handle_cd(sess, cmd)
            self._push_history(sess, r)
            return r
        if cmd == "export" or cmd.startswith("export "):
            r = self._handle_export(sess, cmd)
            self._push_history(sess, r)
            return r
        # 🆕 أوامر aliases: alias / alias name=cmd / unalias name
        if cmd == "alias" or cmd.startswith("alias "):
            body = cmd[len("alias"):].strip()
            if not body:
                listing = "\n".join(f"{k}={v}" for k, v in self.aliases.items()) or "(لا توجد)"
                r = CommandResult(ok=True, cmd=cmd, cwd=sess.cwd, exit_code=0,
                                  stdout=listing, stderr="", duration_ms=0, mode=sess.mode)
            else:
                if body.startswith("rm ") or body.startswith("delete "):
                    ok2, msg = self.del_alias(body.split(None, 1)[1].strip())
                elif "=" in body:
                    name, _, value = body.partition("=")
                    ok2, msg = self.set_alias(name.strip(), value)
                else:
                    ok2, msg = (body in self.aliases, self.aliases.get(body, "غير معرّف"))
                r = CommandResult(ok=ok2, cmd=cmd, cwd=sess.cwd, exit_code=0 if ok2 else 1,
                                  stdout=msg if ok2 else "", stderr="" if ok2 else msg,
                                  duration_ms=0, mode=sess.mode)
            self._push_history(sess, r)
            return r
        # 🆕 audit — عرض سجل التدقيق لآخر N أوامر
        if cmd == "audit" or cmd.startswith("audit "):
            try:
                n = int(cmd.split()[1]) if len(cmd.split()) > 1 else 30
                events = self.role_manager.audit_events(limit=min(n, 200))
                lines = [json.dumps(e, ensure_ascii=False) for e in events]
                r = CommandResult(ok=True, cmd=cmd, cwd=sess.cwd, exit_code=0,
                                  stdout="\n".join(lines) or "(سجل فارغ)", stderr="",
                                  duration_ms=0, mode=sess.mode)
            except Exception as e:
                r = CommandResult(ok=False, cmd=cmd, cwd=sess.cwd, exit_code=1,
                                  stdout="", stderr=str(e), duration_ms=0, mode=sess.mode)
            self._push_history(sess, r)
            return r
        # 🆕 snapshot — حفظ/استعادة حالة الجلسات
        if cmd == "snapshot save" or cmd == "snapshot save_now":
            ok2 = self.save_sessions_snapshot()
            r = CommandResult(ok=ok2, cmd=cmd, cwd=sess.cwd, exit_code=0 if ok2 else 1,
                              stdout="حُفظ snapshot" if ok2 else "فشل الحفظ",
                              stderr="" if ok2 else "فشل الحفظ", duration_ms=0, mode=sess.mode)
            self._push_history(sess, r)
            return r
        if cmd == "snapshot restore":
            n = self.restore_sessions_snapshot()
            r = CommandResult(ok=True, cmd=cmd, cwd=sess.cwd, exit_code=0,
                              stdout=f"استُعيدت {n} جلسة", stderr="",
                              duration_ms=0, mode=sess.mode)
            self._push_history(sess, r)
            return r
        if cmd == "unset" or cmd.startswith("unset "):
            r = self._handle_unset(sess, cmd)
            self._push_history(sess, r)
            return r

        # 🆕 أوامر الخلفية: jobs / tail <id> [n] / kill <id>
        if cmd == "jobs" or cmd.startswith("jobs "):
            try:
                n = int(cmd.split()[1]) if len(cmd.split()) > 1 else 15
                jobs = self.list_jobs()[:n]
                if not jobs:
                    r = CommandResult(ok=True, cmd=cmd, cwd=sess.cwd, exit_code=0,
                                      stdout="(لا توجد مهام خلفية)", stderr="", duration_ms=0,
                                      mode=sess.mode)
                else:
                    lines = []
                    for j in jobs:
                        lines.append(
                            f"[{j['status']}] {j['id']}  {j['cmd'][:80]}  "
                            f"exit={j.get('exit_code') if j['exit_code'] is not None else '—'}  "
                            f"lines={j.get('lines', 0)}"
                        )
                    r = CommandResult(ok=True, cmd=cmd, cwd=sess.cwd, exit_code=0,
                                      stdout="\n".join(lines), stderr="", duration_ms=0,
                                      mode=sess.mode)
            except ValueError:
                r = CommandResult(ok=False, cmd=cmd, cwd=sess.cwd, exit_code=1,
                                  stdout="", stderr="jobs <n>", duration_ms=0, mode=sess.mode)
            self._push_history(sess, r)
            return r
        if cmd.startswith("tail ") or cmd.startswith("tail-job "):
            try:
                parts = cmd.split()
                tid = parts[1]
                n = int(parts[2]) if len(parts) > 2 else 40
            except (ValueError, IndexError):
                r = CommandResult(ok=False, cmd=cmd, cwd=sess.cwd, exit_code=1,
                                  stdout="", stderr="tail <job_id> [n]", duration_ms=0,
                                  mode=sess.mode)
                self._push_history(sess, r)
                return r
            with self._jobs_lock:
                job = self._jobs.get(tid)
            if job is None:
                r = CommandResult(ok=False, cmd=cmd, cwd=sess.cwd, exit_code=1,
                                  stdout="", stderr=f"لا توجد مهمة: {tid}", duration_ms=0,
                                  mode=sess.mode)
                self._push_history(sess, r)
                return r
            lines = job.tail(n)
            status_line = (f"status={job.status} · exit={job.exit_code if job.exit_code is not None else '—'} "
                           f"· {job.duration_s():.1f}s")
            r = CommandResult(ok=True, cmd=cmd, cwd=sess.cwd, exit_code=0,
                              stdout="\n".join(lines), stderr=status_line, duration_ms=0,
                              mode=sess.mode)
            self._push_history(sess, r)
            return r
        if cmd.startswith("kill ") or cmd.startswith("stop "):
            tid = cmd.split(maxsplit=1)[1].strip()
            res = self.stop_job(tid)
            r = CommandResult(ok=res.get("ok"), cmd=cmd, cwd=sess.cwd,
                              exit_code=0 if res.get("ok") else 1,
                              stdout=res.get("msg", ""), stderr=res.get("error", ""),
                              duration_ms=0, mode=sess.mode)
            self._push_history(sess, r)
            return r

        # 🆕 أوامر Kaggle shortcuts (kg status/logs/list/output)
        _kag_ok, _kag_key, _kag_args = self._match_kaggle_cmd(cmd)
        if _kag_ok:
            r = self._run_kaggle_cmd(sess, cmd, _kag_key, _kag_args)
            self._push_history(sess, r)
            return r

        if sess.mode != "admin":
            # 🆕 أوامر kaggle CLI للوكلاء المسجلين لا تحتاج mode=admin
            # (الفحص الدوراني جرى أصلًا في run_agent قبل الوصول هنا)
            is_kag = cmd.lower().startswith("kaggle ") or cmd == "kaggle"
            if is_kag and sess.agent and self.role_manager.role_of(sess.agent).get(
                    "can_use_kaggle_cli"):
                allowed2, _ = self.role_manager.check(
                    sess.agent, cmd, is_kaggle_cli=True)
                if allowed2:
                    # جهّز البيئة ثم نفذ مباشرة (كـ kaggle CLI للوكيل)
                    env = os.environ.copy()
                    env["PYTHONUNBUFFERED"] = "1"
                    env.update(sess.env)
                    if env_extra:
                        env.update(env_extra)
                    return self._run_shell(sess, cmd, time.time(), env, timeout)
            ok, reason = self._is_safe_command(cmd)
            if not ok:
                r = CommandResult(
                    ok=False, cmd=cmd, cwd=sess.cwd, exit_code=126,
                    stdout="", stderr=reason, duration_ms=0, mode=sess.mode, error=reason,
                )
                self._push_history(sess, r)
                return r

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TERM"] = "xterm-256color"
        env.update(sess.env)  # 🆕 متغيرات export المستمرة الخاصة بالجلسة
        if env_extra:
            env.update(env_extra)
        t0 = time.time()
        return self._run_shell(sess, cmd, t0, env, timeout)

    # ── Kaggle shortcuts ──────────────────────────────
    @staticmethod
    def _match_kaggle_cmd(cmd: str):
        try:
            from ai.terminal_smart import is_kaggle_cmd
            return is_kaggle_cmd(cmd)
        except Exception:
            return False, None, []

    def _run_kaggle_cmd(self, sess, cmd, key, args):
        from ai.terminal_smart import KAGGLE_COMMANDS
        t0 = time.time()
        try:
            exit_code, stdout, stderr = KAGGLE_COMMANDS[key](args)
        except Exception as e:
            return CommandResult(ok=False, cmd=cmd, cwd=sess.cwd, exit_code=1,
                                 stdout="", stderr=str(e), duration_ms=0, mode=sess.mode)
        return CommandResult(
            ok=exit_code == 0, cmd=cmd, cwd=sess.cwd, exit_code=exit_code,
            stdout=stdout, stderr=stderr, duration_ms=int((time.time() - t0) * 1000),
            mode=sess.mode, error=(stderr or "") if exit_code != 0 else "",
        )

    def _run_shell(self, sess, cmd, t0, env, timeout):
        # منطق subprocess التنفيذي المشترك (يُستخدم من run ومن bypass kaggle CLI)
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=max(5, min(int(timeout), 300)),
                cwd=sess.cwd,
                env=env,
            )
            out = redact(proc.stdout or "")
            err = redact(proc.stderr or "")
            if len(out) > _MAX_OUTPUT:
                out = out[:_MAX_OUTPUT] + "\n... [truncated]"
            if len(err) > _MAX_OUTPUT:
                err = err[:_MAX_OUTPUT] + "\n... [truncated]"
            r = CommandResult(
                ok=proc.returncode == 0,
                cmd=cmd,
                cwd=sess.cwd,
                exit_code=proc.returncode,
                stdout=out,
                stderr=err,
                duration_ms=int((time.time() - t0) * 1000),
                mode=sess.mode,
            )
        except subprocess.TimeoutExpired:
            r = CommandResult(
                ok=False, cmd=cmd, cwd=sess.cwd, exit_code=124,
                stdout="", stderr="", duration_ms=int((time.time() - t0) * 1000),
                mode=sess.mode, error=f"timeout after {timeout}s",
            )  # placeholder
        except Exception as e:
            r = CommandResult(
                ok=False, cmd=cmd, cwd=sess.cwd, exit_code=1,
                stdout="", stderr="", duration_ms=int((time.time() - t0) * 1000),
                mode=sess.mode, error=str(e),
            )
        self._push_history(sess, r)
        return r

    def run_safe(self, cmd: str, **kwargs) -> CommandResult:
        kwargs["mode"] = "safe"
        return self.run(cmd, **kwargs)

    # ---------------- 🆕 تشغيل خلفي (background jobs) ----------------
    # التنفيذ الحالي عبر run() متزامن بالكامل: يحجب حتى انتهاء الأمر أو
    # انتهاء المهلة، فلا توجد وسيلة لمراقبة أمر طويل (تدريب، بناء...) أو
    # إيقافه أثناء عمله. الطبقة التالية تضيف تنفيذاً حقيقياً بخيط منفصل
    # مع مقبض Popen فعلي يمكن استعلامه أو قتله في أي لحظة.
    def start_background(
        self, cmd: str, session_id: Optional[str] = None,
        mode: Optional[str] = None, env_extra: Optional[Dict[str, str]] = None,
        timeout: int = 300,
    ) -> BackgroundJob:
        cmd = (cmd or "").strip()
        sess = self.get_session(session_id)
        if mode in ("safe", "admin"):
            sess.mode = mode
        job_id = uuid.uuid4().hex[:10]
        job = BackgroundJob(id=job_id, cmd=cmd, session_id=sess.id, cwd=sess.cwd,
                            mode=sess.mode, timeout=int(timeout or 0) or 0)

        if not cmd:
            job.status, job.error = "error", "أمر فارغ"
            with self._jobs_lock:
                self._jobs[job_id] = job
            return job
        if sess.mode != "admin":
            ok, reason = self._is_safe_command(cmd)
            if not ok:
                job.status, job.error = "error", reason
                with self._jobs_lock:
                    self._jobs[job_id] = job
                return job

        with self._jobs_lock:
            self._jobs[job_id] = job

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TERM"] = "xterm-256color"
        env.update(sess.env)
        if env_extra:
            env.update(env_extra)

        def _worker():
            t0 = time.time()
            try:
                proc = subprocess.Popen(
                    cmd, shell=True, cwd=sess.cwd, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                with self._jobs_lock:
                    self._procs[job_id] = proc
                # 🆕 بث حي: قراءة غير حاجزة لسطور stdout/stderr وإلحاقها بالمهمة
                # (نستخدم select على POSIX وسقوط آمن للباقي عبر polling قصير)
                import selectors as _sel
                try:
                    sel = _sel.DefaultSelector()
                    sel.register(proc.stdout, _sel.EVENT_READ, "out")
                    sel.register(proc.stderr, _sel.EVENT_READ, "err")
                    timeout_deadline = (t0 + job.timeout) if job.timeout else None
                    while proc.poll() is None:
                        if timeout_deadline and time.time() >= timeout_deadline:
                            with self._jobs_lock:
                                j = self._jobs.get(job_id)
                                if j and j.status == "running":
                                    j.status, j.error, j.exit_code = "timed_out", (
                                        f"timeout after {job.timeout}s"), 124
                                    try:
                                        proc.kill()
                                    except Exception:
                                        pass
                            break
                        for key, _ in sel.select(timeout=0.15):
                            with key.fileobj as stream:
                                tag = key.data
                                for line in iter(stream.readline, ""):
                                    job.append_line(f"[{tag}] {line.rstrip()}")
                    sel.close()
                except Exception:
                    pass
                # التماس النهائي لبقية المخرجات غير المقروءة
                try:
                    out, err = proc.communicate(timeout=3)
                    for line in (out or "").splitlines():
                        if line.strip():
                            job.append_line(f"[out] {line}")
                    for line in (err or "").splitlines():
                        if line.strip():
                            job.append_line(f"[err] {line}")
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                with self._jobs_lock:
                    j = self._jobs.get(job_id)
                    if j:
                        j.exit_code = proc.returncode
                        j.stdout = redact(j.stdout or "")[:_MAX_OUTPUT]
                        j.stderr = redact(j.stderr or "")[:_MAX_OUTPUT]
                        # إبقاء status=timed_out إن ضُبط أثناء الحلقة
                        if j.status == "running":
                            j.status = "done"
                        j.finished_at = j.finished_at or _now()
                    self._procs.pop(job_id, None)
            except Exception as e:
                with self._jobs_lock:
                    j = self._jobs.get(job_id)
                    if j:
                        j.status, j.error, j.finished_at = "error", str(e), _now()
                    self._procs.pop(job_id, None)
            finally:
                with self._jobs_lock:
                    _j = self._jobs.get(job_id)
                r = CommandResult(
                    ok=(_j.exit_code == 0) if _j else False,
                    cmd=cmd, cwd=sess.cwd, exit_code=_j.exit_code or 0,
                    stdout=_j.stdout if _j else "", stderr=_j.stderr if _j else "",
                    duration_ms=int((time.time() - t0) * 1000), mode=sess.mode,
                    error=_j.error if _j else "",
                )
                self._push_history(sess, r)

        threading.Thread(target=_worker, daemon=True).start()
        return job

    def job_status(self, job_id: str) -> Optional[dict]:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def list_jobs(self, session_id: Optional[str] = None) -> List[dict]:
        with self._jobs_lock:
            jobs = list(self._jobs.values())
        if session_id:
            jobs = [j for j in jobs if j.session_id == session_id]
        return [j.to_dict() for j in sorted(jobs, key=lambda j: j.started_at, reverse=True)]

    def kill_job(self, job_id: str) -> bool:
        """يقتل مهمة خلفية فعلياً (SIGKILL عبر Popen.kill)، وليس مجرد تعليم حالة."""
        with self._jobs_lock:
            proc = self._procs.get(job_id)
            job = self._jobs.get(job_id)
            if not job or job.status != "running":
                return False
            job.status = "killed"
            job.finished_at = _now()
            try:
                if proc is not None:
                    proc.kill()
            except Exception:
                pass
        return True

    def stop_job(self, job_id: str) -> dict:
        """🆕 إيقاف آمن شامل: SIGTERM أولًا ثم SIGKILL بعد 3 ثوانٍ إن لزم.
        يعيد نتيجة قراءة (ok, error) للواجهة."""
        with self._jobs_lock:
            proc = self._procs.get(job_id)
            job = self._jobs.get(job_id)
            if not job:
                return {"ok": False, "error": f"لا توجد مهمة: {job_id}"}
            if job.status != "running":
                return {"ok": False, "error": f"المهمة {job.status} بالفعل — لا تحتاج إيقافًا"}
            job.status = "killed"
            job.finished_at = _now()
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            # SIGKILL احتياط بعد 3 ثوانٍ إن بقي حيًا
            def _force_kill():
                time.sleep(3)
                try:
                    if proc.poll() is None:
                        proc.kill()
                except Exception:
                    pass
            threading.Thread(target=_force_kill, daemon=True).start()
        return {"ok": True, "msg": "أُرسل أمر الإيقاف — إن لم ينته خلال 3 ثوانٍ سيُقتل نهائيًا"}

    def clear_job(self, job_id: str) -> bool:
        """🆕 مسح مهمة منتهية (غير جارية) من سجل المهام لتخفيف اللوحة."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job or job.status == "running":
                return False
            self._jobs.pop(job_id, None)
        return True

    def _push_history(self, sess: TerminalSession, result: CommandResult) -> None:
        entry = result.to_dict()
        sess.history.append(entry)
        if len(sess.history) > _MAX_HISTORY:
            sess.history = sess.history[-_MAX_HISTORY:]
        _append_log({"session": sess.id, **entry})

    def history(self, session_id: Optional[str] = None, limit: int = 30) -> List[dict]:
        sess = self.get_session(session_id)
        return sess.history[-limit:]

    def quick(self, name: str, session_id: Optional[str] = None, mode: str = "safe") -> CommandResult:
        presets = {
            "status": "git status --short --branch",
            "log": "git log -12 --oneline --decorate",
            "diff": "git diff --stat HEAD",
            "pytest": "python3 -m pytest -q --tb=line 2>/dev/null | tail -40",
            "compile_ai": "python3 -m compileall -q ai && echo COMPILE_OK",
            "tree": "ls -la",
            "disk": "du -sh . 2>/dev/null; df -h . | tail -1",
            "python": "python3 --version",
            "branch": "git branch -vv",
            "lfs": "git lfs ls-files 2>/dev/null | head -20 || echo 'no lfs files'",
            "jobs": "jobs",
            "kill_last": None,  # خاص — يُعالج يدويًا أدناه
        }
        cmd = presets.get(name)
        if cmd is None and name == "kill_last":
            # يوقف آخر مهمة خلفية جارية (إن وجدت)
            jobs = self.list_jobs(session_id=session_id)
            running = [j for j in jobs if j.get("status") == "running"]
            if not running:
                return CommandResult(
                    ok=False, cmd=name, cwd=str(self.root), exit_code=1,
                    stdout="", stderr="لا توجد مهام خلفية جارية", duration_ms=0, mode=mode,
                )
            res = self.stop_job(running[0]["id"])
            return CommandResult(
                ok=bool(res.get("ok")), cmd=name, cwd=str(self.root), exit_code=0,
                stdout=res.get("msg", ""), stderr=res.get("error", ""), duration_ms=0, mode=mode,
            )
        if not cmd:
            return CommandResult(
                ok=False, cmd=name, cwd=str(self.root), exit_code=1,
                stdout="", stderr="", duration_ms=0, mode=mode,
                error=f"preset unknown: {name}. available: {', '.join(presets)}",
            )
        return self.run(cmd, session_id=session_id, mode=mode, timeout=90)


_singleton: Optional[NSMTerminal] = None
_slock = threading.Lock()


def get_terminal() -> NSMTerminal:
    global _singleton
    with _slock:
        if _singleton is None:
            _singleton = NSMTerminal()
        return _singleton


def handle_terminal_command(user_input: str) -> Optional[str]:
    """أوامر محادثة: طرفية / terminal / نفّذ طرفية."""
    import json
    t = (user_input or "").strip()
    if not t:
        return None
    if not re.search(r"(طرفية|terminal|\$\s|نف[ّذ]ذ\s*طرفية)", t, re.I):
        return None

    term = get_terminal()
    low = t.lower()

    if re.match(r"^(طرفية|terminal|حالة\s*الطرفية)$", t, re.I):
        sess = term.get_session()
        return (
            "## 💻 NSM Terminal\n```json\n"
            + json.dumps({"session": sess.to_dict(), "sessions": term.list_sessions()}, ensure_ascii=False, indent=2)
            + "\n```\nاستخدم: `طرفية <أمر>` أو `terminal <cmd>`"
        )

    m = re.match(r"^(طرفية|terminal|نف[ّذ]ذ\s*طرفية)\s+(.+)$", t, re.I)
    if m:
        cmd = m.group(2).strip()
        # quick presets
        if cmd.startswith("!"):
            r = term.quick(cmd[1:].strip())
        else:
            r = term.run_safe(cmd)
        return f"## 💻 Terminal\n```\n{r.formatted()}\n```"

    return None
