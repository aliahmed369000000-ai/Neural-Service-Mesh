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

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "cwd": self.cwd,
            "mode": self.mode,
            "history_len": len(self.history),
            "created_at": self.created_at,
            "last": self.history[-1] if self.history else None,
        }


class NSMTerminal:
    """طرفية مشتركة — جلسة واحدة افتراضية + جلسات مسماة."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or ROOT).resolve()
        self._sessions: Dict[str, TerminalSession] = {}
        self._lock = threading.Lock()
        self._default_id = self.create_session(mode="safe").id

    def create_session(self, mode: str = "safe", cwd: Optional[str] = None) -> TerminalSession:
        sid = uuid.uuid4().hex[:10]
        start = Path(cwd).resolve() if cwd else self.root
        try:
            start.relative_to(self.root)
        except Exception:
            start = self.root
        sess = TerminalSession(id=sid, cwd=str(start), mode=mode if mode in ("safe", "admin") else "safe")
        with self._lock:
            self._sessions[sid] = sess
        return sess

    def get_session(self, session_id: Optional[str] = None) -> TerminalSession:
        with self._lock:
            sid = session_id or self._default_id
            if sid not in self._sessions:
                sess = TerminalSession(id=sid, cwd=str(self.root), mode="safe")
                self._sessions[sid] = sess
            return self._sessions[sid]

    def list_sessions(self) -> List[dict]:
        with self._lock:
            return [s.to_dict() for s in self._sessions.values()]

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
        if "$(" in c or "`" in c:
            return False, "استبدال أوامر ($() أو `) غير مسموح في الوضع الآمن — استخدم mode=admin"
        if ">" in c or "<" in c:
            return False, "إعادة توجيه (> أو <) غير مسموحة في الوضع الآمن — استخدم mode=admin"
        if "&" in c.replace("&&", ""):
            return False, "تشغيل بالخلفية (&) غير مسموح في الوضع الآمن — استخدم mode=admin"
        segments = [s.strip() for s in re.split(r"\|\||&&|;|\|", c) if s.strip()]
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
        sess = self.get_session(session_id)
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

        if sess.mode != "admin":
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
        # strip tokens from child env display risk — keep for git if needed but redact output
        if env_extra:
            env.update(env_extra)

        t0 = time.time()
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
            )
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
        }
        cmd = presets.get(name)
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
