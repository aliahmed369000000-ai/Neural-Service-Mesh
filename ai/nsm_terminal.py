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


_MAX_LOG_BYTES = 2_000_000  # 2MB — تدوير تلقائي لمنع نمو غير محدود
_MAX_SESSIONS = 40  # حد أقصى للجلسات المتزامنة — إخلاء الأقدم نشاطاً عند التجاوز
_SESSION_TTL_SECONDS = 6 * 3600  # جلسات خاملة أكثر من 6 ساعات تُخلى تلقائياً


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

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "cwd": self.cwd,
            "mode": self.mode,
            "history_len": len(self.history),
            "created_at": self.created_at,
            "env": dict(self.env),
            "last": self.history[-1] if self.history else None,
        }


@dataclass
class BackgroundJob:
    id: str
    cmd: str
    session_id: str
    cwd: str
    mode: str
    status: str = "running"  # running | done | killed | error
    started_at: str = field(default_factory=_now)
    finished_at: str = ""
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "cmd": self.cmd, "session_id": self.session_id, "cwd": self.cwd,
            "mode": self.mode, "status": self.status, "started_at": self.started_at,
            "finished_at": self.finished_at, "exit_code": self.exit_code,
            "stdout": self.stdout, "stderr": self.stderr, "error": self.error,
        }


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
        self._lock = threading.Lock()
        self._jobs: Dict[str, BackgroundJob] = {}
        self._procs: Dict[str, subprocess.Popen] = {}
        self._jobs_lock = threading.Lock()
        self._default_id: Optional[str] = None  # يُضبط بعد إنشاء الجلسة الافتراضية أدناه
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
        if cmd == "unset" or cmd.startswith("unset "):
            r = self._handle_unset(sess, cmd)
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
        env.update(sess.env)  # 🆕 متغيرات export المستمرة الخاصة بالجلسة
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

    # ---------------- 🆕 تشغيل خلفي (background jobs) ----------------
    # التنفيذ الحالي عبر run() متزامن بالكامل: يحجب حتى انتهاء الأمر أو
    # انتهاء المهلة، فلا توجد وسيلة لمراقبة أمر طويل (تدريب، بناء...) أو
    # إيقافه أثناء عمله. الطبقة التالية تضيف تنفيذاً حقيقياً بخيط منفصل
    # مع مقبض Popen فعلي يمكن استعلامه أو قتله في أي لحظة.
    def start_background(
        self, cmd: str, session_id: Optional[str] = None,
        mode: Optional[str] = None, env_extra: Optional[Dict[str, str]] = None,
    ) -> BackgroundJob:
        cmd = (cmd or "").strip()
        sess = self.get_session(session_id)
        if mode in ("safe", "admin"):
            sess.mode = mode
        job_id = uuid.uuid4().hex[:10]
        job = BackgroundJob(id=job_id, cmd=cmd, session_id=sess.id, cwd=sess.cwd, mode=sess.mode)

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
                out, err = proc.communicate()
                with self._jobs_lock:
                    j = self._jobs.get(job_id)
                    if j:
                        j.exit_code = proc.returncode
                        j.stdout = redact(out or "")[:_MAX_OUTPUT]
                        j.stderr = redact(err or "")[:_MAX_OUTPUT]
                        if j.status == "running":  # لم يُقتل يدوياً أثناء التنفيذ
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
                r = CommandResult(
                    ok=(self._jobs.get(job_id).exit_code == 0) if self._jobs.get(job_id) else False,
                    cmd=cmd, cwd=sess.cwd, exit_code=self._jobs[job_id].exit_code or 0,
                    stdout=self._jobs[job_id].stdout, stderr=self._jobs[job_id].stderr,
                    duration_ms=int((time.time() - t0) * 1000), mode=sess.mode,
                    error=self._jobs[job_id].error,
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
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
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
