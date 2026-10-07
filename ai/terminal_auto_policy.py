"""Safe automatic terminal policy for NSM agents / node hands.

قواعد ثابتة:
- بلا shell=True، بلا مشغّلات (; && || | > >> <)
- بلا كتابة/حذف/شبكة/صلاحيات
- قائمة مسموحة تتوسع بحذر لفحوص وقراءة فقط
"""
from __future__ import annotations

import fnmatch
import os
import posixpath
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TerminalDecision:
    allowed: bool
    reason: str
    command: tuple[str, ...] = ()


# أوامر كاملة مسموحة حرفياً
_ALLOWED_EXACT = {
    ("git", "status"),
    ("git", "status", "-sb"),
    ("git", "status", "--short"),
    ("git", "diff", "--check"),
    ("git", "diff", "--stat"),
    ("git", "diff", "--stat", "HEAD"),
    ("git", "branch"),
    ("git", "branch", "-v"),
    ("git", "branch", "--show-current"),
    ("git", "log", "--oneline", "-5"),
    ("git", "log", "--oneline", "-10"),
    ("git", "log", "--oneline", "-20"),
    ("git", "rev-parse", "HEAD"),
    ("git", "rev-parse", "--abbrev-ref", "HEAD"),
    ("git", "ls-files"),
    ("pwd",),
    ("uname",),
    ("uname", "-a"),
    ("python", "--version"),
    ("python3", "--version"),
    ("pytest", "--version"),
    ("pip", "--version"),
    ("pip3", "--version"),
}

# بادئات مسموحة (الجزء الأول يطابق ثم قيود على الباقي)
_ALLOWED_PREFIXES = {
    ("git", "status"),
    ("git", "diff"),
    ("git", "log"),
    ("git", "show"),
    ("git", "ls-files"),
    ("git", "branch"),
    ("git", "rev-parse"),
}

_BLOCKED_TOKENS = {
    "rm", "rmdir", "del", "sudo", "chmod", "chown", "chgrp",
    "curl", "wget", "ssh", "scp", "nc", "ncat", "telnet",
    "export", "printenv", "env",
    "mkfs", "dd", "shutdown", "reboot", "kill", "killall",
    "apt", "apt-get", "yum", "dnf", "pip", "pip3",  # pip install blocked via deeper check
}

# عبارات محظورة حتى لو ظهرت كجزء من النص
_BLOCKED_PHRASES = (
    "git push", "git reset", "git clean", "git checkout", "git commit",
    "git rebase", "git merge", "git pull", "git fetch", "git remote",
    "pip install", "pip3 install", "pip uninstall",
    "rm -", "sudo ",
)

_SAFE_GIT_SUB = {
    "status", "diff", "log", "show", "ls-files", "branch", "rev-parse",
}

_SAFE_PYTHON_M = {
    "compileall", "py_compile", "pytest", "unittest",
}


def decide(command: str | Sequence[str]) -> TerminalDecision:
    try:
        parts = tuple(command) if not isinstance(command, str) else tuple(shlex.split(command))
    except ValueError:
        return TerminalDecision(False, "تعذر تحليل الأمر")
    if not parts:
        return TerminalDecision(False, "الأمر فارغ")

    lowered = " ".join(parts).lower()
    for phrase in _BLOCKED_PHRASES:
        if phrase in lowered:
            return TerminalDecision(False, f"عبارة محظورة: {phrase.strip()}")

    if any(x in parts for x in (";", "&&", "||", "|", ">", ">>", "<", "`", "$(", "${")):
        return TerminalDecision(False, "shell operators غير مسموحة")

    # رموز خطرة في أي جزء
    for p in parts:
        if any(ch in p for ch in (";", "|", "&", ">", "<", "`", "\n", "\r")):
            return TerminalDecision(False, "محارف shell غير مسموحة في المعاملات")

    # حظر رموز محظورة كأوامر منفصلة (ما عدا pip --version المعالج أدناه)
    head = parts[0].lower()
    if head in _BLOCKED_TOKENS and head not in ("pip", "pip3"):
        return TerminalDecision(False, f"الأمر محظور: {head}")

    # تطابق حرفي
    if parts in _ALLOWED_EXACT:
        return TerminalDecision(True, "فحص قراءة فقط مسموح تلقائياً", parts)

    # pytest / python -m pytest
    if parts[0] in ("pytest", "py.test"):
        if _pytest_args_ok(parts[1:]):
            return TerminalDecision(True, "اختبار pytest مسموح تلقائياً", parts)
        return TerminalDecision(False, "معاملات pytest غير مسموحة")

    if parts[0] in ("python", "python3") and len(parts) >= 2:
        if parts[1] == "--version":
            return TerminalDecision(True, "إصدار Python", parts)
        if parts[1] == "-m" and len(parts) >= 3:
            mod = parts[2]
            if mod in _SAFE_PYTHON_M:
                if mod == "pytest" and not _pytest_args_ok(parts[3:]):
                    return TerminalDecision(False, "معاملات pytest غير مسموحة")
                if mod in ("py_compile", "compileall") and not _path_args_ok(parts[3:]):
                    return TerminalDecision(False, "مسارات غير آمنة")
                if mod == "unittest" and not _unittest_args_ok(parts[3:]):
                    return TerminalDecision(False, "معاملات unittest غير مسموحة")
                return TerminalDecision(True, f"python -m {mod} مسموح", parts)
        # سكربت نسبي بسيط: python path/to/test_x.py
        if len(parts) == 2 and re.match(r"^[\w./-]+\.py$", parts[1]) and ".." not in parts[1]:
            if parts[1].startswith("tests/") or parts[1].startswith("./tests/"):
                return TerminalDecision(True, "تشغيل سكربت اختبار نسبي", parts)

    # pip/pip3 --version فقط
    if parts[0] in ("pip", "pip3") and parts[1:] in (("--version",),):
        return TerminalDecision(True, "إصدار pip", parts)

    # git مع قيود فرعية
    if parts[0] == "git":
        if len(parts) < 2:
            return TerminalDecision(False, "git بلا أمر فرعي")
        sub = parts[1]
        if sub not in _SAFE_GIT_SUB:
            return TerminalDecision(False, f"git {sub} يتطلب موافقة")
        if not _git_args_ok(sub, parts[2:]):
            return TerminalDecision(False, "معاملات git غير مسموحة")
        return TerminalDecision(True, f"git {sub} قراءة فقط", parts)

    # ls / head / tail / wc — قراءة فقط على مسارات بسيطة
    if parts[0] in ("ls", "head", "tail", "wc"):
        if _path_args_ok(parts[1:]):
            return TerminalDecision(True, f"{parts[0]} قراءة مسموحة", parts)
        return TerminalDecision(False, "مسارات غير آمنة")

    return TerminalDecision(False, "الأمر خارج القائمة المسموحة؛ يلزم طلب موافقة", parts)


# ── مسارات آمنة: لا خروج من المشروع ولا ملفات أسرار ────────────────────────
_SECRET_SEGMENT_PATTERNS = (
    ".env*", ".git", ".gitmodules", ".gitconfig", ".git-credentials", ".streamlit",
    ".ssh", ".aws", ".netrc", ".npmrc", ".pypirc", "*.pem", "*.key", "*.p12", "*.pfx",
    "id_rsa*", "id_ed25519*", "*secret*", "*credential*", "*token*", "*password*", "*.kdbx",
)


def _path_is_safe(arg: str) -> bool:
    """مسار نسبي داخل المشروع فقط، بلا '..' ولا مسار مطلق ولا '~'، وبلا أي
    جزء يطابق ملفات الأسرار (.env، .git/config فيه توكن الـremote، .streamlit...)."""
    if not arg or "\x00" in arg or "\\" in arg:
        return False
    if arg.startswith(("/", "~")):
        return False
    segs = [s for s in posixpath.normpath(arg).split("/") if s not in ("", ".")]
    if ".." in segs:
        return False
    for seg in segs:
        low = seg.lower()
        if any(fnmatch.fnmatch(low, pat) for pat in _SECRET_SEGMENT_PATTERNS):
            return False
    return True


def _pytest_args_ok(args: tuple[str, ...]) -> bool:
    """يسمح بمسارات tests/ وملفات test_*.py وعلامات pytest الشائعة فقط."""
    for a in args:
        if a.startswith("-"):
            if a in ("-q", "-v", "-vv", "--tb=line", "--tb=short", "--tb=no",
                     "-x", "--maxfail=1", "--maxfail=3", "-k", "--collect-only"):
                continue
            if a.startswith("--tb=") or a.startswith("--maxfail="):
                continue
            return False
        node = a.split("::", 1)[0]
        is_tests_path = node == "tests" or node.startswith(("tests/", "./tests/"))
        is_test_file = node.endswith(".py") and posixpath.basename(node).startswith("test_")
        if is_tests_path or is_test_file:
            if _path_is_safe(node):
                continue
            return False
        # كلمة بسيطة (قيمة -k)
        if re.match(r"^[\w\[\]-]+$", a):
            continue
        return False
    return True


def _unittest_args_ok(args: tuple[str, ...]) -> bool:
    """unittest بلا discover/-s/-t/-p (كانت تسمح بتشغيل اكتشاف في أي مسار):
    أسماء وحدات tests.* أو ملفات test_*.py آمنة فقط."""
    for a in args:
        if a in ("-v", "-q"):
            continue
        if re.match(r"^tests(\.\w+){0,3}$", a):
            continue
        if a.endswith(".py") and posixpath.basename(a).startswith("test_") and _path_is_safe(a):
            continue
        return False
    return True


_GIT_REF_RE = re.compile(r"^(HEAD(~\d{1,2})?|[0-9a-f]{7,40})$")


def _git_pos_ok(a: str) -> bool:
    """معامل git موضعي: مرجع آمن أو مسار آمن. ':' ممنوعة (HEAD:.env يقرأ blob)."""
    if ":" in a or ".." in a:
        return False
    return bool(_GIT_REF_RE.match(a)) or _path_is_safe(a)


def _git_args_ok(sub: str, args: tuple[str, ...]) -> bool:
    if sub == "log":
        for a in args:
            if a in ("--oneline", "--stat", "--graph", "-5", "-10", "-20", "-n"):
                continue
            if a.startswith("-n") and a[2:].isdigit():
                continue
            if a.isdigit() and int(a) <= 50:
                continue
            if a.startswith("-"):
                return False
            if not _git_pos_ok(a):
                return False
        return True
    if sub == "show":
        for a in args:
            if a in ("--stat", "--name-only", "--oneline"):
                continue
            if a.startswith("-") or not _git_pos_ok(a):
                return False
        return len(args) <= 3
    if sub == "diff":
        for a in args:
            if a in ("--check", "--stat", "--name-only", "--cached", "--shortstat", "-U1", "-U3"):
                continue
            if a.startswith("-") or not _git_pos_ok(a):
                return False
        return True
    if sub == "branch":
        # عرض فقط — بلا أسماء ولا -D/-d/-m/-f/--set-upstream-to (كانت تكتب/تحذف فروعاً)
        return all(a in ("-v", "-vv", "-a", "-r", "--all", "--remotes", "--show-current") for a in args)
    flags = {
        "status": ("-s", "-sb", "-b", "--short", "--branch", "--porcelain", "-uno"),
        "ls-files": ("--cached", "--others", "--modified", "-c", "-o", "-m", "--exclude-standard"),
        "rev-parse": ("--abbrev-ref", "--short", "--show-toplevel", "--is-inside-work-tree"),
    }.get(sub)
    if flags is None:
        return False
    for a in args:
        if a in flags:
            continue
        if a.startswith("-") or not _git_pos_ok(a):
            return False
    return True


def _path_args_ok(args: tuple[str, ...]) -> bool:
    for a in args:
        if a.startswith("-"):
            # أعلام بسيطة فقط
            if a in ("-l", "-la", "-1", "-n", "--lines") or re.match(r"^-\d+$", a):
                continue
            return False
        if not _path_is_safe(a):
            return False
    return True


_PATH_CHECKED_HEADS = {"ls", "head", "tail", "wc", "python", "python3", "pytest", "py.test"}


def _paths_stay_inside(cwd: str, parts: Sequence[str]) -> bool:
    """بعد حلّ الروابط الرمزية: كل معامل غير-علَم يجب أن يبقى داخل cwd وألا يصير
    ملف أسرار (رابط رمزي مثل notes.txt → .env). فحص realpath يحتاج cwd فيجري هنا."""
    if not parts or parts[0] not in _PATH_CHECKED_HEADS:
        return True
    root = os.path.realpath(cwd)
    for a in parts[1:]:
        if a.startswith("-") or "=" in a:
            continue
        real = os.path.realpath(os.path.join(root, a.split("::", 1)[0]))
        if real != root and not real.startswith(root + os.sep):
            return False
        if real != root and not _path_is_safe(os.path.relpath(real, root)):
            return False
    return True


def run_auto(command: str | Sequence[str], *, cwd: str, timeout: int = 60) -> str:
    decision = decide(command)
    if not decision.allowed:
        return f"مرفوض تلقائياً: {decision.reason}"
    if not _paths_stay_inside(cwd, decision.command):
        return "مرفوض تلقائياً: مسار يخرج من المشروع أو يشير (رابط رمزي) إلى ملف أسرار"
    try:
        result = subprocess.run(
            list(decision.command),
            cwd=cwd,
            shell=False,
            capture_output=True,
            text=True,
            timeout=max(1, min(timeout, 120)),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"فشل التشغيل الآمن: {type(exc).__name__}"
    return f"exit={result.returncode}\n{(result.stdout + result.stderr).strip()[-12000:]}"


def explain_policy() -> str:
    return (
        "التشغيل التلقائي: فحوص git (status/diff/log/show/branch/ls-files)، "
        "pytest و python -m py_compile/compileall/unittest، "
        "و ls/head/tail/wc على مسارات نسبية. "
        "الكتابة والحذف والشبكة وGit commit/push/pull وpip install تتطلب موافقة بشرية."
    )


def list_allowed_examples() -> list[str]:
    """أمثلة أوامر مسموحة — للعرض في terminal_policy."""
    return [
        "git status",
        "git status -sb",
        "git diff --stat",
        "git log --oneline -10",
        "git branch --show-current",
        "git rev-parse HEAD",
        "pytest -q tests/",
        "python -m py_compile core/mesh_bundle.py",
        "python3 --version",
        "ls tests",
        "wc -l tests/test_node_hands.py",
        "head -20 README.md",
    ]
