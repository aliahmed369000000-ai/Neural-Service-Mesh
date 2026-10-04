"""Safe automatic terminal policy for NSM agents / node hands.

قواعد ثابتة:
- بلا shell=True، بلا مشغّلات (; && || | > >> <)
- بلا كتابة/حذف/شبكة/صلاحيات
- قائمة مسموحة تتوسع بحذر لفحوص وقراءة فقط
"""
from __future__ import annotations

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


def _pytest_args_ok(args: tuple[str, ...]) -> bool:
    """يسمح بمسارات tests/ وعلامات pytest الشائعة فقط."""
    for a in args:
        if a.startswith("-"):
            if a in ("-q", "-v", "-vv", "--tb=line", "--tb=short", "--tb=no",
                     "-x", "--maxfail=1", "--maxfail=3", "-k", "--collect-only"):
                continue
            if a.startswith("--tb=") or a.startswith("--maxfail="):
                continue
            return False
        # مسار
        if ".." in a or a.startswith("/") or a.startswith("~"):
            return False
        if not (a.startswith("tests/") or a.startswith("./tests/") or a.endswith(".py") or a == "tests"):
            # يسمح بتعبير -k expression ككلمة تالية — إن كانت بعد -k تُقبل في الحلقة السابقة
            if re.match(r"^[\w\[\]-]+$", a):
                continue
            return False
    return True


def _git_args_ok(sub: str, args: tuple[str, ...]) -> bool:
    if sub == "log":
        for a in args:
            if a.startswith("-") and a in ("--oneline", "--stat", "--graph", "-5", "-10", "-20", "-n"):
                continue
            if a.startswith("-n") and a[2:].isdigit():
                continue
            if a.isdigit() and int(a) <= 50:
                continue
            if a in ("HEAD", "HEAD~1", "HEAD~5"):
                continue
            if a.startswith("--"):
                return False
            if ".." in a or a.startswith("-"):
                return False
        return True
    if sub == "show":
        for a in args:
            if a in ("--stat", "--name-only", "--oneline", "HEAD", "HEAD~1"):
                continue
            if re.match(r"^[0-9a-f]{7,40}$", a):
                continue
            if a.startswith("-"):
                return False
        return len(args) <= 3
    if sub == "diff":
        for a in args:
            if a in ("--check", "--stat", "--name-only", "HEAD", "--cached"):
                continue
            if a.startswith("-") and a in ("-U1", "-U3", "--shortstat"):
                continue
            if ".." in a or (a.startswith("-") and a not in ("--check", "--stat", "--name-only", "--cached", "--shortstat")):
                return False
        return True
    if sub in ("status", "branch", "ls-files", "rev-parse"):
        for a in args:
            if a.startswith("-") and len(a) < 20:
                continue
            if a in ("HEAD", "--abbrev-ref", "--short", "--show-current"):
                continue
            if ".." in a:
                return False
        return True
    return False


def _path_args_ok(args: tuple[str, ...]) -> bool:
    for a in args:
        if a.startswith("-"):
            # أعلام بسيطة فقط
            if a in ("-l", "-la", "-1", "-n", "-20", "-5", "-10", "-l", "--lines"):
                continue
            if re.match(r"^-\d+$", a):
                continue
            return False
        if ".." in a or a.startswith("~") or (a.startswith("/") and not a.startswith("/home/workdir")):
            # نمنع مسارات مطلقة خارج بيئة العمل الشائعة
            if a.startswith("/"):
                return False
        if any(ch in a for ch in (";", "|", "&", ">", "<", "`", "$")):
            return False
    return True


def run_auto(command: str | Sequence[str], *, cwd: str, timeout: int = 60) -> str:
    decision = decide(command)
    if not decision.allowed:
        return f"مرفوض تلقائياً: {decision.reason}"
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
