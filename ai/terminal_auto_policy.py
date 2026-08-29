"""Safe automatic terminal policy for NSM agents."""
from __future__ import annotations
import shlex
import subprocess
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class TerminalDecision:
    allowed: bool
    reason: str
    command: tuple[str, ...] = ()

_ALLOWED = {("git", "status"), ("git", "diff", "--check"), ("git", "diff", "--stat")}
_BLOCKED = {"rm", "rmdir", "del", "sudo", "chmod", "chown", "curl", "wget", "ssh", "scp", "export", "printenv", "git push", "git reset", "git clean", "git checkout", "git commit"}

def decide(command: str | Sequence[str]) -> TerminalDecision:
    try: parts = tuple(command) if not isinstance(command, str) else tuple(shlex.split(command))
    except ValueError: return TerminalDecision(False, "تعذر تحليل الأمر")
    if not parts: return TerminalDecision(False, "الأمر فارغ")
    lowered = " ".join(parts).lower()
    if any(x in lowered for x in _BLOCKED): return TerminalDecision(False, "الأمر يتطلب موافقة بشرية أو قد يغيّر النظام")
    if any(x in parts for x in (";", "&&", "||", "|", ">", ">>", "<")): return TerminalDecision(False, "shell operators غير مسموحة")
    if parts[0] == "pytest": return TerminalDecision(True, "اختبار مسموح تلقائياً", parts)
    if parts[:2] == ("python", "-m") and len(parts) >= 3 and parts[2] in {"compileall", "py_compile"}:
        return TerminalDecision(True, "فحص Python مسموح تلقائياً", parts)
    if parts in _ALLOWED: return TerminalDecision(True, "فحص قراءة فقط مسموح تلقائياً", parts)
    return TerminalDecision(False, "الأمر خارج القائمة المسموحة؛ يلزم طلب موافقة", parts)

def run_auto(command: str | Sequence[str], *, cwd: str, timeout: int = 60) -> str:
    decision = decide(command)
    if not decision.allowed: return f"مرفوض تلقائياً: {decision.reason}"
    try:
        result = subprocess.run(decision.command, cwd=cwd, shell=False, capture_output=True, text=True, timeout=max(1, min(timeout, 120)), check=False)
    except (OSError, subprocess.TimeoutExpired) as exc: return f"فشل التشغيل الآمن: {type(exc).__name__}"
    return f"exit={result.returncode}\n{(result.stdout + result.stderr).strip()[-12000:]}"

def explain_policy() -> str:
    return "التشغيل التلقائي يقتصر على فحوص status وdiff وcompileall وpy_compile وpytest. الكتابة والحذف والشبكة وGit commit/push تتطلب موافقة بشرية صريحة."
