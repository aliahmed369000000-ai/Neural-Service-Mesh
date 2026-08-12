"""
NSM Agent Tools — أدوات تشغيلية موحدة للبوت والوكلاء (أسلوب عمل قريب من المساعد الكامل).
================================================================================
- بحث في كود المشروع (grep)
- معلومات Git (status / log / diff)
- py_compile / فحص بناء الجملة
- جلب محتوى صفحة ويب
- البحث عن ملفات
- معلومات النظام
- أوامر bash آمنة محدودة
تُستخدم من code_agent و nsm_chat و nsm_agent_core.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
_MAX_OUT = 12000
_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "checkpoints", "data", ".tools", ".streamlit",
}


def _safe_rel(path: str) -> Optional[Path]:
    if not path or not str(path).strip():
        return None
    try:
        cand = (ROOT / str(path).strip()).resolve()
        cand.relative_to(ROOT.resolve())
        return cand
    except (ValueError, OSError):
        return None


def _run(cmd: List[str], timeout: int = 60, cwd: Optional[str] = None) -> Tuple[int, str]:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=cwd or str(ROOT),
        )
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        if len(out) > _MAX_OUT:
            out = out[:_MAX_OUT] + "\n... [مقطوع]"
        return r.returncode, out
    except subprocess.TimeoutExpired:
        return 1, "timeout"
    except Exception as e:
        return 1, str(e)


def search_code(
    pattern: str,
    path: str = ".",
    glob: str = "*.py",
    max_matches: int = 40,
) -> Dict[str, Any]:
    """بحث نصي/regex في ملفات المشروع (مثل grep)."""
    if not pattern or not pattern.strip():
        return {"ok": False, "msg": "مطلوب نمط البحث", "matches": []}
    base = _safe_rel(path) or ROOT
    if not base.exists():
        return {"ok": False, "msg": f"المسار غير موجود: {path}", "matches": []}

    matches: List[Dict[str, Any]] = []
    try:
        rx = re.compile(pattern, re.I)
    except re.error as e:
        return {"ok": False, "msg": f"regex غير صالح: {e}", "matches": []}

    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            if glob.startswith("*.") and not fname.endswith(glob[1:]):
                continue
            if not glob.startswith("*.") and glob != "*" and fname != glob:
                continue
            fp = Path(dirpath) / fname
            try:
                if fp.stat().st_size > 2_000_000:
                    continue
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = str(fp.relative_to(ROOT))
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    matches.append({
                        "path": rel,
                        "line": i,
                        "text": line.strip()[:200],
                    })
                    if len(matches) >= max_matches:
                        return {
                            "ok": True,
                            "pattern": pattern,
                            "count": len(matches),
                            "truncated": True,
                            "matches": matches,
                        }
    return {"ok": True, "pattern": pattern, "count": len(matches), "truncated": False, "matches": matches}


def find_files(name_glob: str = "*.py", path: str = ".", limit: int = 80) -> Dict[str, Any]:
    """البحث عن ملفات بالاسم/الامتداد."""
    base = _safe_rel(path) or ROOT
    found: List[str] = []
    name_glob = name_glob or "*.py"
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in filenames:
            if name_glob.startswith("*.") and fname.endswith(name_glob[1:]):
                found.append(str((Path(dirpath) / fname).relative_to(ROOT)))
            elif name_glob in fname or fname == name_glob:
                found.append(str((Path(dirpath) / fname).relative_to(ROOT)))
            if len(found) >= limit:
                return {"ok": True, "count": len(found), "truncated": True, "files": found}
    return {"ok": True, "count": len(found), "truncated": False, "files": found}


def git_info(what: str = "status") -> Dict[str, Any]:
    """معلومات Git: status | log | diff | branch | remote."""
    what = (what or "status").strip().lower()
    cmds = {
        "status": ["git", "status", "--short", "--branch"],
        "log": ["git", "log", "-15", "--oneline", "--decorate"],
        "diff": ["git", "diff", "--stat", "HEAD"],
        "branch": ["git", "branch", "-vv"],
        "remote": ["git", "remote", "-v"],
    }
    if what not in cmds:
        return {"ok": False, "msg": f"غير معروف: {what}. المدعوم: {', '.join(cmds)}"}
    code, out = _run(cmds[what], timeout=30)
    return {"ok": code == 0, "what": what, "output": out or "(فارغ)"}


def py_compile_check(path: str) -> Dict[str, Any]:
    """فحص بناء جملة Python عبر py_compile + ast."""
    f = _safe_rel(path)
    if f is None:
        return {"ok": False, "msg": "مسار غير مسموح"}
    if not f.exists():
        return {"ok": False, "msg": f"الملف غير موجود: {path}"}
    if not str(f).endswith(".py"):
        return {"ok": False, "msg": "الملف ليس .py"}
    try:
        src = f.read_text(encoding="utf-8")
        ast.parse(src)
    except SyntaxError as e:
        return {
            "ok": False,
            "path": path,
            "msg": f"SyntaxError: {e.msg}",
            "line": e.lineno,
            "offset": e.offset,
        }
    code, out = _run(["python3", "-m", "py_compile", str(f)], timeout=30)
    return {
        "ok": code == 0,
        "path": path,
        "msg": "سليم — py_compile + ast OK" if code == 0 else out,
    }


def fetch_url(url: str, max_chars: int = 8000) -> Dict[str, Any]:
    """جلب نص صفحة ويب (بدون JS) — مفيد للوثائق والـ API العامة."""
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return {"ok": False, "msg": "الرابط يجب أن يبدأ بـ http:// أو https://"}
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NSM-Agent/1.0 (+https://github.com/aliahmed369000000-ai/Neural-Service-Mesh)"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read(max_chars + 2000)
            ctype = resp.headers.get("Content-Type", "")
        text = raw.decode("utf-8", errors="replace")
        # تبسيط HTML خفيف
        text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", text)
        text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]
        return {
            "ok": True,
            "url": url,
            "content_type": ctype,
            "truncated": truncated,
            "text": text,
        }
    except urllib.error.HTTPError as e:
        return {"ok": False, "msg": f"HTTP {e.code}: {e.reason}", "url": url}
    except Exception as e:
        return {"ok": False, "msg": str(e), "url": url}


def system_info() -> Dict[str, Any]:
    """لمحة عن البيئة (بدون أسرار)."""
    import platform
    import sys
    return {
        "ok": True,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(ROOT),
        "pid": os.getpid(),
        "env_flags": {
            "GITHUB_TOKEN": bool(os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")),
            "GOOGLE_API_KEY": bool(os.environ.get("GOOGLE_API_KEY")),
            "GROQ_API_KEY": bool(os.environ.get("GROQ_API_KEY")),
        },
    }


_SAFE_CMD_PREFIXES = (
    "python ", "python3 ", "pytest", "pip show", "pip list",
    "git status", "git log", "git diff", "git branch", "git remote",
    "git lfs", "ls ", "ls\t", "find ", "wc ", "head ", "tail ",
    "cat ", "sed -n", "grep ", "rg ", "echo ", "uname", "date",
    "pwd", "whoami", "df -h", "du -sh",
)


def run_safe_cmd(cmd: str, timeout: int = 45) -> Dict[str, Any]:
    """تشغيل أوامر قراءة/فحص آمنة فقط (لا حذف ولا شبكة حرة ولا sudo)."""
    c = (cmd or "").strip()
    if not c:
        return {"ok": False, "msg": "أمر فارغ"}
    low = c.lower()
    forbidden = [
        "rm ", "rm\t", "sudo", "chmod", "chown", "mkfs", "dd ",
        ">", ">>", "|", ";", "&&", "$(", "`", "curl ", "wget ",
        "ssh ", "scp ", "nc ", "ncat", "python -c", "eval",
        "os.system", "shutdown", "reboot", "kill ", "pkill",
    ]
    # سماح محدود بـ python -m py_compile و pytest
    if low.startswith("python -m py_compile") or low.startswith("python3 -m py_compile"):
        pass
    elif low.startswith("python -m pytest") or low.startswith("python3 -m pytest"):
        pass
    else:
        for bad in forbidden:
            if bad in low:
                return {"ok": False, "msg": f"أمر مرفوض لأسباب أمان: يحتوي على «{bad.strip()}»"}
        if not any(low.startswith(p.strip()) or low == p.strip() for p in _SAFE_CMD_PREFIXES):
            return {
                "ok": False,
                "msg": "الأمر غير ضمن القائمة الآمنة. المسموح: python/pytest/git status|log|diff|lfs، ls، grep، head، …",
            }
    code, out = _run(["bash", "-lc", c], timeout=timeout)
    return {"ok": code == 0, "cmd": c, "exit_code": code, "output": out or "(لا مخرجات)"}


def format_tool_result(title: str, data: Dict[str, Any]) -> str:
    return f"## {title}\n\n```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```"


def handle_tool_command(user_input: str) -> Optional[str]:
    """أوامر محادثة عربية/إنجليزية للأدوات الجديدة. None إن لم يُطابق."""
    t = (user_input or "").strip()
    if not t:
        return None
    low = t.lower()

    # بحث في الكود
    m = re.match(r"^(ابحث\s*في\s*الكود|grep|search\s*code)\s+(.+)$", t, re.I)
    if m:
        res = search_code(m.group(2).strip())
        return format_tool_result("🔎 بحث في الكود", res)

    # إيجاد ملفات
    m = re.match(r"^(أوجد|اوجد|find\s*files?|ملفات)\s+(.+)$", t, re.I)
    if m:
        res = find_files(m.group(2).strip())
        return format_tool_result("📁 إيجاد ملفات", res)

    # Git
    if re.match(r"^(حالة\s*git|git\s*status)$", t, re.I):
        return format_tool_result("🌿 Git Status", git_info("status"))
    if re.match(r"^(سجل\s*git|git\s*log)$", t, re.I):
        return format_tool_result("📜 Git Log", git_info("log"))
    if re.match(r"^(فرق\s*git|git\s*diff)$", t, re.I):
        return format_tool_result("📝 Git Diff", git_info("diff"))
    if re.match(r"^(فروع\s*git|git\s*branch)$", t, re.I):
        return format_tool_result("🌳 Git Branch", git_info("branch"))

    # py_compile
    m = re.match(r"^(تحقق|compile|py_compile|فحص\s*بناء)\s+(.+)$", t, re.I)
    if m:
        return format_tool_result("🧪 py_compile", py_compile_check(m.group(2).strip()))

    # جلب رابط
    m = re.match(r"^(افتح\s*رابط|جلب|fetch\s*url|open\s*url)\s+(\S+)$", t, re.I)
    if m:
        return format_tool_result("🌐 جلب صفحة", fetch_url(m.group(2).strip()))

    # معلومات النظام
    if re.match(r"^(معلومات\s*النظام|system\s*info|بيئة)$", t, re.I):
        return format_tool_result("🖥️ معلومات النظام", system_info())

    # أمر آمن
    m = re.match(r"^(نفّذ|نفذ|run\s*safe|أمر)\s+(.+)$", t, re.I)
    if m:
        return format_tool_result("⚙️ أمر آمن", run_safe_cmd(m.group(2).strip()))

    # مساعدة الأدوات
    if re.match(r"^(أدوات|tools|مساعدة\s*الأدوات|help\s*tools)$", t, re.I):
        return (
            "## 🧰 أدوات الوكيل المتاحة\n\n"
            "| أمر المحادثة | الوظيفة |\n|---|---|\n"
            "| `ابحث في الكود <نمط>` | grep داخل المشروع |\n"
            "| `أوجد *.py` | إيجاد ملفات |\n"
            "| `حالة git` / `سجل git` / `فرق git` | معلومات Git |\n"
            "| `تحقق path.py` | py_compile + ast |\n"
            "| `افتح رابط https://...` | جلب نص صفحة |\n"
            "| `معلومات النظام` | بيئة التشغيل |\n"
            "| `نفّذ <أمر آمن>` | bash محدود |\n"
            "| `حالة lfs` / `اسحب lfs` | Git LFS |\n"
            "| `افحص` / `عدل` / `أنشئ` / `ارفع` | إدارة ملفات + push |\n"
            "| `ابحث <نص>` | بحث ويب |\n"
        )

    return None
