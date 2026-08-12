"""
Git LFS Helper — قدرات كاملة لـ Git Large File Storage من داخل وكلاء NSM.
=======================================================================
يدعم: install / status / pull / push / track / untrack / ls-files /
       migrate / pointer-check / prune / fetch / env / version
ويُستخدم من nsm_chat و code_agent و nsm_agent_core و model_training_agent.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
_TOOLS_BIN = ROOT / ".tools" / "bin"
_DEFAULT_TRACK_PATTERNS = [
    "*.npy", "*.npz", "*.npz2", "*.pt", "*.pth", "*.bin", "*.onnx",
    "*.h5", "*.hdf5", "*.pb", "*.ckpt", "*.safetensors", "*.gguf", "*.ggml",
    "*.pkl", "*.pickle", "*.joblib", "*.dill",
    "*.zip", "*.tar", "*.tar.gz", "*.tgz", "*.gz", "*.bz2", "*.xz", "*.7z", "*.rar",
    "*.jpg", "*.jpeg", "*.png", "*.gif", "*.webp", "*.bmp", "*.ico", "*.tif", "*.tiff",
    "*.ttf", "*.otf", "*.woff", "*.woff2",
    "*.mp4", "*.webm", "*.mov", "*.avi", "*.mkv",
    "*.mp3", "*.wav", "*.flac", "*.ogg",
    "*.pdf", "*.psd", "*.ai",
]


def _env_path() -> dict:
    env = os.environ.copy()
    if _TOOLS_BIN.is_dir():
        env["PATH"] = str(_TOOLS_BIN) + os.pathsep + env.get("PATH", "")
    return env


def _run(cmd: List[str], timeout: int = 180) -> Tuple[int, str]:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(ROOT), env=_env_path(),
        )
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        return r.returncode, out
    except subprocess.TimeoutExpired:
        return 1, "timeout"
    except FileNotFoundError as e:
        return 1, f"الأمر غير موجود: {e}"
    except Exception as e:
        return 1, str(e)


def has_git_lfs() -> bool:
    code, _ = _run(["git", "lfs", "version"], timeout=15)
    return code == 0


def lfs_version() -> str:
    code, out = _run(["git", "lfs", "version"], timeout=15)
    return out if code == 0 else "غير مثبت"


def ensure_lfs_in_path() -> bool:
    if has_git_lfs():
        return True
    local = _TOOLS_BIN / "git-lfs"
    return local.is_file() and os.access(local, os.X_OK)


def install_lfs() -> Dict[str, Any]:
    if has_git_lfs():
        code, out = _run(["git", "lfs", "install"], timeout=30)
        return {"ok": code == 0, "already": True, "version": lfs_version(), "msg": out or "git lfs install OK"}

    setup = ROOT / "scripts" / "setup_git_lfs.sh"
    if setup.is_file():
        code, out = _run(["bash", str(setup)], timeout=300)
        return {"ok": code == 0 and has_git_lfs(), "already": False, "version": lfs_version(), "msg": out[:1500], "via": "scripts/setup_git_lfs.sh"}

    if os.uname().sysname == "Linux":
        ver = "3.5.1"
        tmp = Path("/tmp") / f"nsm-git-lfs-{os.getpid()}"
        try:
            tmp.mkdir(parents=True, exist_ok=True)
            tgz = tmp / "git-lfs.tgz"
            url = f"https://github.com/git-lfs/git-lfs/releases/download/v{ver}/git-lfs-linux-amd64-v{ver}.tar.gz"
            code, out = _run(["curl", "-fsSL", url, "-o", str(tgz)], timeout=120)
            if code != 0:
                return {"ok": False, "msg": f"فشل التنزيل: {out}"}
            code, out = _run(["tar", "-xzf", str(tgz), "-C", str(tmp)], timeout=60)
            if code != 0:
                return {"ok": False, "msg": f"فشل فك الأرشيف: {out}"}
            _TOOLS_BIN.mkdir(parents=True, exist_ok=True)
            src = next(tmp.glob("git-lfs-*/git-lfs"), None)
            if not src:
                return {"ok": False, "msg": "لم يُعثر على ثنائي git-lfs داخل الأرشيف"}
            dest = _TOOLS_BIN / "git-lfs"
            shutil.copy2(src, dest)
            dest.chmod(0o755)
            code, out = _run(["git", "lfs", "install"], timeout=30)
            return {"ok": code == 0 and has_git_lfs(), "already": False, "version": lfs_version(), "msg": out or f"ثُبّت في {dest}", "via": "direct-download"}
        except Exception as e:
            return {"ok": False, "msg": str(e)}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return {"ok": False, "msg": "ثبّت git-lfs يدوياً من https://git-lfs.com ثم أعد المحاولة"}


def lfs_status(detailed: bool = True) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "git_lfs_installed": has_git_lfs(),
        "version": lfs_version() if has_git_lfs() else None,
        "tools_bin": str(_TOOLS_BIN) if _TOOLS_BIN.is_dir() else None,
    }
    if not has_git_lfs():
        result["ready"] = False
        result["hint"] = "شغّل: bash scripts/setup_git_lfs.sh  أو أمر «تفعيل lfs»"
        return result

    code, out = _run(["git", "lfs", "ls-files"], timeout=60)
    files = []
    if code == 0 and out:
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                files.append({"oid": parts[0], "flag": parts[1], "path": " ".join(parts[2:])})
            elif line.strip():
                files.append({"raw": line.strip()})
    result["lfs_tracked_count"] = len(files)
    result["lfs_files_sample"] = files[:25]

    check_paths = [
        ROOT / "knowledge" / "cognitive_graph.json",
        ROOT / "knowledge" / "cognitive_graph_general_ar.json",
    ]
    rows = []
    for f in check_paths:
        row: Dict[str, Any] = {"path": str(f.relative_to(ROOT)), "exists": f.is_file()}
        if f.is_file():
            row["size"] = f.stat().st_size
            try:
                head = f.read_text(encoding="utf-8", errors="ignore")[:120]
                row["lfs_pointer"] = "git-lfs.github.com" in head or head.startswith("version https://git-lfs")
            except Exception:
                row["lfs_pointer"] = None
        rows.append(row)
    result["knowledge_check"] = rows
    result["ready"] = all(
        r.get("exists") and not r.get("lfs_pointer") and r.get("size", 0) > 10_000 for r in rows
    ) if rows else False

    if detailed:
        code2, env_out = _run(["git", "lfs", "env"], timeout=30)
        result["env_snippet"] = (env_out or "")[:800] if code2 == 0 else None
    return result


def lfs_pull(include: Optional[str] = None) -> Dict[str, Any]:
    if not ensure_lfs_in_path():
        return {"ok": False, "msg": "git-lfs غير متوفر — شغّل تفعيل lfs أولاً"}
    cmd = ["git", "lfs", "pull"]
    if include:
        cmd.extend(["--include", include])
    code, out = _run(cmd, timeout=600)
    return {"ok": code == 0, "msg": out[:2000] or ("OK" if code == 0 else "فشل")}


def lfs_fetch(recent: bool = False) -> Dict[str, Any]:
    if not ensure_lfs_in_path():
        return {"ok": False, "msg": "git-lfs غير متوفر"}
    cmd = ["git", "lfs", "fetch"]
    if recent:
        cmd.append("--recent")
    code, out = _run(cmd, timeout=600)
    return {"ok": code == 0, "msg": out[:2000] or ("OK" if code == 0 else "فشل")}


def lfs_push(remote: str = "origin", branch: str = "main") -> Dict[str, Any]:
    if not ensure_lfs_in_path():
        return {"ok": False, "msg": "git-lfs غير متوفر"}
    code, out = _run(["git", "lfs", "push", remote, branch], timeout=600)
    return {"ok": code == 0, "msg": out[:2000] or ("OK" if code == 0 else "فشل")}


def lfs_track(patterns: Optional[List[str]] = None) -> Dict[str, Any]:
    if not ensure_lfs_in_path():
        return {"ok": False, "msg": "git-lfs غير متوفر"}
    pats = patterns or _DEFAULT_TRACK_PATTERNS
    results = []
    for p in pats:
        code, out = _run(["git", "lfs", "track", p], timeout=30)
        results.append({"pattern": p, "ok": code == 0, "out": out[:200]})
    _run(["git", "add", ".gitattributes"], timeout=15)
    ok_count = sum(1 for r in results if r["ok"])
    return {"ok": ok_count > 0, "tracked": ok_count, "total": len(pats), "details": results[:15], "msg": f"تم تتبع {ok_count}/{len(pats)} نمط"}


def lfs_untrack(patterns: List[str]) -> Dict[str, Any]:
    if not ensure_lfs_in_path():
        return {"ok": False, "msg": "git-lfs غير متوفر"}
    results = []
    for p in patterns:
        code, out = _run(["git", "lfs", "untrack", p], timeout=30)
        results.append({"pattern": p, "ok": code == 0, "out": out[:200]})
    _run(["git", "add", ".gitattributes"], timeout=15)
    return {"ok": True, "details": results}


def lfs_ls_files(limit: int = 50) -> Dict[str, Any]:
    if not ensure_lfs_in_path():
        return {"ok": False, "msg": "git-lfs غير متوفر", "files": []}
    code, out = _run(["git", "lfs", "ls-files"], timeout=60)
    files = [ln.strip() for ln in (out or "").splitlines() if ln.strip()][:limit]
    return {"ok": code == 0, "count": len(files), "files": files}


def lfs_migrate_import(
    include: str = "*.pkl,*.npy,*.pt,*.pth,*.bin,*.h5,*.onnx,*.safetensors",
    everything: bool = False,
) -> Dict[str, Any]:
    if not ensure_lfs_in_path():
        return {"ok": False, "msg": "git-lfs غير متوفر"}
    cmd = ["git", "lfs", "migrate", "import", f"--include={include}"]
    if everything:
        cmd.append("--everything")
    code, out = _run(cmd, timeout=900)
    return {"ok": code == 0, "msg": out[:2500] or ("OK" if code == 0 else "فشل"), "warning": "migrate يعيد كتابة التاريخ — تأكد من التنسيق مع الفريق قبل force-push"}


def lfs_prune() -> Dict[str, Any]:
    if not ensure_lfs_in_path():
        return {"ok": False, "msg": "git-lfs غير متوفر"}
    code, out = _run(["git", "lfs", "prune"], timeout=120)
    return {"ok": code == 0, "msg": out[:1500] or ("OK" if code == 0 else "فشل")}


def is_lfs_pointer(path: str | Path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    try:
        if p.stat().st_size > 500:
            return False
        head = p.read_text(encoding="utf-8", errors="ignore")[:150]
        return "git-lfs.github.com" in head or head.startswith("version https://git-lfs")
    except Exception:
        return False


def handle_lfs_command(user_input: str) -> Optional[str]:
    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(
        r"(git\s*lfs|اسحب\s*lfs|تفعيل\s*lfs|lfs\s*pull|اعداد\s*lfs|إعداد\s*lfs|"
        r"lfs\s*status|حالة\s*lfs|تتبع\s*lfs|lfs\s*track|lfs\s*push|"
        r"رفع\s*lfs|هجرة\s*lfs|lfs\s*migrate|lfs\s*ls|ملفات\s*lfs|"
        r"تثبيت\s*lfs|install\s*lfs|prune\s*lfs)",
        text, re.I,
    ):
        return None

    low = text.lower()

    if re.search(r"(تفعيل|تثبيت|اعداد|إعداد|install|setup)\s*lfs|lfs\s*(install|setup)", low, re.I):
        res = install_lfs()
        st = lfs_status(detailed=False)
        return ("## ⚙️ تفعيل Git LFS\n\n```json\n" + json.dumps({"install": res, "status": st}, ensure_ascii=False, indent=2)
                + "\n```\n\n" + ("✅ جاهز. يمكنك الآن: `اسحب lfs`" if res.get("ok") else "❌ فشل التفعيل — راجع الرسالة أعلاه."))

    if re.search(r"(اسحب|pull)\s*lfs|lfs\s*pull", low, re.I):
        res = lfs_pull()
        st = lfs_status(detailed=False)
        return "## 📥 Git LFS Pull\n\n```json\n" + json.dumps({"pull": res, "status": st}, ensure_ascii=False, indent=2) + "\n```"

    if re.search(r"(رفع|push)\s*lfs|lfs\s*push", low, re.I):
        res = lfs_push()
        return "## 📤 Git LFS Push\n\n```json\n" + json.dumps(res, ensure_ascii=False, indent=2) + "\n```"

    if re.search(r"(تتبع|track)\s*lfs|lfs\s*track", low, re.I):
        extra = re.sub(r".*?(تتبع\s*lfs|lfs\s*track)\s*", "", text, flags=re.I).strip()
        patterns = [p for p in re.split(r"[\s,]+", extra) if p] or None
        res = lfs_track(patterns)
        return "## 📌 Git LFS Track\n\n```json\n" + json.dumps(res, ensure_ascii=False, indent=2) + "\n```"

    if re.search(r"(ملفات\s*lfs|lfs\s*ls|ls-files)", low, re.I):
        res = lfs_ls_files()
        return "## 📋 ملفات Git LFS\n\n```json\n" + json.dumps(res, ensure_ascii=False, indent=2) + "\n```"

    if re.search(r"(هجرة|migrate)\s*lfs|lfs\s*migrate", low, re.I):
        res = lfs_migrate_import(everything=False)
        return "## 🚚 Git LFS Migrate\n\n```json\n" + json.dumps(res, ensure_ascii=False, indent=2) + "\n```\n\n⚠️ يعيد كتابة التاريخ — استخدم بحذر."

    if re.search(r"prune\s*lfs|lfs\s*prune", low, re.I):
        res = lfs_prune()
        return "## 🧹 Git LFS Prune\n\n```json\n" + json.dumps(res, ensure_ascii=False, indent=2) + "\n```"

    st = lfs_status(detailed=True)
    return ("## 📦 حالة Git LFS\n\n```json\n" + json.dumps(st, ensure_ascii=False, indent=2) + "\n```\n\n"
            "أوامر متاحة:\n- `تفعيل lfs` / `تثبيت lfs`\n- `اسحب lfs` / `lfs pull`\n- `رفع lfs` / `lfs push`\n"
            "- `تتبع lfs [أنماط]`\n- `ملفات lfs`\n- `هجرة lfs`\n- `حالة lfs`\n\n"
            "دليل: `docs/GIT_LFS.md` — سكربت: `bash scripts/setup_git_lfs.sh`")
