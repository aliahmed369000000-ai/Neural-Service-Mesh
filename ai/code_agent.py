"""
NSM Code Agent — ai/code_agent.py
===================================
أدوات تحكم كاملة في المشروع من المحادثة:
  افحص / عدل / أنشئ / ارفع / قائمة / اقترح / صحح / ملخص / ابحث
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path
from typing import List, Dict

from ai.web_search_tool import web_search  # 🆕 أداة بحث حقيقية مشتركة (بدون مفتاح API)

ROOT = Path(__file__).parent.parent
_MAX_READ = 5000  # حد القراءة بالحروف


# ══════════════════════════════════════════════════════════════════
# 0. حماية المسارات — كل الدوال أدناه تستقبل path من نص دردشة خام
#    (عبر nsm_chat._handle_code_command)، وكانت تمرّره مباشرة لـ
#    ROOT / path بلا أي تحقق. مسار مثل "../../../etc/passwd" أو مسار
#    مطلق "/etc/passwd" يهرب فعلياً من مجلد المشروع تماماً — Python
#    نفسه يحلّ ".." عند resolve()/فتح الملف، و"/" المطلق يستبدل الجذر
#    بالكامل (Path.__truediv__ الموثَّق). كان هذا يعني أن أي مستخدم في
#    الدردشة يقدر يقرأ (افحص/ملخص/صحح) أو حتى يكتب (عدل/أنشئ) أي ملف
#    على القرص يملك المسار صلاحية الوصول له — مثل secrets.toml أو
#    مفاتيح API في متغيرات البيئة على Streamlit Cloud.
# ══════════════════════════════════════════════════════════════════
def _safe_path(path: str) -> "Path | None":
    """يحلّ مسار مستخدم نسبي إلى مسار مطلق آمن داخل ROOT حصراً فقط.
    يرفض (يعيد None) أي مسار مطلق أو أي ".." يهرب فعلياً خارج مجلد
    المشروع بعد الحلّ (resolve) — سواء وُجد الملف أصلاً (قراءة/تعديل)
    أو لم يوجد بعد (إنشاء)."""
    if not path or not path.strip():
        return None
    try:
        candidate = (ROOT / path.strip()).resolve()
        candidate.relative_to(ROOT.resolve())
    except (ValueError, OSError):
        return None
    return candidate


_UNSAFE_PATH_MSG = "❌ مسار غير مسموح به (خارج مجلد المشروع)."


# ══════════════════════════════════════════════════════════════════
# 1. قراءة ملف
# ══════════════════════════════════════════════════════════════════
def read_file(path: str) -> str:
    try:
        f = _safe_path(path)
        if f is None:
            return _UNSAFE_PATH_MSG
        if not f.exists():
            return f"❌ الملف غير موجود: {path}"
        size = f.stat().st_size
        content = f.read_text(encoding="utf-8", errors="replace")
        if len(content) > _MAX_READ:
            content = content[:_MAX_READ] + f"\n\n... [مقطوع — الحجم الكامل: {size} بايت]"
        return content
    except Exception as e:
        return f"❌ خطأ في القراءة: {e}"


# ══════════════════════════════════════════════════════════════════
# 2. قائمة الملفات
# ══════════════════════════════════════════════════════════════════
def list_files(folder: str = ".") -> str:
    try:
        base = _safe_path(folder)
        if base is None:
            return _UNSAFE_PATH_MSG
        if not base.exists():
            return f"❌ المجلد غير موجود: {folder}"
        skip = {"knowledge", "checkpoints", "data", "__pycache__", ".git"}
        lines = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in skip]
            rel = Path(dirpath).relative_to(ROOT)
            indent = "  " * (len(rel.parts) - (0 if folder == "." else 1))
            if str(rel) != ".":
                lines.append(f"{indent}📁 {Path(dirpath).name}/")
            for f in sorted(filenames):
                if f.endswith(".py"):
                    lines.append(f"{indent}  📄 {f}")
        return "\n".join(lines) if lines else "لا توجد ملفات .py"
    except Exception as e:
        return f"❌ خطأ: {e}"


# ══════════════════════════════════════════════════════════════════
# 3. تعديل ملف
# ══════════════════════════════════════════════════════════════════
def edit_file(path: str, old: str, new: str) -> str:
    try:
        f = _safe_path(path)
        if f is None:
            return _UNSAFE_PATH_MSG
        if not f.exists():
            return f"❌ الملف غير موجود: {path}"
        content = f.read_text(encoding="utf-8")
        if old not in content:
            return f"❌ النص القديم غير موجود في {path}"
        updated = content.replace(old, new, 1)
        f.write_text(updated, encoding="utf-8")
        return f"✅ تم التعديل في {path}"
    except Exception as e:
        return f"❌ خطأ في التعديل: {e}"


# ══════════════════════════════════════════════════════════════════
# 4. إنشاء ملف
# ══════════════════════════════════════════════════════════════════
def create_file(path: str, content: str) -> str:
    try:
        f = _safe_path(path)
        if f is None:
            return _UNSAFE_PATH_MSG
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
        return f"✅ تم إنشاء {path}"
    except Exception as e:
        return f"❌ خطأ في الإنشاء: {e}"


# ══════════════════════════════════════════════════════════════════
# 5. رفع لـ GitHub
# ══════════════════════════════════════════════════════════════════
def git_push(message: str = "NSM auto-commit") -> str:
    """رفع التغييرات لـ GitHub مع دعم Git LFS."""
    try:
        for cfg in [
            ["git", "-C", str(ROOT), "config", "--local", "user.email", "nsm-bot@users.noreply.github.com"],
            ["git", "-C", str(ROOT), "config", "--local", "user.name", "NSM Bot"],
        ]:
            subprocess.run(cfg, capture_output=True)

        r = subprocess.run(["git", "-C", str(ROOT), "add", "-A"], capture_output=True, text=True)
        if r.returncode != 0:
            return f"❌ git add: {(r.stderr or r.stdout).strip()}"

        r = subprocess.run(["git", "-C", str(ROOT), "commit", "-m", message], capture_output=True, text=True)
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        if r.returncode != 0 and "nothing to commit" not in out:
            return f"❌ git commit: {out}"

        auth_remote = None
        try:
            from ai.github_sync import get_authenticated_remote
            auth_remote = get_authenticated_remote()
        except Exception:
            pass

        push_cmd = ["git", "-C", str(ROOT), "push"]
        if auth_remote:
            push_cmd = ["git", "-C", str(ROOT), "push", auth_remote, "HEAD:main"]

        r = subprocess.run(push_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            err = ((r.stdout or "") + (r.stderr or "")).strip()
            if not auth_remote:
                return (f"❌ git push: {err}\n💡 أضف GITHUB_TOKEN و GITHUB_USER و GITHUB_REMOTE في Secrets.")
            return f"❌ git push: {err}"

        lfs_note = ""
        try:
            from ai.git_lfs_helper import has_git_lfs, lfs_push
            if has_git_lfs():
                lr = lfs_push()
                lfs_note = " + Git LFS" if lr.get("ok") else f" (تحذير LFS: {lr.get('msg', '')[:80]})"
        except Exception:
            pass
        return f"✅ رُفع بنجاح لـ GitHub{lfs_note}"
    except Exception as e:
        return f"❌ خطأ في git: {e}"


def git_lfs_status() -> str:
    try:
        from ai.git_lfs_helper import lfs_status
        import json
        return "## 📦 Git LFS\n```json\n" + json.dumps(lfs_status(detailed=True), ensure_ascii=False, indent=2) + "\n```"
    except Exception as e:
        return f"❌ LFS status: {e}"


def git_lfs_pull() -> str:
    try:
        from ai.git_lfs_helper import lfs_pull, lfs_status
        import json
        res = lfs_pull()
        st = lfs_status(detailed=False)
        return "## 📥 LFS Pull\n```json\n" + json.dumps({"pull": res, "status": st}, ensure_ascii=False, indent=2) + "\n```"
    except Exception as e:
        return f"❌ LFS pull: {e}"


# ══════════════════════════════════════════════════════════════════
# 6. اقتراحات تحسين المشروع
# ══════════════════════════════════════════════════════════════════
def project_suggestions(filter_type: str = "") -> str:
    try:
        skip = {"knowledge", "checkpoints", "data", "__pycache__", ".git"}
        all_files = []
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in skip]
            for fname in filenames:
                if fname.endswith(".py"):
                    all_files.append(Path(dirpath) / fname)

        unused, no_try, large, big, duplicates = [], [], [], [], {}

        for fpath in all_files:
            rel = str(fpath.relative_to(ROOT))
            try:
                size = fpath.stat().st_size
                text = fpath.read_text(encoding="utf-8", errors="replace")
                lines = text.count("\n")

                # غير مستخدم (لا يُستورد في أي ملف آخر)
                name = fpath.stem
                imported = any(
                    name in other.read_text(encoding="utf-8", errors="replace")
                    for other in all_files
                    if other != fpath
                )
                if not imported:
                    unused.append(rel)

                # بدون try/except مع عمليات خطرة
                dangerous = any(k in text for k in ["open(", "requests.", "subprocess.", "json.load"])
                if dangerous and "try:" not in text:
                    no_try.append(rel)

                # ملفات ضخمة
                if lines > 600:
                    large.append((rel, lines))
                elif lines > 300:
                    big.append((rel, lines))

                # مكررة
                duplicates.setdefault(fpath.name, []).append(rel)

            except Exception:
                pass

        dupes = {k: v for k, v in duplicates.items() if len(v) > 1}

        ft = filter_type.strip()
        if ft in ("غير مستخدم", "unused"):
            lines_out = [f"📁 ملفات غير مستخدمة ({len(unused)}):"]
            lines_out += [f"  • {f}" for f in unused[:20]]
        elif ft in ("أخطاء", "اخطاء", "errors"):
            lines_out = [f"⚠️ دوال بدون معالجة أخطاء ({len(no_try)}):"]
            lines_out += [f"  • {f}" for f in no_try]
        elif ft in ("كبير", "ضخم", "large"):
            lines_out = [f"📦 ملفات كبيرة ({len(large) + len(big)}):"]
            lines_out += [f"  • {f} ({l} سطر)" for f, l in large + big]
        elif ft in ("مكررة", "duplicate"):
            lines_out = [f"🔁 وحدات مكررة ({len(dupes)}):"]
            lines_out += [f"  • {k}: {', '.join(v)}" for k, v in dupes.items()]
        else:
            lines_out = [
                f"📊 تحليل المشروع — {len(all_files)} ملف Python:",
                f"",
                f"📁 غير مستخدم: {len(unused)} ملف",
                f"⚠️ بدون معالجة أخطاء: {len(no_try)} ملف",
                f"📦 ملفات ضخمة +600 سطر: {len(large)} ملف",
                f"📎 ملفات كبيرة +300 سطر: {len(big)} ملف",
                f"🔁 وحدات مكررة: {len(dupes)} اسم",
                f"",
                f"💡 يمكنك تصفية: اقترح غير مستخدم | أخطاء | كبير | مكررة",
            ]

        return "\n".join(lines_out)
    except Exception as e:
        return f"❌ خطأ في التحليل: {e}"


# ══════════════════════════════════════════════════════════════════
# 7. تصحيح ملف (إضافة try/except)
# ══════════════════════════════════════════════════════════════════
def fix_file(path: str) -> str:
    try:
        f = _safe_path(path)
        if f is None:
            return _UNSAFE_PATH_MSG
        if not f.exists():
            return f"❌ الملف غير موجود: {path}"
        text = f.read_text(encoding="utf-8")

        # فحص وجود عمليات خطرة
        dangerous = [k for k in ["open(", "requests.", "subprocess.", "json.load"] if k in text]
        if not dangerous:
            return f"✅ {path} لا يحتوي على عمليات خطرة — لا حاجة للتصحيح"

        return (
            f"🔍 {path} يحتوي على: {', '.join(dangerous)}\n"
            f"📝 الدوال الخطرة بدون try/except — يُنصح بتغليفها.\n"
            f"💡 استخدم: عدل {path} | الكود_القديم | الكود_الجديد_مع_try"
        )
    except Exception as e:
        return f"❌ خطأ: {e}"


# ══════════════════════════════════════════════════════════════════
# 8. ملخص ملف
# ══════════════════════════════════════════════════════════════════
def summarize_file(path: str) -> str:
    try:
        f = _safe_path(path)
        if f is None:
            return _UNSAFE_PATH_MSG
        if not f.exists():
            return f"❌ الملف غير موجود: {path}"
        text = f.read_text(encoding="utf-8", errors="replace")
        size = f.stat().st_size
        lines = text.count("\n")

        # استخراج الدوال والكلاسات
        funcs, classes, imports = [], [], []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("def "):
                funcs.append(s.split("(")[0].replace("def ", ""))
            elif s.startswith("class "):
                classes.append(s.split("(")[0].replace("class ", "").rstrip(":"))
            elif s.startswith("import ") or s.startswith("from "):
                imports.append(s[:60])

        out = [
            f"📄 {path}",
            f"  الحجم: {lines} سطر | {size} بايت",
        ]
        if classes:
            out.append(f"  الكلاسات ({len(classes)}): {', '.join(classes[:5])}")
        if funcs:
            out.append(f"  الدوال ({len(funcs)}): {', '.join(funcs[:8])}")
        if imports:
            out.append(f"  الاستيرادات: {', '.join(imports[:4])}")

        return "\n".join(out)
    except Exception as e:
        return f"❌ خطأ: {e}"
