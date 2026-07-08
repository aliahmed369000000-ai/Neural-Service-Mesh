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
# 1. قراءة ملف
# ══════════════════════════════════════════════════════════════════
def read_file(path: str) -> str:
    try:
        f = ROOT / path
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
        base = ROOT / folder
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
        f = ROOT / path
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
        f = ROOT / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
        return f"✅ تم إنشاء {path}"
    except Exception as e:
        return f"❌ خطأ في الإنشاء: {e}"


# ══════════════════════════════════════════════════════════════════
# 5. رفع لـ GitHub
# ══════════════════════════════════════════════════════════════════
def git_push(message: str = "NSM auto-commit") -> str:
    try:
        cmds = [
            ["git", "-C", str(ROOT), "add", "-A"],
            ["git", "-C", str(ROOT), "commit", "-m", message],
            ["git", "-C", str(ROOT), "push"],
        ]
        for cmd in cmds:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                if "nothing to commit" in r.stdout + r.stderr:
                    return "ℹ️ لا توجد تغييرات للرفع"
                return f"❌ خطأ: {r.stderr.strip()}"
        return "✅ رُفع بنجاح لـ GitHub"
    except Exception as e:
        return f"❌ خطأ في git: {e}"


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
        f = ROOT / path
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
        f = ROOT / path
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
