"""
NSM Terminal Smart — اقتراحات LLM + تكامل Kaggle CLI
=====================================================
طبقة مستقلة تمدد ai/nsm_terminal.py بأدوات ذكية:

1. Kaggle CLI integration: تنفيذ أوامر kaggle kernels status / logs / list /
   files / output داخل التيرمنال مع معالجة الأخطاء وتلوين المخرجات.
2. Command suggestions (LLM): اقتراح الأمر التالي بناءً على آخر أوامر/مخرجات
   عبر OpenRouter (النماذج السريعة الخفيفة)، مع fallback محلي بدون مفتاح.

لا تستورد مفاتيح API من المحادثة — تستخدم OPENROUTER_API_KEY من secrets/env.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple


# ══════════════════════ Kaggle CLI integration ══════════════════════

def kaggle_binary_available() -> bool:
    """هل أمر kaggle CLI موجود في PATH؟"""
    return shutil.which("kaggle") is not None


def run_kaggle_cmd(args: List[str], timeout: int = 120) -> Tuple[int, str, str]:
    """ينفذ kaggle <args> ويعيد (exit_code, stdout, stderr).
    آمن: لا يمرر المدخلات إلا كقائمة arguments (لا shell=True للأجزاء الخطرة)."""
    if not kaggle_binary_available():
        return 1, "", "أمر kaggle غير مثبت في البيئة — ثبّت: pip install kaggle"
    if not args:
        return 1, "", "استخدام: kaggle kernels status|logs|list|files|output"
    try:
        proc = subprocess.run(
            ["kaggle"] + list(args),
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except FileNotFoundError:
        return 1, "", "أمر kaggle غير موجود في PATH"
    except subprocess.TimeoutExpired:
        return 124, "", f"انتهت مهلة kaggle بعد {timeout}s"


def kaggle_kernel_status(args: List[str]) -> Tuple[int, str, str]:
    """حالة kernel واحد — args: [slug]."""
    kernel_slug = (args or [""])[0]
    if not re.match(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$", kernel_slug):
        return 1, "", "صيغة slug غير صالحة (username/kernel-slug)"
    return run_kaggle_cmd(["kernels", "status", "-k", kernel_slug])


def kaggle_kernel_logs(args: List[str]) -> Tuple[int, str, str]:
    """logs kernel — args: [slug]. مع حد زمن لمنع التضخم."""
    kernel_slug = (args or [""])[0]
    if not re.match(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$", kernel_slug):
        return 1, "", "صيغة slug غير صالحة"
    return run_kaggle_cmd(["kernels", "logs", "-k", kernel_slug], timeout=180)


def kaggle_kernels_list(args: List[str]) -> Tuple[int, str, str]:
    """قائمة kernels المستخدم — args: [username]."""
    user = (args or [""])[0]
    if not re.match(r"^[a-zA-Z0-9._-]+$", user or ""):
        return 1, "", "اسم مستخدم غير صالح"
    return run_kaggle_cmd(["kernels", "list", "--user", user, "--max-page-size", "15"])


def kaggle_kernel_output(args: List[str]) -> Tuple[int, str, str]:
    """تحميل مخرجات kernel إلى مجلد — args: [slug] أو [slug, dest_dir]."""
    slug = args[0] if args else ""
    dest_dir = args[1] if len(args) > 1 else "/tmp/nsm_kag_output"
    if not re.match(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$", slug):
        return 1, "", "صيغة slug غير صالحة"
    os.makedirs(dest_dir, exist_ok=True)
    return run_kaggle_cmd(["kernels", "output", "-k", slug, "-p", dest_dir], timeout=600)


# ══════════════════════ LLM suggestions ══════════════════════

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_SUGGEST_MODEL = "openai/gpt-4.1-mini"
_FAST_MODEL = "google/gemini-2.5-flash"

_PROMPT_TEMPLATE = """أنت مساعد طرفية ذكي لمشروع Neural Service Mesh (NSM) — \
مشروع ذكاء اصطناعي عربي بلغة Python + Streamlit على GitHub.

تاريخ أوامر الطرفية الأخير:
{history}

آخر مخرجات:
{output}

اقترح 3 أوامر مفيدة تاليًا (سطر واحد لكل أمر، بدون شرح، بدون علامات). \
أوامر عملية قصيرة (git/pytest/ls/compilation/streamlit/docker) تناسب السياق.
أجب فقط بسطر واحد يحتوي اقتراحًا واحدًا (الأفضل)."""


def suggest_command(
    history: List[str],
    last_output: str = "",
    api_key: Optional[str] = None,
    model: str = _FAST_MODEL,
    timeout: int = 25,
) -> Tuple[str, bool]:
    """يُقترح الأمر التالي. يعيد (الاقتراح, من_llm).
    بدون مفتاح صالح يرجع fallback محليًا."""
    api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key or not history:
        return _local_fallback(history), False
    last_n = history[-6:]
    out = (last_output or "")[-4000:]
    user = _PROMPT_TEMPLATE.format(
        history="\n".join(f"- {c}" for c in last_n), output=out[:1200] or "—",
    )
    try:
        import requests  # استيراد محلي — متاح في المشروع
        r = requests.post(
            _OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": user}],
                "temperature": 0.4,
                "max_tokens": 120,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        choice = r.json().get("choices") or []
        if choice:
            text = choice[0].get("message", {}).get("content", "").strip()
            text = re.sub(r"^-\s*", "", text).strip()
            text = re.sub(r"[;|`$]", " ", text).strip()[:200]  # لا تسلسلات خطرة
            if text and len(text.split()) <= 12:
                return text, True
    except Exception:
        pass
    return _local_fallback(history), False


def _local_fallback(history: List[str]) -> str:
    """اقتراحات محلية بدون LLM (قواعد بسيطة حسب آخر أمر)."""
    if not history:
        return "git status"
    last = history[-1].strip().lower()
    if last.startswith("git"):
        return "git log --oneline -8" if "status" in last else "git status --short"
    if last.startswith("streamlit"):
        return "curl -s -o /dev/null -w '%{http_code}' http://localhost:8501"
    if "pytest" in last or "compile" in last:
        return "ls -la"
    if last.startswith("cd "):
        return "ls"
    return "ls"


# ══════════════════════ أوامر مدمجة في terminal ══════════════════════

KAGGLE_COMMANDS = {
    # (ملاحظة: handler يستقبل args كـlist — يجب فكها عند الاستدعاء)
    "kg status": lambda a: kaggle_kernel_status(a),
    "kg logs": lambda a: kaggle_kernel_logs(a),
    "kg list": lambda a: kaggle_kernels_list(a),
    "kg output": lambda a: kaggle_kernel_output(a),
}


def is_kaggle_cmd(cmd: str) -> Tuple[bool, Optional[str], List[str]]:
    """هل الأمر kaggle shortcut؟ يعيد (نعم, handler_name, args).
    أمثلة: 'kg status aliahmedmo/nsm-train-xyz'"""
    parts = cmd.split()
    if len(parts) < 2 or parts[0] != "kg":
        return False, None, []
    sub = parts[1]
    # تحويل kg logs إلى kg logs
    key = None
    for k in KAGGLE_COMMANDS:
        if sub == k.split()[1]:
            key = k
            break
    if not key:
        return False, None, []
    return True, key, parts[2:]
