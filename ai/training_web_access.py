"""
Controlled Internet Access for Model Training Agent
===================================================
وصول إنترنت محكوم لوكيل التدريب فقط:

  • قائمة نطاقات مسموحة (whitelist)
  • GET فقط — ممنوع رفع ملفات محلية / أسرار / كود النظام
  • arXiv: بحث أحدث الأوراق
  • Hugging Face: بيانات وصفية للنماذج/المجموعات (بدون تنزيل أوزان ضخمة افتراضياً)
  • بحث مقيّد (عبر web_search_tool مع تصفية النطاقات)
  • حلول برمجية: استعلامات موجّهة لـ Stack Overflow / توثيق

لا يتجاوز وضع NSM_OFFLINE_MODE.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("TrainingWebAccess")

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "model_training" / "web_cache"
ART.mkdir(parents=True, exist_ok=True)

_TIMEOUT = 15
_UA = (
    "Mozilla/5.0 (compatible; NSM-TrainingAgent/1.0; "
    "+https://github.com/aliahmed369000000-ai/Neural-Service-Mesh)"
)
_MAX_RESPONSE_BYTES = 2_000_000  # 2MB سقف للاستجابة
_MAX_CACHE_FILES = 50

# ── Whitelist ──────────────────────────────────────────────────────────────
DEFAULT_WHITELIST: Tuple[str, ...] = (
    "arxiv.org",
    "export.arxiv.org",
    "api.semanticscholar.org",
    "huggingface.co",
    "hf.co",
    "cdn-lfs.huggingface.co",  # ميتا فقط — التنزيل الضخم معطّل افتراضياً
    "stackoverflow.com",
    "api.stackexchange.com",
    "github.com",
    "api.github.com",
    "raw.githubusercontent.com",
    "docs.python.org",
    "pytorch.org",
    "scikit-learn.org",
    "numpy.org",
    "duckduckgo.com",
    "lite.duckduckgo.com",
    "api.duckduckgo.com",
    "en.wikipedia.org",
    "ar.wikipedia.org",
    "kaggle.com",
    "www.kaggle.com",
)

# نطاقات يُحظر الرفع إليها صراحة (جدار حماية البيانات)
BLOCKED_UPLOAD_HOSTS: Set[str] = {
    "pastebin.com",
    "hastebin.com",
    "transfer.sh",
    "file.io",
    "webhook.site",
}

# مسارات محلية لا يجوز إرسال محتواها للخارج
SENSITIVE_PATH_PREFIXES = (
    ".env",
    ".git/",
    "ai/",
    "ui_pages/",
    "core/",
    "scripts/",
    "streamlit_app.py",
    "app_core.py",
    "api_server.py",
    ".streamlit/secrets",
)


def _offline() -> bool:
    try:
        from ai.offline_mode import is_offline
        return bool(is_offline())
    except Exception:
        return os.environ.get("NSM_OFFLINE_MODE", "").strip() in ("1", "true", "yes")


def load_web_policy() -> Dict[str, Any]:
    """تحميل سياسة الويب من guardrails إن وُجدت."""
    path = ROOT / "config" / "training_guardrails.json"
    policy = {
        "enabled": True,
        "whitelist": list(DEFAULT_WHITELIST),
        "allow_model_download": False,
        "allow_dataset_download": False,
        "max_results": 5,
        "timeout_seconds": _TIMEOUT,
    }
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            web = data.get("web_access") or {}
            policy.update({k: v for k, v in web.items() if v is not None})
            if "whitelist" in web and isinstance(web["whitelist"], list):
                policy["whitelist"] = web["whitelist"]
    except Exception as e:
        logger.warning("web policy load: %s", e)
    return policy


def is_host_allowed(url: str, whitelist: Optional[List[str]] = None) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    wl = whitelist or load_web_policy().get("whitelist") or list(DEFAULT_WHITELIST)
    for allowed in wl:
        a = allowed.lower().lstrip(".")
        if host == a or host.endswith("." + a):
            return True
    return False


def _firewall_block_upload(url: str, body: Optional[bytes] = None) -> Optional[str]:
    """يمنع أي طلب يحمل حمولة محلية حساسة أو يستهدف مضيف رفع."""
    host = (urlparse(url).hostname or "").lower()
    if host in BLOCKED_UPLOAD_HOSTS:
        return f"محظور: مضيف رفع غير موثوق ({host})"
    if body:
        # رفض POST/PUT بمحتوى يشبه أسراراً أو كود نظام
        sample = body[:2000].decode("utf-8", errors="ignore")
        if re.search(r"(API_KEY|SECRET|PASSWORD|ghp_|sk-|BEGIN RSA)", sample, re.I):
            return "محظور: محاولة إرسال أسرار محتملة إلى الإنترنت"
        for pref in SENSITIVE_PATH_PREFIXES:
            if pref in sample and len(sample) > 100:
                return f"محظور: محتوى يشبه مسارات نظام حساسة ({pref})"
    return None


def safe_http_get(url: str, accept: str = "application/json,text/plain,*/*") -> Tuple[bool, str]:
    """
    GET آمن: whitelist + سقف حجم + بدون إرسال ملفات محلية.
    يعيد (ok, text_or_error).
    """
    if _offline():
        return False, "وضع عدم الاتصال (NSM_OFFLINE_MODE) — الإنترنت معطّل."

    policy = load_web_policy()
    if not policy.get("enabled", True):
        return False, "وصول الويب لوكيل التدريب معطّل في الإعدادات."

    if not url.startswith("https://") and not url.startswith("http://"):
        return False, "يُسمح فقط بـ http/https."

    if not is_host_allowed(url, policy.get("whitelist")):
        host = urlparse(url).hostname
        return False, f"النطاق غير مسموح (whitelist): {host}"

    block = _firewall_block_upload(url)
    if block:
        return False, block

    timeout = int(policy.get("timeout_seconds") or _TIMEOUT)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": _UA, "Accept": accept},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(_MAX_RESPONSE_BYTES + 1)
            if len(data) > _MAX_RESPONSE_BYTES:
                return False, f"الاستجابة أكبر من الحد ({_MAX_RESPONSE_BYTES} بايت)."
            text = data.decode("utf-8", errors="replace")
            return True, text
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _cache_write(name: str, payload: Dict[str, Any]) -> Path:
    path = ART / f"{name}_{int(datetime.now().timestamp())}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # تنظيف قديم
    files = sorted(ART.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[_MAX_CACHE_FILES:]:
        try:
            old.unlink()
        except Exception:
            pass
    return path


# ── arXiv ──────────────────────────────────────────────────────────────────

def search_arxiv(query: str, max_results: int = 5) -> str:
    """بحث أوراق في arXiv عبر API الرسمي (GET)."""
    q = (query or "").strip() or "neural network training"
    max_results = max(1, min(int(max_results), 10))
    url = (
        "http://export.arxiv.org/api/query?"
        + urllib.parse.urlencode(
            {
                "search_query": f"all:{q}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
    )
    # arxiv API غالباً http على export — اسمح به من whitelist
    ok, body = safe_http_get(url, accept="application/atom+xml")
    if not ok:
        # جرّب https
        url2 = url.replace("http://", "https://", 1)
        ok, body = safe_http_get(url2, accept="application/atom+xml")
    if not ok:
        return f"❌ فشل بحث arXiv: {body}"

    lines = [f"## 📄 arXiv — أحدث نتائج لـ: `{q}`", ""]
    try:
        root = ET.fromstring(body)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entries = root.findall("a:entry", ns)
        papers = []
        for ent in entries:
            title = (ent.findtext("a:title", default="", namespaces=ns) or "").strip()
            title = re.sub(r"\s+", " ", title)
            summary = (ent.findtext("a:summary", default="", namespaces=ns) or "").strip()
            summary = re.sub(r"\s+", " ", summary)[:400]
            published = (ent.findtext("a:published", default="", namespaces=ns) or "")[:10]
            link = ""
            for l in ent.findall("a:link", ns):
                if l.get("type") == "text/html" or l.get("rel") == "alternate":
                    link = l.get("href") or ""
                    break
            if not link:
                link = ent.findtext("a:id", default="", namespaces=ns) or ""
            papers.append({"title": title, "summary": summary, "published": published, "link": link})
            lines.append(f"### {title}")
            lines.append(f"- التاريخ: {published}")
            lines.append(f"- الرابط: {link}")
            lines.append(f"- ملخص: {summary}")
            lines.append("")
        if not papers:
            lines.append("لا نتائج.")
        cache = _cache_write("arxiv", {"query": q, "papers": papers})
        lines.append(f"_حُفظت النتائج في `{cache.relative_to(ROOT)}`_")
    except Exception as e:
        lines.append(f"تعذّر تحليل Atom: {e}")
        lines.append(body[:500])
    return "\n".join(lines)


# ── Hugging Face (metadata only by default) ────────────────────────────────

def search_huggingface(query: str, kind: str = "models", max_results: int = 5) -> str:
    """
    بحث نماذج أو مجموعات بيانات على HF عبر API العامة (ميتا فقط).
    تنزيل الأوزان معطّل افتراضياً (allow_model_download=false).
    """
    q = (query or "").strip() or "bert"
    kind = "datasets" if kind in ("dataset", "datasets", "data") else "models"
    max_results = max(1, min(int(max_results), 10))
    url = f"https://huggingface.co/api/{kind}?search={urllib.parse.quote(q)}&limit={max_results}"
    ok, body = safe_http_get(url)
    if not ok:
        return f"❌ فشل Hugging Face API: {body}"
    try:
        items = json.loads(body)
    except Exception as e:
        return f"❌ JSON غير صالح: {e}"

    policy = load_web_policy()
    lines = [f"## 🤗 Hugging Face — {kind} لـ `{q}`", ""]
    if not policy.get("allow_model_download"):
        lines.append("_تنزيل الأوزان معطّل افتراضياً (حماية المساحة/الأمان). الميثا فقط._")
        lines.append("")

    results = []
    if not isinstance(items, list):
        items = items.get("models") or items.get("datasets") or []
    for it in items[:max_results]:
        if not isinstance(it, dict):
            continue
        mid = it.get("modelId") or it.get("id") or it.get("name") or "?"
        downloads = it.get("downloads") or it.get("downloadsAllTime") or "—"
        tags = ", ".join((it.get("tags") or [])[:6])
        lines.append(f"- **{mid}** | downloads: {downloads} | tags: {tags}")
        results.append({"id": mid, "downloads": downloads, "tags": it.get("tags")})
    if not results:
        lines.append("لا نتائج.")
    cache = _cache_write("hf", {"query": q, "kind": kind, "results": results})
    lines.append(f"\n_حُفظت في `{cache.relative_to(ROOT)}`_")
    return "\n".join(lines)


# ── Restricted web search ──────────────────────────────────────────────────

def restricted_search(query: str, max_results: int = 5, site_filter: Optional[str] = None) -> str:
    """
    بحث عبر الأداة المشتركة ثم تصفية النتائج حسب whitelist.
    site_filter اختياري مثل: site:stackoverflow.com
    """
    if _offline():
        return "وضع عدم الاتصال — البحث معطّل."

    q = (query or "").strip()
    if site_filter and "site:" not in q:
        q = f"{q} {site_filter}"
    try:
        from ai.web_search_tool import web_search
        raw = web_search(q, max_results=max(max_results * 2, 8))
    except Exception as e:
        return f"❌ تعذّر البحث: {e}"

    if not raw or str(raw).startswith("❌"):
        return str(raw)

    policy = load_web_policy()
    wl = policy.get("whitelist") or list(DEFAULT_WHITELIST)
    lines = [f"## 🔍 بحث مقيّد (whitelist) — `{q}`", ""]
    # استخرج روابط من النص
    urls = re.findall(r"https?://[^\s\)\]\>\"']+", str(raw))
    kept = []
    for u in urls:
        u = u.rstrip(".,;")
        if is_host_allowed(u, wl):
            kept.append(u)
    # اعرض النص مع تنويه التصفية
    lines.append(str(raw)[:3500])
    lines.append("")
    lines.append(f"### روابط ضمن whitelist ({len(kept)})")
    for u in kept[:max_results]:
        lines.append(f"- {u}")
    if not kept:
        lines.append("- لم تُعثر روابط ضمن النطاقات المسموحة في هذه النتائج.")
    _cache_write("search", {"query": q, "kept_urls": kept[:20]})
    return "\n".join(lines)


def search_stackoverflow(query: str) -> str:
    return restricted_search(query, site_filter="site:stackoverflow.com")


def search_docs(query: str) -> str:
    return restricted_search(
        query,
        site_filter="site:docs.python.org OR site:pytorch.org OR site:scikit-learn.org",
    )


# ── Policy report ──────────────────────────────────────────────────────────

def web_access_status() -> str:
    policy = load_web_policy()
    offline = _offline()
    lines = [
        "## 🌐 حالة الوصول للإنترنت (محكوم)",
        f"- مفعّل: **{policy.get('enabled')}**",
        f"- وضع offline: **{offline}**",
        f"- تنزيل أوزان نماذج: **{policy.get('allow_model_download')}**",
        f"- تنزيل مجموعات بيانات: **{policy.get('allow_dataset_download')}**",
        f"- timeout: {policy.get('timeout_seconds')}s",
        "",
        "### Whitelist (عينة)",
    ]
    for h in (policy.get("whitelist") or [])[:20]:
        lines.append(f"- `{h}`")
    lines.append("")
    lines.append("### جدار الحماية")
    lines.append("- GET فقط عبر `safe_http_get`")
    lines.append("- ممنوع رفع أسرار/كود نظام")
    lines.append(f"- مضيفات رفع محظورة: {', '.join(sorted(BLOCKED_UPLOAD_HOSTS))}")
    lines.append("")
    lines.append(
        "أوامر: `ابحث arxiv …` · `ابحث huggingface …` · `ابحث stackoverflow …` · "
        "`ابحث توثيق …` · `حالة الويب`"
    )
    return "\n".join(lines)


def handle_web_command(user_input: str) -> Optional[str]:
    text = (user_input or "").strip()
    if not text:
        return None

    if re.search(r"(حالة|status).{0,10}(الويب|web|إنترنت|internet)", text, re.I):
        return web_access_status()
    if text.lower() in ("حالة الويب", "web status", "whitelist"):
        return web_access_status()

    m = re.search(
        r"(?:ابحث|بحث|search).{0,8}(?:arxiv|أوراق|ورقة)\s*(.*)$",
        text,
        re.I,
    )
    if m or re.search(r"^arxiv\s+", text, re.I):
        q = (m.group(1) if m else re.sub(r"^arxiv\s+", "", text, flags=re.I)).strip()
        return search_arxiv(q or "deep learning")

    m = re.search(
        r"(?:ابحث|بحث|search).{0,12}(?:huggingface|hugging\s*face|hf)\s*(.*)$",
        text,
        re.I,
    )
    if m:
        rest = (m.group(1) or "").strip()
        kind = "datasets" if re.search(r"data|مجموعة|dataset", rest, re.I) else "models"
        q = re.sub(r"(dataset|datasets|models|نموذج|بيانات)\s*", "", rest, flags=re.I).strip()
        return search_huggingface(q or "bert", kind=kind)

    if re.search(r"(?:ابحث|بحث|search).{0,12}(stackoverflow|stack\s*overflow|خطأ\s*برمج)", text, re.I):
        q = re.sub(
            r"(?:ابحث|بحث|search).{0,12}(?:stackoverflow|stack\s*overflow|خطأ\s*برمج\w*)\s*",
            "",
            text,
            flags=re.I,
        ).strip()
        return search_stackoverflow(q or "python out of memory pytorch")

    if re.search(r"(?:ابحث|بحث|search).{0,12}(توثيق|docs|documentation)", text, re.I):
        q = re.sub(
            r"(?:ابحث|بحث|search).{0,12}(?:توثيق|docs|documentation)\s*",
            "",
            text,
            flags=re.I,
        ).strip()
        return search_docs(q or "torch.nn.Linear")

    if re.search(r"(?:ابحث\s+ويب|بحث\s+مقيّد|restricted\s*search)\s+(.+)$", text, re.I):
        m2 = re.search(r"(?:ابحث\s+ويب|بحث\s+مقيّد|restricted\s*search)\s+(.+)$", text, re.I)
        return restricted_search((m2.group(1) if m2 else "").strip())

    return None
