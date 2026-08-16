"""
ai/agent_browser.py
===================
🆕 أداة المتصفح للوكلاء — بدون Playwright/سيلينيوم وبلا اعتماديات جديدة.

توفر للوكيل 4 قدرات أساسية (بلا مفاتيح API):
  1. browser_navigate — تحميل صفحة واستخراج نصها وروابطها (HTML → نص نظيف)
  2. browser_api — استدعاء REST API مباشر (GET/POST/PUT/DELETE) مع رؤوس وجسم
  3. browser_download — تنزيل ملف إلى artifacts/ مع التحقق من الحجم والنوع
  4. browser_inspect — فحص روابط/عناوين/صور داخل صفحة معينة

الأمان:
  - HTTPS فقط (HTTP يُحوَّل إلى HTTPS؛ لا بروتوكولات أخرى)
  - timeout إجباري (15 ثوانٍ افتراضيًا، سقف 60)
  - رفض عناوين internal/private (localhost/10.x/192.168/172.16-31/169.254)
    لحماية بيئة التشغيل — يمكن رفعها بـallow_internal=True صراحة
  - حد حجم استجابة (2MB) وحد نص معاد للنموذج (6000 حرف)
  - لا تنفيذ JavaScript داخل النطاق — للنصوص فقط؛
    المحتوى الديناميكي JS ثقيل يُنصح بـ fetch_url البسيط
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts" / "agent_browser"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

_MAX_BODY = 2 * 1024 * 1024        # 2MB
_MAX_TEXT = 6000                    # ما يُعاد للنموذج
_DEFAULT_TIMEOUT = 15
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# عناوين داخلية محظورة افتراضيًا
_INTERNAL_RE = re.compile(
    r"^(localhost|127\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|169\.254\.|0\.0\.0\.0|"
    r"\[::1\]|\[fc|0000:0000)"
)


def _allowed_url(raw: str, allow_internal: bool = False) -> Tuple[bool, str]:
    try:
        u = urllib.parse.urlparse(raw)
    except Exception as e:
        return False, f"عنوان غير صالح: {e}"
    if u.scheme and u.scheme.lower() not in ("https", "http", ""):
        return False, f"بروتوكول غير مسموح: {u.scheme} (HTTPS فقط)"
    target = f"https://{u.netloc}{u.path}" if u.scheme in ("http", "") else raw
    if not allow_internal and _INTERNAL_RE.search(u.netloc):
        return False, "عنوان داخلي محظور (استخدم allow_internal=True صراحة)"
    return True, target


def _strip_html(html: str) -> str:
    """تحويل HTML إلى نص نظيف: إزالة scripts/styles ثم الوسوم ثم ضغط."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<(p|div|tr|li|h[1-6])[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_links(html: str, base_url: str) -> List[str]:
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html)
    out, seen = [], set()
    for h in hrefs:
        if not h or h.startswith(("#", "mailto:", "javascript:")):
            continue
        abs_h = urllib.parse.urljoin(base_url, h)
        if abs_h not in seen and abs_h.startswith("http"):
            seen.add(abs_h)
            out.append(abs_h)
        if len(out) >= 40:
            break
    return out


def navigate(url: str, *, allow_internal: bool = False,
             timeout: int = _DEFAULT_TIMEOUT, max_chars: int = _MAX_TEXT) -> Dict[str, Any]:
    """تحميل صفحة واستخراج نصها وروابطها."""
    ok, target = _allowed_url(url, allow_internal)
    if not ok:
        return {"ok": False, "error": target}
    t0 = time.time()
    try:
        req = urllib.request.Request(target, headers={"User-Agent": UA,
                                                      "Accept": "text/html,application/xhtml+xml"})
        timeout = max(3, min(int(timeout), 60))
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ctype = r.headers.get("Content-Type", "")
            body = r.read(_MAX_BODY)
            encoding = "utf-8"
            m = re.search(r"charset=([\w-]+)", ctype)
            if m:
                encoding = m.group(1)
            html = body.decode(encoding, errors="replace")
        text = _strip_html(html)[:max_chars]
        links = _extract_links(html, target)[:40]
        return {
            "ok": True, "url": target,
            "title": (lambda m: m.group(1) if m else "")(re.search(
                r"<title[^>]*>(.*?)</title>", html, re.S | re.I)),
            "text": text, "links": links,
            "duration_ms": int((time.time() - t0) * 1000),
        }
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}", "url": target,
                "duration_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300], "url": target,
                "duration_ms": int((time.time() - t0) * 1000)}


def api_call(url: str, *, method: str = "GET", headers: Dict[str, str] | None = None,
             body: Any = None, timeout: int = _DEFAULT_TIMEOUT,
             allow_internal: bool = False) -> Dict[str, Any]:
    """استدعاء REST API مباشر (GET/POST/PUT/DELETE/HEAD/OPTIONS)."""
    ok, target = _allowed_url(url, allow_internal)
    if not ok:
        return {"ok": False, "error": target}
    method = (method or "GET").strip().upper()
    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
        return {"ok": False, "error": f"طريقة HTTP غير مدعومة: {method}"}
    t0 = time.time()
    try:
        data = None
        req_headers = {"User-Agent": UA, "Accept": "*/*"}
        if body is not None:
            if isinstance(body, (dict, list)):
                data = json.dumps(body, ensure_ascii=False).encode()
                req_headers["Content-Type"] = "application/json; charset=utf-8"
            elif isinstance(body, str):
                data = body.encode()
        for k, v in (headers or {}).items():
            req_headers[str(k)] = str(v)
        req = urllib.request.Request(target, data=data, headers=req_headers,
                                     method=method)
        timeout = max(3, min(int(timeout), 60))
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp_headers = {k: v for k, v in r.headers.items()}
            resp_body = r.read(_MAX_BODY).decode("utf-8", errors="replace")[:_MAX_TEXT]
        return {"ok": True, "status": r.status, "headers": resp_headers,
                "body": resp_body,
                "duration_ms": int((time.time() - t0) * 1000)}
    except urllib.error.HTTPError as e:
        resp_body = ""
        try:
            resp_body = e.read(_MAX_BODY).decode("utf-8", errors="replace")[:_MAX_TEXT]
        except Exception:
            pass
        return {"ok": False, "status": e.code, "error": f"HTTP {e.code}",
                "body": resp_body, "duration_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300],
                "duration_ms": int((time.time() - t0) * 1000)}


def download(url: str, *, filename: str = "", allow_internal: bool = False,
             timeout: int = _DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """تنزيل ملف إلى artifacts/agent_browser/."""
    ok, target = _allowed_url(url, allow_internal)
    if not ok:
        return {"ok": False, "error": target}
    fname = (filename or Path(urllib.parse.urlparse(target).path).name
             or "download").strip()
    fname = re.sub(r"[^A-Za-z0-9._\-]", "_", fname)[:80]
    dest = ARTIFACTS / fname
    t0 = time.time()
    try:
        req = urllib.request.Request(target, headers={"User-Agent": UA})
        timeout = max(5, min(int(timeout), 120))
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read(_MAX_BODY)
        dest.write_bytes(data)
        return {"ok": True, "saved": str(dest), "size_bytes": len(data),
                "duration_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300],
                "duration_ms": int((time.time() - t0) * 1000)}


def inspect(url: str, *, what: str = "links", allow_internal: bool = False,
            timeout: int = _DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """فحص عناصر صفحة: links|headings|images."""
    ok, target = _allowed_url(url, allow_internal)
    if not ok:
        return {"ok": False, "error": target}
    nav = navigate(target, allow_internal=allow_internal, timeout=timeout)
    if not nav.get("ok"):
        return nav
    t0 = time.time()
    req = urllib.request.Request(target, headers={"User-Agent": UA})
    try:
        timeout = max(3, min(int(timeout), 60))
        with urllib.request.urlopen(req, timeout=timeout) as r:
            html = r.read(_MAX_BODY).decode("utf-8", errors="replace")
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}
    what = (what or "links").lower()
    if what == "headings":
        items = [f"{m.group(1)}|{unescape(m.group(2).strip())}"
                 for m in re.finditer(r"<(h[1-6])[^>]*>(.*?)</\1>", html, re.S | re.I)][:40]
    elif what == "images":
        items = [urllib.parse.urljoin(target, m.group(1))
                 for m in re.finditer(r'src=["\']([^"\']+)["\']', html)
                 if re.search(r"\.(jpe?g|png|gif|webp|svg)($|\?)", m.group(1), re.I)][:40]
    else:
        items = _extract_links(html, target)[:40]
    return {"ok": True, "url": target, "what": what, "items": items,
            "duration_ms": int((time.time() - t0) * 1000)}
