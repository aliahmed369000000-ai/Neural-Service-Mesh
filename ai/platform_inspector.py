"""Read-only, SSRF-resistant public web platform inspector."""
from __future__ import annotations
import html, ipaddress, re, socket, urllib.error, urllib.parse, urllib.request
from typing import Any, Dict

_UA = "NSM-Platform-Inspector/1.0"
_MAX_BYTES = 512_000
_TIMEOUT = 8


def _safe_url(raw: str) -> str:
    url = (raw or "").strip(); p = urllib.parse.urlparse(url)
    if p.scheme not in {"http", "https"} or not p.hostname:
        raise ValueError("يسمح فقط بروابط http وhttps العامة")
    host = p.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise ValueError("الوصول إلى المضيف المحلي غير مسموح")
    try:
        infos = socket.getaddrinfo(host, p.port or (443 if p.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("تعذر حل اسم المضيف") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if any((ip.is_private, ip.is_loopback, ip.is_link_local, ip.is_reserved, ip.is_multicast, ip.is_unspecified)):
            raise ValueError("تم رفض عنوان داخلي أو محجوز حفاظاً على السلامة")
    return urllib.parse.urlunparse((p.scheme, p.netloc, p.path or "/", "", p.query, ""))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _request(url: str, accept: str = "*/*"):
    safe = _safe_url(url)
    req = urllib.request.Request(safe, headers={"User-Agent": _UA, "Accept": accept})
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(req, timeout=_TIMEOUT) as response:
            return response.geturl(), dict(response.headers.items()), response.read(_MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code not in {301, 302, 303, 307, 308} or not exc.headers.get("Location"):
            raise
        target = _safe_url(urllib.parse.urljoin(safe, exc.headers["Location"]))
        req = urllib.request.Request(target, headers={"User-Agent": _UA, "Accept": accept})
        with opener.open(req, timeout=_TIMEOUT) as response:
            return response.geturl(), dict(response.headers.items()), response.read(_MAX_BYTES + 1)


def inspect_platform(url: str) -> Dict[str, Any]:
    safe = _safe_url(url)
    final, headers, raw = _request(safe, "text/html,application/xhtml+xml")
    text = raw[:_MAX_BYTES].decode("utf-8", errors="replace")
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", text)
    title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", match.group(1) if match else ""))).strip()[:200]
    links = []
    for href in re.findall(r"(?is)\b(?:href|src)\s*=\s*['\"]([^'\"]+)", text):
        absolute = urllib.parse.urljoin(final, html.unescape(href))
        if urllib.parse.urlparse(absolute).scheme in {"http", "https"} and absolute not in links:
            links.append(absolute)
    security = {k.lower(): v for k, v in headers.items() if k.lower() in {"content-security-policy", "strict-transport-security", "x-content-type-options", "referrer-policy", "permissions-policy"}}
    return {"ok": True, "url": safe, "final_url": final, "title": title, "content_type": headers.get("Content-Type", ""), "bytes_read": min(len(raw), _MAX_BYTES), "truncated": len(raw) > _MAX_BYTES, "links": links[:100], "link_count": len(links), "forms": len(re.findall(r"(?is)<form\b", text)), "scripts": len(re.findall(r"(?is)<script\b", text)), "security_headers": security, "limits": {"timeout_seconds": _TIMEOUT, "max_bytes": _MAX_BYTES, "actions": "read-only"}}


def format_inspection(report: Dict[str, Any]) -> str:
    if not report.get("ok"):
        return f"فشل فحص المنصة: {report.get('error', 'unknown error')}"
    return (f"## تقرير فحص المنصة\n\nالعنوان: {report['title'] or 'غير متاح'}\nالرابط النهائي: {report['final_url']}\n"
            f"الروابط: {report['link_count']} | النماذج: {report['forms']} | السكربتات: {report['scripts']}\n"
            f"رؤوس الأمان: {', '.join(report['security_headers']) or 'غير موجودة'}\n\n"
            "فحص قراءة فقط؛ لا تسجيل دخول أو إرسال نماذج أو تجاوز حماية.")


def inspect_platform_command(url: str) -> str:
    try:
        return format_inspection(inspect_platform(url))
    except Exception as exc:
        return f"فشل فحص المنصة بأمان: {str(exc)[:240]}"
