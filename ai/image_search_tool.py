"""
NSM Image Search Tool — ai/image_search_tool.py
=================================================
أداة بحث حقيقية عن الصور باستخدام Unsplash API.

الاستخدام:
    from ai.image_search_tool import image_search
    results = image_search("قطط", max_results=9)
    # -> List[Dict]: [{"url", "thumb_url", "description", "author", "author_url", "link"}, ...]

المتطلبات:
    UNSPLASH_ACCESS_KEY في Secrets (مجاني من unsplash.com/developers).

الدالة لا تُرجع أبداً نتائج وهمية: لو فُقد المفتاح أو فشل الطلب تُرجع
قائمة فارغة مع رسالة خطأ صريحة في المفتاح "error" ضمن أول عنصر يُستدعى عبر image_search_safe().
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

_TIMEOUT = 10
_API_URL = "https://api.unsplash.com/search/photos"
_UA = (
    "Mozilla/5.0 (compatible; NSMAgent/1.0; "
    "+https://github.com/aliahmed369000000-ai/Neural-Service-Mesh)"
)


class ImageSearchError(Exception):
    """يُرفع عند فشل البحث عن الصور (مفتاح مفقود أو خطأ من الـ API)."""


def _get_access_key() -> str:
    key = os.environ.get("UNSPLASH_ACCESS_KEY", "").strip()
    if not key:
        raise ImageSearchError(
            "UNSPLASH_ACCESS_KEY غير موجود في Secrets — أضفه لتفعيل البحث عن الصور."
        )
    return key


def image_search(query: str, max_results: int = 9) -> List[Dict[str, str]]:
    """يبحث عن صور حقيقية في Unsplash ويعيد قائمة نتائج منظّمة.

    يرفع ImageSearchError صراحةً عند غياب المفتاح أو فشل الطلب — لا يوجد
    أي fallback وهمي.
    """
    query = (query or "").strip()
    if not query:
        raise ImageSearchError("استعلام البحث فارغ.")

    access_key = _get_access_key()
    max_results = max(1, min(int(max_results or 9), 30))

    params = urllib.parse.urlencode({
        "query": query,
        "per_page": max_results,
        "orientation": "squarish",
    })
    url = f"{_API_URL}?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _UA,
            "Authorization": f"Client-ID {access_key}",
            "Accept-Version": "v1",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        raise ImageSearchError(f"فشل طلب Unsplash (HTTP {e.code}): {body[:200]}") from e
    except urllib.error.URLError as e:
        raise ImageSearchError(f"تعذّر الاتصال بـ Unsplash: {e.reason}") from e
    except Exception as e:  # noqa: BLE001
        raise ImageSearchError(f"خطأ غير متوقع أثناء البحث عن الصور: {e}") from e

    results: List[Dict[str, str]] = []
    for item in payload.get("results", []):
        urls = item.get("urls", {}) or {}
        user = item.get("user", {}) or {}
        results.append({
            "url": urls.get("regular") or urls.get("full") or "",
            "thumb_url": urls.get("thumb") or urls.get("small") or "",
            "description": item.get("description") or item.get("alt_description") or "",
            "author": user.get("name") or "مجهول",
            "author_url": (user.get("links") or {}).get("html", ""),
            "link": (item.get("links") or {}).get("html", ""),
        })

    if not results:
        raise ImageSearchError(f"لا توجد نتائج صور لـ «{query}».")

    return results


def image_search_safe(query: str, max_results: int = 9) -> Dict[str, object]:
    """غلاف آمن يعيد {"ok": bool, "results": [...], "error": Optional[str]}."""
    try:
        return {"ok": True, "results": image_search(query, max_results), "error": None}
    except ImageSearchError as e:
        return {"ok": False, "results": [], "error": str(e)}
