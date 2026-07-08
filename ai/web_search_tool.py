"""
NSM Web Search Tool — ai/web_search_tool.py
=============================================
أداة بحث حقيقية في الإنترنت بدون الحاجة لأي مفتاح API.
مصدر واحد مشترك يُستخدم من:
  - ai/nsm_agent_core.py   (action: "web_search" داخل حلقة الوكيل الرئيسية)
  - ai/code_agent.py       (أمر: "ابحث <نص>" في nsm_chat.py)

الإستراتيجية (بالترتيب، بدون مفتاح API لأي منها):
  1) DuckDuckGo HTML Lite            → نتائج ويب حقيقية (عنوان + رابط + مقتطف)
  2) DuckDuckGo Instant Answer API   → احتياطي JSON رسمي لو فشل (1)

⚠️ ملاحظة صادقة: الخيار (1) يعتمد على هيكل صفحة DuckDuckGo الحالية.
لو غيّروا تصميم الصفحة مستقبلاً، _parse_lite_html() قد تحتاج تحديث بسيط
(هذا حال أي scraping بدون مفتاح API — لا يوجد طريقة مجانية مضمونة للأبد).
الدالة لا تُرجع أبداً "نجاح" وهمي: لو فشل المصدران تُرجع رسالة خطأ صريحة.
"""
from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List

_TIMEOUT = 10
_UA = (
    "Mozilla/5.0 (compatible; NSMAgent/1.0; "
    "+https://github.com/aliahmed369000000-ai/Neural-Service-Mesh)"
)


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _parse_lite_html(html_text: str, max_results: int) -> List[Dict[str, str]]:
    """يستخرج (عنوان، رابط، مقتطف) من صفحة lite.duckduckgo.com/lite/"""
    results: List[Dict[str, str]] = []
    link_pattern = re.compile(
        r'<a[^>]+rel="nofollow"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S
    )
    snippet_pattern = re.compile(r'<td class="result-snippet"[^>]*>(.*?)</td>', re.S)

    links = link_pattern.findall(html_text)
    snippets = snippet_pattern.findall(html_text)

    for i, (url, title) in enumerate(links):
        if not url.startswith("http"):
            continue
        title_clean = html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
        if not title_clean:
            continue
        snippet = ""
        if i < len(snippets):
            snippet = html.unescape(re.sub(r"<[^>]+>", "", snippets[i])).strip()
        results.append({"title": title_clean, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


def _search_duckduckgo_lite(query: str, max_results: int) -> List[Dict[str, str]]:
    q = urllib.parse.quote_plus(query)
    url = f"https://lite.duckduckgo.com/lite/?q={q}"
    html_text = _fetch(url)
    return _parse_lite_html(html_text, max_results)


def _search_instant_answer(query: str) -> List[Dict[str, str]]:
    """احتياطي: DuckDuckGo Instant Answer API (JSON رسمي، بدون مفتاح، نتائج محدودة)."""
    q = urllib.parse.quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
    raw = _fetch(url)
    data = json.loads(raw)

    results: List[Dict[str, str]] = []
    abstract = (data.get("AbstractText") or "").strip()
    if abstract:
        results.append({
            "title": data.get("Heading") or query,
            "url": data.get("AbstractURL") or "",
            "snippet": abstract,
        })
    for topic in data.get("RelatedTopics", [])[:5]:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({
                "title": (topic.get("Text") or "")[:80],
                "url": topic.get("FirstURL") or "",
                "snippet": topic.get("Text") or "",
            })
    return results


def _format_results(query: str, results: List[Dict[str, str]], source: str) -> str:
    lines = [f"🔍 نتائج البحث عن: **{query}** (المصدر: {source})\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r['title']}**")
        if r.get("url"):
            lines.append(f"   {r['url']}")
        if r.get("snippet"):
            snippet = r["snippet"]
            if len(snippet) > 220:
                snippet = snippet[:220] + "..."
            lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def web_search(query: str, max_results: int = 5) -> str:
    """
    بحث حقيقي في الإنترنت بدون مفتاح API.
    يُرجع نصاً منسقاً بالنتائج، أو رسالة خطأ صريحة لو فشلت كل المصادر
    (لا يختلق نتيجة وهمية أبداً).
    """
    query = (query or "").strip()
    if not query:
        return "❌ web_search: مطلوب query (نص البحث)"

    max_results = max(1, min(int(max_results or 5), 10))
    errors: List[str] = []

    try:
        results = _search_duckduckgo_lite(query, max_results)
        if results:
            return _format_results(query, results, source="DuckDuckGo")
    except Exception as e:
        errors.append(f"DuckDuckGo HTML: {e}")

    try:
        results = _search_instant_answer(query)
        if results:
            return _format_results(query, results[:max_results], source="DuckDuckGo Instant Answer")
    except Exception as e:
        errors.append(f"Instant Answer API: {e}")

    detail = " | ".join(errors) if errors else "لا نتائج مطابقة"
    return f"❌ فشل البحث عن '{query}': {detail}"
