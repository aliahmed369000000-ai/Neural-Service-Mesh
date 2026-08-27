"""
NSM Web Search Tool — ai/web_search_tool.py (v2 موسّع)
======================================================
بحث حقيقي متعدد المصادر بدون مفتاح API إجباري.

المصادر (بالترتيب / التجميع):
  1) DuckDuckGo HTML Lite
  2) Wikipedia (ar ثم en) + ملخصات
  3) DuckDuckGo Instant Answer
  4) Google News RSS (أخبار)
  5) Wikidata search (كيانات)
  6) arXiv API (علمي)
  7) Google Trends RSS (رائج)

واجهة:
  web_search(query)              → نص منسّق (متوافق مع القديم)
  web_search_structured(query)   → قائمة dict للوكلاء
  deep_research(query)           → بحث متعدد الزوايا + تجميع
  get_trending_topics(geo)       → مواضيع رائجة
  search_news(query)             → أخبار
"""
from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

try:
    from ai.offline_mode import is_offline, offline_message
except Exception:  # pragma: no cover
    def is_offline() -> bool:
        return False

    def offline_message(what: str = "") -> str:
        return f"وضع عدم اتصال — {what} غير متاح"

_TIMEOUT = 6
# 🛠️ إصلاح: web_search_structured() كانت تستدعي حتى 6-7 مصادر بالتتابع
# (duckduckgo + ddg_instant + wikipedia ar/en + wikidata + news [+ arxiv])،
# كل واحد بمهلة _TIMEOUT ثانية. في أسوأ حالة (مصدر بطيء/محجوب فعلياً على
# Streamlit Community Cloud) كان هذا يعني حظر تشغيل السكربت لعشرات الثواني
# (كان يصل حتى ~80 ثانية مع _TIMEOUT=12)، مما يتسبب بانقطاع WebSocket
# الخاص بـ Streamlit وإعادة تحميل كامل للتطبيق — وهو ما يظهر للمستخدم على
# هيئة "الرجوع فجأة لتبويب الرئيسية" عند الضغط على إرسال في وكيل البحث.
# _TOTAL_BUDGET يفرض سقفاً زمنياً إجمالياً: بمجرد تجاوزه تتوقف الدالة عن
# استدعاء مصادر إضافية وتُعيد ما جمعته حتى تلك اللحظة بدل الاستمرار.
_TOTAL_BUDGET = 15.0
_UA = (
    "Mozilla/5.0 (compatible; NSMAgent/2.0; "
    "+https://github.com/aliahmed369000000-ai/Neural-Service-Mesh)"
)


def _fetch(url: str, timeout: int = _TIMEOUT) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _parse_lite_html(html_text: str, max_results: int) -> List[Dict[str, str]]:
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
        results.append({"title": title_clean, "url": url, "snippet": snippet, "source": "duckduckgo"})
        if len(results) >= max_results:
            break
    return results


def _search_duckduckgo_lite(query: str, max_results: int) -> List[Dict[str, str]]:
    q = urllib.parse.quote_plus(query)
    url = f"https://lite.duckduckgo.com/lite/?q={q}"
    return _parse_lite_html(_fetch(url), max_results)


def _search_instant_answer(query: str) -> List[Dict[str, str]]:
    q = urllib.parse.quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
    data = json.loads(_fetch(url))
    results: List[Dict[str, str]] = []
    abstract = (data.get("AbstractText") or "").strip()
    if abstract:
        results.append({
            "title": data.get("Heading") or query,
            "url": data.get("AbstractURL") or "",
            "snippet": abstract,
            "source": "ddg_instant",
        })
    for topic in data.get("RelatedTopics", [])[:6]:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({
                "title": (topic.get("Text") or "")[:100],
                "url": topic.get("FirstURL") or "",
                "snippet": topic.get("Text") or "",
                "source": "ddg_instant",
            })
        elif isinstance(topic, dict) and "Topics" in topic:
            for sub in topic.get("Topics", [])[:3]:
                if isinstance(sub, dict) and sub.get("Text"):
                    results.append({
                        "title": (sub.get("Text") or "")[:100],
                        "url": sub.get("FirstURL") or "",
                        "snippet": sub.get("Text") or "",
                        "source": "ddg_instant",
                    })
    return results


def _search_wikipedia(query: str, max_results: int, lang: str = "ar") -> List[Dict[str, str]]:
    q = urllib.parse.quote(query)
    search_url = (
        f"https://{lang}.wikipedia.org/w/api.php"
        f"?action=opensearch&search={q}&limit={max_results}&format=json"
    )
    data = json.loads(_fetch(search_url))
    titles: List[str] = data[1] if len(data) > 1 else []
    urls: List[str] = data[3] if len(data) > 3 else []
    results: List[Dict[str, str]] = []
    for i, title in enumerate(titles[:max_results]):
        snippet = ""
        try:
            summary_url = (
                f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
                f"{urllib.parse.quote(title)}"
            )
            summary = json.loads(_fetch(summary_url))
            snippet = (summary.get("extract") or "").strip()
        except Exception:
            pass
        results.append({
            "title": title,
            "url": urls[i] if i < len(urls) else "",
            "snippet": snippet,
            "source": f"wikipedia_{lang}",
        })
    return results


def _search_wikidata(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    q = urllib.parse.quote(query)
    url = (
        "https://www.wikidata.org/w/api.php"
        f"?action=wbsearchentities&search={q}&language=ar&uselang=ar"
        f"&format=json&limit={max_results}"
    )
    try:
        data = json.loads(_fetch(url))
    except Exception:
        url = (
            "https://www.wikidata.org/w/api.php"
            f"?action=wbsearchentities&search={q}&language=en&uselang=en"
            f"&format=json&limit={max_results}"
        )
        data = json.loads(_fetch(url))
    results: List[Dict[str, str]] = []
    for item in data.get("search", [])[:max_results]:
        results.append({
            "title": item.get("label") or query,
            "url": item.get("concepturi") or f"https://www.wikidata.org/wiki/{item.get('id', '')}",
            "snippet": item.get("description") or "",
            "source": "wikidata",
        })
    return results


def _search_news_rss(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """أخبار عبر Google News RSS (بدون مفتاح)."""
    q = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=ar&gl=SA&ceid=SA:ar"
    raw = _fetch(url)
    results: List[Dict[str, str]] = []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return results
    for item in root.findall(".//item")[:max_results]:
        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        title = (title_el.text or "").strip() if title_el is not None else ""
        link = (link_el.text or "").strip() if link_el is not None else ""
        desc = (desc_el.text or "").strip() if desc_el is not None else ""
        desc = re.sub(r"<[^>]+>", "", desc)
        if title:
            results.append({
                "title": title,
                "url": link,
                "snippet": desc[:300],
                "source": "google_news",
            })
    return results


def _search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    q = urllib.parse.quote_plus(query)
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query=all:{q}&start=0&max_results={max_results}"
    )
    raw = _fetch(url)
    results: List[Dict[str, str]] = []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return results
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("a:entry", ns)[:max_results]:
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        title = re.sub(r"\s+", " ", title)
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        summary = re.sub(r"\s+", " ", summary)[:280]
        link = ""
        for l in entry.findall("a:link", ns):
            if l.attrib.get("type") == "text/html" or l.attrib.get("rel") == "alternate":
                link = l.attrib.get("href", "")
                break
        if not link:
            link = entry.findtext("a:id", default="", namespaces=ns) or ""
        if title:
            results.append({
                "title": title,
                "url": link,
                "snippet": summary,
                "source": "arxiv",
            })
    return results


_TRENDS_NS = {"ht": "https://trends.google.com/trends/trendingsearches/daily"}


def _fetch_google_trends_rss(geo: str) -> List[Dict[str, str]]:
    url = f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={geo}"
    raw = _fetch(url)
    root = ET.fromstring(raw)
    results: List[Dict[str, str]] = []
    for item in root.findall(".//item"):
        title_el = item.find("title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        if not title:
            continue
        traffic_el = item.find("ht:approx_traffic", _TRENDS_NS)
        traffic = (traffic_el.text or "").strip() if traffic_el is not None else ""
        news_item = item.find("ht:news_item", _TRENDS_NS)
        news_title, news_url = "", ""
        if news_item is not None:
            nt = news_item.find("ht:news_item_title", _TRENDS_NS)
            nu = news_item.find("ht:news_item_url", _TRENDS_NS)
            news_title = (nt.text or "").strip() if nt is not None else ""
            news_url = (nu.text or "").strip() if nu is not None else ""
        results.append({
            "title": title,
            "traffic": traffic,
            "news_title": news_title,
            "news_url": news_url,
            "source": "google_trends",
        })
    return results


def get_trending_topics(geo: str = "SA", max_results: int = 10) -> List[Dict[str, str]]:
    geo = (geo or "SA").upper()
    try:
        items = _fetch_google_trends_rss(geo)
        return items[:max_results]
    except Exception:
        for fallback in ("EG", "AE", "US"):
            try:
                items = _fetch_google_trends_rss(fallback)
                return items[:max_results]
            except Exception:
                continue
    return []


def search_news(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    query = (query or "").strip()
    if not query:
        return []
    try:
        return _search_news_rss(query, max_results)
    except Exception:
        return []


def _dedupe(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for r in results:
        key = (r.get("url") or r.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def web_search_structured(
    query: str,
    max_results: int = 8,
    include_news: bool = True,
    include_wiki: bool = True,
    include_arxiv: bool = False,
) -> Dict[str, Any]:
    """نتائج منظمة للوكلاء — لا تختلق نتائج."""
    query = (query or "").strip()
    if not query:
        return {"ok": False, "query": query, "results": [], "msg": "query فارغ"}
    if is_offline():
        return {"ok": False, "query": query, "results": [], "msg": offline_message("بحث الويب")}

    max_results = max(1, min(int(max_results or 8), 15))
    errors: List[str] = []
    collected: List[Dict[str, str]] = []

    # 🛠️ سقف زمني إجمالي (انظر تعليق _TOTAL_BUDGET أعلى الملف): بمجرد
    # تجاوزه نتوقف عن استدعاء أي مصدر إضافي بدل إبقاء الطلب معلّقاً
    # لعشرات الثواني (كان هذا يسبب انقطاع WebSocket في Streamlit وإعادة
    # تحميل التطبيق بالكامل — يظهر للمستخدم كعودة مفاجئة لتبويب الرئيسية).
    _start = time.monotonic()

    def _budget_left() -> bool:
        return (time.monotonic() - _start) < _TOTAL_BUDGET

    _sources = [
        ("duckduckgo", lambda: _search_duckduckgo_lite(query, max_results)),
        ("ddg_instant", lambda: _search_instant_answer(query)),
    ]
    if include_wiki:
        _sources.append(("wikipedia_ar", lambda: _search_wikipedia(query, min(5, max_results), lang="ar")))
        _sources.append(("wikipedia_en", lambda: _search_wikipedia(query, min(5, max_results), lang="en")))
        _sources.append(("wikidata", lambda: _search_wikidata(query, 4)))
    if include_news:
        _sources.append(("news", lambda: _search_news_rss(query, min(5, max_results))))
    # arXiv تلقائياً لأسئلة علمية أو بطلب صريح
    sci_hints = ("arxiv", "بحث علمي", "ورقة", "paper", "neural", "transformer", "llm", "algorithm")
    if include_arxiv or any(h in query.lower() for h in sci_hints):
        _sources.append(("arxiv", lambda: _search_arxiv(query, 4)))

    for name, fn in _sources:
        if not _budget_left():
            errors.append(f"{name}: تخطّي — تجاوز السقف الزمني الإجمالي ({_TOTAL_BUDGET:.0f}s)")
            continue
        try:
            collected.extend(fn())
        except Exception as e:
            errors.append(f"{name}: {e}")

    results = _dedupe(collected)[:max_results]
    return {
        "ok": bool(results),
        "query": query,
        "count": len(results),
        "results": results,
        "errors": errors[:8] if not results else [],
        "msg": "OK" if results else (" | ".join(errors) if errors else "لا نتائج"),
    }


def deep_research(query: str, max_per_angle: int = 4) -> Dict[str, Any]:
    """
    بحث عميق: يقسم السؤال لعدة زوايا (تعريف، أخبار، خلفية، تطبيقات)
    ويجمع النتائج — مناسب لتغذية التعلّم الذاتي.
    """
    query = (query or "").strip()
    if not query:
        return {"ok": False, "msg": "query فارغ", "angles": {}}
    if is_offline():
        return {"ok": False, "msg": offline_message("بحث عميق"), "angles": {}}

    angles = {
        "تعريف": f"{query} تعريف شرح",
        "أخبار": f"{query} أخبار أحدث",
        "خلفية": f"{query} تاريخ خلفية",
        "تطبيقات": f"{query} استخدامات تطبيقات",
    }
    # إنجليزي إضافي للموضوعات التقنية
    if re.search(r"[A-Za-z]{3,}", query):
        angles["english"] = query

    angle_results: Dict[str, Any] = {}
    all_hits: List[Dict[str, str]] = []
    for name, q in angles.items():
        res = web_search_structured(
            q,
            max_results=max_per_angle,
            include_news=(name == "أخبار"),
            include_wiki=True,
            include_arxiv=("neural" in query.lower() or "ai" in query.lower() or name == "english"),
        )
        angle_results[name] = res
        all_hits.extend(res.get("results") or [])

    merged = _dedupe(all_hits)[:20]
    return {
        "ok": bool(merged),
        "query": query,
        "angles": {k: {"count": v.get("count", 0), "ok": v.get("ok")} for k, v in angle_results.items()},
        "results": merged,
        "count": len(merged),
        "msg": "OK" if merged else "تعذّر جمع نتائج كافية",
    }


def _format_results(query: str, results: List[Dict[str, str]], source: str) -> str:
    lines = [f"🔍 نتائج البحث عن: **{query}** (المصدر: {source})\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r.get('title', '')}**")
        if r.get("url"):
            lines.append(f"   {r['url']}")
        if r.get("snippet"):
            snippet = r["snippet"]
            if len(snippet) > 240:
                snippet = snippet[:240] + "..."
            lines.append(f"   {snippet}")
        if r.get("source"):
            lines.append(f"   _(source: {r['source']})_")
        lines.append("")
    return "\n".join(lines).strip()


def web_search(query: str, max_results: int = 5) -> str:
    """واجهة متوافقة مع السابق — نص منسّق."""
    query = (query or "").strip()
    if not query:
        return "❌ web_search: مطلوب query (نص البحث)"
    if is_offline():
        return offline_message("بحث الويب")

    max_results = max(1, min(int(max_results or 5), 12))
    structured = web_search_structured(query, max_results=max_results, include_news=True)
    if structured.get("ok") and structured.get("results"):
        sources = sorted({r.get("source", "?") for r in structured["results"]})
        return _format_results(query, structured["results"], source="+".join(sources))
    return f"❌ فشل البحث عن '{query}': {structured.get('msg', 'لا نتائج')}"


def format_trending(geo: str = "SA", max_results: int = 10) -> str:
    items = get_trending_topics(geo=geo, max_results=max_results)
    if not items:
        return f"❌ تعذّر جلب المواضيع الرائجة لـ {geo}"
    lines = [f"📈 المواضيع الرائجة ({geo}):\n"]
    for i, t in enumerate(items, 1):
        traffic = f" — {t.get('traffic')}" if t.get("traffic") else ""
        lines.append(f"{i}. **{t.get('title', '')}**{traffic}")
        if t.get("news_title"):
            lines.append(f"   {t['news_title']}")
        if t.get("news_url"):
            lines.append(f"   {t['news_url']}")
        lines.append("")
    return "\n".join(lines).strip()
