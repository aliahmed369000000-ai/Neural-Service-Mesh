"""
NSM Web Search Tool — ai/web_search_tool.py
=============================================
أداة بحث حقيقية في الإنترنت بدون الحاجة لأي مفتاح API.
مصدر واحد مشترك يُستخدم من:
  - ai/nsm_agent_core.py   (action: "web_search" داخل حلقة الوكيل الرئيسية)
  - ai/code_agent.py       (أمر: "ابحث <نص>" في nsm_chat.py)

الإستراتيجية (بالترتيب، بدون مفتاح API لأي منها):
  1) DuckDuckGo HTML Lite            → نتائج ويب حقيقية (عنوان + رابط + مقتطف)
  2) Wikipedia (عربي ثم إنجليزي)     → 🆕 موثوق جداً من سيرفرات الاستضافة
     السحابية (DuckDuckGo أحياناً يحظر/يبطئ IPs الاستضافة السحابية) —
     ممتاز لأسئلة الحقائق المباشرة (أشخاص، دول، مفاهيم، تواريخ)
  3) DuckDuckGo Instant Answer API   → احتياطي أخير JSON رسمي

⚠️ ملاحظة صادقة: الخيار (1) يعتمد على هيكل صفحة DuckDuckGo الحالية.
لو غيّروا تصميم الصفحة مستقبلاً، _parse_lite_html() قد تحتاج تحديث بسيط
(هذا حال أي scraping بدون مفتاح API — لا يوجد طريقة مجانية مضمونة للأبد).
الدالة لا تُرجع أبداً "نجاح" وهمي: لو فشلت كل المصادر تُرجع رسالة خطأ صريحة.
"""
from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List

from ai.offline_mode import is_offline, offline_message

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


def _search_wikipedia(query: str, max_results: int, lang: str = "ar") -> List[Dict[str, str]]:
    """
    🆕 بحث عبر Wikipedia REST API (opensearch + summary).
    موثوق جداً من سيرفرات الاستضافة السحابية (غالباً لا يحظر IPs
    بعكس DuckDuckGo HTML scraping) — ممتاز لأسئلة الحقائق المباشرة
    (أشخاص، دول، مفاهيم). يُجرَّب العربي أولاً، ثم الإنجليزي احتياطياً.
    """
    q = urllib.parse.quote(query)
    search_url = (
        f"https://{lang}.wikipedia.org/w/api.php"
        f"?action=opensearch&search={q}&limit={max_results}&format=json"
    )
    raw = _fetch(search_url)
    data = json.loads(raw)
    titles: List[str] = data[1] if len(data) > 1 else []
    urls:   List[str] = data[3] if len(data) > 3 else []

    results: List[Dict[str, str]] = []
    for i, title in enumerate(titles[:max_results]):
        snippet = ""
        try:
            summary_url = (
                f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
                f"{urllib.parse.quote(title)}"
            )
            summary_raw = _fetch(summary_url)
            summary = json.loads(summary_raw)
            snippet = (summary.get("extract") or "").strip()
        except Exception:
            pass
        results.append({
            "title": title,
            "url": urls[i] if i < len(urls) else "",
            "snippet": snippet,
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

    if is_offline():
        return offline_message("بحث الويب")

    max_results = max(1, min(int(max_results or 5), 10))
    errors: List[str] = []

    try:
        results = _search_duckduckgo_lite(query, max_results)
        if results:
            return _format_results(query, results, source="DuckDuckGo")
    except Exception as e:
        errors.append(f"DuckDuckGo HTML: {e}")

    # 🆕 Wikipedia (عربي ثم إنجليزي) — بديل موثوق قبل الاستسلام،
    # خصوصاً لأسئلة الحقائق المباشرة (أشخاص/دول/مفاهيم) التي غالباً
    # ما تُطلَب هنا، وأقل عرضة لحظر IPs الاستضافة السحابية.
    for lang in ("ar", "en"):
        try:
            results = _search_wikipedia(query, max_results, lang=lang)
            if results:
                return _format_results(query, results, source=f"Wikipedia ({lang})")
        except Exception as e:
            errors.append(f"Wikipedia ({lang}): {e}")

    try:
        results = _search_instant_answer(query)
        if results:
            return _format_results(query, results[:max_results], source="DuckDuckGo Instant Answer")
    except Exception as e:
        errors.append(f"Instant Answer API: {e}")

    detail = " | ".join(errors) if errors else "لا نتائج مطابقة"
    return f"❌ فشل البحث عن '{query}': {detail}"


# ═════════════════════════════════════════════════════════════════════════════
# المواضيع الرائجة (Trending Topics) — لوكيل صناعة المحتوى
# ═════════════════════════════════════════════════════════════════════════════
# مصدر حقيقي بدون مفتاح API: Google Trends RSS (يومي، حسب الدولة).
# نفس فلسفة web_search أعلاه: لا نتيجة وهمية أبداً — فشل المصدر = رسالة
# خطأ صريحة، وليس بيانات ملفّقة.

_TRENDS_NS = {"ht": "https://trends.google.com/trends/trendingsearches/daily"}


def _fetch_google_trends_rss(geo: str) -> List[Dict[str, str]]:
    """يجلب المواضيع الرائجة اليوم من Google Trends RSS لدولة معيّنة."""
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
        })
    return results


def get_trending_topics(geo: str = "SA", max_results: int = 10) -> List[Dict[str, str]]:
    """
    يُرجع المواضيع الرائجة فعلياً اليوم كبيانات مُهيكلة (للاستخدام البرمجي
    من وكلاء آخرين، مثل وكيل كتابة المقالات القادم).
    geo: رمز الدولة (SA، EG، AE، ...). القيمة الافتراضية SA.
    يرفع Exception صراحة عند الفشل — لا يُرجع قائمة فارغة صامتة ولا بيانات
    ملفّقة، حتى يستطيع المستدعي (البرمجي) التمييز بين "لا نتائج" و"فشل الجلب".
    """
    geo = (geo or "SA").strip().upper() or "SA"
    max_results = max(1, min(int(max_results or 10), 20))
    if is_offline():
        raise Exception(offline_message("المواضيع الرائجة"))
    results = _fetch_google_trends_rss(geo)
    return results[:max_results]


def trending_topics_report(geo: str = "SA", max_results: int = 10) -> str:
    """
    نفس get_trending_topics لكن بصيغة نص منسّق جاهز للعرض المباشر
    (مطابقة لأسلوب web_search أعلاه) — تُستخدم من واجهة المحادثة/الوكلاء.
    """
    geo = (geo or "SA").strip().upper() or "SA"
    try:
        results = get_trending_topics(geo, max_results)
    except Exception as e:
        return f"❌ فشل جلب المواضيع الرائجة لـ '{geo}': {e}"

    if not results:
        return f"❌ لا توجد مواضيع رائجة متاحة حالياً للمنطقة '{geo}'"

    lines = [f"📈 المواضيع الرائجة الآن ({geo}) — المصدر: Google Trends\n"]
    for i, r in enumerate(results, 1):
        traffic_part = f" (بحث تقريبي: {r['traffic']})" if r.get("traffic") else ""
        lines.append(f"{i}. **{r['title']}**{traffic_part}")
        if r.get("news_title"):
            lines.append(f"   خبر ذو صلة: {r['news_title']}")
        if r.get("news_url"):
            lines.append(f"   {r['news_url']}")
        lines.append("")
    return "\n".join(lines).strip()
