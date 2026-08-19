# -*- coding: utf-8 -*-
"""
agent_web_scraping.py — تفاعل الوكلاء مع مواقع خارجية

يتيح للوكلاء:
1. جلب محتوى صفحات الويب (HTML → text)
2. استخراج روابط وصور من صفحة
3. الاتصال بـ REST APIs
4. قراءة JSON/XML
5. البحث في الإنترنت (عبر Google Custom Search أو DuckDuckGo)

لا يحتاج مكتبات خارجية — يعمل بـ Python stdlib فقط.
"""
import json
import re
import html
import urllib.request
import urllib.parse
from typing import Optional


# ── helpers ─────────────────────────────────────────────────────────────────
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}


def fetch_page(url: str, timeout: int = 30) -> str:
    """جلب محتوى صفحة HTML."""
    req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def html_to_text(html_content: str) -> str:
    """تحويل HTML إلى نص نظيف."""
    # إزالة scripts و styles
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html_content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    # إزالة tags
    text = re.sub(r"<[^>]+>", " ", text)
    # فك entities
    text = html.unescape(text)
    # إزالة whitespace زائد
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_links(html_content: str, base_url: str = None) -> list:
    """استخراج كل الروابط من صفحة."""
    pattern = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
    links = pattern.findall(html_content)
    results = []
    seen = set()
    for link in links:
        link = link.strip()
        if not link or link.startswith(("#", "javascript:", "mailto:")):
            continue
        if base_url and link.startswith("/"):
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            link = f"{parsed.scheme}://{parsed.netloc}{link}"
        elif base_url and not link.startswith(("http://", "https://")):
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            link = f"{parsed.scheme}://{parsed.netloc}/{link}"
        if link not in seen:
            seen.add(link)
            results.append(link)
    return results


def extract_images(html_content: str, base_url: str = None) -> list:
    """استخراج كل الصور من صفحة."""
    pattern = re.compile(r'src=["\']([^"\']+\.(?:png|jpe?g|gif|webp|svg))["\']', re.IGNORECASE)
    imgs = pattern.findall(html_content)
    results = []
    seen = set()
    for img in imgs:
        if img not in seen:
            seen.add(img)
            if base_url and img.startswith("/"):
                from urllib.parse import urlparse
                parsed = urlparse(base_url)
                img = f"{parsed.scheme}://{parsed.netloc}{img}"
            results.append(img)
    return results


def fetch_json(url: str, params: dict = None, timeout: int = 30) -> dict:
    """جلب JSON من API."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={**DEFAULT_HEADERS, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def search_duckduckgo(query: str, max_results: int = 10) -> list:
    """بحث عبر DuckDuckGo HTML (بدون API key)."""
    params = {"q": query, "kl": "wt-wt"}
    url = f"https://html.duckduckgo.com/html/?{urllib.parse.urlencode(params)}"
    try:
        html_content = fetch_page(url)
        # استخراج النتائج
        results = []
        # pattern لـ DuckDuckGo results
        pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL
        )
        for match in pattern.finditer(html_content):
            link = html.unescape(match.group(1))
            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            # فك t=redirect
            if "uddg=" in link:
                from urllib.parse import parse_qs, urlparse
                qs = parse_qs(urlparse(link).query)
                if "uddg" in qs:
                    link = qs["uddg"][0]
            results.append({"title": title, "url": link})
            if len(results) >= max_results:
                break
        return results
    except Exception:
        return []


class WebAgent:
    """وكيل الويب — يتفاعل مع المواقع الخارجية."""

    def __init__(self):
        pass

    def fetch(self, url: str, as_text: bool = True) -> dict:
        """جلب صفحة ويب — يرجع {'url', 'status', 'content'}."""
        try:
            raw = fetch_page(url)
            content = html_to_text(raw) if as_text else raw
            return {"url": url, "status": "ok", "content": content[:10000]}
        except Exception as e:
            return {"url": url, "status": "error", "error": str(e)}

    def extract_links(self, url: str) -> dict:
        """استخراج الروابط من صفحة."""
        try:
            raw = fetch_page(url)
            links = extract_links(raw, base_url=url)
            return {"url": url, "links": links[:50]}
        except Exception as e:
            return {"url": url, "error": str(e)}

    def extract_images(self, url: str) -> dict:
        """استخراج الصور من صفحة."""
        try:
            raw = fetch_page(url)
            imgs = extract_images(raw, base_url=url)
            return {"url": url, "images": imgs[:30]}
        except Exception as e:
            return {"url": url, "error": str(e)}

    def api_get(self, url: str, params: dict = None) -> dict:
        """GET request لأي API."""
        try:
            data = fetch_json(url, params=params)
            return {"status": "ok", "data": data}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def search(self, query: str, max_results: int = 10) -> dict:
        """بحث في الإنترنت."""
        results = search_duckduckgo(query, max_results=max_results)
        return {"query": query, "results": results}

    def get_headlines(self) -> dict:
        """جلب عناوين أخبار سريعة (Hacker News API)."""
        try:
            data = fetch_json("https://hacker-news.firebaseio.com/v0/topstories.json")
            ids = data[:10] if isinstance(data, list) else []
            headlines = []
            for i in ids:
                try:
                    story = fetch_json(f"https://hacker-news.firebaseio.com/v0/item/{i}.json")
                    if story and "title" in story:
                        headlines.append({"title": story["title"], "url": story.get("url", "")})
                except Exception:
                    pass
            return {"headlines": headlines}
        except Exception as e:
            return {"error": str(e)}
