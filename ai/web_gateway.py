# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import json
import logging
import time

class UniversalWebGateway:
    """بوابة كونية تسمح للوكلاء بالتفاعل الذكي مع أي موقع أو API دون برمجة مسبقة."""
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.session = requests.Session()
        self.logger = logging.getLogger("NSM-UniversalGateway")

    def interact(self, url: str, action: str = "GET", params: dict = None, data: dict = None, headers: dict = None):
        """تفاعل ديناميكي مع أي URL."""
        try:
            full_headers = self.headers.copy()
            if headers:
                full_headers.update(headers)
            
            response = self.session.request(
                method=action,
                url=url,
                params=params,
                json=data if action in ["POST", "PUT", "PATCH"] else None,
                headers=full_headers,
                timeout=15
            )
            
            return {
                "status": response.status_code,
                "content_type": response.headers.get("Content-Type", ""),
                "data": self._parse_content(response),
                "url": response.url
            }
        except Exception as e:
            return {"error": str(e), "url": url}

    def _parse_content(self, response):
        """تحليل المحتوى بذكاء بناءً على نوعه."""
        content_type = response.headers.get("Content-Type", "").lower()
        
        if "json" in content_type:
            return response.json()
        
        if "html" in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            # استخلاص النصوص الهامة والروابط والجداول
            return {
                "text": soup.get_text(separator=' ', strip=True)[:5000],
                "links": [a.get('href') for a in soup.find_all('a', href=True)][:20],
                "tables": [str(table) for table in soup.find_all('table')][:5]
            }
        
        return response.text[:2000]

    def discover_api(self, domain: str):
        """محاولة اكتشاف نقاط نهاية الـ API لمنصة معينة."""
        common_paths = ["/api", "/v1", "/graphql", "/swagger.json", "/api/v1"]
        results = {}
        for path in common_paths:
            url = f"https://{domain.strip('/')}{path}"
            res = self.interact(url)
            if res.get("status") in [200, 201]:
                results[path] = "Accessible"
        return results

# التوافق مع الكود القديم
class NeuralWebGateway(UniversalWebGateway):
    def search(self, query: str, num_results: int = 5):
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        res = self.interact(search_url)
        if "data" in res and isinstance(res["data"], dict) and "text" in res["data"]:
            # محاكاة بسيطة للبحث القديم
            return [{"title": "Search Result", "snippet": res["data"]["text"][:200]}]
        return []
