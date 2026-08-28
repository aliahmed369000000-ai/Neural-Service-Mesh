# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import json
import logging

class NeuralWebGateway:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.logger = logging.getLogger("NSM-WebGateway")

    def search(self, query: str, num_results: int = 5):
        """بحث سيادي في الإنترنت لاستخلاص المعرفة البحثية."""
        # محاولة البحث عبر DuckDuckGo كخيار أول
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        try:
            response = requests.get(search_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = []
                for item in soup.find_all('div', class_='result', limit=num_results):
                    link = item.find('a', class_='result__a')
                    snippet = item.find('a', class_='result__snippet')
                    if link:
                        results.append({
                            "title": link.text.strip(),
                            "url": link.get('href'),
                            "snippet": snippet.text.strip() if snippet else ""
                        })
                
                # إضافة بحث تخصصي في arXiv للأوراق البحثية إذا كان الاستعلام تقنياً
                if any(k in query.lower() for k in ["research", "paper", "arxiv", "ai", "deep learning"]):
                    arxiv_results = self._search_arxiv(query, limit=3)
                    results.extend(arxiv_results)
                    
                return results
            return []
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def _search_arxiv(self, query: str, limit: int = 3):
        """بحث متخصص في arXiv للأوراق البحثية."""
        arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={limit}"
        try:
            response = requests.get(arxiv_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                papers = []
                for entry in soup.find_all('entry'):
                    papers.append({
                        "title": entry.title.text.strip(),
                        "url": entry.id.text.strip(),
                        "summary": entry.summary.text.strip()[:200] + "...",
                        "source": "arXiv"
                    })
                return papers
            return []
        except Exception:
            return []
