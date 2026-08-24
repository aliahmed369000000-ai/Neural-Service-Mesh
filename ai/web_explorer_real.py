
"""
ai/web_explorer_real.py
=======================
🆕 أداة تصفح الويب والبحث المعمق الحقيقية.
تستخدم BeautifulSoup و Requests لاستخراج المعلومات بدلاً من المحاكاة.
"""
import requests
import json
from bs4 import BeautifulSoup
from typing import Dict, Any

def web_explorer_real(params: Dict[str, Any]) -> str:
    """أداة بحث وتصفح حقيقية."""
    query = params.get("query", "")
    url = params.get("url", "")
    
    if not query and not url:
        return "❌ web_explorer: يجب توفير query للبحث أو url للتصفح."
        
    try:
        if url:
            # تصفح صفحة محددة
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            # استخراج النص النظيف
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator=' ', strip=True)
            return json.dumps({
                "url": url,
                "title": soup.title.string if soup.title else "بدون عنوان",
                "content": text[:5000] # تحديد الحجم
            }, ensure_ascii=False)
        else:
            # محاكاة البحث عبر DuckDuckGo (كحل بسيط ومجاني)
            search_url = f"https://html.duckduckgo.com/html/?q={query}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(search_url, headers=headers, timeout=15)
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            results = []
            for result in soup.find_all('a', class_='result__a')[:5]:
                results.append({
                    "title": result.get_text(),
                    "url": result['href']
                })
            return json.dumps({"query": query, "results": results}, ensure_ascii=False)
            
    except Exception as e:
        return f"❌ web_explorer Error: {e}"
