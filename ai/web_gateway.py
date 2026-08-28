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
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        try:
            response = requests.get(search_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = []
                for link in soup.find_all('a', class_='result__a', limit=num_results):
                    results.append({
                        "title": link.text,
                        "url": link.get('href')
                    })
                return results
            return []
        except Exception as e:
            return []
