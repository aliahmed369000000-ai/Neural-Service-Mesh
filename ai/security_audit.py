# -*- coding: utf-8 -*-
import os
import re
import subprocess
import json
from typing import List, Dict, Any

class SovereignSecurityAudit:
    """
    وحدة البحث الأمني السيادي (Sovereign Security Audit):
    تمكن الوكلاء من استكشاف الثغرات الأمنية في الأكواد والمنصات بشكل مستقل.
    """
    def __init__(self):
        self.vuln_patterns = {
            "Command Injection": [r"os\.system\(", r"subprocess\.run\(.*shell=True", r"eval\("],
            "Secret Leak": [r"ghp_[a-zA-Z0-9]{36}", r"hf_[a-zA-Z0-9]{34}", r"sk-[a-zA-Z0-9]{48}"],
            "Path Traversal": [r"open\(.*\.format\(", r"os\.path\.join\(.*\.\./"],
            "Insecure Auth": [r"password\s*=\s*['\"].*['\"]", r"verify=False"],
            "HF Model RCE": [r"torch\.load\(", r"pickle\.load\(", r"weights_only=False"],
            "HF Space Secrets": [r"st\.secrets", r"os\.environ\.get\(.*TOKEN"],
            "Gradio Insecurity": [r"gr\.Interface\(.*share=True", r"enable_queue=True"],
            "Social API IDOR": [r"api/v1/user/\d+", r"get_profile\?id=", r"settings/update\?uid="],
            "OAuth Vulnerability": [r"redirect_uri=", r"state=.*", r"response_type=token"],
            "Meta Specific": [r"fb_access_token", r"graph\.facebook\.com", r"fbid=\d+"],
            "Instagram Recon": [r"i\.instagram\.com", r"instagram\.com/p/", r"stories/highlights/"],
            "Private Media Leak": [r"cdninstagram\.com", r"display_url", r"is_private\s*:\s*false"],
            "Direct Message Auth": [r"direct_v2/threads/", r"messages/send/", r"share/item/"],
            "TikTok Specific": [r"v16\.tiktokv\.com", r"tiktok\.com/@", r"aweme/v1/feed/"],
            "Private Video Leak": [r"item/detail/", r"video_control", r"is_private\s*:\s*1"],
            "Webview Bridge": [r"TiktokJSBridge", r"bytedance\.on\(", r"invokeMethod"]
        }

    def audit_local_code(self, directory: str = ".") -> Dict[str, Any]:
        """فحص الكود المحلي بحثاً عن ثغرات برمجية شائعة."""
        results = {}
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    try:
                        content = open(path, 'r', encoding='utf-8').read()
                        file_vulns = []
                        for vuln_type, patterns in self.vuln_patterns.items():
                            for pattern in patterns:
                                if re.search(pattern, content):
                                    file_vulns.append(vuln_type)
                        if file_vulns:
                            results[path] = list(set(file_vulns))
                    except Exception:
                        continue
        return results

    def research_vulnerabilities(self, topic: str) -> str:
        """البحث في قواعد البيانات الأمنية والأخبار عن ثغرات معينة."""
        try:
            from ai.web_gateway import NeuralWebGateway
            gw = NeuralWebGateway()
            # توسيع نطاق البحث ليشمل أخبار الأمن ومنصات المكافآت
            queries = [
                f"site:cve.mitre.org {topic} vulnerabilities 2026",
                f"{topic} security bug bounty reports",
                f"{topic} remote code execution exploit 2026"
            ]
            all_results = []
            for q in queries:
                res = gw.search(q)
                if isinstance(res, list):
                    all_results.extend(res)
            return json.dumps(all_results, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"❌ Security research failed: {e}"

    def generate_security_report(self, objective: str) -> str:
        """توليد تقرير أمني سيادي حول منصة أو كود معين."""
        print(f"🛡️ Starting Sovereign Security Audit for: {objective}")
        
        # 1. فحص الكود إذا كان الهدف محلياً
        local_audit = {}
        if objective == "local_project" or objective == ".":
            local_audit = self.audit_local_code()
            
        # 2. بحث خارجي
        research_data = self.research_vulnerabilities(objective)
        
        report = f"""
# 🛡️ تقرير البحث الأمني السيادي (NSM Sovereign Security Report)
**الهدف:** {objective}

## 1. نتائج التدقيق المحلي
{json.dumps(local_audit, indent=2) if local_audit else "لم يتم العثور على ثغرات محلية واضحة أو لم يُطلب تدقيق محلي."}

## 2. نتائج البحث الاستكشافي
{research_data}

## 3. التوصيات الأمنية
- تدوير كافة الأسرار المكتشفة فوراً.
- تجنب استخدام `shell=True` في العمليات الفرعية.
- تطبيق مبدأ "الأقل صلاحية" (Least Privilege) على كافة الوكلاء.
"""
        return report

if __name__ == "__main__":
    auditor = SovereignSecurityAudit()
    print(auditor.generate_security_report("GitHub API"))
