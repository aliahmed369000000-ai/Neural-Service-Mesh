# -*- coding: utf-8 -*-
import re
import json

class NFTSecurityAudit:
    """وحدة البحث الأمني المتخصصة في الـ NFT ومنصات التداول مثل OpenSea."""
    
    def __init__(self):
        self.nft_patterns = {
            "unsafe_minting": r"function\s+mint\(.*\)\s+public",  # Minting بدون قيود
            "metadata_freeze_lack": r"setTokenURI\(.*\)",  # إمكانية تغيير الرابط بعد السك
            "royalty_bypass": r"transferFrom\(.*\)",  # تجاوز منطق العمولات
            "opensea_proxy_vuln": r"isApprovedForAll\(.*\)",  # سوء استخدام تفويض OpenSea
            "lazy_minting_exploit": r"signature\s+verification" # ثغرات التحقق من التوقيع في Lazy Minting
        }

    def audit_nft_contract(self, code: str):
        """فحص كود عقود الـ NFT بحثاً عن ثغرات شائعة."""
        findings = []
        for vuln, pattern in self.nft_patterns.items():
            matches = re.finditer(pattern, code)
            for match in matches:
                findings.append({
                    "vulnerability": vuln,
                    "line": code.count('\n', 0, match.start()) + 1,
                    "snippet": match.group(),
                    "severity": "High" if vuln in ["unsafe_minting", "lazy_minting_exploit"] else "Medium"
                })
        return findings

    def research_nft_bounties(self, platform_name: str):
        """البحث عن برامج مكافآت الـ NFT لمنصة معينة."""
        bounty_info = {
            "OpenSea": {"platform": "HackerOne", "max_bounty": "$25,000", "scope": "Web, API, Smart Contracts"},
            "Rarible": {"platform": "HackerOne", "max_bounty": "$5,000", "scope": "Web, API"},
            "Blur": {"platform": "Immunefi", "max_bounty": "$50,000", "scope": "Smart Contracts, Trading Logic"}
        }
        return bounty_info.get(platform_name, {"platform": "Unknown", "max_bounty": "N/A", "scope": "General"})

    def generate_nft_report(self, target: str, findings: list):
        """توليد تقرير أمني احترافي للـ NFT."""
        report = f"# NFT Security Audit Report: {target}\n\n"
        if not findings:
            report += "✅ No immediate NFT-specific vulnerabilities found in static analysis.\n"
        else:
            report += "## ⚠️ Critical NFT Findings\n\n"
            for f in findings:
                report += f"### [{f['severity']}] {f['vulnerability']}\n"
                report += f"- **Location**: Line {f['line']}\n"
                report += f"- **Snippet**: `{f['snippet']}`\n"
                report += f"- **Risk**: Potential for unauthorized minting or metadata manipulation.\n\n"
        
        bounty = self.research_nft_bounties(target)
        report += f"## 💰 NFT Bounty Information\n"
        report += f"- **Platform**: {bounty['platform']}\n"
        report += f"- **Potential Reward**: {bounty['max_bounty']}\n"
        report += f"- **Official Scope**: {bounty['scope']}\n"
        
        return report
