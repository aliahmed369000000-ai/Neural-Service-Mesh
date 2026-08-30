# -*- coding: utf-8 -*-
import re
import json

class CryptoSecurityAudit:
    """وحدة البحث الأمني المتخصصة في العملات المشفرة والـ DeFi."""
    
    def __init__(self):
        self.vuln_patterns = {
            "reentrancy": r"\.call\{value:.*\}\(\"\"\)",  # نمط محتمل لهجمات Reentrancy في Solidity
            "integer_overflow": r"\+=",  # قديم في Solidity < 0.8
            "oracle_manipulation": r"getReserves\(.*\)",  # نمط التفاعل مع Uniswap oracles
            "insecure_api_key": r"(binance|coinbase|kucoin)_api_key\s*=\s*['\"][a-zA-Z0-9]{32,}['\"]",
            "private_key_leak": r"0x[a-fA-F0-9]{64}"
        }

    def audit_smart_contract(self, code: str):
        """فحص كود العقود الذكية بحثاً عن ثغرات شائعة."""
        findings = []
        for vuln, pattern in self.vuln_patterns.items():
            matches = re.finditer(pattern, code)
            for match in matches:
                findings.append({
                    "vulnerability": vuln,
                    "line": code.count('\n', 0, match.start()) + 1,
                    "snippet": match.group(),
                    "severity": "High" if vuln in ["reentrancy", "private_key_leak"] else "Medium"
                })
        return findings

    def research_crypto_bounties(self, platform_name: str):
        """البحث عن برامج مكافآت الكريبتو لمنصة معينة."""
        # محاكاة البحث في Immunefi/HackerOne
        bounty_info = {
            "Binance": {"platform": "HackerOne", "max_bounty": "$100,000", "scope": "API, Web, Mobile"},
            "Coinbase": {"platform": "HackerOne", "max_bounty": "$250,000", "scope": "Blockchain, Wallets, API"},
            "Uniswap": {"platform": "Immunefi", "max_bounty": "$2,250,000", "scope": "Smart Contracts"}
        }
        return bounty_info.get(platform_name, {"platform": "Unknown", "max_bounty": "N/A", "scope": "General"})

    def generate_crypto_report(self, target: str, findings: list):
        """توليد تقرير أمني احترافي للكريبتو."""
        report = f"# Crypto Security Audit Report: {target}\n\n"
        if not findings:
            report += "✅ No immediate critical vulnerabilities found in static analysis.\n"
        else:
            report += "## ⚠️ Critical Findings\n\n"
            for f in findings:
                report += f"### [{f['severity']}] {f['vulnerability']}\n"
                report += f"- **Location**: Line {f['line']}\n"
                report += f"- **Snippet**: `{f['snippet']}`\n"
                report += f"- **Risk**: High potential for asset loss or unauthorized access.\n\n"
        
        bounty = self.research_crypto_bounties(target)
        report += f"## 💰 Bounty Information\n"
        report += f"- **Platform**: {bounty['platform']}\n"
        report += f"- **Potential Reward**: {bounty['max_bounty']}\n"
        report += f"- **Official Scope**: {bounty['scope']}\n"
        
        return report
