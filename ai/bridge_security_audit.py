# -*- coding: utf-8 -*-
import re
import json

class BridgeSecurityAudit:
    """وحدة البحث الأمني المتخصصة في جسور النقل عبر السلاسل (Cross-Chain Bridges)."""
    
    def __init__(self):
        self.bridge_patterns = {
            "weak_signature_check": r"ecrecover\(.*\)",  # فحص توقيع قد يكون ضعيفاً
            "replay_vulnerability": r"nonces\[.*\]",  # التحقق من وجود نظام النونص لمنع الإعادة
            "lock_mint_mismatch": r"lock\(.*\).*mint\(.*\)",  # منطق القفل والسك المتزامن
            "validator_threshold_low": r"threshold\s*<\s*quorum",  # عتبة تصويت الموثقين
            "bridge_vault_access": r"withdraw\(.*\)\s+public" # الوصول لخزائن الجسر
        }

    def audit_bridge_contract(self, code: str):
        """فحص كود عقود الجسور بحثاً عن ثغرات شائعة."""
        findings = []
        for vuln, pattern in self.bridge_patterns.items():
            matches = re.finditer(pattern, code)
            for match in matches:
                findings.append({
                    "vulnerability": vuln,
                    "line": code.count('\n', 0, match.start()) + 1,
                    "snippet": match.group(),
                    "severity": "Critical" if vuln in ["weak_signature_check", "bridge_vault_access"] else "High"
                })
        return findings

    def research_bridge_bounties(self, bridge_name: str):
        """البحث عن برامج مكافآت الجسور الكبرى."""
        bounty_info = {
            "Wormhole": {"platform": "Immunefi", "max_bounty": "$10,000,000", "scope": "Smart Contracts, Validators"},
            "LayerZero": {"platform": "Immunefi", "max_bounty": "$15,000,000", "scope": "Protocol, Endpoints"},
            "Axelar": {"platform": "Immunefi", "max_bounty": "$2,250,000", "scope": "Smart Contracts, Network"}
        }
        return bounty_info.get(bridge_name, {"platform": "Unknown", "max_bounty": "N/A", "scope": "General"})

    def generate_bridge_report(self, target: str, findings: list):
        """توليد تقرير أمني احترافي لجسور النقل."""
        report = f"# Cross-Chain Bridge Security Audit Report: {target}\n\n"
        if not findings:
            report += "✅ No immediate bridge-specific vulnerabilities found in static analysis.\n"
        else:
            report += "## 🚨 Critical Bridge Findings\n\n"
            for f in findings:
                report += f"### [{f['severity']}] {f['vulnerability']}\n"
                report += f"- **Location**: Line {f['line']}\n"
                report += f"- **Snippet**: `{f['snippet']}`\n"
                report += f"- **Risk**: High risk of funds theft via signature forgery or vault drain.\n\n"
        
        bounty = self.research_bridge_bounties(target)
        report += f"## 💰 Bridge Bounty Information\n"
        report += f"- **Platform**: {bounty['platform']}\n"
        report += f"- **Potential Reward**: {bounty['max_bounty']}\n"
        report += f"- **Official Scope**: {bounty['scope']}\n"
        
        return report
