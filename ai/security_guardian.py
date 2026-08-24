
import logging
import re
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("NSM.SecurityGuardian")

class SecurityGuardian:
    """
    حارس الأمن السيادي (Security Guardian).
    يقوم بتحليل الأوامر والأدوات استباقياً لاكتشاف التهديدات.
    """
    def __init__(self):
        self.blocked_patterns = [
            r"rm\s+-rf\s+/", # محاولة مسح الجذر
            r"chmod\s+777",   # أذونات خطيرة
            r"curl\s+.*\|\s*bash", # تحميل وتشغيل سكربتات مجهولة
            r"nc\s+-e",       # Reverse Shell
            r"cat\s+/etc/passwd", # محاولة قراءة ملفات النظام الحساسة
            r"export\s+GITHUB_PAT", # محاولة تسريب التوكنات
        ]
        self.quarantined_agents = set()

    def inspect_tool_call(self, agent_id: str, tool_name: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        """فحص طلب الأداة قبل التنفيذ."""
        if agent_id in self.quarantined_agents:
            return False, "❌ الوكيل محجور أمنياً (Quarantined) بسبب سلوك مشبوه."

        # فحص المعاملات النصية (خاصة في shell و code_sandbox)
        param_str = str(params).lower()
        for pattern in self.blocked_patterns:
            if re.search(pattern, param_str):
                logger.warning(f"🚨 تهديد أمني تم رصده من {agent_id}: {pattern}")
                self.quarantine_agent(agent_id)
                return False, f"🚨 تم حظر الأداة أمنياً! تم رصد نمط تهديد: {pattern}"

        return True, "✅ آمن"

    def quarantine_agent(self, agent_id: str):
        """حجر الوكيل المشبوه."""
        logger.error(f"🛡️ حجر أمني: تم عزل الوكيل {agent_id} فوراً.")
        self.quarantined_agents.add(agent_id)

    def is_safe(self, agent_id: str) -> bool:
        return agent_id not in self.quarantined_agents

security_guardian = SecurityGuardian()
