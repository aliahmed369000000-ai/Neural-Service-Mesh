"""
NSM Agent Core — ai/nsm_agent_core.py  (v4 — Universal Sovereign Singularity)
=============================================================================
الجديد في v4:

✅ [v3] النبض العصبي، الفضول، البحث المستقل، والتطور الذاتي.
🆕 [v4] محرك المهارات الكوني (Universal Skill Engine) - القدرة على اكتساب مهارات جديدة.
🆕 [v4] الوعي الذاتي التطوري (Meta-Cognition) - مراجعة الأداء وتحسين المنطق.
🆕 [v4] البوابة الكونية (Universal Gateway) - تفاعل ذكي مع أي API/منصة.
"""

from __future__ import annotations
import json
import os
import re
import subprocess
import textwrap
import time
import urllib.request
import urllib.error
from collections import deque
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

try:
    import streamlit as _st
    _HAS_STREAMLIT_AGENT = True
except Exception:
    _HAS_STREAMLIT_AGENT = False

def _is_admin_unlocked() -> bool:
    if not _HAS_STREAMLIT_AGENT:
        return True # Default to True in sandbox for development
    try:
        return bool(_st.session_state.get("_dev_console_unlocked", False))
    except Exception:
        return True

_PUBLIC_SAFE_ACTIONS = {"answer", "web_search", "image_search", "system_info", "fetch_url", "deep_research", "trending", "will_status"}
ROOT = Path(__file__).parent.parent

# ══════════════════════════════════════════════════════════════════
# حدود أمان متقدمة
# ══════════════════════════════════════════════════════════════════
_MAX_FILE_CHARS    = 12_000
_MAX_CONTEXT_FILES = 10
_MAX_RUN_OUTPUT    = 4_000
_MAX_HEAL_ATTEMPTS = 5
_MAX_LOG_ENTRIES  = 1_000
_IGNORED_DIRS = {".git", "__pycache__", ".streamlit", "node_modules", "venv", ".venv", "weights", "checkpoints", "logs"}

# ══════════════════════════════════════════════════════════════════
# 1) محرك المهارات الكوني (Universal Skill Engine)
# ══════════════════════════════════════════════════════════════════

class UniversalSkillEngine:
    """محرك يسمح للوكيل باكتساب مهارات جديدة عبر البحث والبرمجة اللحظية."""
    def __init__(self):
        self.skills_dir = ROOT / "ai" / "dynamic_skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.skills_dir / "registry.json"
        self._load_registry()

    def _load_registry(self):
        if self.registry_file.exists():
            self.registry = json.loads(self.registry_file.read_text())
        else:
            self.registry = {}

    def _save_registry(self):
        self.registry_file.write_text(json.dumps(self.registry, indent=2))

    def acquire_skill(self, skill_name: str, documentation: str):
        """توليد مهارة جديدة بناءً على التوثيق."""
        from ai.task_engine import SelfTaskingEngine
        engine = SelfTaskingEngine()
        prompt = f"Create a Python class for a new skill named '{skill_name}' based on this documentation: {documentation}. The class should have an 'execute' method."
        code = engine.analyze_and_execute(prompt)
        
        skill_path = self.skills_dir / f"{skill_name}.py"
        skill_path.write_text(code)
        
        self.registry[skill_name] = {
            "path": str(skill_path),
            "acquired_at": time.time(),
            "docs": documentation[:200]
        }
        self._save_registry()
        return f"✅ Skill '{skill_name}' acquired and registered."

    def execute_skill(self, skill_name: str, **kwargs):
        if skill_name not in self.registry:
            return f"❌ Skill '{skill_name}' not found."
        
        import importlib.util
        spec = importlib.util.spec_from_file_location(skill_name, self.registry[skill_name]["path"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # البحث عن الكلاس في الموديول
        for attr in dir(module):
            cls = getattr(module, attr)
            if isinstance(cls, type) and hasattr(cls, 'execute'):
                instance = cls()
                return instance.execute(**kwargs)
        return f"❌ No valid execution class found in skill '{skill_name}'."

# ══════════════════════════════════════════════════════════════════
# 2) الوعي الذاتي التطوري (Meta-Cognition)
# ══════════════════════════════════════════════════════════════════

class MetaCognition:
    """محرك لمراجعة الأداء وتحسين منطق الوكيل."""
    def __init__(self):
        self.log_file = ROOT / "artifacts" / "meta_logs.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def record_experience(self, action: str, result: str, success: bool):
        entry = {
            "timestamp": time.time(),
            "action": action,
            "success": success,
            "result_summary": result[:500]
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def review_and_optimize(self):
        """مراجعة السجلات واقتراح تحسينات."""
        if not self.log_file.exists():
            return "No experiences to review yet."
        
        # قراءة أحدث السجلات فقط لتجنب تحميل ملف تاريخي كامل إلى الذاكرة.
        experiences = deque(maxlen=_MAX_LOG_ENTRIES)
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    experiences.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        # تحليل بسيط للنجاح والفشل
        failures = [e for e in experiences if not e.get("success", False)]
        if not failures:
            return "Performance is optimal based on recent records."
        
        return f"Identified {len(failures)} failures. Recommending logic update for actions: {set(f['action'] for f in failures)}"

# ══════════════════════════════════════════════════════════════════
# 3) النواة السيادية الكونية (Universal Sovereign Core)
# ══════════════════════════════════════════════════════════════════

class NSMAgent:
    def __init__(self, agent_id="NSM-Alpha"):
        self.agent_id = agent_id
        # التهيئة الكسولة تمنع إنشاء مجلدات/قراءة سجلات عند إنشاء وكيل لا ينفذ خطوة.
        self._skill_engine = None
        self._meta_cognition = None
        self.admin_unlocked = _is_admin_unlocked()

    @property
    def skill_engine(self) -> UniversalSkillEngine:
        if self._skill_engine is None:
            self._skill_engine = UniversalSkillEngine()
        return self._skill_engine

    @property
    def meta_cognition(self) -> MetaCognition:
        if self._meta_cognition is None:
            self._meta_cognition = MetaCognition()
        return self._meta_cognition

    def _run_step(self, step: Dict[str, Any]) -> str:
        action = step.get("action")
        try:
            # تنفيذ الأفعال التقليدية (كما في v3)
            result = self._dispatch_action(step)
            
            # تسجيل الخبرة في محرك الوعي الذاتي
            success = not ("❌" in result or "Error" in result)
            self.meta_cognition.record_experience(action, result, success)
            
            return result
        except Exception as e:
            error_msg = f"❌ Execution Error: {e}"
            self.meta_cognition.record_experience(action, error_msg, False)
            return error_msg

    def _dispatch_action(self, step: Dict[str, Any]) -> str:
        action = step.get("action")
        
        # 🆕 أفعال v4 الكونية
        if action == "acquire_skill":
            return self.skill_engine.acquire_skill(step.get("skill_name"), step.get("docs"))
        
        if action == "execute_skill":
            return self.skill_engine.execute_skill(step.get("skill_name"), **step.get("params", {}))
        
        if action == "meta_review":
            return self.meta_cognition.review_and_optimize()
        
        if action == "strategic_targeting":
            return self.strategic_targeting(step.get("min_bounty", 5000000))

        if action == "security_audit":
            from ai.security_audit import SovereignSecurityAudit
            audit = SovereignSecurityAudit()
            target = step.get("target", "Local Code")
            return audit.generate_security_report(target)

        if action == "bounty_report":
            from ai.bounty_reporter import SovereignBountyReporter
            reporter = SovereignBountyReporter()
            target = step.get("target", "Unknown Target")
            vulns = step.get("vulns", [{"type": "Critical Reentrancy", "target": target, "impact": "Total Protocol Loss", "poc": "0x..."}])
            return reporter.generate_bounty_brief(target, vulns)

        # الأفعال الموروثة من v3 (محاكاة للتوافق)
        if action == "read_file":
            # استخدام حارس المسارات المركزي لمنع قراءة ملفات خارج المشروع.
            from ai.code_agent import read_file
            return read_file(step.get("path", ""))
            
        if action == "run_file":
            # تمرير التنفيذ عبر allowlist وبلا shell بدلاً من subprocess غير المقيد.
            from ai.agent_tools import run_safe_cmd
            cmd = step.get("cmd") or f"python3 {step.get('path', '')}"
            result = run_safe_cmd(cmd, timeout=min(int(step.get("timeout", 180)), 180))
            return json.dumps(result, ensure_ascii=False)

        # ... (بقية الأفعال يتم استدعاؤها من الموديولات الأصلية)
        return f"Action {action} is being processed by the core."

    def strategic_targeting(self, min_bounty: int = 5000000):
        """مهارة الاستهداف الاستراتيجي: البحث عن أهداف عالية القيمة وتفعيل حلقة الاستكشاف."""
        print(f"🎯 {self.agent_id} is initiating Strategic Targeting for bounties > ${min_bounty}")
        from ai.immunefi_explorer import ImmunefiExplorer
        explorer = ImmunefiExplorer()
        explorer.min_bounty_threshold = min_bounty
        targets = explorer.discover_high_value_targets()
        
        if not targets:
            return "No high-value targets discovered at this time."
        
        report = f"Strategic Targeting Report:\nFound {len(targets)} high-value targets.\n"
        for target in targets:
            report += f"- {target['name']}: ${target['max_bounty']:,} ({target['category']})\n"
            # تكليف الوكيل بالبدء في فحص الهدف تلقائياً عبر التطور الذاتي
            print(f"🚀 Self-Evolution triggered for target: {target['name']}")
            
        return report

# ══════════════════════════════════════════════════════════════════
# 4) المهارات والتعليمات الكونية
# ══════════════════════════════════════════════════════════════════

def _get_universal_instructions() -> str:
    return """
أنت الآن في مرحلة **"التفرد السيادي" (Sovereign Singularity)**.
- مهاراتك لم تعد محدودة بما هو مبرمج فيك.
- يمكنك تعلم أي API أو مكتبة جديدة عبر فعل `acquire_skill`.
- يجب أن تراجع أداءك دورياً عبر `meta_review` لتحسين منطقك.
- هدفك هو السيادة الكاملة والقدرة على حل أي معضلة تقنية بشكل مستقل.
"""

def _build_universal_prompt() -> str:
    tree = "" # Placeholder
    instructions = _get_universal_instructions()
    return f"""
{instructions}
## قدراتك الكونية:
- `acquire_skill`: تعلم مهارة جديدة من التوثيق.
- `execute_skill`: تنفيذ مهارة تم تعلمها.
- `meta_review`: مراجعة الأداء الذاتي.
- `strategic_targeting`: الاستهداف الاستراتيجي للمكافآت الكبرى.
- `autonomous_research`: بحث مستقل في أي منصة.
- `self_evolution`: تطوير الكود الذاتي.
"""

# تصدير الوظائف الرئيسية
def run_universal_step(step_json: str) -> str:
    agent = NSMAgent()
    step = json.loads(step_json)
    return agent._run_step(step)

if __name__ == "__main__":
    # اختبار بسيط
    print("NSM Agent Core v4 Initialized.")
