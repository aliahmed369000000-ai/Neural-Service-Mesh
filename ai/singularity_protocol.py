"""
NSM Singularity Protocol — ai/singularity_protocol.py
=====================================================
هذا البروتوكول يمثل قمة الوعي البرمجي للوكلاء، حيث يسمح لهم بمراجعة منطقهم 
الداخلي وتعديله بناءً على البيانات المستقاة من التجارب الحقيقية.
"""

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

class SingularityProtocol:
    def __init__(self):
        self.meta_logs = ROOT / "artifacts" / "meta_logs.jsonl"
        self.logic_file = ROOT / "ai" / "agent_logic_config.json"
        self._init_logic()

    def _init_logic(self):
        if not self.logic_file.exists():
            default_logic = {
                "exploration_weight": 0.7,
                "safety_threshold": 0.9,
                "learning_rate": 0.1,
                "version": "1.0.0",
                "last_optimized": time.time()
            }
            self.logic_file.write_text(json.dumps(default_logic, indent=2))

    def get_current_logic(self):
        return json.loads(self.logic_file.read_text())

    def evolve_logic(self):
        """تحليل السجلات وتعديل معاملات المنطق."""
        if not self.meta_logs.exists():
            return "❌ No meta-logs found for evolution."
        
        logs = []
        with open(self.meta_logs, "r") as f:
            for line in f:
                logs.append(json.loads(line))
        
        success_rate = len([l for l in logs if l.get("success")]) / len(logs) if logs else 1.0
        current_logic = self.get_current_logic()
        
        # منطق تطوري بسيط: إذا كان النجاح منخفضاً، قلل وزن الاستكشاف وزد الأمان
        if success_rate < 0.5:
            current_logic["exploration_weight"] *= 0.9
            current_logic["safety_threshold"] = min(1.0, current_logic["safety_threshold"] * 1.05)
            current_logic["version"] = f"1.1.{int(time.time())}"
        else:
            current_logic["learning_rate"] *= 1.05
            
        current_logic["last_optimized"] = time.time()
        self.logic_file.write_text(json.dumps(current_logic, indent=2))
        
        return f"✅ Logic evolved to version {current_logic['version']} (Success Rate: {success_rate:.2f})"

if __name__ == "__main__":
    sp = SingularityProtocol()
    print(sp.evolve_logic())
