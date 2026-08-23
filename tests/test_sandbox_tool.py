import sys
import os
import json
from pathlib import Path

# إضافة مسار المشروع لـ sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ai.agent_loop import TOOL_REGISTRY

def test_sandbox_safety():
    print("🧪 بدء اختبار أمان الساندبوكس...")
    
    # 1. اختبار كود آمن
    safe_code = """
class SafeNode:
    def process(self, data):
        return {"result": "Hello from Sandbox", "input": data}
"""
    print("\n[1] اختبار كود آمن...")
    safe_params = {
        "code": safe_code,
        "module_name": "safe_module",
        "class_name": "SafeNode"
    }
    safe_res = json.loads(TOOL_REGISTRY["sandbox_test"].executor(safe_params))
    print(f"النتيجة: {safe_res['verdict']} (Score: {safe_res['score']})")
    
    # 2. اختبار كود غير آمن (محاولة استخدام os.system)
    unsafe_code = """
import os
class UnsafeNode:
    def process(self, data):
        os.system("ls")
        return {"result": "Executed unsafe command"}
"""
    print("\n[2] اختبار كود غير آمن (os.system)...")
    unsafe_params = {
        "code": unsafe_code,
        "module_name": "unsafe_module",
        "class_name": "UnsafeNode"
    }
    unsafe_res = json.loads(TOOL_REGISTRY["sandbox_test"].executor(unsafe_params))
    print(f"النتيجة: {unsafe_res['verdict']} (Score: {unsafe_res['score']})")
    print(f"انتهاكات الأمان: {unsafe_res['safety_violations']}")

if __name__ == "__main__":
    test_sandbox_safety()
