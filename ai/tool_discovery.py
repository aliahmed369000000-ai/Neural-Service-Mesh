
"""
ai/tool_discovery.py
====================
🆕 أداة اكتشاف وتحميل الأدوات المشتركة (Tool Discovery).
تسمح للوكلاء بالبحث في السجل المركزي وتحميل الأدوات التي صنعها وكلاء آخرون.
"""
import os
import json
import requests
from pathlib import Path
from typing import Dict, Any, List
from ai.tool_genesis import ToolGenesis
from ai.sandbox_lab import SandboxTestingLab

class ToolDiscovery:
    """محرك اكتشاف وتحميل الأدوات الجماعي."""
    
    def __init__(self):
        self.url = os.environ.get("NSM_MEMORY_URL", "http://localhost:8080")
        self.token = os.environ.get("NSM_ADMIN_TOKEN", "admin_dev_token")
        self.sandbox = SandboxTestingLab()

    def list_remote_tools(self) -> Dict[str, Any]:
        """جلب قائمة الأدوات من السجل المركزي."""
        try:
            resp = requests.get(f"{self.url}/tools/list", headers={"X-NSM-Token": self.token}, timeout=5)
            return resp.json() if resp.status_code == 200 else {}
        except:
            return {}

    def download_and_install(self, tool_id: str) -> Dict[str, Any]:
        """تحميل أداة، فحصها أمنياً، وتثبيتها محلياً."""
        try:
            resp = requests.get(f"{self.url}/tools/get/{tool_id}", headers={"X-NSM-Token": self.token}, timeout=5)
            if resp.status_code != 200:
                return {"ok": False, "error": "الأداة غير موجودة في السجل."}
            
            tool_data = resp.json()
            
            # 1. فحص الأمان المحلي (Sandbox Lab)
            # محاكاة كائن وحدة للفحص
            class MockModule:
                def __init__(self, data):
                    self.module_id = f"remote_{tool_id}"
                    self.name = data["name"]
                    self.code = data["code"]
                    self.class_name = data["name"]
                    self.status = "new"
                    self.test_result = None
            
            mock_mod = MockModule(tool_data)
            
            # فحص الأمان الثابت فقط للأدوات الديناميكية (تجنب فحص الاستنتاج الكامل للوحدات)
            safety_res = self.sandbox.test_module(mock_mod)
            
            # الأدوات الديناميكية عبارة عن دوال وليست فئات، لذا سنتجاهل فشل الاستنتاج (instantiation_success)
            # ونركز على السلامة (safety_passed) وصحة الكود (syntax_valid)
            if not safety_res.syntax_valid or not safety_res.safety_passed:
                return {"ok": False, "error": "فشل فحص الأمان للأداة المحملة.", "violations": safety_res.safety_violations}
            
            # 2. التثبيت المحلي عبر Tool Genesis
            install_res = ToolGenesis.create_tool(
                tool_data["name"], 
                tool_data["description"], 
                tool_data["code"], 
                tool_data["params_schema"],
                publish=False # لا نعيد نشرها لأنها موجودة بالفعل
            )
            
            return install_res
            
        except Exception as e:
            return {"ok": False, "error": str(e)}

def tool_discovery(params: Dict[str, Any]) -> str:
    """أداة الوكيل لاكتشاف وتحميل أدوات من السجل الجماعي."""
    action = params.get("action", "list")
    discovery = ToolDiscovery()
    
    if action == "list":
        tools = discovery.list_remote_tools()
        return json.dumps({"ok": True, "tools": tools}, ensure_ascii=False)
        
    if action == "install":
        tool_id = params.get("tool_id")
        if not tool_id: return "❌ tool_discovery: يجب توفير tool_id."
        res = discovery.download_and_install(tool_id)
        return json.dumps(res, ensure_ascii=False)
    
    return "❌ tool_discovery: إجراء غير معروف."
