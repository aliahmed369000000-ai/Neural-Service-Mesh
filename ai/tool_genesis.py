
"""
ai/tool_genesis.py
==================
🆕 محرك توليد الأدوات الديناميكي (Tool Genesis).
يسمح للوكيل بابتكار، برمجة، وحفظ أدوات جديدة لنفسه عند الحاجة.
"""
import os
import json
import importlib.util
from pathlib import Path
from typing import Dict, Any, Callable

ROOT = Path(__file__).resolve().parent.parent
DYNAMIC_TOOLS_DIR = ROOT / "ai" / "dynamic_tools"
DYNAMIC_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
if not (DYNAMIC_TOOLS_DIR / "__init__.py").exists():
    (DYNAMIC_TOOLS_DIR / "__init__.py").touch()

class ToolGenesis:
    """محرك خلق الأدوات الذاتي."""
    
    @staticmethod
    def publish_to_registry(name: str, description: str, code: str, schema: Dict[str, Any]):
        """نشر الأداة إلى السجل المركزي المشترك (معطل افتراضياً للأمان)."""
        url = os.environ.get("NSM_MEMORY_URL")
        if not url:
            return False
            
        import requests
        token = os.environ.get("NSM_ADMIN_TOKEN", "")
        
        try:
            payload = {
                "name": name,
                "description": description,
                "code": code,
                "params_schema": schema,
                "agent_id": os.environ.get("AGENT_ID", "sovereign_agent")
            }
            resp = requests.post(f"{url}/tools/publish", json=payload, headers={"X-NSM-Token": token}, timeout=5)
            return resp.status_code == 200
        except:
            return False

    @staticmethod
    def create_tool(name: str, description: str, code: str, params_schema: Dict[str, Any], publish: bool = False) -> Dict[str, Any]:
        """خلق أداة جديدة وحفظها."""
        if not name.isidentifier():
            return {"ok": False, "error": "اسم الأداة غير صالح."}
            
        file_path = DYNAMIC_TOOLS_DIR / f"{name}.py"
        safe_code = json.dumps(code)
        
        # قالب نصي ثابت مع استبدال يدوي لتجنب مشاكل النطاق و f-strings
        # ملاحظة: استخدام exec على المضيف هو تصميم تجريبي (Experimental) وليس sandbox حقيقي.
        tool_template = """
import math
import os
import sys
import json
from typing import Dict, Any

def __NAME__(params: Dict[str, Any]) -> str:
    \"\"\"__DESC__\"\"\"
    try:
        # إعداد النطاق الموحد لـ exec
        l_vars = {
            "params": params,
            "math": math,
            "os": os,
            "sys": sys,
            "json": json,
            "result": None
        }
        # الكود المولد
        source = __CODE__
        # تنفيذ الكود (Warning: Executed on host environment)
        exec(source, l_vars, l_vars)
        res = l_vars.get("result")
        return str(res) if res is not None else "✅ تم التنفيذ بنجاح"
    except Exception as e:
        return f"❌ خطأ في تنفيذ الأداة '__NAME__': {e}"
"""
        tool_content = tool_template.replace("__NAME__", name).replace("__DESC__", description).replace("__CODE__", safe_code)
        
        try:
            file_path.write_text(tool_content.strip() + "\n", encoding="utf-8")
            
            published = False
            if publish:
                published = ToolGenesis.publish_to_registry(name, description, code, params_schema)
                
            return {
                "ok": True,
                "name": name,
                "path": str(file_path),
                "published": published,
                "msg": f"تم خلق الأداة '{name}' بنجاح."
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @staticmethod
    def load_dynamic_tools() -> Dict[str, Callable]:
        """تحميل الأدوات الديناميكية (يجب استخدامه بحذر)."""
        tools = {}
        if not DYNAMIC_TOOLS_DIR.exists():
            return tools
            
        for file in DYNAMIC_TOOLS_DIR.glob("*.py"):
            if file.name == "__init__.py": continue
            
            name = file.stem
            try:
                spec = importlib.util.spec_from_file_location(name, str(file))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, name):
                    tools[name] = getattr(module, name)
            except Exception:
                continue
        return tools

def tool_genesis(params: Dict[str, Any]) -> str:
    """أداة الوكيل لخلق أدوات جديدة لنفسه."""
    name = params.get("name")
    desc = params.get("description", "")
    code = params.get("code")
    schema = params.get("params_schema", params.get("schema", {}))
    
    if not name or not code:
        return "❌ tool_genesis: يجب توفير name و code."
        
    try:
        # تعطيل النشر الشبكي افتراضياً للأمان في البيئة المحلية
        res = ToolGenesis.create_tool(name, desc, code, schema, publish=False)
        return json.dumps(res, ensure_ascii=False)
    except Exception as e:
        return f"❌ tool_genesis: {str(e)}"
