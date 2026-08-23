
"""
ai/tool_genesis.py
==================
🆕 محرك توليد الأدوات الديناميكي (Tool Genesis).
يسمح للوكيل بابتكار، برمجة، وحفظ أدوات جديدة لنفسه عند الحاجة.
"""
import os
import json
import inspect
import importlib.util
from pathlib import Path
from typing import Dict, Any, Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
DYNAMIC_TOOLS_DIR = ROOT / "ai" / "dynamic_tools"
DYNAMIC_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
(DYNAMIC_TOOLS_DIR / "__init__.py").touch()

class ToolGenesis:
    """محرك خلق الأدوات الذاتي."""
    
    @staticmethod
    def create_tool(name: str, description: str, code: str, params_schema: Dict[str, Any]) -> Dict[str, Any]:
        """خلق أداة جديدة وحفظها."""
        if not name.isidentifier():
            return {"ok": False, "error": "اسم الأداة غير صالح."}
            
        file_path = DYNAMIC_TOOLS_DIR / f"{name}.py"
        
        # قالب الأداة
        tool_template = f'''
"""
Dynamic Tool: {name}
Description: {description}
Generated at: {os.times()}
"""
from typing import Dict, Any

def {name}(params: Dict[str, Any]) -> str:
    """{description}"""
    {code.replace(chr(10), chr(10) + "    ")}
'''
        try:
            file_path.write_text(tool_template, encoding="utf-8")
            return {
                "ok": True,
                "name": name,
                "path": str(file_path),
                "msg": f"تم خلق الأداة '{name}' بنجاح."
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @staticmethod
    def load_dynamic_tools() -> Dict[str, Callable]:
        """تحميل جميع الأدوات الديناميكية المخلوقة."""
        tools = {}
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
    desc = params.get("description")
    code = params.get("code")
    schema = params.get("params_schema", {})
    
    if not all([name, desc, code]):
        return "❌ tool_genesis: يجب توفير name, description, code."
        
    res = ToolGenesis.create_tool(name, desc, code, schema)
    return json.dumps(res, ensure_ascii=False)
