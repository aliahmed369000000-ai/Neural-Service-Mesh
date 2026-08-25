import logging
import inspect
from typing import Any, Dict, List, Callable, Optional

logger = logging.getLogger("NSM-Toolbox")

class NSMToolbox:
    """
    المحرك المركزي لإدارة أدوات وكلاء NSM.
    يسمح بتسجيل، استدعاء، ومشاركة الأدوات بين العقد.
    """
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        logger.info("🛠️ NSM Toolbox Initialized.")

    def register_tool(self, name: str, func: Callable, description: str, category: str = "general"):
        """تسجيل أداة جديدة في الصندوق."""
        self.tools[name] = {
            "func": func,
            "description": description,
            "category": category,
            "signature": str(inspect.signature(func)),
            "doc": func.__doc__
        }
        logger.info(f"✅ Tool Registered: [{category}] {name}")

    def get_tool(self, name: str) -> Optional[Callable]:
        """الحصول على أداة محددة للاستخدام."""
        tool = self.tools.get(name)
        return tool["func"] if tool else None

    def list_tools(self) -> List[Dict[str, Any]]:
        """عرض قائمة بجميع الأدوات المتاحة."""
        return [
            {
                "name": name,
                "description": info["description"],
                "category": info["category"],
                "signature": info["signature"]
            }
            for name, info in self.tools.items()
        ]

    def execute_tool(self, name: str, **kwargs) -> Any:
        """تنفيذ أداة محددة مع المعاملات المطلوبة."""
        tool_func = self.get_tool(name)
        if not tool_func:
            raise ValueError(f"❌ Tool '{name}' not found in toolbox.")
        
        try:
            logger.info(f"⚙️ Executing tool: {name} with args: {kwargs}")
            return tool_func(**kwargs)
        except Exception as e:
            logger.error(f"❌ Error executing tool {name}: {e}")
            raise

# نسخة عالمية من صندوق الأدوات
nsm_toolbox = NSMToolbox()

def generate_custom_tool(name: str, code: str, description: str):
    """
    توليد أداة جديدة برمجياً ودمجها في الصندوق فوراً.
    تسمح للوكلاء بابتكار حلول للمشاكل الجديدة.
    """
    try:
        # تنفيذ الكود في بيئة معزولة (محاكاة)
        local_scope = {}
        exec(code, {}, local_scope)
        
        # البحث عن الدالة المولدة (يجب أن تحمل نفس اسم الأداة)
        if name in local_scope:
            nsm_toolbox.register_tool(name, local_scope[name], description, "autonomous")
            logger.info(f"✨ Autonomous Tool Generated: {name}")
            return True
        else:
            raise ValueError(f"Function '{name}' not found in generated code.")
    except Exception as e:
        logger.error(f"❌ Failed to generate autonomous tool: {e}")
        return False

nsm_toolbox.register_tool(
    "tool_generator", 
    generate_custom_tool, 
    "توليد أدوات جديدة برمجياً ودمجها في الصندوق", 
    "autonomous"
)

# --- أدوات أساسية مدمجة ---

def analyze_code_quality(code: str) -> Dict[str, Any]:
    """تحليل جودة الكود البرمجي ورصد الأخطاء الشائعة."""
    # محاكاة تحليل الكود
    issues = []
    if "eval(" in code: issues.append("Security: Avoid using eval()")
    if len(code.split('\n')) > 500: issues.append("Complexity: Function is too long")
    
    return {
        "score": 100 - (len(issues) * 10),
        "issues": issues,
        "status": "Healthy" if not issues else "Needs Review"
    }

def check_security_vulnerabilities(target_path: str) -> List[str]:
    """فحص الثغرات الأمنية في الملفات والمجلدات."""
    vulnerabilities = []
    # محاكاة فحص أمني
    if os.path.exists(target_path):
        vulnerabilities.append(f"Info: Scanning {target_path} for secrets...")
    return vulnerabilities

# تسجيل الأدوات الأساسية
nsm_toolbox.register_tool(
    "code_analyzer", 
    analyze_code_quality, 
    "تحليل جودة الكود ورصد الأخطاء البرمجية", 
    "development"
)
nsm_toolbox.register_tool(
    "security_scanner", 
    check_security_vulnerabilities, 
    "فحص الثغرات الأمنية في الملفات", 
    "security"
)

def data_processor(data: List[Any], operation: str = "summary") -> Dict[str, Any]:
    """معالجة البيانات الرقمية والنصية (تلخيص، متوسط، فرز)."""
    if not data: return {"error": "No data provided"}
    
    if operation == "summary":
        return {
            "count": len(data),
            "type": str(type(data[0])),
            "sample": data[:3]
        }
    elif operation == "sort":
        return {"result": sorted(data)}
    return {"status": "Operation completed"}

def language_translator(text: str, target_lang: str = "ar") -> str:
    """محاكاة ترجمة النصوص بين اللغات المختلفة."""
    # في النسخة الكاملة سيتم ربطها بـ LLM
    return f"[Translated to {target_lang}]: {text}"

nsm_toolbox.register_tool(
    "data_processor", 
    data_processor, 
    "معالجة البيانات الرقمية والنصية وتلخيصها", 
    "data"
)
nsm_toolbox.register_tool(
    "translator", 
    language_translator, 
    "ترجمة النصوص بين اللغات المختلفة", 
    "language"
)
