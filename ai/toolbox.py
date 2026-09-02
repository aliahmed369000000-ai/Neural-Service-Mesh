import logging
import inspect
import os
import threading
import concurrent.futures
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

    # أدوات محظورة دائماً (كود غير موثوق / توليد ديناميكي)
    BLOCKED_TOOLS = frozenset({"tool_generator", "generate_custom_tool", "eval", "exec", "run_code"})

    def execute_tool(self, name: str, timeout: int = 30, **kwargs) -> Any:
        """تنفيذ أداة مسجّلة فقط — مع حظر مسارات الكود غير الموثوق (#15)."""
        if not name or name in self.BLOCKED_TOOLS:
            raise PermissionError(f"Tool '{name}' is blocked for security (#15)")
        # ارفض أي وسيط يبدو ككود قابل للتنفيذ
        for k, v in (kwargs or {}).items():
            if k in ("code", "source", "script", "python", "body") and isinstance(v, str) and len(v) > 0:
                raise PermissionError(
                    f"Refusing kwargs[{k!r}]: untrusted code parameters are not allowed (#15)"
                )
        tool_func = self.get_tool(name)
        if not tool_func:
            raise ValueError(f"❌ Tool '{name}' not found in toolbox.")

        try:
            logger.info(f"⚙️ Executing tool: {name} (Timeout: {timeout}s)")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(tool_func, **kwargs)
                return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.error(f"⏱️ Tool {name} timed out after {timeout}s")
            raise RuntimeError(f"Tool {name} execution timed out.")
        except PermissionError:
            raise
        except Exception as e:
            logger.error(f"❌ Error executing tool {name}: {e}")
            raise

# نسخة عالمية من صندوق الأدوات
nsm_toolbox = NSMToolbox()

def generate_custom_tool(name: str, code: str, description: str):
    """
    #15: مُعطّل عمداً — لا يُسمح بتنفيذ كود غير موثوق عبر exec.
    كان هذا المسار يسمح بتوليد أدوات ديناميكية؛ أُغلق لأسباب أمنية.
    """
    logger.error(
        "🚫 generate_custom_tool blocked: arbitrary code execution is forbidden (#15). "
        f"requested_name={name!r}"
    )
    raise PermissionError(
        "tool_generator is disabled: running untrusted code on mesh nodes is not allowed"
    )

# لا نُسجّل tool_generator كأداة قابلة للاستدعاء عن بُعد

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

def distributed_training_monitor(**kwargs) -> str:
    """أداة لمراقبة حالة التدريب الموزع على عدة GPUs."""
    import torch
    try:
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return "❌ No GPUs detected for distributed training."
        
        status = f"📡 NSM Distributed Swarm: {gpu_count} GPUs Active.\n"
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / 1e9
            status += f" - GPU {i} ({props.name}): {mem_total:.2f} GB VRAM.\n"
        
        return status + "✅ Ready for Surah 4096 Multi-GPU Training."
    except Exception as e:
        return f"❌ Error monitoring GPUs: {e}"

nsm_toolbox.register_tool(
    "distributed_trainer", 
    distributed_training_monitor, 
    "مراقبة وإدارة التدريب الموزع على عدة GPUs", 
    "compute"
)

def training_speed_benchmark(tflops: float = 0.0, samples_per_sec: float = 0.0) -> str:
    """أداة لقياس سرعة التدريب ورصد التحسن في الأداء."""
    status = "📊 NSM Training Speed Benchmark:\n"
    status += f" - Throughput: {samples_per_sec:.2f} samples/sec\n"
    status += f" - Compute Power: {tflops:.2f} TFLOPS\n"
    
    if tflops > 100:
        status += "🚀 Performance Status: EXTREME (ZeRO-3 Aggressive Active)\n"
    elif tflops > 50:
        status += "⚡ Performance Status: HIGH\n"
    else:
        status += "🐢 Performance Status: NORMAL\n"
        
    return status

nsm_toolbox.register_tool(
    "speed_benchmark", 
    training_speed_benchmark, 
    "قياس سرعة التدريب ورصد التحسن في الأداء الحسابي", 
    "compute"
)

def security_monitor_tool() -> str:
    """أداة لرصد محاولات التسلل وحالة تشفير البيانات الحية في السرب."""
    from ai.security_guard import NSMSecurityGuard
    guard = NSMSecurityGuard()
    status = guard.get_security_status()
    
    report = "🛡️ NSM Security Monitor Report:\n"
    report += f" - Status: {status['status']}\n"
    report += f" - Encryption: {status['encryption']}\n"
    report += f" - Integrity Checks: {status['integrity_checks']}\n"
    report += " - Latest Alerts:\n"
    for alert in status['alerts']:
        report += f"   ⚠️ {alert}\n"
        
    return report

nsm_toolbox.register_tool(
    "security_monitor", 
    security_monitor_tool, 
    "رصد محاولات التسلل وحالة تشفير البيانات الحية", 
    "security"
)

def cognitive_evolution_tracker() -> str:
    """أداة لرصد معدل نمو الذكاء والتحسينات المطبقة في السرب العالمي."""
    from ai.cognitive_growth import cognitive_engine
    return cognitive_engine.get_growth_report()

nsm_toolbox.register_tool(
    "cognitive_tracker", 
    cognitive_evolution_tracker, 
    "رصد معدل نمو الذكاء والتحسينات الهيكلية المطبقة", 
    "autonomous"
)
