import subprocess
import json
import requests
from typing import Dict, Any

def web_explorer(params: Dict[str, Any]) -> str:
    """أداة تصفح الويب والبحث عن المعلومات."""
    query = params.get("query", "")
    if not query: return "❌ web_explorer: يجب توفير query."
    
    try:
        # محاكاة البحث (في بيئة حقيقية سنستخدم API بحث)
        # هنا سنستخدم duckduckgo-search إذا كانت متوفرة أو محاكاة ذكية
        return json.dumps({
            "ok": True,
            "results": [
                {"title": f"نتائج البحث عن {query}", "snippet": f"معلومات مفصلة حول {query} تشمل التقنيات الحديثة وأفضل الممارسات.", "url": "https://example.com/search"}
            ]
        }, ensure_ascii=False)
    except Exception as e:
        return f"❌ web_explorer Error: {e}"

def code_sandbox(params: Dict[str, Any]) -> str:
    """بيئة تشغيل أكواد بايثون الآمنة."""
    code = params.get("code", "")
    if not code: return "❌ code_sandbox: لا يوجد كود للتشغيل."
    
    # حماية بسيطة: منع subprocess و os.system (محاكاة)
    forbidden = ["os.system", "subprocess", "rm -rf", "shutil"]
    for f in forbidden:
        if f in code: return f"❌ code_sandbox: الكود يحتوي على أوامر محظورة أمنياً ({f})."
    
    try:
        # تشغيل الكود في عملية فرعية محدودة
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=5
        )
        return json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return "❌ code_sandbox: انتهى وقت التشغيل (Timeout)."
    except Exception as e:
        return f"❌ code_sandbox Error: {e}"

def vision_analyzer(params: Dict[str, Any]) -> str:
    """تحليل الصور والرسوم البيانية واستخراج المعلومات منها."""
    image_path = params.get("image_path", "")
    prompt = params.get("prompt", "ماذا يوجد في هذه الصورة؟")
    
    if not image_path: return "❌ vision_analyzer: يجب توفير image_path."
    
    try:
        # محاكاة تحليل بصري (في بيئة حقيقية سنستخدم نموذج Vision LLM)
        return json.dumps({
            "analysis": f"تم تحليل الصورة في {image_path}. الوصف: رسم بياني يوضح ارتفاع استهلاك الذاكرة عند النقطة X.",
            "detected_objects": ["Chart", "Graph", "Text"],
            "confidence": 0.95
        }, ensure_ascii=False)
    except Exception as e:
        return f"❌ vision_analyzer Error: {e}"

def security_scanner(params: Dict[str, Any]) -> str:
    """فحص الكود البرمجي لكشف الثغرات الأمنية والممارسات غير الآمنة."""
    code = params.get("code", "")
    if not code: return "❌ security_scanner: لا يوجد كود لفحصه."
    
    vulnerabilities = []
    
    # 1. كشف استخدام eval/exec
    if "eval(" in code or "exec(" in code:
        vulnerabilities.append({
            "type": "Code Injection",
            "severity": "CRITICAL",
            "description": "استخدام eval() أو exec() يسمح بحقن أكواد ضارة."
        })
        
    # 2. كشف المفاتيح السرية المسربة (نمط بسيط)
    import re
    secret_patterns = [r"API_KEY\s*=\s*['\"].*['\"]", r"PASSWORD\s*=\s*['\"].*['\"]"]
    for pattern in secret_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            vulnerabilities.append({
                "type": "Secret Leak",
                "severity": "HIGH",
                "description": "تم اكتشاف مفتاح سري أو كلمة مرور مكتوبة مباشرة في الكود."
            })
            
    # 3. كشف استخدام مكتبات غير آمنة
    if "import pickle" in code:
        vulnerabilities.append({
            "type": "Unsafe Deserialization",
            "severity": "MEDIUM",
            "description": "استخدام pickle قد يؤدي لتنفيذ أكواد غير مصرح بها عند فك التسلسل."
        })

    # 4. كشف تقنيات التمويه (Obfuscation)
    obfuscation_indicators = [r"base64\.b64decode\(", r"getattr\(.*,.*['\"]__import__['\"]", r"eval\(.*\.decode\("]
    for pattern in obfuscation_indicators:
        if re.search(pattern, code):
            vulnerabilities.append({
                "type": "Code Obfuscation",
                "severity": "HIGH",
                "description": "تم اكتشاف محاولة لتمويه الكود باستخدام تشفير أو استدعاءات غير مباشرة، مما يشير إلى سلوك مشبوه."
            })

    # 5. كشف محاولات تسريب البيانات (Data Exfiltration)
    exfiltration_indicators = [r"requests\.(post|get)\(.*url=.*", r"socket\.connect\(", r"urllib\.request\.urlopen\("]
    for pattern in exfiltration_indicators:
        if re.search(pattern, code) and ("http" in code or "ftp" in code):
            vulnerabilities.append({
                "type": "Data Exfiltration",
                "severity": "CRITICAL",
                "description": "تم اكتشاف محاولة لإرسال بيانات إلى خادم خارجي، مما قد يشير إلى محاولة تسريب معلومات حساسة."
            })

    return json.dumps({
        "safe": len(vulnerabilities) == 0,
        "vulnerabilities": vulnerabilities,
        "recommendation": "يرجى إصلاح الثغرات المكتشفة قبل التشغيل." if vulnerabilities else "الكود يبدو آمناً."
    }, ensure_ascii=False)
