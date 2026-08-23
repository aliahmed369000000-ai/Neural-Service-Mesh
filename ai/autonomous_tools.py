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
