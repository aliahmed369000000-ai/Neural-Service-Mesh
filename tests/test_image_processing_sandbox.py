import sys
import os
import json
from pathlib import Path

# إضافة مسار المشروع لـ sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ai.agent_loop import TOOL_REGISTRY

def test_image_processing_in_sandbox():
    print("🧪 بدء اختبار معالجة الصور في الساندبوكس...")
    
    # تحديد مسار صورة موجودة
    image_path = str(ROOT / "real_test_image.png")
    
    # كود معالجة الصور: يقوم بفتح الصورة واستخراج أبعادها (محاكاة لمعالجة حقيقية)
    image_code = f"""
from PIL import Image
import os

class ImageProcessorNode:
    def process(self, data):
        path = data.get("image_path")
        if not os.path.exists(path):
            return {{"error": "File not found"}}
            
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            
        return {{
            "status": "success",
            "dimensions": f"{{width}}x{{height}}",
            "mode": mode,
            "file_size_kb": os.path.getsize(path) / 1024
        }}
"""
    
    print(f"\n[1] تشغيل معالج الصور على: {os.path.basename(image_path)}")
    params = {
        "code": image_code,
        "module_name": "image_processor",
        "class_name": "ImageProcessorNode"
    }
    
    # نحتاج لتمرير مسار الصورة في بيانات الاختبار
    # سنقوم بتعديل بسيط في TOOL_REGISTRY executor مؤقتاً أو استخدام منطق مشابه
    try:
        from ai.sandbox_lab import SandboxTestingLab
        class MockModule:
            def __init__(self, mid, name, code, cname):
                self.module_id, self.name, self.code, self.class_name = mid, name, code, cname
                self.status = "new"
                self.test_result = None

        lab = SandboxTestingLab(sandbox_dir=str(ROOT / "artifacts" / "sandbox"))
        # تعديل _build_test_data لتمرير مسار الصورة الحقيقي
        def custom_build_test_data(module):
            return {"image_path": image_path}
        
        lab._build_test_data = custom_build_test_data
        
        mock = MockModule("image_test", "image_processor", image_code, "ImageProcessorNode")
        res = lab.test_module(mock)
        
        print(f"النتيجة النهائية: {json.dumps(res.to_dict(), ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"❌ فشل الاختبار: {e}")

if __name__ == "__main__":
    test_image_processing_in_sandbox()
