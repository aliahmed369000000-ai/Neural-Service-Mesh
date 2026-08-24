
import json
import time

class LoginUIComponent:
    """مكون واجهة مستخدم معيب عمداً لاختبار التصحيح البصري."""
    def __init__(self):
        self.html = """
        <div style="padding: 20px; background: #fff;">
            <h2 style="color: red; margin-left: -50px;">تسجيل الدخول</h2> <!-- خطأ بصري: محاذاة سالبة -->
            <input type="text" placeholder="اسم المستخدم" style="display: block; margin-bottom: 10px;">
            <button style="background: blue; color: blue;">دخول</button> <!-- خطأ بصري: لون النص نفس لون الخلفية -->
        </div>
        """
        self.defects = [
            {"element": "h2", "issue": "negative_margin", "description": "العنوان خارج النطاق البصري بسبب margin-left سالبة"},
            {"element": "button", "issue": "low_contrast", "description": "النص غير مرئي لأن لونه يطابق الخلفية"}
        ]

    def render(self):
        return self.html

    def run_visual_audit(self):
        """محاكاة لعملية التدقيق البصري التي يقوم بها الوكيل."""
        return {
            "status": "failed",
            "visual_defects": self.defects,
            "screenshot_mock": "ui_login_v1_error.png",
            "timestamp": time.time()
        }

if __name__ == "__main__":
    ui = LoginUIComponent()
    audit_results = ui.run_visual_audit()
    print(json.dumps(audit_results, ensure_ascii=False, indent=2))
