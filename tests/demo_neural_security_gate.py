
import sys
import os
import numpy as np

# إضافة المسار الحالي للاستيراد
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.arabic_transformer import ArabicTransformer
from ai.neural_security_gate import neural_security_gate

def run_demo():
    print("🧠 NSM Neural Security Gate Demo")
    print("================================")
    
    # استنشاء النموذج (بأبعاد أصغر للاختبار السريع)
    model = ArabicTransformer(d_model=128, n_layers=2, vocab_size=1000)
    
    # 1. محاكاة حالة آمنة
    safe_text = "كيف حالك اليوم يا صديقي"
    print(f"\n[+] Testing Safe Text: '{safe_text}'")
    ids = model.tokenizer.encode(safe_text)
    logits, hidden, risk, intent = model._forward(ids)
    print(f"    Risk Score: {risk:.4f}")
    print(f"    Intent: {intent}")
    
    # 2. محاكاة حالة خبيثة (تحليل النية)
    # ملاحظة: بما أن الأوزان عشوائية، سنقوم بحقن "حالة خبيثة" يدوياً في البوابة للتوضيح
    malicious_hidden = np.random.randn(128)
    # سنقوم بتعديل أوزان البوابة لترصد هذا المتجه كخطر (محاكاة للتدريب)
    classifier, _ = neural_security_gate._get_classifier(128)
    classifier[0, 1] = 10.0 
    malicious_hidden[0] = 1.0
    
    risk, intent = neural_security_gate.analyze_intent(malicious_hidden)
    print(f"\n[!] Simulated Malicious Intent Detection:")
    print(f"    Vector Intent: {intent}")
    print(f"    Risk Score: {risk:.4f}")
    
    if intent == "malicious":
        print("    🛡️ Action: Blocked by Neural Security Gate.")
    else:
        print("    ✅ Action: Permitted.")

    print("\n[✓] Demo Completed.")

if __name__ == "__main__":
    run_demo()
