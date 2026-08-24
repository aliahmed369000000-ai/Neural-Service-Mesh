import sys
import os
import numpy as np

# إضافة المسار للمشروع
sys.path.append(os.getcwd())

from ai.multimodal_network import MultimodalRoutingCore

def test_dte_evolution_cycle():
    print("--- اختبار تطور الطوبولوجيا الديناميكي (DTE) ---")
    
    # 1. تهيئة الشبكة مع DTE
    model = MultimodalRoutingCore(enable_dte=True)
    
    # 2. محاكاة دورة تدريبية طويلة لتحفيز التطور (101 خطوة)
    print("بدء محاكاة التدريب لتحفيز التطور...")
    dummy_input = np.random.randn(1040)
    
    initial_weights_sum = np.sum(np.abs(model.hidden.weights))
    
    for i in range(101):
        model.train_step(dummy_input, agent_target=0, analysis_target=np.zeros(4))
        if i % 20 == 0:
            print(f"خطوة {i}: الخسارة = {model._last_loss:.6f}")
            
    # 3. التحقق من حدوث التطور
    # بعد 100 خطوة، يجب أن يتم استدعاء _evolve_topology
    final_weights_sum = np.sum(np.abs(model.hidden.weights))
    
    print(f"مجموع الأوزان الأولي: {initial_weights_sum:.4f}")
    print(f"مجموع الأوزان النهائي: {final_weights_sum:.4f}")
    
    # التحقق من أن الأوزان تغيرت (بسبب التقليم أو النمو)
    assert initial_weights_sum != final_weights_sum, "فشل: لم يحدث أي تغيير في طوبولوجيا الأوزان"
    
    # التحقق من إحصائيات النمو
    summary = model.summary()
    print(f"ملخص النموذج: {summary}")
    
    assert summary["train_steps"] == 101, "فشل: لم يتم تسجيل خطوات التدريب بشكل صحيح"
    print("✅ نجح اختبار التطور الطوبولوجي DTE!")

if __name__ == "__main__":
    try:
        test_dte_evolution_cycle()
    except Exception as e:
        print(f"❌ فشل الاختبار: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
