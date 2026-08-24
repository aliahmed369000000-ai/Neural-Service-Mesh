import sys
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.sync_evaluator import sync_evaluator

def test_evaluation():
    print("📊 بدء اختبار تقييم المزامنة...")
    
    # 1. بيانات مرجعية (Ground Truth)
    ground_truth = [
        {"timestamp": 1.0, "text": "مرحباً"},
        {"timestamp": 5.0, "text": "وداعاً"}
    ]
    
    # 2. نتائج مزامنة فعلية (Actual Sync)
    actual_sync = [
        {"timestamp": 1.1, "spoken_text": "مرحباً بكم"},
        {"timestamp": 5.05, "spoken_text": "وداعاً للجميع"}
    ]
    
    # 3. تشغيل التقييم
    metrics = sync_evaluator.evaluate_sync(ground_truth, actual_sync)
    perf = sync_evaluator.run_benchmark(actual_sync)
    
    print(f"\n✅ نتائج التقييم:")
    print(f"- الخطأ الزمني المتوسط: {metrics['temporal_error']:.3f} ثانية")
    print(f"- درجة المطابقة الدلالية: {metrics['score']}%")
    print(f"- سرعة المعالجة: {perf['throughput_fps']:.2f} إطار/ثانية")
    
    assert metrics['temporal_error'] < 0.2
    assert metrics['score'] == 100
    print("\n🎉 نجح اختبار تقييم المزامنة!")

if __name__ == "__main__":
    test_evaluation()
