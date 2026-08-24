import time
import json
import os

class SyncEvaluator:
    """
    نظام تقييم أداء المزامنة السمعية البصرية.
    """
    def __init__(self):
        self.results = []

    def evaluate_sync(self, ground_truth, actual_sync):
        """
        تقييم المزامنة الفعلية مقابل البيانات المرجعية.
        """
        metrics = {
            "temporal_error": 0,
            "semantic_match_count": 0,
            "total_items": len(ground_truth),
            "score": 0
        }
        
        total_error = 0
        for gt in ground_truth:
            # البحث عن أقرب طابع زمني في النتائج الفعلية
            closest = min(actual_sync, key=lambda x: abs(x['timestamp'] - gt['timestamp']))
            error = abs(closest['timestamp'] - gt['timestamp'])
            total_error += error
            
            # تحقق بسيط من التطابق النصي (يمكن تحسينه بـ LLM لاحقاً)
            if gt['text'] in (closest.get('spoken_text') or ""):
                metrics["semantic_match_count"] += 1
        
        metrics["temporal_error"] = total_error / metrics["total_items"] if metrics["total_items"] > 0 else 0
        metrics["score"] = (metrics["semantic_match_count"] / metrics["total_items"]) * 100 if metrics["total_items"] > 0 else 0
        
        return metrics

    def run_benchmark(self, sync_data):
        """
        تشغيل اختبار أداء شامل.
        """
        start_time = time.time()
        # محاكاة عملية معالجة
        time.sleep(0.1) 
        duration = time.time() - start_time
        
        return {
            "throughput_fps": len(sync_data) / duration if duration > 0 else 0,
            "latency_ms": duration * 1000
        }

sync_evaluator = SyncEvaluator()
