import time
import json
import statistics
from ai.autonomous_tools import security_scanner

def run_performance_test():
    print("🚀 بدء اختبار أداء وضغط نظام الدفاع الأمني...")
    
    test_cases = [
        {
            "name": "كود بسيط (Safe)",
            "code": "print('Hello World')"
        },
        {
            "name": "كود متوسط (Complex Safe)",
            "code": """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
print(calculate_fibonacci(10))
"""
        },
        {
            "name": "كود مشبوه (Obfuscated Attack)",
            "code": "import base64; exec(base64.b64decode('cHJpbnQoImhhY2tlZCIp'))"
        },
        {
            "name": "كود ضخم (Large File)",
            "code": "x = 1\n" * 1000 + "eval('x + 1')"
        }
    ]
    
    results = {}
    
    for case in test_cases:
        print(f"\n📊 اختبار: {case['name']}")
        latencies = []
        for i in range(50):  # 50 iterations per case
            start_time = time.perf_counter()
            security_scanner({"code": case['code']})
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000) # ms
        
        results[case['name']] = {
            "avg": statistics.mean(latencies),
            "max": max(latencies),
            "min": min(latencies),
            "p95": statistics.quantiles(latencies, n=20)[18] # 95th percentile
        }
        
        print(f"   - متوسط زمن الاستجابة: {results[case['name']]['avg']:.2f} ms")
        print(f"   - أقصى زمن استجابة: {results[case['name']]['max']:.2f} ms")

    print("\n🏁 ملخص أداء نظام الدفاع الأمني:")
    print("-" * 50)
    print(f"{'الحالة':<25} | {'المتوسط (ms)':<15} | {'P95 (ms)':<10}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<25} | {metrics['avg']:<15.2f} | {metrics['p95']:<10.2f}")
    
    # قياس التأثير على الإنتاجية (Throughput)
    total_start = time.perf_counter()
    total_requests = 200
    for _ in range(total_requests):
        security_scanner({"code": test_cases[1]['code']})
    total_end = time.perf_counter()
    
    throughput = total_requests / (total_end - total_start)
    print(f"\n📈 معدل الإنتاجية (Throughput): {throughput:.2f} فحص/ثانية")

if __name__ == "__main__":
    run_performance_test()
