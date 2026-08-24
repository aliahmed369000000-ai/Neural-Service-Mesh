
import time
import concurrent.futures

def mock_visual_tool(params):
    """محاكاة أداة بصرية تستغرق وقت ثانية واحدة."""
    time.sleep(1)
    return {"status": "success", "data": "visual_result"}

def mock_audio_tool(params):
    """محاكاة أداة صوتية تستغرق وقت ثانية واحدة."""
    time.sleep(1)
    return {"status": "success", "data": "audio_result"}

def run_sequential():
    start = time.time()
    res1 = mock_visual_tool({})
    res2 = mock_audio_tool({})
    return time.time() - start

def run_parallel():
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(mock_visual_tool, {})
        f2 = executor.submit(mock_audio_tool, {})
        concurrent.futures.wait([f1, f2])
    return time.time() - start

if __name__ == "__main__":
    print("⏳ بدء اختبار زمن الاستجابة (Latency Benchmarking)...")
    seq_time = run_sequential()
    print(f"🐢 الزمن التسلسلي: {seq_time:.2f} ثانية")
    
    par_time = run_parallel()
    print(f"⚡ الزمن المتوازي: {par_time:.2f} ثانية")
    
    improvement = ((seq_time - par_time) / seq_time) * 100
    print(f"🚀 نسبة التحسين: {improvement:.1f}%")
