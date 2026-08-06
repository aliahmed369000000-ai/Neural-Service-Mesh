"""
قياس مرجعي دقيق لحزمة واحدة (single batch) — نموذج v3 (120M).

الهدف: عزل تكلفة حزمة واحدة بدقة (وقت + ذاكرة RSS)، بمعزل عن أي متوسط
عبر عدة حزم (PACKS_PER_RUN في train_batch_v3.py يُخفي التفاوت بين أول
حزمة وباقي الحزم — هنا كل حزمة تُقاس على حدة).

يستخدم نفس المعمارية والمسار بالضبط مثل train_batch_v3.py (لا تعديل
على أي كود مُختبَر) — إضافي بالكامل.

الاستخدام:
    python3 benchmark_single_batch_v3.py            # يقيس حزمة PACK_SIZE افتراضية (60)
    python3 benchmark_single_batch_v3.py 40 80 120   # يقيس عدة أحجام حزمة للمقارنة
"""
import sys, time, json, pickle, os, gc
sys.path.insert(0, '.')
import resource
from ai.arabic_transformer import ArabicTransformer

WEIGHTS_DIR = "models/transformer_ckg_v3"
STATE_FILE = "ckg_train_state_v3.json"
DEFAULT_PACK_SIZE = 60


def rss_mb() -> float:
    """ذروة RSS الحالية بالميغابايت (ru_maxrss بالكيلوبايت على لينكس)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_sentences():
    with open('ckg_sentences_v3.pkl', 'rb') as f:
        return pickle.load(f)


def load_model_at_current_position(sentences):
    m = ArabicTransformer(d_model=1216, n_heads=16, d_ff=2560, n_layers=8, vocab_size=8192)
    if os.path.exists(WEIGHTS_DIR):
        m.load(WEIGHTS_DIR)
    if os.path.exists(STATE_FILE):
        pos = json.loads(open(STATE_FILE).read()).get("position", 0)
    else:
        pos = 0
    return m, pos


def bench_one_pack_size(pack_size: int, sentences, pos: int):
    """يقيس حزمة واحدة معزولة بحجم pack_size، بدون تعديل حالة التدريب المحفوظة."""
    gc.collect()
    rss_before = rss_mb()

    m, _ = load_model_at_current_position(sentences)
    rss_after_load = rss_mb()

    end = min(pos + pack_size, len(sentences))
    pack = sentences[pos:end]
    if len(pack) < pack_size:
        pack = sentences[:pack_size]  # نلتف لبداية القائمة لو قربنا من النهاية

    t0 = time.time()
    loss = m.train_step_batch(pack)
    elapsed = time.time() - t0

    rss_after_step = rss_mb()

    return {
        "pack_size": pack_size,
        "elapsed_s": round(elapsed, 3),
        "loss": round(loss, 4),
        "rss_before_mb": round(rss_before, 1),
        "rss_after_load_mb": round(rss_after_load, 1),
        "rss_peak_after_step_mb": round(rss_after_step, 1),
        "model_load_delta_mb": round(rss_after_load - rss_before, 1),
        "step_delta_mb": round(rss_after_step - rss_after_load, 1),
    }


def main():
    pack_sizes = [int(a) for a in sys.argv[1:]] or [DEFAULT_PACK_SIZE]

    # ru_maxrss هي "ذروة تاريخية" ولا تنخفض حتى بعد gc — فلو قِسنا أكثر من
    # حجم حزمة داخل نفس العملية، القياس الثاني يرث ذروة الأول ويصبح غير
    # دقيق. لذلك: حجم واحد يُقاس مباشرة، وأكثر من حجم يُقاس كل واحد منهم
    # في عملية Python منفصلة تماماً (عزل حقيقي للذاكرة والوقت).
    if len(pack_sizes) > 1:
        results = []
        print(f"{'حجم الحزمة':>10} | {'الوقت (ث)':>10} | {'الخسارة':>8} | "
              f"{'ذاكرة النموذج (MB)':>18} | {'ذاكرة الحزمة (MB)':>18} | {'ذروة RSS (MB)':>14}")
        print("-" * 95)
        for ps in pack_sizes:
            out = os.popen(f"{sys.executable} {__file__} {ps}").read()
            for line in out.splitlines():
                if line.strip().startswith(str(ps)):
                    print(line)
            try:
                r = json.loads(open("benchmark_single_batch_v3_result.json").read())[0]
                results.append(r)
            except Exception:
                pass
        with open("benchmark_single_batch_v3_result.json", "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\nتم حفظ كل النتائج (بعزل كامل لكل حجم) في benchmark_single_batch_v3_result.json")
        return

    sentences = load_sentences()
    _, pos = load_model_at_current_position(sentences)

    print(f"موضع التدريب الحالي: {pos}/{len(sentences)}")
    print(f"{'حجم الحزمة':>10} | {'الوقت (ث)':>10} | {'الخسارة':>8} | "
          f"{'ذاكرة النموذج (MB)':>18} | {'ذاكرة الحزمة (MB)':>18} | {'ذروة RSS (MB)':>14}")
    print("-" * 95)

    r = bench_one_pack_size(pack_sizes[0], sentences, pos)
    print(f"{r['pack_size']:>10} | {r['elapsed_s']:>10} | {r['loss']:>8} | "
          f"{r['model_load_delta_mb']:>18} | {r['step_delta_mb']:>18} | "
          f"{r['rss_peak_after_step_mb']:>14}")

    with open("benchmark_single_batch_v3_result.json", "w") as f:
        json.dump([r], f, ensure_ascii=False, indent=2)
    print("\nتم حفظ النتائج التفصيلية في benchmark_single_batch_v3_result.json")


if __name__ == "__main__":
    main()
