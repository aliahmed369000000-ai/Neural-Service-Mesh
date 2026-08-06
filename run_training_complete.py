"""
مشغّل تدريب كامل حتى DONE_ALL — يستدعي train_batch_v3.main() مباشرةً
بدون عمليات shell فرعية، فيستمر في نفس العملية بدون انقطاع.
"""
import sys, time, json, os

sys.path.insert(0, ".")

STATE_FILE = "ckg_train_state_v3.json"
LOG_FILE   = "logs/training_full.log"

os.makedirs("logs", exist_ok=True)


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_pos() -> int:
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return int(json.load(f).get("position", 0))
    except Exception:
        return 0


if __name__ == "__main__":
    import train_batch_v3 as tb

    log("═" * 50)
    log("بدء حلقة التدريب الكاملة — نموذج 200M")
    log("═" * 50)

    run_n = 0
    while True:
        run_n += 1
        pos_before = read_pos()
        log(f"تشغيل #{run_n}  (pos={pos_before})")

        t0 = time.time()
        ret = tb.main()
        elapsed = time.time() - t0

        pos_after = read_pos()
        log(f"  ← انتهى في {elapsed:.0f}ث  pos={pos_after}/26375"
            f"  ({100*pos_after/26375:.1f}%)  exit={ret}")

        if ret == 0 and pos_after == 0:
            # main() طبعت DONE_ALL بالفعل
            log("✅ DONE_ALL — التدريب مكتمل 100%")
            break

        if ret != 0:
            log(f"❌ توقف بخطأ (exit={ret})")
            break

        if pos_after >= 26375:
            log("✅ DONE_ALL — التدريب مكتمل 100%")
            break
