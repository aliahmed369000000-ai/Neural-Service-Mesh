"""
Professional training driver — simulate 5000 rounds via running API.
Target: last_loss < 0.1
Strategy: 100 batches × 50 rounds = 5000 total rounds, no delay.
"""
import requests
import time
import json
import sys

API = "http://localhost:5000"
BATCH_SIZE   = 50
TOTAL_ROUNDS = 5000
TARGET_LOSS  = 0.1
BATCHES      = TOTAL_ROUNDS // BATCH_SIZE


def check_loss():
    try:
        r = requests.get(f"{API}/train/matrix", timeout=10)
        if r.ok:
            d = r.json()
            return d.get("last_loss"), d.get("train_steps", 0)
    except Exception:
        pass
    return None, 0


def run_batch(rounds=50):
    try:
        r = requests.post(
            f"{API}/ai/simulate",
            json={"rounds": rounds, "executions_per_round": 5},
            timeout=120,
        )
        if r.ok:
            d = r.json()
            sim = d.get("simulation", {})
            lp  = d.get("learning_proof", {})
            return sim.get("overall_success_rate", 0), lp.get("improvement_pct", 0)
    except Exception as e:
        print(f"  [batch error] {e}")
    return 0, 0


def save_weights():
    try:
        r = requests.post(f"{API}/train/save-weights", timeout=15)
        if r.ok:
            return r.json()
    except Exception as e:
        print(f"  [save error] {e}")
    return {}


def wait_for_api(max_wait=60):
    print("Waiting for API...", end="", flush=True)
    for _ in range(max_wait):
        try:
            r = requests.get(f"{API}/status", timeout=3)
            if r.ok:
                print(" ready.\n")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(1)
    print(" TIMEOUT")
    return False


def main():
    if not wait_for_api():
        sys.exit(1)

    print("=" * 65)
    print("  PROFESSIONAL TRAINING — 5000 rounds @ 5 exec/round")
    print(f"  Target: loss < {TARGET_LOSS}")
    print("=" * 65)

    loss_initial, steps_initial = check_loss()
    print(f"  Initial loss : {loss_initial}   steps : {steps_initial}")
    print()

    rounds_done = 0
    best_loss   = loss_initial if loss_initial is not None else 999.0

    for batch_num in range(1, BATCHES + 1):
        t0 = time.time()
        sr, improvement = run_batch(BATCH_SIZE)
        elapsed = time.time() - t0

        rounds_done += BATCH_SIZE
        loss, steps = check_loss()
        if loss is not None:
            best_loss = min(best_loss, loss)

        loss_str = f"{loss:.6f}" if loss is not None else "N/A"
        best_str = f"{best_loss:.6f}" if best_loss < 999.0 else "N/A"

        print(
            f"  Batch {batch_num:>3}/{BATCHES}  "
            f"rounds={rounds_done:>5}  "
            f"SR={sr:.1%}  "
            f"loss={loss_str}  "
            f"best={best_str}  "
            f"Δ={improvement:+.1f}%  "
            f"({elapsed:.1f}s)"
        )
        sys.stdout.flush()

        if loss is not None and loss < TARGET_LOSS:
            print(f"\n  ✅ TARGET REACHED — loss {loss:.6f} < {TARGET_LOSS}")
            break

    print()
    print("=" * 65)
    final_loss, final_steps = check_loss()
    print(f"  Final loss   : {final_loss}")
    print(f"  Total steps  : {final_steps}")
    print(f"  Best loss    : {best_loss:.6f}" if best_loss < 999.0 else "  Best loss    : N/A")
    print()

    print("  Saving neural weights...", end="", flush=True)
    save_result = save_weights()
    if save_result.get("saved"):
        print(f" saved → {save_result.get('path')}")
    else:
        print(f" result: {save_result}")

    print("=" * 65)
    print()

    report = {
        "total_rounds": rounds_done,
        "target_loss": TARGET_LOSS,
        "final_loss": final_loss,
        "best_loss": best_loss if best_loss < 999.0 else None,
        "final_steps": final_steps,
        "target_reached": final_loss is not None and final_loss < TARGET_LOSS,
    }
    with open("data/training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved → data/training_report.json")


if __name__ == "__main__":
    main()
