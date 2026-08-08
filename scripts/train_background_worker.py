#!/usr/bin/env python3
"""
Background Worker — يراقب ckg_train_state_v3.json ويطبّق DevOps عند التحسّن/الثبات.
لا يستبدل الأوزان بلا تحقق. التشغيل:
  python3 scripts/train_background_worker.py --once
  python3 scripts/train_background_worker.py --loop --interval 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def cycle() -> dict:
    from ai.autonomous_train_devops import devops_cycle, load_train_state
    state = load_train_state()
    report = devops_cycle(run_train=False)
    # إن تحسّنت الخسارة مؤخراً: اقترح توسيع بيانات (إشارة فقط)
    tail = state.get("loss_history_tail") or []
    improved = False
    if len(tail) >= 8:
        improved = float(sum(tail[-4:]) / 4) < float(sum(tail[-8:-4]) / 4) - 0.05
    report["loss_improving"] = improved
    if improved:
        report["suggest_ar"] = (
            "الخسارة تنخفض — يمكن توسيع ckg_sentences_*.pkl وتشغيل "
            "bash run_training_loop.sh أو train_batch_v3.py بعد بوابة الأمان."
        )
    out = ROOT / "artifacts" / "model_training" / "train_devops" / "worker_last.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    args = ap.parse_args()
    if args.loop:
        while True:
            r = cycle()
            print(json.dumps({"plateau": r.get("plateau"), "improving": r.get("loss_improving")}, ensure_ascii=False))
            time.sleep(max(30, args.interval))
    else:
        r = cycle()
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
