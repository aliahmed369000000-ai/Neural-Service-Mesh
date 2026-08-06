#!/usr/bin/env python3
"""
من داخل Colab (أو أي عقدة تدريب): تدريب محلي ثم دفع الميتا إلى خادم NSM.
  export NSM_REMOTE_WEBHOOK_URL="https://YOUR_HOST/training/remote-results"
  export NSM_REMOTE_WEBHOOK_SECRET="optional-shared-secret"
  python scripts/colab_result_push.py --csv data/samples/classification_demo.csv --epochs 15
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/samples/classification_demo.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--prefer", default="auto")
    parser.add_argument("--webhook", default=os.environ.get("NSM_REMOTE_WEBHOOK_URL") or "")
    parser.add_argument("--secret", default=os.environ.get("NSM_REMOTE_WEBHOOK_SECRET") or "")
    parser.add_argument("--no-train", action="store_true", help="ادفع آخر job فقط إن وُجد")
    args = parser.parse_args()

    os.environ.setdefault("NSM_ALLOW_GPU", "1")

    from ai.remote_gpu_provider import (
        LocalGPUProvider,
        build_result_package,
        push_job_package,
    )

    if args.no_train:
        print("no-train: استخدم package يدوياً")
        return 1

    job = LocalGPUProvider().submit_train_csv(args.csv, epochs=args.epochs, prefer=args.prefer)
    print("job:", job.get("job_id"), job.get("status"), job.get("model_path"))
    pkg = build_result_package(job)
    print("package:", pkg.get("package_id"))

    if args.webhook:
        res = push_job_package(job, args.webhook, args.secret)
        print("push:", res)
    else:
        print("لا webhook — الحزمة محلية تحت artifacts/model_training/remote_jobs/")
    return 0 if job.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
