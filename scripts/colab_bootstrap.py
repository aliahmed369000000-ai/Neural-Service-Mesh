#!/usr/bin/env python3
"""
تهيئة NSM Training Agent داخل Google Colab (أو أي بيئة مشابهة).
الاستخدام في خلية Colab:
  !git clone https://github.com/aliahmed369000000-ai/Neural-Service-Mesh.git
  %cd Neural-Service-Mesh
  %run scripts/colab_bootstrap.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pip_install(packages: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + packages
    subprocess.check_call(cmd)


def print_orchestrator_hints() -> None:
    print(
        """
══════════════════════════════════════
أوامر الوكيل بعد التهيئة:
  from ai.remote_training_orchestrator import handle_orchestrator_command
  print(handle_orchestrator_command("حالة المنصات"))
  print(handle_orchestrator_command("خطة كفاءة"))

تدريب فعّال على GPU Colab:
  from ai.remote_training_orchestrator import efficient_nn_training_source
  exec(compile(efficient_nn_training_source("colab_local"), "t.py", "exec"))

دفع النتائج لخادمك (اختياري):
  export NSM_REMOTE_WEBHOOK_URL=https://YOUR_HOST/training/remote-results
  %run scripts/colab_result_push.py --csv data/samples/classification_demo.csv
══════════════════════════════════════
"""
    )


def main() -> None:
    os.environ.setdefault("NSM_ALLOW_GPU", "1")
    os.environ.setdefault("NSM_OFFLINE_MODE", "0")
    pkgs = ["numpy"]
    try:
        import torch  # noqa: F401
        print("torch موجود مسبقاً")
    except Exception:
        pkgs.append("torch")
    print("تثبيت:", pkgs)
    if pkgs:
        pip_install(pkgs)

    sys.path.insert(0, str(ROOT))
    try:
        from ai.gpu_runtime import device_report_md, detect_device
        print(device_report_md())
        print("force_gpu detect:", detect_device(force_gpu=True))
    except Exception as e:
        print("gpu_runtime:", e)

    try:
        from ai.remote_training_orchestrator import platforms_status
        print(platforms_status())
    except Exception as e:
        print("orchestrator:", e)

    print_orchestrator_hints()


if __name__ == "__main__":
    main()
