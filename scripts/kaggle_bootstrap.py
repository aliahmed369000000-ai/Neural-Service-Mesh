#!/usr/bin/env python3
"""
تهيئة NSM Training Agent داخل Kaggle Notebook.
الاستخدام في خلية Kaggle:

  !git clone https://github.com/aliahmed369000000-ai/Neural-Service-Mesh.git
  %cd Neural-Service-Mesh
  %run scripts/kaggle_bootstrap.py

ثم فعّل Accelerator → GPU T4 x2 من إعدادات الدفتر.
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


def main() -> None:
    os.environ.setdefault("NSM_ALLOW_GPU", "1")
    os.environ.setdefault("NSM_OFFLINE_MODE", "0")
    # على Kaggle غالباً torch موجود مع CUDA
    pkgs = ["numpy"]
    try:
        import torch  # noqa: F401

        print("torch موجود:", getattr(torch, "__version__", "?"))
        print("cuda available:", torch.cuda.is_available())
        print("device_count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
    except Exception:
        pkgs.append("torch")

    # kaggle اختياري (للـAPI من داخل الدفتر نادراً ما يُحتاج)
    try:
        import kaggle  # noqa: F401
    except Exception:
        pass  # لا نفرض التثبيت

    print("تثبيت:", pkgs)
    if pkgs:
        pip_install(pkgs)

    sys.path.insert(0, str(ROOT))

    try:
        from ai.gpu_runtime import device_report_md, detect_device

        print(device_report_md())
        d = detect_device(force_gpu=True)
        print("force_gpu detect:", d)
    except Exception as e:
        print("gpu_runtime:", e)

    try:
        from ai.kaggle_provider import detect_kaggle_gpus, kaggle_notebook_status_report, wrap_model_for_multi_gpu

        print("\n" + kaggle_notebook_status_report())
        print("\nGPU detail:", detect_kaggle_gpus())
    except Exception as e:
        print("kaggle_provider:", e)

    print("\n✅ bootstrap Kaggle اكتمل. أوامر مفيدة:")
    print("  from ai.kaggle_provider import handle_kaggle_command")
    print("  print(handle_kaggle_command('حالة kaggle'))")
    print("  print(handle_kaggle_command('جهّز kaggle'))")


if __name__ == "__main__":
    main()
