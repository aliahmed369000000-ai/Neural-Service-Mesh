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


def main() -> None:
    os.environ.setdefault("NSM_ALLOW_GPU", "1")
    os.environ.setdefault("NSM_OFFLINE_MODE", "0")
    # تبعيات خفيفة للتدريب (Colab غالباً فيه torch مسبقاً)
    pkgs = ["numpy"]
    try:
        import torch  # noqa: F401
        print("torch موجود مسبقاً")
    except Exception:
        pkgs.append("torch")
    print("تثبيت:", pkgs)
    pip_install(pkgs)

    sys.path.insert(0, str(ROOT))
    from ai.gpu_runtime import device_report_md, detect_device

    print(device_report_md())
    d = detect_device(force_gpu=True)
    print("force_gpu detect:", d)


if __name__ == "__main__":
    main()
