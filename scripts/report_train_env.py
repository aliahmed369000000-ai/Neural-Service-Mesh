#!/usr/bin/env python3
"""تقرير سريع لقدرة البيئة على تدريب ArabicTransformer v3 (120M)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# إعادة استخدام منطق train_batch_v3 بدون استيراد ثقيل للنموذج
import importlib.util

spec = importlib.util.spec_from_file_location(
    "train_batch_v3",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_batch_v3.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

avail = mod.available_ram_gb()
pack = mod.choose_pack_size(avail)
packs = mod.choose_packs_per_run(avail)

print("=== NSM Training Environment Report ===")
print(f"Available RAM:     {avail:.2f} GiB")
print(f"Recommended PACK_SIZE:      {pack if pack else 'ABORT (too low)'}")
print(f"Recommended PACKS_PER_RUN:  {packs}")
if pack:
    per_run = pack * packs
    print(f"Sentences / run (approx):   up to {per_run}")
    print("Verdict:                 READY (within adaptive limits)")
else:
    print("Verdict:                 NOT READY — need more RAM for 120M training")
print("Override:  NSM_PACK_SIZE=N NSM_PACKS_PER_RUN=M python3 train_batch_v3.py")
