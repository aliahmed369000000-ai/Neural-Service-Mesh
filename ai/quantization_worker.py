"""
Quantization Worker — ضغط أوزان .npy من float32 إلى int8
======================================================
يحفظ scale + zero_point بجانب الملف. لا يحذف الأصل إلا بعلم.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "model_training" / "quantized"
OUT.mkdir(parents=True, exist_ok=True)


def quantize_array(arr: np.ndarray, bits: int = 8) -> Dict[str, Any]:
    bits = 8 if bits not in (4, 8) else bits
    x = np.asarray(arr, dtype=np.float32)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if bits == 8:
        qmin, qmax = -128, 127
        dtype = np.int8
    else:
        qmin, qmax = -8, 7
        dtype = np.int8  # store nibble-scale in int8 range
    if xmax == xmin:
        scale = 1.0
        zp = 0
        q = np.zeros_like(x, dtype=dtype)
    else:
        scale = (xmax - xmin) / float(qmax - qmin)
        zp = int(round(qmin - xmin / scale))
        q = np.clip(np.round(x / scale + zp), qmin, qmax).astype(dtype)
    return {"q": q, "scale": scale, "zero_point": zp, "bits": bits, "shape": list(x.shape), "dtype_orig": "float32"}


def dequantize(meta: Dict[str, Any], q: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) - float(meta["zero_point"])) * float(meta["scale"])


def quantize_npy_file(path: str, bits: int = 8, keep_original: bool = True) -> Dict[str, Any]:
    src = Path(path)
    if not src.is_file():
        return {"ok": False, "error": f"missing {path}"}
    arr = np.load(str(src), allow_pickle=False)
    orig_bytes = src.stat().st_size
    pack = quantize_array(arr, bits=bits)
    stem = src.stem
    out_dir = OUT / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    qpath = out_dir / f"{stem}.int{bits}.npy"
    mpath = out_dir / f"{stem}.int{bits}.meta.json"
    np.save(str(qpath), pack["q"])
    meta = {k: v for k, v in pack.items() if k != "q"}
    meta.update({"source": str(src), "at": datetime.now(timezone.utc).isoformat()})
    mpath.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    new_bytes = qpath.stat().st_size + mpath.stat().st_size
    return {
        "ok": True,
        "source": str(src),
        "quantized": str(qpath.relative_to(ROOT)),
        "meta": str(mpath.relative_to(ROOT)),
        "orig_bytes": orig_bytes,
        "new_bytes": new_bytes,
        "ratio": round(new_bytes / max(orig_bytes, 1), 3),
        "keep_original": keep_original,
    }


def quantize_models_dir(pattern: str = "models/**/*.npy", bits: int = 8, limit: int = 20) -> Dict[str, Any]:
    files = list(ROOT.glob(pattern))[:limit]
    results = []
    for f in files:
        if f.stat().st_size < 1024:
            continue
        results.append(quantize_npy_file(str(f), bits=bits))
    return {"ok": True, "n": len(results), "results": results}


def handle_quant_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(r"(تكميم|كم[ّ]?م|quantiz|ضغط\s*اوزان)", text, re.I):
        return None
    bits = 8
    if re.search(r"int4|4\s*بت", text, re.I):
        bits = 4
    # prefer routing weights as demo if large transformer missing
    candidates = list((ROOT / "models").glob("**/*.npy")) if (ROOT / "models").is_dir() else []
    if candidates:
        # largest first
        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
        r = quantize_npy_file(str(candidates[0]), bits=bits)
        return "## 📦 تكميم أوزان\n```json\n" + json.dumps(r, ensure_ascii=False, indent=2) + "\n```"
    r = quantize_models_dir(bits=bits, limit=5)
    return "## 📦 تكميم أوزان\n```json\n" + json.dumps(r, ensure_ascii=False, indent=2, default=str)[:3000] + "\n```"
