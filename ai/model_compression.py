"""
Model Compression — تكميم وتقليم للشبكات
========================================
  • Quantization: float32 → int8 (مقاييس لكل-موتّر) مع تقدير خطأ إعادة البناء
  • Pruning: إزالة أوزان ضعيفة بالنسبة المئوية (magnitude pruning)
يعمل على state_dict لـ PyTorch أو قواميس numpy، بدون اعتماد إلزامي إضافي.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ModelCompression")

ROOT = Path(__file__).resolve().parent.parent
COMP_DIR = ROOT / "artifacts" / "model_training" / "architect" / "compression"
COMP_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tensor_to_numpy(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def quantize_int8(weight: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """تكميم متماثل per-tensor إلى int8. يعيد (q, scale, zero_point=0)."""
    w = weight.astype(np.float32)
    max_abs = float(np.max(np.abs(w))) if w.size else 1.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    return q, scale, 0.0


def dequantize_int8(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale


def prune_magnitude(weight: np.ndarray, sparsity: float) -> Tuple[np.ndarray, float]:
    """يصفّر أصغر |w| حسب نسبة sparsity ∈ [0,1]."""
    sparsity = float(np.clip(sparsity, 0.0, 0.95))
    w = weight.astype(np.float32).copy()
    if w.size == 0 or sparsity <= 0:
        return w, 0.0
    flat = np.abs(w).ravel()
    thresh = np.quantile(flat, sparsity)
    mask = np.abs(w) >= thresh
    kept = float(mask.mean())
    w = w * mask
    return w, 1.0 - kept


@dataclass
class CompressionReport:
    ok: bool
    method: str
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    sparsity: float = 0.0
    reconstruction_mse: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    output_path: str = ""
    created_at: str = field(default_factory=_now)
    narrative_ar: str = ""

    def to_markdown(self) -> str:
        return "\n".join(
            [
                f"## 📦 تقرير ضغط النموذج — {self.method}",
                f"- الحجم الأصلي: **{self.original_bytes:,}** بايت",
                f"- بعد الضغط: **{self.compressed_bytes:,}** بايت",
                f"- نسبة الضغط: **{self.compression_ratio:.2f}×**",
                f"- sparsity: **{self.sparsity:.1%}**" if self.sparsity else "",
                f"- MSE إعادة البناء: **{self.reconstruction_mse:.6f}**",
                f"- المخرج: `{self.output_path}`",
                "",
                self.narrative_ar,
            ]
        )


def _state_dict_from_demo() -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    return {
        "net.0.weight": rng.normal(size=(64, 32)).astype(np.float32),
        "net.0.bias": rng.normal(size=(64,)).astype(np.float32),
        "net.2.weight": rng.normal(size=(2, 64)).astype(np.float32),
        "net.2.bias": rng.normal(size=(2,)).astype(np.float32),
    }


def load_state_flexible(path: Optional[Path] = None) -> Dict[str, np.ndarray]:
    if path and path.is_file():
        try:
            import torch

            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict) and "state_dict" in obj:
                obj = obj["state_dict"]
            if isinstance(obj, dict):
                return {k: _tensor_to_numpy(v) for k, v in obj.items() if hasattr(v, "shape") or isinstance(v, (list, np.ndarray))}
        except Exception as e:
            logger.warning("load failed %s: %s", path, e)
    return _state_dict_from_demo()


def compress_quantize(
    state: Optional[Dict[str, np.ndarray]] = None,
    model_path: Optional[str] = None,
) -> CompressionReport:
    state = state or load_state_flexible(Path(model_path) if model_path else None)
    orig = 0
    comp = 0
    mse_acc = []
    q_state: Dict[str, Any] = {}
    for k, w in state.items():
        arr = np.asarray(w, dtype=np.float32)
        orig += arr.nbytes
        if arr.size < 8 or arr.ndim == 0:
            q_state[k] = {"type": "fp32", "data": arr}
            comp += arr.nbytes
            continue
        q, scale, zp = quantize_int8(arr)
        recon = dequantize_int8(q, scale)
        mse_acc.append(float(np.mean((arr - recon) ** 2)))
        q_state[k] = {"type": "int8", "scale": scale, "zero_point": zp, "data": q}
        comp += q.nbytes + 8  # scale overhead approx
    ratio = (orig / comp) if comp else 1.0
    out = COMP_DIR / f"quant_int8_{int(time.time())}.npz"
    np.savez_compressed(out, **{k: v["data"] if isinstance(v, dict) else v for k, v in q_state.items()})
    meta = {k: {kk: vv for kk, vv in val.items() if kk != "data"} for k, val in q_state.items() if isinstance(val, dict)}
    meta_path = out.with_suffix(".json")
    meta_path.write_text(
        json.dumps({"tensors": meta, "method": "int8_per_tensor", "created_at": _now()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    mse = float(np.mean(mse_acc)) if mse_acc else 0.0
    report = CompressionReport(
        ok=True,
        method="quantization_int8",
        original_bytes=orig,
        compressed_bytes=comp,
        compression_ratio=ratio,
        reconstruction_mse=mse,
        output_path=str(out.relative_to(ROOT)),
        narrative_ar=(
            f"تم تكميم الأوزان إلى int8 بنسبة ضغط تقريبية {ratio:.1f}× "
            f"وخطأ إعادة بناء MSE={mse:.6f}. مناسب للأجهزة الضعيفة مع اختبار دقة سريع بعد النشر."
        ),
        details={"meta": str(meta_path.relative_to(ROOT))},
    )
    (COMP_DIR / f"report_quant_{int(time.time())}.md").write_text(report.to_markdown(), encoding="utf-8")
    return report


def compress_prune(
    state: Optional[Dict[str, np.ndarray]] = None,
    model_path: Optional[str] = None,
    sparsity: float = 0.5,
) -> CompressionReport:
    state = state or load_state_flexible(Path(model_path) if model_path else None)
    orig = 0
    nonzero_before = 0
    nonzero_after = 0
    pruned: Dict[str, np.ndarray] = {}
    for k, w in state.items():
        arr = np.asarray(w, dtype=np.float32)
        orig += arr.nbytes
        nonzero_before += int(np.count_nonzero(arr))
        if arr.ndim >= 2:
            pw, _ = prune_magnitude(arr, sparsity)
        else:
            pw = arr  # لا تقلّم الانحياز عادة
        pruned[k] = pw
        nonzero_after += int(np.count_nonzero(pw))
    actual_sp = 1.0 - (nonzero_after / max(nonzero_before, 1))
    # حجم فعّال تقريبي عند التخزين المتناثر
    comp = int(orig * (1 - actual_sp) + 64 * len(pruned))
    out = COMP_DIR / f"pruned_{int(time.time())}.npz"
    np.savez_compressed(out, **pruned)
    report = CompressionReport(
        ok=True,
        method=f"magnitude_pruning_{sparsity:.0%}",
        original_bytes=orig,
        compressed_bytes=comp,
        compression_ratio=(orig / comp) if comp else 1.0,
        sparsity=actual_sp,
        reconstruction_mse=0.0,
        output_path=str(out.relative_to(ROOT)),
        narrative_ar=(
            f"تقليم بالمقدار بنسبة مستهدفة {sparsity:.0%} → sparsity فعلي {actual_sp:.1%}. "
            "احذف الروابط الضعيفة ثم أعد تدريباً قصيراً (fine-tune) لاسترجاع الدقة قبل النشر على الجوال/الراوتر."
        ),
    )
    (COMP_DIR / f"report_prune_{int(time.time())}.md").write_text(report.to_markdown(), encoding="utf-8")
    return report


def compress_pipeline(sparsity: float = 0.4) -> Dict[str, Any]:
    """تقليم ثم تكميم — مسار أجهزة ضعيفة."""
    state = _state_dict_from_demo()
    # prune in place
    pruned = {}
    for k, w in state.items():
        arr = np.asarray(w, dtype=np.float32)
        pruned[k] = prune_magnitude(arr, sparsity)[0] if arr.ndim >= 2 else arr
    pr = compress_prune(state=state, sparsity=sparsity)
    qr = compress_quantize(state=pruned)
    return {"prune": asdict(pr), "quantize": asdict(qr)}


def handle_compression_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(r"(ضغط|كم[ّ]?م|quantiz|prun|تقليم|compression|خف[ّ]?ف\s*نموذج)", text, re.I):
        return None
    if re.search(r"(تقليم|prun)", text, re.I):
        sp = 0.5
        m = re.search(r"(\d{1,2})\s*%", text)
        if m:
            sp = max(0.05, min(0.9, int(m.group(1)) / 100.0))
        r = compress_prune(sparsity=sp)
        return r.to_markdown()
    if re.search(r"(كم[ّ]?م|quantiz|int8)", text, re.I):
        r = compress_quantize()
        return r.to_markdown()
    # pipeline كامل
    pipe = compress_pipeline(0.4)
    return (
        "## 📦 مسار ضغط كامل (تقليم → تكميم)\n\n"
        + "```json\n"
        + json.dumps(pipe, ensure_ascii=False, indent=2)[:3500]
        + "\n```"
    )
