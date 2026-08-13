"""
GPU Runtime Manager — إدارة نقل التدريب إلى CUDA ومراقبة VRAM
=============================================================
  1) Device detection (CUDA / CPU)
  2) نقل النموذج والبيانات إلى الجهاز
  3) تقدير batch size حسب VRAM
  4) عند OOM: تصغير الدفعة وإعادة المحاولة

يحترم NSM_ALLOW_GPU و budget.prefer_cpu_for_toy في guardrails.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("GPURuntime")


@dataclass
class DeviceInfo:
    device_str: str
    cuda: bool
    name: Optional[str] = None
    total_vram_gb: Optional[float] = None
    free_vram_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    reason: str = ""


def _env_force_gpu() -> bool:
    return os.environ.get("NSM_ALLOW_GPU", "").strip().lower() in ("1", "true", "yes")


def _prefer_cpu_for_toy() -> bool:
    try:
        from ai.training_sandbox import load_guardrails

        return bool((load_guardrails().get("budget") or {}).get("prefer_cpu_for_toy", True))
    except Exception:
        return True


def _allow_gpu_cfg() -> bool:
    try:
        from ai.training_sandbox import load_guardrails

        return bool((load_guardrails().get("budget") or {}).get("allow_gpu", True))
    except Exception:
        return True


def detect_device(force_gpu: Optional[bool] = None) -> DeviceInfo:
    """يفحص وجود CUDA ويختار الجهاز."""
    try:
        import torch
    except Exception as e:
        return DeviceInfo("cpu", False, reason=f"torch unavailable: {e}")

    cuda_ok = bool(torch.cuda.is_available())
    name = None
    total = free = None
    ver = getattr(getattr(torch, "version", None), "cuda", None)

    if cuda_ok:
        try:
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total = float(props.total_memory) / (1024**3)
            free_b, _total_b = torch.cuda.mem_get_info(0)
            free = float(free_b) / (1024**3)
        except Exception as e:
            logger.warning("cuda props: %s", e)

    use_gpu = False
    reason = "cpu_default"
    force = _env_force_gpu() if force_gpu is None else force_gpu

    if not cuda_ok:
        reason = "cuda_not_available"
    elif not _allow_gpu_cfg():
        reason = "allow_gpu=false in guardrails"
    elif force:
        use_gpu = True
        reason = "NSM_ALLOW_GPU or force"
    elif not _prefer_cpu_for_toy():
        use_gpu = True
        reason = "prefer_cpu_for_toy=false"
    else:
        reason = "prefer_cpu_for_toy (set NSM_ALLOW_GPU=1 to use GPU)"

    return DeviceInfo(
        device_str="cuda" if use_gpu else "cpu",
        cuda=cuda_ok,
        name=name,
        total_vram_gb=total,
        free_vram_gb=free,
        cuda_version=ver,
        reason=reason,
    )


def torch_device(force_gpu: Optional[bool] = None):
    import torch

    info = detect_device(force_gpu=force_gpu)
    return torch.device(info.device_str), info


def vram_snapshot() -> Dict[str, Any]:
    info = detect_device(force_gpu=True)  # report hardware even if policy prefers cpu
    out = {
        "cuda_available": info.cuda,
        "name": info.name,
        "total_vram_gb": info.total_vram_gb,
        "free_vram_gb": info.free_vram_gb,
        "cuda_version": info.cuda_version,
        "policy_device": detect_device().device_str,
        "policy_reason": detect_device().reason,
    }
    try:
        import torch

        if torch.cuda.is_available():
            out["allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
            out["reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
    except Exception:
        pass
    return out


def suggest_batch_size(
    n_samples: int,
    n_features: int = 64,
    free_vram_gb: Optional[float] = None,
    base: int = 32,
) -> int:
    """
    تقدير محافظ لحجم الدفعة حسب VRAM المتاح.
    على CPU أو VRAM غير معروف: دفعة معتدلة حسب عدد العيّنات.
    """
    n = max(1, int(n_samples))
    if free_vram_gb is None:
        snap = vram_snapshot()
        free_vram_gb = snap.get("free_vram_gb")

    # CPU / unknown
    if free_vram_gb is None or free_vram_gb <= 0:
        return max(4, min(base, n, 64))

    # rough: ~2-4MB per sample for small MLP; scale down aggressively under 2GB free
    if free_vram_gb < 1.0:
        bs = 4
    elif free_vram_gb < 2.0:
        bs = 8
    elif free_vram_gb < 4.0:
        bs = 16
    elif free_vram_gb < 8.0:
        bs = 32
    else:
        bs = 64

    # feature pressure
    if n_features > 512:
        bs = max(4, bs // 2)
    if n_features > 2048:
        bs = max(2, bs // 2)

    return max(1, min(bs, n, 128))


def is_oom_error(err: BaseException) -> bool:
    msg = f"{type(err).__name__}: {err}".lower()
    return any(
        k in msg
        for k in (
            "out of memory",
            "cuda out of memory",
            "cudnn out of memory",
            "oom",
            "cannot allocate memory",
        )
    )


def empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_with_oom_backoff(
    train_fn: Callable[[int], Any],
    initial_batch: int,
    min_batch: int = 1,
    max_retries: int = 4,
) -> Tuple[Any, int, List[str]]:
    """
    يشغّل train_fn(batch_size)؛ عند OOM يصغّر الدفعة للنصف ويعيد المحاولة.
    """
    bs = max(min_batch, int(initial_batch))
    log: List[str] = []
    last_err: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            empty_cache()
            result = train_fn(bs)
            log.append(f"ok batch={bs} attempt={attempt}")
            return result, bs, log
        except Exception as e:
            last_err = e
            if is_oom_error(e) and bs > min_batch:
                new_bs = max(min_batch, bs // 2)
                log.append(f"OOM batch={bs} → retry batch={new_bs}")
                empty_cache()
                bs = new_bs
                continue
            raise
    raise RuntimeError(f"OOM retries exhausted: {last_err}")


def device_report_md() -> str:
    snap = vram_snapshot()
    pol = detect_device()
    lines = [
        "## 🖥️ حالة الحوسبة (GPU Runtime)",
        f"- CUDA متاح في النظام: **{snap.get('cuda_available')}**",
        f"- اسم GPU: {snap.get('name') or '—'}",
        f"- VRAM كلي / حر: {snap.get('total_vram_gb') and round(snap['total_vram_gb'], 2)} / "
        f"{snap.get('free_vram_gb') and round(snap['free_vram_gb'], 2)} GB",
        f"- CUDA toolkit: {snap.get('cuda_version') or '—'}",
        f"- allocated/reserved: {snap.get('allocated_gb') and round(snap['allocated_gb'], 3)} / "
        f"{snap.get('reserved_gb') and round(snap['reserved_gb'], 3)} GB",
        f"- جهاز السياسة الحالي: **{pol.device_str}** ({pol.reason})",
        "",
        "لتفعيل GPU حتى مع prefer_cpu_for_toy: `NSM_ALLOW_GPU=1`",
    ]
    return "\n".join(lines)



def nvidia_smi_text(query: bool = True) -> str:
    """مخرجات nvidia-smi (نص) أو رسالة إن تعذّر."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return "nvidia-smi غير متاح على هذا الجهاز"
    try:
        if query:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,utilization.memory,"
                    "memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            out = (r.stdout or r.stderr or "").strip()
            if out:
                return out
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
        return ((r.stdout or "") + (r.stderr or "")).strip()[-4000:] or "لا مخرجات"
    except Exception as e:
        return f"nvidia-smi error: {e}"


def gpu_monitor_snapshot() -> Dict[str, Any]:
    """لقطة موحّدة: torch VRAM + nvidia-smi + اقتراح batch."""
    from datetime import datetime, timezone
    snap = vram_snapshot()
    smi = nvidia_smi_text(query=True)
    try:
        suggested = suggest_batch_size(base_batch=16, model_size_hint="medium")
    except Exception:
        suggested = None
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "vram": snap,
        "nvidia_smi_csv": smi,
        "suggested_batch": suggested,
        "tips_ar": [
            "Utilization منخفض جداً أثناء Running = غالباً تحميل بيانات أو عنق CPU",
            "memory.used ≈ total → خفّض BATCH",
            "Dual T4: تأكد أن التدريب يرى gpus>=2",
        ],
    }


def kaggle_gpu_monitor_snippet() -> str:
    """مقطع Python لمراقبة GPU داخل نواة Kaggle."""
    lines = [
        "def _nsm_gpu_monitor(tag=''):",
        "    import subprocess",
        "    try:",
        "        import torch",
        "        n = torch.cuda.device_count() if torch.cuda.is_available() else 0",
        "        print(f'[GPU {tag}] cuda={torch.cuda.is_available()} n={n}')",
        "        if torch.cuda.is_available():",
        "            for i in range(n):",
        "                a = torch.cuda.memory_allocated(i) / 1e9",
        "                r = torch.cuda.memory_reserved(i) / 1e9",
        "                print(f'  dev{i} alloc={a:.2f}GB reserved={r:.2f}GB name={torch.cuda.get_device_name(i)}')",
        "    except Exception as e:",
        "        print('torch monitor:', e)",
        "    try:",
        "        print(subprocess.check_output([",
        "            'nvidia-smi',",
        "            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',",
        "            '--format=csv,noheader',",
        "        ], text=True, timeout=10).strip())",
        "    except Exception as e:",
        "        print('smi:', e)",
    ]
    return "\n".join(lines) + "\n"
