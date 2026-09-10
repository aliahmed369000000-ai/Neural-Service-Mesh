"""اكتشاف قدرات العقدة فعلياً دون تشغيل موارد خارجية أو تحميل نماذج."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import time
from typing import Any, Dict, List


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def collect_capabilities() -> Dict[str, Any]:
    """يجمع قدرات قابلة لإعادة الإنتاج؛ لا يدّعي توفر GPU أو نموذج دون فحص."""
    caps: List[str] = ["CPU", "text", "storage", "checkpoint"]
    facts: Dict[str, Any] = {
        "platform": platform.system().lower(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
        "modules": {},
        "cuda": {"available": False, "device_count": 0, "names": []},
        "binaries": {"git": shutil.which("git") is not None},
    }

    for module in ("aiohttp", "websockets", "numpy", "torch", "tensorflow", "huggingface_hub"):
        facts["modules"][module] = _has_module(module)

    if facts["modules"].get("aiohttp") or facts["modules"].get("websockets"):
        caps.append("web")
    if facts["modules"].get("torch"):
        try:
            import torch
            cuda = bool(torch.cuda.is_available())
            count = int(torch.cuda.device_count()) if cuda else 0
            names = [str(torch.cuda.get_device_name(i)) for i in range(count)] if cuda else []
            facts["cuda"] = {"available": cuda, "device_count": count, "names": names}
            if count:
                caps.append("GPU_LOW")
                if count >= 2:
                    caps.append("GPU_HIGH")
            caps.append("tf_engine")
        except Exception as exc:
            facts["cuda_error"] = type(exc).__name__
    if facts["modules"].get("tensorflow"):
        caps.append("tf_engine")

    # إزالة التكرار مع ترتيب ثابت حتى تكون البصمة قابلة للمقارنة.
    caps = sorted(set(caps))
    canonical = json.dumps({"capabilities": caps, "facts": facts}, sort_keys=True, separators=(",", ":"))
    return {
        "capabilities": caps,
        "facts": facts,
        "attestation_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20],
        "measured_at": time.time(),
        "source": "local_probe",
    }


def capability_names() -> List[str]:
    return list(collect_capabilities()["capabilities"])
