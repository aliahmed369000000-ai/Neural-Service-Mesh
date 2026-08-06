"""
Hardware-Aware Optimization — وعي عتادي للكود والتدريب
=======================================================
  • بصمة GPU (T4 / V100 / A100 / H100 / CPU)
  • توصيات batch / AMP / TF32 / channels_last
  • مقتطفات Triton اختيارية (إن وُجد) أو PyTorch عالي الكفاءة
لا يدّعي توليد CUDA تجاري كامل؛ يوجّه الاستغلال الأقصى بشكل عملي وآمن.
"""
from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("HardwareAware")

ROOT = Path(__file__).resolve().parent.parent
HW_DIR = ROOT / "artifacts" / "model_training" / "meta_ai" / "hardware"
HW_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


GPU_PROFILES: Dict[str, Dict[str, Any]] = {
    "cpu": {
        "amp": False,
        "batch": 32,
        "compile": False,
        "notes_ar": "لا CUDA — ركّز على vectorization وعدد خيوط محدودة.",
    },
    "t4": {
        "amp": True,
        "batch": 128,
        "compile": True,
        "tensor_cores": True,
        "notes_ar": "Turing — AMP مفيد جداً؛ Dual T4 → DataParallel.",
    },
    "v100": {
        "amp": True,
        "batch": 256,
        "compile": True,
        "tensor_cores": True,
        "notes_ar": "Volta — فعّل TF32 بحذر؛ VRAM 16/32GB.",
    },
    "a100": {
        "amp": True,
        "batch": 512,
        "compile": True,
        "tf32": True,
        "notes_ar": "Ampere — TF32 + torch.compile غالباً يرفعان الإنتاجية.",
    },
    "h100": {
        "amp": True,
        "batch": 512,
        "compile": True,
        "tf32": True,
        "fp8_hint": True,
        "notes_ar": "Hopper — استغل FP8/Transformer Engine عند التوفر؛ Triton لنوى مخصصة.",
    },
    "l4": {
        "amp": True,
        "batch": 192,
        "compile": True,
        "notes_ar": "Ada Lovelace منخفض الطاقة — AMP + batch متوسط.",
    },
}


def detect_gpu_family() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        name = (torch.cuda.get_device_name(0) or "").lower()
        for key in ("h100", "a100", "v100", "l4", "t4"):
            if key in name:
                return key
        if "4090" in name or "3090" in name or "ada" in name:
            return "l4"
        return "t4"
    except Exception:
        return "cpu"


@dataclass
class HardwarePlan:
    ok: bool
    family: str
    device_name: str
    recommendations: Dict[str, Any]
    code_snippet: str
    narrative_ar: str
    created_at: str = field(default_factory=_now)

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "## ⚙️ تحسين واعٍ بالعتاد",
                f"- العائلة: **{self.family}**",
                f"- الجهاز: `{self.device_name}`",
                f"- توصيات: `{json.dumps(self.recommendations, ensure_ascii=False)}`",
                "",
                self.narrative_ar,
                "",
                "### مقتطف موصى",
                "```python",
                self.code_snippet,
                "```",
            ]
        )


def plan_for_device(force_family: Optional[str] = None) -> HardwarePlan:
    family = (force_family or detect_gpu_family()).lower()
    if family not in GPU_PROFILES:
        family = "cpu"
    profile = dict(GPU_PROFILES[family])
    device_name = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    snippet = textwrap.dedent(
        f'''
        import torch
        torch.backends.cuda.matmul.allow_tf32 = {str(bool(profile.get("tf32", False)))}
        torch.backends.cudnn.allow_tf32 = {str(bool(profile.get("tf32", False)))}
        use_amp = {str(bool(profile.get("amp", False)))}
        batch_size = {int(profile.get("batch", 32))}
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        # اختياري: model = torch.compile(model)  # PyTorch 2+
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        # تدريب:
        # with torch.cuda.amp.autocast(enabled=use_amp):
        #     loss = criterion(model(x), y)
        '''
    ).strip()

    # Triton تلميح فقط — لا نفرض التثبيت
    if family in ("a100", "h100", "t4"):
        snippet += textwrap.dedent(
            '''

            # تلميح Triton (اختياري إن magentos/triton متاح):
            # اكتب نواة fused لـ elementwise إن كان الملف الشخصي يظهر bottleneck في kernel launch
            '''
        )

    narrative = (
        f"اكتُشفت عائلة `{family}`. {profile.get('notes_ar', '')} "
        f"batch مقترح={profile.get('batch')}, AMP={profile.get('amp')}. "
        "هذا يقارب استغلال الترانزستورات عبر إعدادات الإطار لا عبر CUDA يدوي غير آمن."
    )
    plan = HardwarePlan(
        ok=True,
        family=family,
        device_name=device_name,
        recommendations={k: v for k, v in profile.items() if k != "notes_ar"},
        code_snippet=snippet,
        narrative_ar=narrative,
    )
    path = HW_DIR / f"hw_plan_{family}.md"
    path.write_text(plan.to_markdown(), encoding="utf-8")
    return plan


def handle_hardware_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(
        r"(عتاد|hardware|كرت\s*الشاش|gpu\s*profile|تحسين\s*عتاد|triton|cuda\s*optim|h100|a100|t4\s*optim)",
        text,
        re.I,
    ):
        return None
    force = None
    for key in ("h100", "a100", "v100", "l4", "t4", "cpu"):
        if re.search(rf"\b{key}\b", text, re.I):
            force = key
            break
    return plan_for_device(force_family=force).to_markdown()
