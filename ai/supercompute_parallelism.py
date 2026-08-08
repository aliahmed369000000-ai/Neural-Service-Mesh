"""
Supercomputing Orchestration — توازي ثلاثي الأبعاد + توسّع عنقودي
================================================================
  • خطط Data / Tensor / Pipeline Parallelism
  • قوالب إعداد DeepSpeed / Megatron-أسلوب (JSON/YAML نصي)
  • نموذج توسّع ديناميكي (scale up/down) — قرار لا تنفيذ سحابي أعمى

بدون مفاتيح سحابة لا يُستأجر عتاد حقيقي؛ المخرج خطة + ملفات تشغيل.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("SuperCompute")

ROOT = Path(__file__).resolve().parent.parent
SC_DIR = ROOT / "artifacts" / "model_training" / "super_ai" / "compute"
SC_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ParallelismPlan:
    ok: bool
    model_params_b: float
    n_gpus: int
    data_parallel: int
    tensor_parallel: int
    pipeline_parallel: int
    micro_batch: int
    global_batch: int
    deepspeed_config: Dict[str, Any]
    narrative_ar: str
    k8s_hint: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "## 🖥️ توازي ثلاثي الأبعاد (3D Parallelism)",
                f"- حجم النموذج التقريبي: **{self.model_params_b:.1f}B** معلمة",
                f"- GPUs: **{self.n_gpus}**",
                f"- DP={self.data_parallel} · TP={self.tensor_parallel} · PP={self.pipeline_parallel}",
                f"- micro_batch={self.micro_batch} → global_batch={self.global_batch}",
                "",
                self.narrative_ar,
                "",
                "### DeepSpeed (مقتطف)",
                "```json",
                json.dumps(self.deepspeed_config, ensure_ascii=False, indent=2)[:2000],
                "```",
                "",
                "### توسّع عنقودي",
                f"```json\n{json.dumps(self.k8s_hint, ensure_ascii=False, indent=2)}\n```",
            ]
        )


def plan_3d_parallelism(
    model_params_b: float = 7.0,
    n_gpus: int = 8,
    target_global_batch: int = 1024,
) -> ParallelismPlan:
    """
    قواعد تقريبية شائعة:
      - TP ∈ {1,2,4,8} حسب عرض النموذج وNVLink
      - PP لزيادة العمق عند نفاد الذاكرة
      - DP = n_gpus / (TP * PP)
    """
    n_gpus = max(1, int(n_gpus))
    # اختيار TP
    if n_gpus >= 8 and model_params_b >= 30:
        tp = 8
    elif n_gpus >= 4 and model_params_b >= 7:
        tp = 4
    elif n_gpus >= 2 and model_params_b >= 3:
        tp = 2
    else:
        tp = 1
    # PP
    if model_params_b >= 70 and n_gpus >= 16:
        pp = 4
    elif model_params_b >= 13 and n_gpus >= 8:
        pp = 2
    else:
        pp = 1
    # ضمان القسمة
    while tp * pp > n_gpus and tp > 1:
        tp //= 2
    while tp * pp > n_gpus and pp > 1:
        pp //= 2
    dp = max(1, n_gpus // (tp * pp))
    micro = max(1, target_global_batch // max(dp, 1))
    # صغّر micro إن كان النموذج ضخماً
    if model_params_b >= 30:
        micro = min(micro, 2)
    elif model_params_b >= 7:
        micro = min(micro, 4)
    global_batch = micro * dp

    ds = {
        "train_batch_size": global_batch,
        "train_micro_batch_size_per_gpu": micro,
        "gradient_accumulation_steps": max(1, target_global_batch // max(global_batch, 1)),
        "zero_optimization": {
            "stage": 3 if model_params_b >= 13 else 2,
            "offload_param": {"device": "cpu"} if model_params_b >= 30 else None,
            "overlap_comm": True,
        },
        "fp16": {"enabled": True},
        "bf16": {"enabled": model_params_b >= 7},
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
        "nsm_parallelism": {
            "data_parallel_size": dp,
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "framework_hints": ["DeepSpeed", "Megatron-LM style TP/PP"],
        },
    }
    # تنظيف None
    if ds["zero_optimization"]["offload_param"] is None:
        del ds["zero_optimization"]["offload_param"]

    k8s = {
        "strategy": "Ray + Kubernetes Job",
        "min_replicas": max(1, n_gpus // 8),
        "max_replicas": max(n_gpus, 8),
        "scale_up_when": "step_time > budget OR queue_utilization > 0.9",
        "scale_down_when": "queue_empty for 10m",
        "spot_preferred": True,
        "note_ar": "التنفيذ الفعلي يتطلب kubeconfig + GPU operator — هنا قرار وتوصيف فقط.",
    }

    narrative = (
        f"لنموذج ~{model_params_b}B على {n_gpus} GPU: "
        f"قسّم الطبقات TP={tp}، عمق الخط PP={pp}، وكرر البيانات DP={dp}. "
        f"ZeRO stage مناسب للذاكرة. ابدأ على حصة مجانية/Spot ثم وسّع العنقود ديناميكياً."
    )
    plan = ParallelismPlan(
        ok=True,
        model_params_b=model_params_b,
        n_gpus=n_gpus,
        data_parallel=dp,
        tensor_parallel=tp,
        pipeline_parallel=pp,
        micro_batch=micro,
        global_batch=global_batch,
        deepspeed_config=ds,
        narrative_ar=narrative,
        k8s_hint=k8s,
    )
    out = SC_DIR / f"parallelism_{n_gpus}gpu_{model_params_b}b.json"
    out.write_text(json.dumps(asdict(plan), ensure_ascii=False, indent=2), encoding="utf-8")
    (out.with_suffix(".md")).write_text(plan.to_markdown(), encoding="utf-8")
    return plan


def handle_supercompute_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(
        r"(توازي\s*ثلاث|3d\s*parallel|deepspeed|megatron|عنقود|supercompute|توزيع\s*gpu|pipeline\s*parallel)",
        text,
        re.I,
    ):
        return None
    params_b, gpus = 7.0, 8
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", text, re.I)
    if m:
        params_b = float(m.group(1))
    m = re.search(r"(\d+)\s*(?:gpu|كروت|بطاق)", text, re.I)
    if m:
        gpus = max(1, min(1024, int(m.group(1))))
    return plan_3d_parallelism(model_params_b=params_b, n_gpus=gpus).to_markdown()
