"""
AI Architect & Orchestrator — المهندس المعماري الخبير للذكاء الاصطناعي
=====================================================================
يرفع وكيل التدريب من «تشغيل تجارب» إلى حكم هندسي:

  1) LLM-as-a-Judge     → ai.model_judge
  2) Hyperparameter Tuning → ai.hyperparam_tuner
  3) Model Compression  → ai.model_compression
  4) Federated Learning → ai.federated_learning

ويتنسّق مع المنصات البعيدة (Kaggle/Colab) عبر remote_training_orchestrator.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("AIArchitect")

ROOT = Path(__file__).resolve().parent.parent
ARCH_DIR = ROOT / "artifacts" / "model_training" / "architect"
ARCH_DIR.mkdir(parents=True, exist_ok=True)


def architect_status() -> str:
    lines = [
        "## 🏛️ المهندس المعماري للذكاء الاصطناعي — الحالة",
        "",
        "### القدرات",
        "1. **تحكيم ذاتي للنماذج** (`حكّم نموذج`) — مقاييس + قاضٍ LLM اختياري + خطة إصلاح",
        "2. **بحث فائق للمعلمات** (`بحث فائق`) — random + bayesian-like لتقليل هدر GPU",
        "3. **ضغط النماذج** (`كمّم نموذج` / `قلّم نموذج` / `اضغط نموذج`)",
        "4. **تعلم موحّد خاص** (`تدريب اتحادي`) — FedAvg بدون رفع البيانات الخام",
        "",
        "### التكامل",
        "- منصات بعيدة: `حالة المنصات` · `درّب بعيد kaggle` · `مهمة colab`",
        "- مخرجات: `artifacts/model_training/architect/`",
        "",
    ]
    # وجود الوحدات
    mods = [
        ("model_judge", "ai.model_judge"),
        ("hyperparam_tuner", "ai.hyperparam_tuner"),
        ("model_compression", "ai.model_compression"),
        ("federated_learning", "ai.federated_learning"),
        ("remote_orchestrator", "ai.remote_training_orchestrator"),
    ]
    lines.append("### الوحدات")
    for name, imp in mods:
        try:
            __import__(imp)
            lines.append(f"- `{name}`: ✅")
        except Exception as e:
            lines.append(f"- `{name}`: ❌ ({e})")
    lines += [
        "",
        "### دورة معمارية مقترحة",
        "1. `بحث فائق 12 تجربة`",
        "2. `درّب بعيد kaggle وادفع` بالمعلمات الفائزة",
        "3. `حكّم نموذج`",
        "4. عند الحاجة: `اضغط نموذج` أو `تدريب اتحادي 5 عملاء`",
    ]
    return "\n".join(lines)


def run_architect_cycle(n_tune: int = 8) -> str:
    """دورة مختصرة: tune → judge → compress → fed sample."""
    parts: List[str] = ["## 🏛️ دورة المهندس المعماري (مختصرة)", ""]
    try:
        from ai.hyperparam_tuner import run_tuning

        tr = run_tuning(n_trials=n_tune)
        parts.append(tr.to_markdown())
        parts.append("")
    except Exception as e:
        parts.append(f"### Tuning فشل: {e}")
    try:
        from ai.model_judge import judge_demo

        jr = judge_demo(use_llm=False)
        parts.append(jr.to_markdown())
        parts.append("")
    except Exception as e:
        parts.append(f"### Judge فشل: {e}")
    try:
        from ai.model_compression import compress_pipeline

        pipe = compress_pipeline(0.35)
        parts.append("### ضغط")
        parts.append(
            f"- quant ratio≈{pipe['quantize'].get('compression_ratio', 0):.2f}× | "
            f"prune sparsity≈{pipe['prune'].get('sparsity', 0):.1%}"
        )
        parts.append("")
    except Exception as e:
        parts.append(f"### Compression فشل: {e}")
    try:
        from ai.federated_learning import run_federated

        fr = run_federated(n_clients=4, rounds=4, samples_per_client=80, local_epochs=4)
        parts.append(fr.to_markdown())
    except Exception as e:
        parts.append(f"### Federated فشل: {e}")
    report = "\n".join(parts)
    out = ARCH_DIR / "last_architect_cycle.md"
    out.write_text(report, encoding="utf-8")
    return report


def handle_architect_command(user_input: str) -> Optional[str]:
    """
    موجّه أوامر المهندس المعماري.
    يعيد نصاً أو None لتمرير الرسالة لوحدات أخرى.
    """
    text = (user_input or "").strip()
    if not text:
        return None

    # حالة المعماري / قدرات
    if re.search(
        r"(مهندس\s*معمار|ai\s*architect|architect\s*status|قدرات\s*المعماري|حالة\s*المعماري)",
        text,
        re.I,
    ) or text.lower() in ("architect", "معماري"):
        return architect_status()

    # دورة كاملة
    if re.search(r"(دورة\s*معمار|architect\s*cycle|شغ[ّ]?ل\s*المعماري)", text, re.I):
        n = 8
        m = re.search(r"(\d+)\s*تجرب", text)
        if m:
            n = max(4, min(20, int(m.group(1))))
        return run_architect_cycle(n_tune=n)

    # تفويض للوحدات المتخصصة
    try:
        from ai.model_judge import handle_judge_command

        r = handle_judge_command(text)
        if r is not None:
            return r
    except Exception as e:
        logger.warning("judge: %s", e)

    try:
        from ai.hyperparam_tuner import handle_tune_command

        r = handle_tune_command(text)
        if r is not None:
            return r
    except Exception as e:
        logger.warning("tune: %s", e)

    try:
        from ai.model_compression import handle_compression_command

        r = handle_compression_command(text)
        if r is not None:
            return r
    except Exception as e:
        logger.warning("compression: %s", e)

    try:
        from ai.federated_learning import handle_federated_command

        r = handle_federated_command(text)
        if r is not None:
            return r
    except Exception as e:
        logger.warning("federated: %s", e)

    return None
