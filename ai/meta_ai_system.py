"""
Meta-AI System — الذكاء الخارق لإدارة الأنظمة
=============================================
طبقة فوق العالِم والم معماری:
  1) Reasoning Traces
  2) Neuroevolution & NAS
  3) Hardware-Aware Optimization
  4) Persistent Vector Memory
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("MetaAI")

ROOT = Path(__file__).resolve().parent.parent
META_DIR = ROOT / "artifacts" / "model_training" / "meta_ai"
META_DIR.mkdir(parents=True, exist_ok=True)


def meta_status() -> str:
    lines = [
        "## 🧠 Meta-AI System — إدارة الأنظمة بتفكير عميق",
        "",
        "### القدرات",
        "1. **تفكير عميق** — `فكر عميق لتصنيف نصوص` · `نقد ذاتي فشل: OOM`",
        "2. **تطور جيني / NAS** — `تطور جيني 5 أجيال` · `nas 8 شبكات`",
        "3. **وعي عتادي** — `تحسين عتاد` · `hardware h100`",
        "4. **ذاكرة متجهة مستمرة** — `تذكر: …` · `استرجع: …` · `إحصاء الذاكرة`",
        "",
        "### التكامل مع الطبقات الأدنى",
        "Meta-AI → Scientist → Architect → Orchestrator → Training Agent",
        "",
    ]
    for mod in (
        "ai.reasoning_traces",
        "ai.neuroevolution_nas",
        "ai.hardware_aware",
        "ai.persistent_memory",
    ):
        try:
            __import__(mod)
            lines.append(f"- `{mod}`: ✅")
        except Exception as e:
            lines.append(f"- `{mod}`: ❌ {e}")
    return "\n".join(lines)


def run_meta_cycle() -> str:
    parts = ["## 🧠 دورة Meta-AI مختصرة", ""]
    try:
        from ai.reasoning_traces import plan_architectures

        parts.append(plan_architectures("مشروع تجريبي Meta-AI").to_markdown())
        parts.append("")
    except Exception as e:
        parts.append(f"تفكير: {e}")
    try:
        from ai.neuroevolution_nas import run_nas

        parts.append(run_nas(generations=3, population=6).to_markdown())
        parts.append("")
    except Exception as e:
        parts.append(f"NAS: {e}")
    try:
        from ai.hardware_aware import plan_for_device

        parts.append(plan_for_device().to_markdown())
        parts.append("")
    except Exception as e:
        parts.append(f"عتاد: {e}")
    try:
        from ai.persistent_memory import recall_similar, memory_stats

        parts.append("### ذاكرة")
        parts.append(str(memory_stats()))
        hits = recall_similar("خطة تدريب شبكة", top_k=3)
        for h in hits:
            parts.append(f"- {h['score']:.2f} {h['kind']}: {h['text'][:100]}")
    except Exception as e:
        parts.append(f"ذاكرة: {e}")
    report = "\n".join(parts)
    (META_DIR / "last_meta_cycle.md").write_text(report, encoding="utf-8")
    return report


def handle_meta_command(user_input: str) -> Optional[str]:
    text = (user_input or "").strip()
    if not text:
        return None
    if re.search(r"(meta\s*-?\s*ai|حال[ةه]\s*meta|ذكاء\s*خارق|نظام\s*meta|ميتا)", text, re.I) or text.lower() in (
        "meta",
        "meta-ai",
        "ميتا",
    ):
        return meta_status()
    if re.search(r"(دور[ةه]\s*meta|meta\s*cycle|شغ[ّ]?ل\s*meta)", text, re.I):
        return run_meta_cycle()

    for mod, fn in (
        ("ai.reasoning_traces", "handle_reasoning_command"),
        ("ai.neuroevolution_nas", "handle_nas_command"),
        ("ai.hardware_aware", "handle_hardware_command"),
        ("ai.persistent_memory", "handle_memory_command"),
    ):
        try:
            m = __import__(mod, fromlist=[fn])
            r = getattr(m, fn)(text)
            if r is not None:
                return r
        except Exception as e:
            logger.warning("%s: %s", mod, e)
    return None
