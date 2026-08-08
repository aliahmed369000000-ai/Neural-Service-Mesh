#!/usr/bin/env python3
"""
MoE Continual Layer — تطوير مستمر خفيف للتصنيف والتوجيه
======================================================
- يسجّل تصنيفات حية (سؤال → فئة → ثقة)
- يقترح الفئات الأضعف / الأكثر التباساً
- يقدّم خطوة تكيّف صغيرة (train_on_context) عند ثقة منخفضة + تلميح كلامي واضح

لا يعتمد على مفاتيح API — يعمل محلياً بالكامل.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "artifacts" / "hierarchical_moe" / "continual_log.jsonl"
STATS_PATH = ROOT / "artifacts" / "hierarchical_moe" / "continual_stats.json"


def _append_log(row: Dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def record_classification(
    question: str,
    result: Dict[str, Any],
    adapted: bool = False,
) -> None:
    """تسجيل نتيجة تصنيف واحدة."""
    _append_log(
        {
            "ts": time.time(),
            "q": (question or "")[:240],
            "top": result.get("top"),
            "confidence": result.get("confidence"),
            "source": result.get("source"),
            "adapted": adapted,
        }
    )


def load_recent(n: int = 200) -> List[Dict[str, Any]]:
    if not LOG_PATH.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return rows[-n:]


def stats_report(n: int = 300) -> str:
    rows = load_recent(n)
    if not rows:
        return "## 📈 إحصاء MoE المستمر\n\nلا توجد سجلات بعد. اسأل النظام ليبدأ التسجيل."
    tops = Counter(r.get("top") or "unknown" for r in rows)
    confs = [float(r.get("confidence") or 0) for r in rows]
    low = [r for r in rows if float(r.get("confidence") or 0) < 0.45]
    lines = [
        "## 📈 إحصاء MoE المستمر",
        "",
        f"- عينات أخيرة: **{len(rows)}**",
        f"- متوسط الثقة: **{(sum(confs)/len(confs)):.3f}**",
        f"- تصنيفات منخفضة الثقة (<0.45): **{len(low)}**",
        "",
        "### أكثر الفئات ظهوراً",
    ]
    for cat, c in tops.most_common(8):
        lines.append(f"- `{cat}`: {c}")
    if low:
        lines.append("\n### أمثلة ثقة منخفضة")
        for r in low[-5:]:
            lines.append(
                f"- «{(r.get('q') or '')[:60]}» → `{r.get('top')}` ({r.get('confidence')})"
            )
    return "\n".join(lines)


def classify_and_adapt(
    question: str,
    min_confidence: float = 0.4,
    adapt: bool = True,
) -> Dict[str, Any]:
    """
    صنّف السؤال؛ إن كانت الثقة منخفضة ووُجد تلميح كلامي واضح،
    نفّذ خطوة تكيّف صغيرة على راوتر الفئات.
    """
    from ai.moe_ckg_bridge import get_moe_bridge, keyword_category_scores

    bridge = get_moe_bridge()
    result = bridge.classify(question)
    adapted = False
    adapt_info: Dict[str, Any] = {}

    conf = float(result.get("confidence") or 0)
    kw = keyword_category_scores(question or "")
    if (
        adapt
        and bridge.available
        and conf < min_confidence
        and kw
    ):
        preferred = [c for c, _ in sorted(kw.items(), key=lambda x: -x[1])[:2]]
        try:
            from ai.knowledge_trainer import VectorEncoder
            vec = VectorEncoder.encode(question, domain="general")
            adapt_info = bridge.train_on_context(vec, preferred, steps=2, lr=5e-4)
            adapted = True
            # إعادة التصنيف بعد التكيّف
            result = bridge.classify(question, context_vector=vec)
            result["adapted"] = True
            result["adapt_info"] = adapt_info
        except Exception as e:
            result["adapt_error"] = str(e)

    record_classification(question, result, adapted=adapted)
    return result


def continual_dashboard() -> str:
    """لوحة قصيرة للتطوير المستمر."""
    from ai.moe_ckg_bridge import get_moe_bridge

    bridge = get_moe_bridge()
    parts = [
        bridge.health_report(),
        "",
        stats_report(250),
        "",
        "### أوامر مفيدة",
        "- `صنّف: ...` أو `classify: ...`",
        "- `صحة moe` · `إحصاء moe`",
        "- `ملخص moe` · `تقرير موازنة`",
    ]
    return "\n".join(parts)
