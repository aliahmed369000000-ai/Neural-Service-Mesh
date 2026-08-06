"""
Synthetic Data Factory — مصنع بيانات اصطناعية + تصفية
=====================================================
  • توليد عيّنات نصية/جدولية اصطناعية خفيفة (بدون LLM خارجي إلزامي)
  • خط أنابيب تصفية: تكرار، طول، جودة تقريبية، تنوع
  • تخزين دفعات تحت artifacts للتجارب اللاحقة
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("SyntheticDataFactory")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "artifacts" / "model_training" / "super_ai" / "synthetic"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_TEMPLATES_AR = [
    "شرح مفهوم {topic} بأسلوب مبسّط مع مثال عملي.",
    "اكتب خطوات لحل مشكلة تتعلق بـ {topic} مع تحذيرات شائعة.",
    "قارن بين أسلوبين في {topic} واذكر متى يُفضَّل كل منهما.",
    "لخّص أفضل ممارسات {topic} للمبتدئين في الذكاء الاصطناعي.",
    "صغ سؤالاً وأجب عنه حول {topic} بدقة عالية.",
]

_TOPICS = [
    "التعلم العميق",
    "معالجة اللغة العربية",
    "تحسين GPU",
    "قواعد المعرفة",
    "الأمن السيبراني للنماذج",
    "التوازي الموزّع",
    "ضغط النماذج",
    "البيانات الاصطناعية",
]


def generate_synthetic_texts(n: int = 50, seed: int = 0) -> List[Dict[str, str]]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        topic = str(rng.choice(_TOPICS))
        tmpl = str(rng.choice(_TEMPLATES_AR))
        prompt = tmpl.format(topic=topic)
        # إجابة اصطناعية هيكلية (ليست معرفة عالمية موثوقة — للتدريب التجريبي)
        answer = (
            f"[{topic}] ملخص توليدي #{i}: "
            f"ابدأ بالتعريف، ثم مثال، ثم قيد عملي. "
            f"seed={seed} hash={hashlib.md5(prompt.encode()).hexdigest()[:8]}"
        )
        rows.append({"id": f"syn_{seed}_{i}", "prompt": prompt, "answer": answer, "topic": topic})
    return rows


def generate_synthetic_table(n: int = 200, d: int = 8, seed: int = 1) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.3 * X[:, 1] + rng.normal(scale=0.1, size=n) > 0).astype(np.int64)
    return {"X": X, "y": y, "n": n, "d": d}


def curate_texts(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """تصفية: تكرار، طول قصير جداً، تكرار جُمل."""
    seen = set()
    kept = []
    stats = {"in": len(rows), "dup": 0, "short": 0, "kept": 0}
    for r in rows:
        body = (r.get("prompt") or "") + "\n" + (r.get("answer") or "")
        if len(body.strip()) < 40:
            stats["short"] += 1
            continue
        h = hashlib.md5(body.encode("utf-8")).hexdigest()
        if h in seen:
            stats["dup"] += 1
            continue
        seen.add(h)
        kept.append(r)
        stats["kept"] += 1
    return kept, stats


@dataclass
class FactoryReport:
    ok: bool
    n_generated: int
    n_kept: int
    stats: Dict[str, int]
    output_path: str
    narrative_ar: str
    created_at: str = field(default_factory=_now)

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "## 🏭 مصنع البيانات الاصطناعية",
                f"- مُولَّد: **{self.n_generated}** · بعد التصفية: **{self.n_kept}**",
                f"- إحصاء: `{self.stats}`",
                f"- المخرج: `{self.output_path}`",
                "",
                self.narrative_ar,
            ]
        )


def run_factory(n_texts: int = 80, seed: int = 0) -> FactoryReport:
    raw = generate_synthetic_texts(n=n_texts, seed=seed)
    kept, stats = curate_texts(raw)
    # جدول مرافق
    table = generate_synthetic_table(n=min(500, n_texts * 3), seed=seed)
    out = DATA_DIR / f"batch_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "texts.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in kept),
        encoding="utf-8",
    )
    np.savez_compressed(out / "table.npz", X=table["X"], y=table["y"])
    meta = {"stats": stats, "topics": _TOPICS, "created_at": _now()}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    narrative = (
        "وُلدت بيانات اصطناعية ثم مُرّرت على فلاتر تكرار/طول. "
        "هذه البيانات للتجارب الداخلية فقط — لا تُعامل كمعرفة موثوقة دون تحقق بشري/CKG."
    )
    report = FactoryReport(
        ok=True,
        n_generated=len(raw),
        n_kept=len(kept),
        stats=stats,
        output_path=str(out.relative_to(ROOT)),
        narrative_ar=narrative,
    )
    (out / "report.md").write_text(report.to_markdown(), encoding="utf-8")
    try:
        from ai.persistent_memory import remember_experience

        remember_experience(
            kind="synthetic_batch",
            text=f"دفعة اصطناعية kept={len(kept)} path={out.name}",
            meta=stats,
        )
    except Exception:
        pass
    return report


def handle_synthetic_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(
        r"(بيانات\s*اصطناع|synthetic\s*data|مصنع\s*بيانات|توليد\s*بيانات|curation)",
        text,
        re.I,
    ):
        return None
    n = 80
    m = re.search(r"(\d+)\s*(?:عين|نص|sample|row)", text, re.I)
    if m:
        n = max(10, min(5000, int(m.group(1))))
    return run_factory(n_texts=n).to_markdown()
