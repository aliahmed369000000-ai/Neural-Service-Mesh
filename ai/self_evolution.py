"""
Self-Evolution — ترقية جينية مسؤولة للوكيل نفسه
===============================================
  • سجل إصدارات الوكيل (v1 → v2) مع درجات كفاءة
  • اقتراح ترقية عند تفوّق نسخة تجريبية
  • لا يحذف كود الإنتاج تلقائياً — يتطلب تأكيداً صريحاً

Kernel optimization: يصدر تلميحات C++/CUDA/Triton مرتبطة بالعتاد المكتشف.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("SelfEvolution")

ROOT = Path(__file__).resolve().parent.parent
EVO_DIR = ROOT / "artifacts" / "model_training" / "super_ai" / "evolution"
EVO_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY = EVO_DIR / "agent_versions.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> Dict[str, Any]:
    if REGISTRY.is_file():
        try:
            return json.loads(REGISTRY.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "active": "v1.0.0",
        "versions": [
            {
                "id": "v1.0.0",
                "score": 0.70,
                "notes": "خط الأساس NSM Agent",
                "created_at": _now(),
            }
        ],
        "history": [],
    }


def _save(data: Dict[str, Any]) -> None:
    REGISTRY.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class EvolutionReport:
    ok: bool
    action: str
    active: str
    candidate: Optional[str]
    should_promote: bool
    reason_ar: str
    kernel_hints: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)

    def to_markdown(self) -> str:
        lines = [
            "## 🔁 التطور الذاتي للوكيل",
            f"- النسخة النشطة: **{self.active}**",
            f"- المرشّح: **{self.candidate or '—'}**",
            f"- ترقية؟ **{'نعم' if self.should_promote else 'لا'}**",
            f"- الإجراء: {self.action}",
            f"- السبب: {self.reason_ar}",
            "",
            "### تلميحات نواة منخفضة المستوى",
        ]
        for h in self.kernel_hints:
            lines.append(f"- {h}")
        lines += [
            "",
            "_الترقية الفعلية للكود تتطلّب مراجعة بشرية + CI؛ لن يُحذف الفرع الرئيسي تلقائياً._",
        ]
        return "\n".join(lines)


def propose_agent_version(score: float, notes: str = "") -> EvolutionReport:
    data = _load()
    active = data.get("active", "v1.0.0")
    active_score = 0.7
    for v in data.get("versions", []):
        if v.get("id") == active:
            active_score = float(v.get("score", 0.7))
    # رقم إصدار جديد
    n = len(data.get("versions", [])) + 1
    cand = f"v1.{n}.0"
    should = float(score) >= active_score + 0.03
    entry = {
        "id": cand,
        "score": float(score),
        "notes": notes or "مرشّح تلقائي من دورة تقييم",
        "created_at": _now(),
        "promoted": False,
    }
    data.setdefault("versions", []).append(entry)
    data.setdefault("history", []).append(
        {"event": "propose", "candidate": cand, "score": score, "at": _now()}
    )
    _save(data)

    hints = [
        "PyTorch: channels_last + torch.compile على Ampere/Hopper.",
        "Triton: ادمج elementwise + activation في نواة واحدة إن ظهر bottleneck في profiler.",
        "CUDA graphs: للإستدلال المتكرر بأشكال ثابتة فقط.",
        "تجنّب كتابة نوى CUDA غير مختبرة في مسار الإنتاج.",
    ]
    try:
        from ai.hardware_aware import detect_gpu_family

        fam = detect_gpu_family()
        hints.insert(0, f"العائلة المكتشفة: {fam} — طبّق ملف hardware_aware المناسب.")
    except Exception:
        pass

    reason = (
        f"المرشّح {cand} بدرجة {score:.3f} مقابل النشط {active} ({active_score:.3f}). "
        + (
            "يفوق العتبة (+3%) — جاهز لمسار CI/CD مع تأكيد بشري."
            if should
            else "دون عتبة الترقية — أبقِ النسخة النشطة."
        )
    )
    return EvolutionReport(
        ok=True,
        action="propose",
        active=active,
        candidate=cand,
        should_promote=should,
        reason_ar=reason,
        kernel_hints=hints,
    )


def activate_version(version_id: str, confirm: str = "") -> EvolutionReport:
    data = _load()
    if confirm.strip().lower() not in ("promote", "ترقية", "confirm"):
        return EvolutionReport(
            ok=False,
            action="denied",
            active=data.get("active", "v1.0.0"),
            candidate=version_id,
            should_promote=False,
            reason_ar="ارفض الترقية دون تأكيد صريح: أكّد ترقية الوكيل promote",
            kernel_hints=[],
        )
    ids = {v["id"] for v in data.get("versions", [])}
    if version_id not in ids:
        return EvolutionReport(
            ok=False,
            action="missing",
            active=data.get("active", "v1.0.0"),
            candidate=version_id,
            should_promote=False,
            reason_ar=f"الإصدار {version_id} غير موجود في السجل.",
            kernel_hints=[],
        )
    old = data.get("active")
    data["active"] = version_id
    for v in data["versions"]:
        if v["id"] == version_id:
            v["promoted"] = True
    data.setdefault("history", []).append(
        {"event": "activate", "from": old, "to": version_id, "at": _now()}
    )
    _save(data)
    return EvolutionReport(
        ok=True,
        action="activate",
        active=version_id,
        candidate=version_id,
        should_promote=True,
        reason_ar=f"نُقلت الصلاحية المنطقية من {old} إلى {version_id} في السجل (بدون حذف ملفات).",
        kernel_hints=[],
    )


def handle_evolution_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if re.search(r"(سجل\s*الوكيل|agent\s*versions|اصدارات\s*الوكيل)", text, re.I):
        data = _load()
        return (
            "## 📚 إصدارات الوكيل\n\n```json\n"
            + json.dumps(data, ensure_ascii=False, indent=2)[:3500]
            + "\n```"
        )
    if re.search(r"(فعّل\s*وكيل|activate\s*agent|رق[ّ]?ي\s*الوكيل)", text, re.I):
        m = re.search(r"v\d+\.\d+\.\d+", text, re.I)
        vid = m.group(0) if m else ""
        conf = "promote" if re.search(r"promote|تأكيد|أكد", text, re.I) else ""
        if not vid:
            return "حدد إصداراً مثل: `رقِّ الوكيل v1.2.0 تأكيد promote`"
        return activate_version(vid, confirm=conf).to_markdown()
    if re.search(
        r"(تطور\s*ذاتي|self\s*-?evolv|نسخ[ةه]\s*وكيل|agent\s*v2|ترقية\s*جيني[ةه]\s*للوكيل)",
        text,
        re.I,
    ):
        score = 0.78
        m = re.search(r"score\s*[=:]\s*([0-9.]+)", text, re.I)
        if m:
            score = float(m.group(1))
            if score > 1:
                score /= 100.0
        return propose_agent_version(score).to_markdown()
    return None
