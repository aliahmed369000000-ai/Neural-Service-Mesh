"""
Cognitive Edge Runtime — مسار استدلال طرفي خفيف (ليس نظام تشغيل حقيقي)
======================================================================
يبني «ملف تشغيل طرفي» لـ NeuralCore/numpy على أجهزة ضعيفة:
  • تعطيل الأطر الثقيلة
  • quantize تلميحات
  • واجهة استدعاء offline

رقاقة ASIC مذكورة كخريطة تصميم نظرية فقط — لا تصنيع سيليكون هنا.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
EDGE_DIR = ROOT / "artifacts" / "model_training" / "civilization" / "edge_runtime"
EDGE_DIR.mkdir(parents=True, exist_ok=True)


def edge_runtime_profile() -> Dict[str, Any]:
    return {
        "name": "NSM-Cognitive-Edge-Profile",
        "runtime": "python+numpy-only",
        "requires": ["numpy"],
        "discouraged": ["torch large", "transformers full", "cloud roundtrip"],
        "pipeline": [
            "tokenize text",
            "CKG concept match (local JSON subset)",
            "NeuralCore/DeepRouting forward (numpy)",
            "ranked concepts answer",
        ],
        "memory_budget_mb": 256,
        "offline": True,
        "asic_note_ar": (
            "تصميم ASIC/رقاقة مخصصة يتطلب شريك تصنيع وRTL — "
            "هذا الملف يحدد فقط نواة الخوارزميات المرشّحة للتسريع: "
            "matmul 784×784، softmax-4، تشابه متجهات CKG."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def export_edge_bundle() -> Dict[str, Any]:
    profile = edge_runtime_profile()
    out = EDGE_DIR / "edge_profile.json"
    out.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    stub = EDGE_DIR / "edge_infer_stub.py"
    stub.write_text(
        '''# NSM Edge infer stub — offline numpy path
"""شغّل على الجهاز الطرفي بعد نسخ models/ + knowledge subset."""
def infer(question: str) -> str:
    try:
        from ai.reasoning_pipeline import ReasoningPipeline
        p = ReasoningPipeline(train_on_query=False, use_deep_routing=True)
        r = p.answer(question)
        return getattr(r, "answer_text", str(r))
    except Exception as e:
        return f"edge_fallback: {e}"
if __name__ == "__main__":
    print(infer("ما الأمانة؟"))
''',
        encoding="utf-8",
    )
    return {"ok": True, "profile": str(out.relative_to(ROOT)), "stub": str(stub.relative_to(ROOT)), **profile}


def handle_edge_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(r"(cognitive\s*os|نظام\s*تشغيل\s*معرف|edge\s*runtime|رقاقه\s*nsm|asic|طرف[يى]\s*معرف)", text, re.I):
        return None
    r = export_edge_bundle()
    return (
        "## 📱 مسار Cognitive Edge\n"
        f"- ملف: `{r['profile']}`\n- stub: `{r['stub']}`\n"
        f"- {r['asic_note_ar']}\n\n"
        "```json\n"
        + json.dumps({k: r[k] for k in ("name", "runtime", "memory_budget_mb", "offline", "pipeline")}, ensure_ascii=False, indent=2)
        + "\n```"
    )
