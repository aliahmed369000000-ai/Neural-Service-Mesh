"""
Cognitive Swarm Simulation — محاكاة عقول/مدارس فكرية
=====================================================
Personas تتناظر حول معضلة؛ تُسجَّل الحجج كمعرفة اصطناعية تجريبية.
ليست بديلاً عن الاجتهاد البشري أو الفتوى.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = ROOT / "artifacts" / "model_training" / "civilization" / "cognitive_swarm"
SIM_DIR.mkdir(parents=True, exist_ok=True)

PERSONAS: Dict[str, Dict[str, str]] = {
    "ahl_alathar": {
        "title": "مدرسة أثرية",
        "style": "تستند إلى النصوص والآثار، حذرة من الرأي المجرّد.",
    },
    "rationalist": {
        "title": "اتجاه عقلي كلامي",
        "style": "يرجّح النظر العقلي مع ضبط المقاصد ودرء التعارض.",
    },
    "natural_philosopher": {
        "title": "فيلسوف طبيعي",
        "style": "يربط الظواهر بأسباب طبيعية قابلة للملاحظة والتجريب.",
    },
    "ethicist_tech": {
        "title": "أخلاقي تقني",
        "style": "يركّز على الضرر/النفع، الخصوصية، والمساءلة في الأنظمة الذكية.",
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_debate(topic: str, rounds: int = 2) -> Dict[str, Any]:
    topic = (topic or "أخلاقيات الذكاء الاصطناعي").strip()
    rounds = max(1, min(5, int(rounds)))
    transcript: List[Dict[str, str]] = []
    for r in range(rounds):
        for key, persona in PERSONAS.items():
            # رد هيكلي قالب — يمكن لاحقاً تغذيته بـ free_router/CKG
            arg = (
                f"[{persona['title']}] حول «{topic}»: {persona['style']} "
                f"في الجولة {r+1} أؤكد ضرورة ضبط الاستدلال بمصدر معرفي موثوق "
                f"وتجنب التعميم غير المبرهن."
            )
            transcript.append({"round": str(r + 1), "persona": key, "text": arg})
    synthesis = (
        f"تركيب تجريبي لمعضلة «{topic}»: اتفقت الشخصيات على الحاجة لمرجعية معرفية "
        "ومحاسبة أخلاقية، واختلفت في وزن النص مقابل النظر. "
        "يُحفظ كنص اصطناعي للمراجعة — لا يُنشر كحكم نهائي."
    )
    report = {
        "ok": True,
        "topic": topic,
        "rounds": rounds,
        "personas": list(PERSONAS.keys()),
        "transcript": transcript,
        "synthesis_ar": synthesis,
        "created_at": _now(),
        "label": "pure_synthetic_theology_experimental",
    }
    out = SIM_DIR / f"debate_{int(datetime.now().timestamp())}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["path"] = str(out.relative_to(ROOT))
    try:
        from ai.persistent_memory import remember_experience
        remember_experience("cognitive_debate", synthesis, {"topic": topic})
    except Exception:
        pass
    return report


def handle_swarm_sim_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(r"(محاكاة|محاكاه|cognitive\s*swarm|تناظر\s*مدارس|حوار\s*فلسف|عقول\s*متعدد)", text, re.I):
        return None
    topic = "أخلاقيات الذكاء الاصطناعي"
    m = re.search(r"(?:حول|عن|topic)[:\s]+(.+)$", text, re.I)
    if m:
        topic = m.group(1).strip()[:200]
    r = run_debate(topic)
    lines = [
        f"## 🧠 محاكاة عقول متعددة — {r['topic']}",
        f"- جولات: {r['rounds']} · شخصيات: {', '.join(r['personas'])}",
        f"- ملف: `{r.get('path')}`",
        "",
        "### تركيب",
        r["synthesis_ar"],
        "",
        "### مقتطف",
    ]
    for t in r["transcript"][:4]:
        lines.append(f"- {t['text'][:160]}…")
    return "\n".join(lines)
