"""
Continuous Active Re-Training — حصاد فجوات → خطة تدريب خلفية
لا يستبدل الأوزان على main دون تحقق؛ يكتب خطة + يطلق إشارة مبوّبة.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "model_training" / "active_retrain"
OUT.mkdir(parents=True, exist_ok=True)


def harvest_gaps() -> Dict[str, Any]:
    gaps = []
    try:
        from ai.sovereignty_loop import knowledge_pulse
        pulse = knowledge_pulse()
        gaps.extend(pulse.get("gap_hints") or [])
    except Exception as e:
        pulse = {"error": str(e)}
    # أحداث اجتماعية سلبية كإشارة ضعف محتوى
    try:
        from ai.social_agent import get_recent_events
        for row in get_recent_events(30):
            # platform,event_type,author,content,...
            if len(row) >= 4 and str(row[1]) in ("reply_fail", "publish_blocked"):
                gaps.append(str(row[3])[:160])
    except Exception:
        pass
    return {"gaps": gaps[:30], "n": len(gaps), "pulse": pulse if isinstance(pulse, dict) else {}}


def plan_retrain(epochs: int = 20) -> Dict[str, Any]:
    harvest = harvest_gaps()
    epochs = max(1, min(100, int(epochs)))
    plan = {
        "at": datetime.now(timezone.utc).isoformat(),
        "epochs": epochs,
        "gaps_n": harvest["n"],
        "suggested_pkl": "data/ckg_sentences_v4.pkl",
        "script": "run_training_loop.sh",
        "replace_weights": False,  # لا استبدال أعمى
        "steps_ar": [
            "جمع/تنظيف جمل الفجوات إلى pkl جديد عند توفر البيانات",
            f"تشغيل تدريب مبوّب حتى {epochs} عصراً (أو Kaggle)",
            "التحقق عبر nsm_answer_verifier على عيّنة",
            "استبدال الأوزان فقط بعد اجتياز العتبة",
        ],
        "harvest": harvest,
    }
    # بوابة العالم
    try:
        from world_model.environment_model import EnvironmentModel
        safety = EnvironmentModel(model_dir=str(ROOT / "world_model")).assess_training_safety(
            "active_retrain", estimated_vram_mb=6144
        )
        plan["safety"] = safety
        plan["green_light"] = bool(safety.get("green_light"))
    except Exception as e:
        plan["safety"] = {"error": str(e)}
        plan["green_light"] = False
    path = OUT / f"plan_{int(datetime.now().timestamp())}.json"
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    plan["path"] = str(path.relative_to(ROOT))
    return plan


def handle_active_retrain_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(r"(إ?عادة\s*تدريب\s*نشط|active\s*retrain|تنفس\s*معرف|تحديث\s*معرفي\s*تلقائي)", text, re.I):
        return None
    epochs = 20
    m = re.search(r"(\d+)\s*(?:عصر|epoch)", text, re.I)
    if m:
        epochs = int(m.group(1))
    plan = plan_retrain(epochs=epochs)
    return "## 🌬️ تنفس معرفي / إعادة تدريب نشط\n```json\n" + json.dumps(plan, ensure_ascii=False, indent=2, default=str)[:3500] + "\n```"
