"""
Sensors → Training Pipeline Bridge
==================================
يرصد أسئلة متكررة/فاشلة من الاجتماعي والحساسات، يفحص qa_engine،
ويُصفّ قائمة للدفعة التدريبية القادمة.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "model_training" / "sensor_bridge"
OUT.mkdir(parents=True, exist_ok=True)
QUEUE = OUT / "pending_train_questions.jsonl"


def _collect_candidate_questions() -> List[str]:
    qs: List[str] = []
    try:
        from ai.social_agent import get_recent_events
        for row in get_recent_events(50):
            if len(row) >= 4:
                content = str(row[3] or "").strip()
                if len(content) >= 12 and ("؟" in content or "?" in content):
                    qs.append(content[:240])
    except Exception:
        pass
    try:
        from ai.sovereignty_loop import knowledge_pulse
        pulse = knowledge_pulse()
        for g in pulse.get("gap_hints") or []:
            if len(str(g)) >= 8:
                qs.append(str(g)[:240])
    except Exception:
        pass
    return qs


def _weak_answer(question: str) -> Dict[str, Any]:
    try:
        from knowledge.qa_engine import answer_question
        ans = answer_question(question)
        text = ""
        if isinstance(ans, dict):
            text = str(ans.get("summary") or ans.get("answer") or "")
        else:
            text = str(ans)
        weak = len(text.strip()) < 40
        return {"question": question, "weak": weak, "preview": text[:160]}
    except Exception as e:
        return {"question": question, "weak": True, "error": str(e)}


def bridge_cycle(top_n: int = 10) -> Dict[str, Any]:
    qs = _collect_candidate_questions()
    counts = Counter(qs)
    ranked = [q for q, _ in counts.most_common(top_n)]
    weak_items = []
    for q in ranked:
        r = _weak_answer(q)
        if r.get("weak"):
            weak_items.append(r)
            with QUEUE.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**r, "at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
    report = {
        "at": datetime.now(timezone.utc).isoformat(),
        "candidates": len(qs),
        "unique": len(counts),
        "weak_for_training": weak_items,
        "queue": str(QUEUE.relative_to(ROOT)),
        "next_ar": "بعد ملء الطابور: صياغة جمل → pkl → devops تدريب / Kaggle",
    }
    (OUT / "last_bridge.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def handle_sensor_bridge_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(r"(جسر\s*حساسات|sensors?\s*bridge|حساسات\s*تدريب)", text, re.I):
        return None
    r = bridge_cycle()
    return "## 📡 جسر الحساسات → التدريب\n```json\n" + json.dumps(r, ensure_ascii=False, indent=2)[:3500] + "\n```"
