"""
Civilizational Guardrails — حارس معرفي ضد التشويه (دفاعي)
========================================================
  • يفحص ادّعاء نصي مقابل مفاهيم CKG المحلية
  • يصنّف: مدعوم / غير مدعوم / ملتبس
  • يقترح رداً تحصينياً قصيراً — دون نشر تلقائي على الشبكات
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
GR_DIR = ROOT / "artifacts" / "model_training" / "civilization" / "guardrails"
GR_DIR.mkdir(parents=True, exist_ok=True)


def _ckg_names() -> List[str]:
    try:
        from ai.grand_knowledge_mesh import _load_ckg_names
        return _load_ckg_names(3000)
    except Exception:
        return []


def analyze_claim(claim: str) -> Dict[str, Any]:
    claim = (claim or "").strip()
    names = _ckg_names()
    hits = [n for n in names if n and n in claim]
    # إشارات خطر بسيطة
    risk_flags = []
    if re.search(r"(مؤامر|مختلق|زيف|لا\s*أصل)", claim):
        risk_flags.append("لغة اتهام/نفي حاد")
    if len(hits) == 0:
        status = "unsupported_or_unknown"
        score = 0.25
    elif len(hits) >= 2:
        status = "partially_grounded"
        score = 0.7
    else:
        status = "weakly_grounded"
        score = 0.5
    rebuttal = (
        "تحصين مقترح: أعد صياغة الادّعاء مع مصادر أولية، "
        f"واربطه بمفاهيم CKG ذات الصلة: {', '.join(hits[:5]) or '— لا تطابق مباشر —'}. "
        "تجنّب التعميم؛ افصل الثابت عن التفسير."
    )
    report = {
        "ok": True,
        "claim": claim[:500],
        "status": status,
        "grounding_score": score,
        "ckg_hits": hits[:20],
        "risk_flags": risk_flags,
        "rebuttal_ar": rebuttal,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": "defensive_review_only_no_auto_publish",
    }
    out = GR_DIR / f"claim_{int(datetime.now().timestamp())}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["path"] = str(out.relative_to(ROOT))
    return report


def handle_guard_command(user_input: str) -> Optional[str]:
    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(r"(حارس\s*معرف|guardrail|تحصين\s*معرف|كشف\s*تزييف|تحقق\s*ادعا)", text, re.I):
        return None
    m = re.search(r"(?:ادعا[ءي]|claim|نص)[:\s]+(.+)$", text, re.I)
    claim = m.group(1).strip() if m else text
    # strip command words
    claim = re.sub(r"^(حارس\s*معرفي|تحقق\s*ادعاء|كشف\s*تزييف)[:\s]*", "", claim, flags=re.I)
    r = analyze_claim(claim)
    return (
        "## 🛡️ حارس الأمان المعرفي\n"
        f"- الحالة: **{r['status']}** · درجة التأسيس: **{r['grounding_score']}**\n"
        f"- مفاهيم CKG: {', '.join(r['ckg_hits'][:8]) or '—'}\n"
        f"- {r['rebuttal_ar']}\n"
        f"- ملف: `{r.get('path')}`\n"
    )
