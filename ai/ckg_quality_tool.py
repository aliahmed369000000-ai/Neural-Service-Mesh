"""واجهة أمر لتقرير جودة CKG."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_report_mod():
    spec = importlib.util.spec_from_file_location(
        "ckg_quality_report", ROOT / "scripts" / "ckg_quality_report.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def handle_ckg_quality_command(user_input: str) -> Optional[str]:
    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(r"(تقرير\s*ckg|فحص\s*ckg|جودة\s*ckg|ckg\s*quality|جودة\s*اجابات)", text, re.I):
        return None
    mod = _load_report_mod()
    n = 8
    m = re.search(r"(\d+)", text)
    if m:
        n = max(3, min(15, int(m.group(1))))
    report = mod.build_report(n_questions=n)
    paths = mod.write_outputs(report)
    summary = {
        "ckg_ready": report["ckg_ready"],
        "lfs_blocked": report["lfs_blocked"],
        "avg_answer_quality": report.get("avg_answer_quality"),
        "diagnosis_ar": report.get("diagnosis_ar"),
        "outputs": paths,
        "sample": [
            {
                "q": a.get("question"),
                "overall": (a.get("scores") or {}).get("overall"),
                "ok": a.get("ok"),
            }
            for a in (report.get("answers") or [])[:5]
        ],
    }
    return (
        "## 📊 تقرير جودة CKG والإجابات\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n```\n\nشغّل أيضاً: `python3 scripts/ckg_quality_report.py`"
    )
