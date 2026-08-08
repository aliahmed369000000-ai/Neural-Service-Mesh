"""أوامر تذكير Git LFS من داخل الوكيل."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent


def lfs_status() -> dict:
    files = [
        ROOT / "knowledge" / "cognitive_graph.json",
        ROOT / "knowledge" / "cognitive_graph_general_ar.json",
    ]
    rows = []
    for f in files:
        row = {"path": str(f.relative_to(ROOT)), "exists": f.is_file()}
        if f.is_file():
            row["size"] = f.stat().st_size
            head = f.read_text(encoding="utf-8", errors="ignore")[:80]
            row["lfs_pointer"] = "git-lfs.github.com" in head or head.startswith("version https://git-lfs")
        rows.append(row)
    ready = all(
        r.get("exists") and not r.get("lfs_pointer") and r.get("size", 0) > 10_000 for r in rows
    )
    return {"ready": ready, "files": rows}


def handle_lfs_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(r"(git\s*lfs|اسحب\s*lfs|تفعيل\s*lfs|lfs\s*pull|اعداد\s*lfs)", text, re.I):
        return None
    st = lfs_status()
    return (
        "## 📦 Git LFS\n\n```json\n"
        + json.dumps(st, ensure_ascii=False, indent=2)
        + "\n```\n\nللجميع:\n```bash\nbash scripts/setup_git_lfs.sh\n```\n"
        "ثم: `python3 scripts/ckg_quality_report.py --ckg-only`\nدليل: `docs/GIT_LFS.md`"
    )
