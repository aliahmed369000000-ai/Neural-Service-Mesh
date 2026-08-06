"""
تصدير بيانات من معرفة NSM (كيانات، جذور، عيّنات قرآنية) إلى CSV قابل للتدريب.
يُستخدم من المصنع عند أهداف CKG / المعرفة الإسلامية.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "artifacts" / "model_training" / "ckg_exports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_entity_label(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return "أخرى"
    if "إله" in s or s.startswith("الذات"):
        return "إلهيات"
    if "نبي" in s or "رسول" in s:
        return "أنبياء_ورسل"
    if "أمة" in s or "مملكة" in s:
        return "أمم_وممالك"
    if "قصة" in s or "فتية" in s:
        return "قصص_قرآنية"
    if "ملكة" in s or "امرأة" in s or "أم " in s:
        return "شخصيات_نسائية"
    if "ملك" in s or "قائد" in s or "شخصية" in s:
        return "شخصيات"
    if "ظاهرة" in s or "سُنّة" in s or "سنة" in s:
        return "سنن_و مفاهيم"
    return "أخرى"


def export_entity_type_csv(max_rows: int = 500) -> Tuple[str, Dict[str, Any]]:
    """تصنيف نوع الكيان من knowledge/entities.json → text,label."""
    path = ROOT / "knowledge" / "entities.json"
    if not path.is_file():
        raise FileNotFoundError("knowledge/entities.json غير موجود")
    data = json.loads(path.read_text(encoding="utf-8"))
    ents = data.get("entities") or {}
    rows: List[Dict[str, str]] = []
    if isinstance(ents, dict):
        items = ents.items()
    else:
        items = [(e.get("name", f"e{i}"), e) for i, e in enumerate(ents)]

    for name, info in items:
        if not isinstance(info, dict):
            continue
        label = _normalize_entity_label(str(info.get("type") or "غير_مصنف"))
        summary = str(info.get("summary") or "")
        attrs = " ".join(str(a) for a in (info.get("attributes") or [])[:8])
        related = " ".join(str(c) for c in (info.get("related_concepts") or [])[:8])
        text = f"{name}. {summary} {attrs} {related}".strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) < 8:
            continue
        rows.append({"text": text, "label": label})
        if len(rows) >= max_rows:
            break

    if len(rows) < 4:
        raise ValueError(f"عيّنات غير كافية من الكيانات ({len(rows)})")

    # توازن تقريبي: كرر الأصناف النادرة إن كان العدد قليلاً جداً
    labels = [r["label"] for r in rows]
    counts = Counter(labels)
    if len(counts) >= 2 and min(counts.values()) == 1 and len(rows) < 40:
        for lab, c in list(counts.items()):
            if c < 2:
                sample = next(r for r in rows if r["label"] == lab)
                rows.append(dict(sample))

    out = OUT_DIR / "ckg_entity_types.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label"])
        w.writeheader()
        w.writerows(rows)

    meta = {
        "task": "classification",
        "source": "knowledge/entities.json",
        "n_rows": len(rows),
        "labels": dict(Counter(r["label"] for r in rows)),
        "path": str(out.relative_to(ROOT)),
    }
    (OUT_DIR / "ckg_entity_types.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(out.relative_to(ROOT)), meta


def export_root_category_csv(max_rows: int = 800) -> Tuple[str, Dict[str, Any]]:
    """تصنيف جذور عربية حسب category إن وُجد، وإلا bucket بالتردد."""
    path = ROOT / "knowledge" / "arabic_roots_index.json"
    if not path.is_file():
        raise FileNotFoundError("arabic_roots_index.json غير موجود")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Dict[str, str]] = []
    for root, info in data.items():
        if not isinstance(info, dict):
            continue
        cat = str(info.get("category") or "").strip()
        freq = int(info.get("frequency") or 0)
        if not cat:
            if freq >= 100:
                cat = "شائع_جداً"
            elif freq >= 20:
                cat = "شائع"
            else:
                cat = "نادر"
        concept = str(info.get("concept_name") or root)
        tokens = " ".join(str(t) for t in (info.get("tokens") or [])[:6])
        text = f"جذر {root} مفهوم {concept} أمثلة {tokens} تكرار {freq}"
        rows.append({"text": text, "label": cat})
        if len(rows) >= max_rows:
            break

    out = OUT_DIR / "ckg_root_categories.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label"])
        w.writeheader()
        w.writerows(rows)
    meta = {
        "task": "classification",
        "source": "knowledge/arabic_roots_index.json",
        "n_rows": len(rows),
        "labels": dict(Counter(r["label"] for r in rows)),
        "path": str(out.relative_to(ROOT)),
    }
    (OUT_DIR / "ckg_root_categories.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(out.relative_to(ROOT)), meta


def export_for_goal(domain_hint: str, raw_text: str = "") -> Tuple[str, Dict[str, Any]]:
    """اختيار أفضل تصدير حسب الهدف."""
    t = (raw_text or "") + " " + domain_hint
    if re.search(r"جذر|roots|صرف", t, re.I):
        return export_root_category_csv()
    # افتراضي للكيانات / CKG / إسلامي
    try:
        return export_entity_type_csv()
    except Exception:
        return export_root_category_csv()


def export_status() -> str:
    files = sorted(OUT_DIR.glob("*.csv"))
    lines = ["## 📦 صادرات CKG للتدريب", f"- المجلد: `artifacts/model_training/ckg_exports/`"]
    if not files:
        lines.append("لا صادرات بعد. ستُنشأ تلقائياً عند هدف CKG في المصنع.")
    for p in files:
        lines.append(f"- `{p.relative_to(ROOT)}` ({p.stat().st_size} بايت)")
    return "\n".join(lines)
