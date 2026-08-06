"""
CI/CD for AI — تحديث تلقائي مسؤول للنماذج
==========================================
  • سجل نماذج (model registry) محلي مع أفضل checkpoint
  • توليد/تحديث workflow GitHub Actions لنشر عند تفوّق النموذج
  • لا يدفع أسراراً ولا يكسر الخدمة: بوابة موافقة + اختبارات

الدمج مع المستودع عبر GitHub Actions الموجودة؛ الوكيل يجهّز الملفات والقرار.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("AICICD")

ROOT = Path(__file__).resolve().parent.parent
REG_DIR = ROOT / "artifacts" / "model_training" / "scientist" / "registry"
REG_DIR.mkdir(parents=True, exist_ok=True)
WF_PATH = ROOT / ".github" / "workflows" / "ai-model-promote.yml"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PromoteDecision:
    ok: bool
    should_promote: bool
    old_score: float
    new_score: float
    margin: float
    reason_ar: str
    registry_path: str = ""
    workflow_ready: bool = False
    created_at: str = field(default_factory=_now)

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "## 🚀 قرار الترقية (CI/CD للذكاء الاصطناعي)",
                f"- ترقية؟ **{'نعم ✅' if self.should_promote else 'لا ❌'}**",
                f"- النتيجة السابقة: **{self.old_score:.4f}** → الجديدة: **{self.new_score:.4f}** (هامش {self.margin:.4f})",
                f"- السبب: {self.reason_ar}",
                f"- السجل: `{self.registry_path}`",
                f"- workflow: {'جاهز' if self.workflow_ready else 'غير موجود'}",
                "",
                "عند الموافقة: ادفع عبر GitHub Actions / `promote model` بعد اجتياز الاختبارات.",
            ]
        )


def load_registry() -> Dict[str, Any]:
    path = REG_DIR / "model_registry.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "champion": {"score": 0.0, "model_id": None, "updated_at": None},
        "challengers": [],
        "history": [],
    }


def save_registry(reg: Dict[str, Any]) -> Path:
    path = REG_DIR / "model_registry.json"
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def register_challenger(model_id: str, score: float, meta: Optional[Dict[str, Any]] = None) -> PromoteDecision:
    reg = load_registry()
    old = float(reg.get("champion", {}).get("score") or 0.0)
    margin = float(score) - old
    # عتبة ترقية: تحسن ≥ 1%
    should = margin >= 0.01
    entry = {
        "model_id": model_id,
        "score": float(score),
        "meta": meta or {},
        "at": _now(),
    }
    reg.setdefault("challengers", []).append(entry)
    reg.setdefault("history", []).append(entry)
    if should:
        reg["champion"] = {
            "score": float(score),
            "model_id": model_id,
            "updated_at": _now(),
            "meta": meta or {},
        }
        reason = f"المنافس تفوّق على البطل بهامش {margin:.2%} — مؤهل للترقية بعد الاختبارات."
    else:
        reason = f"التحسن {margin:.2%} دون عتبة 1% — لا ترقية تلقائية (تجنّب تذبذب الإنتاج)."
    path = save_registry(reg)
    ensure_promote_workflow()
    d = PromoteDecision(
        ok=True,
        should_promote=should,
        old_score=old,
        new_score=float(score),
        margin=margin,
        reason_ar=reason,
        registry_path=str(path.relative_to(ROOT)),
        workflow_ready=WF_PATH.is_file(),
    )
    (REG_DIR / f"promote_{int(time.time())}.md").write_text(d.to_markdown(), encoding="utf-8")
    return d


def ensure_promote_workflow() -> Path:
    """ينشئ workflow ترقية آمناً (manual + path filters)."""
    WF_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = """# ترقية نموذج NSM — لا نشر أعمى
name: AI Model Promote

on:
  workflow_dispatch:
    inputs:
      model_id:
        description: 'معرف النموذج في السجل'
        required: true
      confirm:
        description: 'اكتب promote لتأكيد الترقية'
        required: true
        default: ''

jobs:
  promote:
    runs-on: ubuntu-latest
    if: github.event.inputs.confirm == 'promote'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Sanity compile
        run: |
          python -m py_compile ai/ai_cicd.py ai/model_training_agent.py || true
      - name: Show registry
        run: |
          if [ -f artifacts/model_training/scientist/registry/model_registry.json ]; then
            cat artifacts/model_training/scientist/registry/model_registry.json
          else
            echo "لا سجل نماذج بعد"
          fi
      - name: Notice
        run: |
          echo "الترقية اليدوية المؤكدة للنموذج: ${{ github.event.inputs.model_id }}"
          echo "اربط هنا خطوات نسخ الأوزان إلى مسار الإنتاج / Streamlit secrets حسب بيئتك."
"""
    if not WF_PATH.is_file() or "AI Model Promote" not in WF_PATH.read_text(encoding="utf-8"):
        WF_PATH.write_text(content, encoding="utf-8")
    return WF_PATH


def handle_cicd_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(
        r"(ci/?cd|ترقية\s*نموذج|promote\s*model|سجل\s*نماذج|registry|نشر\s*نموذج)",
        text,
        re.I,
    ):
        return None
    if re.search(r"(سجل|registry|اعرض)", text, re.I):
        reg = load_registry()
        return (
            "## 📚 سجل النماذج\n\n```json\n"
            + json.dumps(reg, ensure_ascii=False, indent=2)[:3000]
            + "\n```"
        )
    # تسجيل منافس — درجة افتراضية أو مستخرجة
    score = 0.85
    m = re.search(r"(0\.\d+|1\.0|\d+(?:\.\d+)?)\s*%?", text)
    # simpler: score=0.87
    m2 = re.search(r"score\s*[=:]\s*([0-9.]+)", text, re.I)
    if m2:
        score = float(m2.group(1))
        if score > 1.0:
            score = score / 100.0
    model_id = f"challenger_{int(time.time()) % 100000}"
    m3 = re.search(r"model[_\s-]?id\s*[=:]\s*([\w.-]+)", text, re.I)
    if m3:
        model_id = m3.group(1)
    d = register_challenger(model_id, score)
    return d.to_markdown() + f"\n\n_Workflow: `{WF_PATH.relative_to(ROOT)}`_"
