"""
Kaggle Training Connector — جسر أوزان/بيانات ↔ Kaggle Dual T4
=============================================================
يلفّ ai.kaggle_provider لإعداد مهمة إعادة تدريب وجلب النتائج.
لا يفرض رفع Git تلقائي بدون تحقق؛ يوفّر مسارات واضحة للتحديث.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("KaggleTrainingConnector")

ROOT = Path(__file__).resolve().parent.parent
CONN_DIR = ROOT / "artifacts" / "model_training" / "connectors" / "kaggle"
CONN_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def status() -> Dict[str, Any]:
    try:
        from ai.kaggle_provider import kaggle_status_text
        return {"ok": True, "text": kaggle_status_text() if callable(kaggle_status_text) else str(kaggle_status_text)}
    except Exception:
        pass
    try:
        from ai.kaggle_provider import handle_kaggle_command
        t = handle_kaggle_command("حالة kaggle")
        return {"ok": bool(t), "text": t or "لا حالة"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def queue_retrain_job(epochs: int = 20, title: Optional[str] = None) -> Dict[str, Any]:
    """
    يجهّز مهمة Kaggle لإعادة تدريب (prepare). الدفع الاختياري عبر أوامر الوكيل.
    """
    epochs = max(1, min(200, int(epochs)))
    title = title or f"nsm-continuous-retrain-{epochs}ep"
    result: Dict[str, Any] = {"ok": False, "epochs": epochs, "title": title, "at": _now()}
    try:
        from ai.kaggle_provider import handle_kaggle_command
        prep = handle_kaggle_command(f"جهّز kaggle")
        result["prepare_preview"] = (prep or "")[:1500]
        result["ok"] = prep is not None
        result["next_ar"] = (
            "راجع المهمة ثم نفّذ: `ادفع kaggle` / `درّب بعيد kaggle وادفع`. "
            "بعد انتهاء التدريب: `حمّل kaggle` وانسخ الأوزان إلى models/ بعد التحقق."
        )
    except Exception as e:
        result["error"] = str(e)
    path = CONN_DIR / f"job_{int(datetime.now().timestamp())}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["job_file"] = str(path.relative_to(ROOT))
    return result


def suggest_models_update_git() -> str:
    return (
        "بعد تحميل أوزان جديدة من Kaggle:\n"
        "1) ضعها تحت `models/` أو المسار المعتمد للـ ArabicTransformer\n"
        "2) اختبر محلياً (py_compile + سؤال تجريبي)\n"
        "3) `git add models/ && git commit && git push` — يفضّل CI قبل الإنتاج\n"
        "لا ترفع أوزاناً غير مختبرة مباشرة على main إن كان لديك حماية فرع."
    )
