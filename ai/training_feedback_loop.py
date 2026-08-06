"""
Training Feedback Loop & Self-Correction — حلقة تعلّم ذاتي لوكيل التدريب
======================================================================
  1) Self-Correction: قراءة الأخطاء (OOM/فشل)، تقليل الحجم، إعادة المحاولة
  2) Model Registry: حفظ أفضل نسخة ومقارنة بالمسجّلة سابقاً
  3) Data Drift Monitoring: خط أساس إحصائي + إشارة إعادة تدريب

كل الكتابة محصورة تحت artifacts/model_training/ (عبر sandbox إن وُجد).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("TrainingFeedbackLoop")

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "model_training"
REGISTRY = ART / "registry"
DRIFT = ART / "drift"
LOOP_LOG = ART / "logs"
for d in (REGISTRY, DRIFT, LOOP_LOG):
    d.mkdir(parents=True, exist_ok=True)

try:
    from ai.training_sandbox import (
        EarlyStopping,
        assert_read_allowed,
        assert_write_allowed,
        clamp_epochs,
        clamp_samples,
        detect_compute,
        load_guardrails,
        run_with_timeout,
        SandboxTimeout,
    )
    _SB = True
except Exception:
    _SB = False

    def clamp_epochs(e):  # type: ignore
        return max(1, min(int(e), 50))

    def clamp_samples(n):  # type: ignore
        return max(1, min(int(n), 5000))

    def detect_compute():  # type: ignore
        return {"device": "cpu"}

    def assert_write_allowed(p):  # type: ignore
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        return Path(p)

    def assert_read_allowed(p):  # type: ignore
        return Path(p)

    def run_with_timeout(fn, seconds):  # type: ignore
        return fn()

    class SandboxTimeout(Exception):
        pass


# ── تصنيف الأخطاء واقتراح التصحيح ─────────────────────────────────────────

@dataclass
class CorrectionPlan:
    action: str
    reason: str
    params: Dict[str, Any] = field(default_factory=dict)


def classify_error(error_text: str) -> CorrectionPlan:
    t = (error_text or "").lower()
    if any(k in t for k in ("oom", "out of memory", "killed", "cannot allocate", "cuda out of memory")):
        return CorrectionPlan(
            action="reduce_resources",
            reason="نقص ذاكرة (OOM)",
            params={"epochs_factor": 0.5, "samples_factor": 0.5, "prefer": "torch"},
        )
    if any(k in t for k in ("timeout", "تجاوز المهلة", "sandboxtimeout")):
        return CorrectionPlan(
            action="reduce_runtime",
            reason="تجاوز المهلة الزمنية",
            params={"epochs_factor": 0.6, "timeout_factor": 1.0},
        )
    if any(k in t for k in ("filenotfound", "لم يُعثر", "no such file", "permission")):
        return CorrectionPlan(
            action="fix_path",
            reason="مسار بيانات أو صلاحيات",
            params={"fallback_dataset": "data/samples/classification_demo.csv"},
        )
    if any(k in t for k in ("shape", "size mismatch", "dimension", "mat1 and mat2")):
        return CorrectionPlan(
            action="simplify_model",
            reason="عدم توافق أبعاد",
            params={"prefer": "torch", "epochs_factor": 0.8},
        )
    return CorrectionPlan(
        action="retry_conservative",
        reason="خطأ عام — إعادة بمحاولة محافظة",
        params={"epochs_factor": 0.7, "samples_factor": 0.7},
    )


def analyze_metrics(metric_name: str, metric_value: float, history: Optional[List[float]] = None) -> str:
    """تحليل بسيط لمنحنى التعلّم / جودة النتيجة."""
    lines = ["## 📊 تحليل النتائج", f"- المقياس: **{metric_name}** = `{metric_value:.4f}`"]
    low = False
    if metric_name.lower() in ("accuracy", "f1", "r2"):
        if metric_value < 0.55:
            low = True
            lines.append("- ⚠️ دقة منخفضة — يُفضّل زيادة البيانات أو تبسيط/تغيير البنية.")
        elif metric_value < 0.75:
            lines.append("- أداء متوسط — جرّب مزيداً من الحقب أو ميزات أفضل.")
        else:
            lines.append("- ✅ أداء جيد على مجموعة الاختبار الحالية.")
    elif metric_name.lower() in ("mse", "loss", "val_loss"):
        if metric_value > 5.0:
            low = True
            lines.append("- ⚠️ خسارة مرتفعة — راجع التطبيع ومعدل التعلم وحجم البيانات.")
        else:
            lines.append("- الخسارة ضمن نطاق مقبول لهذه المرحلة.")

    if history and len(history) >= 4:
        first = sum(history[: len(history) // 4]) / max(1, len(history) // 4)
        last = sum(history[-len(history) // 4 :]) / max(1, len(history) // 4)
        if metric_name.lower() in ("mse", "loss", "val_loss"):
            if last > first * 0.98:
                lines.append("- المنحنى شبه مستوٍ أو متدهور — احتمال underfitting أو الحاجة لبيانات أكثر.")
            else:
                lines.append(f"- تحسّن المنحنى: {first:.4f} → {last:.4f}")
        else:
            if last < first:
                lines.append("- تراجع المقياس عبر الحقب — راقب overfitting.")
            else:
                lines.append(f"- تحسّن المقياس: {first:.4f} → {last:.4f}")

    if low:
        lines.append(
            "- اقتراح الوكيل: أعد التشغيل بـ `صحّح وأعد التدريب` أو زد عيّنات البيانات الحقيقية."
        )
    return "\n".join(lines)


def _parse_metric_from_result(text: str) -> Tuple[str, float]:
    for name in ("Accuracy", "F1", "R²", "R2", "MSE"):
        m = re.search(rf"{name}\s*=\s*([0-9.]+)", text or "", re.I)
        if m:
            key = name.replace("R²", "R2")
            return key, float(m.group(1))
    m = re.search(r"(accuracy|f1|mse|r2)\s*[=:]\s*([0-9.]+)", text or "", re.I)
    if m:
        return m.group(1), float(m.group(2))
    return "unknown", 0.0


# ── Model Registry ─────────────────────────────────────────────────────────

def _registry_index_path() -> Path:
    return REGISTRY / "index.json"


def load_registry() -> Dict[str, Any]:
    p = _registry_index_path()
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"version": 1, "models": [], "champion_id": None}


def save_registry(reg: Dict[str, Any]) -> None:
    p = assert_write_allowed(_registry_index_path()) if _SB else _registry_index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def register_model(
    model_path: str,
    task: str,
    metric_name: str,
    metric_value: float,
    dataset: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """تسجيل نموذج ومقارنته بالبطل الحالي."""
    src = Path(model_path)
    if not src.is_file():
        src = ROOT / model_path
    if not src.is_file():
        return {"ok": False, "error": f"ملف غير موجود: {model_path}"}

    reg = load_registry()
    mid = f"mdl_{int(time.time())}_{hashlib.sha1(str(src).encode()).hexdigest()[:8]}"
    dest = REGISTRY / f"{mid}{src.suffix or '.pt'}"
    try:
        dest.write_bytes(src.read_bytes())
    except Exception as e:
        return {"ok": False, "error": str(e)}

    entry = {
        "id": mid,
        "path": str(dest.relative_to(ROOT)),
        "source": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
        "task": task,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "dataset": dataset,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "extra": extra or {},
    }
    reg.setdefault("models", []).append(entry)

    # اختيار البطل: أعلى accuracy/f1/r2 أو أقل mse
    def score(e: Dict[str, Any]) -> float:
        n = (e.get("metric_name") or "").lower()
        v = float(e.get("metric_value") or 0)
        if n in ("mse", "loss", "val_loss"):
            return -v
        return v

    champions = sorted(reg["models"], key=score, reverse=True)
    reg["champion_id"] = champions[0]["id"] if champions else None
    is_champion = reg["champion_id"] == mid
    save_registry(reg)
    return {
        "ok": True,
        "entry": entry,
        "is_champion": is_champion,
        "champion_id": reg["champion_id"],
        "total_models": len(reg["models"]),
    }


def registry_report() -> str:
    reg = load_registry()
    lines = ["## 🗂️ Model Registry", f"- عدد النماذج: **{len(reg.get('models') or [])}**"]
    champ = reg.get("champion_id")
    lines.append(f"- البطل الحالي: `{champ or '—'}`")
    models = sorted(
        reg.get("models") or [],
        key=lambda e: e.get("registered_at") or "",
        reverse=True,
    )
    for e in models[:15]:
        star = " ⭐" if e.get("id") == champ else ""
        lines.append(
            f"- `{e.get('id')}`{star} — {e.get('task')} — "
            f"{e.get('metric_name')}={e.get('metric_value')} — `{e.get('path')}`"
        )
    if not models:
        lines.append("لا نماذج بعد. شغّل تدريباً ثم `سجّل أفضل نموذج`.")
    return "\n".join(lines)


def get_champion() -> Optional[Dict[str, Any]]:
    reg = load_registry()
    cid = reg.get("champion_id")
    for e in reg.get("models") or []:
        if e.get("id") == cid:
            return e
    return None


# ── Data Drift ─────────────────────────────────────────────────────────────

def _feature_stats(X: np.ndarray) -> Dict[str, Any]:
    return {
        "mean": X.mean(axis=0).tolist(),
        "std": (X.std(axis=0) + 1e-12).tolist(),
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
    }


def set_drift_baseline(dataset_csv: str, target_col: Optional[str] = None) -> str:
    from ai.model_training_agent import _infer_target_and_matrix, _load_csv_table, _text_to_bow

    path = Path(dataset_csv)
    if not path.is_file():
        path = ROOT / dataset_csv
    header, data = _load_csv_table(path)
    bundle = _infer_target_and_matrix(header, data, target_col=target_col)
    if bundle["feature_mode"] == "text":
        X = _text_to_bow(bundle["texts"] or [])
    else:
        X = np.asarray(bundle["X"], dtype=np.float64)
    stats = _feature_stats(X)
    payload = {
        "dataset": str(dataset_csv),
        "target": bundle.get("target_name"),
        "feature_mode": bundle["feature_mode"],
        "stats": stats,
        "set_at": datetime.now(timezone.utc).isoformat(),
    }
    out = DRIFT / "baseline.json"
    if _SB:
        assert_write_allowed(out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return (
        f"## 📌 خط أساس Drift\n"
        f"- dataset: `{dataset_csv}`\n"
        f"- n={stats['n']} d={stats['d']}\n"
        f"- محفوظ: `{out.relative_to(ROOT)}`"
    )


def check_drift(dataset_csv: str, target_col: Optional[str] = None, threshold: float = 0.35) -> str:
    """مقارنة إحصائيات البيانات الحالية بخط الأساس (متوسط المسافة المعيارية)."""
    base_path = DRIFT / "baseline.json"
    if not base_path.is_file():
        return "⚠️ لا يوجد خط أساس. نفّذ أولاً: `ثبّت خط أساس drift` على بيانات مرجعية."

    baseline = json.loads(base_path.read_text(encoding="utf-8"))
    from ai.model_training_agent import _infer_target_and_matrix, _load_csv_table, _text_to_bow

    path = Path(dataset_csv)
    if not path.is_file():
        path = ROOT / dataset_csv
    header, data = _load_csv_table(path)
    bundle = _infer_target_and_matrix(header, data, target_col=target_col)
    if bundle["feature_mode"] == "text":
        X = _text_to_bow(bundle["texts"] or [])
    else:
        X = np.asarray(bundle["X"], dtype=np.float64)

    bmean = np.array(baseline["stats"]["mean"], dtype=np.float64)
    bstd = np.array(baseline["stats"]["std"], dtype=np.float64)
    d = min(len(bmean), X.shape[1])
    if d == 0:
        return "❌ أبعاد غير متوافقة للفحص."
    cur_mean = X[:, :d].mean(axis=0)
    # متوسط |μ_cur - μ_base| / σ_base
    score = float(np.mean(np.abs(cur_mean - bmean[:d]) / bstd[:d]))
    drifted = score >= threshold
    report = {
        "dataset": dataset_csv,
        "drift_score": score,
        "threshold": threshold,
        "drifted": drifted,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    out = DRIFT / f"check_{int(time.time())}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (DRIFT / "last_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "## 🌊 فحص Data Drift",
        f"- score: **{score:.4f}** (عتبة {threshold})",
        f"- الحالة: {'⚠️ يوجد انجراف — يُنصح بإعادة التدريب' if drifted else '✅ لا انجراف ملحوظ'}",
        f"- التقرير: `{out.relative_to(ROOT)}`",
    ]
    if drifted:
        lines.append("- إجراء مقترح: `أعد التدريب بسبب drift` أو `صحّح وأعد التدريب`.")
    return "\n".join(lines)


def maybe_retrain_on_drift(
    dataset_csv: str,
    threshold: float = 0.35,
    epochs: int = 15,
) -> str:
    drift_msg = check_drift(dataset_csv, threshold=threshold)
    last = json.loads((DRIFT / "last_check.json").read_text(encoding="utf-8"))
    if not last.get("drifted"):
        return drift_msg + "\n\nلم يُشغَّل تدريب (لا انجراف)."
    # إعادة تدريب + تسجيل
    from ai.model_training_agent import train_from_csv

    result = train_from_csv(dataset_csv, epochs=clamp_epochs(epochs), prefer="auto")
    metric_name, metric_value = _parse_metric_from_result(result)
    # ابحث عن آخر ملف pt
    pts = sorted((ART).glob("torch_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    reg_info = {}
    if pts:
        reg_info = register_model(
            str(pts[0]),
            task="classification",
            metric_name=metric_name,
            metric_value=metric_value,
            dataset=dataset_csv,
            extra={"trigger": "data_drift"},
        )
    return (
        drift_msg
        + "\n\n## 🔄 إعادة تدريب بسبب Drift\n"
        + result
        + "\n\n### Registry\n"
        + json.dumps(reg_info, ensure_ascii=False, indent=2)
    )


# ── Self-Correction Loop ───────────────────────────────────────────────────

def self_correct_and_train(
    dataset: str = "data/samples/classification_demo.csv",
    epochs: int = 20,
    prefer: str = "auto",
    max_retries: int = 3,
) -> str:
    """
    يحاول التدريب؛ عند الفشل يقرأ الخطأ ويعدّل المعاملات ويعيد المحاولة.
    عند النجاح يسجّل النموذج ويحلّل المقاييس.
    """
    from ai.model_training_agent import train_from_csv

    epochs = clamp_epochs(epochs)
    attempts: List[Dict[str, Any]] = []
    last_error = ""
    result_text = ""
    current_epochs = epochs
    current_prefer = prefer
    current_dataset = dataset

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            result_text = train_from_csv(
                current_dataset,
                epochs=current_epochs,
                prefer=current_prefer,
            )
            # اعتبر فشل منطقي إذا بدأت الرسالة بـ ❌
            if result_text.strip().startswith("❌"):
                raise RuntimeError(result_text[:500])
            elapsed = time.time() - t0
            metric_name, metric_value = _parse_metric_from_result(result_text)
            analysis = analyze_metrics(metric_name, metric_value)

            pts = sorted(ART.glob("torch_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            reg_info: Dict[str, Any] = {}
            if pts and metric_name != "unknown":
                reg_info = register_model(
                    str(pts[0]),
                    task="classification" if metric_name.lower() in ("accuracy", "f1") else "regression",
                    metric_name=metric_name,
                    metric_value=metric_value,
                    dataset=current_dataset,
                    extra={"attempt": attempt, "self_correction": True},
                )

            attempts.append(
                {
                    "attempt": attempt,
                    "ok": True,
                    "elapsed_s": round(elapsed, 2),
                    "epochs": current_epochs,
                    "prefer": current_prefer,
                    "metric": {metric_name: metric_value},
                }
            )
            log_path = LOOP_LOG / f"feedback_{int(time.time())}.json"
            log_path.write_text(
                json.dumps(
                    {"attempts": attempts, "final": "success", "registry": reg_info},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            champ = "⭐ بطل جديد" if reg_info.get("is_champion") else "مسجّل"
            return (
                f"## 🔁 حلقة التصحيح الذاتي — نجاح (محاولة {attempt}/{max_retries})\n\n"
                f"- dataset: `{current_dataset}`\n"
                f"- epochs: {current_epochs} | prefer: {current_prefer}\n"
                f"- registry: {champ}\n"
                f"- سجل الحلقة: `{log_path.relative_to(ROOT)}`\n\n"
                f"{result_text}\n\n{analysis}\n\n"
                f"### Registry\n{registry_report()}"
            )
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"
            plan = classify_error(last_error)
            attempts.append(
                {
                    "attempt": attempt,
                    "ok": False,
                    "error": str(e)[:400],
                    "plan": asdict(plan),
                }
            )
            # طبّق التصحيح
            if plan.action == "fix_path":
                current_dataset = plan.params.get("fallback_dataset", current_dataset)
            if "epochs_factor" in plan.params:
                current_epochs = clamp_epochs(
                    max(3, int(current_epochs * float(plan.params["epochs_factor"])))
                )
            if plan.params.get("prefer"):
                current_prefer = str(plan.params["prefer"])
            if attempt >= max_retries:
                break

    log_path = LOOP_LOG / f"feedback_fail_{int(time.time())}.json"
    log_path.write_text(
        json.dumps({"attempts": attempts, "final": "failed", "last_error": last_error[:1000]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return (
        f"## ❌ حلقة التصحيح — فشلت بعد {max_retries} محاولات\n\n"
        f"آخر خطأ:\n```\n{last_error[:1200]}\n```\n"
        f"سجل: `{log_path.relative_to(ROOT)}`\n"
        f"المحاولات: {json.dumps(attempts, ensure_ascii=False, indent=2)}"
    )


# ── نشر تجريبي بسيط (استدلال على البطل) ───────────────────────────────────

def predict_with_champion_demo(features: Optional[List[float]] = None) -> str:
    """
    نشر تجريبي: تحميل بطل السجل إن كان MLP torch وحساب استدلال بسيط.
    ليس بديلاً عن API إنتاجي كامل — للتحقق من سلسلة registry → inference.
    """
    champ = get_champion()
    if not champ:
        return "لا يوجد بطل في السجل. درّب وسجّل نموذجاً أولاً."
    path = ROOT / champ["path"]
    if not path.is_file():
        return f"ملف البطل مفقود: {champ['path']}"
    try:
        import torch
        import torch.nn as nn

        blob = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or "state_dict" not in blob:
            return f"صيغة غير مدعومة للاستدلال التجريبي: {champ['path']}"
        n_features = int(blob.get("n_features") or (len(features) if features else 4))
        task = blob.get("task") or champ.get("task") or "classification"
        # بنية افتراضية مطابقة لـ train_torch_on_arrays
        n_out = 2 if task == "classification" else 1
        hidden = 64
        model = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_out),
        )
        model.load_state_dict(blob["state_dict"])
        model.eval()
        if features is None:
            features = [0.0] * n_features
        x = torch.tensor([features[:n_features]], dtype=torch.float32)
        with torch.no_grad():
            out = model(x)
            if task == "classification":
                prob = torch.softmax(out, dim=-1).numpy().tolist()[0]
                pred = int(out.argmax(-1).item())
                return (
                    f"## 🚀 استدلال تجريبي (Champion)\n"
                    f"- model: `{champ['id']}`\n"
                    f"- pred_class: **{pred}**\n"
                    f"- probs: {prob}\n"
                    f"- metric المسجّل: {champ.get('metric_name')}={champ.get('metric_value')}"
                )
            val = float(out.item())
            return (
                f"## 🚀 استدلال تجريبي (Champion)\n"
                f"- model: `{champ['id']}`\n"
                f"- y_hat: **{val:.4f}**\n"
                f"- metric: {champ.get('metric_name')}={champ.get('metric_value')}"
            )
    except Exception as e:
        return f"تعذّر الاستدلال التجريبي: {type(e).__name__}: {e}"


def feedback_status() -> str:
    reg = load_registry()
    drift_base = (DRIFT / "baseline.json").is_file()
    last_drift = DRIFT / "last_check.json"
    last = None
    if last_drift.is_file():
        try:
            last = json.loads(last_drift.read_text(encoding="utf-8"))
        except Exception:
            pass
    lines = [
        "## 🔄 حالة حلقة التغذية الراجعة",
        f"- Registry models: {len(reg.get('models') or [])} | champion: `{reg.get('champion_id')}`",
        f"- Drift baseline: {'✅' if drift_base else '❌'}",
        f"- آخر فحص drift: {last}",
        f"- Compute: {detect_compute()}",
        "",
        "أوامر: `صحّح وأعد التدريب` · `سجّل النماذج` · `ثبّت خط أساس drift` · "
        "`افحص drift` · `استدلال البطل`",
    ]
    return "\n".join(lines)


def handle_feedback_command(user_input: str) -> Optional[str]:
    text = (user_input or "").strip()
    if not text:
        return None

    if re.search(r"(حالة|status).{0,12}(feedback|تغذية|تصحيح|حلقة)", text, re.I):
        return feedback_status()
    if re.search(r"(صحّح|صحح|self.?correct|أعد\s*التدريب|اعادة\s*التدريب).{0,20}(وأعد|self)?", text, re.I) or re.search(
        r"صحّح وأعد التدريب|self.?correction", text, re.I
    ):
        ds = "data/samples/classification_demo.csv"
        m = re.search(r"((?:data|artifacts)[\w./-]+\.csv)", text, re.I)
        if m:
            ds = m.group(1)
        return self_correct_and_train(dataset=ds)
    if re.search(r"(سجل|سجّل|registry|قائمة).{0,15}(نموذج|models|النماذج)|نماذج\s*السجل|model\s*registry", text, re.I):
        return registry_report()
    if re.search(r"(ثبّت|ثبت|set).{0,15}(خط\s*أساس|baseline).{0,10}(drift)?", text, re.I):
        ds = "data/samples/classification_demo.csv"
        m = re.search(r"((?:data|artifacts)[\w./-]+\.csv)", text, re.I)
        if m:
            ds = m.group(1)
        return set_drift_baseline(ds)
    if re.search(r"(افحص|فحص|check).{0,10}(drift|انجراف)", text, re.I):
        ds = "data/samples/classification_demo.csv"
        m = re.search(r"((?:data|artifacts)[\w./-]+\.csv)", text, re.I)
        if m:
            ds = m.group(1)
        return check_drift(ds)
    if re.search(r"(أعد|اعادة|retrain).{0,15}(drift|انجراف)", text, re.I):
        ds = "data/samples/classification_demo.csv"
        m = re.search(r"((?:data|artifacts)[\w./-]+\.csv)", text, re.I)
        if m:
            ds = m.group(1)
        return maybe_retrain_on_drift(ds)
    if re.search(r"(استدلال|predict|inference).{0,12}(البطل|champion)", text, re.I):
        return predict_with_champion_demo()
    if text in ("feedback", "registry", "drift"):
        return feedback_status() if text == "feedback" else (
            registry_report() if text == "registry" else check_drift("data/samples/classification_demo.csv")
        )
    return None
