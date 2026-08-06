"""
Autonomous AI Factory — منظومة إنتاجية مستقلة لوكيل التدريب
==========================================================
  1) Goal-Driven Autonomy: هدف عام → خطة مهام ذاتية
  2) Multi-Agent Collaboration: جامع بيانات / مهندس / جودة / منسّق
  3) Human-in-the-Loop: طابور موافقات قبل النشر أو التكاليف العالية

لا يعتمد على AutoGen/LangChain — يتكامل مع وحدات NSM الموجودة
(training_web_access, training_feedback_loop, training_sandbox, model_training_agent).
"""
from __future__ import annotations

import json
import logging
import re
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("TrainingFactory")

ROOT = Path(__file__).resolve().parent.parent
FACTORY = ROOT / "artifacts" / "model_training" / "factory"
APPROVALS = FACTORY / "approvals"
RUNS = FACTORY / "runs"
for d in (FACTORY, APPROVALS, RUNS):
    d.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_factory_cfg() -> Dict[str, Any]:
    defaults = {
        "require_approval_for_deploy": True,
        "require_approval_above_accuracy_gain": 0.03,
        "max_autonomous_steps": 8,
        "default_target_accuracy": 0.90,
        "auto_approve_toy": True,
        "crisis_after_failures": 3,
    }
    path = ROOT / "config" / "training_guardrails.json"
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            fac = data.get("factory") or {}
            defaults.update({k: v for k, v in fac.items() if v is not None})
    except Exception as e:
        logger.warning("factory cfg: %s", e)
    return defaults


# ── Goal → Plan ────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    agent: str  # data | engineer | qa | coordinator
    action: str
    description: str
    status: str = "pending"  # pending|running|done|failed|skipped|awaiting_approval
    result_preview: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactoryGoal:
    goal_id: str
    raw_text: str
    target_metric: str = "accuracy"
    target_value: float = 0.90
    domain_hint: str = "general"
    created_at: str = field(default_factory=_now)


def parse_goal(text: str) -> FactoryGoal:
    """يستخرج هدفاً قابلاً للقياس من نص عربي/إنجليزي عام."""
    t = (text or "").strip()
    # أزل بادئات الأمر
    t = re.sub(
        r"^(هدف|goal|مصنع|factory|شغّل\s*المصنع|نفّذ\s*هدفا?)\s*[:：\-]?\s*",
        "",
        t,
        flags=re.I,
    ).strip()
    target = 0.90
    m = re.search(r"(\d{1,3})\s*%", t)
    if m:
        target = min(0.99, max(0.5, int(m.group(1)) / 100.0))
    m2 = re.search(r"(?:دقة|accuracy)\s*(?:أعلى\s*من|>=|>|فوق)?\s*(0?\.\d+)", t, re.I)
    if m2:
        target = float(m2.group(1))

    metric = "accuracy"
    if re.search(r"انحدار|mse|rmse", t, re.I):
        metric = "mse"

    domain = "general"
    if re.search(r"لهج|dialect", t, re.I):
        domain = "arabic_dialects"
    elif re.search(r"مشاعر|sentiment", t, re.I):
        domain = "sentiment"
    elif re.search(r"جذر|صرف|roots", t, re.I):
        domain = "ckg_roots"
    elif re.search(r"ckg|كيان|معرفة|قرآن|islami|دين|مفهوم", t, re.I):
        domain = "ckg"
    elif re.search(r"تصنيف|classif", t, re.I):
        domain = "classification"

    return FactoryGoal(
        goal_id=f"goal_{uuid.uuid4().hex[:10]}",
        raw_text=t or text,
        target_metric=metric,
        target_value=target,
        domain_hint=domain,
    )


def build_plan(goal: FactoryGoal) -> List[PlanStep]:
    """خطة ذاتية متعددة الوكلاء حسب الهدف."""
    steps: List[PlanStep] = [
        PlanStep(
            agent="coordinator",
            action="define_success",
            description=f"تعريف النجاح: {goal.target_metric} ≥ {goal.target_value} | مجال={goal.domain_hint}",
        ),
        PlanStep(
            agent="data",
            action="discover_data",
            description="اكتشاف/جلب مصادر بيانات مناسبة (محلية + ويب محكوم)",
        ),
        PlanStep(
            agent="data",
            action="prepare_dataset",
            description="تجهيز عيّنة تدريب محلية آمنة (Toy أو CSV مشروع)",
        ),
        PlanStep(
            agent="engineer",
            action="select_model",
            description="اختيار بنية/مسار تدريب مناسب للهدف",
        ),
        PlanStep(
            agent="engineer",
            action="train",
            description="تنفيذ التدريب مع sandbox + early stopping + تصحيح ذاتي",
        ),
        PlanStep(
            agent="qa",
            action="evaluate",
            description="تقييم المقاييس ومقارنة بالهدف",
        ),
        PlanStep(
            agent="qa",
            action="bias_check",
            description="فحص سريع للتحيز/الثغرات على العيّنة",
        ),
        PlanStep(
            agent="coordinator",
            action="propose_deploy",
            description="اقتراح تسجيل/نشر النموذج وطلب موافقة بشرية إن لزم",
        ),
    ]
    return steps


# ── Multi-agent workers ────────────────────────────────────────────────────

def _agent_data_discover(goal: FactoryGoal) -> Tuple[str, Dict[str, Any]]:
    notes = []
    meta: Dict[str, Any] = {"sources": []}
    # محلي
    samples = list((ROOT / "data" / "samples").glob("*.csv")) if (ROOT / "data" / "samples").is_dir() else []
    for p in samples:
        meta["sources"].append({"type": "local", "path": str(p.relative_to(ROOT))})
    notes.append(f"ملفات محلية: {len(samples)}")

    # ويب محكوم
    try:
        from ai.training_web_access import search_arxiv, search_huggingface, _offline
        if not _offline():
            q = goal.raw_text[:80]
            if goal.domain_hint == "arabic_dialects":
                q = "Arabic dialect identification neural network"
            elif goal.domain_hint == "sentiment":
                q = "Arabic sentiment classification"
            elif goal.domain_hint in ("ckg", "ckg_roots", "islamic_knowledge"):
                q = "Arabic knowledge graph concept classification Quran"
            arx = search_arxiv(q, max_results=3)
            notes.append("arXiv: تم الجلب")
            meta["arxiv_preview"] = arx[:500]
            hf_q = "arabic" if "arab" in goal.domain_hint or "لهج" in goal.raw_text else "classification"
            hf = search_huggingface(hf_q, kind="models", max_results=3)
            notes.append("HuggingFace: ميثا نماذج")
            meta["hf_preview"] = hf[:400]
        else:
            notes.append("وضع offline — تخطي الويب")
    except Exception as e:
        notes.append(f"ويب: {e}")
    return " | ".join(notes), meta


def _agent_data_prepare(goal: FactoryGoal) -> Tuple[str, Dict[str, Any]]:
    """اختيار/تصدير أفضل CSV حسب المجال — CKG يُصدَّر من معرفة المشروع."""
    mapping = {
        "sentiment": "data/samples/text_sentiment_demo.csv",
        "arabic_dialects": "data/samples/text_sentiment_demo.csv",
        "classification": "data/samples/classification_demo.csv",
        "general": "data/samples/classification_demo.csv",
    }
    meta: Dict[str, Any] = {}
    path = mapping.get(goal.domain_hint)

    if goal.domain_hint in ("ckg", "ckg_roots", "islamic_knowledge") or path is None and re.search(
        r"ckg|كيان|معرفة|قرآن", goal.raw_text, re.I
    ):
        try:
            from ai.ckg_training_export import export_for_goal

            path, meta = export_for_goal(goal.domain_hint, goal.raw_text)
            return (
                f"تصدير CKG → `{path}` (n={meta.get('n_rows')}, labels={meta.get('labels')})",
                {"dataset": path, "ckg_export": meta},
            )
        except Exception as e:
            return f"❌ فشل تصدير CKG: {e}", {"dataset": None}

    if goal.target_metric == "mse":
        path = "data/samples/regression_demo.csv"
    path = path or mapping["general"]
    full = ROOT / path
    if not full.is_file():
        found = list((ROOT / "data" / "samples").glob("*.csv"))
        if found:
            path = str(found[0].relative_to(ROOT))
        else:
            return "❌ لا بيانات محلية", {"dataset": None}
    return f"اختيار البيانات: `{path}`", {"dataset": path}


def _agent_engineer_select(goal: FactoryGoal, dataset: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    prefer = "torch"
    if dataset and "text" in dataset:
        prefer = "text"
    elif goal.domain_hint in ("sentiment", "arabic_dialects", "ckg", "ckg_roots", "islamic_knowledge"):
        prefer = "text"
    elif goal.target_metric == "mse":
        prefer = "torch"
    return f"النموذج المختار: prefer={prefer}", {"prefer": prefer, "epochs": 20}


def _agent_engineer_train(
    goal: FactoryGoal, dataset: str, prefer: str, epochs: int
) -> Tuple[str, Dict[str, Any]]:
    try:
        from ai.training_feedback_loop import self_correct_and_train, _parse_metric_from_result

        result = self_correct_and_train(
            dataset=dataset,
            epochs=epochs,
            prefer=prefer,
            max_retries=3,
        )
        name, val = _parse_metric_from_result(result)
        return result[:2500], {"metric_name": name, "metric_value": val, "raw": result[:1500]}
    except Exception as e:
        # fallback مباشر
        try:
            from ai.model_training_agent import train_from_csv

            result = train_from_csv(dataset, epochs=epochs, prefer=prefer)
            from ai.training_feedback_loop import _parse_metric_from_result

            name, val = _parse_metric_from_result(result)
            return result[:2500], {"metric_name": name, "metric_value": val}
        except Exception as e2:
            return f"❌ فشل التدريب: {e} / {e2}", {"metric_name": "unknown", "metric_value": 0.0}


def _agent_qa_evaluate(goal: FactoryGoal, metric_name: str, metric_value: float) -> Tuple[str, Dict[str, Any]]:
    ok = False
    if goal.target_metric in ("accuracy",) and metric_name.lower() in ("accuracy", "f1"):
        ok = metric_value >= goal.target_value
    elif goal.target_metric == "mse" and metric_name.lower() == "mse":
        ok = metric_value <= goal.target_value  # إن كان الهدف حداً أعلى للخسارة
    else:
        # افتراضي accuracy-like
        ok = metric_value >= goal.target_value if metric_name.lower() != "mse" else metric_value < 1.0

    gap = goal.target_value - metric_value if metric_name.lower() != "mse" else metric_value - goal.target_value
    msg = (
        f"QA: {metric_name}={metric_value:.4f} | الهدف={goal.target_value} | "
        f"{'✅ تحقق' if ok else '⚠️ لم يتحقق'} (فجوة≈{gap:.4f})"
    )
    return msg, {"goal_met": ok, "gap": gap, "metric_name": metric_name, "metric_value": metric_value}


def _agent_qa_bias(dataset: str) -> Tuple[str, Dict[str, Any]]:
    """فحص سطحي: توازن الأصناف إن وُجد عمود label."""
    try:
        from ai.model_training_agent import _load_csv_table, _infer_target_and_matrix

        header, data = _load_csv_table(ROOT / dataset)
        bundle = _infer_target_and_matrix(header, data)
        y = bundle["y"]
        import numpy as np

        vals, counts = np.unique(y, return_counts=True)
        dist = {str(v): int(c) for v, c in zip(vals, counts)}
        total = sum(dist.values()) or 1
        ratios = {k: round(v / total, 3) for k, v in dist.items()}
        imbalanced = any(r < 0.15 or r > 0.85 for r in ratios.values()) if len(ratios) > 1 else False
        msg = f"توزيع الأصناف: {ratios}" + (" — ⚠️ عدم توازن" if imbalanced else " — متوازن نسبياً")
        return msg, {"class_dist": dist, "imbalanced": imbalanced}
    except Exception as e:
        return f"تخطي فحص التحيز: {e}", {"imbalanced": None}


# ── Approvals (Human-in-the-Loop) ──────────────────────────────────────────

def create_approval_request(
    run_id: str,
    title: str,
    details: Dict[str, Any],
    cost_estimate_usd: float = 0.0,
) -> Dict[str, Any]:
    req = {
        "id": f"apr_{uuid.uuid4().hex[:10]}",
        "run_id": run_id,
        "title": title,
        "details": details,
        "cost_estimate_usd": cost_estimate_usd,
        "status": "pending",  # pending|approved|rejected
        "created_at": _now(),
        "resolved_at": None,
        "resolver_note": "",
    }
    path = APPROVALS / f"{req['id']}.json"
    path.write_text(json.dumps(req, ensure_ascii=False, indent=2), encoding="utf-8")
    # index
    idx_path = APPROVALS / "index.json"
    idx = []
    if idx_path.is_file():
        try:
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            idx = []
    idx.insert(0, {"id": req["id"], "title": title, "status": "pending", "created_at": req["created_at"]})
    idx_path.write_text(json.dumps(idx[:100], ensure_ascii=False, indent=2), encoding="utf-8")
    return req


def list_approvals(status: Optional[str] = "pending") -> str:
    items = sorted(APPROVALS.glob("apr_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    lines = ["## 🛂 طلبات الموافقة (Human-in-the-Loop)", ""]
    shown = 0
    for p in items:
        try:
            req = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if status and req.get("status") != status:
            continue
        shown += 1
        lines.append(
            f"- `{req['id']}` — **{req.get('status')}** — {req.get('title')} "
            f"— تكلفة≈${req.get('cost_estimate_usd', 0)}"
        )
        if req.get("details"):
            lines.append(f"  - {json.dumps(req['details'], ensure_ascii=False)[:200]}")
    if shown == 0:
        lines.append("لا طلبات" + (f" بحالة `{status}`" if status else "") + ".")
    lines.append("")
    lines.append("للموافقة: `وافق apr_xxxx` | للرفض: `ارفض apr_xxxx`")
    return "\n".join(lines)


def resolve_approval(approval_id: str, approve: bool, note: str = "") -> str:
    path = APPROVALS / f"{approval_id}.json"
    if not path.is_file():
        # try prefix match
        matches = list(APPROVALS.glob(f"{approval_id}*.json"))
        if not matches:
            return f"❌ طلب غير موجود: {approval_id}"
        path = matches[0]
    req = json.loads(path.read_text(encoding="utf-8"))
    req["status"] = "approved" if approve else "rejected"
    req["resolved_at"] = _now()
    req["resolver_note"] = note
    path.write_text(json.dumps(req, ensure_ascii=False, indent=2), encoding="utf-8")
    # إن وُفّق — سجّل البطل إن أمكن
    if approve:
        try:
            from ai.training_feedback_loop import register_model, registry_report
            details = req.get("details") or {}
            model_path = details.get("model_path")
            if model_path:
                register_model(
                    model_path,
                    task=details.get("task") or "classification",
                    metric_name=details.get("metric_name") or "Accuracy",
                    metric_value=float(details.get("metric_value") or 0),
                    dataset=details.get("dataset") or "",
                    extra={"approved": True, "approval_id": req["id"]},
                )
                return (
                    f"✅ تمت الموافقة على `{req['id']}` ونُشر/سُجّل النموذج.\n\n"
                    + registry_report()
                )
        except Exception as e:
            return f"✅ موافقة مسجّلة لكن فشل التسجيل: {e}"
    return f"{'✅ موافقة' if approve else '🛑 رفض'} لـ `{req['id']}` — {note}"


# ── Orchestrator ───────────────────────────────────────────────────────────

def run_factory(goal_text: str, auto_approve_toy: Optional[bool] = None) -> str:
    """تشغيل دورة مصنع كاملة من هدف عام."""
    cfg = _load_factory_cfg()
    if auto_approve_toy is None:
        auto_approve_toy = bool(cfg.get("auto_approve_toy", True))

    goal = parse_goal(goal_text)
    plan = build_plan(goal)
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    state: Dict[str, Any] = {
        "run_id": run_id,
        "goal": asdict(goal),
        "started_at": _now(),
        "steps": [],
        "status": "running",
        "crisis": False,
        "metric_name": "unknown",
        "metric_value": 0.0,
        "dataset": None,
        "prefer": "torch",
        "epochs": 20,
        "failures": 0,
    }

    lines = [
        f"## 🏭 مصنع الذكاء الاصطناعي — تشغيل `{run_id}`",
        f"**الهدف:** {goal.raw_text}",
        f"**نجاح محدد:** {goal.target_metric} ≥ {goal.target_value} | مجال={goal.domain_hint}",
        "",
        "### الخطة الذاتية",
    ]
    for i, s in enumerate(plan, 1):
        lines.append(f"{i}. [{s.agent}] {s.action}: {s.description}")
    lines.append("")

    max_steps = int(cfg.get("max_autonomous_steps") or 8)
    for i, step in enumerate(plan[:max_steps]):
        step.status = "running"
        lines.append(f"#### ▶ خطوة {i+1}: `{step.agent}` / `{step.action}`")
        try:
            if step.action == "define_success":
                preview, meta = step.description, {"target": goal.target_value}
            elif step.action == "discover_data":
                preview, meta = _agent_data_discover(goal)
            elif step.action == "prepare_dataset":
                preview, meta = _agent_data_prepare(goal)
                state["dataset"] = meta.get("dataset")
            elif step.action == "select_model":
                preview, meta = _agent_engineer_select(goal, state.get("dataset"))
                state["prefer"] = meta.get("prefer", "torch")
                state["epochs"] = int(meta.get("epochs") or 20)
            elif step.action == "train":
                if not state.get("dataset"):
                    raise RuntimeError("لا dataset")
                preview, meta = _agent_engineer_train(
                    goal,
                    state["dataset"],
                    state.get("prefer") or "torch",
                    int(state.get("epochs") or 20),
                )
                state["metric_name"] = meta.get("metric_name") or "unknown"
                state["metric_value"] = float(meta.get("metric_value") or 0)
                if str(preview).startswith("❌"):
                    state["failures"] += 1
                    step.status = "failed"
                else:
                    step.status = "done"
            elif step.action == "evaluate":
                preview, meta = _agent_qa_evaluate(
                    goal, state.get("metric_name") or "unknown", float(state.get("metric_value") or 0)
                )
                state["goal_met"] = meta.get("goal_met")
            elif step.action == "bias_check":
                preview, meta = _agent_qa_bias(state.get("dataset") or "data/samples/classification_demo.csv")
            elif step.action == "propose_deploy":
                # موافقة
                need = bool(cfg.get("require_approval_for_deploy", True))
                gain = float(state.get("metric_value") or 0)
                # ابحث آخر pt
                art = ROOT / "artifacts" / "model_training"
                pts = sorted(art.glob("torch_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                model_path = str(pts[0].relative_to(ROOT)) if pts else ""
                details = {
                    "metric_name": state.get("metric_name"),
                    "metric_value": state.get("metric_value"),
                    "dataset": state.get("dataset"),
                    "goal_met": state.get("goal_met"),
                    "model_path": model_path,
                    "task": "classification",
                }
                cost = 0.0 if auto_approve_toy else 5.0
                if need and not (auto_approve_toy and "samples/" in str(state.get("dataset") or "")):
                    req = create_approval_request(
                        run_id,
                        title=f"نشر نموذج للدقة {state.get('metric_value')}",
                        details=details,
                        cost_estimate_usd=cost,
                    )
                    step.status = "awaiting_approval"
                    preview = (
                        f"🛂 يطلب موافقتك: `{req['id']}`\n"
                        f"نموذج جديد بمقياس {details['metric_name']}={details['metric_value']}. "
                        f"تكلفة تقديرية ${cost}.\n"
                        f"قل: `وافق {req['id']}` أو `ارفض {req['id']}`"
                    )
                    meta = {"approval_id": req["id"]}
                else:
                    # موافقة تلقائية للـtoy
                    try:
                        from ai.training_feedback_loop import register_model

                        if model_path:
                            reg = register_model(
                                model_path,
                                task="classification",
                                metric_name=str(details["metric_name"] or "Accuracy"),
                                metric_value=float(details["metric_value"] or 0),
                                dataset=str(details.get("dataset") or ""),
                                extra={"factory_run": run_id, "auto_approved_toy": True},
                            )
                            preview = f"تسجيل تلقائي (toy): {json.dumps(reg, ensure_ascii=False)[:300]}"
                        else:
                            preview = "لا ملف نموذج للتسجيل"
                    except Exception as e:
                        preview = f"تسجيل: {e}"
                    meta = {"auto_approved": True}
                    step.status = "done"
            else:
                preview, meta = f"تخطي {step.action}", {}

            if step.status == "running":
                step.status = "done"
            step.result_preview = str(preview)[:1500]
            step.meta = meta
            lines.append(str(preview)[:1200])
        except Exception as e:
            state["failures"] += 1
            step.status = "failed"
            step.result_preview = f"{type(e).__name__}: {e}"
            lines.append(f"❌ {step.result_preview}")
            if state["failures"] >= int(cfg.get("crisis_after_failures") or 3):
                state["crisis"] = True
                lines.append(
                    "\n🚨 **أزمة (Crisis):** فشل متكرر — يُطلب تدخل بشري لإعادة التوجيه.\n"
                    "راجع السجلات أو عدّل الهدف/البيانات ثم أعد تشغيل المصنع."
                )
                break

        state["steps"].append(
            {
                "agent": step.agent,
                "action": step.action,
                "status": step.status,
                "preview": step.result_preview[:500],
                "meta": step.meta,
            }
        )
        lines.append("")

    state["finished_at"] = _now()
    state["status"] = "crisis" if state.get("crisis") else "completed"
    run_path = RUNS / f"{run_id}.json"
    run_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    lines.append(f"---\nسجل التشغيل: `{run_path.relative_to(ROOT)}`")
    lines.append(f"الحالة النهائية: **{state['status']}** | metric={state.get('metric_name')}={state.get('metric_value')}")
    return "\n".join(lines)


def factory_status() -> str:
    cfg = _load_factory_cfg()
    runs = sorted(RUNS.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    pending = list(APPROVALS.glob("apr_*.json"))
    pend_n = 0
    for p in pending:
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("status") == "pending":
                pend_n += 1
        except Exception:
            pass
    lines = [
        "## 🏭 حالة مصنع الذكاء الاصطناعي",
        f"- إعدادات: موافقة نشر={cfg.get('require_approval_for_deploy')} | "
        f"auto_approve_toy={cfg.get('auto_approve_toy')} | "
        f"أزمة بعد {cfg.get('crisis_after_failures')} فشلاً",
        f"- تشغيلات محفوظة: {len(runs)}",
        f"- موافقات معلّقة: **{pend_n}**",
        "",
        "### آخر التشغيلات",
    ]
    for p in runs[:8]:
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
            lines.append(
                f"- `{st.get('run_id')}` — {st.get('status')} — "
                f"{(st.get('goal') or {}).get('raw_text', '')[:60]} — "
                f"{st.get('metric_name')}={st.get('metric_value')}"
            )
        except Exception:
            lines.append(f"- `{p.name}`")
    lines.append("")
    lines.append(
        "أوامر: `هدف: …` أو `شغّل المصنع: …` · `موافقات` · `وافق apr_…` · `ارفض apr_…` · `حالة المصنع`"
    )
    return "\n".join(lines)


def handle_factory_command(user_input: str) -> Optional[str]:
    text = (user_input or "").strip()
    if not text:
        return None

    if re.search(r"(حالة|status).{0,10}(المصنع|factory)", text, re.I) or text in (
        "حالة المصنع",
        "factory status",
    ):
        return factory_status()

    if re.search(r"^(موافقات|approvals|طلبات\s*الموافقة)\s*$", text, re.I):
        return list_approvals("pending")

    m = re.search(r"^(وافق|approve)\s+(apr_[\w]+)", text, re.I)
    if m:
        return resolve_approval(m.group(2), True, note="user approved")

    m = re.search(r"^(ارفض|reject)\s+(apr_[\w]+)", text, re.I)
    if m:
        return resolve_approval(m.group(2), False, note="user rejected")

    # هدف عام / تشغيل مصنع
    if re.search(
        r"^(هدف|goal|مصنع|factory|شغّل\s*المصنع|نفّذ\s*الهدف|تشغيل\s*ذاتي)\s*[:：\-]?\s*.+",
        text,
        re.I,
    ) or re.search(
        r"(اجعل|حق[ّق]ق|ارفع\s*الدقة|تعرف\s*على).{5,}",
        text,
        re.I,
    ):
        # تجنّب التقاط أوامر أخرى قصيرة
        if len(text) < 8:
            return None
        # لا يلتقط أوامر التدريب المباشرة القصيرة
        if re.match(r"^(درّب|درب|حالة|قائمة|ابحث|صحّح|سجل|ثبّت|افحص)\b", text):
            return None
        return run_factory(text)

    return None
