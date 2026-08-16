"""
Model Training Agent — وكيل عام لإدارة وتدريب نماذج الذكاء الاصطناعي
====================================================================
وكيل أوسع من تدريب CKG فقط: يدير دورة حياة أي نموذج تقريباً —

  • نماذج NSM الداخلية (ArabicTransformer / CKG، NeuralCore، KnowledgeTrainer…)
  • نماذج scikit-learn الكلاسيكية (تصنيف/انحدار) إن توفرت المكتبة
  • نماذج PyTorch البسيطة (شبكة كثيفة) عند توفر torch
  • خطط وأوامر عامة لأي مهمة يتصفها المستخدم (بيانات → خوارزمية → تدريب → تقييم → نشر)

الأوامر النصية العربية تُنفَّذ كأدوات حقيقية؛ أي نص غير مطابق يمر للمحادثة العادية.
لا يرفع استثناءات للواجهة — كل فشل يُعاد كتقرير نصي واضح.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ModelTrainingAgent")

# ═══════════════════════════════════════════════════════════════════════════
# منهجية NSM الموروثة (methodology_engine) — دورة plan/inspect/execute/verify
# ودرس تلقائي من فشل التدريب. fallback آمن كامل: إن غابت الوحدة يعمل
# الوكيل كالمعتاد تمامًا.
# ═══════════════════════════════════════════════════════════════════════════
try:
    from ai.methodology_engine import (
        method_record_lesson as _meth_record_lesson,
        method_step as _meth_step,
        method_task_finished as _meth_task_finished,
        method_task_started as _meth_task_started,
    )
    _METH_OK = True
except Exception:  # pragma: no cover
    _METH_OK = False

    def _meth_task_started(task_id, request, plan=None):
        return None

    def _meth_step(task_id="", step_type="execute", note="", ok=True, meta=None):
        return None

    def _meth_task_finished(task_id="", status="done", ok=True, result_summary=""):
        return None

    def _meth_record_lesson(principle_id, error_context="", lesson=""):
        return None


# ═══════════════════════════════════════════════════════════════════════════
# قناة الكيرنل المنعزل (nb_kernel) — تدريب PyTorch الثقيل في عملية معزولة
# عن Streamlit بدل التنفيذ داخل العملية نفسها، مع fallback كامل للآلية
# الحالية حتى لا يُكسر أي سلوك قائم.
# ═══════════════════════════════════════════════════════════════════════════
_TRAIN_KERNEL_SESSION = "mta_kernel"
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts" / "model_training"
_ARTIFACTS = ARTIFACTS  # أمان: أي مرجع قديم لـ _ARTIFACTS يبقى صالحًا
_MTA_LESSONS_PATH = ARTIFACTS / "mta_lessons.json"
_MTA_LESSONS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _kernel_available() -> bool:
    """هل الكيرنل المنعزل جاهز للاستخدام؟ (ipykernel + حد الجلسات)."""
    try:
        from ai.nb_kernel import _available_for_kernel
        return bool(_available_for_kernel())
    except Exception:
        return False


def _kern_run_cell(source: str, timeout: int = 300) -> Dict[str, Any]:
    """تنفيذ خلية داخل كيرنل التدريب المنعزل (ثابت لكل الوكيل)."""
    try:
        from ai.nb_kernel import run_cell_kernel
        return dict(run_cell_kernel(_TRAIN_KERNEL_SESSION, source, timeout=timeout))
    except Exception as exc:
        return {"ok": False, "outputs": [], "timeout": False,
                "error": f"kernel unavailable: {type(exc).__name__}: {exc}"}


def _kern_collect_text(res: Dict[str, Any]) -> str:
    """تجميع نصوص مخرجات خلية الكيرنل (stdout/stderr/أخطاء)."""
    parts: List[str] = []
    errs: List[str] = []
    for o in res.get("outputs") or []:
        t = o.get("text") or ""
        if o.get("type") == "stream" and o.get("name") == "stdout":
            parts.append(t)
        else:
            errs.append(t)
    err = res.get("error") or ""
    if err and err not in ("None", ""):
        errs.append(str(err))
    joined = "".join(parts)
    if errs:
        joined += ("\n" if joined else "") + "⚠ kernel:\n" + "".join(errs)
    return joined


def _read_mta_lessons() -> Dict[str, Any]:
    """قراءة سجل دروس الوكيل (batch مفضّل ونصائح من أخطاء سابقة)."""
    if _MTA_LESSONS_PATH.is_file():
        data = _load_json(_MTA_LESSONS_PATH)
        if isinstance(data, dict):
            return data
    return {"preferred_batch": None, "lessons": [], "oom_count": 0}


def _write_mta_lessons(data: Dict[str, Any]) -> None:
    try:
        _MTA_LESSONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_MTA_LESSONS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("mta_lessons write: %s", exc)


def _suggest_kernel_batch(n: int, n_features: int = 64, base: int = 16) -> int:
    """حجم batch مقترح للتدريب عبر الكيرنل — يبدأ من base ويصغّر بعد OOM.
    درس من منهجية الوالد: من الفشل نتعلم — batch أصغر تلقائيًا."""
    lessons = _read_mta_lessons()
    pref = lessons.get("preferred_batch")
    if pref is not None:
        return max(1, min(int(pref), n))
    return max(4, min(base, n))


def _record_kernel_oom_lesson(
    context: str = "",
    current_batch: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """درس تلقائي من منهجية الوالد: من فشل OOM نتعلم — batch أصغر تلقائيًا.
    يسجّل الدرس في ذاكرة المنهجية الموروثة ويحدّث المفضّل المحفوظ."""
    lessons = _read_mta_lessons()
    prev = lessons.get("preferred_batch")
    new_batch = None
    if current_batch is not None:
        new_batch = max(1, int(current_batch) // 2)
    elif prev is not None:
        new_batch = max(1, int(prev) // 2)
    lessons["oom_count"] = int(lessons.get("oom_count", 0)) + 1
    if new_batch is not None:
        lessons["preferred_batch"] = new_batch
    lesson_txt = (
        f"فشل التدريب بنقص ذاكرة (OOM) عند batch={current_batch}. "
        f"الدرس: استخدم batch={new_batch} في المرة القادمة "
        f"(تعلّم تلقائي — OOM count={lessons['oom_count']})."
    )
    lessons.setdefault("lessons", []).append({
        "ts": time.time(), "kind": "oom_backoff",
        "from_batch": current_batch, "to_batch": new_batch,
        "count": lessons["oom_count"], "note": lesson_txt,
    })
    _write_mta_lessons(lessons)
    try:
        _meth_record_lesson(
            principle_id=5,  # "التعلم من الأخطاء"
            error_context=(context or "")[:400],
            lesson=lesson_txt,
        )
    except Exception as exc:
        logger.warning("mta oom lesson: %s", exc)
    return lessons


def _torch_kernel_source(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    epochs: int,
    batch_size: int,
    run_id: Optional[str],
    checkpoint_dir: str,
    best_ckpt: Optional[str],
) -> str:
    """بناء كود PyTorch كامل بصيغة خلية كيرنل — نفس منطق train_torch_on_arrays
    لكن يُنفَّذ في عملية معزولة (لا يحجب الشات، ويقبض OOM/Timeout)."""
    return (
        "import json, time\n"
        "import numpy as np, torch, torch.nn as nn\n"
        f"_X = {json.dumps(X.tolist())}\n"
        f"_y = {json.dumps(y.tolist())}\n"
        f"_task = {json.dumps(task)}\n"
        f"_epochs = {int(epochs)}\n"
        f"_batch_size = {int(batch_size)}\n"
        f"_run_id = {json.dumps(run_id)}\n"
        f"_ckpt_dir = {json.dumps(checkpoint_dir)}\n"
        f"_best_ckpt = {json.dumps(best_ckpt)}\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "n_samples, n_features = _X.shape[0], _X.shape[1]\n"
        "n_out = (int(np.max(_y)) + 1) if _task == 'classification' else 1\n"
        "model = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), "
        "nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_out)).to(device)\n"
        "opt = torch.optim.Adam(model.parameters(), lr=1e-3)\n"
        "loss_fn = nn.CrossEntropyLoss() if _task == 'classification' "
        "else nn.MSELoss()\n"
        "X_t = torch.tensor(_X, dtype=torch.float32, device=device)\n"
        "y_t = torch.tensor(_y, dtype=torch.long if _task == 'classification' "
        "else torch.float32, device=device)\n"
        "if _task != 'classification': y_t = y_t.reshape(-1, 1)\n"
        "idx = np.random.permutation(n_samples)\n"
        "split = max(1, int(0.8 * n_samples))\n"
        "tr, te = idx[:split], idx[split:] if split < n_samples else idx[-1:]\n"
        "hist, val_hist, best_val = [], [], float('inf')\n"
        "import os; os.makedirs(_ckpt_dir, exist_ok=True)\n"
        "for ep in range(_epochs):\n"
        "    model.train()\n"
        "    perm = np.random.permutation(tr)\n"
        "    ep_loss, n_b = 0.0, 0\n"
        "    for i in range(0, len(perm), _batch_size):\n"
        "        bi = perm[i:i + _batch_size]\n"
        "        opt.zero_grad(); loss = loss_fn(model(X_t[bi]), y_t[bi])\n"
        "        loss.backward(); opt.step()\n"
        "        ep_loss += float(loss.item()); n_b += 1\n"
        "    hist.append(ep_loss / max(1, n_b))\n"
        "    model.eval()\n"
        "    with torch.no_grad():\n"
        "        vl = float(loss_fn(model(X_t[te]), y_t[te]).item())\n"
        "    val_hist.append(vl)\n"
        "    if vl < best_val: best_val = vl\n"
        "    if (ep + 1) % 5 == 0 or vl <= best_val:\n"
        "        torch.save({'epoch': ep + 1, 'model_state': model.state_dict(), "
        "'optimizer_state': opt.state_dict(), 'best_val_loss': best_val, "
        "'history': hist, 'val_history': val_hist, 'task': _task, "
        "'n_features': n_features, 'batch_size': _batch_size}, "
        "os.path.join(_ckpt_dir, 'latest.pt'))\n"
        "    if vl <= best_val:\n"
        "        torch.save({'epoch': ep + 1, 'model_state': model.state_dict(), "
        "'optimizer_state': opt.state_dict(), 'best_val_loss': best_val, "
        "'history': hist, 'val_history': val_hist, 'task': _task, "
        "'n_features': n_features, 'batch_size': _batch_size}, "
        "os.path.join(_ckpt_dir, 'best.pt'))\n"
        "    if len(hist) > 2 and abs(hist[-1] - hist[-2]) < 1e-4 and vl < hist[-1]:\n"
        "        break\n"
        "print(json.dumps({'epochs': ep + 1, 'train_loss_last5': hist[-5:], "
        "'val_loss_last5': val_hist[-5:], 'best_val_loss': best_val, "
        "'device': str(device), 'batch_size': _batch_size, "
        "'checkpoint_dir': _ckpt_dir}))\n"
    )


def train_via_kernel(
    source: str,
    timeout: int = 300,
    fallback: Optional[Callable[[], str]] = None,
) -> str:
    """
    تنفيذ كود تدريب (PyTorch/شامل) داخل كيرنل معزول عن عملية Streamlit
    — يمنع حجب الشات أثناء تدريب طويل، ويقبض الأخطاء (OOM/Timeout) بأمان.

    عند توفر الكيرنل يُنفَّذ هناك؛ وإلا يُنفَّذ fallback الممرَّر (الآلية
    القديمة). إن لم يُمرَّر fallback يُعاد تقرير فشل واضح — الوكيل لا يتجمد.
    """
    if not _kernel_available():
        if fallback is not None:
            try:
                return fallback()
            except Exception as exc:
                return f"❌ {type(exc).__name__}: {exc}"
        return ("⚠ الكيرنل المنعزل غير متاح في هذه البيئة."
                " جرّب إعادة تحميل التطبيق.")
    res = _kern_run_cell(source, timeout=timeout)
    txt = _kern_collect_text(res)
    if res.get("ok") and not res.get("timeout"):
        return txt or "✅ نفّذ الكيرنل الخلية بنجاح (بلا مخرجات نصية)."
    if res.get("timeout"):
        note = f"⏱ تجاوزت الخلية المهلة ({timeout}s) — أُوقفت بأمان دون حجب الشات."
        try:
            from ai.nb_kernel import restart_kernel
            restart_kernel(_TRAIN_KERNEL_SESSION)
        except Exception:
            pass
        return (note + "\n\n" + txt).strip()
    return f"❌ فشل التنفيذ في الكيرنل المنعزل:\n\n{txt}".strip()

ARTIFACTS.mkdir(parents=True, exist_ok=True)
# مزامنة اختيارية للـ checkpoints مع تخزين خارجي (معطّلة افتراضياً — راجع
# ai/checkpoint_storage.py). لا تكسر الوكيل إن تعذّر الاستيراد.
try:
    from ai.checkpoint_storage import (
        sync_checkpoint_after_save as _ckpt_sync_after_save,
        restore_checkpoint_if_missing as _ckpt_restore_if_missing,
    )
except Exception:  # pragma: no cover
    def _ckpt_sync_after_save(run_id, files):
        return []

    def _ckpt_restore_if_missing(run_id, checkpoint_dir, which="latest"):
        return False

# Sandbox / Guardrails (اختياري — لا يكسر الوكيل إن تعذّر الاستيراد)
try:
    from ai.training_sandbox import (
        clamp_epochs as _sb_clamp_epochs,
        clamp_samples as _sb_clamp_samples,
        detect_compute as _sb_detect_compute,
        list_mission_logs as _sb_list_missions,
        run_first_mission as _sb_run_first_mission,
        run_second_mission as _sb_run_second_mission,
        run_mission as _sb_run_mission,
        sandbox_status_report as _sb_status,
        EarlyStopping as _EarlyStopping,
        assert_write_allowed as _sb_assert_write,
    )
    _SANDBOX_OK = True
except Exception:
    _SANDBOX_OK = False

    def _sb_clamp_epochs(e):
        return max(1, min(int(e), 50))

    def _sb_clamp_samples(n):
        return max(1, min(int(n), 5000))


try:
    from ai.training_feedback_loop import handle_feedback_command as _fb_handle
    _FEEDBACK_OK = True
except Exception:
    _fb_handle = None
    _FEEDBACK_OK = False

try:
    from ai.training_web_access import handle_web_command as _web_handle
    _WEB_ACCESS_OK = True
except Exception:
    _web_handle = None
    _WEB_ACCESS_OK = False

try:
    from ai.training_factory import handle_factory_command as _factory_handle
    _FACTORY_OK = True
except Exception:
    _factory_handle = None
    _FACTORY_OK = False

try:
    from ai.aiaas_platform import handle_aiaas_command as _aiaas_handle
    _AIAAS_OK = True
except Exception:
    _aiaas_handle = None
    _AIAAS_OK = False

try:
    from ai.apex_autonomy import handle_apex_command as _apex_handle
    _APEX_OK = True
except Exception:
    _apex_handle = None
    _APEX_OK = False

try:
    from ai.remote_gpu_provider import handle_remote_gpu_command as _remote_gpu_handle
    _REMOTE_GPU_OK = True
except Exception:
    _remote_gpu_handle = None
    _REMOTE_GPU_OK = False

try:
    from ai.kaggle_provider import handle_kaggle_command as _kaggle_handle
    _KAGGLE_OK = True
except Exception:
    _kaggle_handle = None
    _KAGGLE_OK = False

try:
    from ai.remote_training_orchestrator import handle_orchestrator_command as _orch_handle
    _ORCH_OK = True
except Exception:
    _orch_handle = None
    _ORCH_OK = False

try:
    from ai.ai_architect import handle_architect_command as _architect_handle
    _ARCHITECT_OK = True
except Exception:
    _architect_handle = None
    _ARCHITECT_OK = False

try:
    from ai.scientist_manager import handle_scientist_command as _scientist_handle
    _SCIENTIST_OK = True
except Exception:
    _scientist_handle = None
    _SCIENTIST_OK = False

try:
    from ai.meta_ai_system import handle_meta_command as _meta_handle
    _META_OK = True
except Exception:
    _meta_handle = None
    _META_OK = False

try:
    from ai.super_ai_orchestrator import handle_super_command as _super_handle
    _SUPER_OK = True
except Exception:
    _super_handle = None
    _SUPER_OK = False

try:
    from ai.commercial_economy import handle_economic_command as _economic_handle
    _ECONOMIC_OK = True
except Exception:
    _economic_handle = None
    _ECONOMIC_OK = False

try:
    from ai.production_activation import handle_production_command as _prod_handle
    _PROD_OK = True
except Exception:
    _prod_handle = None
    _PROD_OK = False

try:
    from ai.civilization_layer import handle_civilization_command as _civ_handle
    _CIV_OK = True
except Exception:
    _civ_handle = None
    _CIV_OK = False

try:
    from ai.sovereignty_loop import handle_sovereignty_command as _sov_handle
    _SOV_OK = True
except Exception:
    _sov_handle = None
    _SOV_OK = False

try:
    from ai.social_swarm import handle_social_swarm_command as _social_swarm_handle
    _SOCIAL_SWARM_OK = True
except Exception:
    _social_swarm_handle = None
    _SOCIAL_SWARM_OK = False

try:
    from ai.active_retrain_loop import handle_active_retrain_command as _retrain_handle
    _RETRAIN_OK = True
except Exception:
    _retrain_handle = None
    _RETRAIN_OK = False

try:
    from world_model.predictive_sim import handle_predictive_command as _pred_handle
    _PRED_OK = True
except Exception:
    _pred_handle = None
    _PRED_OK = False


try:
    from ai.autonomous_train_devops import handle_train_devops_command as _devops_handle
    _DEVOPS_OK = True
except Exception:
    _devops_handle = None
    _DEVOPS_OK = False

try:
    from ai.reinforcement_learning import handle_rl_command as _rl_handle
    _RL_OK = True
except Exception:
    _rl_handle = None
    _RL_OK = False

try:
    from ai.stripe_billing import handle_billing_command as _bill_handle
    _BILL_OK = True
except Exception:
    _bill_handle = None
    _BILL_OK = False

try:
    from ai.ckg_quality_tool import handle_ckg_quality_command as _ckgq_handle
    _CKGQ_OK = True
except Exception:
    _ckgq_handle = None
    _CKGQ_OK = False

try:
    from ai.git_lfs_helper import handle_lfs_command as _lfs_handle
    _LFS_OK = True
except Exception:
    _lfs_handle = None
    _LFS_OK = False

try:
    from ai.cognitive_microkernel import handle_kernel_command as _kern_handle
    _KERN_OK = True
except Exception:
    _kern_handle = None
    _KERN_OK = False

try:
    from ai.cosmic_mesh import handle_mesh_command as _mesh_handle
    _MESH_OK = True
except Exception:
    _mesh_handle = None
    _MESH_OK = False

try:
    from ai.mcp_internal_gateway import handle_gateway_command as _gw_handle
    _GW_OK = True
except Exception:
    _gw_handle = None
    _GW_OK = False

try:
    from ai.quantization_worker import handle_quant_command as _quant_handle
    _QUANT_OK = True
except Exception:
    _quant_handle = None
    _QUANT_OK = False

try:
    from ai.sensors_training_bridge import handle_sensor_bridge_command as _sbridge_handle
    _SBRIDGE_OK = True
except Exception:
    _sbridge_handle = None
    _SBRIDGE_OK = False

try:
    from ai.continuous_training_agent import handle_continuous_command as _cont_handle
    _CONT_OK = True
except Exception:
    _cont_handle = None
    _CONT_OK = False

try:
    from ai.hierarchical_moe import handle_moe_command as _moe_handle
    _MOE_OK = True
except Exception:
    _moe_handle = None
    _MOE_OK = False

try:
    from ai.command_lexicon import (
        handle_help_command as _help_handle,
        rewrite_to_canonical as _rewrite_cmd,
        normalize_ar as _norm_ar,
    )
    _LEXICON_OK = True
except Exception:
    _help_handle = None
    _rewrite_cmd = lambda t: t
    _norm_ar = lambda t: (t or "").strip().lower()
    _LEXICON_OK = False

try:
    from ai.gpu_runtime import (
        torch_device as _gpu_torch_device,
        suggest_batch_size as _gpu_suggest_batch,
        run_with_oom_backoff as _gpu_oom_backoff,
        empty_cache as _gpu_empty_cache,
        device_report_md as _gpu_report,
        detect_device as _gpu_detect,
    )
    _GPU_OK = True
except Exception:
    _GPU_OK = False
    def _gpu_torch_device(force_gpu=None):
        import torch
        return torch.device("cpu"), None
    def _gpu_suggest_batch(n, n_features=64, free_vram_gb=None, base=32):
        return max(4, min(32, n))
    def _gpu_oom_backoff(fn, initial_batch, min_batch=1, max_retries=4):
        return fn(initial_batch), initial_batch, ["cpu_fallback"]
    def _gpu_empty_cache():
        pass
    def _gpu_report():
        return "GPU runtime غير محمّل"
    def _gpu_detect(force_gpu=None):
        return None

# ── مسارات NSM المعروفة (اختيارية — أحد الأهداف وليست الوحيدة) ──────────────
STATE_V3 = ROOT / "ckg_train_state_v3.json"
SENTENCES_V3 = ROOT / "ckg_sentences_v3.pkl"
SENTENCES_V2 = ROOT / "ckg_sentences_v2.pkl"
SENTENCES_V1 = ROOT / "ckg_sentences.pkl"
GENERAL_AR = ROOT / "ckg_sentences_general_ar.pkl"
WEIGHTS_V3 = ROOT / "models" / "transformer_ckg_v3"
TRAIN_V3 = ROOT / "train_batch_v3.py"
TRAIN_BATCH = ROOT / "train_batch.py"
TRAIN_YEMENI = ROOT / "train_yemeni.py"
TRAIN_PILOT = ROOT / "train_pilot_general_ar.py"

_MAX_PACKS = 2
_MIN_RAM_GB = 1.2
_TIMEOUT_S = 600

# ── اكتشاف المكتبات الاختيارية ─────────────────────────────────────────────
try:
    import sklearn  # noqa: F401
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_squared_error,
        r2_score,
        classification_report,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    _SKLEARN_OK = True
except Exception:
    _SKLEARN_OK = False

# 🚀 تأجيل تحميل torch: يُحمل عند أول استخدام فعلي (يوفّر ~1.5s عند بدء التطبيق)
_TORCH_ATTEMPTED = False

def _ensure_torch_agent():
    # type: () -> bool
    '''تحميل تأجيلي لـ torch — أول نداء فقط.'''
    global _TORCH_OK, _TORCH_ATTEMPTED, torch, nn, optim
    if _TORCH_ATTEMPTED:
        return _TORCH_OK
    _TORCH_ATTEMPTED = True
    try:
        import torch as _t
        import torch.nn as _nn
        import torch.optim as _optim
        torch, nn, optim = _t, _nn, _optim
        _TORCH_OK = True
    except Exception:
        _TORCH_OK = False
    return _TORCH_OK


def _ram_gb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        return (info.get("MemAvailable") or info.get("MemFree", 0)) / (1024.0 * 1024.0)
    except Exception:
        return 1.0


def _load_json(path: Path) -> Optional[dict]:
    try:
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("json %s: %s", path, e)
    return None


def _count_pkl(path: Path) -> Optional[int]:
    try:
        if not path.is_file():
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        return len(data) if hasattr(data, "__len__") else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 1) اكتشاف البيئة والأهداف المتاحة
# ═══════════════════════════════════════════════════════════════════════════

def inventory() -> str:
    """جرد شامل: مكتبات، سكربتات تدريب، بيانات، أوزان، موارد."""
    lines = [
        "## 🧭 جرد بيئة التدريب (عام — ليس CKG فقط)",
        "",
        "### المكتبات",
        f"- numpy: ✅ ({np.__version__})",
        f"- scikit-learn: {'✅' if _SKLEARN_OK else '❌ غير مثبت'}",
        f"- PyTorch: {'✅ ' + torch.__version__ if _ensure_torch_agent() else '❌ غير متاح'}",
        "",
        "### أهداف NSM الداخلية (اختيارية)",
    ]
    for label, path in [
        ("train_batch_v3.py (ArabicTransformer/CKG)", TRAIN_V3),
        ("train_batch.py", TRAIN_BATCH),
        ("train_yemeni.py", TRAIN_YEMENI),
        ("train_pilot_general_ar.py", TRAIN_PILOT),
        ("ai/knowledge_trainer.py", ROOT / "ai" / "knowledge_trainer.py"),
        ("ai/experience_trainer.py", ROOT / "ai" / "experience_trainer.py"),
        ("ai/arabic_transformer.py", ROOT / "ai" / "arabic_transformer.py"),
        ("ai/neural_core.py", ROOT / "ai" / "neural_core.py"),
    ]:
        lines.append(f"- `{label}`: {'✅' if path.is_file() else '❌'}")

    lines.append("")
    lines.append("### مصادر بيانات معروفة")
    for label, path in [
        ("ckg_sentences_v3.pkl", SENTENCES_V3),
        ("ckg_sentences_v2.pkl", SENTENCES_V2),
        ("ckg_sentences.pkl", SENTENCES_V1),
        ("ckg_sentences_general_ar.pkl", GENERAL_AR),
    ]:
        n = _count_pkl(path)
        if n is not None:
            mb = path.stat().st_size / (1024 * 1024)
            lines.append(f"- `{label}`: **{n:,}** عينة ({mb:.1f} MB)")
        else:
            lines.append(f"- `{label}`: غير متاح")

    # أي CSV/JSON في data/ أو artifacts/
    extra: List[str] = []
    for folder in (ROOT / "data", ARTIFACTS, ROOT / "knowledge_sources"):
        if not folder.is_dir():
            continue
        for p in folder.rglob("*"):
            if p.suffix.lower() in {".csv", ".json", ".jsonl", ".tsv", ".npy", ".npz"} and p.is_file():
                try:
                    mb = p.stat().st_size / (1024 * 1024)
                    if mb < 200:  # تجاهل ملفات ضخمة جداً في العرض
                        extra.append(f"  - `{p.relative_to(ROOT)}` ({mb:.2f} MB)")
                except Exception:
                    pass
    if extra:
        lines.append("")
        lines.append("### ملفات بيانات إضافية (عينة)")
        lines.extend(extra[:25])
        if len(extra) > 25:
            lines.append(f"  - … و{len(extra) - 25} ملف إضافي")

    lines.append("")
    lines.append(f"### الموارد: رام متاحة ≈ **{_ram_gb():.2f} GB**")
    lines.append(f"### مجلد مخرجات الوكيل: `{ARTIFACTS.relative_to(ROOT)}/`")
    lines.append("")
    lines.append(
        "الوكيل يدعم: (0) مهندس معماري — تحكيم/بحث فائق/ضغط/اتحاد، (1) تدريب عام sklearn/torch، (2) تشغيل سكربتات NSM، (2b) Kaggle API + Dual T4 — أوامر `حالة kaggle` / `جهّز kaggle`، "
        "(3) تخطيط دورة حياة لأي مهمة تصفها."
    )
    return "\n".join(lines)


def lifecycle_plan(task_hint: str = "") -> str:
    """خطة دورة حياة عامة لأي نموذج — مع تخصيص اختياري حسب وصف المهمة."""
    hint = (task_hint or "").strip()
    specialized = ""
    if hint:
        specialized = f"\n### تخصيص حسب مهمتك\n> {hint}\n"

    return f"""## 🗺️ خطة دورة حياة تدريب نموذج (عامة)
{specialized}
### 1) جمع ومعالجة البيانات
- حدّد المصدر: ملفات محلية، CKG، واجهات API، أو توليد اصطناعي للتجربة.
- نظّف القيم الناقصة، وحِّد الترميز، اقسم Train/Val/Test (مثلاً 70/15/15).
- أمر مفيد: **جرد البيئة** لرؤية ما هو متاح الآن.

### 2) اختيار الخوارزمية / البنية
| نوع المهمة | خيارات سريعة |
|------------|----------------|
| تصنيف جدولي | LogisticRegression، RandomForest، GradientBoosting (sklearn) |
| انحدار جدولي | Ridge، LinearRegression، غابات (sklearn) |
| نص عربي / معرفة NSM | ArabicTransformer، NeuralCore، KnowledgeTrainer |
| شبكة عصبية عامة | MLP بـ PyTorch (طبقات كثيفة) |
| مخصّص | سكربت `train_*.py` في جذر المشروع |

### 3) ضبط المعلمات الفائقة
- ابدأ بإعدادات افتراضية معقولة ثم راقب الخسارة/الدقة.
- للـCKG: `PACK_SIZE` حسب الرام (انظر **اقترح إعدادات ckg**).
- للشبكات: LR، batch size، epochs، weight decay، early stopping.

### 4) إدارة التدريب
- راقب الخسارة والتحقّق؛ أوقف عند أفضل أداء (early stopping).
- احفظ checkpoints دورياً؛ سجّل البذور والبيئة لإعادة الإنتاج.
- أوامر: **درّب تصنيف تجريبي** / **درّب انحدار تجريبي** / **درّب شبكة torch** / **شغّل تدريب ckg**.

### 5) التقييم والاختبار
- مقاييس: Accuracy/F1 للتصنيف، RMSE/R² للانحدار، loss للعصبية.
- افحص التحيز والأخطاء المنهجية على شريحة اختبار محجوبة.
- لـNSM المعرفي: Faithfulness Verifier عند توفره.

### 6) النشر والحفظ
- احفظ النموذج تحت `artifacts/model_training/`.
- أوزان NSM الكبيرة تبقى محلية (مُتجاهَلة في git عمداً).
- وثّق الإصدار، البيانات، والمقاييس بجانب الملف.

---
**مبدأ الوكيل:** لا يختلق نتائج تدريب — إما يشغّل أداة حقيقية أو يوضح أن الأمر يحتاج وصفاً/بيانات أوضح.
"""


# ═══════════════════════════════════════════════════════════════════════════
# 2) أدوات CKG / NSM (هدف واحد من عدة أهداف)
# ═══════════════════════════════════════════════════════════════════════════

def ckg_status() -> str:
    lines = ["## 📊 حالة تدريب CKG / ArabicTransformer v3 (هدف NSM)", ""]
    state = _load_json(STATE_V3)
    n = _count_pkl(SENTENCES_V3)
    if state:
        pos = int(state.get("position", 0) or 0)
        total = n or pos or 1
        if n:
            total = n
        pct = 100.0 * pos / total if total else 0.0
        tail = state.get("loss_history_tail") or []
        recent = tail[-8:] if tail else []
        avg = sum(recent) / len(recent) if recent else float("nan")
        lines += [
            f"- التقدّم: **{pos:,}/{total:,} ({pct:.1f}%)**",
            f"- runs: {state.get('runs', '—')} | tokenizer: `{state.get('tokenizer_version', '—')}`",
            f"- model: `{state.get('model_version', '—')}`",
            f"- آخر حزمة: size={state.get('last_pack_size')} packs={state.get('last_packs_per_run')} "
            f"elapsed={state.get('last_elapsed_s')}s RAM={state.get('last_avail_ram_gb')}GB",
        ]
        if recent:
            lines.append(f"- آخر loss: {', '.join(f'{x:.3f}' for x in recent)} (متوسط {avg:.3f})")
    else:
        lines.append("لا يوجد `ckg_train_state_v3.json`.")
    lines.append(f"- أوزان: {'موجودة' if WEIGHTS_V3.is_dir() else 'غير موجودة محلياً (طبيعي — gitignore)'}")
    lines.append(f"- رام الآن: {_ram_gb():.2f} GB")
    return "\n".join(lines)


def ckg_recommend() -> str:
    avail = _ram_gb()
    safety, ref_pack, ref_peak = 0.35, 80, 2.93
    budget = max(0.0, avail - safety)
    if budget < 1.2:
        pack, packs = 0, 0
        note = "رام غير كافية لتدريب ArabicTransformer (~120M). استخدم بيئة ≥ 3.5 GB."
    else:
        pack = max(4, min(80, (int(ref_pack * (budget / ref_peak)) // 4) * 4))
        packs = 8 if avail >= 3.2 else (4 if avail >= 2.4 else (2 if avail >= 1.8 else 1))
        note = "إعدادات مناسبة لهذه الجلسة."
    return (
        f"## ⚙️ إعدادات مقترحة لتدريب CKG v3\n\n"
        f"- رام: **{avail:.2f} GB**\n"
        f"- PACK_SIZE=`{pack}` | PACKS_PER_RUN=`{packs}`\n"
        f"- Tokenizer: word-v1 (متوافق مع الحالة الحالية)\n"
        f"- بنية مرجعية: D_MODEL=2304, 16 طبقة, LR=1e-4\n\n"
        f"{note}\n\n"
        f"```bash\nNSM_PACK_SIZE={pack or 16} NSM_PACKS_PER_RUN={max(packs, 1)} python3 train_batch_v3.py\n```"
    )


def ckg_loss_trend() -> str:
    state = _load_json(STATE_V3)
    if not state:
        return "لا يوجد سجل خسارة CKG."
    tail = list(state.get("loss_history_tail") or [])
    if len(tail) < 4:
        return f"نقاط غير كافية ({len(tail)})."
    q = max(1, len(tail) // 4)
    first, last = sum(tail[:q]) / q, sum(tail[-q:]) / q
    delta = last - first
    trend = "تحسّن" if delta < -0.05 else ("تدهور" if delta > 0.05 else "مستقر")
    return (
        f"## 📈 اتجاه خسارة CKG\n\n"
        f"- نقاط: {len(tail)} | أول ربع: {first:.3f} | آخر ربع: {last:.3f} | Δ={delta:+.3f} → **{trend}**\n"
        f"- آخر 12: {', '.join(f'{x:.3f}' for x in tail[-12:])}"
    )


def run_ckg_step(packs: int = 1, pack_size: Optional[int] = None, dry_run: bool = False) -> str:
    packs = max(1, min(int(packs), _MAX_PACKS))
    avail = _ram_gb()
    if not TRAIN_V3.is_file():
        return "سكربت train_batch_v3.py غير موجود."
    if dry_run:
        return (
            f"🧪 dry-run CKG: packs={packs}, size={pack_size or 'auto'}, RAM={avail:.2f}GB\n"
            f"```bash\nNSM_PACKS_PER_RUN={packs}"
            + (f" NSM_PACK_SIZE={pack_size}" if pack_size else "")
            + " python3 train_batch_v3.py\n```"
        )
    if avail < _MIN_RAM_GB:
        return (
            f"❌ رُفض: رام {avail:.2f} GB < {_MIN_RAM_GB}. "
            "النموذج كبير — شغّل على بيئة أقوى أو استخدم dry-run."
        )
    env = os.environ.copy()
    env["NSM_PACKS_PER_RUN"] = str(packs)
    if pack_size:
        env["NSM_PACK_SIZE"] = str(int(pack_size))
    env.pop("NSM_RESET_TRAIN", None)
    try:
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(TRAIN_V3)],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
        )
        out = (proc.stdout or "")[-3500:]
        err = (proc.stderr or "")[-1200:]
        body = f"رمز={proc.returncode} | {time.time() - t0:.1f}s\n```\n{out.strip()}\n```"
        if proc.returncode != 0 and err.strip():
            body += f"\nstderr:\n```\n{err.strip()}\n```"
        return body
    except subprocess.TimeoutExpired:
        return f"انتهت المهلة ({_TIMEOUT_S}s)."
    except Exception as e:
        return f"فشل: {type(e).__name__}: {e}"


def run_nsm_script(script_name: str, dry_run: bool = False) -> str:
    """تشغيل أي سكربت train_*.py معروف في جذر المشروع."""
    mapping = {
        "v3": TRAIN_V3,
        "ckg": TRAIN_V3,
        "batch": TRAIN_BATCH,
        "yemeni": TRAIN_YEMENI,
        "pilot": TRAIN_PILOT,
        "general": TRAIN_PILOT,
    }
    key = script_name.strip().lower()
    path = mapping.get(key)
    if path is None:
        # اسم ملف مباشر
        cand = ROOT / script_name
        if cand.is_file() and cand.suffix == ".py":
            path = cand
    if path is None or not path.is_file():
        return (
            f"سكربت غير معروف: `{script_name}`.\n"
            f"المتاح: v3/ckg, batch, yemeni, pilot/general أو اسم ملف train_*.py"
        )
    if dry_run:
        return f"🧪 dry-run: `python3 {path.name}`"
    avail = _ram_gb()
    if avail < 1.0:
        return f"❌ رام منخفضة ({avail:.2f} GB)."
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
        )
        out = (proc.stdout or "")[-3000:]
        return f"تشغيل `{path.name}` → رمز {proc.returncode}\n```\n{out.strip()}\n```"
    except Exception as e:
        return f"فشل تشغيل {path.name}: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# 3) تدريب عام — sklearn
# ═══════════════════════════════════════════════════════════════════════════

def train_sklearn_demo(
    task: str = "classification",
    n_samples: int = 800,
    n_features: int = 12,
    model_name: str = "auto",
) -> str:
    if not _SKLEARN_OK:
        return (
            "❌ scikit-learn غير مثبت في هذه البيئة.\n"
            "ثبّته بـ `pip install scikit-learn` أو استخدم **درّب شبكة torch** / أهداف NSM."
        )
    task = task.lower().strip()
    n_samples = max(50, min(int(n_samples), 5000))
    n_features = max(2, min(int(n_features), 64))
    t0 = time.time()

    try:
        if task in ("regression", "انحدار", "reg"):
            X, y = make_regression(
                n_samples=n_samples, n_features=n_features, noise=12.0, random_state=42
            )
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            name = model_name if model_name != "auto" else "ridge"
            if name in ("linear", "linreg"):
                model = Pipeline([("scaler", StandardScaler()), ("m", LinearRegression())])
            else:
                model = Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=1.0))])
                name = "ridge"
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mse = float(mean_squared_error(y_test, pred))
            r2 = float(r2_score(y_test, pred))
            out_path = ARTIFACTS / f"sklearn_{name}_reg_{int(time.time())}.joblib"
            try:
                import joblib

                joblib.dump(model, out_path)
                saved = str(out_path.relative_to(ROOT))
            except Exception:
                saved = "(تعذّر الحفظ — joblib غير متاح)"
            return (
                f"## ✅ تدريب انحدار (sklearn / {name})\n\n"
                f"- عينات: {n_samples} | ميزات: {n_features}\n"
                f"- MSE={mse:.4f} | R²={r2:.4f}\n"
                f"- المدة: {time.time() - t0:.2f}s\n"
                f"- محفوظ: `{saved}`"
            )

        # classification default
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_redundant=0,
            n_classes=2,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        name = model_name if model_name != "auto" else "rf"
        if name in ("logreg", "logistic", "lr"):
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("m", LogisticRegression(max_iter=500, random_state=42)),
                ]
            )
            name = "logreg"
        elif name in ("gb", "gbm", "boost"):
            model = GradientBoostingClassifier(random_state=42)
            name = "gbm"
        else:
            model = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
            name = "rf"
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, pred))
        f1 = float(f1_score(y_test, pred, average="weighted"))
        report = classification_report(y_test, pred, digits=3)
        out_path = ARTIFACTS / f"sklearn_{name}_clf_{int(time.time())}.joblib"
        try:
            import joblib

            joblib.dump(model, out_path)
            saved = str(out_path.relative_to(ROOT))
        except Exception:
            # fallback pickle
            out_path = ARTIFACTS / f"sklearn_{name}_clf_{int(time.time())}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(model, f)
            saved = str(out_path.relative_to(ROOT))
        return (
            f"## ✅ تدريب تصنيف (sklearn / {name})\n\n"
            f"- عينات: {n_samples} | ميزات: {n_features}\n"
            f"- Accuracy={acc:.4f} | F1={f1:.4f}\n"
            f"- المدة: {time.time() - t0:.2f}s\n"
            f"- محفوظ: `{saved}`\n\n"
            f"```\n{report}\n```"
        )
    except Exception as e:
        return f"❌ فشل تدريب sklearn: {type(e).__name__}: {e}\n```\n{traceback.format_exc()[-800:]}\n```"


# ═══════════════════════════════════════════════════════════════════════════
# 4) تدريب عام — PyTorch MLP
# ═══════════════════════════════════════════════════════════════════════════

def train_torch_mlp(
    task: str = "classification",
    n_samples: int = 600,
    n_features: int = 16,
    epochs: int = 25,
    hidden: int = 64,
) -> str:
    if not _TORCH_OK:
        return "❌ PyTorch غير متاح في هذه البيئة."

    task = task.lower().strip()
    n_samples = max(80, min(int(n_samples), 4000))
    n_features = max(2, min(int(n_features), 128))
    epochs = _sb_clamp_epochs(max(3, min(int(epochs), 100)))
    hidden = max(8, min(int(hidden), 512))
    t0 = time.time()
    device, _dev_info = _gpu_torch_device()

    try:
        rng = np.random.default_rng(42)
        if task in ("regression", "انحدار", "reg"):
            X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
            true_w = rng.normal(size=(n_features, 1)).astype(np.float32)
            y = (X @ true_w + 0.1 * rng.normal(size=(n_samples, 1))).astype(np.float32)
            n_out = 1
            loss_fn = nn.MSELoss()
            metric_name = "MSE"
        else:
            X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
            logits = X[:, : min(4, n_features)].sum(axis=1)
            y = (logits > 0).astype(np.int64)
            n_out = 2
            loss_fn = nn.CrossEntropyLoss()
            metric_name = "Accuracy"
            task = "classification"

        idx = rng.permutation(n_samples)
        split = int(n_samples * 0.75)
        tr, te = idx[:split], idx[split:]
        Xtr = torch.tensor(X[tr], device=device)
        Xte = torch.tensor(X[te], device=device)
        if n_out == 1:
            ytr = torch.tensor(y[tr], device=device)
            yte = torch.tensor(y[te], device=device)
        else:
            ytr = torch.tensor(y[tr], device=device)
            yte = torch.tensor(y[te], device=device)

        model = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_out),
        ).to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3)

        history: List[float] = []
        val_history: List[float] = []
        stopped_early = False
        es = None
        if _SANDBOX_OK:
            try:
                es = _EarlyStopping.from_config()
            except Exception:
                es = None
        model.train()
        actual_epochs = 0
        for ep in range(epochs):
            opt.zero_grad()
            out = model(Xtr)
            loss = loss_fn(out, ytr if n_out == 1 else ytr)
            loss.backward()
            opt.step()
            history.append(float(loss.item()))
            actual_epochs = ep + 1
            model.eval()
            with torch.no_grad():
                vo = model(Xte)
                vl = float(loss_fn(vo, yte if n_out == 1 else yte).item())
            model.train()
            val_history.append(vl)
            if es is not None and es.step(vl):
                stopped_early = True
                break

        model.eval()
        with torch.no_grad():
            out_te = model(Xte)
            if n_out == 1:
                metric = float(loss_fn(out_te, yte).item())
            else:
                pred = out_te.argmax(dim=-1)
                metric = float((pred == yte).float().mean().item())

        out_path = ARTIFACTS / f"torch_mlp_{task}_{int(time.time())}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "n_features": n_features,
                "hidden": hidden,
                "n_out": n_out,
                "task": task,
                "epochs": epochs,
                "metric_name": metric_name,
                "metric": metric,
                "loss_history": history[-20:],
            },
            out_path,
        )

        return (
            f"## ✅ تدريب شبكة PyTorch (MLP)\n\n"
            f"- المهمة: **{task}** | عينات={n_samples} | ميزات={n_features}\n"
            f"- بنية: {n_features}→{hidden}→{hidden // 2}→{n_out} | epochs={epochs}\n"
            f"- {metric_name} (اختبار) = **{metric:.4f}**\n"
            f"- آخر خسائر: {', '.join(f'{x:.4f}' for x in history[-6:])}\n"
            f"- المدة: {time.time() - t0:.2f}s\n"
            f"- محفوظ: `{out_path.relative_to(ROOT)}`"
        )
    except Exception as e:
        return f"❌ فشل torch MLP: {type(e).__name__}: {e}\n```\n{traceback.format_exc()[-800:]}\n```"


# ═══════════════════════════════════════════════════════════════════════════
# 5) اقتراح خوارزمية حسب وصف المهمة
# ═══════════════════════════════════════════════════════════════════════════

def suggest_algorithm(description: str) -> str:
    d = (description or "").strip().lower()
    lines = ["## 🧠 اقتراح خوارزميات / بنى", ""]
    if not d:
        lines.append("صف المهمة بإيجاز (مثلاً: تصنيف نصوص عربية، توقع سعر، كشف شذوذ…).")
        return "\n".join(lines)

    lines.append(f"> المهمة: {description.strip()}")
    lines.append("")

    suggestions: List[str] = []
    if any(k in d for k in ("قرآن", "ckg", "معرفة", "مفاهيم", "عربي نص", "نص عربي", "transformer")):
        suggestions.append(
            "- **ArabicTransformer / train_batch_v3** — الأنسب للمعرفة العربية المبنية على CKG داخل NSM."
        )
        suggestions.append("- **KnowledgeTrainer** — لدمج حقائق جديدة في الرسم المعرفي.")
    if any(k in d for k in ("تصنيف", "classif", "فئة", "spam", "مشاعر", "sentiment")):
        suggestions.append("- **sklearn:** LogisticRegression أو RandomForest أو GradientBoosting.")
        suggestions.append("- **PyTorch MLP** — إن كانت الميزات كثيفة/غير خطية.")
    if any(k in d for k in ("انحدار", "regress", "توقع", "سعر", "رقم")):
        suggestions.append("- **sklearn:** Ridge / LinearRegression.")
        suggestions.append("- **PyTorch MLP** للانحدار غير الخطي.")
    if any(k in d for k in ("صورة", "vision", "صوت", "audio", "video")):
        suggestions.append(
            "- يحتاج نموذجاً متخصصاً (CNN/Transformer بصري أو صوتي). "
            "في NSM يوجد مكوّنات صوت/فيديو اختيارية عبر torch — حدّد المهمة بدقة أكثر."
        )
    if any(k in d for k in ("وكيل", "agent", "rl", "تعزيز")):
        suggestions.append("- خارج النطاق الافتراضي الحالي؛ يمكن تخطيط الدورة ثم ربط بيئة RL لاحقاً.")

    if not suggestions:
        suggestions.append("- ابدأ بـ **جرد البيئة** ثم **درّب تصنيف تجريبي** أو **درّب شبكة torch** للتحقق من المسار.")
        suggestions.append("- إن كانت البيانات جدولية: sklearn أولاً (سريع وقابل للتفسير).")
        suggestions.append("- إن كانت معرفة NSM عربية: مسارات CKG / NeuralCore.")

    lines.extend(suggestions)
    lines.append("")
    lines.append("أوامر تنفيذ سريعة: `درّب تصنيف تجريبي` · `درّب انحدار تجريبي` · `درّب شبكة torch` · `حالة ckg`")
    return "\n".join(lines)


def list_saved_models() -> str:
    lines = ["## 💾 نماذج محفوظة بواسطة الوكيل", ""]
    if not ARTIFACTS.is_dir():
        return "\n".join(lines + ["لا يوجد مجلد مخرجات بعد."])
    files = sorted(ARTIFACTS.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        lines.append("لا توجد ملفات بعد. درّب نموذجاً أولاً.")
        return "\n".join(lines)
    for p in files[:30]:
        try:
            mb = p.stat().st_size / (1024 * 1024)
            lines.append(f"- `{p.relative_to(ROOT)}` ({mb:.3f} MB)")
        except Exception:
            lines.append(f"- `{p.name}`")
    return "\n".join(lines)


def training_dashboard() -> str:
    """لوحة تحكم موحّدة لوكيل تدريب النماذج — نظرة عامة + توصيات فورية."""
    lines = [
        "## 🎛️ لوحة تحكم تدريب النماذج",
        "",
        "أنا **مدير تدريب النماذج** داخل NSM: أراقب البيئة، أقترح الخطوة التالية،",
        "وأشغّل أدوات حقيقية (بدون اختلاق مقاييس).",
        "",
    ]

    # موارد سريعة
    ram = _ram_gb()
    lines.append("### 📡 الموارد")
    lines.append(f"- رام متاحة ≈ **{ram:.2f} GB**")
    try:
        gpu_txt = _gpu_report()
        first = (gpu_txt or "").splitlines()[0] if gpu_txt else "GPU: غير معروف"
        lines.append(f"- {first}")
    except Exception:
        lines.append("- GPU: تعذّر التقرير")
    lines.append(f"- scikit-learn: {'✅' if _SKLEARN_OK else '❌'}")
    lines.append(f"- PyTorch: {'✅' if _TORCH_OK else '❌'}")
    lines.append("")

    # آخر مهام التدريب الموحّدة
    lines.append("### 📜 آخر مهام التدريب")
    try:
        runs = _read_training_runs(8)
        if not runs:
            lines.append("- لا سجل بعد. استخدم: `مهمة تدريب data/samples/classification_demo.csv الهدف=label`")
        else:
            for r in reversed(runs[-5:]):
                st = r.get("status", "?")
                plan = r.get("plan") or {}
                lines.append(
                    f"- `{r.get('id','?')}` · **{st}** · {plan.get('path') or r.get('path')} "
                    f"({plan.get('task','?')}/{plan.get('engine','?')})"
                )
        lines.append("- أوامر: `سجل مهام التدريب` · `مهمة تدريب <csv> الهدف=label` · أضف `نفّذ` للتشغيل")
    except Exception as e:
        lines.append(f"- تعذّر قراءة السجل: {e}")
    lines.append("")

    # CKG
    lines.append("### 🧬 حالة CKG / ArabicTransformer")
    try:
        st = _load_json(STATE_V3) if STATE_V3.is_file() else None
        if st:
            epoch = st.get("epoch") or st.get("global_step") or st.get("steps") or "?"
            loss = st.get("last_loss") or st.get("loss") or st.get("avg_loss")
            packs = st.get("packs_done") or st.get("packs") or "?"
            lines.append(f"- حالة محفوظة: ✅ | خطوة/عصر: **{epoch}** | حزم: **{packs}**")
            if loss is not None:
                lines.append(f"- آخر خسارة مسجّلة: **{loss}**")
        else:
            lines.append("- لا يوجد `ckg_train_state_v3.json` بعد.")
    except Exception as e:
        lines.append(f"- تعذّر قراءة حالة CKG: {e}")
    lines.append(f"- سكربت v3: {'✅' if TRAIN_V3.is_file() else '❌'}")
    lines.append("")

    # Hierarchical MoE
    lines.append("### 🧩 Hierarchical MoE")
    try:
        from ai.moe_ckg_bridge import get_moe_bridge
        br = get_moe_bridge()
        if br.available:
            m = br.moe
            lines.append(
                f"- جاهز ✅ | فئات **{len(m._group_order)}** | خبراء **{m.total_experts()}**"
            )
            lines.append(
                f"- best: temp={getattr(m,'router_temperature','?')} · "
                f"shared={getattr(m,'shared_coeff','?')} · residual={getattr(m,'input_residual','?')}"
            )
            sample = br.classify("ما حكم الصلاة؟")
            lines.append(f"- عينة: `{sample.get('top')}` (ثقة {sample.get('confidence')})")
        else:
            lines.append(f"- غير متاح: {br._load_error}")
    except Exception as e:
        lines.append(f"- تعذّر فحص MoE: {e}")
    lines.append("- أوامر: `صحة moe` · `صنّف: ...` · `إحصاء moe` · `ملخص moe`")
    lines.append("")

    # نماذج محفوظة (مختصر)
    lines.append("### 💾 آخر النماذج المحفوظة")
    try:
        if ARTIFACTS.is_dir():
            files = sorted(
                [p for p in ARTIFACTS.rglob("*") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:5]
            if files:
                for p in files:
                    mb = p.stat().st_size / (1024 * 1024)
                    rel = p.relative_to(ROOT) if p.is_relative_to(ROOT) else p.name
                    lines.append(f"- `{rel}` ({mb:.2f} MB)")
            else:
                lines.append("- لا توجد مخرجات بعد.")
        else:
            lines.append("- مجلد المخرجات غير موجود بعد.")
    except Exception:
        lines.append("- تعذّر سرد المخرجات.")
    lines.append("")

    # تدريب مستمر
    lines.append("### 🔄 التدريب الذاتي المستمر")
    try:
        ct_log = ROOT / "artifacts" / "model_training" / "continuous" / "training_triggers.jsonl"
        if ct_log.is_file():
            raw = ct_log.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            lines.append(f"- سجل الأحداث: **{len(raw)}** حدث")
            if raw:
                try:
                    last = json.loads(raw[-1])
                    q = (last.get("quality") or {}).get("score")
                    action = (last.get("plan") or {}).get("action")
                    lines.append(f"- آخر تقييم جودة: **{q}** | إجراء: **{action}**")
                except Exception:
                    lines.append("- آخر حدث موجود (تعذّر تفصيله).")
        else:
            lines.append("- لم يُسجَّل أي دورة تدريب مستمر بعد. جرّب: `تدريب مستمر`")
    except Exception as e:
        lines.append(f"- {e}")
    lines.append("")

    # توصيات ذكية
    lines.append("### 🎯 الخطوة التالية المقترحة")
    lines.append(smart_train_next(recommend_only=True))
    lines.append("")
    lines.append("---")
    lines.append(
        "**أوامر سريعة:** `جرد` · `حالة ckg` · `شغّل تدريب ckg تجريبي` · "
        "`درّب شبكة torch` · `درّب تصنيف تجريبي` · `نماذج محفوظة` · `تدريب مستمر` · `خطة`"
    )
    return "\n".join(lines)


def smart_train_next(recommend_only: bool = False) -> str:
    """يختار بأمان أفضل خطوة تدريب تالية حسب حالة البيئة.

    إن recommend_only=True يعيد توصية نصية فقط بدون تشغيل ثقيل.
    """
    ram = _ram_gb()

    has_v3_script = TRAIN_V3.is_file()
    has_sentences = any(
        p.is_file() for p in (SENTENCES_V3, SENTENCES_V2, SENTENCES_V1, GENERAL_AR)
    )
    st = _load_json(STATE_V3) if STATE_V3.is_file() else None

    if has_v3_script and has_sentences and ram >= 1.2:
        if st is None:
            if not recommend_only:
                return run_ckg_step(packs=1, dry_run=False)
            return (
                "① **شغّل تدريب ckg تجريبي** (حزمة واحدة) للتحقق من المسار.\n"
                "② بعدها راقب بـ `حالة ckg` و`خسارة`.\n"
                "③ عند الاستقرار زد الحزم تدريجياً."
            )
        return (
            "① راجع `حالة ckg` و`خسارة`.\n"
            "② إن كان الاتجاه جيداً: `شغّل تدريب ckg` بحزمة أو حزمتين.\n"
            "③ أو جرّب `درّب شبكة torch` / `درّب تصنيف تجريبي` للتحقق من مسارات أخرى."
        )

    if _SKLEARN_OK and ram >= 0.8:
        return (
            "① ابدأ بـ **درّب تصنيف تجريبي** للتحقق السريع من مسار sklearn.\n"
            "② إن كان لديك CSV: `قائمة csv` ثم `درّب من csv ...`\n"
            "③ لشبكات أعقد: ثبّت torch ثم `درّب شبكة torch`."
        )

    if _TORCH_OK and ram >= 1.0:
        return (
            "① **درّب شبكة torch** (MLP تجريبي) للتحقق من المسار.\n"
            "② راقب الرام — قلّل الحقب إن لزم.\n"
            "③ احفظ النتائج تحت `artifacts/model_training/`."
        )

    return (
        "① نفّذ **جرد** لرؤية المكتبات والبيانات المتاحة.\n"
        "② ثبّت scikit-learn أو torch حسب الحاجة.\n"
        "③ إن كانت رام منخفضة (<1 GB) استخدم `dry-run` أو Colab (`colab`)."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5b) تدريب على CSV حقيقي + رفع ملفات + CNN + نص
# ═══════════════════════════════════════════════════════════════════════════

UPLOAD_DIR = ARTIFACTS / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_CSV_SEARCH_ROOTS = (
    ROOT / "data",
    ROOT / "knowledge_sources",
    ARTIFACTS,
    ROOT / "artifacts",
)


def _find_csv_files(limit: int = 40) -> List[Path]:
    found: List[Path] = []
    seen = set()
    for root in _CSV_SEARCH_ROOTS:
        if not root.is_dir():
            continue
        try:
            for p in root.rglob("*.csv"):
                if not p.is_file():
                    continue
                key = str(p.resolve())
                if key in seen:
                    continue
                seen.add(key)
                found.append(p)
                if len(found) >= limit:
                    return found
        except Exception:
            continue
    return found


def list_csv_datasets() -> str:
    files = _find_csv_files()
    lines = ["## 📄 ملفات CSV المتاحة للتدريب", ""]
    if not files:
        lines.append("لا توجد ملفات CSV تحت data/ أو artifacts/.")
        lines.append("ارفع ملفاً من واجهة الوكيل أو ضع CSV في `data/samples/`.")
        return "\n".join(lines)
    for p in files:
        try:
            mb = p.stat().st_size / (1024 * 1024)
            rel = p.relative_to(ROOT)
            # peek header
            header = ""
            with open(p, encoding="utf-8", errors="replace") as f:
                header = f.readline().strip()[:120]
            lines.append(f"- `{rel}` ({mb:.2f} MB) — أعمدة: `{header}`")
        except Exception as e:
            lines.append(f"- `{p.name}` (خطأ قراءة: {e})")
    lines.append("")
    lines.append("للتدريب: `درّب على csv data/samples/classification_demo.csv`")
    lines.append("أو: `درّب على csv <المسار> الهدف=label`")
    return "\n".join(lines)


def _load_csv_table(path: Path, max_rows: int = 5000):
    """يحمّل CSV إلى قائمة صفوف + رؤوس. بدون pandas."""
    import csv as _csv

    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        first_line = f.readline()
        if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
            raise ValueError(
                f"الملف `{path.name}` هو مؤشر Git LFS (pointer) وليس البيانات الفعلية — "
                "لم يتم سحب محتوى LFS في هذه البيئة. شغّل `git lfs pull` محلياً، "
                "أو استثنِ هذا الملف من LFS في .gitattributes إن كان صغيراً بما يكفي "
                "ليعمل كملف Git عادي (كما تم مع بيانات data/samples/)."
            )
        f.seek(0)
        reader = _csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("ملف فارغ")
    header = [h.strip() for h in rows[0]]
    data = rows[1 : max_rows + 1]
    return header, data


def _infer_target_and_matrix(header: List[str], data: List[List[str]], target_col: Optional[str] = None):
    """يستنتج عمود الهدف ويبني X (float) و y."""
    if not data:
        raise ValueError("لا توجد صفوف بيانات")

    col_idx = {h: i for i, h in enumerate(header)}
    if target_col and target_col in col_idx:
        t_i = col_idx[target_col]
    else:
        # تفضيل أسماء شائعة ثم آخر عمود
        for cand in ("label", "target", "y", "class", "الهدف", "التصنيف", "label_id"):
            if cand in col_idx:
                t_i = col_idx[cand]
                break
        else:
            t_i = len(header) - 1

    # أعمدة نصية vs رقمية
    text_cols: List[int] = []
    num_cols: List[int] = []
    for i, h in enumerate(header):
        if i == t_i:
            continue
        # عيّنة
        sample_vals = [r[i] for r in data[:30] if i < len(r)]
        numeric_ok = 0
        for v in sample_vals:
            try:
                float(v.replace(",", ""))
                numeric_ok += 1
            except Exception:
                pass
        if sample_vals and numeric_ok / max(1, len(sample_vals)) >= 0.7:
            num_cols.append(i)
        else:
            text_cols.append(i)

    y_raw = [r[t_i] if t_i < len(r) else "" for r in data]

    # هدف رقمي؟
    y_num = []
    y_num_ok = True
    for v in y_raw:
        try:
            y_num.append(float(str(v).replace(",", "")))
        except Exception:
            y_num_ok = False
            break

    # تصنيف إذا قيم فريدة قليلة
    unique = list(dict.fromkeys(y_raw))
    is_classification = (not y_num_ok) or (len(unique) <= max(15, int(0.05 * len(y_raw)) + 1))

    if is_classification:
        label_to_id = {lab: i for i, lab in enumerate(unique)}
        y = np.array([label_to_id[v] for v in y_raw], dtype=np.int64)
        task = "classification"
        label_map = {i: lab for lab, i in label_to_id.items()}
    else:
        y = np.array(y_num, dtype=np.float64)
        task = "regression"
        label_map = None

    if num_cols:
        X_list = []
        for r in data:
            row = []
            for i in num_cols:
                try:
                    row.append(float(str(r[i]).replace(",", "")))
                except Exception:
                    row.append(0.0)
            X_list.append(row)
        X = np.array(X_list, dtype=np.float64)
        feature_mode = "numeric"
        feature_names = [header[i] for i in num_cols]
        texts = None
    elif text_cols:
        # عمود نص أساسي = أطول/أول نصي
        ti = text_cols[0]
        texts = [r[ti] if ti < len(r) else "" for r in data]
        X = None
        feature_mode = "text"
        feature_names = [header[ti]]
    else:
        raise ValueError("لم يُعثر على أعمدة ميزات صالحة")

    return {
        "X": X,
        "y": y,
        "task": task,
        "feature_mode": feature_mode,
        "feature_names": feature_names,
        "texts": texts,
        "target_name": header[t_i],
        "label_map": label_map,
        "n_samples": len(data),
    }


def _text_to_bow(texts: List[str], vocab_size: int = 512) -> np.ndarray:
    """Bag-of-hashed-words بسيط (بدون مكتبات إضافية)."""
    mat = np.zeros((len(texts), vocab_size), dtype=np.float64)
    for i, t in enumerate(texts):
        for tok in str(t).lower().replace("،", " ").replace(".", " ").split():
            if len(tok) < 2:
                continue
            h = 2166136261
            for ch in tok:
                h ^= ord(ch)
                h = (h * 16777619) & 0xFFFFFFFF
            mat[i, h % vocab_size] += 1.0
        s = mat[i].sum()
        if s > 0:
            mat[i] /= s
    return mat



# ═══════════════════════════════════════════════════════════════════════════
# 3b) دورة تدريب موحّدة — معاينة / تنفيذ / سجل مهام
# ═══════════════════════════════════════════════════════════════════════════

TRAINING_RUNS_LOG = ARTIFACTS / "training_runs.jsonl"
CHECKPOINTS_DIR = ARTIFACTS / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
_ALLOWED_DATA_ROOTS = (
    ROOT / "data",
    ROOT / "artifacts",
    ROOT / "knowledge_sources",
    ARTIFACTS,
)


def _safe_resolve_data_path(path_str: str) -> Path:
    """يحل المسار ويرفض أي ملف خارج مجلدات المشروع المسموحة."""
    raw = (path_str or "").strip().strip("`\"'")
    if not raw:
        raise ValueError("مسار فارغ")
    if ".." in Path(raw).parts:
        raise PermissionError("مسارات تحتوي .. غير مسموحة")
    candidates = []
    p = Path(raw)
    if p.is_file():
        candidates.append(p.resolve())
    candidates.append((ROOT / raw).resolve())
    # بحث بالاسم تحت data/
    name = Path(raw).name
    for root in _ALLOWED_DATA_ROOTS:
        if root.is_dir():
            for hit in root.rglob(name):
                if hit.is_file():
                    candidates.append(hit.resolve())
                    break
    allowed = [r.resolve() for r in _ALLOWED_DATA_ROOTS if r.exists()]
    for c in candidates:
        try:
            if not c.is_file():
                continue
            ok = any(str(c).startswith(str(a) + sep) or c == a for a in allowed for sep in ("/",))
            # أيضاً أي مسار تحت ROOT
            if str(c).startswith(str(ROOT.resolve()) + "/"):
                # رفض المسارات الحساسة
                rel = c.relative_to(ROOT.resolve())
                if rel.parts and rel.parts[0] in {".git", ".env", "secrets"}:
                    raise PermissionError(f"مسار محظور: {rel}")
                return c
        except PermissionError:
            raise
        except Exception:
            continue
    raise FileNotFoundError(f"لم يُعثر على ملف آمن داخل المشروع: {path_str}")


def _append_training_run(row: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_RUNS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_training_runs(limit: int = 30) -> List[dict]:
    if not TRAINING_RUNS_LOG.is_file():
        return []
    rows = []
    with open(TRAINING_RUNS_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows[-limit:]


# ── Checkpoints / استئناف بعد الفشل / rollback ──────────────────────────────

def _checkpoint_dir(run_id: str) -> Path:
    d = CHECKPOINTS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_checkpoint(
    run_id: str,
    epoch: int,
    model,
    optimizer,
    val_loss: Optional[float],
    best_val_loss: float,
    history: List[float],
    val_history: List[float],
    task: str,
    n_features: int,
    batch_size: int,
    is_best: bool = False,
) -> None:
    """يحفظ حالة التدريب الحالية (latest) وأفضل نسخة (best) بشكل منفصل."""
    import torch

    d = _checkpoint_dir(run_id)
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
        "best_val_loss": float(best_val_loss),
        "history": list(history),
        "val_history": list(val_history),
        "task": task,
        "n_features": int(n_features),
        "batch_size": int(batch_size),
        "ts": time.time(),
    }
    torch.save(payload, d / "latest.pt")
    if is_best:
        torch.save(payload, d / "best.pt")
    meta = {k: v for k, v in payload.items() if k not in ("model_state", "optimizer_state")}
    try:
        with open(d / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, default=str)
    except Exception:
        pass

    # مزامنة اختيارية مع تخزين خارجي (best-effort، لا تؤثر على التدريب المحلي)
    files_to_sync = [d / "latest.pt"]
    if is_best:
        files_to_sync.append(d / "best.pt")
    try:
        _ckpt_sync_after_save(run_id, files_to_sync)
    except Exception as e:
        logger.warning(f"sync خارجي فشل (يُهمَل، التدريب المحلي سليم): {e}")


def _load_checkpoint(run_id: str, which: str = "latest") -> Optional[dict]:
    import torch

    d = _checkpoint_dir(run_id)
    p = d / f"{which}.pt"
    if not p.is_file():
        # قد يكون الملف المحلي مفقوداً لأن القرص لا يبقى بين الجلسات
        # (مثلاً على Streamlit Community Cloud) — نحاول استرجاعه من
        # التخزين الخارجي إن كان مفعّلاً قبل التسليم بعدم وجوده.
        try:
            _ckpt_restore_if_missing(run_id, d, which)
        except Exception as e:
            logger.warning(f"restore خارجي فشل (يُهمَل): {e}")
    if not p.is_file():
        return None
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        # إصدارات torch أقدم لا تدعم weights_only
        return torch.load(p, map_location="cpu")
    except Exception:
        return None


def find_resumable_run(path_str: Optional[str] = None, target_col: Optional[str] = None) -> Optional[str]:
    """يبحث عن آخر مهمة (failed أو غير مكتملة) عندها checkpoint قابل للاستئناف."""
    rows = _read_training_runs(limit=200)
    for r in reversed(rows):
        if path_str and r.get("path") != path_str:
            continue
        if target_col and r.get("target") != target_col:
            continue
        run_id = r.get("id")
        if not run_id:
            continue
        if (_checkpoint_dir(run_id) / "latest.pt").is_file():
            return run_id
    return None


def resume_training_mission(
    path_str: Optional[str] = None,
    target_col: Optional[str] = None,
    epochs: int = 15,
    run_id: Optional[str] = None,
) -> str:
    """
    يستأنف آخر مهمة تدريب فاشلة/متوقفة من آخر checkpoint محفوظ،
    بدل إعادة التدريب من الصفر.
    """
    if not run_id:
        run_id = find_resumable_run(path_str=path_str, target_col=target_col)
    if not run_id:
        return (
            "## ⚠️ لا يوجد ما يُستأنف\n\n"
            "ما لقيت أي مهمة تدريب سابقة عندها checkpoint محفوظ "
            "(الاستئناف مدعوم حالياً لمسار Torch MLP).\n"
            "جرّب: `مهمة تدريب <ملف.csv> الهدف=<العمود> نفّذ` أولاً."
        )
    rows = _read_training_runs(limit=200)
    src = next((r for r in reversed(rows) if r.get("id") == run_id), None)
    if src is None:
        return f"## ⚠️ لم يُعثر على سجل للمهمة `{run_id}` رغم وجود checkpoint."

    ck_meta = _load_checkpoint(run_id, "latest") or {}
    prev_epoch = int(ck_meta.get("epoch", 0))

    resumed = dict(src)
    resumed["status"] = "running"
    resumed["ts"] = time.time()
    resumed["resumed_from_epoch"] = prev_epoch
    _append_training_run(resumed)

    path_used = src.get("path")
    target_used = src.get("target")
    plan = src.get("plan") or {}
    prefer = plan.get("engine") if plan.get("engine") in ("torch",) else "torch"

    try:
        path = _safe_resolve_data_path(path_used)
        header, data = _load_csv_table(path)
        bundle = _infer_target_and_matrix(header, data, target_col=target_used)
        if bundle["feature_mode"] == "text":
            raise ValueError("الاستئناف غير مدعوم بعد لبيانات نصية")
        X = bundle["X"]
        mu = X.mean(axis=0)
        sig = X.std(axis=0) + 1e-8
        X = (X - mu) / sig
        y = bundle["y"]
        result = train_torch_on_arrays(
            X, y, bundle["task"], epochs=epochs, run_id=run_id, resume=True
        )
        done = dict(resumed)
        done.update({"status": "completed", "ts": time.time(), "result_preview": (result or "")[:500]})
        _append_training_run(done)
        return (
            f"## ▶️ استئناف مهمة `{run_id}` من الحقبة {prev_epoch}\n\n"
            f"- الملف: `{path_used}` | الهدف: `{target_used}`\n\n---\n{result}"
        )
    except Exception as e:
        fail = dict(resumed)
        fail.update({"status": "failed", "ts": time.time(), "error": f"{type(e).__name__}: {e}"})
        _append_training_run(fail)
        return f"## ❌ فشل الاستئناف\n\n{type(e).__name__}: {e}"


def rollback_to_best(run_id: Optional[str] = None) -> str:
    """
    يستعيد أفضل نسخة محفوظة (أقل val_loss) من آخر مهمة تدريب،
    ويجعلها النموذج النهائي المحفوظ في قائمة النماذج المحفوظة.
    """
    import torch

    if not run_id:
        rows = _read_training_runs(limit=200)
        for r in reversed(rows):
            rid = r.get("id")
            if rid and (_checkpoint_dir(rid) / "best.pt").is_file():
                run_id = rid
                break
    if not run_id:
        return "## ⚠️ لا توجد نسخة `best` محفوظة للرجوع إليها بعد."

    best = _load_checkpoint(run_id, "best")
    if best is None:
        return f"## ⚠️ لا توجد نسخة `best` محفوظة للمهمة `{run_id}`."

    outp = ARTIFACTS / f"rollback_best_{run_id}_{int(time.time())}.pt"
    torch.save(best, outp)
    return (
        f"## ⏪ تم الرجوع لأفضل نسخة من المهمة `{run_id}`\n\n"
        f"- أفضل val_loss: **{best.get('best_val_loss')}** عند الحقبة {best.get('epoch')}\n"
        f"- محفوظة الآن كنموذج نهائي: `{outp.relative_to(ROOT)}`"
    )


def inspect_training_data(path_str: str, target_col: Optional[str] = None) -> dict:
    """فحص البيانات والهدف ونوع المهمة + اختيار محرك مقترح."""
    path = _safe_resolve_data_path(path_str)
    header, data = _load_csv_table(path)
    bundle = _infer_target_and_matrix(header, data, target_col=target_col)
    task = bundle["task"]
    n_samples = int(bundle["n_samples"])
    if bundle["feature_mode"] == "text":
        n_features = 512  # BoW لاحقاً
    else:
        n_features = int(bundle["X"].shape[1]) if bundle["X"] is not None else 0
    n_classes = None
    if task == "classification":
        n_classes = int(len(set(bundle["y"].tolist()))) if hasattr(bundle["y"], "tolist") else None
    # اختيار محرك تلقائي
    if bundle["feature_mode"] == "text":
        engine = "torch" if _TORCH_OK else "sklearn"
    elif n_samples < 5000 and n_features < 200 and _SKLEARN_OK:
        engine = "sklearn"
    elif _TORCH_OK:
        engine = "torch"
    elif _SKLEARN_OK:
        engine = "sklearn"
    else:
        engine = "none"
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "abs_path": str(path),
        "target": bundle["target_name"],
        "task": task,
        "feature_mode": bundle["feature_mode"],
        "feature_names": bundle["feature_names"],
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "engine": engine,
        "sklearn_ok": _SKLEARN_OK,
        "torch_ok": _TORCH_OK,
        "bundle": bundle,
    }


def format_training_plan(info: dict, execute: bool = False) -> str:
    lines = [
        "## 📋 خطة مهمة تدريب" + (" (تنفيذ)" if execute else " (معاينة — Dry-run)"),
        "",
        f"- الملف: `{info['path']}`",
        f"- الهدف: `{info['target']}`",
        f"- نوع المهمة: **{info['task']}**",
        f"- العينات: **{info['n_samples']}** · الميزات: **{info['n_features']}**",
    ]
    if info.get("n_classes") is not None:
        lines.append(f"- عدد الفئات: **{info['n_classes']}**")
    lines.append(f"- نمط الميزات: **{info['feature_mode']}** → {info['feature_names'][:8]}")
    lines.append(f"- المحرك المقترح: **{info['engine']}** (sklearn={'✅' if info['sklearn_ok'] else '❌'} · torch={'✅' if info['torch_ok'] else '❌'})")
    lines.append("")
    if not execute:
        lines.append("> هذه **معاينة آمنة**. لتشغيل التدريب فعلياً أضف: `نفّذ` أو `execute`.")
        lines.append(">")
        lines.append(f"> مثال: `مهمة تدريب {info['path']} الهدف={info['target']} نفّذ`")
    return "\n".join(lines)


def run_training_mission(
    path_str: str,
    target_col: Optional[str] = None,
    epochs: int = 30,
    prefer: str = "auto",
    execute: bool = False,
) -> str:
    """
    دورة موحّدة: فحص → خطة → (اختياري) تنفيذ مع سجل حالات
    planned | running | completed | failed | rejected

    مع منهجية NSM الموروثة: تسجّل المهمة وخطواتها (plan/inspect/execute/
    verify) في ذاكرة المنهجية حتى تظهر في لوحة المراقبة (قسم المنهجية)
    وتتعلّم منها الدروس. إن غابت الوحدة تعمل كالمعتاد تمامًا.
    """
    run_id = f"run_{int(time.time())}"
    base = {
        "id": run_id,
        "ts": time.time(),
        "path": path_str,
        "target": target_col,
        "execute": bool(execute),
    }
    try:
        info = inspect_training_data(path_str, target_col=target_col)
    except PermissionError as e:
        base.update({"status": "rejected", "error": str(e)})
        _append_training_run(base)
        return f"## 🚫 مرفوض\n\n{e}"
    except FileNotFoundError as e:
        base.update({"status": "rejected", "error": str(e)})
        _append_training_run(base)
        return f"## 🚫 مرفوض\n\n{e}"
    except Exception as e:
        base.update({"status": "failed", "error": f"{type(e).__name__}: {e}"})
        _append_training_run(base)
        return f"## ❌ فشل الفحص\n\n{type(e).__name__}: {e}"

    base["status"] = "planned"
    base["plan"] = {
        "task": info["task"],
        "n_samples": info["n_samples"],
        "n_features": info["n_features"],
        "n_classes": info.get("n_classes"),
        "engine": info["engine"],
        "path": info["path"],
        "target": info["target"],
    }
    _append_training_run(base)

    # منهجية الوالد: تسجيل المهمة وخطة خطواتها في ذاكرة المنهجية الموروثة
    try:
        _meth_task_started(
            run_id,
            request=(f"تدريب نموذج: {path_str}" +
                     (f" الهدف={target_col}" if target_col else "")),
            plan=[
                {"step": "inspect", "desc": "فحص البيانات واختيار المحرك"},
                {"step": "plan", "desc": "بناء خطة التدريب (epochs, engine)"},
                {"step": "execute", "desc": "تنفيذ التدريب (kernel أو مباشر)"},
                {"step": "verify", "desc": "التحقق من النتيجة وحفظ checkpoint"},
                {"step": "reflect", "desc": "تسجيل النتيجة/الدرس المنهجي"},
            ],
        )
        _meth_step(run_id, step_type="inspect", ok=True,
                   note=(f"البيانات: {info['n_samples']} عينة · "
                         f"{info['n_features']} ميزة · "
                         f"المهمة={info['task']} · المحرك={info['engine']}"))
    except Exception:
        pass

    plan_txt = format_training_plan(info, execute=execute)
    if not execute:
        try:
            _meth_step(run_id, step_type="plan", ok=True,
                       note="خطة معاينة آمنة — لا تنفيذ")
            _meth_task_finished(run_id, status="done", ok=True,
                                result_summary="خطة تدريب معاينة بدون تنفيذ")
        except Exception:
            pass
        return plan_txt

    # تنفيذ فعلي
    running = dict(base)
    running["status"] = "running"
    running["ts"] = time.time()
    _append_training_run(running)

    eng = prefer if prefer not in ("auto", "") else info["engine"]
    if eng == "none":
        fail = dict(base)
        fail.update({"status": "failed", "error": "لا يتوفر sklearn ولا torch"})
        _append_training_run(fail)
        return plan_txt + "\n\n❌ لا يتوفر محرك تدريب."

    try:
        result = train_from_csv(
            info["abs_path"],
            target_col=info["target"],
            epochs=epochs,
            prefer=eng if eng in ("sklearn", "torch", "text", "cnn") else "auto",
            run_id=run_id,
        )
        done = dict(base)
        has_ckpt = (_checkpoint_dir(run_id) / "latest.pt").is_file()
        done.update({
            "status": "completed",
            "ts": time.time(),
            "engine": eng,
            "result_preview": (result or "")[:500],
            "has_checkpoint": has_ckpt,
        })
        _append_training_run(done)

        # منهجية الوالد: خطوة تحقق وتسجيل النتيجة في ذاكرة المنهجية
        try:
            _meth_step(run_id, step_type="execute", ok=True,
                       note=f"التدريب عبر {eng} اكتمل")
            _meth_step(run_id, step_type="verify", ok=bool(has_ckpt),
                       note=("checkpoint محفوظ" if has_ckpt else "بلا checkpoint"))
            _meth_task_finished(
                run_id, status="done", ok=True,
                result_summary=(result or "")[:400])
        except Exception:
            pass
        return plan_txt + "\n\n---\n" + result
    except Exception as e:
        err_txt = f"{type(e).__name__}: {e}"
        fail = dict(base)
        has_ckpt = (_checkpoint_dir(run_id) / "latest.pt").is_file()
        fail.update({
            "status": "failed",
            "ts": time.time(),
            "error": err_txt,
            "has_checkpoint": has_ckpt,
        })
        _append_training_run(fail)

        # منهجية الوالد: فشل تنفيذ → خطوة reflection + درس تلقائي
        try:
            _meth_step(run_id, step_type="verify", ok=False,
                       note=f"فشل التنفيذ: {err_txt}")
            _meth_task_finished(run_id, status="failed", ok=False,
                                result_summary=err_txt[:400])
            if "Memory" in err_txt or "CUDA" in err_txt or "OOM" in err_txt:
                _record_kernel_oom_lesson(context=err_txt[:400])
        except Exception:
            pass
        note = (
            f"\n\n> 💾 يوجد checkpoint محفوظ لهذه المهمة (`{run_id}`) — "
            f"استخدم **استأنف التدريب** بدل البدء من الصفر."
            if has_ckpt else ""
        )
        return plan_txt + f"\n\n## ❌ فشل التنفيذ\n\n{type(e).__name__}: {e}" + note


def run_kaggle_training(
    preset: str = "small",
    n: Optional[int] = None,
    epochs: int = 15,
    batch: int = 16,
    fresh: bool = True,
    auto_push: bool = True,
) -> str:
    """
    تشغيل تدريب SurahChain على GPU Kaggle عبر Kaggle API — بنفس منهجية NSM:

      جهّز (generate kernel script + metadata)
      → ادفع (`kaggle kernels push`)
      → راقب حتى RUNNING ثم اكتمال (COMPLETE/FAILED/ERROR)
      → اقرأ Logs عند الانتهاء
      → إن فشل: درس منهجي تلقائي (OOM/ذاكرة) + تقرير الإصلاح المقترح
      → AUTO_PUSH=1 بعد نجاح التدريب (يرفع checkpoint لـ GitHub)

    الافتراضي: preset small / d_model=128 (كما طلب المستخدم).
    لا يتطلب مفاتيح في الكود: Kaggle keys من Streamlit Secrets
    وGITHUB_TOKEN من Kaggle Secrets.
    """
    run_id = f"kag_{int(time.time())}"
    base = {"id": run_id, "ts": time.time(), "type": "kaggle", "preset": preset}
    _KAGGLE_CFGS = {
        "small": {"preset": "small", "d_model": 128, "n": 30000, "batch": 16},
        "medium": {"preset": "medium", "d_model": 256, "n": 60000, "batch": 24},
        "large": {"preset": "large", "d_model": 512, "n": 100000, "batch": 8},
        "smoke": {"preset": "small", "d_model": 128, "n": 2000, "batch": 8},
    }
    cfg = _KAGGLE_CFGS.get((preset or "").lower())
    if cfg is None:
        return "## ❌ preset غير معروف — استخدم small/medium/large/smoke"
    n = n or cfg["n"]
    batch = batch or cfg["batch"]
    d_model = cfg["d_model"]

    lines = [
        "## 🟠 دورة تدريب Kaggle (GPU T4) — SurahChain",
        f"- preset=**{cfg['preset']}** · d_model=**{d_model}** · N=**{n}** · epochs=**{epochs}** · batch=**{batch}**",
        f"- AUTO_PUSH={'مفعّل ✅' if auto_push else 'معطّل ❌'} · fresh={'نعم' if fresh else 'استكمال'}",
        "",
    ]

    try:
        from ai.kaggle_provider import (
            ensure_kaggle_env, start_surahchain_training_api,
            status_kaggle_kernel, download_kaggle_output,
            _kaggle_cli_available, _kaggle_py_available,
        )
    except Exception as e:
        return "## ❌ وحدة kaggle_provider غير متاحة\n\n" + str(e)

    ok_cred, msg = ensure_kaggle_env()
    cli_ok = _kaggle_cli_available() or _kaggle_py_available()
    base["checks"] = {"creds": ok_cred, "creds_msg": msg, "cli": cli_ok}
    if not ok_cred or not cli_ok:
        base.update({"status": "rejected", "error": msg})
        _append_training_run(base)
        lines.append("### 🚫 الجاهزية")
        lines.append(f"- بيانات Kaggle (Streamlit Secrets): {'✅' if ok_cred else '❌ ' + (msg or '')}")
        lines.append(f"- Kaggle CLI: {'✅' if cli_ok else '❌ '}")
        lines.append("> ضع KAGGLE_USERNAME + KAGGLE_KEY في Streamlit Secrets وGITHUB_TOKEN في Kaggle Secrets ثم أعد المحاولة.")
        return "\n".join(lines)

    base["status"] = "running"
    try:
        _meth_task_started(
            run_id,
            request=f"تدريب Kaggle SurahChain preset={cfg['preset']} d={d_model} N={n} epochs={epochs}",
            plan={"preset": cfg["preset"], "d_model": d_model, "n": n,
                  "epochs": epochs, "batch": batch, "fresh": fresh,
                  "auto_push": auto_push, "gpu": "T4"},
        )
    except Exception:
        pass

    # ── 1) جهّز + ادفع kernel ──
    _meth_step(run_id, step_type="plan", ok=True,
               note=f"خطة Kaggle: clone repo → train_pretrain_torch ({cfg['preset']}) → AUTO_PUSH")
    launch = start_surahchain_training_api(
        preset=str(cfg["preset"]), n=int(n), epochs=int(epochs),
        batch=int(batch), fresh=bool(fresh), auto_push=bool(auto_push),
    )
    if not launch.get("ok"):
        base.update({"status": "failed", "error": launch.get("error")})
        _append_training_run(base)
        try:
            _meth_task_finished(run_id, status="failed", ok=False,
                                result_summary=(launch.get("error") or "")[:400])
        except Exception:
            pass
        lines.append("### ❌ فشل الدفع")
        lines.append("```")
        lines.append(json.dumps(launch, ensure_ascii=False, indent=2))
        lines.append("```")
        return "\n".join(lines)

    job_id = launch.get("job_id") or run_id
    slug = (launch.get("push") or {}).get("kernel_slug") or launch.get("kernel_url") or ""
    base["job_id"] = job_id
    base["kernel_url"] = launch.get("kernel_url")
    lines.append("### 🚀 تم الدفع")
    lines.append(f"- رابط الكيرنل: {launch.get('kernel_url') or 'https://www.kaggle.com/code/' + slug}")
    lines.append("")
    _meth_step(run_id, step_type="execute", ok=True,
               note=f"kernel={slug} pushed via Kaggle API")

    # ── 2) مراقبة حتى RUNNING ثم اكتمال ──
    status_txt = ""
    final_status = None
    waited = 0
    max_wait = 60  # مراقبة قصيرة في الجلسة: Kaggle kernel قد يستغرق ساعات تدريبًا
    while waited < max_wait:
        time.sleep(5)
        waited += 5
        try:
            st = status_kaggle_kernel(job_id)
            status_txt = st.get("status_raw") or json.dumps(st, ensure_ascii=False)
        except Exception as e:
            status_txt = f"status err: {e}"
        low = status_txt.lower()
        # نواصل المراقبة حتى حالة نهائية (complete/failed/error) — "running" ليست نهائية
        for cand in ("complete", "error", "failed", "killed", "cancelled"):
            if cand in low:
                final_status = cand
                break
        if final_status:
            break
    base["poll_seconds"] = waited
    base["status_raw"] = status_txt[:600]
    base["final_status"] = final_status

    # ── 3) قراءة Logs عند أي حالة نهائية ──
    if final_status in ("complete", "failed", "error"):
        try:
            dl = download_kaggle_output(job_id)
            base["output_files"] = dl.get("files") or []
            out_dir = ROOT / (dl.get("output_dir") or "")
            tail = ""
            for cand in ("logs", "output"):
                p = out_dir / cand if out_dir.is_dir() else None
                if p and p.is_dir():
                    logs = sorted(p.iterdir())[-1:]
                    if logs:
                        tail = logs[0].read_text(errors="ignore")[-3000:]
                        break
            base["log_tail"] = tail
            if final_status != "complete":
                err_ctx = tail or status_txt
                try:
                    if "Memory" in err_ctx or "CUDA" in err_ctx or "OOM" in err_ctx:
                        _record_kernel_oom_lesson(context=err_ctx[:400],
                                                  current_batch=batch)
                except Exception:
                    pass
        except Exception as e:
            base["download_error"] = str(e)

    _meth_step(run_id, step_type="verify", ok=final_status == "complete",
               note=f"kaggle status={final_status} waited={waited}s")
    base["status"] = "completed" if final_status == "complete" else "running"
    _meth_task_finished(
        run_id, status="done" if final_status == "complete" else "failed",
        ok=bool(final_status == "complete"),
        result_summary=f"preset={cfg['preset']} d={d_model} N={n} "
                       f"epochs={epochs} kaggle_status={final_status}",
    )
    _append_training_run(base)

    lines.append("### 📡 حالة الكيرنل")
    lines.append(f"- الحالة الآن: **{final_status or 'لم تُحدد بعد — راقب من Kaggle'}**")
    if final_status == "complete":
        lines.append("- ✅ التدريب انتهى — الـcheckpoint يُرفع تلقائيًا لـ GitHub (AUTO_PUSH=1) إن وُجد GITHUB_TOKEN في Kaggle Secrets.")
    elif final_status in ("failed", "error"):
        lines.append("- ❌ انتهى بفشل — راجع Logs في Kaggle. إن كان السبب ذاكرة/توصيلات: الدرس المسجّل يُصيّر الـbatch للنصف تلقائيًا.")
    else:
        lines.append("- ⏳ ما زال في الطابور/التشغيل — Kaggle قد يستغرق دقائق إلى ساعات حسب preset.")
        lines.append("> راقب يدويًا: `حالة kaggle ` أو افتح رابط الكيرنل أعلاه.")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({k: v for k, v in base.items() if k != "status_raw"},
                            ensure_ascii=False, indent=2, default=str)[:1500])
    lines.append("```")
    return "\n".join(lines)


def list_training_runs(limit: int = 15) -> str:
    rows = _read_training_runs(limit=max(5, min(int(limit), 50)))
    lines = ["## 📜 سجل مهام التدريب", ""]
    if not rows:
        lines.append("لا توجد مهام مسجّلة بعد.")
        lines.append("جرّب: `مهمة تدريب data/samples/classification_demo.csv الهدف=label`")
        return "\n".join(lines)
    for r in reversed(rows):
        st = r.get("status", "?")
        icon = {
            "planned": "📋",
            "running": "⏳",
            "completed": "✅",
            "failed": "❌",
            "rejected": "🚫",
        }.get(st, "•")
        plan = r.get("plan") or {}
        lines.append(
            f"{icon} `{r.get('id')}` · **{st}** · "
            f"{plan.get('path') or r.get('path')} · "
            f"task={plan.get('task', '?')} · engine={plan.get('engine', r.get('engine', '?'))}"
        )
        if r.get("error"):
            lines.append(f"   ↳ {r['error'][:120]}")
    return "\n".join(lines)


def train_from_csv(
    path_str: str,
    target_col: Optional[str] = None,
    epochs: int = 30,
    prefer: str = "auto",
    run_id: Optional[str] = None,
    resume: bool = False,
) -> str:
    """تدريب على ملف CSV من القرص (مسار نسبي أو مطلق داخل المشروع)."""
    try:
        path = _safe_resolve_data_path(path_str)
    except PermissionError as e:
        return f"🚫 مرفوض: {e}"
    except FileNotFoundError as e:
        return f"❌ {e}\nاستخدم **قائمة csv** لعرض المتاح."
    except Exception as e:
        return f"❌ مسار غير صالح: {type(e).__name__}: {e}"

    try:
        header, data = _load_csv_table(path)
        bundle = _infer_target_and_matrix(header, data, target_col=target_col)
    except Exception as e:
        return f"❌ فشل قراءة/تفسير CSV: {type(e).__name__}: {e}"

    task = bundle["task"]
    lines = [
        f"## 📥 تدريب من CSV",
        f"- الملف: `{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}`",
        f"- عينات: **{bundle['n_samples']}** | المهمة: **{task}** | الهدف: `{bundle['target_name']}`",
        f"- نمط الميزات: **{bundle['feature_mode']}** → {bundle['feature_names'][:8]}",
        "",
    ]

    # جهّز X
    if bundle["feature_mode"] == "text":
        X = _text_to_bow(bundle["texts"] or [])
        lines.append(f"- تحويل النص إلى BoW بحجم {X.shape[1]}")
    else:
        X = bundle["X"]
        # تطبيع
        mu = X.mean(axis=0)
        sig = X.std(axis=0) + 1e-8
        X = (X - mu) / sig

    y = bundle["y"]
    n = len(y)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    split = max(1, int(n * 0.75))
    tr, te = idx[:split], idx[split:]
    if len(te) == 0:
        te = tr[-max(1, len(tr) // 5) :]

    use_torch = prefer in ("auto", "torch", "pytorch", "cnn", "text")
    use_sk = prefer in ("auto", "sklearn", "sk") and _SKLEARN_OK and bundle["feature_mode"] == "numeric"

    results = []

    if use_sk and bundle["feature_mode"] == "numeric":
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

            if task == "classification":
                m = RandomForestClassifier(n_estimators=60, random_state=42)
                m.fit(X[tr], y[tr])
                pred = m.predict(X[te])
                acc = float(accuracy_score(y[te], pred))
                f1 = float(f1_score(y[te], pred, average="weighted"))
                results.append(f"sklearn RF: Accuracy={acc:.4f} F1={f1:.4f}")
                out = ARTIFACTS / f"csv_rf_clf_{int(time.time())}.pkl"
            else:
                m = RandomForestRegressor(n_estimators=60, random_state=42)
                m.fit(X[tr], y[tr])
                pred = m.predict(X[te])
                mse = float(mean_squared_error(y[te], pred))
                r2 = float(r2_score(y[te], pred))
                results.append(f"sklearn RF: MSE={mse:.4f} R²={r2:.4f}")
                out = ARTIFACTS / f"csv_rf_reg_{int(time.time())}.pkl"
            with open(out, "wb") as f:
                pickle.dump({"model": m, "path": str(path), "task": task}, f)
            results.append(f"محفوظ: `{out.relative_to(ROOT)}`")
        except Exception as e:
            results.append(f"sklearn فشل: {e}")

    if use_torch and _TORCH_OK:
        try:
            if prefer in ("cnn",) and bundle["feature_mode"] == "numeric":
                results.append(train_torch_cnn_on_arrays(X, y, task, epochs=epochs))
            elif bundle["feature_mode"] == "text" or prefer in ("text",):
                results.append(
                    train_torch_text_on_texts(
                        bundle["texts"] or [], y, task, epochs=epochs
                    )
                )
            else:
                # MLP على المصفوفة (يدعم checkpoint/استئناف عبر run_id)
                results.append(
                    train_torch_on_arrays(
                        X, y, task, epochs=epochs, run_id=run_id, resume=resume
                    )
                )
        except Exception as e:
            results.append(f"torch فشل: {e}")
    elif use_torch and not _TORCH_OK:
        results.append("PyTorch غير متاح.")

    if not results:
        results.append("لم يُشغَّل أي محرك تدريب (تحقق من المكتبات).")

    lines.append("### النتائج")
    body = "\n".join(lines)
    for r in results:
        if isinstance(r, str) and r.startswith("##"):
            body += "\n\n" + r
        else:
            body += "\n- " + str(r)
    return body


def train_torch_on_arrays(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    epochs: int = 25,
    run_id: Optional[str] = None,
    checkpoint_every: int = 5,
    resume: bool = False,
) -> str:
    epochs = _sb_clamp_epochs(max(3, min(int(epochs), 120)))
    checkpoint_every = max(1, int(checkpoint_every))

    # ═══ منهجية الوالد: التدريب الثقيل في كيرنل منعزل ═══
    # إن توفر الكيرنل المنعزل، يُنفَّذ التدريب في عملية معزولة عن Streamlit
    # (لا حجب للشات، وقبض آمن لـ OOM/Timeout) مع حفظ checkpoint ودرس تلقائي
    # عند OOM. عند غيابه يعمل بالآلية الحالية دون أي تغيير.
    if _kernel_available():
        ckpt_dir = str(_checkpoint_dir(run_id)) if run_id else str(
            ARTIFACTS / "model_training" / "fallback")
        bs = _suggest_kernel_batch(int(X.shape[0]),
                                   n_features=int(X.shape[1]))
        source = _torch_kernel_source(
            X, y, task, epochs, bs, run_id, ckpt_dir, None)
        direct = lambda: _run(batch_size=bs)  # noqa: E731
        res = train_via_kernel(source, timeout=600, fallback=direct)
        if "❌ فشل التنفيذ في الكيرنل" in res and "Memory" in res:
            _record_kernel_oom_lesson(context=res[:400], current_batch=bs)
        return res

    import torch
    import torch.nn as nn
    device, dev_info = _gpu_torch_device()
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    free_v = getattr(dev_info, "free_vram_gb", None) if dev_info else None
    init_bs = _gpu_suggest_batch(n_samples, n_features=n_features, free_vram_gb=free_v)

    def _run(batch_size: int) -> str:
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        n = X_t.shape[0]
        idx = np.random.permutation(n)
        split = max(1, int(0.8 * n))
        tr, te = idx[:split], idx[split:] if split < n else idx[-1:]
        if task == "classification":
            n_out = int(y.max()) + 1 if y.size else 2
            y_t = torch.tensor(y, dtype=torch.long, device=device)
            loss_fn = nn.CrossEntropyLoss()
        else:
            n_out = 1
            y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
            loss_fn = nn.MSELoss()
        model = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_out),
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        hist: List[float] = []
        val_hist: List[float] = []
        stopped_early = False
        es = None
        if _SANDBOX_OK:
            try:
                from ai.training_sandbox import EarlyStopping, load_guardrails
                _es_cfg = (load_guardrails().get("early_stopping") or {})
                if _es_cfg.get("enabled", True):
                    es = EarlyStopping.from_config()
            except Exception:
                es = _EarlyStopping.from_config() if _SANDBOX_OK else None

        start_epoch = 0
        best_val = float("inf")
        resume_note = ""
        if resume and run_id:
            ck = _load_checkpoint(run_id, "latest")
            if ck is not None and int(ck.get("n_features", -1)) == n_features and ck.get("task") == task:
                try:
                    model.load_state_dict(ck["model_state"])
                    opt.load_state_dict(ck["optimizer_state"])
                    start_epoch = int(ck.get("epoch", 0))
                    hist = list(ck.get("history") or [])
                    val_hist = list(ck.get("val_history") or [])
                    best_val = float(ck.get("best_val_loss", float("inf")))
                    resume_note = f" | استؤنف من الحقبة {start_epoch}"
                except Exception:
                    start_epoch = 0

        model.train()
        actual_epochs = start_epoch
        tr_idx = tr
        for ep in range(start_epoch, start_epoch + epochs):
            model.train()
            # mini-batches
            perm = np.random.permutation(tr_idx)
            ep_loss = 0.0
            n_batches = 0
            for i in range(0, len(perm), batch_size):
                bi = perm[i : i + batch_size]
                xb = X_t[bi]
                yb = y_t[bi]
                opt.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                n_batches += 1
            train_l = ep_loss / max(1, n_batches)
            hist.append(train_l)
            actual_epochs = ep + 1
            model.eval()
            with torch.no_grad():
                v_out = model(X_t[te])
                v_loss = float(loss_fn(v_out, y_t[te]).item())
            model.train()
            val_hist.append(v_loss)
            is_best = v_loss < best_val
            if is_best:
                best_val = v_loss
            if run_id and (actual_epochs % checkpoint_every == 0 or is_best):
                try:
                    _save_checkpoint(
                        run_id, actual_epochs, model, opt, v_loss, best_val,
                        hist, val_hist, task, n_features, batch_size, is_best=is_best,
                    )
                except Exception:
                    pass
            if es is not None and es.step(v_loss):
                stopped_early = True
                break
        if run_id:
            try:
                last_v = val_hist[-1] if val_hist else None
                _save_checkpoint(
                    run_id, actual_epochs, model, opt, last_v, best_val,
                    hist, val_hist, task, n_features, batch_size,
                    is_best=(last_v is not None and last_v <= best_val),
                )
            except Exception:
                pass
        model.eval()
        with torch.no_grad():
            out = model(X_t[te])
            if task == "classification":
                metric = float((out.argmax(-1) == y_t[te]).float().mean().item())
                mname = "Accuracy"
            else:
                metric = float(loss_fn(out, y_t[te]).item())
                mname = "MSE"
        outp = ARTIFACTS / f"torch_mlp_csv_{task}_{int(time.time())}.pt"
        if _SANDBOX_OK:
            try:
                _sb_assert_write(outp)
            except Exception:
                pass
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(
            {
                "state_dict": cpu_state,
                "task": task,
                "n_features": n_features,
                "actual_epochs": actual_epochs,
                "early_stopped": stopped_early,
                "val_loss_history": val_hist[-20:],
                "device": str(device),
                "batch_size": batch_size,
            },
            outp,
        )
        _gpu_empty_cache()
        es_note = f" | early_stop@{actual_epochs}" if stopped_early else f" | epochs={actual_epochs}"
        dev_note = f" | device={device}"
        if dev_info is not None:
            dev_note += f" ({getattr(dev_info, 'reason', '')})"
        ckpt_note = f" | checkpoint كل {checkpoint_every} حقب → `{run_id}`" if run_id else ""
        return (
            f"## ✅ Torch MLP على بيانات حقيقية\n"
            f"- {mname}={metric:.4f} | planned={epochs}{es_note}{resume_note} | features={n_features} | batch={batch_size}{dev_note}\n"
            f"- train loss: {', '.join(f'{x:.4f}' for x in hist[-5:])}\n"
            f"- val loss: {', '.join(f'{x:.4f}' for x in val_hist[-5:])}\n"
            f"- best val_loss حتى الآن: **{best_val:.4f}**{ckpt_note}\n"
            f"- `{outp.relative_to(ROOT)}`"
        )

    try:
        result, used_bs, oom_log = _gpu_oom_backoff(_run, init_bs, min_batch=1, max_retries=4)
        if len(oom_log) > 1:
            result += "\n- OOM backoff: " + " → ".join(oom_log)
        return result
    except Exception as e:
        _gpu_empty_cache()
        return f"❌ فشل تدريب MLP على {device}: {type(e).__name__}: {e}"


def train_torch_cnn_on_arrays(
    X: np.ndarray, y: np.ndarray, task: str, epochs: int = 25
) -> str:
    """Conv1d على متجه الميزات (مناسب لبيانات جدولية/إشارات قصيرة)."""
    if not _TORCH_OK:
        return "PyTorch غير متاح"
    epochs = _sb_clamp_epochs(max(3, min(int(epochs), 100)))
    n, f = X.shape
    # حشّ إلى طول زوجي مناسب
    length = max(8, f)
    if f < length:
        pad = np.zeros((n, length - f), dtype=np.float64)
        Xc = np.concatenate([X, pad], axis=1)
    else:
        Xc = X[:, :length]
        length = Xc.shape[1]

    device, _dev_info = _gpu_torch_device()
    Xt = torch.tensor(Xc, dtype=torch.float32, device=device).unsqueeze(1)  # B,1,L
    idx = np.random.default_rng(1).permutation(n)
    split = max(1, int(n * 0.75))
    te = idx[split:] if len(idx[split:]) > 0 else idx[-1:]
    tr = idx[:split]

    class SmallCNN(nn.Module):
        def __init__(self, length: int, n_out: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(4),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 4, 64),
                nn.ReLU(),
                nn.Linear(64, n_out),
            )

        def forward(self, x):
            return self.fc(self.conv(x))

    if task == "classification":
        n_out = int(y.max()) + 1
        yt = torch.tensor(y, dtype=torch.long, device=device)
        loss_fn = nn.CrossEntropyLoss()
    else:
        n_out = 1
        yt = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
        loss_fn = nn.MSELoss()

    model = SmallCNN(length, n_out).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    hist = []
    val_hist = []
    stopped_early = False
    es = None
    if _SANDBOX_OK:
        try:
            es = _EarlyStopping.from_config()
        except Exception:
            es = None
    model.train()
    actual_epochs = 0
    for _ in range(epochs):
        opt.zero_grad()
        out = model(Xt[tr])
        loss = loss_fn(out, yt[tr])
        loss.backward()
        opt.step()
        hist.append(float(loss.item()))
        actual_epochs += 1
        model.eval()
        with torch.no_grad():
            v_out = model(Xt[te])
            v_loss = float(loss_fn(v_out, yt[te]).item())
        model.train()
        val_hist.append(v_loss)
        if es is not None and es.step(v_loss):
            stopped_early = True
            break
    model.eval()
    with torch.no_grad():
        out = model(Xt[te])
        if task == "classification":
            metric = float((out.argmax(-1) == yt[te]).float().mean().item())
            mname = "Accuracy"
        else:
            metric = float(loss_fn(out, yt[te]).item())
            mname = "MSE"
    outp = ARTIFACTS / f"torch_cnn_{task}_{int(time.time())}.pt"
    torch.save({"state_dict": model.state_dict(), "task": task, "length": length}, outp)
    return (
        f"## ✅ Torch CNN-1D\n"
        f"- {mname}={metric:.4f} | epochs={epochs} | length={length}\n"
        f"- loss: {', '.join(f'{x:.4f}' for x in hist[-5:])}\n"
        f"- `{outp.relative_to(ROOT)}`"
    )


def train_torch_text_on_texts(
    texts: List[str], y: np.ndarray, task: str, epochs: int = 20, max_len: int = 32
) -> str:
    """Tokenizer حرفي بسيط + Embedding + TransformerEncoder layer."""
    if not _TORCH_OK:
        return "PyTorch غير متاح"
    epochs = _sb_clamp_epochs(max(3, min(int(epochs), 80)))
    # بناء قاموس أحرف
    chars = sorted({c for t in texts for c in str(t)})[:200]
    stoi = {c: i + 2 for i, c in enumerate(chars)}  # 0 pad, 1 unk
    vocab = len(stoi) + 2

    def encode(t: str) -> List[int]:
        ids = [stoi.get(c, 1) for c in str(t)[:max_len]]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids[:max_len]

    X = torch.tensor([encode(t) for t in texts], dtype=torch.long)
    n = len(texts)
    idx = np.random.default_rng(2).permutation(n)
    split = max(1, int(n * 0.75))
    te = idx[split:] if len(idx[split:]) > 0 else idx[-1:]
    tr = idx[:split]

    class TinyTextTransformer(nn.Module):
        def __init__(self, vocab: int, n_out: int, d: int = 64, nhead: int = 4):
            super().__init__()
            self.emb = nn.Embedding(vocab, d, padding_idx=0)
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=nhead, dim_feedforward=128, batch_first=True, dropout=0.1
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Linear(d, n_out)

        def forward(self, x):
            h = self.emb(x)
            mask = x == 0
            h = self.enc(h, src_key_padding_mask=mask)
            # mean pool over non-pad
            mask_f = (~mask).float().unsqueeze(-1)
            pooled = (h * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
            return self.head(pooled)

    if task == "classification":
        n_out = int(y.max()) + 1
        yt = torch.tensor(y, dtype=torch.long)
        loss_fn = nn.CrossEntropyLoss()
    else:
        n_out = 1
        yt = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        loss_fn = nn.MSELoss()

    model = TinyTextTransformer(vocab, n_out)
    opt = optim.Adam(model.parameters(), lr=2e-3)
    hist = []
    val_hist = []
    stopped_early = False
    es = None
    if _SANDBOX_OK:
        try:
            es = _EarlyStopping.from_config()
        except Exception:
            es = None
    model.train()
    actual_epochs = 0
    for _ in range(epochs):
        opt.zero_grad()
        out = model(X[tr])
        loss = loss_fn(out, yt[tr])
        loss.backward()
        opt.step()
        hist.append(float(loss.item()))
        actual_epochs += 1
        model.eval()
        with torch.no_grad():
            v_out = model(X[te])
            v_loss = float(loss_fn(v_out, yt[te]).item())
        model.train()
        val_hist.append(v_loss)
        if es is not None and es.step(v_loss):
            stopped_early = True
            break
    model.eval()
    with torch.no_grad():
        out = model(X[te])
        if task == "classification":
            metric = float((out.argmax(-1) == yt[te]).float().mean().item())
            mname = "Accuracy"
        else:
            metric = float(loss_fn(out, yt[te]).item())
            mname = "MSE"
    outp = ARTIFACTS / f"torch_text_{task}_{int(time.time())}.pt"
    torch.save(
        {"state_dict": model.state_dict(), "stoi": stoi, "task": task, "max_len": max_len},
        outp,
    )
    return (
        f"## ✅ Torch Text Transformer (صغير)\n"
        f"- {mname}={metric:.4f} | epochs={epochs} | vocab_chars={vocab} | max_len={max_len}\n"
        f"- loss: {', '.join(f'{x:.4f}' for x in hist[-5:])}\n"
        f"- `{outp.relative_to(ROOT)}`"
    )


def save_upload_and_train(
    filename: str,
    raw_bytes: bytes,
    target_col: Optional[str] = None,
    prefer: str = "auto",
    epochs: int = 25,
) -> str:
    """حفظ ملف مرفوع ثم تدريبه (CSV أو نص سطور label\\ttext)."""
    safe = Path(filename).name.replace("..", "_")
    dest = UPLOAD_DIR / f"{int(time.time())}_{safe}"
    dest.write_bytes(raw_bytes)
    if safe.lower().endswith(".csv"):
        return (
            f"تم حفظ الرفع: `{dest.relative_to(ROOT)}`\n\n"
            + train_from_csv(str(dest), target_col=target_col, epochs=epochs, prefer=prefer)
        )
    # محاولة تفسير كنص: كل سطر label||text أو text فقط
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        tmp = UPLOAD_DIR / f"{dest.stem}_as.csv"
        import csv as _csv

        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["text", "label"])
            for ln in lines[:3000]:
                if "||" in ln:
                    a, b = ln.split("||", 1)
                    w.writerow([b.strip(), a.strip()])
                elif "\t" in ln:
                    a, b = ln.split("\t", 1)
                    w.writerow([b.strip(), a.strip()])
                else:
                    w.writerow([ln, "0"])
        return (
            f"تم تحويل النص إلى CSV: `{tmp.relative_to(ROOT)}`\n\n"
            + train_from_csv(str(tmp), target_col="label", epochs=epochs, prefer="text")
        )
    except Exception as e:
        return f"حُفظ الملف `{dest.relative_to(ROOT)}` لكن تعذّر التدريب التلقائي: {e}"


def train_torch_cnn_demo(epochs: int = 20) -> str:
    """تجربة CNN على بيانات اصطناعية جدولية."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(500, 24))
    y = (X[:, :3].sum(axis=1) > 0).astype(np.int64)
    return train_torch_cnn_on_arrays(X, y, "classification", epochs=epochs)


def train_torch_text_demo(epochs: int = 15) -> str:
    pos = ["رائع جدا", "ممتاز", "أحب هذا", "تجربة جيدة", "أنصح به"] * 30
    neg = ["سيء", "فظيع", "لا أنصح", "مخيب", "ضعيف"] * 30
    texts = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    return train_torch_text_on_texts(texts, y, "classification", epochs=epochs)



# ═══════════════════════════════════════════════════════════════════════════
# 6) موجّه الأوامر النصية
# ═══════════════════════════════════════════════════════════════════════════

def handle_training_command(user_input: str) -> Optional[str]:
    """
    يفسّر أوامر عربية/إنجليزية شائعة.
    يعيد نص الأداة أو None لتمرير الرسالة للمحادثة.
    """
    text = (user_input or "").strip()
    if not text:
        return None

    # مساعدة وصياغة موحّدة (أوامر / مساعدة / ماذا تستطيع)
    if _LEXICON_OK and _help_handle is not None:
        try:
            help_out = _help_handle(text)
            if help_out is not None:
                return help_out
        except Exception as _h_err:
            logger.warning("lexicon help: %s", _h_err)
        try:
            text = _rewrite_cmd(text)
        except Exception:
            pass

    low = text.lower()

    # ── لوحة التحكم والخطوة التالية (أولوية عالية قبل الموجّهات الفرعية) ──
    if re.search(
        r"(لوحة\s*التحكم|dashboard|نظرة\s*عامة\s*(على\s*)?(التدريب)?|"
        r"حالة\s*التدريب|وضع\s*التدريب|ملخص\s*التدريب|تقرير\s*التدريب|"
        r"^لوحة$|^dashboard$)",
        text,
        re.I,
    ):
        return training_dashboard()

    if re.search(
        r"(^ماذا\s*بعد$|^الخطوة\s*التالية$|next\s*step|"
        r"اقترح\s*(ال)?خطوة|ابدأ\s*تدريب\s*ذكي|تدريب\s*ذكي|smart\s*train)",
        text,
        re.I,
    ):
        execute = bool(re.search(r"(ابدأ|شغّل|نفّذ|execute|start)", text, re.I))
        return smart_train_next(recommend_only=not execute)

    if _KERN_OK and _kern_handle is not None:
        try:
            kn = _kern_handle(text)
            if kn is not None:
                return kn
        except Exception as _kn_err:
            logger.warning("kernel: %s", _kn_err)
    if _MESH_OK and _mesh_handle is not None:
        try:
            mh = _mesh_handle(text)
            if mh is not None:
                return mh
        except Exception as _mh_err:
            logger.warning("mesh: %s", _mh_err)

    if _LFS_OK and _lfs_handle is not None:
        try:
            lf = _lfs_handle(text)
            if lf is not None:
                return lf
        except Exception as _lf_err:
            logger.warning("lfs: %s", _lf_err)

    if _CKGQ_OK and _ckgq_handle is not None:
        try:
            cq = _ckgq_handle(text)
            if cq is not None:
                return cq
        except Exception as _cq_err:
            logger.warning("ckg_quality: %s", _cq_err)

    if _BILL_OK and _bill_handle is not None:
        try:
            bl = _bill_handle(text)
            if bl is not None:
                return bl
        except Exception as _bl_err:
            logger.warning("billing: %s", _bl_err)

    if _RL_OK and _rl_handle is not None:
        try:
            rl = _rl_handle(text)
            if rl is not None:
                return rl
        except Exception as _rl_err:
            logger.warning("rl: %s", _rl_err)

    if _DEVOPS_OK and _devops_handle is not None:
        try:
            dv = _devops_handle(text)
            if dv is not None:
                return dv
        except Exception as _dv_err:
            logger.warning("devops: %s", _dv_err)
    if _GW_OK and _gw_handle is not None:
        try:
            gw = _gw_handle(text)
            if gw is not None:
                return gw
        except Exception as _gw_err:
            logger.warning("gateway: %s", _gw_err)
    if _QUANT_OK and _quant_handle is not None:
        try:
            qt = _quant_handle(text)
            if qt is not None:
                return qt
        except Exception as _qt_err:
            logger.warning("quant: %s", _qt_err)
    if _SBRIDGE_OK and _sbridge_handle is not None:
        try:
            sb = _sbridge_handle(text)
            if sb is not None:
                return sb
        except Exception as _sb_err:
            logger.warning("sbridge: %s", _sb_err)

    if _PRED_OK and _pred_handle is not None:
        try:
            pd = _pred_handle(text)
            if pd is not None:
                return pd
        except Exception as _pd_err:
            logger.warning("predictive: %s", _pd_err)

    if _RETRAIN_OK and _retrain_handle is not None:
        try:
            rt = _retrain_handle(text)
            if rt is not None:
                return rt
        except Exception as _rt_err:
            logger.warning("active_retrain: %s", _rt_err)

    # السرب الاجتماعي
    if _SOCIAL_SWARM_OK and _social_swarm_handle is not None:
        try:
            ss = _social_swarm_handle(text)
            if ss is not None:
                return ss
        except Exception as _ss_err:
            logger.warning("social_swarm: %s", _ss_err)

    # سيادة التشغيل: MCP/WorldModel/Sensors
    if _SOV_OK and _sov_handle is not None:
        try:
            sv = _sov_handle(text)
            if sv is not None:
                return sv
        except Exception as _sv_err:
            logger.warning("sovereignty: %s", _sv_err)

    # طبقة الحضارة / ما بعد القمة
    if _CIV_OK and _civ_handle is not None:
        try:
            cv = _civ_handle(text)
            if cv is not None:
                return cv
        except Exception as _cv_err:
            logger.warning("civilization: %s", _cv_err)

    # تفعيل الإنتاجية + تدريب مستمر
    if _PROD_OK and _prod_handle is not None:
        try:
            pr = _prod_handle(text)
            if pr is not None:
                return pr
        except Exception as _pr_err:
            logger.warning("production: %s", _pr_err)
    if _CONT_OK and _cont_handle is not None:
        try:
            ct = _cont_handle(text)
            if ct is not None:
                return ct
        except Exception as _ct_err:
            logger.warning("continuous: %s", _ct_err)

    # Hierarchical Dynamic MoE (خليط خبراء هرمي)
    if _MOE_OK and _moe_handle is not None:
        try:
            moe_out = _moe_handle(text)
            if moe_out is not None:
                return moe_out
        except Exception as _moe_err:
            logger.warning("hierarchical_moe: %s", _moe_err)

    # المحرك الاقتصادي (AIaaS / سوق / Arbitrage / بيانات)
    if _ECONOMIC_OK and _economic_handle is not None:
        try:
            ec = _economic_handle(text)
            if ec is not None:
                return ec
        except Exception as _ec_err:
            logger.warning("economic: %s", _ec_err)

    # Super AI Orchestrator — حوسبة فائقة / بيانات / تطور / سرب
    if _SUPER_OK and _super_handle is not None:
        try:
            su = _super_handle(text)
            if su is not None:
                return su
        except Exception as _su_err:
            logger.warning("super-ai: %s", _su_err)

    # Meta-AI — تفكير عميق / NAS / عتاد / ذاكرة متجهة
    if _META_OK and _meta_handle is not None:
        try:
            mt = _meta_handle(text)
            if mt is not None:
                return mt
        except Exception as _mt_err:
            logger.warning("meta-ai: %s", _mt_err)

    # العالِم المبتكر + المدير الأمني والمالي
    if _SCIENTIST_OK and _scientist_handle is not None:
        try:
            sc = _scientist_handle(text)
            if sc is not None:
                return sc
        except Exception as _sc_err:
            logger.warning("scientist: %s", _sc_err)

    # المهندس المعماري (تحكيم / بحث فائق / ضغط / اتحادي)
    if _ARCHITECT_OK and _architect_handle is not None:
        try:
            ar = _architect_handle(text)
            if ar is not None:
                return ar
        except Exception as _ar_err:
            logger.warning("architect: %s", _ar_err)

    # ── تدريب Kaggle SurahChain المباشر (نفس منهجية NSM) ──
    # أولوية أعلى من المنسّق العام لأن أوامره أدق — قبل orchestrator
    m_kagtrain = re.search(
        r"(?:ادرب|درّ?ب|شغّل?|نفّ?ذ|أطلق|ابدأ|جهّ?ز|حضّر?|prepare)\s*"
        r"(?:تدريب\s*)?(?:kaggle|كاجل|كاغل)(?:\s+(surahchain|سلسلة\s*السور))?|"
        r"(?:train|run|launch)\s*kaggle\s*(?:surah)?",
        low,
    )
    if m_kagtrain:
        kpreset = "small"
        kepochs = 15
        kfresh = True
        kap = True
        for m2 in re.finditer(r"(?:preset|نمط)\s*[=:]?\s*(small|medium|large|smoke)", low):
            kpreset = m2.group(1)
        m_ep = re.search(r"(?:epochs?|عصور|حقب)\s*[=:]?\s*(\d+)", text, re.I)
        if m_ep:
            kepochs = max(1, min(200, int(m_ep.group(1))))
        if re.search(r"(استكمال|resume|continue)", low):
            kfresh = False
        if re.search(r"(بدون\s*(push|رفع|رفع\s*تلقائي)|skip\s*push|auto_push\s*=\s*0)", low):
            kap = False
        return run_kaggle_training(
            preset=kpreset, epochs=kepochs, fresh=kfresh, auto_push=kap)

    # منسّق المنصات البعيدة (Kaggle + Colab + كفاءة التدريب)
    if _ORCH_OK and _orch_handle is not None:
        try:
            oc = _orch_handle(text)
            if oc is not None:
                return oc
        except Exception as _oc_err:
            logger.warning("orchestrator: %s", _oc_err)

        # Kaggle (API + Dual T4) — قبل remote العام لأن أوامره أكثر تحديداً
    if _KAGGLE_OK and _kaggle_handle is not None:
        try:
            kg = _kaggle_handle(text)
            if kg is not None:
                return kg
        except Exception as _kg_err:
            logger.warning("kaggle: %s", _kg_err)

    if _REMOTE_GPU_OK and _remote_gpu_handle is not None:
        try:
            rg = _remote_gpu_handle(text)
            if rg is not None:
                return rg
        except Exception as _rg_err:
            logger.warning("remote gpu: %s", _rg_err)

    # Apex autonomy (mergers / synthetic / DAO sim)
    if _APEX_OK and _apex_handle is not None:
        try:
            apx = _apex_handle(text)
            if apx is not None:
                return apx
        except Exception as _apx_err:
            logger.warning("apex handle: %s", _apx_err)

    # AIaaS platform commands
    if _AIAAS_OK and _aiaas_handle is not None:
        try:
            aa = _aiaas_handle(text)
            if aa is not None:
                return aa
        except Exception as _aa_err:
            logger.warning("aiaas handle: %s", _aa_err)

    # مصنع مستقل (أهداف عامة + موافقات)
    if _FACTORY_OK and _factory_handle is not None:
        try:
            fac = _factory_handle(text)
            if fac is not None:
                return fac
        except Exception as _fac_err:
            logger.warning("factory handle: %s", _fac_err)

    # حلقة التغذية الراجعة / registry / drift أولاً
    if _FEEDBACK_OK and _fb_handle is not None:
        try:
            fb = _fb_handle(text)
            if fb is not None:
                return fb
        except Exception as _fb_err:
            logger.warning("feedback handle: %s", _fb_err)

    # وصول إنترنت محكوم (arxiv / HF / بحث مقيّد)
    if _WEB_ACCESS_OK and _web_handle is not None:
        try:
            web = _web_handle(text)
            if web is not None:
                return web
        except Exception as _web_err:
            logger.warning("web handle: %s", _web_err)

    # جرد / inventory
    if re.search(r"(جرد|مخزون|ما\s*المتاح|inventory|بيئة\s*التدريب|ماذا\s*تستطيع)", text, re.I):
        return inventory()

    # خطة عامة (+ تلميح اختياري بعد : أو -)
    if re.search(r"(خطة|دورة\s*حياة|lifecycle)\s*(تدريب|نموذج)?", text, re.I) or low in (
        "خطة",
        "plan",
        "lifecycle",
    ):
        hint = ""
        m = re.search(r"(?:خطة|lifecycle|plan)\s*[:\-–]\s*(.+)$", text, re.I)
        if m:
            hint = m.group(1).strip()
        elif "خطة" in text and len(text) > 12:
            hint = re.sub(r"خطة\s*(دورة\s*حياة)?\s*(تدريب)?", "", text, flags=re.I).strip(' :-–')
        return lifecycle_plan(hint)

    # اقتراح خوارزمية
    if re.search(r"(اقترح|ما\s*أفضل|أي\s*خوارزم|suggest).{0,20}(خوارزم|نموذج|algorithm|model)", text, re.I):
        desc = re.sub(
            r"(اقترح|ما\s*أفضل|أي)\s*(خوارزم\w*|نموذج\w*|algorithm|model)\s*(ل|لـ|for)?\s*",
            "",
            text,
            flags=re.I,
        ).strip()
        return suggest_algorithm(desc or text)
    if re.search(r"^(اقترح\s+نموذج|suggest\s+model)\b", text, re.I):
        return suggest_algorithm(re.sub(r"^(اقترح\s+نموذج|suggest\s+model)\s*", "", text, flags=re.I))

    # نماذج محفوظة
    if re.search(r"(نماذج\s*محفوظة|قائمة\s*النماذج|saved\s*models|list\s*models)", text, re.I):
        return list_saved_models()

    # ── CKG / NSM محدد ────────────────────────────────────────────────────
    if re.search(r"(حالة|وضع|تقرير).{0,12}(ckg|سي\s*كي\s*جي|التدريب\s*المعرفي)|ckg\s*status", text, re.I):
        return ckg_status()
    if re.search(r"(حالة|وضع)\s*(التدريب)?\s*$", text.strip()) and "torch" not in low and "sklearn" not in low:
        # "حالة التدريب" العامة → اعرض الجرد + ملخص CKG إن وُجد
        return inventory() + "\n\n---\n\n" + ckg_status()

    if re.search(
        r"(اقترح|إعدادات).{0,15}(ckg|حزم|pack)|إعدادات\s*ckg|ckg\s*config",
        text,
        re.I,
    ):
        return ckg_recommend()

    if re.search(r"(تحليل|اتجاه).{0,10}(خسارة|loss).*ckg|(خسارة|loss)\s*ckg", text, re.I):
        return ckg_loss_trend()
    if re.search(r"(تحليل|اتجاه).{0,10}(خسارة|loss)|loss\s*trend", text, re.I):
        return ckg_loss_trend()

    dry = bool(re.search(r"تجريب|dry\s*-?run|محاكاة|بدون\s*تشغيل", text, re.I))

    if re.search(r"(شغّل|شغل|ابدأ).{0,15}(تدريب\s*)?(ckg|v3|المعرفي)|train\s*ckg|run\s*ckg", text, re.I):
        packs = 1
        m = re.search(r"(\d+)\s*(حزم|حزمة|packs?)", text)
        if m:
            packs = int(m.group(1))
        return run_ckg_step(packs=packs, dry_run=dry)

    if re.search(r"(شغّل|شغل)\s*(سكربت|script)?\s*(train[\w_\.]*)", text, re.I):
        m = re.search(r"(train[\w_\.]*)", text, re.I)
        return run_nsm_script(m.group(1) if m else "v3", dry_run=dry)
    if re.search(r"(شغّل|شغل).{0,10}(yemeni|pilot|batch|general)", text, re.I):
        for k in ("yemeni", "pilot", "batch", "general", "v3"):
            if k in low:
                return run_nsm_script(k, dry_run=dry)

    # ── تدريب عام sklearn / torch ─────────────────────────────────────────
    if re.search(r"(درّب|درب|train).{0,15}(تصنيف|classif)", text, re.I):
        n = 800
        m = re.search(r"(\d+)\s*(عينة|samples?)", text)
        if m:
            n = int(m.group(1))
        model = "auto"
        if re.search(r"logreg|logistic|لوجست", text, re.I):
            model = "logreg"
        elif re.search(r"boost|gbm|تعزيز", text, re.I):
            model = "gb"
        elif re.search(r"forest|غاب", text, re.I):
            model = "rf"
        return train_sklearn_demo(task="classification", n_samples=n, model_name=model)

    if re.search(r"(درّب|درب|train).{0,15}(انحدار|regress)", text, re.I):
        n = 800
        m = re.search(r"(\d+)\s*(عينة|samples?)", text)
        if m:
            n = int(m.group(1))
        return train_sklearn_demo(task="regression", n_samples=n)

    if re.search(r"(درّب|درب|train).{0,20}(torch|pytorch|شبكة|mlp|عصبي)", text, re.I):
        epochs = 25
        m = re.search(r"(\d+)\s*(حقب|epochs?|عصر)", text, re.I)
        if m:
            epochs = int(m.group(1))
        task = "regression" if re.search(r"انحدار|regress", text, re.I) else "classification"
        return train_torch_mlp(task=task, epochs=epochs)

    # سجل مهام التدريب
    if re.search(
        r"(سجل\s*مهام\s*التدريب|سجل\s*التدريب|training\s*runs|سجل\s*المهام)",
        text,
        re.I,
    ):
        return list_training_runs()

    # استئناف التدريب من آخر checkpoint بعد فشل
    if re.search(
        r"(استأنف|استكمل|أكمل)\s*(ال)?تدريب|resume\s*training|continue\s*training",
        text,
        re.I,
    ):
        target = None
        mt = re.search(r"(?:هدف|target|label)\s*[=:]\s*([\w\u0600-\u06FF]+)", text, re.I)
        if mt:
            target = mt.group(1)
        path_csv = None
        mp = re.search(r"((?:[\w./\\-]+/)*[\w.-]+\.csv)", text, re.I)
        if mp:
            path_csv = mp.group(1)
        epochs = 15
        me = re.search(r"(\d+)\s*(حقب|epochs?)", text, re.I)
        if me:
            epochs = int(me.group(1))
        return resume_training_mission(path_str=path_csv, target_col=target, epochs=epochs)

    # الرجوع لأفضل نسخة محفوظة (rollback)
    if re.search(
        r"(ارجع|استرجع|رجّع)\s*(ل|إلى)?\s*أفضل\s*(نموذج|نسخة)|rollback\s*(to\s*)?best|"
        r"استعد\s*أفضل",
        text,
        re.I,
    ):
        m_run = re.search(r"run_\d+", text)
        return rollback_to_best(run_id=m_run.group(0) if m_run else None)

    # مهمة تدريب موحّدة (معاينة افتراضياً)
    m_mission = re.search(
        r"(?:مهمة\s*تدريب|training\s*mission|خطة\s*تدريب\s*على)\s+((?:[\w./\\-]+/)*[\w.-]+\.csv)",
        text,
        re.I,
    )
    if m_mission:
        path_csv = m_mission.group(1)
        target = None
        mt = re.search(r"(?:هدف|target|label)\s*[=:]\s*([\w\u0600-\u06FF]+)", text, re.I)
        if mt:
            target = mt.group(1)
        epochs = 30
        me = re.search(r"(\d+)\s*(حقب|epochs?)", text, re.I)
        if me:
            epochs = int(me.group(1))
        prefer = "auto"
        if re.search(r"sklearn|غاب|forest", text, re.I):
            prefer = "sklearn"
        elif re.search(r"torch|pytorch|mlp", text, re.I):
            prefer = "torch"
        execute = bool(re.search(r"(نف[ّ]?ذ|شغ[ّ]?ل|execute|run\s*now|ابدأ\s*التنفيذ)", text, re.I))
        return run_training_mission(
            path_csv, target_col=target, epochs=epochs, prefer=prefer, execute=execute
        )

    # قائمة CSV
    if re.search(r"(قائمة|عرض|list).{0,12}(csv|بيانات)|ملفات\s*csv|csv\s*files", text, re.I):
        return list_csv_datasets()

    # درّب على csv path
    # درّب على csv path
    m_csv = re.search(
        r"(درّب|درب|train).{0,40}?((?:[\w./\\-]+/)*[\w.-]+\.csv)",
        text,
        re.I,
    )
    if m_csv:
        path_csv = m_csv.group(2)
        target = None
        mt = re.search(r"(?:هدف|target|label)\s*[=:]\s*([\w\u0600-\u06FF]+)", text, re.I)
        if mt:
            target = mt.group(1)
        epochs = 30
        me = re.search(r"(\d+)\s*(حقب|epochs?)", text, re.I)
        if me:
            epochs = int(me.group(1))
        prefer = "auto"
        if re.search(r"\bcnn\b|شبكة\s*التفاف", text, re.I):
            prefer = "cnn"
        elif re.search(r"نص|text|transformer", text, re.I):
            prefer = "text"
        elif re.search(r"sklearn|غاب|forest", text, re.I):
            prefer = "sklearn"
        return train_from_csv(path_csv, target_col=target, epochs=epochs, prefer=prefer)

    if re.search(r"(درّب|درب|train).{0,15}(cnn|التفاف|convolution)", text, re.I):
        epochs = 20
        m = re.search(r"(\d+)\s*(حقب|epochs?)", text, re.I)
        if m:
            epochs = int(m.group(1))
        return train_torch_cnn_demo(epochs=epochs)

    if re.search(r"(درّب|درب|train).{0,20}(نص|text\s*transformer|transformer\s*نص)", text, re.I):
        epochs = 15
        m = re.search(r"(\d+)\s*(حقب|epochs?)", text, re.I)
        if m:
            epochs = int(m.group(1))
        return train_torch_text_demo(epochs=epochs)

    if re.search(r"(colab|كوكلاب|google\s*colab)", text, re.I):
        return (
            "## 📒 تشغيل الوكيل على Google Colab\n\n"
            "1. Runtime → Change runtime type → **GPU (T4)**\n"
            "2. ارفع أو افتح الدفتر: `notebooks/NSM_Colab_Training_Agent.ipynb`\n"
            "3. نفّذ الخلايا: clone → pip → (اختياري Drive) → حالة gpu → تدريب\n"
            "4. اضبط في الدفتر: `os.environ['NSM_ALLOW_GPU']='1'`\n\n"
            "التهيئة: `scripts/colab_bootstrap.py`\n"
            "الدليل: `notebooks/README_COLAB.md`\n\n"
            "_التحكم في Colab من متصفح خارجي غير مدعوم في المستودع._"
        )

    if re.search(r"(حالة|status).{0,10}(gpu|cuda|vram|كرت)", text, re.I) or text.lower() in ("gpu", "حالة gpu"):
        return _gpu_report()

    # Sandbox / أول مهمة
    if re.search(r"(حالة|status).{0,12}(sandbox|عزل|حواجز|guardrail)|sandbox\s*status", text, re.I):
        if _SANDBOX_OK:
            return _sb_status()
        return "وحدة sandbox غير محمّلة."
    if re.search(r"(سجل|قائمة).{0,12}(مهام|missions)|list\s*missions", text, re.I):
        if _SANDBOX_OK:
            return _sb_list_missions()
        return "وحدة sandbox غير محمّلة."
    if re.search(
        r"(أول\s*مهمة|المهمة\s*الأولى|first\s*mission|أطلق\s*المهمة\s*الأولى)",
        text,
        re.I,
    ):
        if not _SANDBOX_OK:
            return "وحدة sandbox غير محمّلة."
        dry = bool(re.search(r"تجريب|dry\s*-?run|محاكاة", text, re.I))
        return _sb_run_first_mission(dry_run=dry)

    if re.search(
        r"(ثاني(?:ة)?\s*مهمة|المهمة\s*الثانية|second\s*mission|أطلق\s*المهمة\s*الثانية)",
        text,
        re.I,
    ):
        if not _SANDBOX_OK:
            return "وحدة sandbox غير محمّلة."
        dry = bool(re.search(r"تجريب|dry\s*-?run|محاكاة", text, re.I))
        return _sb_run_second_mission(dry_run=dry)

    # لوحة التحكم / الخطوة التالية الذكية
    if re.search(
        r"(لوحة\s*التحكم|dashboard|نظرة\s*عامة|حالة\s*التدريب|وضع\s*التدريب|"
        r"ملخص\s*التدريب|تقرير\s*التدريب)",
        text,
        re.I,
    ):
        return training_dashboard()

    if re.search(
        r"(الخطوة\s*التالية|ماذا\s*بعد|next\s*step|اقترح\s*(خطوة|تدريب)|"
        r"ابدأ\s*تدريب\s*ذكي|تدريب\s*ذكي|smart\s*train)",
        text,
        re.I,
    ):
        # "ابدأ تدريب ذكي" ينفّذ إن أمكن؛ الباقي توصية فقط
        execute = bool(re.search(r"(ابدأ|شغّل|نفّذ|execute|start)", text, re.I))
        return smart_train_next(recommend_only=not execute)

    # أوامر قصيرة
    aliases = {
        "جرد": inventory,
        "inventory": inventory,
        "مخزون": inventory,
        "خطة": lifecycle_plan,
        "plan": lifecycle_plan,
        "خطة تدريب": lifecycle_plan,
        "ckg": ckg_status,
        "حالة ckg": ckg_status,
        "وضع ckg": ckg_status,
        "خسارة": ckg_loss_trend,
        "loss": ckg_loss_trend,
        "محفوظات": list_saved_models,
        "نماذج": list_saved_models,
        "نماذج محفوظة": list_saved_models,
        "csv": list_csv_datasets,
        "قائمة csv": list_csv_datasets,
        "لوحة": training_dashboard,
        "لوحة التحكم": training_dashboard,
        "dashboard": training_dashboard,
        "نظرة عامة": training_dashboard,
        "ماذا بعد": (lambda: smart_train_next(True)),
        "الخطوة التالية": (lambda: smart_train_next(True)),
        "sandbox": (_sb_status if _SANDBOX_OK else inventory),
        "أول مهمة": (lambda: _sb_run_first_mission(False) if _SANDBOX_OK else "لا sandbox"),
        "ثاني مهمة": (lambda: _sb_run_second_mission(False) if _SANDBOX_OK else "لا sandbox"),
        "المهمة الثانية": (lambda: _sb_run_second_mission(False) if _SANDBOX_OK else "لا sandbox"),
        "أوامر": (lambda: _help_handle("أوامر") if _help_handle else inventory),
        "مساعدة": (lambda: _help_handle("مساعدة") if _help_handle else inventory),
    }
    key = text.strip().lower()
    if key in aliases:
        fn = aliases[key]
        return fn() if fn is not lifecycle_plan else lifecycle_plan()
    # مطابقة بعد التطبيع
    if _LEXICON_OK:
        nkey = _norm_ar(text)
        for ak, fn in list(aliases.items()):
            if _norm_ar(ak) == nkey:
                return fn() if fn is not lifecycle_plan else lifecycle_plan()

    return None
