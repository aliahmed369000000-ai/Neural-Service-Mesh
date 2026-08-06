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

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts" / "model_training"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

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

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


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
        f"- PyTorch: {'✅ ' + torch.__version__ if _TORCH_OK else '❌ غير متاح'}",
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
        "الوكيل يدعم: (1) تدريب عام sklearn/torch، (2) تشغيل سكربتات NSM، "
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
    epochs = max(3, min(int(epochs), 100))
    hidden = max(8, min(int(hidden), 512))
    t0 = time.time()
    device = torch.device("cpu")

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
        model.train()
        for ep in range(epochs):
            opt.zero_grad()
            out = model(Xtr)
            loss = loss_fn(out, ytr if n_out == 1 else ytr)
            loss.backward()
            opt.step()
            history.append(float(loss.item()))

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
    low = text.lower()

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

    # أوامر قصيرة
    aliases = {
        "جرد": inventory,
        "inventory": inventory,
        "خطة": lifecycle_plan,
        "plan": lifecycle_plan,
        "ckg": ckg_status,
        "حالة ckg": ckg_status,
        "خسارة": ckg_loss_trend,
        "loss": ckg_loss_trend,
        "محفوظات": list_saved_models,
        "نماذج": list_saved_models,
    }
    if text.strip().lower() in aliases:
        fn = aliases[text.strip().lower()]
        return fn() if fn is not lifecycle_plan else lifecycle_plan()

    return None
