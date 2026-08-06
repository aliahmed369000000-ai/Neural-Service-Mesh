"""
Model Training Agent — وكيل إدارة دورة حياة تدريب النماذج في NSM
================================================================
يوفّر أدوات حقيقية (وليست مجرد محادثة) لإدارة دورة حياة النموذج:

  1. جمع/فحص البيانات (ckg_sentences_*.pkl، حالة التدريب)
  2. اقتراح إعدادات الحزم والمعلمات حسب الرام المتاحة
  3. تشغيل خطوة تدريب آمنة (train_batch_v3) مع حدود واضحة
  4. تحليل اتجاه الخسارة والتقدّم
  5. تقرير دورة الحياة (بيانات → تدريب → تقييم → نشر/حفظ)

لا يعتمد على LangChain/AutoGen/MLflow — يتكامل مع البنية الموجودة
(ArabicTransformer + train_batch_v3 + ckg_train_state).
أي فشل في أداة لا يرفع استثناءً للواجهة؛ يُعاد تقرير نصي واضح.
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ModelTrainingAgent")

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "ckg_train_state_v3.json"
SENTENCES_V3 = ROOT / "ckg_sentences_v3.pkl"
SENTENCES_V2 = ROOT / "ckg_sentences_v2.pkl"
SENTENCES_V1 = ROOT / "ckg_sentences.pkl"
WEIGHTS_DIR = ROOT / "models" / "transformer_ckg_v3"
TRAIN_SCRIPT = ROOT / "train_batch_v3.py"

# حدود أمان لتشغيل التدريب من داخل الوكيل (منع استنزاف موارد غير متوقع)
_MAX_PACKS_PER_AGENT_RUN = 2
_MIN_AVAIL_RAM_GB = 1.2
_DEFAULT_TIMEOUT_S = 600


def _available_ram_gb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        avail_kb = info.get("MemAvailable") or info.get("MemFree", 0)
        return avail_kb / (1024.0 * 1024.0)
    except Exception:
        return 1.0


def _load_json(path: Path) -> Optional[dict]:
    try:
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("load_json %s: %s", path, e)
    return None


def _count_sentences(path: Path) -> Optional[int]:
    try:
        if not path.is_file():
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        return len(data) if hasattr(data, "__len__") else None
    except Exception as e:
        logger.warning("count_sentences %s: %s", path, e)
        return None


def get_training_status() -> str:
    """تقرير حالة التدريب الحالية (v3) + الملفات المتوفرة."""
    lines: List[str] = ["## 📊 حالة تدريب النماذج (CKG / ArabicTransformer v3)", ""]

    state = _load_json(STATE_FILE)
    n_sent = _count_sentences(SENTENCES_V3)

    if state:
        pos = int(state.get("position", 0) or 0)
        total = n_sent or int(state.get("total_sentences_seen", 0) or 0) or pos
        # إذا كان لدينا عدد الجمل الحقيقي، استخدمه كمقام
        if n_sent and n_sent > 0:
            total = n_sent
        pct = (100.0 * pos / total) if total else 0.0
        tail = state.get("loss_history_tail") or []
        recent = tail[-8:] if tail else []
        avg_recent = sum(recent) / len(recent) if recent else float("nan")

        lines.append(f"- **التقدّم:** `{pos:,}` / `{total:,}` جملة (**{pct:.1f}%**)")
        lines.append(f"- **عدد التشغيلات (runs):** {state.get('runs', '—')}")
        lines.append(f"- **إجمالي الجمل المُشاهَدة:** {state.get('total_sentences_seen', '—'):,}"
                     if isinstance(state.get("total_sentences_seen"), int)
                     else f"- **إجمالي الجمل المُشاهَدة:** {state.get('total_sentences_seen', '—')}")
        lines.append(f"- **إصدار التوكنايزر:** `{state.get('tokenizer_version', '—')}`")
        lines.append(f"- **إصدار النموذج:** `{state.get('model_version', '—')}`")
        lines.append(f"- **آخر حزمة:** size={state.get('last_pack_size', '—')} × "
                     f"packs={state.get('last_packs_per_run', '—')} | "
                     f"elapsed={state.get('last_elapsed_s', '—')} ث | "
                     f"RAM={state.get('last_avail_ram_gb', '—')} GB")
        if recent:
            lines.append(f"- **آخر خسائر (loss):** {', '.join(f'{x:.3f}' for x in recent)}")
            lines.append(f"- **متوسط آخر {len(recent)}:** `{avg_recent:.3f}`")
        if pos >= total and total > 0:
            lines.append("")
            lines.append("✅ **التدريب مكتمل (100%)** حسب ملف الحالة.")
        else:
            remaining = max(0, (total or 0) - pos)
            lines.append(f"- **المتبقي تقريباً:** {remaining:,} جملة")
    else:
        lines.append("⚠️ لا يوجد ملف حالة `ckg_train_state_v3.json` — لم يبدأ تدريب v3 بعد أو الملف غير متاح.")

    lines.append("")
    lines.append("### 📁 مصادر البيانات")
    for label, path in [
        ("ckg_sentences_v3.pkl", SENTENCES_V3),
        ("ckg_sentences_v2.pkl", SENTENCES_V2),
        ("ckg_sentences.pkl", SENTENCES_V1),
    ]:
        n = _count_sentences(path)
        size_mb = path.stat().st_size / (1024 * 1024) if path.is_file() else 0
        if n is not None:
            lines.append(f"- `{label}`: **{n:,}** جملة ({size_mb:.1f} MB)")
        else:
            lines.append(f"- `{label}`: غير متاح")

    lines.append("")
    lines.append("### 🧠 أوزان النموذج")
    if WEIGHTS_DIR.is_dir():
        files = list(WEIGHTS_DIR.glob("*"))
        lines.append(f"- المجلد موجود: `{WEIGHTS_DIR.relative_to(ROOT)}` ({len(files)} ملف)")
        for p in sorted(files)[:12]:
            try:
                sz = p.stat().st_size / (1024 * 1024)
                lines.append(f"  - `{p.name}` ({sz:.2f} MB)")
            except Exception:
                lines.append(f"  - `{p.name}`")
        if len(files) > 12:
            lines.append(f"  - … و{len(files) - 12} ملف إضافي")
    else:
        lines.append(f"- ⚠️ مجلد الأوزان غير موجود: `{WEIGHTS_DIR.relative_to(ROOT)}` "
                     "(متوقع — المجلد مُتجاهَل في git ويُنشأ محلياً عند أول تدريب)")

    avail = _available_ram_gb()
    lines.append("")
    lines.append(f"### 💻 موارد البيئة الحالية")
    lines.append(f"- الرام المتاحة تقريباً: **{avail:.2f} GB**")
    if avail < _MIN_AVAIL_RAM_GB:
        lines.append("- ⚠️ الرام منخفضة — تشغيل تدريب كامل قد يفشل (OOM). يُفضّل بيئة ≥ 3.5 GB.")
    return "\n".join(lines)


def recommend_config() -> str:
    """اقتراح إعدادات حزم ومعلمات حسب الرام الحالية."""
    avail = _available_ram_gb()
    lines = [
        "## ⚙️ اقتراح إعدادات التدريب",
        "",
        f"- الرام المتاحة: **{avail:.2f} GB**",
        "",
    ]

    # نفس منطق train_batch_v3 تقريباً
    safety = 0.35
    ref_pack, ref_peak = 80, 2.93
    budget = max(0.0, avail - safety)
    if budget < 1.2:
        pack = 0
        packs = 0
        advice = "الرام غير كافية لتشغيل تدريب آمن. استخدم جهازاً برام ≥ 3.5 GB أو قلل النموذج مؤقتاً."
    else:
        ratio = budget / ref_peak
        pack = max(4, min(80, int(ref_pack * ratio)))
        pack = max(4, (pack // 4) * 4)
        if avail >= 3.2:
            packs = 8
        elif avail >= 2.4:
            packs = 4
        elif avail >= 1.8:
            packs = 2
        else:
            packs = 1
        advice = "الإعدادات أدناه مناسبة لهذه الجلسة. يمكن تجاوزها بمتغيرات البيئة NSM_PACK_SIZE / NSM_PACKS_PER_RUN."

    lines.append(f"- **PACK_SIZE المقترح:** `{pack}`")
    lines.append(f"- **PACKS_PER_RUN المقترح:** `{packs}`")
    lines.append(f"- **Tokenizer الافتراضي:** `word` (word-v1) — متوافق مع الحالة الحالية")
    lines.append("")
    lines.append("### معلمات فائقة مرجعية (من الكود الحالي)")
    lines.append("- D_MODEL=2304 | N_HEADS=16 | N_LAYERS=16 | D_FF=8384 | MAX_SEQ_LEN=128")
    lines.append("- VOCAB_SIZE=8192 | LEARNING_RATE=1e-4 | CLIP_GRAD=1.0")
    lines.append("")
    lines.append(f"**التوصية:** {advice}")
    lines.append("")
    lines.append("لتشغيل يدوي:")
    if pack > 0:
        lines.append(
            f"```bash\nNSM_PACK_SIZE={pack} NSM_PACKS_PER_RUN={packs} python3 train_batch_v3.py\n```"
        )
    else:
        lines.append("```bash\n# انتظر موارد كافية أو شغّل على بيئة أقوى\npython3 train_batch_v3.py\n```")
    return "\n".join(lines)


def analyze_loss_trend() -> str:
    """تحليل بسيط لاتجاه الخسارة من ملف الحالة."""
    state = _load_json(STATE_FILE)
    if not state:
        return "⚠️ لا يوجد ملف حالة لتحليل الخسارة."

    tail = list(state.get("loss_history_tail") or [])
    if len(tail) < 4:
        return f"البيانات غير كافية للتحليل (يوجد {len(tail)} نقطة فقط). شغّل خطوات تدريب إضافية أولاً."

    first = sum(tail[: max(1, len(tail) // 4)]) / max(1, len(tail) // 4)
    last = sum(tail[-max(1, len(tail) // 4) :]) / max(1, len(tail) // 4)
    overall_avg = sum(tail) / len(tail)
    delta = last - first
    trend = "📉 تحسّن (انخفاض)" if delta < -0.05 else ("📈 تدهور (ارتفاع)" if delta > 0.05 else "➡️ مستقر تقريباً")

    lines = [
        "## 📈 تحليل اتجاه الخسارة (Loss)",
        "",
        f"- عدد نقاط السجل: **{len(tail)}**",
        f"- متوسط الربع الأول: `{first:.3f}`",
        f"- متوسط الربع الأخير: `{last:.3f}`",
        f"- المتوسط الكلي: `{overall_avg:.3f}`",
        f"- التغيّر (آخر − أول): `{delta:+.3f}` → **{trend}**",
        "",
        "آخر 12 قيمة:",
        "```",
        ", ".join(f"{x:.3f}" for x in tail[-12:]),
        "```",
        "",
    ]
    if last > 6.5:
        lines.append("⚠️ الخسارة ما زالت مرتفعة نسبياً — قد تحتاج مزيداً من الحقب أو مراجعة معدل التعلم/التوكنايزر.")
    elif last < 4.0:
        lines.append("✅ مستوى خسارة منخفض نسبياً مقارنة بسجل المشروع السابق.")
    else:
        lines.append("الخسارة في نطاق متوسط مقبول لمرحلة التدريب الحالية.")
    return "\n".join(lines)


def run_training_step(
    packs: int = 1,
    pack_size: Optional[int] = None,
    dry_run: bool = False,
) -> str:
    """
    تشغيل خطوة تدريب واحدة عبر train_batch_v3.py بأمان محدود.

    - يرفض التشغيل إذا كانت الرام منخفضة جداً (ما لم dry_run=True).
    - يحدّ عدد الحزم بـ _MAX_PACKS_PER_AGENT_RUN.
    - لا يعيد ضبط التدريب من الصفر أبداً من هنا.
    """
    packs = max(1, min(int(packs), _MAX_PACKS_PER_AGENT_RUN))
    avail = _available_ram_gb()

    if not TRAIN_SCRIPT.is_file():
        return f"❌ سكربت التدريب غير موجود: `{TRAIN_SCRIPT.name}`"

    if dry_run:
        return (
            f"🧪 **وضع تجريبي (dry-run)** — لن يُشغَّل التدريب فعلياً.\n\n"
            f"- الحزم المطلوبة: {packs}\n"
            f"- pack_size: {pack_size or 'تلقائي'}\n"
            f"- الرام المتاحة: {avail:.2f} GB\n"
            f"- الأمر المقترح:\n"
            f"```bash\n"
            f"NSM_PACKS_PER_RUN={packs}"
            + (f" NSM_PACK_SIZE={pack_size}" if pack_size else "")
            + f" python3 {TRAIN_SCRIPT.name}\n```"
        )

    if avail < _MIN_AVAIL_RAM_GB:
        return (
            f"❌ **رُفض التشغيل:** الرام المتاحة ({avail:.2f} GB) أقل من الحد الآمن "
            f"({_MIN_AVAIL_RAM_GB} GB).\n"
            "النموذج ~120M باراميتر ويتطلب عادةً ≥ 3.5 GB لتجنب OOM.\n"
            "استخدم `dry_run` أو شغّل على بيئة أقوى، أو اطلب «اقترح إعدادات»."
        )

    env = os.environ.copy()
    env["NSM_PACKS_PER_RUN"] = str(packs)
    if pack_size is not None and pack_size > 0:
        env["NSM_PACK_SIZE"] = str(int(pack_size))

    # لا نمرّر NSM_RESET_TRAIN أبداً من الوكيل
    env.pop("NSM_RESET_TRAIN", None)

    lines = [
        f"🚀 بدء خطوة تدريب (حتى {packs} حزمة)…",
        f"- الرام المتاحة عند البدء: {avail:.2f} GB",
        "",
    ]

    try:
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=_DEFAULT_TIMEOUT_S,
        )
        elapsed = time.time() - t0
        out = (proc.stdout or "")[-4000:]
        err = (proc.stderr or "")[-1500:]
        lines.append(f"- رمز الخروج: `{proc.returncode}` | المدة: {elapsed:.1f} ث")
        if out.strip():
            lines.append("")
            lines.append("**مخرجات التدريب:**")
            lines.append("```")
            lines.append(out.strip())
            lines.append("```")
        if err.strip() and proc.returncode != 0:
            lines.append("")
            lines.append("**stderr:**")
            lines.append("```")
            lines.append(err.strip())
            lines.append("```")
        if proc.returncode == 0:
            lines.append("")
            lines.append("✅ انتهت الخطوة. حدّث الحالة عبر أمر «حالة التدريب».")
        else:
            lines.append("")
            lines.append("⚠️ انتهت الخطوة برمز غير صفري — راجع المخرجات أعلاه.")
    except subprocess.TimeoutExpired:
        lines.append(f"❌ انتهت المهلة ({_DEFAULT_TIMEOUT_S} ث). أوقف التدريب لحماية الموارد.")
    except Exception as e:
        lines.append(f"❌ فشل التشغيل: {type(e).__name__}: {e}")

    return "\n".join(lines)


def lifecycle_plan() -> str:
    """خطة دورة حياة كاملة مخصّصة لمشروع NSM."""
    state = _load_json(STATE_FILE) or {}
    pos = int(state.get("position", 0) or 0)
    n = _count_sentences(SENTENCES_V3) or 26375
    pct = 100.0 * pos / n if n else 0.0

    return f"""## 🗺️ خطة دورة حياة تدريب النموذج (NSM)

### 1) جمع ومعالجة البيانات
- المصدر الأساسي: `ckg_sentences_v3.pkl` ({n:,} جملة مستخرجة من CKG/قرآن).
- مصادر إضافية: v2 / العام (`ckg_sentences_general_ar.pkl`).
- التنظيف: يتم مسبقاً عبر خط الأنابيب (تطبيع عربي + جذور + ربط CKG).
- **الحالة الحالية للبيانات:** جاهزة للتدريب.

### 2) اختيار النموذج وضبط المعلمات
- النموذج: `ArabicTransformer` v3 (~120M، numpy أساسي).
- التوكنايزر الحالي في الحالة: `{state.get('tokenizer_version', 'word-v1')}`.
- المعلمات: D_MODEL=2304، 16 طبقة، LR=1e-4.
- استخدم أمر **«اقترح إعدادات»** لضبط PACK_SIZE حسب الرام.

### 3) إدارة التدريب
- السكربت: `train_batch_v3.py` (يستأنف تلقائياً من `ckg_train_state_v3.json`).
- التقدّم الحالي: **{pos:,}/{n:,} ({pct:.1f}%)**.
- الأوامر المقترحة:
  - `حالة التدريب` → مراقبة
  - `شغّل خطوة تدريب` → تشغيل محدود وآمن
  - `تحليل الخسارة` → تقييم اتجاه التعلّم
- الإيقاف المبكر: راقب انخفاض الخسارة؛ عند الاستقرار يمكن إيقاف الحقب الإضافية.

### 4) التقييم والاختبار
- تتبع `loss_history_tail` في ملف الحالة.
- ربط لاحق بـ Faithfulness Verifier (`ai/nsm_answer_verifier.py`) للتحقق من تأسيس الإجابات.
- اختبارات وحدة موجودة: `ai/test_arabic_transformer_*.py`، `ai/test_knowledge_trainer_ckg.py`.

### 5) النشر والحفظ
- الأوزان تُحفظ محلياً في `models/transformer_ckg_v3/` (مُتجاهَلة في git عمداً).
- ملف الحالة فقط يُرفع عادةً لتسجيل التقدّم.
- الواجهة الحية تستهلك النموذج عبر `ReasoningPipeline` بعد توفر الأوزان.

---
**ملاحظة أمان:** هذا الوكيل لا يعيد التدريب من الصفر ولا يرفع أوزاناً كبيرة إلى git.
التدريب الكامل إلى 100% يحتاج بيئة برام كافية (≥ 3.5 GB متاحة مفضّل).
"""


def handle_training_command(user_input: str) -> Optional[str]:
    """
    يفسّر أوامر عربية شائعة لوكيل التدريب.
    يعيد نص النتيجة إن طابق أمراً، وإلا None لتمرير الرسالة للمحادثة العادية.
    """
    text = (user_input or "").strip()
    if not text:
        return None
    low = text.lower()

    # حالة / تقرير
    if re.search(r"(حالة|وضع|تقرير).{0,12}(تدريب|النموذج|الشبكة)|training\s*status|status", text, re.I):
        return get_training_status()
    if re.search(r"^(حالة التدريب|وضع التدريب|تقرير التدريب)\s*$", text.strip()):
        return get_training_status()

    # اقتراح إعدادات
    if re.search(
        r"(اقترح.{0,20}إعدادات|إعدادات.{0,15}تدريب|توصية.{0,15}(تدريب|حزم|معلمات)|"
        r"ضبط.{0,15}(معلمات|هايبر|حزم)|recommend|hyperparam)",
        text,
        re.I,
    ):
        return recommend_config()

    # تحليل خسارة
    if re.search(r"(تحليل|اتجاه).{0,10}(خسارة|loss)|loss\s*trend", text, re.I):
        return analyze_loss_trend()

    # خطة دورة حياة
    if re.search(r"(خطة|دورة\s*حياة|lifecycle).{0,15}(تدريب|نموذج)?|lifecycle", text, re.I):
        return lifecycle_plan()

    # تشغيل خطوة (مع دعم dry-run)
    dry = bool(re.search(r"تجريب|dry\s*-?run|بدون\s*تشغيل|محاكاة", text, re.I))
    if re.search(
        r"(شغّل|شغل|ابدأ|نفّذ|نفذ).{0,15}(تدريب|خطوة|حزمة)|run\s*train|train\s*step",
        text,
        re.I,
    ):
        packs = 1
        m = re.search(r"(\d+)\s*(حزم|حزمة|packs?)", text)
        if m:
            packs = int(m.group(1))
        pack_size = None
        m2 = re.search(r"(?:حجم|size)\s*[:=]?\s*(\d+)", text, re.I)
        if m2:
            pack_size = int(m2.group(1))
        return run_training_step(packs=packs, pack_size=pack_size, dry_run=dry)

    # أوامر قصيرة مباشرة
    if text.strip() in ("حالة", "status"):
        return get_training_status()
    if text.strip() in ("إعدادات", "config"):
        return recommend_config()
    if text.strip() in ("خسارة", "loss"):
        return analyze_loss_trend()
    if text.strip() in ("خطة", "plan"):
        return lifecycle_plan()

    return None
