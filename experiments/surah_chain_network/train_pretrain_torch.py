"""
تدريب Pre-training لـ SurahChain LM على بيانات نصية عامة من الإنترنت
(بدون CKG) — مع دعم الاستكمال من آخر checkpoint (resume).

المصدر: Jr23xd23/ArabicText-Large عبر prepare_pretrain_data.py

الاستخدام:
  # سعة أكبر (موصى به للتقوية)
  SCN_PRESET=medium SCN_FRESH=1 \\
    python experiments/surah_chain_network/train_pretrain_torch.py

  # أو يدوياً
  SCN_N=30000 SCN_D_MODEL=128 SCN_N_PRE=2 SCN_N_POST=2 \\
  SCN_FRESH=1 SCN_BATCH=16 \\
    python experiments/surah_chain_network/train_pretrain_torch.py

قاعدة: شبكة الـ114 لا تُدمج ولا تُغيَّر أبعادها افتراضياً (من surah_layer_dims.json).
التقوية عبر d_model + الانتباه قبل/بعد السلسلة فقط.

presets:
  small   → d=128, سلسلة 114 كما هي, pre/post=2
  medium  → d=256, سلسلة 114 كما هي, pre/post=4
  large   → d=512, سلسلة 114 كما هي, pre/post=6
  xlarge  → معمارية نماذج كبيرة: d=8192 | FFN=28672 (SwiGLU) | GQA
            (64 رأس استعلام + 8 KV × 128/رأس) | RMSNorm Pre-Norm |
            pre/post=8 — سلسلة 114 تبقى كما هي والتوسع أثناء التدريب
            (يتطلب GPU ذاكرة ≥ 32GB + torch.compile)
"""
from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))

from hybrid_experiment_torch import HybridExperimentModelTorch

# ----- إعدادات أساسية -----
N = int(os.environ.get("SCN_N", "8000"))
EPOCHS = int(os.environ.get("SCN_EPOCHS", "15"))
BATCH = int(os.environ.get("SCN_BATCH", "16"))
D_MODEL = int(os.environ.get("SCN_D_MODEL", "128"))
N_HEADS = int(os.environ.get("SCN_N_HEADS", "8"))
N_KV_HEADS = int(os.environ.get("SCN_N_KV_HEADS", "0")) or None  # 0 ⇒ MHA عادي
D_HEAD = int(os.environ.get("SCN_D_HEAD", "0")) or None
D_FF = int(os.environ.get("SCN_D_FF", "0")) or None
N_PRE = int(os.environ.get("SCN_N_PRE", "2"))
N_POST = int(os.environ.get("SCN_N_POST", "2"))
BASE_LR = float(os.environ.get("SCN_LR", "1e-3"))
MAX_LEN = int(os.environ.get("SCN_MAX_LEN", "96"))
COMPILE = os.environ.get("SCN_COMPILE", "0") == "1"
# ── توفير ذاكرة للحجوم الكبيرة ───────────────────────────────────────────
# USE_8BIT_ADAM:Adam-8bit (bitsandbytes) يخفض حالة المحسّن ≈40% من VRAM.
# GRAD_ACCUM>1 ينفّذ micro-batches متعددة قبل خطوة محسّن واحدة —
# يضاعف حجم الدفعة الفعلي (BATCH×GRAD_ACCUM) دون زيادة ذاكرة activations.
USE_8BIT_ADAM = os.environ.get("SCN_USE_8BIT_ADAM", "0") == "1"
GRAD_ACCUM = max(1, int(os.environ.get("SCN_GRAD_ACCUM", "1")))
# ── إشارة التوقف الآمن (زر التوقف في واجهة Streamlit / Kaggle) ────────────
# وجود الملف يعني: أكمل العصر الجاري واحفظ checkpoint ثم توقف نظيفًا.
USE_QK_NORM = os.environ.get("SCN_QK_NORM", "1") == "1"
USE_GATED_ATTN = os.environ.get("SCN_GATED_ATTN", "1") == "1"
CHAIN_SCALE = float(os.environ.get("SCN_CHAIN_SCALE", "1"))
WARMUP_RATIO = 0.1
# توسيع احترافي (غير عشوائي):
# - صبر أطول قبل اعتبار الهضبة
# - لا توسيع في بداية الجولة (بعد resume/قفزة LR)
# - تهدئة بين كل توسيعين
# - شرط «هضبة حقيقية» على نافذة عصور وليس مجرد أسوأ من best مرة أو مرتين
PATIENCE = int(os.environ.get("SCN_EXPAND_PATIENCE", "6"))
MAX_EXPANDS = int(os.environ.get("SCN_MAX_EXPANDS", "3"))
EXPAND_MIN_EPOCH = int(os.environ.get("SCN_EXPAND_MIN_EPOCH", "12"))  # عمر النموذج الأدنى
EXPAND_COOLDOWN = int(os.environ.get("SCN_EXPAND_COOLDOWN", "8"))  # عصور بين توسيعين
EXPAND_WARMUP_RUN = int(os.environ.get("SCN_EXPAND_WARMUP_RUN", "4"))  # عصور بعد بدء الجولة
EXPAND_FLAT_REL = float(os.environ.get("SCN_EXPAND_FLAT_REL", "0.015"))  # هضبة: تذبذب نسبي صغير
STOP_PATIENCE = int(os.environ.get("SCN_STOP_PATIENCE", "0"))
UNTIL_END = os.environ.get("SCN_UNTIL_END", "0") == "1"
# ── Pre-tokenization (تسريع CPU 2-4×): SCN_PRE_TOKENIZE=1 يشفر كل النصوص
#     مرة واحدة في البداية ويحفظ التسلسلات (tokens.pkl) — في كل خطوة
#     يُبنى الـbatch من تسلسلات جاهزة بدل إعادة tokenize نص خام ──
PRE_TOKENIZE = os.environ.get("SCN_PRE_TOKENIZE", "0").strip() == "1"
TOKEN_CACHE = os.environ.get("SCN_TOKEN_CACHE", "").strip()
FRESH = os.environ.get("SCN_FRESH", "0") == "1"
RESUME_PATH = os.environ.get("SCN_RESUME_PATH", "").strip()
# ── NSM resume ذكي: "auto" يستأنف تلقائيًا من آخر checkpoint مرفوع على GitHub
#     حتى مع SCN_FRESH=1 — حتى لا يضيع التدريب عند انقطاع الجلسة ──
SCN_RESUME = os.environ.get("SCN_RESUME", "").strip().lower()
CHECKPOINT_EVERY = int(os.environ.get("SCN_CHECKPOINT_EVERY", "2"))  # حفظ مرفوع كل كذا عصر
# حد زمني للجلسة (ساعات) — أقل من حد Kaggle 12س حتى لا يُقطع العمل دون حفظ
MAX_HOURS = float(os.environ.get("SCN_MAX_HOURS", "0") or 0)  # 0 = بلا حد
# حفظ وطباعة أثناء العصر (حتى لا تضيع ساعات بلا checkpoint إذا أُلغي الكيرنل)
SAVE_EVERY_STEPS = int(os.environ.get("SCN_SAVE_EVERY_STEPS", "0") or 0)  # 0 = فقط نهاية العصر
LOG_EVERY_STEPS = int(os.environ.get("SCN_LOG_EVERY_STEPS", "0") or 0)
# ── رفع سريع أول الجولة: epoch الأول والثاني يُرفعان فورًا —
#     أقصى حماية من موت مبكر قبل أول رفع دوري ──
FIRST_FAST = os.environ.get("SCN_FIRST_FAST", "1").strip().lower() in ("1", "true", "yes")
UPLOAD_RETRIES = max(1, int(os.environ.get("SCN_UPLOAD_RETRIES", "3")))

# ----- presets للسعة -----
PRESET = os.environ.get("SCN_PRESET", "").strip().lower()
if PRESET == "small":
    D_MODEL, N_HEADS, N_PRE, N_POST = 128, 8, 2, 2
    CHAIN_SCALE = 1.0
    N = max(N, 30000)
elif PRESET == "medium":
    D_MODEL, N_HEADS, N_PRE, N_POST = 256, 8, 4, 4
    # سلسلة 114 كما في surah_layer_dims.json — لا دمج ولا تغيير أبعاد
    CHAIN_SCALE = float(os.environ.get("SCN_CHAIN_SCALE", "1"))
    N = max(N, 60000)
    BATCH = int(os.environ.get("SCN_BATCH", "16"))
    BASE_LR = float(os.environ.get("SCN_LR", "5e-4"))
elif PRESET == "large":
    D_MODEL, N_HEADS, N_PRE, N_POST = 512, 8, 6, 6
    CHAIN_SCALE = float(os.environ.get("SCN_CHAIN_SCALE", "1"))
    N = max(N, 100000)
    BATCH = int(os.environ.get("SCN_BATCH", "8"))
    BASE_LR = float(os.environ.get("SCN_LR", "3e-4"))
    MAX_LEN = int(os.environ.get("SCN_MAX_LEN", "128"))
elif PRESET == "xlarge":
    # معمارية نماذج كبيرة: GQA (64 رأس استعلام + 8 KV heads × 128/رأس)،
    # SwiGLU FFN (3.5× d_model = 28672)، RMSNorm Pre-Norm.
    # سلسلة السور الـ114 تبقى كما هي والتوسع يتم أثناء التدريب.
    D_MODEL, N_HEADS, N_PRE, N_POST = 8192, 64, 8, 8
    N_KV_HEADS, D_HEAD = 8, 128
    D_FF = int(os.environ.get("SCN_D_FF", "28672"))
    CHAIN_SCALE = float(os.environ.get("SCN_CHAIN_SCALE", "1"))
    N = max(N, int(os.environ.get("SCN_N", "100000")))
    BATCH = int(os.environ.get("SCN_BATCH", "1"))
    BASE_LR = float(os.environ.get("SCN_LR", "1e-4"))
    MAX_LEN = int(os.environ.get("SCN_MAX_LEN", "64"))
    COMPILE = os.environ.get("SCN_COMPILE", "0") == "1"
    # توفير ذاكرة إلزامي لـxlarge على Kaggle:
    # Adam-8bit + تجميع تدرجات — بدونهما يتجاوز الطلب 16GB (T4) بكثير.
    USE_8BIT_ADAM = os.environ.get("SCN_USE_8BIT_ADAM", "1") == "1"
    GRAD_ACCUM = max(1, int(os.environ.get("SCN_GRAD_ACCUM", "2")))
    # استقرار: التوسيع الذاتي أسمح عند هذا الحجم
    MAX_EXPANDS = int(os.environ.get("SCN_MAX_EXPANDS", "5"))
    # ── وضع TPU v5e-8 (PyTorch/XLA) ─────────────────────────────────────
    # SCN_TPU=1: النقل إلى Kaggle TPU v5e-8 (8 شرائح × 16GB = 128GB HBM).
    # - bitsandbytes غير مدعوم على TPU → AdamW عادي مع bf16
    # - XLA JIT compiler يجمع الرسم بيانيًا تلقائيًا → COMPILE=0
    # - دفعات أكبر ممكنة: BATCH أعلى + تجميع تدرجات أوسع (effective batch أضخم)
    # - d_model=8192: ~15GB أوزان + ~31GB حالة محسّن FP32 ≈ 46GB —
    #   لا يكفي على T4 (16GB) لكن مريح على شريحة TPU v5e واحدة (16GB حرة
    #   بعد الأوزان عبر bf16 + 8-bit-like packing أو micro-batch صغيرة)
    SCN_TPU = os.environ.get("SCN_TPU", "0") == "1"
    if SCN_TPU:
        # Adam-8bit لا يعمل على TPU — البديل: bf16 للمحسّن والأوزان
        USE_8BIT_ADAM = False
        COMPILE = False
        # دفعات أكبر على TPU: micro-batch per step أعلى، وتجميع أوسع
        BATCH = int(os.environ.get("SCN_BATCH", "2"))
        GRAD_ACCUM = max(1, int(os.environ.get("SCN_GRAD_ACCUM", "8")))
        print(
            f"✅ وضع TPU (SCN_TPU=1): 8-bit Adam معطّل (غير مدعوم على XLA) | "
            f"torch.compile معطّل (XLA JIT) | BATCH={BATCH} × "
            f"accum={GRAD_ACCUM} → effective_batch={BATCH * GRAD_ACCUM}"
        )
elif PRESET in ("xlarge-6144", "xlarge6144"):
    # معمارية نماذج كبيرة مخفّفة: d=6144 (48 رأس × 128/رأس، 6 KV heads، FFN=21504)
    # ~4.2B params | bf16 أوزان ≈ 8.4GB | ~25GB إجمالي تدريب على TPU
    # أخف من d=8192 بـ 44% → يعمل بشكل مريح على شريحة TPU v5e واحدة
    D_MODEL, N_HEADS, N_PRE, N_POST = 6144, 48, 8, 8
    N_KV_HEADS, D_HEAD = 6, 128
    D_FF = int(os.environ.get("SCN_D_FF", "21504"))
    CHAIN_SCALE = float(os.environ.get("SCN_CHAIN_SCALE", "1"))
    N = max(N, int(os.environ.get("SCN_N", "25000")))
    BATCH = int(os.environ.get("SCN_BATCH", "4"))
    BASE_LR = float(os.environ.get("SCN_LR", "1e-4"))
    MAX_LEN = int(os.environ.get("SCN_MAX_LEN", "64"))
    COMPILE = os.environ.get("SCN_COMPILE", "0") == "1"
    USE_8BIT_ADAM = os.environ.get("SCN_USE_8BIT_ADAM", "1") == "1"
    GRAD_ACCUM = max(1, int(os.environ.get("SCN_GRAD_ACCUM", "4")))
    MAX_EXPANDS = int(os.environ.get("SCN_MAX_EXPANDS", "5"))
    SCN_TPU = os.environ.get("SCN_TPU", "0") == "1"
    if SCN_TPU:
        USE_8BIT_ADAM = False
        COMPILE = False
        # d=6144 أخف → دفعات أكبر على TPU
        BATCH = int(os.environ.get("SCN_BATCH", "8"))
        GRAD_ACCUM = max(1, int(os.environ.get("SCN_GRAD_ACCUM", "4")))
        print(
            f"✅ وضع TPU d=6144 (SCN_TPU=1): bf16 | BATCH={BATCH} × "
            f"accum={GRAD_ACCUM} → effective_batch={BATCH * GRAD_ACCUM}"
        )

# تجاوز اختياري بعد الـpreset (لتقليل الذاكرة على TPU)
if os.environ.get("SCN_N_PRE", "").strip():
    N_PRE = int(os.environ["SCN_N_PRE"])
if os.environ.get("SCN_N_POST", "").strip():
    N_POST = int(os.environ["SCN_N_POST"])
if os.environ.get("SCN_CHAIN_SCALE", "").strip() and PRESET:
    CHAIN_SCALE = float(os.environ["SCN_CHAIN_SCALE"])

if N_KV_HEADS is not None and N_HEADS % N_KV_HEADS != 0:

    raise ValueError(f"n_heads ({N_HEADS}) يجب أن يقبل القسمة على n_kv_heads ({N_KV_HEADS})")

if UNTIL_END:
    EPOCHS = max(EPOCHS, int(os.environ.get("SCN_MAX_EPOCHS", "80")))
    if STOP_PATIENCE <= 0:
        STOP_PATIENCE = int(os.environ.get("SCN_STOP_PATIENCE", "6"))

# checkpoints منفصلة لكل سعة حتى لا تُOverwrite تجربة d=128
TAG = os.environ.get("SCN_TAG", "").strip()
if not TAG:
    TAG = f"d{D_MODEL}_s{str(CHAIN_SCALE).replace('.', 'p')}"
CKPT_DIR = _HERE / "checkpoints"
CKPT_BEST = CKPT_DIR / f"best_pretrain_{TAG}.pt"
CKPT_LATEST = CKPT_DIR / f"latest_pretrain_{TAG}.pt"
VOCAB_PATH = _HERE / f"tokenizer_vocab_pretrain_{TAG}.json"
STATE_FILE = CKPT_DIR / f"pretrain_state_{TAG}.json"
PROGRESS_FILE = CKPT_DIR / f"progress_{TAG}.json"
PRETRAIN_CACHE = _HERE / "data" / "pretrain_sentences.pkl"
# ── إشارة التوقف الآمن (زر التوقف في واجهة Streamlit / Kaggle) ────────────
# وجود الملف يعني: أكمل العصر الجاري واحفظ checkpoint ثم توقف نظيفًا —
# ثم يُعيد _upload_checkpoint() آخر حالة إلى GitHub لاستئناف آمن لاحقًا.
STOP_SIGNAL_FILE = CKPT_DIR / "STOP"
# توافق مع الملفات القديمة عند small/d128
if TAG in ("d128_s1", "d128_s1p0") or (D_MODEL == 128 and CHAIN_SCALE == 1.0 and not os.environ.get("SCN_TAG")):
    if not os.environ.get("SCN_TAG"):
        # ابقِ الأسماء القديمة لـ d128 الافتراضي إن وُجدت
        _old_best = CKPT_DIR / "best_pretrain_torch.pt"
        _old_latest = CKPT_DIR / "latest_pretrain_torch.pt"
        if _old_latest.exists() or _old_best.exists():
            CKPT_BEST = _old_best
            CKPT_LATEST = _old_latest
            STATE_FILE = CKPT_DIR / "pretrain_torch_state.json"
            PROGRESS_FILE = CKPT_DIR / "progress_torch.json"
            VOCAB_PATH = _HERE / "tokenizer_vocab_pretrain.json"


def load_pretrain_sentences(max_n: int) -> list:
    """يحمّل من الكاش؛ إن لم يوجد أو كان أصغر من المطلوب يشغّل التحضير.

    وضع العامل الموزّع: SCN_WORKER_CACHE=<مسار cache.pkl منفصل للعامل>
    يتجاوز الكاش المشترك كليًا حتى لا يحصل كل عامل على نفس البيانات.

    SCN_RESPECT_CURSOR=1: يتجاوز الجمل التي سجّلها training_cursor.json كمستخدمة
    حتى لا يُعاد تدريب نفس الشريحة في جولة لاحقة (بعد دمج بيانات جديدة).
    """
    worker_cache = os.environ.get("SCN_WORKER_CACHE", "").strip()
    if worker_cache:
        from pathlib import Path as _P
        wcp = _P(worker_cache)
        if wcp.is_file():
            try:
                with open(wcp, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, list):
                    out = [s.strip() for s in data if isinstance(s, str) and len(s.strip()) >= 20]
                    random.Random(0).shuffle(out)
                    print(f"كاش العامل ({worker_cache}): {len(out)} مقطع")
                    return out[:max_n]
            except Exception as e:
                print(f"كاش العامل فشل: {e}")
        return []  # لا fallback للكاش المشترك — العامل يحمّل بياناته بنفسه
    if PRETRAIN_CACHE.exists():
        with open(PRETRAIN_CACHE, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            out = [s.strip() for s in data if isinstance(s, str) and len(s.strip()) >= 20]
            # احترام مؤشر التدريب: تخطّي الجمل المستخدمة سابقاً
            if os.environ.get("SCN_RESPECT_CURSOR", "0") == "1":
                cursor_path = _HERE / "data" / "training_cursor.json"
                used = 0
                if cursor_path.exists():
                    try:
                        used = int(json.loads(cursor_path.read_text(encoding="utf-8")).get("used_count") or 0)
                    except Exception:
                        used = 0
                if used > 0 and used < len(out):
                    print(f"مؤشر التدريب: تخطّي أول {used} جملة مستخدمة سابقاً")
                    out = out[used:]
                elif used >= len(out):
                    print(f"تحذير: used_count={used} >= المتاح {len(out)} — استخدام كامل المجموعة")
            random.Random(0).shuffle(out)
            if len(out) >= max_n:
                return out[:max_n]
            print(f"كاش موجود لكن أصغر من المطلوب ({len(out)} < {max_n}) — توسيع...")

    print("تحضير/توسيع بيانات Pre-training...")
    from prepare_pretrain_data import load_and_prepare, CACHE_FILE, CACHE_DIR

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["SCN_N"] = str(max_n)
    sentences = load_and_prepare(max_n)
    if PRETRAIN_CACHE.exists():
        try:
            with open(PRETRAIN_CACHE, "rb") as f:
                old = pickle.load(f)
            if isinstance(old, list):
                seen = set(sentences)
                for s in old:
                    if isinstance(s, str) and s not in seen and len(s.strip()) >= 20:
                        sentences.append(s.strip())
                        seen.add(s)
        except Exception:
            pass
    random.Random(0).shuffle(sentences)
    sentences = sentences[:max_n]
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(sentences, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"حُفظ الكاش: {CACHE_FILE} ({len(sentences)} مقطع)")
    return sentences


def _write_progress(obj: Dict) -> None:
    """NSM Live Logs: يكتب تقدم التدريب إلى ملف JSON كل عصر (مع flush فوري)."""
    try:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        entry = {**obj, "updated_at": time.time()}
        tmp = PROGRESS_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, default=str)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        try:
            tmp.replace(PROGRESS_FILE)
        except Exception:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, default=str)
    except Exception as e:
        print("[progress] write skipped:", e)


def read_progress(path: Path | None = None) -> Dict | None:
    """NSM Live Logs: يقرأ آخر تقدم مسجّل — للعرض الحي في Streamlit."""
    if path is None:
        base = Path(__file__).resolve().parent.parent / "surah_chain_network" / "checkpoints"
        path = base / "progress_torch.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _upload_checkpoint(ep: int) -> None:
    """رفع checkpoint الحالي + state إلى GitHub (فرع main) — استئناف آمن.

    يعمل حتى خارج repo (clone مؤقت في /tmp) — مثالي لـ Kaggle kernels.
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    if not token:
        print("[checkpoint] لا GITHUB_TOKEN — تخطي الرفع الدوري")
        return
    repo = os.environ.get("SCN_REPO", "aliahmed369000000-ai/Neural-Service-Mesh")
    branch = os.environ.get("SCN_BRANCH", "main")
    tmp = Path("/tmp/nsm_ckpt_push")
    import shutil
    import subprocess

    shutil.rmtree(str(tmp), ignore_errors=True)
    # ── إعادة محاولة للرفع: الكيرنل قد يموت أثناء clone — retry مع backoff ──
    last_err = None
    for attempt in range(1, UPLOAD_RETRIES + 1):
        last_err = _upload_checkpoint_once(token, repo, branch, tmp, ep, attempt)
        if last_err is None:
            break
        import shutil as _sh
        _sh.rmtree(str(tmp), ignore_errors=True)
        if attempt < UPLOAD_RETRIES:
            wait = 5 * attempt  # 5s → 10s → 15s
            print(f"[checkpoint] إعادة محاولة رفع #{attempt + 1}/{UPLOAD_RETRIES} بعد {wait}s…")
            time.sleep(wait)
    if last_err is not None:
        print(f"[checkpoint] استُنفدت {UPLOAD_RETRIES} محاولات — لن يُفقد التدريب، سيُرفع في الجولة التالية")


def _upload_checkpoint_once(token: str, repo: str, branch: str, tmp: Path,
                            ep: int, attempt: int = 1) -> str | None:
    """محاولة واحدة للرفع — ترجع None عند النجاح أو نص خطأ."""
    import shutil
    import subprocess
    shutil.rmtree(str(tmp), ignore_errors=True)
    try:
        # إصلاح timeout: sparse checkout ضحل (فقط checkpoints) + timeout=900
        r = subprocess.run(
            ["git", "clone", "-q", "--depth", "1", "--single-branch",
             "--branch", branch, "--filter=blob:none", "--sparse",
             f"https://x-access-token:{token}@github.com/{repo}.git", str(tmp)],
            capture_output=True, text=True, timeout=900,
        )
        if r.returncode != 0:
            print(f"[checkpoint] clone فشل: {r.stderr[-200:]}")
            return "clone failed"
        # sparse: جلب checkpoints فقط (بدون باقي المستودع)
        subprocess.run(["git", "-C", str(tmp), "sparse-checkout", "set",
                        "experiments/surah_chain_network/checkpoints"],
                       capture_output=True, check=False)
        dest = tmp / "experiments" / "surah_chain_network" / "checkpoints"
        dest.mkdir(parents=True, exist_ok=True)
        # رفع state + vocab فقط — الأوزان الثقيلة (.pt) تُرفع عبر distributed_worker.py على فرع dist
        files = [
            (STATE_FILE, f"pretrain_state_{TAG}.json"),
            (VOCAB_PATH, f"tokenizer_vocab_pretrain_{TAG}.json"),
        ]
        for src, name in files:
            if src.is_file():
                shutil.copy(str(src), str(dest / name))
        subprocess.run(["git", "-C", str(tmp), "add", "-f", "experiments/surah_chain_network/checkpoints/"],
                       capture_output=True, check=False)
        st = subprocess.run(["git", "-C", str(tmp), "status", "--porcelain"],
                            capture_output=True, text=True)
        if not st.stdout.strip():
            print(f"[checkpoint] لا تغييرات (epoch {ep}) — تخطي")
            return
        subprocess.run(
            ["git", "-C", str(tmp), "-c", "user.email=nsm-bot@users.noreply.github.com",
             "-c", "user.name=NSM Bot", "commit", "-q", "-m",
             f"NSM: checkpoint epoch {ep} (surahchain {TAG})"],
            capture_output=True, check=False)
        r2 = subprocess.run(["git", "-C", str(tmp), "push", "-q", "origin", branch],
                            capture_output=True, text=True, timeout=600)
        if r2.returncode != 0:
            print(f"[checkpoint] push فشل: {r2.stderr[-200:]}")
            return "push failed"
        # ── تحقق فعلي من الرفع: ls-remote يطابق الـcommit المرفوع ──
        local_head = subprocess.run(
            ["git", "-C", str(tmp), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip()
        rv = subprocess.run(
            ["git", "-C", str(tmp), "ls-remote", "origin", f"refs/heads/{branch}"],
            capture_output=True, text=True, timeout=120)
        remote_head = rv.stdout.split("\t")[0].strip() if rv.returncode == 0 else ""
        if remote_head and remote_head == local_head:
            print(f"[checkpoint] رُفعت checkpoint epoch {ep} إلى GitHub ✅ (ls-remote مُطابَق)")
        elif remote_head:
            print(f"[checkpoint] رُفعت checkpoint epoch {ep} لكن remote={remote_head[:12]} ≠ local={local_head[:12]} — إعادة المحاولة")
            return "remote mismatch"
        else:
            print(f"[checkpoint] push نجح لكن ls-remote غير متاح — نعتبرها مرفوعة (attempt {attempt})")
        return None
    except Exception as e:
        print(f"[checkpoint] خطأ: {e}")
        return str(e)[:200]
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


def _fetch_uploaded_checkpoint() -> Path | None:
    """يسحب آخر checkpoint مرفوعة على GitHub (branch main) إلى مجلد checkpoints.

    يبحث عن: latest_pretrain_{TAG}.pt ثم latest_pretrain_torch.pt.
    يُستخدم مع SCN_RESUME=auto (يتجاوز SCN_FRESH=1).
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    if not token:
        print("[resume] لا GITHUB_TOKEN — لا يمكن سحب checkpoint المرفوعة")
        return None
    repo = os.environ.get("SCN_REPO", "aliahmed369000000-ai/Neural-Service-Mesh")
    branch = os.environ.get("SCN_BRANCH", "main")
    candidates = [
        f"experiments/surah_chain_network/checkpoints/latest_pretrain_{TAG}.pt",
        "experiments/surah_chain_network/checkpoints/latest_pretrain_torch.pt",
    ]
    import base64
    import urllib.request

    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    for rel in candidates:
        u = f"https://raw.githubusercontent.com/{repo}/{branch}/{rel}"
        dest = CKPT_DIR / rel.split("/")[-1]
        try:
            req = urllib.request.Request(u, headers={"Authorization": f"Basic {basic}", "User-Agent": "nsm-bot"})
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 200:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r.read())
            print(f"[resume] سُحبت checkpoint المرفوعة: {rel}")
            return dest
        except Exception as e:
            print(f"[resume] لم تُوجد {rel} ({type(e).__name__})")
            if dest.exists():
                dest.unlink()
    return None


def _pick_resume_path():
    if FRESH and SCN_RESUME != "auto":
        return None
    if SCN_RESUME == "auto":
        # استئناف تلقائي: المرفوع على GitHub له الأولوية (آخر تدريب ناجح)،
        # ثم المحلي (جولة سابقة منقطعة)
        remote = _fetch_uploaded_checkpoint()
        if remote is not None:
            return remote
    if RESUME_PATH:
        p = Path(RESUME_PATH)
        return p if p.exists() else None
    if CKPT_LATEST.exists():
        return CKPT_LATEST
    if CKPT_BEST.exists():
        return CKPT_BEST
    return None


_CRASH_STATE: Dict = {}


def _handle_fatal_signal(sig: int, frame) -> None:
    """موت مفاجئ (SIGTERM/SIGINT من Kaggle stop أو انقطاع) —
    احفظ checkpoint وارفع ما وصل إليه التدريب فورًا قبل الموت."""
    sig_name = "SIGTERM" if sig == 15 else "SIGINT"
    print(f"\n⚠ إشارة {sig_name} — موت مفاجئ! حفظ طوارئ ورفع...")
    m = _CRASH_STATE.get("model")
    train_meta = _CRASH_STATE.get("train_meta")
    ep = _CRASH_STATE.get("epoch")
    try:
        if m is not None and CKPT_DIR.exists():
            try:
                m.save(str(CKPT_LATEST), train_meta=train_meta)
                print("[emergency] حُفظت checkpoint الطوارئ ✅")
            except Exception as e:
                print(f"[emergency] حفظ فشل: {e}")
            try:
                _upload_checkpoint(ep or 0)
                print(f"[emergency] رُفعت checkpoint الطوارئ epoch {ep} ✅")
            except Exception as e:
                print(f"[emergency] رفع فشل: {e}")
    except Exception:
        pass
    sys.exit(0)


def _should_upload(ep: int, start_epoch: int, every: int = CHECKPOINT_EVERY,
                   fast: bool = FIRST_FAST) -> bool:
    """منطق قرار الرفع الدوري: رفع دوري كل `every` عصور + رفع سريع
    أول عصورين من الجولة (FIRST_FAST) للحماية من الموت المبكر."""
    into_run = ep - start_epoch
    if fast and 1 <= into_run <= 2:
        return True
    return every > 0 and into_run % every == 0


# ── معالج الموت المفاجئ: Kaggle يبعث SIGTERM قبل القتل —
#     التسجيل هنا module-level حتى يعمل فور استيراد السكربت
#     (حتى قبل بدء main) — بدون هذا handler يضيع كل ما بعد آخر رفع دوري ──
import signal as _sig
for _s in (_sig.SIGTERM, _sig.SIGINT):
    try:
        _sig.signal(_s, _handle_fatal_signal)
    except Exception:
        pass


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("SurahChain Pre-training (بيانات عامة — بدون CKG)")
    print("=" * 60)

    texts = load_pretrain_sentences(N)
    if len(texts) < 50:
        print(f"بيانات قليلة جداً ({len(texts)}). شغّل prepare_pretrain_data أولاً.")
        sys.exit(1)
    print(f"مقاطع التدريب: {len(texts)}")
    print(f"عينة: {texts[0][:120]}...")

    resume_path = _pick_resume_path()
    start_epoch = 0
    best = float("inf")
    history = []
    global_step = 0
    n_expands = 0
    expand_log = []
    total_seconds_prev = 0.0

    print(
        f"preset={PRESET or '-'} | tag={TAG} | d_model={D_MODEL} | "
        f"chain_scale={CHAIN_SCALE} | pre/post={N_PRE}/{N_POST} | N={N}"
    )
    print(f"checkpoints: {CKPT_LATEST.name} / {CKPT_BEST.name}")
    m = HybridExperimentModelTorch(
        d_model=D_MODEL,
        vocab_size=8192,
        lr=BASE_LR,
        n_heads=N_HEADS,
        n_pre=N_PRE,
        n_post=N_POST,
        compile_model=COMPILE,
        use_qk_norm=USE_QK_NORM,
        use_gated_attn=USE_GATED_ATTN,
        chain_scale=CHAIN_SCALE,
        n_kv_heads=N_KV_HEADS,
        d_head=D_HEAD,
        d_ff=D_FF,
    )
    core = getattr(m.model, "_orig_mod", m.model)
    gqa_str = f" | GQA: {core.n_kv_heads} KV heads × {core.d_head}/رأس" if core.n_kv_heads != core.n_heads else " | انتباه: MHA"
    print(f"QK-Norm={USE_QK_NORM} | Gated-Attention={USE_GATED_ATTN}{gqa_str}")
    pc = core.param_count()
    print(f"FFN={'SwiGLU' if hasattr(core.pre_blocks[0].ffn, 'w_up') else 'GELU'} | d_ff={core.d_ff} | params={pc['total']:,}")

    if resume_path is not None:
        print(f"استكمال من: {resume_path}")
        meta = m.load(str(resume_path), load_optimizer=True) or {}
        # توافق مع checkpoints قديمة: اقرأ state.json إن لم تُحفظ train_meta داخل .pt
        if not meta and STATE_FILE.exists():
            try:
                meta = json.loads(STATE_FILE.read_text())
                print(f"  → استُعيدت البيانات من {STATE_FILE.name} (صيغة قديمة)")
            except Exception:
                meta = {}
        start_epoch = int(meta.get("epoch") or meta.get("epochs_completed") or meta.get("epochs") or 0)
        best = float(meta.get("best_loss", best))
        history = list(meta.get("history") or [])
        global_step = int(meta.get("global_step", 0))
        n_expands = int(meta.get("n_expands", 0))
        expand_log = list(meta.get("expand_log") or [])
        total_seconds_prev = float(
            meta.get("total_seconds") or meta.get("seconds") or 0
        )
        try:
            m.tokenizer.save(str(VOCAB_PATH))
        except Exception:
            pass
        print(
            f"  → استُكمل من العصر {start_epoch} | best_loss={best:.4f} | "
            f"steps={global_step} | expands={n_expands}"
        )
        print(f"  → سيشغّل {EPOCHS} عصر إضافي (حتى {start_epoch + EPOCHS})")
    else:
        print("بدء تدريب جديد (لا checkpoint أو SCN_FRESH=1)")
        n_vocab = m.build_tokenizer_from_texts(
            texts, max_vocab=min(8192, max(4000, len(texts) // 2))
        )
        m.tokenizer.save(str(VOCAB_PATH))
        print(f"قاموس: {n_vocab}")

    # ── Pre-tokenization: تشفير كل النصوص مرة واحدة (تسريع 2-4× على CPU) ─
    token_seqs = None
    if PRE_TOKENIZE:
        # مسار كاش التسلسلات: SCN_TOKEN_CACHE ثم الكاش المنفصل للعامل ثم افتراضي
        token_cache_path = TOKEN_CACHE
        if not token_cache_path:
            wcache = os.environ.get("SCN_WORKER_CACHE", "").strip()
            if wcache:
                token_cache_path = wcache.replace("cache.pkl", "tokens.pkl")
            else:
                token_cache_path = str(_HERE / f"pretrain_tokens_{TAG}.pkl")
        if Path(token_cache_path).is_file():
            try:
                with open(token_cache_path, "rb") as f:
                    token_seqs, saved_len = pickle.load(f)
                if isinstance(token_seqs, list) and len(token_seqs) == len(texts):
                    print(f"✅ pre-tokenize: حُمّلت {len(token_seqs)} تسلسلات من الكاش")
                else:
                    print(f"⚠ كاش التسلسلات لا يطابق عدد المقاطع — يُعاد بناؤه")
                    token_seqs = None
            except Exception as e:
                print(f"⚠ قراءة كاش التسلسلات فشلت ({e}) — يُعاد بناؤه")
                token_seqs = None
        if token_seqs is None:
            print(f"pre-tokenize: تشفير {len(texts)} مقطع مرة واحدة (max_len={MAX_LEN})...")
            t0 = time.time()
            token_seqs = []
            for t in texts:
                ids = m.tokenizer.encode(t, MAX_LEN)
                token_seqs.append(ids.tolist() if hasattr(ids, "tolist") else list(ids))
            Path(token_cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(token_cache_path, "wb") as f:
                pickle.dump((token_seqs, MAX_LEN), f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✅ pre-tokenize: {len(token_seqs)} تسلسل في {time.time() - t0:.1f}ث — {token_cache_path}")
        # استبدال النصوص الخام بالتسلسلات المشفرة — _encode_batch يقبل الاثنين
        train_data = token_seqs
        print(f"✅ وضع pre-tokenized: كل خطوة تُبنى من تسلسلات جاهزة (بلا tokenize حي)")
    else:
        train_data = texts

    print("params:", m.param_count())

    steps_per_epoch = max(1, (len(texts) + BATCH - 1) // BATCH)
    remaining = max(1, EPOCHS * steps_per_epoch)
    if global_step > 0:
        # استكمال: بدون warmup، وcosine على المتبقي فقط من LR الحالي (لا قفزة لـ base_lr الكامل)
        warmup = 0
        total_steps = remaining
        # اضبط base_lr إلى آخر lr محفوظ إن وُجد حتى لا يرتفع فجأة
        if getattr(m, "lr", None) and m.lr < m.base_lr:
            m.base_lr = max(m.lr, m.base_lr * 0.1)
    else:
        total_steps = remaining
        warmup = max(1, int(remaining * WARMUP_RATIO))

    step0_for_lr = global_step
    print(
        f"epochs=+{EPOCHS} (من {start_epoch}) batch={BATCH} "
        f"steps_per_ep={steps_per_epoch} device={m.device}"
    )

    t0 = time.time()
    no_improve = 0
    end_epoch = start_epoch + EPOCHS
    stopped_early = False
    final_epoch = start_epoch

    if UNTIL_END or STOP_PATIENCE > 0:
        print(
            f"وضع الإنهاء: max_epochs=+{EPOCHS} | "
            f"stop_patience={STOP_PATIENCE} | until_end={UNTIL_END}"
        )

    # ── 8-bit Adam: تخفيض ≈40% من VRAM للحجوم الكبيرة ─────────────────────
    # bitsandbytes غير مدعوم على TPU/XLA — تعطيل تلقائي في وضع TPU
    global USE_8BIT_ADAM  # NSM: بدون هذا التعيين المحلي يسبّب UnboundLocalError (تعديل محلي لاحق + قراءة قبلها)
    SCN_TPU_ANY = os.environ.get("SCN_TPU", "0") == "1"
    if SCN_TPU_ANY and USE_8BIT_ADAM:
        USE_8BIT_ADAM = False
        print("✅ وضع TPU: 8-bit Adam معطّل تلقائيًا (bitsandbytes لا يعمل على XLA)")
    if USE_8BIT_ADAM:
        try:
            from bitsandbytes.optim import AdamW8bit
            m.opt = AdamW8bit(m.model.parameters(), lr=m.lr, weight_decay=0.01)
            print("✅ optimizer: AdamW-8bit (bitsandbytes) — توفير VRAM")
        except Exception as e:
            print(f"⚠ AdamW-8bit غير متاح ({e}) — الاستمرار بـAdamW")
            USE_8BIT_ADAM = False
    if GRAD_ACCUM > 1:
        print(f"✅ gradient accumulation: {GRAD_ACCUM} micro-batches/step")
    stop_reason = ""
    for ep in range(start_epoch + 1, end_epoch + 1):
        order = list(train_data)
        random.shuffle(order)
        ep_losses = []
        micro_buf = []
        for i in range(0, len(order), BATCH):
            batch = order[i : i + BATCH]
            # step نسبي للجولة عند الاستكمال حتى لا يُحسب progress على أفق قديم/جديد بشكل يرفع LR
            if global_step > 0 and warmup == 0:
                run_step = global_step - step0_for_lr
            else:
                run_step = global_step
            if GRAD_ACCUM > 1:
                # تجميع تدرجات: zero_grad كل epoch، خطوة محسّن كل GRAD_ACCUM
                accum_loss = 0.0
                accum_n = 0
                for mi in range(0, len(batch), 1):
                    micro = batch[mi: mi + 1]
                    loss = m.train_batch(
                        micro,
                        max_len=MAX_LEN,
                        step=run_step,
                        total_steps=total_steps,
                        warmup_steps=warmup,
                        accum=True,
                    )
                    if loss == loss:  # not NaN
                        accum_loss += loss
                        accum_n += 1
                if accum_n > 0:
                    micro_buf.append(accum_loss / accum_n)
                    if len(micro_buf) >= GRAD_ACCUM or i + BATCH >= len(order):
                        m.accum_step()
                        ep_losses.extend(micro_buf)
                        micro_buf.clear()
            else:
                loss = m.train_batch(
                    batch,
                    max_len=MAX_LEN,
                    step=run_step,
                    total_steps=total_steps,
                    warmup_steps=warmup,
                )
                if loss == loss:  # not NaN
                    ep_losses.append(loss)
            global_step += 1
            # طباعة دورية أثناء العصر
            if LOG_EVERY_STEPS > 0 and global_step % LOG_EVERY_STEPS == 0 and ep_losses:
                recent = ep_losses[-min(20, len(ep_losses)):]
                print(
                    f"  step {global_step} ep={ep} loss≈{sum(recent)/len(recent):.4f} "
                    f"lr={m.lr:.6f}",
                    flush=True,
                )
            # حفظ منتصف-العصر حتى لا يضيع الجهد عند إلغاء الجلسة
            if SAVE_EVERY_STEPS > 0 and global_step % SAVE_EVERY_STEPS == 0:
                mid_meta = {
                    "epoch": ep,
                    "best_loss": best,
                    "history": history,
                    "global_step": global_step,
                    "n_expands": n_expands,
                    "expand_log": expand_log,
                    "n_sentences": len(texts),
                    "d_model": D_MODEL,
                    "mid_epoch": True,
                    "total_seconds": total_seconds_prev + (time.time() - t0),
                }
                try:
                    m.save(str(CKPT_LATEST), train_meta=mid_meta)
                    print(f"  💾 mid-save step={global_step} → {CKPT_LATEST.name}", flush=True)
                    _upload_checkpoint(ep)
                except Exception as _se:
                    print(f"  ⚠ mid-save failed: {_se}", flush=True)
        mean_l = sum(ep_losses) / max(1, len(ep_losses))
        history.append({"epoch": ep, "loss": mean_l, "lr": m.lr})
        mark = ""
        final_epoch = ep
        train_meta = {
            "epoch": ep,
            "best_loss": best,
            "history": history,
            "global_step": global_step,
            "n_expands": n_expands,
            "expand_log": expand_log,
            "n_sentences": len(texts),
            "d_model": D_MODEL,
            "total_seconds": total_seconds_prev + (time.time() - t0),
        }
        if mean_l < best - 1e-5:
            best = mean_l
            train_meta["best_loss"] = best
            m.save(str(CKPT_BEST), train_meta=train_meta)
            mark = " *best*"
            no_improve = 0
        else:
            no_improve += 1
            # --- توسيع احترافي عند هضبة حقيقية فقط ---
            epochs_into_run = ep - start_epoch
            last_expand_ep = expand_log[-1]["epoch"] if expand_log else -10**9
            since_expand = ep - last_expand_ep
            # نافذة أخيرة: هل الـloss مستقر (هضبة) أم ما زال يتذبذب نازلاً؟
            window = [h["loss"] for h in history[-PATIENCE:]]
            flat = False
            if len(window) >= max(3, PATIENCE // 2):
                w_mean = sum(window) / len(window)
                w_span = max(window) - min(window)
                flat = (w_mean > 0 and (w_span / w_mean) <= EXPAND_FLAT_REL)
            # تحسّن ضعيف جداً عبر النافذة (ليس هبوطاً واضحاً)
            weak_trend = False
            if len(window) >= 3:
                weak_trend = window[-1] > window[0] - 1e-3  # لم ينخفض بوضوح من أول النافذة لآخرها

            can_expand = (
                n_expands < MAX_EXPANDS
                and no_improve >= PATIENCE
                and ep >= EXPAND_MIN_EPOCH
                and epochs_into_run >= EXPAND_WARMUP_RUN
                and since_expand >= EXPAND_COOLDOWN
                and flat
                and weak_trend
            )
            if can_expand:
                info = m.expand_narrowest(delta=1)
                if info is not None:
                    n_expands += 1
                    no_improve = 0
                    expand_log.append({"epoch": ep, **info})
                    train_meta["n_expands"] = n_expands
                    train_meta["expand_log"] = expand_log
                    mark += (
                        f" *expand#{n_expands} L{info['layer_idx']} "
                        f"{info['old']}→out{info['new_out']}*"
                    )
                    print(
                        f"  → توسيع مدروس: طبقة {info['layer_idx']} "
                        f"{info['old']} → out={info['new_out']} "
                        f"next_in={info['next_new_in']} "
                        f"(no_improve={PATIENCE}, flat_window=OK)"
                    )
            elif no_improve >= PATIENCE and n_expands < MAX_EXPANDS:
                # تشخيص: لماذا لم يتوسع؟
                reasons = []
                if ep < EXPAND_MIN_EPOCH:
                    reasons.append(f"min_epoch({EXPAND_MIN_EPOCH})")
                if epochs_into_run < EXPAND_WARMUP_RUN:
                    reasons.append(f"run_warmup({EXPAND_WARMUP_RUN})")
                if since_expand < EXPAND_COOLDOWN:
                    reasons.append(f"cooldown({EXPAND_COOLDOWN})")
                if not flat:
                    reasons.append("not_flat")
                if not weak_trend:
                    reasons.append("still_trending_down")
                if reasons and (no_improve == PATIENCE or no_improve % 5 == 0):
                    print(f"  · لا توسيع بعد: {', '.join(reasons)}")

            # إيقاف عند استقرار الـloss (نهاية التدريب)
            if STOP_PATIENCE > 0 and no_improve >= STOP_PATIENCE:
                mark += " *early-stop*"
                train_meta["best_loss"] = best
                train_meta["stopped_early"] = True
                m.save(str(CKPT_LATEST), train_meta=train_meta)
                print(
                    f"epoch {ep:03d}/{end_epoch}  loss={mean_l:.4f}  "
                    f"lr={m.lr:.6f}{mark}"
                )
                print(
                    f"⏹ توقف تلقائي: لا تحسّن لمدة {STOP_PATIENCE} عصور "
                    f"(best_loss={best:.4f})"
                )
                stopped_early = True
                break
        train_meta["best_loss"] = best
        m.save(str(CKPT_LATEST), train_meta=train_meta)
        _CRASH_STATE.update({"model": m, "train_meta": train_meta, "epoch": ep})
        # ── NSM: رفع checkpoint دوري إلى GitHub كل CHECKPOINT_EVERY عصور ──
        #     مع رفع سريع أول عصورين (FIRST_FAST) للحماية من الموت المبكر ──
        should_upload = _should_upload(ep, start_epoch)
        if should_upload:
            _upload_checkpoint(ep)
        # ── إشارة التوقف الآمن (زر التوقف) ──────────────────────────────
        if STOP_SIGNAL_FILE.exists():
            stop_reason = "stop_signal"
            train_meta["stopped_early"] = True
            train_meta["stop_reason"] = "إشارة توقف آمنة (زر التوقف)"
            print("⏹ إشارة توقف آمنة — إيقاف نظيف بعد حفظ checkpoint")
            _upload_checkpoint(ep)
            STOP_SIGNAL_FILE.unlink(missing_ok=True)
            break
        print(f"epoch {ep:03d}/{end_epoch}  loss={mean_l:.4f}  lr={m.lr:.6f}{mark}", flush=True)
        _write_progress({
            "epoch": ep, "end_epoch": end_epoch, "loss": mean_l, "best_loss": best,
            "lr": m.lr, "global_step": global_step,
            "started_at": t0, "elapsed": time.time() - t0, "mark": mark or None,
        })
        # إيقاف آمن قبل حد Kaggle 12 ساعة
        if MAX_HOURS > 0 and (time.time() - t0) / 3600.0 >= MAX_HOURS:
            print(f"⏹ تجاوز الحد الزمني SCN_MAX_HOURS={MAX_HOURS}h — حفظ وإيقاف آمن")
            train_meta["best_loss"] = best
            train_meta["stopped_early"] = True
            train_meta["stop_reason"] = "max_hours"
            m.save(str(CKPT_LATEST), train_meta=train_meta)
            _upload_checkpoint(ep)
            stopped_early = True
            stop_reason = "max_hours"
            final_epoch = ep
            break

    elapsed = time.time() - t0
    state = {
        "data_source": "Jr23xd23/ArabicText-Large (pretrain)",
        "n_sentences": len(texts),
        "epochs_completed": final_epoch,
        "epochs_this_run": max(0, final_epoch - start_epoch),
        "epochs_planned": EPOCHS,
        "best_loss": best,
        "history": history,
        "seconds_this_run": round(elapsed, 1),
        "total_seconds": round(total_seconds_prev + elapsed, 1),
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_pre": N_PRE,
        "n_post": N_POST,
        "chain_scale": CHAIN_SCALE,
        "preset": PRESET or None,
        "tag": TAG,
        "device": str(m.device),
        "backend": "pytorch",
        "real_batch": True,
        "n_expands": n_expands,
        "expand_log": expand_log,
        "no_ckg": True,
        "resumed_from": str(resume_path) if resume_path else None,
        "global_step": global_step,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason or None,
        "until_end": UNTIL_END,
        "stop_patience": STOP_PATIENCE,
        "use_8bit_adam": USE_8BIT_ADAM,
        "grad_accum": GRAD_ACCUM,
        "params": m.param_count(),
    }
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    # تحديث مؤشر التدريب حتى لا تُعاد نفس الجمل في الجولة التالية
    try:
        cursor_path = _HERE / "data" / "training_cursor.json"
        cursor_path.parent.mkdir(parents=True, exist_ok=True)
        cur = {}
        if cursor_path.exists():
            try:
                cur = json.loads(cursor_path.read_text(encoding="utf-8"))
            except Exception:
                cur = {}
        prev = int(cur.get("used_count") or 0)
        # عند SCN_RESPECT_CURSOR=1 نضيف فقط ما درّبنا عليه هذه الجولة
        # وإلا نعتبر أننا استخدمنا len(texts) من بداية الملف
        add_n = len(texts) if os.environ.get("SCN_RESPECT_CURSOR", "0") != "1" else len(texts)
        cur["version"] = 1
        cur["used_count"] = prev + add_n if os.environ.get("SCN_RESPECT_CURSOR", "0") == "1" else max(prev, len(texts))
        cur["last_train_at"] = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat()
        runs = cur.get("runs") or []
        runs.append({
            "at": cur["last_train_at"],
            "used_this_run": len(texts),
            "used_total": cur["used_count"],
            "tag": TAG,
            "preset": PRESET or None,
            "epochs": final_epoch,
        })
        cur["runs"] = runs[-50:]
        cursor_path.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"مؤشر التدريب → used_count={cur['used_count']} (كان {prev})")
    except Exception as _ce:
        print(f"تحذير: تعذّر تحديث training_cursor: {_ce}")
    print("-" * 50)
    print(f"أفضل loss={best:.4f}  زمن_هذه_الجولة={elapsed:.1f}s  device={m.device}")
    print(
        f"العصر النهائي: {final_epoch} | توسيعات: {n_expands} | "
        f"early_stop={stopped_early}"
    )
    if stopped_early and stop_reason == "stop_signal":
        print("✅ توقف آمن بطلب المستخدم — رُفعت آخر checkpoint لاستئناف لاحق")
    elif stopped_early:
        print("✅ اكتمل التدريب حتى استقرار الـloss (النهاية العملية)")
    else:
        print(f"للاستكمال لاحقاً: أعد نفس الأمر (سيُحمّل {CKPT_LATEST.name} تلقائياً)")
    for prompt in ("الصبر", "المعرفة", "اللغة", "العلم", "التاريخ"):
        print(f"  generate({prompt!r}) → {m.generate(prompt, max_new_tokens=24)}")




def _maybe_auto_push_after_train() -> None:
    """إن AUTO_PUSH=1 ارفع النتائج بعد انتهاء main بنجاح."""
    import os
    if os.environ.get("AUTO_PUSH", "").strip().lower() not in ("1", "true", "yes"):
        return
    try:
        from experiments.surah_chain_network.run_train_then_push import push_artifacts
        print("--- AUTO_PUSH: رفع النتائج ---")
        print(push_artifacts())
    except Exception as e:
        print("AUTO_PUSH failed:", e)

if __name__ == "__main__":
    main()
    _maybe_auto_push_after_train()
