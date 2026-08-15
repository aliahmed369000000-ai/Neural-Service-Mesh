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
FRESH = os.environ.get("SCN_FRESH", "0") == "1"
RESUME_PATH = os.environ.get("SCN_RESUME_PATH", "").strip()
# ── NSM resume ذكي: "auto" يستأنف تلقائيًا من آخر checkpoint مرفوع على GitHub
#     حتى مع SCN_FRESH=1 — حتى لا يضيع التدريب عند انقطاع الجلسة ──
SCN_RESUME = os.environ.get("SCN_RESUME", "").strip().lower()
CHECKPOINT_EVERY = int(os.environ.get("SCN_CHECKPOINT_EVERY", "2"))  # حفظ مرفوع كل كذا عصر

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
    MAX_LEN = int(os.environ.get("SCN_MAX_LEN", "128"))
    COMPILE = os.environ.get("SCN_COMPILE", "1") == "1"
    # استقرار: التوسيع الذاتي أسمح عند هذا الحجم
    MAX_EXPANDS = int(os.environ.get("SCN_MAX_EXPANDS", "5"))

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
    """يحمّل من الكاش؛ إن لم يوجد أو كان أصغر من المطلوب يشغّل التحضير."""
    if PRETRAIN_CACHE.exists():
        with open(PRETRAIN_CACHE, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            out = [s.strip() for s in data if isinstance(s, str) and len(s.strip()) >= 20]
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
    try:
        r = subprocess.run(
            ["git", "clone", "-q", "--branch", branch,
             f"https://x-access-token:{token}@github.com/{repo}.git", str(tmp)],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            print(f"[checkpoint] clone فشل: {r.stderr[-200:]}")
            return
        dest = tmp / "experiments" / "surah_chain_network" / "checkpoints"
        dest.mkdir(parents=True, exist_ok=True)
        files = [
            (CKPT_LATEST, f"latest_pretrain_{TAG}.pt"),
            (CKPT_BEST, f"best_pretrain_{TAG}.pt"),
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
        if r2.returncode == 0:
            print(f"[checkpoint] رُفعت checkpoint epoch {ep} إلى GitHub ✅")
        else:
            print(f"[checkpoint] push فشل: {r2.stderr[-200:]}")
    except Exception as e:
        print(f"[checkpoint] خطأ: {e}")
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

    for ep in range(start_epoch + 1, end_epoch + 1):
        order = list(texts)
        random.shuffle(order)
        ep_losses = []
        for i in range(0, len(order), BATCH):
            batch = order[i : i + BATCH]
            # step نسبي للجولة عند الاستكمال حتى لا يُحسب progress على أفق قديم/جديد بشكل يرفع LR
            if global_step > 0 and warmup == 0:
                run_step = global_step - step0_for_lr
            else:
                run_step = global_step
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
        # ── NSM: رفع checkpoint دوري إلى GitHub كل CHECKPOINT_EVERY عصور ──
        if CHECKPOINT_EVERY > 0 and (ep - start_epoch) % CHECKPOINT_EVERY == 0:
            _upload_checkpoint(ep)
        print(f"epoch {ep:03d}/{end_epoch}  loss={mean_l:.4f}  lr={m.lr:.6f}{mark}", flush=True)
        _write_progress({
            "epoch": ep, "end_epoch": end_epoch, "loss": mean_l, "best_loss": best,
            "lr": m.lr, "global_step": global_step,
            "started_at": t0, "elapsed": time.time() - t0, "mark": mark or None,
        })

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
        "until_end": UNTIL_END,
        "stop_patience": STOP_PATIENCE,
        "params": m.param_count(),
    }
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    print("-" * 50)
    print(f"أفضل loss={best:.4f}  زمن_هذه_الجولة={elapsed:.1f}s  device={m.device}")
    print(
        f"العصر النهائي: {final_epoch} | توسيعات: {n_expands} | "
        f"early_stop={stopped_early}"
    )
    if stopped_early:
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
