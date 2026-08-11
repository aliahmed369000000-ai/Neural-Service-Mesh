"""
تدريب Pre-training لـ SurahChain LM على بيانات نصية عامة من الإنترنت
(بدون CKG) — مع دعم الاستكمال من آخر checkpoint (resume).

المصدر: Jr23xd23/ArabicText-Large عبر prepare_pretrain_data.py

الاستخدام:
  # 1) تحضير البيانات
  pip install datasets
  SCN_N=30000 python experiments/surah_chain_network/prepare_pretrain_data.py

  # 2) تدريب جديد
  SCN_N=30000 SCN_EPOCHS=10 SCN_D_MODEL=128 SCN_BATCH=16 \\
    python experiments/surah_chain_network/train_pretrain_torch.py

  # 3) استكمال من آخر checkpoint (افتراضي إن وُجد latest)
  SCN_N=30000 SCN_EPOCHS=10 \\
    python experiments/surah_chain_network/train_pretrain_torch.py

  # بدء من الصفر رغم وجود checkpoint
  SCN_FRESH=1 SCN_N=30000 SCN_EPOCHS=5 \\
    python experiments/surah_chain_network/train_pretrain_torch.py

ملاحظات Termux:
  - SCN_EPOCHS = عدد الحقب الإضافية عند الاستكمال (وليس الإجمالي)
  - عند الاستكمال يُحافظ على الأوزان + الـoptimizer + التاريخ
  - يُحفظ checkpoint كل عصر (latest) وأفضل loss (best)
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

CKPT_DIR = _HERE / "checkpoints"
CKPT_BEST = CKPT_DIR / "best_pretrain_torch.pt"
CKPT_LATEST = CKPT_DIR / "latest_pretrain_torch.pt"
VOCAB_PATH = _HERE / "tokenizer_vocab_pretrain.json"
STATE_FILE = CKPT_DIR / "pretrain_torch_state.json"
PRETRAIN_CACHE = _HERE / "data" / "pretrain_sentences.pkl"

N = int(os.environ.get("SCN_N", "8000"))
EPOCHS = int(os.environ.get("SCN_EPOCHS", "15"))
BATCH = int(os.environ.get("SCN_BATCH", "16"))
D_MODEL = int(os.environ.get("SCN_D_MODEL", "128"))
N_HEADS = int(os.environ.get("SCN_N_HEADS", "8"))
N_PRE = int(os.environ.get("SCN_N_PRE", "2"))
N_POST = int(os.environ.get("SCN_N_POST", "2"))
BASE_LR = float(os.environ.get("SCN_LR", "1e-3"))
MAX_LEN = int(os.environ.get("SCN_MAX_LEN", "96"))
COMPILE = os.environ.get("SCN_COMPILE", "0") == "1"
# تحسينات الانتباه (لا تمس سلسلة السور)
USE_QK_NORM = os.environ.get("SCN_QK_NORM", "1") == "1"
USE_GATED_ATTN = os.environ.get("SCN_GATED_ATTN", "1") == "1"
WARMUP_RATIO = 0.1
PATIENCE = int(os.environ.get("SCN_EXPAND_PATIENCE", "2"))
MAX_EXPANDS = int(os.environ.get("SCN_MAX_EXPANDS", "5"))
# إيقاف عند استقرار الـloss (بدون تحسّن لهذا العدد من العصور)
STOP_PATIENCE = int(os.environ.get("SCN_STOP_PATIENCE", "0"))  # 0 = معطّل
# وضع «حتى النهاية»: حد أقصى عالٍ + إيقاف تلقائي عند الاستقرار
UNTIL_END = os.environ.get("SCN_UNTIL_END", "0") == "1"
if UNTIL_END:
    EPOCHS = max(EPOCHS, int(os.environ.get("SCN_MAX_EPOCHS", "80")))
    if STOP_PATIENCE <= 0:
        STOP_PATIENCE = int(os.environ.get("SCN_STOP_PATIENCE", "6"))
# SCN_FRESH=1 يجبر البدء من الصفر؛ غير ذلك يستكمل إن وُجد latest
FRESH = os.environ.get("SCN_FRESH", "0") == "1"
RESUME_PATH = os.environ.get("SCN_RESUME_PATH", "").strip()


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


def _pick_resume_path():
    if FRESH:
        return None
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
    )
    print(f"QK-Norm={USE_QK_NORM} | Gated-Attention={USE_GATED_ATTN}")

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
    total_steps = max(global_step + EPOCHS * steps_per_epoch, 1)
    warmup = max(1, int((EPOCHS * steps_per_epoch) * WARMUP_RATIO))
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
            loss = m.train_batch(
                batch,
                max_len=MAX_LEN,
                step=global_step,
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
            # توسيع ذاتي اختياري عند هضبة قصيرة
            if no_improve >= PATIENCE and n_expands < MAX_EXPANDS:
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
                        f"  → توسيع ذاتي: طبقة {info['layer_idx']} "
                        f"{info['old']} → out={info['new_out']} "
                        f"next_in={info['next_new_in']}"
                    )
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
        print(f"epoch {ep:03d}/{end_epoch}  loss={mean_l:.4f}  lr={m.lr:.6f}{mark}")

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


if __name__ == "__main__":
    main()
