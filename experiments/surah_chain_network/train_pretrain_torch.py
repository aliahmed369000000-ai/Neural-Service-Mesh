"""
تدريب Pre-training لـ SurahChain LM على بيانات نصية عامة من الإنترنت
(بدون CKG).

المصدر: Jr23xd23/ArabicText-Large عبر prepare_pretrain_data.py

الاستخدام:
  # 1) تحضير البيانات (مرة واحدة)
  pip install datasets
  python experiments/surah_chain_network/prepare_pretrain_data.py

  # 2) التدريب
  python experiments/surah_chain_network/train_pretrain_torch.py

  # أمثلة متقدمة
  SCN_N=12000 SCN_EPOCHS=20 SCN_D_MODEL=256 SCN_BATCH=32 \\
    python experiments/surah_chain_network/train_pretrain_torch.py
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
BATCH = int(os.environ.get("SCN_BATCH", "32"))
D_MODEL = int(os.environ.get("SCN_D_MODEL", "256"))
N_HEADS = int(os.environ.get("SCN_N_HEADS", "8"))
N_PRE = int(os.environ.get("SCN_N_PRE", "2"))
N_POST = int(os.environ.get("SCN_N_POST", "2"))
BASE_LR = float(os.environ.get("SCN_LR", "1e-3"))
MAX_LEN = int(os.environ.get("SCN_MAX_LEN", "96"))
COMPILE = os.environ.get("SCN_COMPILE", "0") == "1"
WARMUP_RATIO = 0.1
PATIENCE = int(os.environ.get("SCN_EXPAND_PATIENCE", "2"))
MAX_EXPANDS = int(os.environ.get("SCN_MAX_EXPANDS", "5"))


def load_pretrain_sentences(max_n: int) -> list[str]:
    """يحمّل من الكاش؛ إن لم يوجد يشغّل التحضير تلقائياً."""
    if not PRETRAIN_CACHE.exists():
        print("كاش Pre-training غير موجود — تشغيل prepare_pretrain_data...")
        from prepare_pretrain_data import load_and_prepare, CACHE_FILE, CACHE_DIR

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        sentences = load_and_prepare(max_n)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(sentences, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"حُفظ الكاش: {CACHE_FILE}")
        return sentences[:max_n]

    with open(PRETRAIN_CACHE, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"كاش غير صالح: {PRETRAIN_CACHE}")
    out = [s.strip() for s in data if isinstance(s, str) and len(s.strip()) >= 20]
    random.Random(0).shuffle(out)
    return out[:max_n]


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("SurahChain Pre-training (بيانات عامة من الإنترنت — بدون CKG)")
    print("=" * 60)

    texts = load_pretrain_sentences(N)
    if len(texts) < 50:
        print(f"بيانات قليلة جداً ({len(texts)}). شغّل prepare_pretrain_data أولاً.")
        sys.exit(1)
    print(f"مقاطع التدريب: {len(texts)}")
    print(f"عينة: {texts[0][:120]}...")

    m = HybridExperimentModelTorch(
        d_model=D_MODEL,
        vocab_size=8192,
        lr=BASE_LR,
        n_heads=N_HEADS,
        n_pre=N_PRE,
        n_post=N_POST,
        compile_model=COMPILE,
    )
    n_vocab = m.build_tokenizer_from_texts(
        texts, max_vocab=min(8192, max(4000, len(texts) // 2))
    )
    m.tokenizer.save(str(VOCAB_PATH))
    print(f"قاموس: {n_vocab}")
    print("params:", m.param_count())

    steps_per_epoch = max(1, (len(texts) + BATCH - 1) // BATCH)
    total_steps = EPOCHS * steps_per_epoch
    warmup = max(1, int(total_steps * WARMUP_RATIO))
    print(
        f"epochs={EPOCHS} batch={BATCH} steps={total_steps} "
        f"warmup={warmup} device={m.device} compile={COMPILE}"
    )

    best = float("inf")
    history = []
    global_step = 0
    t0 = time.time()
    no_improve = 0
    n_expands = 0
    expand_log = []

    for ep in range(1, EPOCHS + 1):
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
        if mean_l < best - 1e-5:
            best = mean_l
            m.save(str(CKPT_BEST))
            mark = " *best*"
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE and n_expands < MAX_EXPANDS:
                info = m.expand_narrowest(delta=1)
                if info is not None:
                    n_expands += 1
                    no_improve = 0
                    expand_log.append({"epoch": ep, **info})
                    mark += (
                        f" *expand#{n_expands} L{info['layer_idx']} "
                        f"{info['old']}→out{info['new_out']}*"
                    )
                    print(
                        f"  → توسيع ذاتي: طبقة {info['layer_idx']} "
                        f"{info['old']} → out={info['new_out']} "
                        f"next_in={info['next_new_in']}"
                    )
        print(f"epoch {ep:03d}/{EPOCHS}  loss={mean_l:.4f}  lr={m.lr:.6f}{mark}")

    m.save(str(CKPT_LATEST))
    state = {
        "data_source": "Jr23xd23/ArabicText-Large (pretrain)",
        "n_sentences": len(texts),
        "epochs": EPOCHS,
        "best_loss": best,
        "history": history,
        "seconds": round(time.time() - t0, 1),
        "d_model": D_MODEL,
        "device": str(m.device),
        "backend": "pytorch",
        "real_batch": True,
        "n_expands": n_expands,
        "expand_log": expand_log,
        "no_ckg": True,
    }
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    print("-" * 50)
    print(f"أفضل loss={best:.4f}  زمن={state['seconds']}s  device={m.device}")
    print(f"توسيعات ذاتية: {n_expands}")
    for prompt in ("الصبر", "المعرفة", "اللغة", "العلم", "التاريخ"):
        print(f"  generate({prompt!r}) → {m.generate(prompt, max_new_tokens=24)}")


if __name__ == "__main__":
    main()
