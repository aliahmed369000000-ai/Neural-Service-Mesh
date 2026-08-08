"""
تدريب طويل SurahChain LM + Multi-Head Attention على جمل CKG.

  SCN_N=5000 SCN_EPOCHS=20 python experiments/surah_chain_network/train_ckg_lm.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))

from hybrid_experiment import HybridExperimentModel

CKPT_DIR = _HERE / "checkpoints"
CKPT_BEST = CKPT_DIR / "best_ckg_lm.pkl"
CKPT_LATEST = CKPT_DIR / "latest_ckg_lm.pkl"
VOCAB_PATH = _HERE / "tokenizer_vocab_strong.json"
STATE_FILE = CKPT_DIR / "ckg_lm_train_state.json"

N = int(os.environ.get("SCN_N", "5000"))
EPOCHS = int(os.environ.get("SCN_EPOCHS", "20"))
BATCH = int(os.environ.get("SCN_BATCH", "16"))
D_MODEL = int(os.environ.get("SCN_D_MODEL", "256"))
N_HEADS = int(os.environ.get("SCN_N_HEADS", "8"))
N_PRE = int(os.environ.get("SCN_N_PRE", "2"))
N_POST = int(os.environ.get("SCN_N_POST", "2"))
BASE_LR = float(os.environ.get("SCN_LR", "8e-4"))
MAX_LEN = int(os.environ.get("SCN_MAX_LEN", "128"))
WARMUP_RATIO = 0.1


def load_ckg_sentences(max_n: int) -> list:
    paths = [
        _REPO / "ckg_sentences_v3.pkl",
        _REPO / "ckg_sentences.pkl",
        _REPO / "ckg_sentences_general_ar.pkl",
    ]
    all_s = []
    for p in paths:
        if not p.exists():
            continue
        with open(p, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            all_s.extend(
                [s.strip() for s in data if isinstance(s, str) and len(s.strip()) >= 8]
            )
    seen, out = set(), []
    rng = np.random.default_rng(0)
    rng.shuffle(all_s)
    for s in all_s:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_n:
            break
    return out


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    texts = load_ckg_sentences(N)
    if len(texts) < 10:
        raise SystemExit("لا جمل CKG — ضع ckg_sentences_v3.pkl في جذر المستودع")
    print(f"جمل CKG: {len(texts)}")
    print("عينة:", texts[:2])

    m = HybridExperimentModel(
        d_model=D_MODEL,
        lr=BASE_LR,
        tokenizer="strong",
        n_heads=N_HEADS,
        n_pre=N_PRE,
        n_post=N_POST,
    )
    n_vocab = m.build_tokenizer_from_texts(
        texts, max_vocab=min(8192, max(3000, len(texts)))
    )
    m.tokenizer.save(str(VOCAB_PATH))
    print(f"قاموس: {n_vocab}")
    print("params:", m.param_count())

    steps_per_epoch = max(1, (len(texts) + BATCH - 1) // BATCH)
    total_steps = EPOCHS * steps_per_epoch
    warmup = max(1, int(total_steps * WARMUP_RATIO))
    print(
        f"epochs={EPOCHS} batch={BATCH} steps={total_steps} warmup={warmup} "
        f"heads={N_HEADS} pre={N_PRE} post={N_POST}"
    )

    best = float("inf")
    history = []
    global_step = 0
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        order = list(texts)
        np.random.shuffle(order)
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
            ep_losses.append(loss)
            global_step += 1
        mean_l = float(np.mean(ep_losses))
        history.append({"epoch": ep, "loss": mean_l, "lr": m.lr})
        mark = ""
        if mean_l < best:
            best = mean_l
            with open(CKPT_BEST, "wb") as f:
                pickle.dump(m, f)
            mark = " *best*"
        print(f"epoch {ep:03d}/{EPOCHS}  loss={mean_l:.4f}  lr={m.lr:.6f}{mark}")

    with open(CKPT_LATEST, "wb") as f:
        pickle.dump(m, f)
    state = {
        "n_sentences": len(texts),
        "epochs": EPOCHS,
        "best_loss": best,
        "history": history,
        "seconds": round(time.time() - t0, 1),
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_pre": N_PRE,
        "n_post": N_POST,
        "vocab": n_vocab,
    }
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    print("-" * 50)
    print(f"أفضل loss={best:.4f}  زمن={state['seconds']}s")
    for prompt in ("الصبر", "القرآن", "سورة", "الإيمان", "العلم"):
        gen = m.generate(prompt, max_new_tokens=24, temperature=0.8, top_k=30, max_ctx=96)
        print(f"  generate({prompt!r}) → {gen}")


if __name__ == "__main__":
    main()
