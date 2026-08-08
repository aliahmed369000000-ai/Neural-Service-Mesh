"""
تدريب SurahChain كشبكة نموذج لغوي:
  - WordTokenizer يُبنى من البيانات
  - دفعات (batches)
  - LR: warmup + cosine decay
  - حفظ أفضل checkpoint + تجربة generate

الاستخدام من جذر المستودع:
  python experiments/surah_chain_network/train_lm.py
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

from hybrid_data import SENTENCES
from hybrid_experiment import HybridExperimentModel

CKPT_DIR = _HERE / "checkpoints"
CKPT_BEST = CKPT_DIR / "best_lm.pkl"
CKPT_LATEST = CKPT_DIR / "latest_lm.pkl"
VOCAB_PATH = _HERE / "tokenizer_vocab.json"
STATE_FILE = CKPT_DIR / "lm_train_state.json"

BASE_LR = float(os.environ.get("SCN_LR", "1.5e-3"))
EPOCHS = int(os.environ.get("SCN_EPOCHS", "30"))
BATCH_SIZE = int(os.environ.get("SCN_BATCH", "8"))
D_MODEL = int(os.environ.get("SCN_D_MODEL", "256"))
MAX_LEN = int(os.environ.get("SCN_MAX_LEN", "32"))
WARMUP_RATIO = 0.1


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    texts = list(SENTENCES)

    m = HybridExperimentModel(d_model=D_MODEL, lr=BASE_LR, tokenizer="word")
    n_vocab = m.build_tokenizer_from_texts(texts)
    m.tokenizer.save(str(VOCAB_PATH))
    print(f"قاموس WordTokenizer: {n_vocab} رمز → {VOCAB_PATH}")
    print("params:", m.param_count())

    steps_per_epoch = max(1, (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE)
    total_steps = EPOCHS * steps_per_epoch
    warmup = max(1, int(total_steps * WARMUP_RATIO))
    print(f"epochs={EPOCHS} batch={BATCH_SIZE} total_steps={total_steps} warmup={warmup}")

    best = float("inf")
    history = []
    global_step = 0
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        order = list(texts)
        np.random.shuffle(order)
        ep_losses = []
        for i in range(0, len(order), BATCH_SIZE):
            batch = order[i : i + BATCH_SIZE]
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
        marker = ""
        if mean_l < best:
            best = mean_l
            with open(CKPT_BEST, "wb") as f:
                pickle.dump(m, f)
            marker = " *best*"
        print(f"epoch {ep:03d}/{EPOCHS}  loss={mean_l:.4f}  lr={m.lr:.5f}{marker}")

    with open(CKPT_LATEST, "wb") as f:
        pickle.dump(m, f)
    state = {
        "epochs": EPOCHS,
        "best_loss": best,
        "history": history,
        "seconds": round(time.time() - t0, 1),
        "d_model": D_MODEL,
        "vocab": n_vocab,
    }
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))

    print("-" * 50)
    print(f"أفضل loss={best:.4f}  زمن={state['seconds']}s")
    for prompt in ("الصبر", "العلم", "القرآن"):
        gen = m.generate(prompt, max_new_tokens=16, temperature=0.85, top_k=25)
        print(f"  generate({prompt!r}) → {gen}")


if __name__ == "__main__":
    main()
