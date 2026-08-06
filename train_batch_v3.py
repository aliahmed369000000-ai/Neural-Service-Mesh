"""
تدريب ArabicTransformer v3 (120M) على ckg_sentences_v3.pkl
مع تكيّف تلقائي لحجم الحزمة حسب الرام المتاحة.

يكمل تلقائياً من آخر checkpoint (ckg_train_state_v3.json + models/transformer_ckg_v3).

تجاوز يدوي (اختياري):
  NSM_PACK_SIZE=40 NSM_PACKS_PER_RUN=8 python3 train_batch_v3.py
  NSM_RESET_TRAIN=1 python3 train_batch_v3.py   # إعادة تدريب من الصفر
  NSM_TOKENIZER=bpe python3 train_batch_v3.py          # BPE
  NSM_TOKENIZER=wordpiece python3 train_batch_v3.py       # WordPiece
  NSM_TOKENIZER=sentencepiece python3 train_batch_v3.py   # SentencePiece-style
  NSM_TOKENIZER=unigram python3 train_batch_v3.py         # Unigram LM
  NSM_TOKENIZER=char python3 train_batch_v3.py            # Character-level
  NSM_TOKENIZER=byte_bpe python3 train_batch_v3.py        # Byte-level BPE
  NSM_TOKENIZER=modern_bbpe python3 train_batch_v3.py     # GPT-4/tiktoken-style (موصى به)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

sys.path.insert(0, ".")

WEIGHTS_DIR = "models/transformer_ckg_v3"
STATE_FILE = "ckg_train_state_v3.json"
SENTENCES_FILE = "ckg_sentences_v3.pkl"

# قياس مرجعي: PACK_SIZE=80 → ذروة RSS ~2.93GB على بيئة ~3.7GB متاحة
_REF_PACK_SIZE = 80
_REF_PEAK_GB = 2.93
_SAFETY_MARGIN_GB = 0.35


def available_ram_gb() -> float:
    """MemAvailable من /proc/meminfo (GiB)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])  # kB
        avail_kb = info.get("MemAvailable") or info.get("MemFree", 0)
        return avail_kb / (1024.0 * 1024.0)
    except Exception:
        return 1.5


def choose_pack_size(avail_gb: float) -> int:
    """PACK_SIZE آمن تقريباً حسب الرام. 0 = غير كافٍ."""
    budget = max(0.0, avail_gb - _SAFETY_MARGIN_GB)
    if budget < 1.2:
        return 0
    ratio = budget / _REF_PEAK_GB
    size = int(_REF_PACK_SIZE * ratio)
    size = max(4, min(_REF_PACK_SIZE, size))
    return max(4, (size // 4) * 4)


def choose_packs_per_run(avail_gb: float) -> int:
    """عدد الحزم لكل استدعاء (الذاكرة تستقر بعد أول حزمتين)."""
    if avail_gb >= 3.2:
        return 16
    if avail_gb >= 2.4:
        return 8
    if avail_gb >= 1.8:
        return 4
    return 2


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {
        "position": 0,
        "loss_history_tail": [],
        "runs": 0,
        "total_sentences_seen": 0,
    }


def save_state(state: dict) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, STATE_FILE)


def main() -> int:
    avail = available_ram_gb()

    env_pack = os.environ.get("NSM_PACK_SIZE", "").strip()
    env_packs = os.environ.get("NSM_PACKS_PER_RUN", "").strip()
    pack_size = int(env_pack) if env_pack.isdigit() else choose_pack_size(avail)
    packs_per_run = int(env_packs) if env_packs.isdigit() else choose_packs_per_run(avail)

    print(
        f"ENV: available_ram≈{avail:.2f} GiB | pack_size={pack_size} | "
        f"packs_per_run={packs_per_run}"
    )
    if env_pack or env_packs:
        print("ENV: تجاوز يدوي نشط (NSM_PACK_SIZE / NSM_PACKS_PER_RUN)")

    if pack_size <= 0:
        print(
            f"ABORT: الرام المتاحة ({avail:.2f} GiB) غير كافية لتدريب آمن "
            f"(120M يحتاج تقريباً ≥1.5–2 GiB متاح مع هامش أمان)."
        )
        return 2

    if not os.path.exists(SENTENCES_FILE):
        print(f"ABORT: ملف البيانات غير موجود: {SENTENCES_FILE}")
        return 1

    with open(SENTENCES_FILE, "rb") as f:
        sentences = pickle.load(f)
    n = len(sentences)

    from ai.arabic_transformer import ArabicTransformer, WordTokenizer

    reset = os.environ.get("NSM_RESET_TRAIN", "").strip() in ("1", "true", "yes")
    tok_mode = os.environ.get("NSM_TOKENIZER", "word").strip().lower()
    if tok_mode in ("wp",):
        tok_mode = "wordpiece"
    if tok_mode in ("spm", "sp"):
        tok_mode = "sentencepiece"
    if tok_mode in ("unigramlm",):
        tok_mode = "unigram"
    if tok_mode in ("character",):
        tok_mode = "char"
    if tok_mode in ("bbpe", "bytebpe"):
        tok_mode = "byte_bpe"
    if tok_mode in ("modernbbpe", "tiktoken", "gpt4"):
        tok_mode = "modern_bbpe"
    if tok_mode not in ("word", "bpe", "wordpiece", "sentencepiece", "unigram", "char", "byte_bpe", "modern_bbpe"):
        tok_mode = "word"
    TOKENIZER_VERSION = {
        "bpe": "bpe-v1",
        "wordpiece": "wordpiece-v1",
        "sentencepiece": "sentencepiece-v1",
        "unigram": "unigram-v1",
        "char": "char-v1",
        "byte_bpe": "byte-bpe-v1",
        "modern_bbpe": "modern-bbpe-v1",
        "word": "word-v1",
    }[tok_mode]

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if tok_mode == "bpe":
        from ai.bpe_tokenizer import BPETokenizer
        vocab_path = os.path.join(WEIGHTS_DIR, "bpe_tokenizer.json")
        tok = BPETokenizer(vocab_size=8192, vocab_path=vocab_path if os.path.exists(vocab_path) else None)
        if not os.path.exists(vocab_path) or len(tok.token_to_id) <= 20:
            print("Training BPE tokenizer on sentences…")
            n_vocab = tok.train(sentences)
            tok.save(vocab_path)
            print(f"BPE vocab: {n_vocab} merges={len(tok.merges)} → {vocab_path}")
        else:
            print(f"Loaded BPE vocab ({len(tok.token_to_id)} tokens, {len(tok.merges)} merges)")
    elif tok_mode == "wordpiece":
        from ai.wordpiece_tokenizer import WordPieceTokenizer
        vocab_path = os.path.join(WEIGHTS_DIR, "wordpiece_tokenizer.json")
        tok = WordPieceTokenizer(vocab_size=8192, vocab_path=vocab_path if os.path.exists(vocab_path) else None)
        if not os.path.exists(vocab_path) or len(tok.token_to_id) <= 20:
            print("Training WordPiece tokenizer on sentences…")
            n_vocab = tok.train(sentences)
            tok.save(vocab_path)
            print(f"WordPiece vocab: {n_vocab} → {vocab_path}")
        else:
            print(f"Loaded WordPiece vocab ({len(tok.token_to_id)} tokens)")
    elif tok_mode == "sentencepiece":
        from ai.sentencepiece_tokenizer import SentencePieceTokenizer
        vocab_path = os.path.join(WEIGHTS_DIR, "sentencepiece_tokenizer.json")
        tok = SentencePieceTokenizer(vocab_size=8192, vocab_path=vocab_path if os.path.exists(vocab_path) else None)
        if not os.path.exists(vocab_path) or len(tok.token_to_id) <= 20:
            print("Training SentencePiece tokenizer on sentences…")
            n_vocab = tok.train(sentences)
            tok.save(vocab_path)
            print(f"SentencePiece vocab: {n_vocab} merges={len(tok.merges)} → {vocab_path}")
        else:
            print(f"Loaded SentencePiece vocab ({len(tok.token_to_id)} tokens, {len(tok.merges)} merges)")
    elif tok_mode == "unigram":
        from ai.unigram_tokenizer import UnigramTokenizer
        vocab_path = os.path.join(WEIGHTS_DIR, "unigram_tokenizer.json")
        tok = UnigramTokenizer(vocab_size=8192, vocab_path=vocab_path if os.path.exists(vocab_path) else None)
        if not os.path.exists(vocab_path) or len(tok.token_to_id) <= 20:
            print("Training Unigram LM tokenizer on sentences…")
            n_vocab = tok.train(sentences)
            tok.save(vocab_path)
            print(f"Unigram vocab: {n_vocab} → {vocab_path}")
        else:
            print(f"Loaded Unigram vocab ({len(tok.token_to_id)} tokens)")
    elif tok_mode == "char":
        from ai.char_tokenizer import CharTokenizer
        vocab_path = os.path.join(WEIGHTS_DIR, "char_tokenizer.json")
        tok = CharTokenizer(vocab_size=512, vocab_path=vocab_path if os.path.exists(vocab_path) else None)
        if not os.path.exists(vocab_path) or len(tok.token_to_id) <= 20:
            print("Building Char tokenizer…")
            n_vocab = tok.train(sentences)
            tok.save(vocab_path)
            print(f"Char vocab: {n_vocab} → {vocab_path}")
        else:
            print(f"Loaded Char vocab ({len(tok.token_to_id)} tokens)")
    elif tok_mode == "byte_bpe":
        from ai.byte_bpe_tokenizer import ByteBPETokenizer
        vocab_path = os.path.join(WEIGHTS_DIR, "byte_bpe_tokenizer.json")
        tok = ByteBPETokenizer(vocab_size=8192, vocab_path=vocab_path if os.path.exists(vocab_path) else None)
        if not os.path.exists(vocab_path) or len(tok.merges) == 0:
            print("Training Byte-level BPE…")
            n_vocab = tok.train(sentences)
            tok.save(vocab_path)
            print(f"Byte-BPE vocab: {n_vocab} merges={len(tok.merges)} → {vocab_path}")
        else:
            print(f"Loaded Byte-BPE vocab ({len(tok.token_to_id)} tokens, {len(tok.merges)} merges)")
    elif tok_mode == "modern_bbpe":
        from ai.modern_bbpe_tokenizer import ModernBBPETokenizer
        vocab_path = os.path.join(WEIGHTS_DIR, "modern_bbpe_tokenizer.json")
        tok = ModernBBPETokenizer(vocab_size=16000, vocab_path=vocab_path if os.path.exists(vocab_path) else None)
        if not os.path.exists(vocab_path) or len(tok.merges) == 0:
            print("Training Modern BBPE (GPT-4/tiktoken-style)…")
            n_vocab = tok.train(sentences)
            tok.save(vocab_path)
            print(f"Modern-BBPE vocab: {n_vocab} merges={len(tok.merges)} → {vocab_path}")
        else:
            print(f"Loaded Modern-BBPE vocab ({len(tok.token_to_id)} tokens, {len(tok.merges)} merges)")
    else:
        vocab_path = os.path.join(WEIGHTS_DIR, "tokenizer_vocab.json")
        tok = WordTokenizer(vocab_size=8192, vocab_path=vocab_path if os.path.exists(vocab_path) else None)
        if not os.path.exists(vocab_path) or len(getattr(tok, "word_to_id", {})) <= 64:
            print("Building WordTokenizer vocabulary from training sentences…")
            n_vocab = tok.build_from_texts(sentences, max_vocab=8192)
            tok.save(vocab_path)
            print(f"Vocab size: {n_vocab} → {vocab_path}")
        else:
            print(f"Loaded existing vocab ({len(tok.word_to_id)} tokens) from {vocab_path}")

    print("Loading ArabicTransformer (120M: d_model=1216, n_layers=8)…")
    model = ArabicTransformer(
        d_model=1216, n_heads=16, d_ff=2560, n_layers=8, vocab_size=8192,
        tokenizer=tok, weights_dir=WEIGHTS_DIR,
        tokenizer_type=tok_mode,
    )

    state = load_state()
    prev_tok_ver = state.get("tokenizer_version")
    weights_compatible = (
        not reset
        and prev_tok_ver == TOKENIZER_VERSION
        and os.path.exists(os.path.join(WEIGHTS_DIR, "embedding.npy"))
    )
    if weights_compatible:
        model.load(WEIGHTS_DIR)
        print(f"Loaded weights from {WEIGHTS_DIR} (tokenizer_version={TOKENIZER_VERSION})")
    else:
        if reset:
            print("NSM_RESET_TRAIN=1 → بدء أوزان جديدة وتصفير موضع التدريب")
        elif prev_tok_ver and prev_tok_ver != TOKENIZER_VERSION:
            print(
                f"Tokenizer تغيّر ({prev_tok_ver} → {TOKENIZER_VERSION}) — "
                "تجاهل الأوزان القديمة وبدء تدريب من الصفر"
            )
        else:
            print(f"No compatible checkpoint at {WEIGHTS_DIR} — بدء أوزان جديدة")
        state = {
            "position": 0,
            "loss_history_tail": [],
            "runs": 0,
            "total_sentences_seen": 0,
            "tokenizer_version": TOKENIZER_VERSION,
        }

    state["tokenizer_version"] = TOKENIZER_VERSION
    pos = int(state.get("position", 0))
    if pos >= n:
        print(f"DONE_ALL: التدريب مكتمل بالفعل ({pos}/{n})")
        return 0

    print(f"Resume from position {pos}/{n} ({100.0 * pos / n:.1f}%)")
    start_pos = pos

    t0 = time.time()
    losses: list[float] = []
    packs_done = 0
    for _ in range(packs_per_run):
        if pos >= n:
            break
        end = min(pos + pack_size, n)
        pack = sentences[pos:end]
        loss = float(model.train_step_batch(pack))
        losses.append(loss)
        pos = end
        packs_done += 1
        print(f"  pack {packs_done}/{packs_per_run}  pos={pos}/{n}  loss={loss:.4f}")

    elapsed = time.time() - t0
    model.save(WEIGHTS_DIR)

    processed = pos - start_pos
    prev_tail = list(state.get("loss_history_tail") or [])
    new_tail = (prev_tail + [round(x, 3) for x in losses])[-64:]

    state["position"] = pos
    state["loss_history_tail"] = new_tail
    state["runs"] = int(state.get("runs", 0)) + 1
    state["total_sentences_seen"] = int(state.get("total_sentences_seen", 0)) + processed
    state["last_pack_size"] = pack_size
    state["last_packs_per_run"] = packs_done
    state["last_elapsed_s"] = round(elapsed, 1)
    state["last_avail_ram_gb"] = round(avail, 2)
    state["tokenizer_version"] = TOKENIZER_VERSION
    state["model_version"] = getattr(model, "VERSION", "3.1")
    save_state(state)

    pct = 100.0 * pos / n
    avg_loss = sum(losses) / len(losses) if losses else float("nan")
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = n - pos
    eta_s = remaining / rate if rate > 0 else float("inf")
    eta_str = f"{eta_s / 60:.0f} min" if eta_s < 1e12 else "n/a"

    print(
        f"[{pos}/{n}] ({pct:.1f}%) avg_loss={avg_loss:.3f} elapsed={elapsed:.1f}s "
        f"({packs_done} حزم × حتى {pack_size} جملة | {processed} جملة هذا التشغيل | "
        f"{rate:.1f} جملة/ث | ETA≈{eta_str})"
    )
    if pos >= n:
        print("DONE_ALL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
