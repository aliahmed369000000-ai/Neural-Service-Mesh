import sys, time, json, pickle, os
sys.path.insert(0, '.')
from ai.arabic_transformer import ArabicTransformer

WEIGHTS_DIR = "models/transformer_ckg_v3"
STATE_FILE = "ckg_train_state_v3.json"
PACK_SIZE = 60          # جمل لكل حزمة ملصوقة (sequence packing) — أفضل توازن ذاكرة/سرعة
PACKS_PER_RUN = 4        # عدد الحزم في نفس الاستدعاء (19-20s/حزمة => ~80s إجمالي، آمن)

with open('ckg_sentences_v3.pkl', 'rb') as f:
    sentences = pickle.load(f)
N = len(sentences)

m = ArabicTransformer(d_model=1216, n_heads=16, d_ff=2560, n_layers=8, vocab_size=8192)
if os.path.exists(WEIGHTS_DIR):
    m.load(WEIGHTS_DIR)

if os.path.exists(STATE_FILE):
    state = json.loads(open(STATE_FILE).read())
else:
    state = {"position": 0, "loss_history_tail": []}

pos = state["position"]
if pos >= N:
    print(f"DONE_ALL: التدريب مكتمل بالفعل ({pos}/{N})")
    sys.exit(0)

t0 = time.time()
losses = []
for _ in range(PACKS_PER_RUN):
    if pos >= N:
        break
    end = min(pos + PACK_SIZE, N)
    pack = sentences[pos:end]
    loss = m.train_step_batch(pack)
    losses.append(loss)
    pos = end

elapsed = time.time() - t0
m.save(WEIGHTS_DIR)
state["position"] = pos
state["loss_history_tail"] = [round(x, 3) for x in losses]
with open(STATE_FILE, 'w') as f:
    json.dump(state, f)

pct = pos / N * 100
avg_loss = sum(losses) / len(losses)
print(f"[{pos}/{N}] ({pct:.1f}%) avg_loss={avg_loss:.3f} elapsed={elapsed:.1f}s "
      f"({PACKS_PER_RUN} حزم × {PACK_SIZE} جملة = حتى {PACKS_PER_RUN*PACK_SIZE} جملة/استدعاء)")
if pos >= N:
    print("DONE_ALL")

