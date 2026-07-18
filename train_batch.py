import sys, time, json, pickle, os
sys.path.insert(0, '.')
from ai.arabic_transformer import ArabicTransformer

# قابلة للتخصيص عبر متغيرات بيئة (مثلاً للتدريب على ckg_sentences_v2.pkl
# الأنظف/الأوسع بدل الجيل الأول) — الافتراضي يطابق السلوك القديم تماماً.
WEIGHTS_DIR   = os.environ.get("CKG_WEIGHTS_DIR", "models/transformer_ckg_v1")
STATE_FILE    = os.environ.get("CKG_STATE_FILE", "ckg_train_state.json")
SENTENCES_FILE = os.environ.get("CKG_SENTENCES_FILE", "ckg_sentences.pkl")
BATCH_SIZE = int(os.environ.get("CKG_BATCH_SIZE", "300"))

with open(SENTENCES_FILE, 'rb') as f:
    sentences = pickle.load(f)
N = len(sentences)

m = ArabicTransformer()
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

end = min(pos + BATCH_SIZE, N)
batch = sentences[pos:end]

t0 = time.time()
losses = []
for s in batch:
    losses.append(m.train_step(s))
elapsed = time.time() - t0

m.save(WEIGHTS_DIR)
state["position"] = end
state["loss_history_tail"] = [round(x, 3) for x in losses[-10:]]
with open(STATE_FILE, 'w') as f:
    json.dump(state, f)

pct = end / N * 100
avg_loss = sum(losses) / len(losses)
print(f"[{end}/{N}] ({pct:.1f}%) avg_loss={avg_loss:.3f} elapsed={elapsed:.1f}s")
if end >= N:
    print("DONE_ALL")
