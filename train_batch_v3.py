import sys, time, json, pickle, os
sys.path.insert(0, '.')
from ai.arabic_transformer import ArabicTransformer

WEIGHTS_DIR = "models/transformer_ckg_v3"
STATE_FILE = "ckg_train_state_v3.json"
PACK_SIZE = 80           # جمل لكل حزمة ملصوقة (sequence packing).
                         # قياس فعلي (benchmark_single_batch_v3.py + فحص استقرار عبر
                         # 6 حزم متتالية): ذروة RSS مستقرة عند ~2.93GB من ~3.7GB
                         # متاحة — هامش أمان ~0.8GB. حجم 90 يقترب من ~3.2GB (هامش أضيق)،
                         # وحجم 100 يقترب من ~3.68GB (خطر OOM فعلي) — الكلفة تكبر أسرع
                         # من خطي مع الحجم (attention على التسلسل الملصوق كامل)، فـ80
                         # هو أقصى توسيع آمن لحجم الحزمة نفسه.
PACKS_PER_RUN = 16       # عدد الحزم في نفس الاستدعاء. الذاكرة تستقر بعد أول حزمتين ولا
                         # تتراكم مع زيادة عدد الحزم (تم التحقق فعلياً حتى 6 حزم متتالية) —
                         # فرفع العدد لا يزيد خطر OOM، فقط وقت التنفيذ الإجمالي. لا يوجد
                         # حد زمني صارم للاستدعاء الواحد في بيئة التشغيل الحالية، فرُفع
                         # العدد من 4 إلى 16 (240 → 1,280 جملة/استدعاء تقريباً).

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
m.save(WEIGHTS_DIR)  # atomic (write-temp-then-replace) — راجع arabic_transformer.py
state["position"] = pos
state["loss_history_tail"] = [round(x, 3) for x in losses]

# كتابة atomic لملف الحالة نفسه: لو الانقطاع حصل هنا بالذات، الملف
# القديم يفضل سليم بدل JSON نص-مكتوب/تالف.
tmp_state = STATE_FILE + ".tmp"
with open(tmp_state, 'w') as f:
    json.dump(state, f)
os.replace(tmp_state, STATE_FILE)

pct = pos / N * 100
avg_loss = sum(losses) / len(losses)
print(f"[{pos}/{N}] ({pct:.1f}%) avg_loss={avg_loss:.3f} elapsed={elapsed:.1f}s "
      f"({PACKS_PER_RUN} حزم × {PACK_SIZE} جملة = حتى {PACKS_PER_RUN*PACK_SIZE} جملة/استدعاء)")
if pos >= N:
    print("DONE_ALL")

