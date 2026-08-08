"""
تدريب شبكة Surah-Chain التجريبية (114 طبقة، أبعادها = عدد آيات السور
المتتالية في القرآن، من ملف شبكهه_114-1.xlsx) — بدون learning rate
decay، بمعدل تعلّم ثابت طوال التدريب.

⚠️ تجريبية بحتة: غير متصلة بمسار الإجابة الحي (qa_engine.py). الهدف
إثبات مفهوم: هل تقدر شبكة FC عميقة جداً (114 طبقة) بأبعاد غير منتظمة
تُستخدم كـ"طبقة وسطى مخفية" بين embedding وoutput head من نفس مشروع
NSM؟ — الإجابة: نعم، لكن بحاجة LayerNorm+Residual (بدونهم vanishing
gradient كامل، جرّبنا بأنفسنا).

ملاحظة عن الاستقرار: بدون decay، الشبكة بتوصل لنتائج جيدة لكن بتتعرض
لقفزات دورية (رصدنا 4 قفزات >30% خلال 300 عصر) بسبب ضيق نقطة الاختناق
(7 أبعاد فقط) + عمق الشبكة (114 طبقة). للتعامل مع هذا: هذا السكربت
يحفظ "أفضل نسخة" (Best checkpoint) بدل "آخر نسخة" فقط — إصلاح اكتُشف
ضرورياً بعد ملاحظة إن آخر عصر مش بالضرورة أفضل عصر.
"""
import sys, time, json, pickle, os
sys.path.insert(0, os.path.dirname(__file__))
from hybrid_experiment import HybridExperimentModel
from hybrid_data import SENTENCES
import numpy as np

CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
CKPT_BEST = os.path.join(CKPT_DIR, 'best_model.pkl')
CKPT_LATEST = os.path.join(CKPT_DIR, 'latest_model.pkl')
STATE_FILE = os.path.join(CKPT_DIR, 'train_state.json')

LR = 1.5e-3        # ثابت طوال التدريب — بدون decay، بقرار صريح
TOTAL_EPOCHS = 300
EPOCHS_PER_RUN = 40  # آمن زمنياً لبيئات محدودة (~60-70s لكل تشغيلة)


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    if os.path.exists(CKPT_LATEST):
        with open(CKPT_LATEST, 'rb') as f:
            m = pickle.load(f)
        with open(STATE_FILE) as f:
            state = json.load(f)
        print(f"استؤنف من العصر {state['epoch']}/{TOTAL_EPOCHS}")
    else:
        m = HybridExperimentModel(d_model=256, lr=LR)
        state = {'epoch': 0, 'best_avg': float('inf'), 'best_epoch': 0,
                  'loss_tail': [], 'spike_epochs': []}
        print("بدء تدريب جديد من الصفر (بدون decay)")

    start_epoch = state['epoch']
    end_epoch = min(start_epoch + EPOCHS_PER_RUN, TOTAL_EPOCHS)
    prev_avg = state['loss_tail'][-1] if state['loss_tail'] else None

    t0 = time.time()
    for epoch in range(start_epoch, end_epoch):
        m.lr = LR  # بدون أي decay — ثابت دايمًا
        sentences = list(SENTENCES)
        np.random.shuffle(sentences)
        losses = [m.train_step(s) for s in sentences]
        losses = [x for x in losses if x is not None]
        avg = float(np.mean(losses))

        if prev_avg is not None and avg > prev_avg * 1.3:
            state['spike_epochs'].append(
                {'epoch': epoch + 1, 'from': round(prev_avg, 3), 'to': round(avg, 3)}
            )
        prev_avg = avg

        # حفظ "أفضل نسخة" فقط لما فعلاً تتحسّن — يحمي من قفزات مؤقتة
        if avg < state['best_avg']:
            state['best_avg'] = avg
            state['best_epoch'] = epoch + 1
            with open(CKPT_BEST, 'wb') as f:
                pickle.dump(m, f)

        state['loss_tail'] = (state['loss_tail'] + [round(avg, 3)])[-10:]

    dt = time.time() - t0
    state['epoch'] = end_epoch
    with open(CKPT_LATEST, 'wb') as f:
        pickle.dump(m, f)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    pct = end_epoch / TOTAL_EPOCHS * 100
    print(f"[{end_epoch}/{TOTAL_EPOCHS}] ({pct:.1f}%) elapsed={dt:.1f}s")
    print(f"أفضل متوسط: {state['best_avg']:.3f} (عصر {state['best_epoch']})")
    print(f"قفزات >30% حتى الآن: {len(state['spike_epochs'])}")
    if end_epoch >= TOTAL_EPOCHS:
        print("DONE_ALL")


if __name__ == '__main__':
    main()
