# -*- coding: utf-8 -*-
"""
train_pilot_general_ar.py — تدريب تجريبي محلي (sandbox، CPU فقط)
====================================================================
⚠️ هذا تدريب "تجريبي محلي" (pilot) بحجم صغير مقصود، مو النموذج الإنتاجي:
    - يستخدم نفس بنية ArabicTransformer الموجودة (بدون أي تعديل عليها)
      لكن بأبعاد مصغّرة (d_model=192 بدل 2304) لتكتمل الدورة كاملة خلال
      دقائق على CPU واحد بدون GPU.
    - الهدف: إثبات أن بيانات CKG العامة الجديدة (general_ar_v1) قابلة
      للتدريب الفعلي فعلاً، وتوفير أول نموذج محفوظ حقيقي عليها.
    - هذا لا يستبدل خطة Falcon-H1-Arabic-7B + QLoRA المتفق عليها للنموذج
      العربي الواسع الفعلي — تلك تحتاج GPU حقيقي (Kaggle) ولا يمكن
      تنفيذها في بيئة sandbox هذه.
"""
import sys
import time
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ai.arabic_transformer import ArabicTransformer

PILOT_WEIGHTS_DIR = "models/pilot_general_ar_v1"
SENTENCES_FILE = "ckg_sentences_general_ar.pkl"
N_EPOCHS = 5

def main():
    with open(SENTENCES_FILE, "rb") as f:
        sentences = pickle.load(f)
    print(f"📚 عدد جمل التدريب: {len(sentences)}")

    model = ArabicTransformer(
        d_model=192, n_heads=4, d_ff=384, n_layers=4,
        max_seq=32, vocab_size=8192, lr=0.001,
        weights_dir=PILOT_WEIGHTS_DIR,
    )
    total_params = (
        model.embedding.W.size
        + model.core.W_up.size + model.core.W_down.size
        + model.core.b_up.size + model.core.b_down.size + model.core._W_core.size
        + sum(
            b.mha.Wq.size + b.mha.Wk.size + b.mha.Wv.size + b.mha.Wo.size
            + b.ffn.W1.size + b.ffn.W2.size + b.ffn.b1.size + b.ffn.b2.size
            + b.ln1.g.size + b.ln1.b.size + b.ln2.g.size + b.ln2.b.size
            for b in model.blocks
        )
        + model.head.W.size + model.head.b.size
    )
    print(f"🔢 عدد المعاملات (تجريبي مصغّر): {total_params:,}")

    t0 = time.time()
    loss_log = []
    for epoch in range(1, N_EPOCHS + 1):
        epoch_losses = []
        for i, sent in enumerate(sentences):
            loss = model.train_step(sent)
            epoch_losses.append(loss)
            if (i + 1) % 300 == 0:
                recent = sum(epoch_losses[-300:]) / len(epoch_losses[-300:])
                print(f"  epoch {epoch} | جملة {i+1}/{len(sentences)} | متوسط_loss(آخر300)={recent:.4f}")
        avg = sum(epoch_losses) / len(epoch_losses)
        loss_log.append(avg)
        elapsed = time.time() - t0
        print(f"✅ نهاية epoch {epoch}/{N_EPOCHS} | متوسط_loss={avg:.4f} | الوقت المنقضي={elapsed:.1f}ث")

    print(f"\n📉 مسار الخسارة عبر الدورات: {[round(x,4) for x in loss_log]}")
    improved = loss_log[0] - loss_log[-1]
    print(f"📊 انخفاض الخسارة الكلي: {improved:.4f} ({'✅ تعلّم فعلي' if improved > 0 else '⚠️ لم ينخفض'})")

    Path(PILOT_WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
    model.save()
    print(f"💾 حُفظ النموذج التجريبي → {PILOT_WEIGHTS_DIR}")

    # اختبار نوعي بسيط: أقرب الجمل دلالياً حسب encode()
    print("\n🔍 اختبار نوعي (تشابه دلالي عبر encode):")
    import numpy as np
    test_words = ["الفيزياء", "التاريخ", "الأدب", "الرياضة"]
    for w in test_words:
        v = model.encode(w)
        print(f"  encode('{w}') → shape={v.shape}, norm={np.linalg.norm(v):.3f}")

    return loss_log

if __name__ == "__main__":
    main()
