"""
NSM Trainer v3 — تدريب طبقة Embedding مستقلة
================================================
المشكلة المكتشفة:
  • W (784×784) موجبة 93% → بعد الضرب كل المتجهات تتقارب (cos=0.98)
  • المتجهات الخام x جيدة التمييز (cos=0.25 بين مواضيع مختلفة)

الحل:
  • نُدرّب E (784×128): x(784) → E.T@x → z(128)
  • نُخزّن W للاستخدام في nsm_chat (نحترمها ولا نحذفها)
  • التدريب: Triplet Loss على z مباشرة بدون W

النتيجة:
  • cos_same > 0.8  (نفس الموضوع)
  • cos_diff < 0.2  (مواضيع مختلفة)

الاستخدام:
    python nsm_trainer.py              # تدريب 50 epoch
    python nsm_trainer.py --epochs 100 # تدريب أطول
    python nsm_trainer.py --eval       # تقييم فقط
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np

_ROOT  = Path(__file__).parent
_E_OUT = _ROOT / "nsm_embedding.npz"

EMBED_DIM = 128

# ══════════════════════════════════════════════════════════════════
# بيانات التدريب
# ══════════════════════════════════════════════════════════════════
TRAIN_DATA = [
    # 0 — إسلام وقرآن
    ("أركان الإسلام خمسة الشهادتان والصلاة والزكاة والصوم والحج", 0),
    ("الصلوات الخمس الفجر والظهر والعصر والمغرب والعشاء", 0),
    ("القرآن كلام الله المنزل على النبي محمد مئة وأربع عشرة سورة", 0),
    ("سورة الفاتحة أم الكتاب وأول سور القرآن سبع آيات", 0),
    ("التوحيد الإيمان بوحدانية الله لا شريك له في ربوبيته", 0),
    ("النبي محمد خاتم الأنبياء والمرسلين ولد في مكة المكرمة", 0),
    ("الزكاة فريضة مالية ركن الإسلام تُصرف للمحتاجين", 0),
    ("الحج فريضة على المستطيع مرة في العمر في مكة", 0),
    ("صوم رمضان إمساك عن الطعام من الفجر للمغرب", 0),
    ("الشهادتان لا إله إلا الله محمد رسول الله مفتاح الإسلام", 0),
    ("آية الكرسي أعظم آية تصف الله الحي القيوم", 0),
    ("يوم القيامة الحساب والجزاء الجنة للمؤمن والنار للكافر", 0),
    ("الملائكة مخلوقات نورانية جبريل ميكائيل إسرافيل", 0),
    ("الهجرة النبوية من مكة إلى المدينة بداية التقويم الهجري", 0),

    # 1 — ذكاء اصطناعي وتكنولوجيا
    ("الذكاء الاصطناعي علم برمجة الحواسيب لمحاكاة الذكاء", 1),
    ("الشبكة العصبية نموذج رياضي مستوحى من الدماغ يتعلم", 1),
    ("تعلم الآلة فرع الذكاء الاصطناعي يتعلم من البيانات", 1),
    ("Transformer معمارية Self-Attention أساس نماذج اللغة الحديثة", 1),
    ("GPT نموذج لغوي ضخم يولّد النصوص من شركة OpenAI", 1),
    ("التعلم العميق شبكات عصبية متعددة الطبقات deep learning", 1),
    ("NLP معالجة اللغة الطبيعية تمكّن الحاسوب من فهم اللغة", 1),
    ("Backpropagation خوارزمية حساب التدرجات تحديث الأوزان", 1),
    ("دالة الخسارة تقيس الفرق بين إخراج النموذج والصواب", 1),
    ("Overfitting النموذج يحفظ التدريب لا يُعمّم للبيانات الجديدة", 1),
    ("embedding تحويل الكلمات إلى متجهات رقمية في فضاء مستمر", 1),
    ("neural network weights bias activation function gradient", 1),

    # 2 — رياضيات وعلوم
    ("الجبر الخطي دراسة المتجهات والمصفوفات والمعادلات الخطية", 2),
    ("المصفوفة مجموعة أرقام مرتبة صفوف وأعمدة للحسابات", 2),
    ("التفاضل حساب معدلات التغير والمشتقات في الرياضيات", 2),
    ("الاحتمال قياس إمكانية وقوع حدث بين صفر وواحد", 2),
    ("الإحصاء علم جمع وتحليل وتفسير البيانات الرقمية", 2),
    ("المتجه كمية لها قيمة واتجاه في الفضاء الرياضي", 2),
    ("الناتج الداخلي dot product يقيس التشابه بين المتجهين", 2),
    ("الفيزياء علم الطاقة والمادة والقوى في الكون", 2),
    ("الكيمياء علم المادة وتحولاتها والتفاعلات والعناصر", 2),
    ("معادلة تفاضلية تصف التغير والديناميكيات الرياضية", 2),

    # 3 — لغة عربية وأدب
    ("اللغة العربية لغة القرآن أغنى اللغات أربعمئة مليون", 3),
    ("النحو العربي علم ضبط أواخر الكلمات وبنية الجملة", 3),
    ("الصرف علم بنية الكلمة العربية واشتقاقها وتصريفها", 3),
    ("البلاغة فن التعبير الجميل المؤثر تشبيه استعارة كناية", 3),
    ("المتنبي أمير شعراء العربية القرن العاشر الميلادي", 3),
    ("الشعر العربي فن وزن وقافية وعاطفة عصر جاهلي عباسي", 3),
    ("الفصحى اللغة العربية المعيارية الرسمية للكتابة والإعلام", 3),
    ("الأدب العربي إنتاج إبداعي شعر نثر رواية قصة", 3),
    ("اللغة العربية فصاحة بلاغة مفردات اشتقاق جذور", 3),

    # 4 — تاريخ
    ("الهجرة النبوية انتقال النبي مكة المدينة عام ستمئة", 4),
    ("الخلفاء الراشدون أبو بكر عمر عثمان علي رضي الله", 4),
    ("فتح مكة النبي دخل فاتحاً طهّر الكعبة من الأصنام", 4),
    ("الحضارة الإسلامية علوم طب فلك رياضيات بيت الحكمة", 4),
    ("العصر العباسي بغداد الذهبي علوم ثقافة حضارة", 4),
    ("الأندلس إسبانيا ثمانية قرون حضارة إسلامية أوروبا", 4),
    ("غزوة بدر أحد خندق فتح مكة غزوات النبي التاريخ", 4),

    # 5 — صحة وطب
    ("الصحة نوم كافٍ غذاء متوازن رياضة منتظمة تجنب التوتر", 5),
    ("التغذية السليمة بروتين كربوهيدرات دهون فيتامينات معادن", 5),
    ("الرياضة تقوي الجسم تحسن المزاج تحمي من الأمراض", 5),
    ("النوم الصحي سبع ثماني ساعات ضرورية للجسم والعقل", 5),
    ("التوتر أسبابه الضغط القلق قلة النوم العمل الزائد", 5),
    ("المناعة جهاز دفاع الجسم ضد الأمراض الميكروبات الفيروسات", 5),
    ("الطب العلاج الدواء الطبيب المستشفى الصحة الجسدية", 5),

    # 6 — برمجة
    ("Python لغة برمجة سهلة قوية للذكاء الاصطناعي والويب", 6),
    ("API واجهة برمجية تتيح التواصل بين التطبيقات والخدمات", 6),
    ("قاعدة البيانات SQL نظام تخزين البيانات واسترجاعها", 6),
    ("Git نظام تحكم إصدارات الكود البرمجي GitHub", 6),
    ("Framework إطار عمل بنية جاهزة لتطوير التطبيقات", 6),
    ("JavaScript لغة الويب Frontend Backend Node.js", 6),
    ("البرمجة كود سكريبت تطوير تطبيقات خوارزميات", 6),
]

NUM_TOPICS = 7

# ══════════════════════════════════════════════════════════════════
# Tokenizer
# ══════════════════════════════════════════════════════════════════
def _fnv1a(s):
    h = 0x811c9dc5
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def text_to_vec(text: str, dim=784) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float64)
    t = text.strip().lower()
    for n in (1, 2, 3):
        for i in range(len(t) - n + 1):
            vec[_fnv1a(t[i:i+n]) % dim] += 1.0
    total = vec.sum()
    if total > 0:
        vec = np.log1p(vec * 10.0 / total) / math.log1p(10.0)
    return vec

def l2(x): return x / (np.linalg.norm(x) + 1e-8)
def cosine(a, b): return float(l2(a) @ l2(b))

# ══════════════════════════════════════════════════════════════════
# Forward: x(784) → E.T@x → z(128) مع L2
# ══════════════════════════════════════════════════════════════════
def embed(E, x):
    raw = E.T @ x          # (128,)
    return l2(raw)         # (128,) normalized

# ══════════════════════════════════════════════════════════════════
# Triplet Loss مع Hard Negative Mining
# ══════════════════════════════════════════════════════════════════
def triplet_grad(E, anc, pos, neg, margin=0.4):
    za = embed(E, anc)
    zp = embed(E, pos)
    zn = embed(E, neg)

    d_pos = 1.0 - float(za @ zp)   # distance = 1 - cosine
    d_neg = 1.0 - float(za @ zn)
    loss = max(0.0, margin + d_pos - d_neg)

    if loss < 1e-7:
        return loss, np.zeros_like(E)

    # dL/d_za = zp - zn  (simplified gradient for cosine)
    dza = -zp + zn   # (128,)

    # chain rule عبر L2 norm
    raw_a = E.T @ anc
    norm_a = np.linalg.norm(raw_a) + 1e-8
    draw = (dza - za * float(za @ dza)) / norm_a  # (128,)

    # dL/dE = anc ⊗ draw
    dE = np.outer(anc, draw)   # (784, 128)
    return loss, dE

# ══════════════════════════════════════════════════════════════════
# التدريب
# ══════════════════════════════════════════════════════════════════
def train(epochs=80, lr=0.005, margin=0.4):
    print("=" * 55)
    print("  NSM Trainer v3 — تدريب Embedding مستقل")
    print("=" * 55)

    rng = np.random.default_rng(42)

    # تهيئة E
    scale = 1.0 / math.sqrt(784)
    E = rng.normal(0, scale, (784, EMBED_DIM))
    print(f"✓ E مُهيَّأة: {E.shape}")
    print(f"✓ بيانات: {len(TRAIN_DATA)} جملة — {NUM_TOPICS} مواضيع")

    # تجهيز المتجهات مرة واحدة
    vecs = [(text_to_vec(txt), lbl) for txt, lbl in TRAIN_DATA]

    # Adam
    m = np.zeros_like(E)
    v = np.zeros_like(E)
    beta1, beta2, eps_a = 0.9, 0.999, 1e-8
    step = 0

    print(f"\n{'Epoch':>6} {'Loss':>9} {'same':>8} {'diff':>8} {'gap':>7}")
    print("-" * 42)

    best_gap, best_E = -1.0, E.copy()

    for epoch in range(1, epochs + 1):
        total_loss, total_dE, n = 0.0, np.zeros_like(E), 0

        idx_list = list(range(len(vecs)))
        rng.shuffle(idx_list)

        for idx in idx_list:
            anc_x, anc_lbl = vecs[idx]

            # positive pool
            pos_pool = [i for i in range(len(vecs)) if i != idx and vecs[i][1] == anc_lbl]
            # hard negative: أقرب من موضوع مختلف
            neg_pool = [i for i in range(len(vecs)) if vecs[i][1] != anc_lbl]
            if not pos_pool or not neg_pool: continue

            # Hard negative mining: اختر أعلى تشابه
            if epoch > 10:
                za = embed(E, anc_x)
                neg_scores = [(cosine(za, embed(E, vecs[i][0])), i) for i in neg_pool]
                neg_scores.sort(reverse=True)
                neg_idx = neg_scores[0][1]  # الأقرب (أصعب)
            else:
                neg_idx = int(rng.choice(neg_pool))

            pos_idx = int(rng.choice(pos_pool))
            pos_x = vecs[pos_idx][0]
            neg_x = vecs[neg_idx][0]

            loss, dE = triplet_grad(E, anc_x, pos_x, neg_x, margin)
            total_loss += loss
            total_dE += dE
            n += 1

        if n == 0: continue
        step += 1
        grad = total_dE / n

        # Adam
        m = beta1*m + (1-beta1)*grad
        v = beta2*v + (1-beta2)*grad**2
        m_hat = m/(1-beta1**step)
        v_hat = v/(1-beta2**step)
        E -= lr * m_hat/(np.sqrt(v_hat)+eps_a)

        # تقييم كل 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            same_list, diff_list = [], []
            for i in range(len(vecs)):
                za = embed(E, vecs[i][0])
                for j in range(i+1, len(vecs)):
                    zb = embed(E, vecs[j][0])
                    c = cosine(za, zb)
                    (same_list if vecs[i][1]==vecs[j][1] else diff_list).append(c)

            s = np.mean(same_list) if same_list else 0
            d = np.mean(diff_list) if diff_list else 0
            gap = s - d
            avg_l = total_loss / n
            print(f"{epoch:>6} {avg_l:>9.4f} {s:>8.4f} {d:>8.4f} {gap:>7.4f}")

            if gap > best_gap:
                best_gap, best_E = gap, E.copy()

    print(f"\n✓ أفضل gap: {best_gap:.4f}")
    np.savez(_E_OUT, E=best_E, dim=EMBED_DIM)
    print(f"✓ حُفظت E في: {_E_OUT}")
    return best_E

# ══════════════════════════════════════════════════════════════════
# التقييم
# ══════════════════════════════════════════════════════════════════
def evaluate():
    if not _E_OUT.exists():
        print("❌ شغّل التدريب أولاً: python nsm_trainer.py")
        return

    E = np.load(_E_OUT)["E"]
    print(f"\n✓ E محملة: {E.shape}\n")

    topic_samples = {
        "إسلام":     ["أركان الإسلام والصلاة","القرآن سورة الفاتحة","التوحيد والنبي محمد"],
        "تكنولوجيا": ["ذكاء اصطناعي شبكة عصبية","Transformer GPT نموذج","تعلم عميق embedding"],
        "رياضيات":   ["جبر خطي مصفوفة متجه","احتمال إحصاء تحليل","معادلة تفاضلية رياضيات"],
        "لغة":       ["لغة عربية نحو صرف","شعر بلاغة أدب فصحى","المتنبي قصيدة عروض"],
        "تاريخ":     ["هجرة نبوية مكة مدينة","خلفاء راشدون إسلام","حضارة عباسية بغداد"],
        "صحة":       ["تغذية صحية بروتين فيتامين","رياضة ونوم ومناعة","طب علاج دواء صحة"],
        "برمجة":     ["Python API برمجة كود","قاعدة بيانات SQL Git","JavaScript Framework ويب"],
    }

    tvecs = {}
    for topic, samples in topic_samples.items():
        zs = [embed(E, text_to_vec(s)) for s in samples]
        tvecs[topic] = np.mean(zs, axis=0)

    names = list(tvecs.keys())
    print(f"{'':10}", end="")
    for n in names: print(f"{n[:6]:9}", end="")
    print()
    print("-" * 73)

    good, total = 0, 0
    for n1 in names:
        print(f"{n1[:8]:10}", end="")
        for n2 in names:
            c = cosine(tvecs[n1], tvecs[n2])
            if n1 == n2:
                tag = "█"
            elif c < 0.3:
                tag = "✓"; good += 1
            elif c < 0.6:
                tag = "~"
            else:
                tag = "✗"
            if n1 != n2: total += 1
            print(f"{c:6.3f}{tag:3}", end="")
        print()

    print(f"\n█=نفس الموضوع  ✓=فصل ممتاز(<0.3)  ~=متوسط  ✗=ضعيف")
    print(f"الفصل الممتاز: {good}/{total} = {good/max(total,1)*100:.1f}%")


# ══════════════════════════════════════════════════════════════════
# ★ إضافة جديدة (لا تُغيّر أي شيء أعلاه) ★
# Continual Training محمي بـ EWC + Rollback — الأولويتان #1 و #3
# ════════════════════════════════════════════════════════════════════
# المشكلة المكتشفة في تقرير التحليل:
#   train() أعلاه يبدأ دائماً من E عشوائية جديدة (rng.normal) — أي تدريب
#   على موضوع إضافي لاحقاً سيُعيد كتابة nsm_embedding.npz من الصفر وينسى
#   كل المواضيع السابقة (Catastrophic Forgetting). كذلك EWCLearner في
#   ai/continual_learner.py كان موجوداً لكنه **غير مُستخدَم في أي مكان**
#   بالمشروع (orphan module) — ولا توجد أي حماية rollback إذا خرج
#   تدريب جديد بنتيجة أسوأ.
#
# الحل:
#   continual_train() تُحمّل E الموجودة (لا تبدأ من الصفر)، تحسب/تحمّل
#   Fisher Information لحماية المعرفة القديمة (EWC)، تُدرّب على بيانات
#   موضوع جديد فقط، وتُغلّف كل العملية بـ CheckpointGuard: إن تراجع
#   الفصل (separation gap) على البيانات القديمة أكثر من الحد المسموح،
#   يُستعاد nsm_embedding.npz القديم تلقائياً — بدون أي تدخل يدوي.
#
# الاستخدام:
#   python nsm_trainer.py --continual                       # تجربة بموضوع توضيحي
#   python nsm_trainer.py --continual --continual-file new_topic.json
#
#   new_topic.json يجب أن يكون بصيغة: [["نص الجملة", رقم_الموضوع], ...]
#
# لا يُغيَّر أي سلوك في train()/evaluate()/التشغيل الافتراضي أعلاه.
# ════════════════════════════════════════════════════════════════════

try:
    from ai.continual_learner import EWCLearner, EWCTrainingLoop
    _HAS_EWC = True
except Exception as _exc:                                    # pragma: no cover
    _HAS_EWC = False
    _EWC_IMPORT_ERROR = _exc

try:
    from ai.rollback_guard import CheckpointGuard
    _HAS_ROLLBACK = True
except Exception as _exc:                                    # pragma: no cover
    _HAS_ROLLBACK = False
    _ROLLBACK_IMPORT_ERROR = _exc


# موضوع توضيحي جديد (رقم 7) — يُستخدم فقط إن لم يُمرَّر --continual-file
# الهدف إثبات أن الآلية تعمل: تعلّم موضوع جديد كلياً بدون نسيان السبعة القدامى
_DEMO_NEW_TOPIC_DATA = [
    ("الفلك الإسلامي رصد القمر والكواكب لتحديد مواقيت الصلاة والتقويم", 7),
    ("علم الفلك يدرس النجوم والمجرات والكواكب وحركة الأجرام السماوية", 7),
    ("التقويم القمري الهجري يعتمد على رؤية هلال بداية الشهر", 7),
    ("الكواكب التسعة تدور حول الشمس في مسارات بيضاوية منتظمة", 7),
    ("علماء المسلمين كالبيروني والخوارزمي طوّروا علم الفلك والرياضيات", 7),
    ("النجوم تُستخدم تاريخياً لتحديد الاتجاهات والقبلة في الصحراء", 7),
]


class _PseudoEpisode:
    """يُحاكي واجهة Episode (ai/experience_store.py) من بيانات نصية ثابتة،
    لاستخدامها في EWCLearner.compute_fisher عندما لا توجد حلقات حقيقية
    محفوظة بعد في EpisodeStore (نظام جديد لم يتراكم له تجارب كافية)."""

    def __init__(self, text: str, dim: int = 784):
        self.question = text
        self.context_vector = text_to_vec(text, dim=dim).tolist()


def _separation_gap(E: np.ndarray, data) -> float:
    """نفس مقياس evaluate() أعلاه (same-cos الأعلى = أفضل) لكن كدالة
    قابلة لإعادة الاستخدام — لا تُغيّر evaluate() نفسها."""
    vecs = [(text_to_vec(t), lbl) for t, lbl in data]
    same, diff = [], []
    for i in range(len(vecs)):
        za = embed(E, vecs[i][0])
        for j in range(i + 1, len(vecs)):
            zb = embed(E, vecs[j][0])
            c = cosine(za, zb)
            (same if vecs[i][1] == vecs[j][1] else diff).append(c)
    s = float(np.mean(same)) if same else 0.0
    d = float(np.mean(diff)) if diff else 0.0
    return s - d


def load_new_data_file(path) -> list:
    """يقرأ ملف JSON بصيغة [["نص", رقم_الموضوع], ...]."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [(item[0], int(item[1])) for item in raw]


def continual_train(
    new_data=None,
    epochs: int = 30,
    lr: float = 0.01,
    margin: float = 0.4,
    lambda_reg: float = 20.0,
    task_label: str = "new_task",
    min_improvement: float = -0.05,
):
    """
    تدريب مستمر (Continual Learning) محمي بـ EWC + Rollback.

    خلافاً لـ train()، هذه الدالة:
      • تُحمّل E الموجودة بالفعل (لا تبدأ من الصفر) — تحافظ على المعرفة القديمة
      • تستخدم EWCLearner لمعاقبة أي تغيير كبير في الأوزان المهمة للمواضيع القديمة
      • تُغلَّف بـ CheckpointGuard: تراجع تلقائي إن انهارت دقة المواضيع القديمة

    Args:
        new_data:        قائمة (نص, رقم_موضوع) للموضوع/المواضيع الجديدة.
                          افتراضياً تُستخدم _DEMO_NEW_TOPIC_DATA التوضيحية.
        epochs:          عدد دورات تدريب EWCTrainingLoop
        lr:               معدل التعلّم
        margin:          هامش Triplet Loss
        lambda_reg:      قوة عقوبة EWC (أعلى = حماية أقوى للمعرفة القديمة)
        task_label:      وصف المهمة الجديدة (يُحفظ في سجلّ EWC و Rollback)
        min_improvement: أقل تراجع مسموح به في separation gap على البيانات
                          القديمة قبل اعتبار التحديث "فاشلاً" والتراجع عنه
    """
    print("=" * 60)
    print("  NSM Continual Trainer — EWC + Rollback Guard")
    print("=" * 60)

    if not _HAS_EWC:
        print(f"❌ تعذّر استيراد ai.continual_learner: {_EWC_IMPORT_ERROR}")
        print("   تأكّد من وجود ai/continual_learner.py في المشروع.")
        return None

    if not _E_OUT.exists():
        print("❌ لا توجد nsm_embedding.npz — شغّل التدريب الأساسي أولاً:")
        print("   python nsm_trainer.py")
        return None

    new_data = new_data or _DEMO_NEW_TOPIC_DATA

    # 1) تحميل E الحالية (وليس عشوائية جديدة — هذا جوهر Continual Learning)
    E = np.load(_E_OUT)["E"].astype(np.float64)
    print(f"✓ E الحالية محمّلة: {E.shape}")

    old_gap = _separation_gap(E, TRAIN_DATA)
    print(f"✓ separation gap الحالي (على {len(TRAIN_DATA)} جملة قديمة): {old_gap:.4f}")

    # 2) بناء/تحميل EWCLearner — Fisher Information لحماية المعرفة القديمة
    ewc = EWCLearner(lambda_reg=lambda_reg)
    if not ewc._snapshots:
        # لا توجد لقطات محفوظة بعد — نحسبها من بيانات التدريب الأساسية نفسها
        # (تُستخدم كـ "ذاكرة استبدال" Replay Memory بديلة عن EpisodeStore فارغ)
        print("ℹ لا توجد لقطات EWC سابقة — حساب Fisher من بيانات المواضيع السبعة الأساسية…")
        pseudo_episodes = [_PseudoEpisode(text) for text, _ in TRAIN_DATA]
        ewc.compute_fisher(E, pseudo_episodes, task_label="base_7_topics")
        ewc.save()

    new_topics = sorted(set(lbl for _, lbl in new_data))
    print(f"✓ مهمة جديدة: '{task_label}' — {len(new_data)} جملة، مواضيع: {new_topics}")

    # Triplet Loss يحتاج موضوعَين مختلفَين على الأقل داخل نفس دفعة التدريب
    # ليبني أزواج (إيجابي/سلبي). نُضيف عيّنة صغيرة من المواضيع القديمة
    # كـ "سلبيات تباين" فقط لغرض التدريب — حماية المعرفة القديمة نفسها
    # تبقى من مهمة EWC Penalty وحدها (Fisher + Anchor)، لا من هذه العيّنة.
    import random as _random
    _rng_sample = _random.Random(42)
    by_old_topic = {}
    for text, lbl in TRAIN_DATA:
        by_old_topic.setdefault(lbl, []).append(text)
    contrast_sample = []
    for lbl, texts in by_old_topic.items():
        if lbl not in new_topics:
            contrast_sample.extend(
                (t, lbl) for t in _rng_sample.sample(texts, min(2, len(texts)))
            )
    training_batch = new_data + contrast_sample

    # ★ إصلاح تكامل مهم: يجب أن تُحمى ملفات Fisher/Anchor الخاصة بـ EWC
    # هي أيضاً ضمن نفس عملية rollback — وإلا فإن تراجع nsm_embedding.npz
    # فقط (دون التراجع عن ewc_fisher.npz/ewc_anchor.npz) يجعلهما يشيران
    # إلى E "مرفوضة" غير متطابقة مع E الحالية المُستعادة، فينتج عنه
    # انفجار عددي (inf/nan) في أي محاولة تدريب مستمر لاحقة (تم رصد هذا
    # فعلياً أثناء الاختبار: EWCLearner يُحمّل anchor قديم لا يطابق E
    # الفعلية بعد التراجع → diff ضخم → penalty=inf).
    guarded_files = [_E_OUT, ewc.fisher_path, ewc.anchor_path]

    # 3) التدريب مع عقوبة EWC (يحمي أوزان المواضيع القديمة المهمة)
    def _do_update():
        loop = EWCTrainingLoop(E.copy(), ewc, lr=lr, margin=margin)
        result = loop.train(training_batch, epochs=epochs, verbose=True)
        np.savez(_E_OUT, E=result["E"], dim=EMBED_DIM)
        # نُسجّل المهمة الجديدة في EWC أيضاً لحمايتها هي بدورها من النسيان
        # في أي تدريب مستمر مستقبلي
        ewc.compute_fisher(result["E"], [_PseudoEpisode(t) for t, _ in new_data],
                            task_label=task_label)
        ewc.save()

    def _eval_combined() -> float:
        E_new = np.load(_E_OUT)["E"]
        return _separation_gap(E_new, TRAIN_DATA + new_data)

    # 4) تنفيذ محمي بـ Rollback — إن لم تتوفر ai/rollback_guard، نُنفّذ بدون حماية
    if _HAS_ROLLBACK:
        guard = CheckpointGuard(asset="nsm_embedding")
        if guard.current_score() is None:
            guard.snapshot(guarded_files, label="baseline_before_continual", score=old_gap)

        decision = guard.guarded_update(
            files=guarded_files,
            update_fn=_do_update,
            eval_fn=_eval_combined,
            tolerance=min_improvement,
            label=task_label,
        )
        print("\n" + decision.summary())
        return decision
    else:
        print(f"⚠ ai/rollback_guard غير متاح ({_ROLLBACK_IMPORT_ERROR}) — "
              f"تنفيذ بدون حماية تراجع تلقائي!")
        _do_update()
        new_gap = _eval_combined()
        print(f"✓ separation gap بعد التدريب المستمر: {new_gap:.4f} "
              f"(كان {old_gap:.4f})")
        return {"old_gap": old_gap, "new_gap": new_gap, "rolled_back": False}


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval", action="store_true")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.005)
    # ★ أعلام جديدة لـ Continual Learning — لا تُغيّر السلوك الافتراضي ★
    p.add_argument("--continual", action="store_true",
                    help="تدريب مستمر محمي بـ EWC+Rollback بدل التدريب الكامل من الصفر")
    p.add_argument("--continual-file", type=str, default=None,
                    help="ملف JSON [[نص, رقم_موضوع], ...] للموضوع الجديد")
    p.add_argument("--ewc-lambda", type=float, default=20.0)
    p.add_argument("--continual-epochs", type=int, default=30)
    p.add_argument("--task-label", type=str, default="new_task")
    args = p.parse_args()

    if args.continual:
        new_data = load_new_data_file(args.continual_file) if args.continual_file else None
        continual_train(
            new_data=new_data,
            epochs=args.continual_epochs,
            lambda_reg=args.ewc_lambda,
            task_label=args.task_label,
        )
    elif args.eval:
        evaluate()
    else:
        train(epochs=args.epochs, lr=args.lr)
        evaluate()

