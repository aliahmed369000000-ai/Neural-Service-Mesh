#!/usr/bin/env python3
"""
تحسين دقة الراوتر الهرمي (Group Router).
- راوتر MLP أعمق
- مسقط قابل للتعلم 784→d_model
- label smoothing + أوزان فئات
- جمل تمييزية قوية للفئات الضعيفة سابقاً (فقه/عقيدة/طب/عام)
"""
from __future__ import annotations

import json
import pickle
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
from ai.hierarchical_moe import DEFAULT_CATEGORIES, DEFAULT_SAVE_DIR, HierarchicalMoE
from ai.knowledge_trainer import VectorEncoder
from ai.moe_ckg_bridge import _KEYWORD_TO_CATEGORY

CATEGORY_ORDER = list(DEFAULT_CATEGORIES.keys())

# جمل عالية التمييز لكل فئة (تُكرر لتعزيز الحدود بين الفئات)
DISC: Dict[str, List[str]] = {
    "fiqh": [
        "ما حكم الصلاة في المذهب الشافعي؟",
        "فتوى فقهية في الزكاة والصوم",
        "أحكام الطهارة والوضوء عند الحنفية",
        "مسألة نكاح وطلاق في الفقه المالكي",
        "الحج والعمرة أحكام فقهية",
    ],
    "aqidah": [
        "أركان الإيمان والتوحيد",
        "عقيدة أهل السنة في الأسماء والصفات",
        "مسائل الشرك والقدر في العقيدة",
        "الفرق بين التوحيد والشرك",
    ],
    "tafsir": [
        "تفسير سورة البقرة آية الكرسي",
        "معنى آيات القرآن في سورة الإخلاص",
        "التفسير الموضوعي للقرآن",
    ],
    "hadith": [
        "حديث صحيح البخاري عن الصدق",
        "تخريج حديث وإسناده",
        "علم مصطلح الحديث والرواية",
    ],
    "seerah": [
        "غزوة بدر في السيرة النبوية",
        "هجرة النبي من مكة إلى المدينة",
        "سيرة الخلفاء الراشدين",
    ],
    "medicine": [
        "أعراض المرض والعلاج الطبي",
        "إسعافات أولية للحروق والجروح",
        "التغذية الصحية والفيتامينات",
        "مراجعة الطبيب عند الحمى",
    ],
    "programming": [
        "اكتب دالة python باستخدام list و dict",
        "خوارزمية فرز وAPI بـ FastAPI",
        "كود جافاسكربت وgit commit",
    ],
    "general": [
        "ملخص عام مبسط للمبتدئين",
        "سؤال يومي غير متخصص",
        "شرح مختصر بدون تفاصيل تقنية",
    ],
    "cooking": [
        "وصفة طبخ كبسة بالمكونات والخطوات",
        "طريقة تحضير أكلة عربية في المطبخ",
    ],
    "career": [
        "كتابة سيرة ذاتية لمقابلة عمل",
        "مهارات وظيفية للحصول على وظيفة",
    ],
    "tajweed": [
        "أحكام التجويد والنون الساكنة",
        "مخارج الحروف وحفظ القرآن",
    ],
    "usul": [
        "أصول الفقه والقواعد الفقهية",
        "مقاصد الشريعة الضرورية",
    ],
    "law": [
        "عقد قانوني ودعوى في المحكمة",
        "القانون المدني والتجاري",
    ],
    "environment": [
        "تغير المناخ والتلوث البيئي",
        "الاستدامة وإعادة التدوير",
    ],
}

_TOPICS = [
    "الصلاة", "العلم", "الأسرة", "الصحة", "المال", "التقنية", "الرياضة",
    "البيئة", "التاريخ", "البرمجة", "السفر", "التعليم", "الفن", "القانون",
]


def label_sentence(text: str) -> str:
    for pat, cat in _KEYWORD_TO_CATEGORY:
        if re.search(pat, text or "", re.I):
            return cat
    return "general"


def build_data(per_cat: int = 120, ckg_limit: int = 6000) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    # تمييزية
    for cat, rows in DISC.items():
        for _ in range(25):
            texts.extend(rows)
            labels.extend([cat] * len(rows))
    # قوالب عامة لكل فئة
    for cat in CATEGORY_ORDER:
        for i in range(per_cat):
            t = _TOPICS[i % len(_TOPICS)]
            if cat in DISC:
                base = DISC[cat][i % len(DISC[cat])]
                texts.append(base + f" — سياق {t}")
            else:
                texts.append(f"موضوع متخصص في مجال {cat} حول {t}")
            labels.append(cat)
    # CKG
    ckg = []
    for path in [
        ROOT / "ckg_sentences_v3.pkl",
        ROOT / "ckg_sentences.pkl",
        ROOT / "ckg_sentences_general_ar.pkl",
    ]:
        if not path.is_file():
            continue
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str) and len(item) > 10:
                        ckg.append(item.strip())
        except Exception:
            pass
    random.shuffle(ckg)
    for s in ckg[:ckg_limit]:
        texts.append(s)
        labels.append(label_sentence(s))

    # توازن
    by = defaultdict(list)
    for t, l in zip(texts, labels):
        by[l].append(t)
    target = 220
    rng = random.Random(42)
    out_t, out_l = [], []
    for cat in CATEGORY_ORDER:
        pool = by.get(cat) or [f"مجال {cat}"]
        chosen = (
            rng.sample(pool, target)
            if len(pool) >= target
            else [rng.choice(pool) for _ in range(target)]
        )
        out_t.extend(chosen)
        out_l.extend([cat] * len(chosen))
    paired = list(zip(out_t, out_l))
    rng.shuffle(paired)
    return [a for a, _ in paired], [b for _, b in paired]


class LearnableProjector(nn.Module):
    def __init__(self, d_in: int = 784, d_out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out * 2),
            nn.GELU(),
            nn.Linear(d_out * 2, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=== بيانات محسّنة للراوتر ===", flush=True)
    texts, labels = build_data()
    print(f"n={len(texts)} cats={len(CATEGORY_ORDER)}", flush=True)

    print("=== تشفير ===", flush=True)
    enc = VectorEncoder()
    cat_to_i = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    X = np.zeros((len(texts), 784), np.float64)
    y = np.zeros(len(texts), np.int64)

    def one(i: int):
        v = enc.encode(texts[i], domain="general", importance=0.7, certainty=0.85)
        return i, v, cat_to_i[labels[i]]

    with ThreadPoolExecutor(4) as ex:
        for fut in as_completed([ex.submit(one, i) for i in range(len(texts))]):
            i, v, yi = fut.result()
            X[i] = v
            y[i] = yi

    # تطبيع
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Xt = torch.from_numpy(X.astype(np.float32))
    yt = torch.from_numpy(y).long()

    d_model = 128
    print("=== بناء نموذج براوتر MLP ===", flush=True)
    model = HierarchicalMoE(
        d_model=d_model,
        d_ff=d_model * 4,
        categories={k: list(v) for k, v in DEFAULT_CATEGORIES.items()},
        top_k_groups=3,
        top_k_experts=3,
        dropout=0.05,
        lb_coeff=0.01,
    )
    # تقليل الضوضاء أثناء تحسين الدقة
    model.group_router.noise_std = 0.15
    model.group_router.jitter = True

    projector = LearnableProjector(784, d_model)
    params = list(model.group_router.parameters()) + list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    cc = np.bincount(y, minlength=len(CATEGORY_ORDER)).astype(np.float64)
    cc = np.maximum(cc, 1.0)
    cw = torch.from_numpy((cc.sum() / (len(CATEGORY_ORDER) * cc)).astype(np.float32))

    # train/val split
    n = len(y)
    idx = np.random.permutation(n)
    n_val = max(500, n // 10)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def eval_acc(ids):
        model.eval()
        projector.eval()
        with torch.no_grad():
            xb = projector(Xt[ids])
            pred = model.group_router.gate(xb).argmax(-1)
            return float((pred == yt[ids]).float().mean().item())

    print("=== تدريب الراوتر ===", flush=True)
    bs = 128
    t0 = time.time()
    best_val = 0.0
    best_state = None
    history = []

    for ep in range(1, 21):
        model.train()
        projector.train()
        # تعطيل jitter في نصف التدريب الأخير لدقة أعلى
        model.group_router.jitter = ep < 12
        np.random.shuffle(tr_idx)
        loss_s = acc_s = nb = 0
        for s in range(0, len(tr_idx), bs):
            b = tr_idx[s : s + bs]
            xb = projector(Xt[b])
            yb = yt[b]
            opt.zero_grad()
            logits = model.group_router.gate(xb)
            loss = F.cross_entropy(logits, yb, weight=cw, label_smoothing=0.06)
            # aux خفيف من الراوتر
            _, _, _, aux = model.group_router(xb)
            loss = loss + 0.05 * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            loss_s += float(loss.item())
            acc_s += float((logits.argmax(-1) == yb).float().mean().item())
            nb += 1
        sched.step()
        tr_acc = acc_s / max(nb, 1)
        val_acc = eval_acc(val_idx)
        history.append({"epoch": ep, "train_acc": tr_acc, "val_acc": val_acc})
        print(
            f"  ep{ep:02d}/20  loss={loss_s/nb:.4f}  train={tr_acc:.3f}  val={val_acc:.3f}",
            flush=True,
        )
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {
                "router": {k: v.cpu().clone() for k, v in model.group_router.state_dict().items()},
                "proj": {k: v.cpu().clone() for k, v in projector.state_dict().items()},
            }

    if best_state:
        model.group_router.load_state_dict(best_state["router"])
        projector.load_state_dict(best_state["proj"])

    # دقة لكل فئة على كل البيانات
    model.eval()
    projector.eval()
    per = {}
    with torch.no_grad():
        xb_all = projector(Xt)
        pred_all = model.group_router.gate(xb_all).argmax(-1)
        overall = float((pred_all == yt).float().mean().item())
        for i, c in enumerate(CATEGORY_ORDER):
            m = yt == i
            if int(m.sum()) == 0:
                continue
            per[c] = float((pred_all[m] == yt[m]).float().mean().item())
    macro = float(np.mean(list(per.values()))) if per else 0.0
    elapsed = round(time.time() - t0, 1)
    print(f"=== overall={overall:.3f} macro={macro:.3f} best_val={best_val:.3f} time={elapsed}s ===", flush=True)
    print("أفضل:", sorted(per.items(), key=lambda x: -x[1])[:8], flush=True)
    print("أضعف:", sorted(per.items(), key=lambda x: x[1])[:6], flush=True)

    # حفظ المسقط كـ npy تقريبي (متوسط أوزان للطبقة الأخيرة للمسقط) + state كامل
    DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"projector": projector.state_dict(), "d_model": d_model},
        DEFAULT_SAVE_DIR / "moe_learnable_projector.pt",
    )
    # للتوافق مع الجسر: مصفوفة تقريبية من طبقة خطية فعّالة عبر عينات
    with torch.no_grad():
        # استخدم Jacobian تقريبي: ضوضاء أساس
        eye = torch.eye(784)
        # تقريب خطي محلي عند الصفر
        base = projector(torch.zeros(1, 784)).squeeze(0)
        cols = []
        eps = 1e-2
        for i in range(784):
            x = torch.zeros(1, 784)
            x[0, i] = eps
            cols.append(((projector(x).squeeze(0) - base) / eps).numpy())
        P = np.stack(cols, axis=1)  # (d_model, 784)
    np.save(DEFAULT_SAVE_DIR / "moe_projector.npy", P.astype(np.float64))

    # حفظ النموذج (خبراء منتهيّة + راوتر محسّن)
    path = model.save(DEFAULT_SAVE_DIR / "hierarchical_moe.pt")
    meta = {
        "router_accuracy_train": {
            "overall_acc": overall,
            "macro_acc": macro,
            "best_val_acc": best_val,
            "per_category_acc": per,
            "history": history,
            "elapsed_sec": elapsed,
            "label_smoothing": 0.06,
            "router": "MLP+LayerNorm",
            "learnable_projector": True,
        }
    }
    side = path.with_suffix(".json")
    try:
        base = json.loads(side.read_text(encoding="utf-8")) if side.is_file() else {}
    except Exception:
        base = {}
    base.update(meta)
    base["group_order"] = list(model._group_order)
    base["experts"] = {c: model.groups[c]._id_order for c in model._group_order}
    side.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
