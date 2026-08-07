#!/usr/bin/env python3
"""
تدريب Hierarchical Dynamic MoE على جمل CKG (إشراف ضعيف بالفئات).
================================================================
1) يحمّل ckg_sentences_v3.pkl (أو البديل)
2) يصنّف كل جملة لفئة MoE بالكلمات المفتاحية
3) يشفرها إلى متجه 784 عبر VectorEncoder
4) يدرّب راوتر الفئات + الشبكة الهرمية (main + load-balance aux)
5) يحفظ الأوزان في artifacts/hierarchical_moe/

الاستخدام:
    python3 train_hierarchical_moe.py
    python3 train_hierarchical_moe.py --epochs 5 --max-samples 4000
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent

from ai.hierarchical_moe import (
    HierarchicalMoE,
    build_default_moe,
    DEFAULT_SAVE_DIR,
    DEFAULT_CATEGORIES,
)
from ai.knowledge_trainer import VectorEncoder
from ai.moe_ckg_bridge import _KEYWORD_TO_CATEGORY, map_cluster_to_category

CATEGORY_ORDER = list(DEFAULT_CATEGORIES.keys())  # fiqh, tafsir, ...


# جمل بذرية للفئات غير المغطاة جيداً في CKG (برمجة، رياضة، …)
SEED_SENTENCES: Dict[str, List[str]] = {
    "programming": [
        "كيف أكتب دالة python لحساب المتوسط؟",
        "ما الفرق بين list و dict في البرمجة؟",
        "شرح خوارزمية البحث الثنائي مع مثال كود",
        "كيف أبني API بسيط باستخدام FastAPI؟",
        "ما هو git commit وكيف أرفع التغييرات؟",
        "تحسين أداء حلقة for في جافاسكربت",
        "مفهوم class والوراثة في البرمجة الكائنية",
        "تصحيح خطأ IndexError في بايثون",
    ],
    "sports": [
        "تمارين لياقة يومية للمبتدئين",
        "قوانين كرة القدم في ركلة الجزاء",
        "كيف أحسّن التحمل في الجري؟",
        "أفضل تمارين بناء عضلات البطن",
        "نظام غذائي مرتبط بالتدريب الرياضي",
        "استراتيجية فريق في مباراة كرة قدم",
    ],
    "science": [
        "قانون نيوتن الثاني في الفيزياء",
        "تفاعل كيميائي بين حمض وقاعدة",
        "دورة الخلية في علم الأحياء",
        "ما هو التمثيل الضوئي في النباتات؟",
        "شرح البنية الذرية بشكل مبسط",
    ],
    "math": [
        "حل معادلة من الدرجة الثانية",
        "مساحة المثلث باستخدام الارتفاع",
        "مقدمة في الإحصاء والمتوسط الحسابي",
        "ما هو المشتق في التفاضل؟",
        "ضرب المصفوفات في الجبر الخطي",
    ],
    "medicine": [
        "أعراض نزلات البرد الشائعة وعلاجها المنزلي",
        "أسس التغذية السليمة اليومية",
        "الوقاية من الأمراض المعدية",
        "متى يجب مراجعة الطبيب عند الحمى؟",
    ],
    "technology": [
        "ما هو الذكاء الاصطناعي وكيف يعمل؟",
        "أساسيات أمن الشبكات والحماية",
        "الفرق بين CPU و GPU",
        "مفاهيم الحوسبة السحابية",
    ],
    "business": [
        "خطة تسويق لمنتج جديد",
        "إدارة التدفق النقدي في شركة ناشئة",
        "مهارات القيادة وإدارة الفرق",
        "مقدمة في الاستثمار والتمويل",
    ],
    "education": [
        "أساليب تدريس فعّالة للطلاب",
        "تصميم منهج تعليمي واضح",
        "تحفيز التعلم الذاتي لدى المتعلمين",
    ],
    "history": [
        "أحداث مهمة في التاريخ الإسلامي",
        "أسباب الحرب العالمية الثانية",
        "نشأة الحضارات القديمة",
    ],
    "media": [
        "كتابة محتوى جذاب للمنصات الاجتماعية",
        "أساسيات الصحافة الرقمية",
        "استراتيجية نشر على يوتيوب",
    ],
    "psychology": [
        "إدارة القلق بطرق عملية",
        "مراحل النمو النفسي للطفل",
        "تحفيز السلوك الإيجابي",
    ],
    "law": [
        "مبادئ العقد في القانون المدني",
        "حقوق الأطراف في الدعوى القضائية",
        "مفهوم التشريع والأنظمة",
    ],
    "arabic": [
        "قواعد النحو في إعراب الفعل",
        "أساليب البلاغة في التشبيه",
        "ميزان الصرف للأفعال العربية",
        "ترجمة نص عربي إلى لغة أخرى بدقة",
    ],
}


def label_sentence(text: str) -> str:
    for pat, cat in _KEYWORD_TO_CATEGORY:
        if re.search(pat, text or "", re.I):
            return cat
    low = (text or "").lower()
    for key, cat in [
        ("quran", "tafsir"),
        ("fiqh", "fiqh"),
        ("hadith", "hadith"),
        ("aqidah", "aqidah"),
        ("seerah", "seerah"),
        ("arabic", "arabic"),
        ("nahw", "arabic"),
        ("python", "programming"),
        ("code", "programming"),
        ("مجال", "general"),
    ]:
        if key in low:
            return cat
    return "general"


def load_sentences(max_samples: int) -> List[str]:
    candidates = [
        ROOT / "ckg_sentences_v3.pkl",
        ROOT / "ckg_sentences.pkl",
        ROOT / "ckg_sentences_v2.pkl",
        ROOT / "ckg_sentences_general_ar.pkl",
    ]
    sentences: List[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str) and item.strip():
                        sentences.append(item.strip())
                    elif isinstance(item, dict):
                        t = item.get("text") or item.get("sentence") or item.get("s")
                        if t:
                            sentences.append(str(t).strip())
            print(f"  + {path.name}: {len(data) if hasattr(data, '__len__') else '?'} → إجمالي {len(sentences)}")
        except Exception as e:
            print(f"  ! فشل {path.name}: {e}")
    # بذور التخصصات العامة (تُكرر لتقوية الإشارة)
    for cat, seeds in SEED_SENTENCES.items():
        for _ in range(40):
            for s in seeds:
                sentences.append(s)
        print(f"  + seeds[{cat}]: {len(seeds)}×40")
    # إزالة تكرار مع الحفاظ على ترتيب تقريبي
    seen = set()
    uniq = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    # أعد إضافة البذور بعد إزالة التكرار (مكررة عمداً للتوازن)
    for cat, seeds in SEED_SENTENCES.items():
        for _ in range(30):
            uniq.extend(seeds)
    random.shuffle(uniq)
    if max_samples > 0 and len(uniq) > max_samples:
        uniq = uniq[:max_samples]
    return uniq


def encode_batch(texts: List[str], labels: List[str], encoder: VectorEncoder) -> Tuple[np.ndarray, np.ndarray]:
    X = np.zeros((len(texts), 784), dtype=np.float64)
    y = np.zeros(len(texts), dtype=np.int64)
    cat_to_i = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    # domain في VectorEncoder محدود — نمرّر general للتخصصات الجديدة مع النص الحامل للإشارة
    known_domains = {
        "fiqh", "tafsir", "hadith", "aqidah", "seerah", "arabic", "general",
        "history", "science", "education",
    }
    for i, (t, lab) in enumerate(zip(texts, labels)):
        domain = lab if lab in known_domains else "general"
        X[i] = encoder.encode(t, domain=domain, importance=0.6, certainty=0.75)
        y[i] = cat_to_i.get(lab, cat_to_i["general"])
    return X, y


def build_projector(d_model: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return (rng.randn(d_model, 784).astype(np.float64) * (2.0 / np.sqrt(784)))


def project_np(X: np.ndarray, P: np.ndarray) -> torch.Tensor:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norms
    out = Xn @ P.T
    return torch.from_numpy(out.astype(np.float32))


def train(
    epochs: int = 4,
    max_samples: int = 6000,
    batch_size: int = 64,
    lr: float = 1e-3,
    d_model: int = 128,
    seed: int = 42,
) -> HierarchicalMoE:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=== تحميل الجمل ===")
    sentences = load_sentences(max_samples)
    if len(sentences) < 50:
        raise RuntimeError(f"جمل غير كافية للتدريب: {len(sentences)}")

    labels = [label_sentence(s) for s in sentences]
    counts = Counter(labels)
    print("توزيع الفئات (قبل التوازن):", dict(counts))

    # توازن بالعينات: سقف/أرضية لكل فئة
    by_cat: Dict[str, List[str]] = {c: [] for c in CATEGORY_ORDER}
    for s, lab in zip(sentences, labels):
        by_cat.setdefault(lab, []).append(s)
    target_per = max(80, min(600, max_samples // max(1, len(CATEGORY_ORDER))))
    balanced_s: List[str] = []
    balanced_l: List[str] = []
    rng = random.Random(seed)
    for cat in CATEGORY_ORDER:
        pool = by_cat.get(cat) or []
        if not pool:
            continue
        if len(pool) >= target_per:
            chosen = rng.sample(pool, target_per)
        else:
            chosen = [rng.choice(pool) for _ in range(target_per)]
        balanced_s.extend(chosen)
        balanced_l.extend([cat] * len(chosen))
    # اخلط
    paired = list(zip(balanced_s, balanced_l))
    rng.shuffle(paired)
    sentences = [a for a, _ in paired]
    labels = [b for _, b in paired]
    counts = Counter(labels)
    print("توزيع الفئات (بعد التوازن):", dict(counts), "n=", len(sentences))

    print("=== تشفير 784 ===")
    encoder = VectorEncoder()
    X, y = encode_batch(sentences, labels, encoder)

    # توازن تقريبي: أوزان عكسية للفئات
    class_count = np.bincount(y, minlength=len(CATEGORY_ORDER)).astype(np.float64)
    class_count = np.maximum(class_count, 1.0)
    class_w = (class_count.sum() / (len(CATEGORY_ORDER) * class_count)).astype(np.float32)
    weight_tensor = torch.from_numpy(class_w)

    print("=== بناء النموذج ===")
    path = DEFAULT_SAVE_DIR / "hierarchical_moe.pt"
    model = None
    if path.is_file():
        try:
            model = HierarchicalMoE.load(path)
            if set(model._group_order) != set(CATEGORY_ORDER):
                print(
                    f"  فئات مختلفة (محمّل={len(model._group_order)}، مطلوب={len(CATEGORY_ORDER)}) — إعادة بناء"
                )
                model = None
            else:
                print(f"  استكمال من {path} ({model.total_experts()} خبراء)")
        except Exception as e:
            print(f"  تحميل فشل ({e}) — نموذج جديد")
            model = None
    if model is None:
        model = build_default_moe(d_model=d_model, top_k_groups=2, top_k_experts=3)
        print(f"  نموذج جديد: {len(CATEGORY_ORDER)} فئات، {model.total_experts()} خبراء")

    # تأكد من تطابق d_model
    if model.d_model != d_model:
        print(f"  d_model المحمّل={model.d_model} — نستخدمه")
        d_model = model.d_model

    P = build_projector(d_model, seed=seed)
    # حفظ المسقط مع الميتا
    projector_path = DEFAULT_SAVE_DIR / "moe_projector.npy"
    DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(projector_path, P)

    device = torch.device("cpu")
    model = model.to(device)
    model.train()

    # درّب الراوتر الرئيسي + الخبراء
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)

    n = len(y)
    idx_all = np.arange(n)
    history = []

    print(f"=== تدريب: epochs={epochs} batch={batch_size} n={n} ===")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        np.random.shuffle(idx_all)
        total_loss = 0.0
        total_ce = 0.0
        total_aux = 0.0
        total_acc = 0.0
        nb = 0
        for start in range(0, n, batch_size):
            batch_idx = idx_all[start : start + batch_size]
            xb = project_np(X[batch_idx], P).to(device)
            yb = torch.from_numpy(y[batch_idx]).long().to(device)

            opt.zero_grad()
            # راوتر المجموعات — CE على logits (وليس softmax)
            logits = model.group_router.gate(xb)
            ce = F.cross_entropy(logits, yb, weight=weight_tensor.to(device))
            g_idx, g_w, probs, aux = model.group_router(xb)
            # forward كامل (خبراء) + هدف إعادة البناء الخفيف
            out, aux2 = model(xb)
            recon = F.mse_loss(out, xb.detach())
            loss = ce + 0.5 * recon + 0.25 * (aux + aux2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            pred = logits.argmax(dim=-1)
            acc = (pred == yb).float().mean().item()
            total_loss += float(loss.item())
            total_ce += float(ce.item())
            total_aux += float((aux + aux2).item())
            total_acc += acc
            nb += 1

        avg = {
            "epoch": ep,
            "loss": total_loss / max(nb, 1),
            "ce": total_ce / max(nb, 1),
            "aux": total_aux / max(nb, 1),
            "acc": total_acc / max(nb, 1),
        }
        history.append(avg)
        print(
            f"  epoch {ep}/{epochs}  loss={avg['loss']:.4f}  "
            f"ce={avg['ce']:.4f}  aux={avg['aux']:.4f}  acc={avg['acc']:.3f}"
        )

    elapsed = time.time() - t0
    model.eval()

    # تقييم سريع
    with torch.no_grad():
        xb = project_np(X[: min(512, n)], P)
        yb = torch.from_numpy(y[: min(512, n)]).long()
        _, _, probs, _ = model.group_router(xb)
        acc = (probs.argmax(-1) == yb).float().mean().item()
    print(f"=== انتهى خلال {elapsed:.1f}s | acc@eval≈{acc:.3f} ===")

    # حفظ
    saved = model.save(path)
    meta_extra = {
        "train": {
            "epochs": epochs,
            "max_samples": max_samples,
            "n_used": n,
            "batch_size": batch_size,
            "lr": lr,
            "d_model": d_model,
            "label_counts": dict(counts),
            "history": history,
            "eval_acc": acc,
            "elapsed_sec": round(elapsed, 2),
            "projector": str(projector_path.relative_to(ROOT)),
        }
    }
    side = path.with_suffix(".json")
    try:
        base = json.loads(side.read_text(encoding="utf-8")) if side.is_file() else {}
    except Exception:
        base = {}
    base.update(meta_extra)
    side.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"حُفظ: {saved}")
    print(model.summary())
    print(model.load_balance_report())
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    args = ap.parse_args()
    train(
        epochs=args.epochs,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
    )


if __name__ == "__main__":
    main()
