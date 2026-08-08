#!/usr/bin/env python3
"""تدريب واسع متوازٍ لـ Hierarchical MoE — راوتر ثم كل الخبراء."""
from __future__ import annotations

import json
import pickle
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
from ai.hierarchical_moe import DEFAULT_CATEGORIES, DEFAULT_SAVE_DIR, HierarchicalMoE
from ai.knowledge_trainer import VectorEncoder
from ai.moe_ckg_bridge import _KEYWORD_TO_CATEGORY

CATEGORY_ORDER = list(DEFAULT_CATEGORIES.keys())

_TOPICS = [
    "الصلاة", "الزكاة", "العلم", "العمل", "الأسرة", "الصحة", "الوقت",
    "الأخلاق", "الشباب", "المال", "التعليم", "التقنية", "الصدق", "الرياضة",
    "البيئة", "التاريخ", "اللغة", "البرمجة", "السوق", "القانون", "الفن",
]

_TEMPLATES: Dict[str, List[str]] = {
    "fiqh": ["ما حكم {x} في الفقه؟", "رأي المذاهب في {x}", "فتوى عن {x}"],
    "tafsir": ["تفسير يتعلق بـ {x}", "معنى قرآني عن {x}"],
    "hadith": ["حديث عن {x}", "تخريج رواية في {x}"],
    "aqidah": ["مسألة عقدية في {x}", "التوحيد و{x}"],
    "seerah": ["من السيرة النبوية عن {x}", "الصحابة و{x}"],
    "usul": ["قاعدة أصولية في {x}", "مقاصد الشريعة و{x}"],
    "tajweed": ["حكم تجويدي في {x}", "حفظ وتجويد متعلق بـ {x}"],
    "islamic_finance": ["تمويل إسلامي لـ {x}", "زكاة متعلقة بـ {x}"],
    "arabic": ["إعراب وبيان عن {x}", "بلاغة التعبير عن {x}"],
    "literature": ["نص أدبي عن {x}", "شعر يتناول {x}"],
    "programming": ["كود python لـ {x}", "خوارزمية لـ {x}", "API لخدمة {x}"],
    "technology": ["ذكاء اصطناعي لـ {x}", "أمن سيبراني لـ {x}", "سحابة لـ {x}"],
    "science": ["تفسير علمي لـ {x}", "تجربة علمية عن {x}"],
    "math": ["مسألة رياضيات في {x}", "إحصاء عن {x}"],
    "astronomy": ["فلك و{x}", "رصد {x}"],
    "geography": ["جغرافيا {x}", "خريطة {x}"],
    "engineering": ["تصميم هندسي لـ {x}", "حسابات هندسية لـ {x}"],
    "medicine": ["صحة و{x}", "إسعاف في {x}", "تغذية عند {x}"],
    "psychology": ["جانب نفسي في {x}", "سلوك متعلق بـ {x}"],
    "sports": ["تمرين رياضي لـ {x}", "قانون رياضي في {x}"],
    "family": ["نصيحة أسرية عن {x}", "تربية و{x}"],
    "cooking": ["وصفة طبخ لـ {x}", "تحضير {x}"],
    "travel": ["رحلة وسفر إلى {x}", "عمرة وجانب {x}"],
    "environment": ["بيئة ومناخ و{x}", "استدامة في {x}"],
    "agriculture": ["زراعة {x}", "مزرعة و{x}"],
    "business": ["خطة عمل لـ {x}", "تسويق {x}"],
    "economy": ["تحليل اقتصادي لـ {x}", "سوق {x}"],
    "law": ["جانب قانوني في {x}", "نظام يتعلق بـ {x}"],
    "career": ["وظيفة في {x}", "سيرة ذاتية لمسار {x}"],
    "education": ["تدريس {x}", "منهج عن {x}"],
    "history": ["تاريخ {x}", "حدث تاريخي عن {x}"],
    "media": ["محتوى إعلامي عن {x}", "منشور عن {x}"],
    "philosophy": ["تأمل فلسفي في {x}", "أخلاق و{x}"],
    "art": ["تصميم فني لـ {x}", "خط عربي و{x}"],
    "general": ["شرح عام عن {x}", "ملخص لموضوع {x}"],
}


def label_sentence(text: str) -> str:
    for pat, cat in _KEYWORD_TO_CATEGORY:
        if re.search(pat, text or "", re.I):
            return cat
    return "general"


def build_corpus(per_cat: int = 90, ckg_limit: int = 8000) -> Tuple[List[str], List[str]]:
    print("=== بناء بيانات واسعة ===", flush=True)
    texts: List[str] = []
    labels: List[str] = []
    for cat in CATEGORY_ORDER:
        tpls = _TEMPLATES.get(cat, _TEMPLATES["general"])
        for i in range(per_cat):
            s = tpls[i % len(tpls)].replace("{x}", _TOPICS[i % len(_TOPICS)])
            texts.append(s)
            labels.append(cat)
    print(f"  قوالب: {len(texts)}", flush=True)

    ckg: List[str] = []
    for path in [
        ROOT / "ckg_sentences_v3.pkl",
        ROOT / "ckg_sentences.pkl",
        ROOT / "ckg_sentences_v2.pkl",
        ROOT / "ckg_sentences_general_ar.pkl",
    ]:
        if not path.is_file():
            continue
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str) and len(item.strip()) > 8:
                        ckg.append(item.strip())
        except Exception:
            pass
    random.shuffle(ckg)
    ckg = ckg[:ckg_limit]
    for s in ckg:
        texts.append(s)
        labels.append(label_sentence(s))
    print(f"  + CKG {len(ckg)} → {len(texts)}", flush=True)

    by: Dict[str, List[str]] = defaultdict(list)
    for t, l in zip(texts, labels):
        by[l].append(t)
    target = max(per_cat, 200)
    bal_t, bal_l = [], []
    rng = random.Random(42)
    for cat in CATEGORY_ORDER:
        pool = by.get(cat) or [f"موضوع في مجال {cat}"]
        chosen = (
            rng.sample(pool, target)
            if len(pool) >= target
            else [rng.choice(pool) for _ in range(target)]
        )
        bal_t.extend(chosen)
        bal_l.extend([cat] * len(chosen))
    paired = list(zip(bal_t, bal_l))
    rng.shuffle(paired)
    print("  توزيع متوازن:", len(paired), "عينة", flush=True)
    return [a for a, _ in paired], [b for _, b in paired]


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    texts, labels = build_corpus(per_cat=90, ckg_limit=8000)
    print("=== تشفير متوازٍ ===", flush=True)
    enc = VectorEncoder()
    cat_to_i = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    X = np.zeros((len(texts), 784), np.float64)
    y = np.zeros(len(texts), np.int64)

    def one(i: int):
        v = enc.encode(texts[i], domain="general", importance=0.65, certainty=0.8)
        return i, v, cat_to_i[labels[i]]

    with ThreadPoolExecutor(4) as ex:
        for fut in as_completed([ex.submit(one, i) for i in range(len(texts))]):
            i, v, yi = fut.result()
            X[i] = v
            y[i] = yi
    print("encoded", len(texts), flush=True)

    d_model = 128
    P = np.random.RandomState(42).randn(d_model, 784).astype(np.float64) * (
        2 / np.sqrt(784)
    )
    DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DEFAULT_SAVE_DIR / "moe_projector.npy", P)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xp = torch.from_numpy(((X / norms) @ P.T).astype(np.float32))
    yt = torch.from_numpy(y).long()

    model = HierarchicalMoE(
        d_model=d_model,
        d_ff=d_model * 4,
        categories={k: list(v) for k, v in DEFAULT_CATEGORIES.items()},
        top_k_groups=3,
        top_k_experts=3,
        dropout=0.05,
        lb_coeff=0.01,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"فئات={len(CATEGORY_ORDER)} خبراء={model.total_experts()} params={n_params:,}", flush=True)

    cc = np.bincount(y, minlength=len(CATEGORY_ORDER)).astype(np.float64)
    cc = np.maximum(cc, 1.0)
    cw = torch.from_numpy((cc.sum() / (len(CATEGORY_ORDER) * cc)).astype(np.float32))
    bs = 256
    hist = []
    t0 = time.time()

    print("=== Phase1: راوتر الفئات (كل الفئات بالتوازي في الدفعات) ===", flush=True)
    opt = torch.optim.Adam(model.group_router.parameters(), lr=2e-3)
    for ep in range(1, 10):
        idx = np.random.permutation(len(y))
        loss_s = acc_s = n = 0
        for s in range(0, len(y), bs):
            b = idx[s : s + bs]
            xb, yb = Xp[b], yt[b]
            opt.zero_grad()
            logits = model.group_router.gate(xb)
            loss = F.cross_entropy(logits, yb, weight=cw)
            loss.backward()
            opt.step()
            loss_s += float(loss.item())
            acc_s += float((logits.argmax(-1) == yb).float().mean().item())
            n += 1
        print(f"  p1 ep{ep}/9 loss={loss_s/n:.4f} acc={acc_s/n:.3f}", flush=True)
        hist.append({"phase": 1, "epoch": ep, "acc": acc_s / n})

    print("=== Phase2: كل الخبراء (forward هرمي متوازٍ داخل الدفعة) ===", flush=True)
    opt2 = torch.optim.Adam(model.parameters(), lr=8e-4)
    model.train()
    for ep in range(1, 5):
        idx = np.random.permutation(len(y))
        loss_s = acc_s = n = 0
        for s in range(0, len(y), bs):
            b = idx[s : s + bs]
            xb, yb = Xp[b], yt[b]
            opt2.zero_grad()
            logits = model.group_router.gate(xb)
            ce = F.cross_entropy(logits, yb, weight=cw)
            out, aux = model(xb)
            loss = ce + 0.35 * F.mse_loss(out, xb.detach()) + 0.15 * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            loss_s += float(loss.item())
            acc_s += float((logits.argmax(-1) == yb).float().mean().item())
            n += 1
        print(f"  p2 ep{ep}/4 loss={loss_s/n:.4f} acc={acc_s/n:.3f}", flush=True)
        hist.append({"phase": 2, "epoch": ep, "acc": acc_s / n})

    model.eval()
    per = {}
    with torch.no_grad():
        for i, c in enumerate(CATEGORY_ORDER):
            m = yt == i
            if int(m.sum()) == 0:
                continue
            pred = model.group_router.gate(Xp[m][:200]).argmax(-1)
            per[c] = float((pred == yt[m][:200]).float().mean().item())
    macro = float(np.mean(list(per.values()))) if per else 0.0
    elapsed = round(time.time() - t0, 1)
    print(f"=== انتهى {elapsed}s | macro-acc={macro:.3f} ===", flush=True)
    print("أفضل:", sorted(per.items(), key=lambda x: -x[1])[:6], flush=True)
    print("أضعف:", sorted(per.items(), key=lambda x: x[1])[:5], flush=True)

    path = model.save(DEFAULT_SAVE_DIR / "hierarchical_moe.pt")
    meta = {
        "wide_parallel_train": {
            "d_model": d_model,
            "n_samples": int(len(y)),
            "n_experts": model.total_experts(),
            "n_categories": len(CATEGORY_ORDER),
            "macro_acc": macro,
            "per_category_acc": per,
            "history": hist,
            "elapsed_sec": elapsed,
            "phases": "router_then_all_experts_parallel_batches",
            "note": "MoE هرمي متخصص (~17M) — ليس بديلاً عن LLM بمليارات المعاملات",
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
