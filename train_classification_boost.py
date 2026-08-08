#!/usr/bin/env python3
"""
دفعة تحسين تصنيف: أزواج التباس + مزج مسقط/راوتر موجود.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.hierarchical_moe import DEFAULT_CATEGORIES, DEFAULT_SAVE_DIR, HierarchicalMoE
from ai.knowledge_trainer import VectorEncoder

CATEGORY_ORDER = list(DEFAULT_CATEGORIES.keys())

# أزواج التباس شائعة (نص، فئة صحيحة)
HARD = [
    ("سيرة ذاتية للعمل", "career"),
    ("كتابة cv ومهارات وظيفية", "career"),
    ("مقابلة عمل لوظيفة مطور", "career"),
    ("السيرة النبوية وغزوة بدر", "seerah"),
    ("هجرة النبي إلى المدينة", "seerah"),
    ("وصفة طبخ كبسة", "cooking"),
    ("طريقة تحضير أكلة في المطبخ", "cooking"),
    ("أعراض المرض ومراجعة الطبيب", "medicine"),
    ("إسعافات أولية للحروق", "medicine"),
    ("قانون نيوتن الثاني في الفيزياء", "science"),
    ("تجربة كيميائية في المختبر", "science"),
    ("عقد قانوني في المحكمة", "law"),
    ("دعوى قضائية ومحام", "law"),
    ("ما حكم الصلاة الشافعي", "fiqh"),
    ("أركان الإيمان والتوحيد", "aqidah"),
    ("كود python و list", "programming"),
    ("تغير المناخ والتلوث", "environment"),
    ("أحكام التجويد والنون الساكنة", "tajweed"),
    ("أصول الفقه ومقاصد الشريعة", "usul"),
    ("خطة تسويق لمنتج", "business"),
]


class LearnableProjector(nn.Module):
    def __init__(self, d_out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, d_out * 2),
            nn.GELU(),
            nn.Linear(d_out * 2, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)
    path = DEFAULT_SAVE_DIR / "hierarchical_moe.pt"
    model = HierarchicalMoE.load(path, map_location="cpu")
    model.group_router.noise_std = 0.05
    model.group_router.jitter = False

    proj_path = DEFAULT_SAVE_DIR / "moe_learnable_projector.pt"
    projector = LearnableProjector(model.d_model)
    if proj_path.is_file():
        blob = torch.load(proj_path, map_location="cpu", weights_only=False)
        projector.load_state_dict(blob["projector"])

    enc = VectorEncoder()
    cat_to_i = {c: i for i, c in enumerate(model._group_order)}

    rows = []
    for text, cat in HARD:
        for _ in range(15):
            rows.append((text, cat))
        # تنويع خفيف
        rows.append((text + " مع شرح مبسط", cat))
        rows.append(("أريد معرفة: " + text, cat))

    random.shuffle(rows)
    X = []
    y = []
    for t, c in rows:
        v = enc.encode(t, domain="general", importance=0.7, certainty=0.9)
        v = v / (np.linalg.norm(v) + 1e-8)
        X.append(v.astype(np.float32))
        y.append(cat_to_i.get(c, cat_to_i.get("general", 0)))
    Xt = torch.from_numpy(np.stack(X))
    yt = torch.tensor(y, dtype=torch.long)

    params = list(model.group_router.parameters()) + list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=8e-4, weight_decay=1e-4)

    model.train()
    projector.train()
    print(f"fine-tune hard cases n={len(y)}", flush=True)
    for ep in range(1, 16):
        opt.zero_grad()
        logits = model.group_router.gate(projector(Xt))
        loss = F.cross_entropy(logits, yt, label_smoothing=0.04)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        acc = (logits.argmax(-1) == yt).float().mean().item()
        if ep % 3 == 0 or ep == 1:
            print(f"  ep{ep} loss={loss.item():.4f} acc={acc:.3f}", flush=True)

    model.eval()
    projector.eval()
    with torch.no_grad():
        pred = model.group_router.gate(projector(Xt)).argmax(-1)
        print("final hard-acc", float((pred == yt).float().mean()), flush=True)

    model.save(path)
    torch.save({"projector": projector.state_dict(), "d_model": model.d_model}, proj_path)

    # حدّث npy توافقياً
    with torch.no_grad():
        base = projector(torch.zeros(1, 784)).squeeze(0)
        cols = []
        eps = 1e-2
        for i in range(784):
            x = torch.zeros(1, 784)
            x[0, i] = eps
            cols.append(((projector(x).squeeze(0) - base) / eps).numpy())
        P = np.stack(cols, axis=1)
    np.save(DEFAULT_SAVE_DIR / "moe_projector.npy", P.astype(np.float64))

    side = path.with_suffix(".json")
    try:
        meta = json.loads(side.read_text(encoding="utf-8")) if side.is_file() else {}
    except Exception:
        meta = {}
    meta["classification_boost"] = {
        "hard_pairs": len(HARD),
        "note": "hybrid keywords + hard-negative router fine-tune",
    }
    side.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
