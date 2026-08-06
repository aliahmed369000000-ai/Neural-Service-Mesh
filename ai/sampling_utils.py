"""
تقنيات عيّنة حديثة للتوليد (nucleus / top-k / repetition penalty).
تعمل على توزيع احتمالات numpy — مشتركة بين ArabicTransformer والمسارات الأخرى.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def apply_temperature(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64).flatten()
    p = np.clip(p, 1e-12, None)
    if temperature is None or abs(temperature - 1.0) < 1e-8:
        p = p / p.sum()
        return p
    logp = np.log(p) / max(temperature, 1e-6)
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()


def top_k_filter(probs: np.ndarray, top_k: int = 0) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64).flatten().copy()
    if not top_k or top_k <= 0 or top_k >= p.size:
        return p / max(p.sum(), 1e-12)
    idx = np.argpartition(p, -top_k)[-top_k:]
    mask = np.zeros_like(p)
    mask[idx] = p[idx]
    s = mask.sum()
    return mask / s if s > 0 else p / max(p.sum(), 1e-12)


def top_p_filter(probs: np.ndarray, top_p: float = 0.9) -> np.ndarray:
    """Nucleus sampling (Holtzman et al.)."""
    p = np.asarray(probs, dtype=np.float64).flatten().copy()
    if top_p is None or top_p >= 1.0 or top_p <= 0:
        return p / max(p.sum(), 1e-12)
    order = np.argsort(p)[::-1]
    sorted_p = p[order]
    cum = np.cumsum(sorted_p)
    keep = cum <= top_p
    if not np.any(keep):
        keep[0] = True
    else:
        # أبقِ الرمز الذي يتجاوز العتبة قليلاً
        last = np.where(keep)[0]
        if len(last) < len(keep):
            keep[last[-1] + 1 if last[-1] + 1 < len(keep) else last[-1]] = True
    mask = np.zeros_like(p)
    mask[order[keep]] = p[order[keep]]
    s = mask.sum()
    return mask / s if s > 0 else p / max(p.sum(), 1e-12)


def apply_repetition_penalty(
    probs: np.ndarray,
    recent_ids: list,
    penalty: float = 1.15,
) -> np.ndarray:
    """يخفض احتمال الرموز التي ظهرت مؤخراً."""
    p = np.asarray(probs, dtype=np.float64).flatten().copy()
    if not recent_ids or penalty is None or penalty <= 1.0:
        return p / max(p.sum(), 1e-12)
    for i in set(int(x) for x in recent_ids):
        if 0 <= i < p.size:
            p[i] = p[i] / penalty
    return p / max(p.sum(), 1e-12)


def sample_token(
    probs: np.ndarray,
    temperature: float = 0.85,
    top_k: int = 50,
    top_p: float = 0.92,
    recent_ids: Optional[list] = None,
    repetition_penalty: float = 1.1,
) -> int:
    p = apply_temperature(probs, temperature)
    p = apply_repetition_penalty(p, recent_ids or [], repetition_penalty)
    p = top_k_filter(p, top_k)
    p = top_p_filter(p, top_p)
    return int(np.random.choice(len(p), p=p))
