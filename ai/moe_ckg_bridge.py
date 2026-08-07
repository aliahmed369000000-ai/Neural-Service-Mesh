"""
MoE ↔ CKG Bridge — ai/moe_ckg_bridge.py
=======================================
يربط Hierarchical Dynamic MoE بمسار الاستدلال الحي:

  سؤال → CKG (مفاهيم) → context_vector[784]
       → إسقاط إلى d_model → HierarchicalMoE
       → أوزان فئات/خبراء → تعزيز ترتيب المفاهيم

لا يكسّر التدفق الحالي: أي فشل في التحميل يُبتلع ويُعاد ranking بدون تعديل.

الاستخدام من ReasoningPipeline:
    from ai.moe_ckg_bridge import get_moe_bridge
    bridge = get_moe_bridge()
    ranked = bridge.rerank_concepts(ranked, context_vector, question)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("MoECKGBridge")

ROOT = Path(__file__).resolve().parent.parent
MOE_PATH = ROOT / "artifacts" / "hierarchical_moe" / "hierarchical_moe.pt"

# ── خريطة clusters/كلمات → فئات MoE ─────────────────────────────────────────
_CLUSTER_TO_CATEGORY: Dict[str, str] = {
    "fiqh": "fiqh",
    "فقه": "fiqh",
    "احكام": "fiqh",
    "أحكام": "fiqh",
    "madhhab": "fiqh",
    "tafsir": "tafsir",
    "تفسير": "tafsir",
    "quran": "tafsir",
    "قرآن": "tafsir",
    "قران": "tafsir",
    "ayah": "tafsir",
    "hadith": "hadith",
    "حديث": "hadith",
    "sunnah": "hadith",
    "سنة": "hadith",
    "aqidah": "aqidah",
    "عقيدة": "aqidah",
    "tawhid": "aqidah",
    "توحيد": "aqidah",
    "seerah": "seerah",
    "سيرة": "seerah",
    "تاريخ": "seerah",
    "history": "seerah",
    "arabic": "arabic",
    "لغة": "arabic",
    "نحو": "arabic",
    "بلاغة": "arabic",
    "nahw": "arabic",
    "general": "general",
    "عام": "general",
}

_KEYWORD_TO_CATEGORY: List[Tuple[str, str]] = [
    (r"فق[هھ]|حنفي|مالكي|شافعي|حنبلي|صلاة|زكاة|صوم|حج|طهارة", "fiqh"),
    (r"تفسير|آية|آيات|سورة|قرآن|قران", "tafsir"),
    (r"حديث|أحاديث|رواة|إسناد|صحيح|ضعيف|بخاري|مسلم", "hadith"),
    (r"عقيدة|توحيد|إيمان|أسماء\s*الله|قدر", "aqidah"),
    (r"سيرة|غزوة|صحابة|خلافة|نبوي", "seerah"),
    (r"نحو|صرف|بلاغة|إعراب|لغة\s*عربي", "arabic"),
]


def map_cluster_to_category(cluster: str) -> str:
    """يحوّل اسم cluster من CKG إلى مفتاح فئة MoE."""
    if not cluster:
        return "general"
    c = cluster.strip().lower()
    if c in _CLUSTER_TO_CATEGORY:
        return _CLUSTER_TO_CATEGORY[c]
    for key, cat in _CLUSTER_TO_CATEGORY.items():
        if key in c:
            return cat
    return "general"


def infer_categories_from_text(text: str) -> List[str]:
    """استخراج فئات محتملة من نص السؤال (للتعزيز الرمزي)."""
    found: List[str] = []
    for pat, cat in _KEYWORD_TO_CATEGORY:
        if re.search(pat, text or "", re.I):
            if cat not in found:
                found.append(cat)
    return found or ["general"]


class MoECKGBridge:
    """
    جسر خفيف: يحمل HierarchicalMoE مرة واحدة، يسقط المتجه 784→d_model،
    ويُرجع معاملات تعزيز لكل فئة لاستخدامها في إعادة ترتيب المفاهيم.
    """

    def __init__(
        self,
        d_model: Optional[int] = None,
        moe_path: Optional[Path] = None,
        blend: float = 0.25,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.blend = float(max(0.0, min(1.0, blend)))
        self.moe = None
        self.projector = None  # np: (d_model, 784) أو torch Linear
        self._torch = None
        self._device = "cpu"
        self._load_error: Optional[str] = None

        if not enabled:
            return

        try:
            import torch
            from ai.hierarchical_moe import HierarchicalMoE, build_default_moe

            self._torch = torch
            path = Path(moe_path) if moe_path else MOE_PATH
            if path.is_file():
                self.moe = HierarchicalMoE.load(path, map_location="cpu")
                logger.info("MoE محمّل من %s (%d خبراء)", path, self.moe.total_experts())
            else:
                self.moe = build_default_moe(d_model=d_model or 128)
                logger.info("MoE افتراضي جديد (%d خبراء)", self.moe.total_experts())

            self.moe.eval()
            dm = self.moe.d_model
            proj_path = MOE_PATH.parent / "moe_projector.npy"
            if proj_path.is_file():
                P = np.load(str(proj_path))
                if getattr(P, "shape", None) == (dm, 784):
                    self.projector = P.astype(np.float64)
                    logger.info("مسقط MoE محمّل من %s", proj_path)
                else:
                    rng = np.random.RandomState(42)
                    self.projector = rng.randn(dm, 784).astype(np.float64) * (2.0 / np.sqrt(784))
            else:
                rng = np.random.RandomState(42)
                self.projector = rng.randn(dm, 784).astype(np.float64) * (2.0 / np.sqrt(784))
        except Exception as e:
            self._load_error = str(e)
            self.moe = None
            logger.warning("MoECKGBridge تعطّل: %s", e)

    @property
    def available(self) -> bool:
        return self.enabled and self.moe is not None and self.projector is not None

    def project(self, context_vector: np.ndarray) -> "torch.Tensor":
        """(784,) أو (B,784) → (B, d_model) torch."""
        torch = self._torch
        v = np.asarray(context_vector, dtype=np.float64)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        if v.shape[-1] != 784:
            # pad / truncate
            fixed = np.zeros((v.shape[0], 784), dtype=np.float64)
            n = min(784, v.shape[-1])
            fixed[:, :n] = v[:, :n]
            v = fixed
        # L2 normalize ثم إسقاط
        norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        v = v / norms
        projected = v @ self.projector.T  # (B, d_model)
        return torch.from_numpy(projected.astype(np.float32))

    def category_weights(
        self, context_vector: np.ndarray, question: str = ""
    ) -> Dict[str, float]:
        """
        يُرجع dict: category → وزن نسبي من راوتر المستوى 1 (+ تلميح نصي).
        """
        if not self.available:
            return {c: 1.0 for c in infer_categories_from_text(question)}

        torch = self._torch
        with torch.no_grad():
            x = self.project(context_vector)
            # راوتر المجموعات فقط
            g_idx, g_w, g_probs, _ = self.moe.group_router(x)
            order = self.moe._group_order
            probs = g_probs[0].cpu().numpy()  # (n_groups,)
            weights = {order[i]: float(probs[i]) for i in range(len(order))}

        # تعزيز رمزي خفيف من كلمات السؤال
        for cat in infer_categories_from_text(question):
            weights[cat] = weights.get(cat, 0.05) + 0.15

        # تطبيع
        s = sum(weights.values()) or 1.0
        return {k: v / s for k, v in weights.items()}

    def route_info(
        self, context_vector: np.ndarray, question: str = ""
    ) -> Dict[str, Any]:
        """تفاصيل المسار الهرمي للشفافية."""
        if not self.available:
            return {"available": False, "error": self._load_error}

        torch = self._torch
        with torch.no_grad():
            x = self.project(context_vector)
            out, aux, info = self.moe(x, return_info=True)
        return {
            "available": True,
            "groups_selected": info.get("groups_selected"),
            "group_weights": info.get("group_weights"),
            "aux_loss": info.get("aux_loss"),
            "text_hint_categories": infer_categories_from_text(question),
            "category_weights": self.category_weights(context_vector, question),
        }

    def rerank_concepts(
        self,
        ranked: List[Dict[str, Any]],
        context_vector: np.ndarray,
        question: str = "",
        score_key: str = "score",
    ) -> List[Dict[str, Any]]:
        """
        يعيد ترتيب قائمة المفاهيم (قواميس من _decide) بمزج وزن MoE.

        score_new = (1 - blend) * score_old + blend * score_old * (1 + cat_w)
        """
        if not self.available or not ranked or self.blend <= 0:
            return ranked

        cat_w = self.category_weights(context_vector, question)
        out: List[Dict[str, Any]] = []
        for item in ranked:
            row = dict(item)
            cluster = str(row.get("cluster") or row.get("domain") or "")
            cat = map_cluster_to_category(cluster)
            # إن لم يوجد cluster، حاول من الاسم
            if cat == "general" and row.get("name"):
                hinted = infer_categories_from_text(str(row["name"]))
                cat = hinted[0] if hinted else "general"
            w = cat_w.get(cat, cat_w.get("general", 0.1))
            old = float(row.get(score_key) or 0.0)
            new = (1.0 - self.blend) * old + self.blend * old * (1.0 + w)
            row[score_key] = new
            row["moe_category"] = cat
            row["moe_weight"] = round(w, 4)
            out.append(row)

        out.sort(key=lambda r: float(r.get(score_key) or 0.0), reverse=True)
        return out

    def train_on_context(
        self,
        context_vector: np.ndarray,
        preferred_categories: Sequence[str],
        steps: int = 3,
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        """
        تدريب خفيف موجّه: يدفع راوتر المجموعات نحو الفئات المفضلة
        (مثلاً من clusters المطابقة في CKG) مع aux load-balance.
        """
        if not self.available or not preferred_categories:
            return {"loss": 0.0, "skipped": 1.0}

        torch = self._torch
        self.moe.train()
        opt = torch.optim.Adam(
            list(self.moe.group_router.parameters()),
            lr=lr,
        )
        order = self.moe._group_order
        target_idx = []
        for c in preferred_categories:
            if c in order:
                target_idx.append(order.index(c))
        if not target_idx:
            self.moe.eval()
            return {"loss": 0.0, "skipped": 1.0}

        target = torch.zeros(len(order))
        for i in target_idx:
            target[i] = 1.0 / len(target_idx)

        last = 0.0
        x = self.project(context_vector)
        for _ in range(max(1, steps)):
            opt.zero_grad()
            _, _, probs, aux = self.moe.group_router(x)
            # CE تقريبي: -Σ t log p
            main = -torch.sum(target.to(probs.device) * torch.log(probs[0] + 1e-8))
            loss = main + aux
            loss.backward()
            opt.step()
            last = float(loss.item())

        self.moe.eval()
        return {"loss": last, "steps": float(steps)}

    def save(self, path: Optional[Path] = None) -> Optional[Path]:
        if not self.available:
            return None
        p = Path(path) if path else MOE_PATH
        return self.moe.save(p)


# ── Singleton ────────────────────────────────────────────────────────────────
_bridge: Optional[MoECKGBridge] = None


def get_moe_bridge(
    force_reload: bool = False,
    blend: float = 0.25,
    enabled: bool = True,
) -> MoECKGBridge:
    global _bridge
    if _bridge is None or force_reload:
        _bridge = MoECKGBridge(blend=blend, enabled=enabled)
    return _bridge


def moe_boost_pipeline_ranked(
    ranked: List[Any],
    context_vector: np.ndarray,
    question: str = "",
    blend: float = 0.25,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    واجهة مريحة لـ ReasoningPipeline:

    تقبل قائمة كائنات (MatchedConcept-like أو dict) وتعيد نفس النوع مرتّباً.
    """
    bridge = get_moe_bridge(blend=blend)
    if not bridge.available or not ranked:
        return ranked, {"moe_applied": False, "error": bridge._load_error}

    # حوّل إلى dicts
    dicts: List[Dict[str, Any]] = []
    is_dict = isinstance(ranked[0], dict)
    for r in ranked:
        if isinstance(r, dict):
            dicts.append(dict(r))
        else:
            dicts.append(
                {
                    "name": getattr(r, "name", ""),
                    "cluster": getattr(r, "cluster", ""),
                    "strength": getattr(r, "strength", 0.0),
                    "score": getattr(r, "score", 0.0),
                    "frequency": getattr(r, "frequency", 0),
                    "_orig": r,
                }
            )

    reranked = bridge.rerank_concepts(dicts, context_vector, question)
    info = {
        "moe_applied": True,
        "blend": blend,
        "category_weights": bridge.category_weights(context_vector, question),
    }

    if is_dict:
        return reranked, info

    # أعد تطبيق score على الكائنات الأصلية مع الحفاظ على الترتيب الجديد
    out_objs = []
    for d in reranked:
        orig = d.get("_orig")
        if orig is not None:
            try:
                orig.score = float(d["score"])
            except Exception:
                pass
            # حقول شفافية اختيارية
            try:
                setattr(orig, "moe_category", d.get("moe_category"))
                setattr(orig, "moe_weight", d.get("moe_weight"))
            except Exception:
                pass
            out_objs.append(orig)
        else:
            out_objs.append(d)
    return out_objs, info
