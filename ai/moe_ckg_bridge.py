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
    "fiqh": "fiqh", "فقه": "fiqh", "احكام": "fiqh", "أحكام": "fiqh", "madhhab": "fiqh",
    "tafsir": "tafsir", "تفسير": "tafsir", "quran": "tafsir", "قرآن": "tafsir", "قران": "tafsir",
    "hadith": "hadith", "حديث": "hadith", "sunnah": "hadith", "سنة": "hadith",
    "aqidah": "aqidah", "عقيدة": "aqidah", "tawhid": "aqidah", "توحيد": "aqidah",
    "seerah": "seerah", "سيرة": "seerah",
    "usul": "usul", "أصول": "usul", "مقاصد": "usul",
    "tajweed": "tajweed", "تجويد": "tajweed", "قراءات": "tajweed", "حفظ": "tajweed",
    "islamic_finance": "islamic_finance", "مرابحة": "islamic_finance", "صكوك": "islamic_finance",
    "arabic": "arabic", "لغة": "arabic", "نحو": "arabic", "بلاغة": "arabic",
    "literature": "literature", "أدب": "literature", "شعر": "literature",
    "programming": "programming", "code": "programming", "برمجة": "programming",
    "technology": "technology", "تقنية": "technology", "ai": "technology",
    "science": "science", "علوم": "science", "physics": "science",
    "math": "math", "رياضيات": "math",
    "astronomy": "astronomy", "فلك": "astronomy", "نجوم": "astronomy",
    "geography": "geography", "جغرافيا": "geography", "خريطة": "geography",
    "engineering": "engineering", "هندسة": "engineering",
    "medicine": "medicine", "طب": "medicine", "صحة": "medicine",
    "psychology": "psychology", "نفس": "psychology",
    "sports": "sports", "رياضة": "sports", "football": "sports",
    "family": "family", "أسرة": "family", "زواج": "family", "تربية": "family",
    "cooking": "cooking", "طبخ": "cooking", "وصفة": "cooking",
    "travel": "travel", "سفر": "travel", "سياحة": "travel", "عمرة": "travel",
    "environment": "environment", "بيئة": "environment", "مناخ": "environment",
    "agriculture": "agriculture", "زراعة": "agriculture", "مزرعة": "agriculture",
    "business": "business", "أعمال": "business", "تسويق": "business",
    "economy": "economy", "اقتصاد": "economy", "تضخم": "economy",
    "law": "law", "قانون": "law",
    "career": "career", "وظيفة": "career", "سيرة_ذاتية": "career",
    "education": "education", "تعليم": "education",
    "history": "history", "تاريخ": "history",
    "media": "media", "إعلام": "media",
    "philosophy": "philosophy", "فلسفة": "philosophy", "منطق": "philosophy",
    "art": "art", "فن": "art", "تصميم": "art", "خط": "art",
    "general": "general", "عام": "general",
}

_KEYWORD_TO_CATEGORY: List[Tuple[str, str]] = [
    # أولوية عالية: تمييز الالتباس الشائع أولاً
    (r"سيرة\s*ذاتية|مقابلة\s*عمل|وظيفة|مهارة\s*وظيف|cv\b|resume", "career"),
    (r"قانون\s*نيوتن|فيزياء|كيمياء|أحياء|بيولوجيا|علوم\s*طبيعي", "science"),
    (r"وصفة|مطبخ|أكلة|مكونات\s*الطبخ|طبخ", "cooking"),
    (r"سيرة\s*نبوية|غزوة|صحابة|خلافة\s*الراشد|الهجرة\s*النبوية|السيرة\s*النبوية", "seerah"),
    # دين
    (r"فق[هھ]|حنفي|مالكي|شافعي|حنبلي|صلاة|زكاة|صوم|حج|طهارة|وضوء|نكاح|طلاق|فتوى", "fiqh"),
    (r"تفسير|آية|آيات|سورة|قرآن|قران", "tafsir"),
    (r"حديث|أحاديث|رواة|إسناد|بخاري|ترمذي|تخريج", "hadith"),
    (r"عقيدة|توحيد|إيمان|أسماء\s*الله|قدر|شرك|أركان\s*الإيمان", "aqidah"),
    (r"أصول\s*الفقه|قواعد\s*فقهي|مقاصد\s*الشريعة", "usul"),
    (r"تجويد|قراءات|حفظ\s*القرآن|مخارج\s*الحروف", "tajweed"),
    (r"تمويل\s*إسلامي|مرابحة|صكوك|تأمين\s*تكافلي|\bربا\b", "islamic_finance"),
    (r"نحو|صرف|بلاغة|إعراب|لغة\s*عربي", "arabic"),
    (r"شعر|قصيدة|رواية|أدب|نقد\s*أدبي", "literature"),
    # تقنية
    (r"برمج|برمجة|python|جافا|javascript|كود|خوارزمي|\bgit\b|\bapi\b|android|ios", "programming"),
    (r"ذكاء\s*اصطناعي|شبكات|حاسوب|سيرفر|أمن\s*سيبراني|سحابة|devops|docker", "technology"),
    (r"رياضيا|رياضيات|جبر|هندسة\s*مستوية|إحصاء|تفاضل|تكامل", "math"),
    (r"فلك|نجوم|كوكب|مجرة|تلسكوب", "astronomy"),
    (r"جغرافيا|قارة|خريطة|تضاريس", "geography"),
    (r"هندسة\s*مدنية|كهربائية|ميكانيك|مهندس", "engineering"),
    (r"طبي[ب]?|مرض|علاج|دواء|إسعاف|مستشفى|أعراض|عيادة", "medicine"),
    (r"علم\s*نفس|قلق|اكتئاب|سلوك|شخصية", "psychology"),
    (r"رياضة|كرة\s*قدم|مباراة|لياقة|أولمبي|جيم", "sports"),
    (r"أسرة|زواج|تربية\s*أطفال|مراهق", "family"),
    (r"سفر|سياحة|تأشيرة|عمرة|فندق|رحلة", "travel"),
    (r"بيئة|تلوث|مناخ|استدامة|تدوير", "environment"),
    (r"زراعة|محصول|ماشية|مزرعة", "agriculture"),
    (r"تسويق|مبيعات|إدارة\s*أعمال|شركة|ريادة", "business"),
    (r"اقتصاد|تضخم|ناتج\s*محلي|سوق\s*مالية|عرض\s*وطلب", "economy"),
    (r"محكمة|دعوى|تشريع|محام|عقد\s*قانوني|القانون\s*المدني", "law"),
    (r"تعليم|مناهج|تدريس|اختبار|منهج|امتحان", "education"),
    (r"تاريخ|حضارة|عثمان|حرب\s*عالمية|مؤرخ", "history"),
    (r"إعلام|صحافة|محتوى|سوشيال|يوتيوب|فيديو", "media"),
    (r"فلسفة|أخلاق|منطق\s*صوري", "philosophy"),
    (r"تصميم|خط\s*عربي|فن\s*تشكيلي|رسم|جرافيك", "art"),
    # سيرة عامة متأخرة (بعد career)
    (r"السيرة(?!\s*ذاتية)|سيرة\s*ال(?!ذات)|\bنبوي\b", "seerah"),
    (r"صحة(?!\s*الطعام)|تغذية|فيتامين", "medicine"),
    (r"قانون(?!\s*نيوتن)", "law"),
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
    scores = keyword_category_scores(text)
    if not scores:
        return ["general"]
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [c for c, _ in ranked if _ > 0]


def keyword_category_scores(text: str) -> Dict[str, float]:
    """درجات كلمات مفتاحية مرتبة بالأولوية (أول تطابق أقوى)."""
    scores: Dict[str, float] = {}
    if not text:
        return scores
    for i, (pat, cat) in enumerate(_KEYWORD_TO_CATEGORY):
        if re.search(pat, text, re.I):
            # أول الأنماط في القائمة أعلى وزناً
            w = 1.0 - min(0.5, i * 0.008)
            scores[cat] = max(scores.get(cat, 0.0), w)
    return scores


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
                self.moe.apply_best_config()
                logger.info("MoE محمّل من %s (%d خبراء) [best config]", path, self.moe.total_experts())
            else:
                self.moe = build_default_moe(d_model=d_model or 128)
                logger.info("MoE افتراضي جديد (%d خبراء) [best config]", self.moe.total_experts())

            self.moe.eval()
            dm = self.moe.d_model
            self._torch_projector = None
            learnable = MOE_PATH.parent / "moe_learnable_projector.pt"
            proj_path = MOE_PATH.parent / "moe_projector.npy"
            if learnable.is_file():
                try:
                    import torch.nn as nn
                    blob = torch.load(str(learnable), map_location="cpu", weights_only=False)
                    # LearnableProjector: Linear(784,2d)->GELU->Linear(2d,d)->LN
                    class _Proj(nn.Module):
                        def __init__(self, d_out):
                            super().__init__()
                            self.net = nn.Sequential(
                                nn.Linear(784, d_out * 2),
                                nn.GELU(),
                                nn.Linear(d_out * 2, d_out),
                                nn.LayerNorm(d_out),
                            )
                        def forward(self, x):
                            return self.net(x)
                    proj = _Proj(dm)
                    proj.load_state_dict(blob["projector"])
                    proj.eval()
                    self._torch_projector = proj
                    logger.info("مسقط MoE القابل للتعلم محمّل من %s", learnable)
                except Exception as e:
                    logger.warning("فشل تحميل المسقط القابل للتعلم: %s", e)
            if self._torch_projector is None and proj_path.is_file():
                P = np.load(str(proj_path))
                if getattr(P, "shape", None) == (dm, 784):
                    self.projector = P.astype(np.float64)
                    logger.info("مسقط MoE محمّل من %s", proj_path)
                else:
                    rng = np.random.RandomState(42)
                    self.projector = rng.randn(dm, 784).astype(np.float64) * (2.0 / np.sqrt(784))
            elif self._torch_projector is None:
                rng = np.random.RandomState(42)
                self.projector = rng.randn(dm, 784).astype(np.float64) * (2.0 / np.sqrt(784))
        except Exception as e:
            self._load_error = str(e)
            self.moe = None
            logger.warning("MoECKGBridge تعطّل: %s", e)

    @property
    def available(self) -> bool:
        has_proj = self.projector is not None or getattr(self, "_torch_projector", None) is not None
        return self.enabled and self.moe is not None and has_proj

    def project(self, context_vector: np.ndarray) -> "torch.Tensor":
        """(784,) أو (B,784) → (B, d_model) torch."""
        torch = self._torch
        v = np.asarray(context_vector, dtype=np.float64)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        if v.shape[-1] != 784:
            fixed = np.zeros((v.shape[0], 784), dtype=np.float64)
            n = min(784, v.shape[-1])
            fixed[:, :n] = v[:, :n]
            v = fixed
        norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        v = v / norms
        if getattr(self, "_torch_projector", None) is not None:
            with torch.no_grad():
                return self._torch_projector(torch.from_numpy(v.astype(np.float32)))
        projected = v @ self.projector.T
        return torch.from_numpy(projected.astype(np.float32))

    def category_weights(
        self, context_vector: np.ndarray, question: str = ""
    ) -> Dict[str, float]:
        """
        تصنيف هجين: راوتر عصبي + درجات كلمات مفتاحية (أوّل مطابقة أقوى).
        عند وجود إشارة نصية واضحة يُرفع وزن المزج الرمزي تلقائياً.
        """
        kw = keyword_category_scores(question or "")
        if not self.available:
            if not kw:
                return {"general": 1.0}
            s = sum(kw.values()) or 1.0
            return {k: v / s for k, v in kw.items()}

        torch = self._torch
        with torch.no_grad():
            x = self.project(context_vector)
            g_idx, g_w, g_probs, _ = self.moe.group_router(x)
            order = self.moe._group_order
            probs = g_probs[0].cpu().numpy()
            weights = {order[i]: float(probs[i]) for i in range(len(order))}

        # مزج تكيّفي: إشارة كلمات قوية → اعتماد أكبر على القواعد
        if kw:
            max_kw = max(kw.values())
            # إشارة نصية واضحة → اعتماد أقوى على القواعد (يحل التباس سيرة/وظيفة، طبخ/طب، …)
            alpha = 0.50 + 0.48 * max_kw  # 0.50..0.98
            top_cat = max(kw.items(), key=lambda x: x[1])[0]
            for cat, sc in kw.items():
                weights[cat] = weights.get(cat, 0.0) * (1.0 - alpha) + sc * alpha
            # تعزيز إضافي للفئة الأولى نصياً
            weights[top_cat] = weights.get(top_cat, 0.0) + 0.25 * alpha
            for cat in list(weights.keys()):
                if cat not in kw:
                    weights[cat] *= (1.0 - 0.65 * alpha)

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


    def classify(
        self,
        question: str,
        context_vector: Optional[np.ndarray] = None,
        top_n: int = 3,
    ) -> Dict[str, Any]:
        """
        تصنيف عالي المستوى للسؤال: فئة + ثقة + بدائل.
        إن لم يُمرَّر context_vector يُبنى من النص عبر VectorEncoder إن أمكن.
        """
        q = (question or "").strip()
        if context_vector is None:
            try:
                from ai.knowledge_trainer import VectorEncoder
                context_vector = VectorEncoder.encode(q, domain="general", importance=0.65, certainty=0.8)
            except Exception as e:
                kw = keyword_category_scores(q)
                if not kw:
                    return {"top": "general", "confidence": 0.2, "alternatives": [], "source": "fallback", "error": str(e)}
                ranked = sorted(kw.items(), key=lambda x: -x[1])
                s = sum(v for _, v in ranked) or 1.0
                return {
                    "top": ranked[0][0],
                    "confidence": float(ranked[0][1] / s),
                    "alternatives": [{"category": c, "weight": float(v / s)} for c, v in ranked[1:top_n]],
                    "source": "keywords_only",
                }

        weights = self.category_weights(context_vector, q)
        ranked = sorted(weights.items(), key=lambda x: -x[1])
        top_cat, top_w = ranked[0]
        # ثقة معدّلة بالفجوة عن الثاني
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        gap = top_w - second
        confidence = float(min(1.0, top_w + 0.5 * gap))

        experts: List[str] = []
        if self.available and top_cat in getattr(self.moe, "groups", {}):
            try:
                experts = list(self.moe.groups[top_cat]._id_order)[:5]
            except Exception:
                experts = []

        return {
            "top": top_cat,
            "confidence": round(confidence, 4),
            "weight": round(float(top_w), 4),
            "alternatives": [
                {"category": c, "weight": round(float(w), 4)} for c, w in ranked[1:top_n]
            ],
            "experts": experts,
            "source": "hybrid_moe",
            "available": self.available,
        }

    def health_report(self) -> str:
        """تقرير صحة نظام MoE للوحة التحكم."""
        lines = ["## 🩺 صحة Hierarchical MoE", ""]
        if not self.available:
            lines.append(f"- **الحالة:** غير متاح — `{self._load_error}`")
            return "\n".join(lines)
        m = self.moe
        lines.append("- **الحالة:** ✅ جاهز")
        lines.append(f"- **فئات:** {len(m._group_order)} · **خبراء:** {m.total_experts()}")
        lines.append(
            f"- **Best config:** temp={getattr(m,'router_temperature', '?')} · "
            f"shared={getattr(m,'shared_coeff','?')} · "
            f"threshold={getattr(m,'weight_threshold','?')} · "
            f"residual={getattr(m,'input_residual','?')}"
        )
        lines.append(f"- **مسقط قابل للتعلم:** {'نعم' if getattr(self,'_torch_projector',None) is not None else 'npy/ثابت'}")
        lines.append(f"- **blend مع CKG:** {self.blend}")
        # عيّنة تصنيف
        sample = self.classify("ما حكم الصلاة في المذهب الشافعي؟")
        lines.append(
            f"- **عينة تصنيف:** `{sample.get('top')}` "
            f"(ثقة {sample.get('confidence')})"
        )
        return "\n".join(lines)

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
