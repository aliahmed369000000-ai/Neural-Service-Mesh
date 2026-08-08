"""
Hierarchical Dynamic Mixture-of-Experts (MoE) — ai/hierarchical_moe.py
======================================================================
نظام MoE هرمي ديناميكي بـ PyTorch لمشروع Neural Service Mesh.

المتطلبات المغطّاة:
  • عدد خبراء غير محدود عملياً — يبدأ بعدد قليل ويتوسع لاحقاً
  • Top-K Routing: تفعيل 3–5 خبراء فقط لكل مدخل (قابل للضبط)
  • Dynamic Experts: إضافة خبير جديد دون إعادة بناء/تدريب النظام كاملاً
  • Hierarchical (مستويان):
      المستوى 1 → راوتر رئيسي يختار مجموعة/فئة (فقه، تفسير، حديث…)
      المستوى 2 → راوتر فرعي داخل المجموعة يختار الخبير المحدد
  • Load Balancing: خسارة مساعدة تمنع الاعتماد الدائم على نفس الخبراء

تقنيات MoE المتقدمة المدمجة:
  • Shared Expert (DeepSeek-style): مسار مشترك دائماً + خبراء موجَّهة
  • Router Temperature: حدة/ليونة قرار التوجيه
  • Router Z-Loss (ST-MoE): استقرار logits البوابة
  • Capacity Factor: حد سعة لكل خبير مع إسقاط الفائض
  • Noisy Top-K: استكشاف أثناء التدريب (Switch/GShard)
  • Expert Dropout: تنظيم لمساهمة الخبراء الموجَّهة
  • MLP Gate + LayerNorm: راوتر أعمق من Linear واحد

الاستخدام السريع:
    from ai.hierarchical_moe import HierarchicalMoE, DEFAULT_CATEGORIES

    moe = HierarchicalMoE(d_model=128, top_k_groups=2, top_k_experts=3)
    # أوزان عشوائية أولية + خبراء افتراضيون لكل فئة إسلامية

    x = torch.randn(4, 128)          # (batch, d_model)
    out, aux = moe(x)                # out: (B, d_model), aux فيه load-balance loss

    # إضافة خبير ديناميكي لاحقاً بدون إعادة تدريب الباقي
    moe.add_expert("fiqh", name="fiqh_dhahiri", description="فقه ظاهري")

    moe.save("artifacts/hierarchical_moe/moe.pt")
    moe2 = HierarchicalMoE.load("artifacts/hierarchical_moe/moe.pt")
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("HierarchicalMoE")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAVE_DIR = ROOT / "artifacts" / "hierarchical_moe"
DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── فئات المستوى 1 + خبراء المستوى 2 (دين موسّع + تخصصات واسعة) ────────────
DEFAULT_CATEGORIES: Dict[str, List[str]] = {
    # ── الدين الإسلامي ──
    "fiqh": [
        "fiqh_hanafi", "fiqh_maliki", "fiqh_shafii", "fiqh_hanbali",
        "fiqh_general", "fiqh_contemporary",
    ],
    "tafsir": [
        "tafsir_general", "tafsir_linguistic", "tafsir_thematic", "tafsir_scientific",
    ],
    "hadith": [
        "hadith_riwaya", "hadith_diraya", "hadith_ahkam", "hadith_takhrij",
    ],
    "aqidah": [
        "aqidah_general", "aqidah_comparative", "aqidah_tawhid", "aqidah_sects",
    ],
    "seerah": [
        "seerah_prophet", "seerah_khulafa", "seerah_events", "seerah_companions",
    ],
    "usul": [
        "usul_fiqh", "usul_qawaid", "usul_maqasid",
    ],
    "tajweed": [
        "tajweed_rules", "tajweed_qiraat", "tajweed_hifz",
    ],
    "islamic_finance": [
        "isf_banking", "isf_contracts", "isf_zakat_calc",
    ],
    # ── لغات وآداب ──
    "arabic": [
        "arabic_nahw", "arabic_balagha", "arabic_sarf", "arabic_translation",
    ],
    "literature": [
        "lit_poetry", "lit_prose", "lit_criticism", "lit_arabic_classic",
    ],
    # ── تقنية وبرمجة ──
    "programming": [
        "prog_python", "prog_web", "prog_algorithms", "prog_systems",
        "prog_mobile", "prog_data",
    ],
    "technology": [
        "tech_ai", "tech_networks", "tech_hardware", "tech_security",
        "tech_cloud", "tech_devops",
    ],
    # ── علوم ورياضيات ──
    "science": [
        "science_physics", "science_chemistry", "science_biology", "science_general",
    ],
    "math": [
        "math_algebra", "math_geometry", "math_stats", "math_general", "math_calculus",
    ],
    "astronomy": [
        "astro_solar", "astro_observation", "astro_general",
    ],
    "geography": [
        "geo_physical", "geo_human", "geo_maps",
    ],
    "engineering": [
        "eng_civil", "eng_electrical", "eng_mechanical", "eng_general",
    ],
    # ── صحة ونفس ──
    "medicine": [
        "med_general", "med_nutrition", "med_public_health", "med_first_aid",
    ],
    "psychology": [
        "psych_general", "psych_development", "psych_clinical",
    ],
    # ── مجتمع وحياة ──
    "sports": [
        "sports_football", "sports_fitness", "sports_general", "sports_olympic",
    ],
    "family": [
        "family_parenting", "family_marriage", "family_youth",
    ],
    "cooking": [
        "cook_recipes", "cook_healthy", "cook_arabic",
    ],
    "travel": [
        "travel_planning", "travel_hajj_umrah", "travel_culture",
    ],
    "environment": [
        "env_climate", "env_conservation", "env_sustainability",
    ],
    "agriculture": [
        "agri_crops", "agri_livestock", "agri_modern",
    ],
    # ── أعمال واقتصاد وقانون ──
    "business": [
        "biz_marketing", "biz_finance", "biz_management", "biz_entrepreneurship",
    ],
    "economy": [
        "econ_macro", "econ_micro", "econ_markets",
    ],
    "law": [
        "law_general", "law_civil", "law_commercial",
    ],
    "career": [
        "career_jobs", "career_cv", "career_skills",
    ],
    # ── تعليم وإعلام وفكر ──
    "education": [
        "edu_teaching", "edu_curriculum", "edu_learning", "edu_exams",
    ],
    "history": [
        "hist_islamic", "hist_world", "hist_general",
    ],
    "media": [
        "media_content", "media_social", "media_journalism", "media_video",
    ],
    "philosophy": [
        "phil_general", "phil_ethics", "phil_logic",
    ],
    "art": [
        "art_design", "art_calligraphy", "art_visual",
    ],
    "general": [
        "general_assistant", "general_research", "general_daily",
    ],
}



# ── أفضل إعدادات MoE (ممارسات Switch / ST-MoE / DeepSeek / Mixtral) ────────
BEST_MOE_CONFIG: Dict[str, Any] = {
    "top_k_groups": 3,
    "top_k_experts": 3,
    "use_shared_expert": True,
    "shared_coeff": 0.35,
    "router_temperature": 0.85,   # أحدّ قليلاً من 1.0 → ثقة أعلى
    "z_loss_coeff": 1e-3,
    "capacity_factor": 1.5,        # سعة أوفر قليلاً لتقليل الإسقاط الضار
    "expert_dropout": 0.08,       # تنظيم أثناء التدريب
    "lb_coeff": 0.02,             # موازنة حمل أقوى
    "weight_threshold": 0.15,     # إهمال أوزان Top-K الضعيفة (Mixtral-style)
    "input_residual": True,       # مسار متبقٍ من الدخل (Transformer FFN style)
}


# ══════════════════════════════════════════════════════════════════════════════
# 1) بيانات تعريف الخبير
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ExpertMeta:
    """بيانات تعريفية لخبير — تُحفظ مع الأوزان."""
    expert_id: str
    category: str
    name: str
    description: str = ""
    created_at: float = field(default_factory=time.time)
    usage_count: int = 0
    total_gate_weight: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExpertMeta":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ══════════════════════════════════════════════════════════════════════════════
# 2) شبكة الخبير الواحد (FFN مستقلة)
# ══════════════════════════════════════════════════════════════════════════════
class ExpertFFN(nn.Module):
    """خبير عصبي بسيط: Linear → GELU → Dropout → Linear (residual-friendly)."""

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        d_ff = d_ff or (d_model * 4)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# 3) راوتر Top-K مع ضوضاء اختيارية (لـ load balancing أثناء التدريب)
# ══════════════════════════════════════════════════════════════════════════════
class TopKRouter(nn.Module):
    """
    راوتر MLP متقدم: d_model → hidden → n_targets + Top-K.

    تقنيات:
      - Temperature scaling على logits
      - Noisy Top-K (Switch/GShard) أثناء التدريب
      - Capacity factor لإسقاط الفائض عن سعة الخبير
      - Load-balancing aux: n * Σ (f_i · P_i)
      - Router z-loss (ST-MoE): mean(logsumexp(logits)^2)
    """

    def __init__(
        self,
        d_model: int,
        n_targets: int,
        top_k: int = 2,
        noise_std: float = 1.0,
        jitter: bool = True,
        hidden_mult: float = 2.0,
        dropout: float = 0.1,
        temperature: float = 0.85,
        z_loss_coeff: float = 1e-3,
        capacity_factor: float = 1.5,
        weight_threshold: float = 0.15,
    ):
        super().__init__()
        assert n_targets >= 1
        self.n_targets = n_targets
        self.top_k = max(1, min(top_k, n_targets))
        self.noise_std = noise_std
        self.jitter = jitter
        self.temperature = max(1e-3, float(temperature))
        self.z_loss_coeff = float(z_loss_coeff)
        self.capacity_factor = float(capacity_factor)
        self.weight_threshold = float(weight_threshold)
        hidden = max(64, int(d_model * hidden_mult))
        self.hidden = hidden
        self.gate = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_targets, bias=False),
        )
        self._init_gate()

    def _init_gate(self) -> None:
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        last = self.gate[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=0.02)

    def expand_targets(self, new_n: int) -> None:
        """توسيع الطبقة الأخيرة فقط عند إضافة منافذ جديدة."""
        if new_n <= self.n_targets:
            return
        last = self.gate[-1]
        assert isinstance(last, nn.Linear)
        old = last.weight.data  # (n_old, hidden)
        hid = old.shape[1]
        new_last = nn.Linear(hid, new_n, bias=False)
        nn.init.normal_(new_last.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            new_last.weight[: self.n_targets].copy_(old)
        self.gate[-1] = new_last
        self.n_targets = new_n
        self.top_k = min(self.top_k, new_n)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, d_model) — تمثيل مجمّع (mean pool أو CLS)
        Returns:
            top_idx:   (B, top_k) فهارس المختارين
            top_w:     (B, top_k) أوزان بعد softmax على الـ top-k فقط (مجموع=1)
            probs:     (B, n_targets) احتمالات كاملة (قبل top-k)
            aux_loss:  خسارة موازنة الحمل (scalar)
        """
        logits = self.gate(x)  # (B, n)
        # Temperature scaling (ST-MoE / adaptive routing sharpness)
        logits_t = logits / self.temperature

        if self.training and self.jitter and self.noise_std > 0:
            # Noisy Top-K (Switch / GShard) — استكشاف أثناء التدريب
            noise = torch.randn_like(logits_t) * self.noise_std
            scores = logits_t + noise * F.softplus(logits_t)
        else:
            scores = logits_t

        probs = F.softmax(logits_t.float(), dim=-1)

        k = min(self.top_k, self.n_targets)
        top_val, top_idx = torch.topk(scores, k=k, dim=-1)
        top_w = F.softmax(top_val.float(), dim=-1).to(dtype=x.dtype)

        # Threshold routing (Mixtral-style): أزل المساهمات الضعيفة جداً
        if self.weight_threshold > 0:
            tmax = top_w.max(dim=-1, keepdim=True).values.clamp_min(1e-9)
            top_w = top_w.masked_fill(top_w < (tmax * self.weight_threshold), 0.0)
            top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        B = x.shape[0]
        # ── Capacity factor (Switch Transformer): حد أقصى للتوكينات لكل خبير ──
        if self.training and self.capacity_factor > 0 and B > 1:
            capacity = max(1, int(self.capacity_factor * B * k / max(1, self.n_targets)))
            # أعد توزيع الاختيارات التي تتجاوز السعة (drop overflow)
            for ki in range(k):
                idx_k = top_idx[:, ki]
                for expert_i in idx_k.unique().tolist():
                    positions = (idx_k == expert_i).nonzero(as_tuple=False).view(-1)
                    if positions.numel() > capacity:
                        # احتفظ بأعلى أوزان البوابة فقط
                        scores_pos = top_val[positions, ki]
                        keep = torch.topk(scores_pos, k=capacity, dim=0).indices
                        drop = torch.ones(positions.numel(), dtype=torch.bool, device=x.device)
                        drop[keep] = False
                        drop_idx = positions[drop]
                        # حوّل الفائض إلى ثاني أفضل عام (أو صفر وزن)
                        top_w[drop_idx, ki] = 0.0
            # أعد تطبيع الأوزان بعد الإسقاط
            top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-9)

        # Load-balancing aux (Switch style)
        mask = torch.zeros(B, self.n_targets, device=x.device, dtype=x.dtype)
        mask.scatter_(1, top_idx, 1.0)
        f = mask.mean(dim=0)
        P = probs.mean(dim=0)
        lb_loss = self.n_targets * torch.sum(f * P)

        # Router z-loss (ST-MoE): يستقرّ لوجيتات البوابة ويمنع انفجارها
        # z = logsumexp(logits); loss = mean(z^2)
        z = torch.logsumexp(logits.float(), dim=-1)
        z_loss = torch.mean(z ** 2)
        aux_loss = lb_loss + self.z_loss_coeff * z_loss

        return top_idx, top_w, probs, aux_loss


# ══════════════════════════════════════════════════════════════════════════════
# 4) مجموعة خبراء (فئة) — المستوى 2
# ══════════════════════════════════════════════════════════════════════════════
class ExpertGroup(nn.Module):
    """فئة معرفية تحتوي راوتراً فرعياً + قاموس خبراء ديناميكي."""

    def __init__(
        self,
        category: str,
        d_model: int,
        d_ff: int,
        expert_names: Sequence[str],
        top_k: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.category = category
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.top_k = top_k

        self.experts = nn.ModuleDict()
        self.meta: Dict[str, ExpertMeta] = {}
        self._id_order: List[str] = []

        for name in expert_names:
            self._register_expert(name, description=f"{category}/{name}")

        n = max(1, len(self._id_order))
        self.router = TopKRouter(d_model, n_targets=n, top_k=min(top_k, n))

    def _register_expert(self, name: str, description: str = "") -> str:
        eid = name if name not in self.experts else f"{name}_{len(self.experts)}"
        self.experts[eid] = ExpertFFN(self.d_model, self.d_ff, self.dropout)
        self.meta[eid] = ExpertMeta(
            expert_id=eid,
            category=self.category,
            name=name,
            description=description,
        )
        self._id_order.append(eid)
        return eid

    def add_expert(self, name: str, description: str = "") -> str:
        """إضافة خبير جديد ديناميكياً — يوسّع الراوتر الفرعي دون لمس الخبراء القدامى."""
        if name in self.experts:
            logger.warning("الخبير '%s' موجود مسبقاً في '%s'", name, self.category)
            return name
        eid = self._register_expert(name, description=description)
        self.router.expand_targets(len(self._id_order))
        logger.info("أُضيف الخبير '%s' إلى فئة '%s' (إجمالي %d)", eid, self.category, len(self._id_order))
        return eid

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: (B, d_model)
        Returns:
            out: (B, d_model)
            aux: scalar load-balance loss
            info: تفاصيل الاختيار للشفافية/التدقيق
        """
        if not self._id_order:
            return x, x.new_zeros(()), {"experts": [], "weights": []}

        top_idx, top_w, probs, aux = self.router(x)
        B, K = top_idx.shape
        out = torch.zeros_like(x)

        chosen_names: List[List[str]] = [[] for _ in range(B)]
        chosen_w: List[List[float]] = [[] for _ in range(B)]

        for ki in range(K):
            idx_k = top_idx[:, ki]  # (B,)
            w_k = top_w[:, ki]  # (B,)
            # اجمع العينات حسب الخبير لتقليل الاستدعاءات
            for expert_i in idx_k.unique().tolist():
                mask = idx_k == expert_i
                if not mask.any():
                    continue
                eid = self._id_order[int(expert_i)]
                expert_out = self.experts[eid](x[mask])
                out[mask] = out[mask] + expert_out * w_k[mask].unsqueeze(-1)
                # إحصائيات
                self.meta[eid].usage_count += int(mask.sum().item())
                self.meta[eid].total_gate_weight += float(w_k[mask].sum().item())
                for b in mask.nonzero(as_tuple=False).view(-1).tolist():
                    chosen_names[b].append(eid)
                    chosen_w[b].append(float(w_k[b].item()))

        info = {
            "category": self.category,
            "experts": chosen_names,
            "weights": chosen_w,
            "router_probs_mean": probs.detach().mean(0).cpu().tolist(),
        }
        return out, aux, info

    def expert_count(self) -> int:
        return len(self._id_order)

    def list_experts(self) -> List[dict]:
        return [self.meta[e].to_dict() for e in self._id_order]


# ══════════════════════════════════════════════════════════════════════════════
# 5) النظام الهرمي الكامل — المستوى 1 + 2
# ══════════════════════════════════════════════════════════════════════════════
class HierarchicalMoE(nn.Module):
    """
    Mixture-of-Experts هرمي ديناميكي.

    forward(x):
      1) الراوتر الرئيسي يختار top_k_groups فئات
      2) داخل كل فئة مختارة: الراوتر الفرعي يختار top_k_experts
      3) المخرجات تُوزَن وتُجمع
      4) تُرجع أيضاً aux_loss لموازنة الحمل على المستويين
    """

    def __init__(
        self,
        d_model: int = 128,
        d_ff: Optional[int] = None,
        categories: Optional[Dict[str, List[str]]] = None,
        top_k_groups: int = 3,
        top_k_experts: int = 3,
        dropout: float = 0.05,
        lb_coeff: float = 0.02,
        use_shared_expert: bool = True,
        shared_coeff: float = 0.35,
        router_temperature: float = 0.85,
        z_loss_coeff: float = 1e-3,
        capacity_factor: float = 1.5,
        expert_dropout: float = 0.08,
        weight_threshold: float = 0.15,
        input_residual: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or (d_model * 4)
        self.top_k_groups = top_k_groups
        self.top_k_experts = top_k_experts
        self.lb_coeff = lb_coeff
        self.dropout_p = dropout
        self.use_shared_expert = use_shared_expert
        self.shared_coeff = float(shared_coeff)
        self.router_temperature = float(router_temperature)
        self.z_loss_coeff = float(z_loss_coeff)
        self.capacity_factor = float(capacity_factor)
        self.expert_dropout = float(expert_dropout)
        self.weight_threshold = float(weight_threshold)
        self.input_residual = bool(input_residual)

        cats = categories or {k: list(v) for k, v in DEFAULT_CATEGORIES.items()}
        self.groups = nn.ModuleDict()
        self._group_order: List[str] = []

        for cat, experts in cats.items():
            self.groups[cat] = ExpertGroup(
                category=cat,
                d_model=d_model,
                d_ff=self.d_ff,
                expert_names=experts,
                top_k=top_k_experts,
                dropout=dropout,
            )
            self._group_order.append(cat)

        n_g = max(1, len(self._group_order))
        self.group_router = TopKRouter(
            d_model,
            n_targets=n_g,
            top_k=min(top_k_groups, n_g),
            temperature=router_temperature,
            z_loss_coeff=z_loss_coeff,
            capacity_factor=capacity_factor,
            weight_threshold=weight_threshold,
        )
        # مزامنة إعدادات الراوترات الفرعية على أفضل القيم
        for g in self.groups.values():
            g.router.temperature = router_temperature
            g.router.z_loss_coeff = z_loss_coeff
            g.router.capacity_factor = capacity_factor
            g.router.weight_threshold = weight_threshold

        # Shared Expert (أسلوب DeepSeek-MoE): مسار دائماً نشط + خبراء موجَّهة
        if use_shared_expert:
            self.shared_expert = ExpertFFN(d_model, self.d_ff, dropout)
        else:
            self.shared_expert = None

        self.out_norm = nn.LayerNorm(d_model)

    # ── ديناميكية ────────────────────────────────────────────────────────────
    def add_category(self, category: str, initial_experts: Optional[List[str]] = None) -> None:
        """إضافة فئة جديدة (مستوى 1) مع خبراء ابتدائيين."""
        if category in self.groups:
            logger.warning("الفئة '%s' موجودة مسبقاً", category)
            return
        names = initial_experts or [f"{category}_general"]
        self.groups[category] = ExpertGroup(
            category=category,
            d_model=self.d_model,
            d_ff=self.d_ff,
            expert_names=names,
            top_k=self.top_k_experts,
            dropout=self.dropout_p,
        )
        self._group_order.append(category)
        self.group_router.expand_targets(len(self._group_order))
        logger.info("أُضيفت الفئة '%s' مع خبراء: %s", category, names)

    def add_expert(
        self,
        category: str,
        name: str,
        description: str = "",
        create_category: bool = True,
    ) -> str:
        """
        إضافة خبير جديد ديناميكياً.
        إذا لم تكن الفئة موجودة و create_category=True تُنشأ تلقائياً.
        الخبراء الموجودون وأوزانهم لا تُمس.
        """
        if category not in self.groups:
            if not create_category:
                raise KeyError(f"الفئة '{category}' غير موجودة")
            self.add_category(category, initial_experts=[name])
            # حدّث الوصف إن لزم
            if name in self.groups[category].meta:
                self.groups[category].meta[name].description = description
            return name
        return self.groups[category].add_expert(name, description=description)

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(
        self, x: torch.Tensor, return_info: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: (B, d_model) أو (B, T, d_model) — إن كان ثلاثي الأبعاد يُعمل mean-pool للراوتر
        Returns:
            out: نفس شكل x إن كان (B,d) أو (B,T,d) بعد بثّ الأوزان
            aux_loss: خسارة موازنة الحمل الكلية (للجمع مع loss الرئيسي)
            info (اختياري): تفاصيل المسار الهرمي
        """
        token_level = x.dim() == 3
        if token_level:
            # راوتر على تمثيل مجمّع؛ الخبراء تُطبَّق على كل الرموز
            pooled = x.mean(dim=1)  # (B, d)
        else:
            pooled = x

        g_idx, g_w, g_probs, g_aux = self.group_router(pooled)
        B, Kg = g_idx.shape

        if token_level:
            out = torch.zeros_like(x)
        else:
            out = torch.zeros_like(pooled)

        total_aux = g_aux
        route_info: List[dict] = []

        for ki in range(Kg):
            gi = g_idx[:, ki]
            gw = g_w[:, ki]
            for group_i in gi.unique().tolist():
                mask = gi == group_i
                if not mask.any():
                    continue
                cat = self._group_order[int(group_i)]
                group: ExpertGroup = self.groups[cat]

                if token_level:
                    # طبّق على الرموز: نمرّر pooled للراوتر الفرعي، والخبراء على كل توكن
                    sub_pooled = pooled[mask]
                    # احصل على اختيار الخبراء من pooled
                    e_idx, e_w, e_probs, e_aux = group.router(sub_pooled)
                    total_aux = total_aux + e_aux
                    # طبّق الخبراء على x[mask] → (Bm, T, d)
                    sub_x = x[mask]
                    Bm, T, D = sub_x.shape
                    flat = sub_x.reshape(Bm * T, D)
                    sub_out = torch.zeros_like(flat)
                    K2 = e_idx.shape[1]
                    for kj in range(K2):
                        # بثّ اختيار الخبير عبر الزمن
                        expert_ids = e_idx[:, kj]  # (Bm,)
                        weights = e_w[:, kj]  # (Bm,)
                        for ei in expert_ids.unique().tolist():
                            m2 = expert_ids == ei
                            if not m2.any():
                                continue
                            eid = group._id_order[int(ei)]
                            # كل الرموز للعينات المختارة
                            token_mask = m2.unsqueeze(1).expand(Bm, T).reshape(Bm * T)
                            w_flat = weights.unsqueeze(1).expand(Bm, T).reshape(Bm * T)
                            active = token_mask
                            if not active.any():
                                continue
                            y = group.experts[eid](flat[active])
                            sub_out[active] = sub_out[active] + y * w_flat[active].unsqueeze(-1)
                            group.meta[eid].usage_count += int(active.sum().item())
                    sub_out = sub_out.view(Bm, T, D)
                    # وزن المجموعة
                    out[mask] = out[mask] + sub_out * gw[mask].view(-1, 1, 1)
                    route_info.append(
                        {
                            "category": cat,
                            "group_weight_mean": float(gw[mask].mean().item()),
                            "n_samples": int(mask.sum().item()),
                        }
                    )
                else:
                    sub_out, e_aux, info = group(pooled[mask])
                    total_aux = total_aux + e_aux
                    out[mask] = out[mask] + sub_out * gw[mask].unsqueeze(-1)
                    info["group_weight_mean"] = float(gw[mask].mean().item())
                    route_info.append(info)

        # Expert dropout (تدريب): إسقاط عشوائي لمساهمة الخبراء الموجَّهة
        if self.training and self.expert_dropout > 0:
            keep = (torch.rand(B, device=out.device) > self.expert_dropout).to(out.dtype)
            while keep.dim() < out.dim():
                keep = keep.unsqueeze(-1)
            out = out * keep

        # Shared expert (DeepSeek-style residual pathway)
        if self.shared_expert is not None:
            if token_level:
                B_, T_, D_ = x.shape
                shared_out = self.shared_expert(x.reshape(B_ * T_, D_)).view(B_, T_, D_)
            else:
                shared_out = self.shared_expert(pooled)
            out = out * (1.0 - self.shared_coeff) + shared_out * self.shared_coeff

        # Residual connection من الدخل (أفضل استقرار وتمثيل)
        if self.input_residual:
            residual = x if token_level else pooled
            out = out + residual

        out = self.out_norm(out)
        aux_loss = self.lb_coeff * total_aux

        if return_info:
            info_out = {
                "groups_selected": [
                    [self._group_order[int(i)] for i in row.tolist()]
                    for row in g_idx
                ],
                "group_weights": g_w.detach().cpu().tolist(),
                "routes": route_info,
                "aux_loss": float(aux_loss.detach().item()),
                "shared_expert": bool(self.shared_expert is not None),
                "shared_coeff": self.shared_coeff,
                "router_temperature": self.router_temperature,
                "capacity_factor": self.capacity_factor,
            }
            return out, aux_loss, info_out
        return out, aux_loss

    # ── إحصائيات وإدارة ──────────────────────────────────────────────────────
    def total_experts(self) -> int:
        return sum(g.expert_count() for g in self.groups.values())

    def list_categories(self) -> List[str]:
        return list(self._group_order)

    def list_all_experts(self) -> List[dict]:
        out: List[dict] = []
        for cat in self._group_order:
            out.extend(self.groups[cat].list_experts())
        return out

    def load_balance_report(self) -> str:
        """تقرير نصي عن توزيع الاستخدام — لكشف الخبراء المهمَلين."""
        lines = ["## ⚖️ تقرير موازنة حمل الخبراء (Load Balance)", ""]
        for cat in self._group_order:
            g: ExpertGroup = self.groups[cat]
            lines.append(f"### {cat} ({g.expert_count()} خبراء)")
            metas = g.list_experts()
            if not metas:
                lines.append("  (فارغ)")
                continue
            total_u = sum(m["usage_count"] for m in metas) or 1
            for m in sorted(metas, key=lambda d: -d["usage_count"]):
                pct = 100.0 * m["usage_count"] / total_u
                bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
                lines.append(
                    f"  - `{m['expert_id']}`: {m['usage_count']} ({pct:.1f}%) {bar}"
                )
            lines.append("")
        lines.append(f"**إجمالي الخبراء:** {self.total_experts()}")
        return "\n".join(lines)

    def summary(self) -> str:
        lines = [
            "## 🧩 Hierarchical Dynamic MoE — ملخص",
            f"- d_model={self.d_model} · d_ff={self.d_ff}",
            f"- top_k_groups={self.top_k_groups} · top_k_experts={self.top_k_experts}",
            f"- lb_coeff={self.lb_coeff}",
            f"- shared_expert={self.use_shared_expert} (coeff={self.shared_coeff})",
            f"- temperature={self.router_temperature} · z_loss={self.z_loss_coeff} · capacity={self.capacity_factor}",
            f"- فئات: {len(self._group_order)} → {self._group_order}",
            f"- إجمالي الخبراء: **{self.total_experts()}**",
            "",
        ]
        for cat in self._group_order:
            eids = self.groups[cat]._id_order
            lines.append(f"  • **{cat}**: {', '.join(eids)}")
        n_params = sum(p.numel() for p in self.parameters())
        lines.append(f"\n- معاملات قابلة للتدريب: **{n_params:,}**")
        return "\n".join(lines)

    def apply_best_config(self) -> "HierarchicalMoE":
        """تطبيق أفضل إعدادات MoE المعتمدة على النموذج الحالي."""
        cfg = BEST_MOE_CONFIG
        self.top_k_groups = int(cfg["top_k_groups"])
        self.top_k_experts = int(cfg["top_k_experts"])
        self.shared_coeff = float(cfg["shared_coeff"])
        self.router_temperature = float(cfg["router_temperature"])
        self.z_loss_coeff = float(cfg["z_loss_coeff"])
        self.capacity_factor = float(cfg["capacity_factor"])
        self.expert_dropout = float(cfg["expert_dropout"])
        self.lb_coeff = float(cfg["lb_coeff"])
        self.weight_threshold = float(cfg["weight_threshold"])
        self.input_residual = bool(cfg["input_residual"])
        self.use_shared_expert = bool(cfg["use_shared_expert"])
        if self.use_shared_expert and self.shared_expert is None:
            self.shared_expert = ExpertFFN(self.d_model, self.d_ff, self.dropout_p)
        # مزامنة الراوترات
        self.group_router.top_k = min(self.top_k_groups, self.group_router.n_targets)
        self.group_router.temperature = self.router_temperature
        self.group_router.z_loss_coeff = self.z_loss_coeff
        self.group_router.capacity_factor = self.capacity_factor
        self.group_router.weight_threshold = self.weight_threshold
        for g in self.groups.values():
            g.router.top_k = min(self.top_k_experts, g.router.n_targets)
            g.router.temperature = self.router_temperature
            g.router.z_loss_coeff = self.z_loss_coeff
            g.router.capacity_factor = self.capacity_factor
            g.router.weight_threshold = self.weight_threshold
        return self

    # ── حفظ / تحميل ──────────────────────────────────────────────────────────
    def save(self, path: Optional[str | Path] = None) -> Path:
        path = Path(path) if path else DEFAULT_SAVE_DIR / "hierarchical_moe.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "top_k_groups": self.top_k_groups,
            "top_k_experts": self.top_k_experts,
            "lb_coeff": self.lb_coeff,
            "dropout": self.dropout_p,
            "use_shared_expert": self.use_shared_expert,
            "shared_coeff": self.shared_coeff,
            "router_temperature": self.router_temperature,
            "z_loss_coeff": self.z_loss_coeff,
            "capacity_factor": self.capacity_factor,
            "expert_dropout": self.expert_dropout,
            "weight_threshold": self.weight_threshold,
            "input_residual": self.input_residual,
            "best_config_applied": True,
            "group_order": list(self._group_order),
            "experts": {
                cat: self.groups[cat]._id_order for cat in self._group_order
            },
            "expert_meta": {
                cat: {eid: self.groups[cat].meta[eid].to_dict() for eid in self.groups[cat]._id_order}
                for cat in self._group_order
            },
        }
        payload = {
            "meta": meta,
            "state_dict": self.state_dict(),
        }
        torch.save(payload, path)
        # JSON مرافق للقراءة البشرية
        side = path.with_suffix(".json")
        side.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("حُفظ HierarchicalMoE في %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path, map_location: str | None = None) -> "HierarchicalMoE":
        path = Path(path)
        payload = torch.load(path, map_location=map_location or "cpu", weights_only=False)
        meta = payload["meta"]
        categories = {cat: list(eids) for cat, eids in meta["experts"].items()}
        model = cls(
            d_model=meta["d_model"],
            d_ff=meta["d_ff"],
            categories=categories,
            top_k_groups=meta.get("top_k_groups", BEST_MOE_CONFIG["top_k_groups"]),
            top_k_experts=meta.get("top_k_experts", BEST_MOE_CONFIG["top_k_experts"]),
            dropout=meta.get("dropout", 0.05),
            lb_coeff=meta.get("lb_coeff", BEST_MOE_CONFIG["lb_coeff"]),
            use_shared_expert=meta.get("use_shared_expert", BEST_MOE_CONFIG["use_shared_expert"]),
            shared_coeff=meta.get("shared_coeff", BEST_MOE_CONFIG["shared_coeff"]),
            router_temperature=meta.get("router_temperature", BEST_MOE_CONFIG["router_temperature"]),
            z_loss_coeff=meta.get("z_loss_coeff", BEST_MOE_CONFIG["z_loss_coeff"]),
            capacity_factor=meta.get("capacity_factor", BEST_MOE_CONFIG["capacity_factor"]),
            expert_dropout=meta.get("expert_dropout", BEST_MOE_CONFIG["expert_dropout"]),
            weight_threshold=meta.get("weight_threshold", BEST_MOE_CONFIG["weight_threshold"]),
            input_residual=meta.get("input_residual", BEST_MOE_CONFIG["input_residual"]),
        )
        # استعادة ترتيب الفئات إن اختلف
        model._group_order = list(meta["group_order"])
        # إن كان الحفظ قديماً بدون best_config — طبّق الأفضل
        if not meta.get("best_config_applied"):
            model.apply_best_config()
        # استعادة meta الاستخدام
        for cat, emap in meta.get("expert_meta", {}).items():
            if cat not in model.groups:
                continue
            for eid, md in emap.items():
                if eid in model.groups[cat].meta:
                    model.groups[cat].meta[eid] = ExpertMeta.from_dict(md)
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
        if missing:
            logger.warning("مفاتيح ناقصة عند التحميل: %s", missing[:8])
        if unexpected:
            logger.warning("مفاتيح زائدة عند التحميل: %s", unexpected[:8])
        return model


# ══════════════════════════════════════════════════════════════════════════════
# 6) أدوات تدريب مساعدة + جسر أوامر نصية
# ══════════════════════════════════════════════════════════════════════════════
def train_step(
    model: HierarchicalMoE,
    x: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    main_loss_fn=None,
) -> Dict[str, float]:
    """خطوة تدريب واحدة: loss رئيسي + aux load-balancing."""
    model.train()
    optimizer.zero_grad()
    out, aux = model(x)
    if main_loss_fn is None:
        main_loss = F.mse_loss(out, target)
    else:
        main_loss = main_loss_fn(out, target)
    loss = main_loss + aux
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "main": float(main_loss.item()),
        "aux": float(aux.item()),
    }


def build_default_moe(
    d_model: int = 128,
    top_k_groups: int | None = None,
    top_k_experts: int | None = None,
) -> HierarchicalMoE:
    """مصنع سريع — دائماً بأفضل إعدادات MoE المعتمدة."""
    cfg = BEST_MOE_CONFIG
    return HierarchicalMoE(
        d_model=d_model,
        top_k_groups=top_k_groups if top_k_groups is not None else int(cfg["top_k_groups"]),
        top_k_experts=top_k_experts if top_k_experts is not None else int(cfg["top_k_experts"]),
        categories={k: list(v) for k, v in DEFAULT_CATEGORIES.items()},
        use_shared_expert=bool(cfg["use_shared_expert"]),
        shared_coeff=float(cfg["shared_coeff"]),
        router_temperature=float(cfg["router_temperature"]),
        z_loss_coeff=float(cfg["z_loss_coeff"]),
        capacity_factor=float(cfg["capacity_factor"]),
        expert_dropout=float(cfg["expert_dropout"]),
        lb_coeff=float(cfg["lb_coeff"]),
        weight_threshold=float(cfg["weight_threshold"]),
        input_residual=bool(cfg["input_residual"]),
    )


def handle_moe_command(user_input: str) -> Optional[str]:
    """
    أوامر نصية عربية لإدارة الـ MoE (تُستدعى من وكيل التدريب أو الجسر).
    تعيد نصاً أو None إن لم يُطابق الأمر.
    """
    import re

    text = (user_input or "").strip()
    if not text:
        return None

    # تفعيل فقط عند ذكر MoE / خبراء / راوتر هرمي
    if not re.search(
        r"(moe|مxture|خليط\s*خبراء|خبراء\s*هرمي|hierarchical\s*moe|"
        r"راوتر\s*هرمي|نظام\s*الخبراء|mixture\s*of\s*experts)",
        text,
        re.I,
    ) and not re.search(
        r"(أضف\s*خبير|اضف\s*خبير|add\s*expert|قائمة\s*خبراء|"
        r"ملخص\s*moe|تقرير\s*موازنة|load\s*balance)",
        text,
        re.I,
    ):
        return None

    state_path = DEFAULT_SAVE_DIR / "hierarchical_moe.pt"

    def _load_or_create() -> HierarchicalMoE:
        if state_path.is_file():
            try:
                return HierarchicalMoE.load(state_path)
            except Exception as e:
                logger.warning("فشل تحميل MoE: %s — إنشاء جديد", e)
        return build_default_moe()

    # ملخص
    if re.search(r"(ملخص|summary|حالة).{0,12}(moe|خبراء|هرمي)", text, re.I) or re.search(
        r"(moe|خبراء).{0,8}(ملخص|summary|حالة)", text, re.I
    ):
        m = _load_or_create()
        return m.summary()

    # تقرير موازنة
    if re.search(r"(موازنة|load\s*balance|توزيع\s*خبراء|تقرير\s*حمل)", text, re.I):
        m = _load_or_create()
        return m.load_balance_report()

    # قائمة خبراء
    if re.search(r"(قائمة\s*خبراء|list\s*experts|عرض\s*خبراء)", text, re.I):
        m = _load_or_create()
        lines = ["## 📋 قائمة خبراء Hierarchical MoE", ""]
        for e in m.list_all_experts():
            lines.append(
                f"- `{e['category']}/{e['expert_id']}` — {e.get('description') or e['name']} "
                f"(استخدام: {e['usage_count']})"
            )
        lines.append(f"\n**المجموع:** {m.total_experts()}")
        return "\n".join(lines)

    # إضافة خبير: أضف خبير فقه اسمه fiqh_zahiri
    m_add = re.search(
        r"(?:أضف|اضف|add)\s*خبير\s+(?:(?:في|لفئة|category)\s+)?(\w+)\s+(?:اسم(?:ه|ها)?\s+)?(\w+)",
        text,
        re.I,
    )
    if m_add:
        cat, name = m_add.group(1), m_add.group(2)
        m = _load_or_create()
        eid = m.add_expert(cat, name, description=f"خبير مضاف ديناميكياً: {name}")
        m.save(state_path)
        return (
            f"✅ أُضيف الخبير `{eid}` تحت الفئة `{cat}`.\n"
            f"إجمالي الخبراء الآن: **{m.total_experts()}**\n"
            f"حُفظ في `{state_path.relative_to(ROOT)}`"
        )

    # إضافة فئة
    m_cat = re.search(
        r"(?:أضف|اضف|add)\s*فئة\s+(\w+)",
        text,
        re.I,
    )
    if m_cat:
        cat = m_cat.group(1)
        m = _load_or_create()
        m.add_category(cat)
        m.save(state_path)
        return f"✅ أُضيفت الفئة `{cat}` مع خبير ابتدائي.\n" + m.summary()

    # تهيئة / بناء
    if re.search(r"(ابنِ|ابني|أنشئ|انشئ|build|init).{0,15}(moe|خبراء|هرمي)", text, re.I):
        m = build_default_moe()
        m.save(state_path)
        return "✅ تم بناء Hierarchical MoE الافتراضي وحفظه.\n\n" + m.summary()

    # اختبار سريع
    if re.search(r"(اختبر|test).{0,10}(moe|خبراء)", text, re.I):
        m = _load_or_create()
        m.eval()
        with torch.no_grad():
            x = torch.randn(2, m.d_model)
            out, aux, info = m(x, return_info=True)
        return (
            f"## 🧪 اختبار Forward\n"
            f"- شكل المخرج: `{tuple(out.shape)}`\n"
            f"- aux_loss: **{aux.item():.6f}**\n"
            f"- مجموعات مختارة: {info['groups_selected']}\n"
            f"- أوزان المجموعات: {[[round(w, 3) for w in row] for row in info['group_weights']]}\n"
        )

    # افتراضي عند ذكر MoE
    if re.search(r"\bmoe\b|خليط\s*خبراء|هرمي", text, re.I):
        m = _load_or_create()
        return (
            m.summary()
            + "\n\n---\n**أوامر:** `ملخص moe` · `قائمة خبراء` · `تقرير موازنة` · "
            "`أضف خبير fiqh اسمه fiqh_zahiri` · `أضف فئة usul` · `اختبر moe` · `ابنِ moe`"
        )

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 7) فحص ذاتي سريع عند التشغيل المباشر
# ══════════════════════════════════════════════════════════════════════════════
def _self_test() -> None:
    print("=== Hierarchical MoE self-test ===")
    m = build_default_moe(d_model=64, top_k_groups=2, top_k_experts=2)
    print(m.summary())
    x = torch.randn(4, 64)
    out, aux, info = m(x, return_info=True)
    assert out.shape == x.shape, out.shape
    assert aux.ndim == 0
    print("forward OK", out.shape, "aux", float(aux.detach()))
    print("routes sample:", info["groups_selected"][0])

    # ديناميكية
    m.add_expert("fiqh", "fiqh_zahiri", "فقه ظاهري")
    m.add_category("usul", ["usul_general"])
    assert m.total_experts() >= 15
    out2, aux2 = m(x)
    assert out2.shape == x.shape

    # تدريب خطوة
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    stats = train_step(m, x, torch.randn_like(x), opt)
    print("train_step", stats)

    # حفظ/تحميل
    p = DEFAULT_SAVE_DIR / "_test_moe.pt"
    m.save(p)
    m2 = HierarchicalMoE.load(p)
    assert m2.total_experts() == m.total_experts()
    out3, _ = m2(x)
    assert out3.shape == x.shape
    p.unlink(missing_ok=True)
    p.with_suffix(".json").unlink(missing_ok=True)
    print("save/load OK")
    print("ALL PASSED")


if __name__ == "__main__":
    _self_test()
