"""
NSM Semantic Router — التوجيه الدلالي الذكي
============================================
يُصنِّف كل استعلام إلى فئة دلالية (عربي/إسلامي، برمجة، إبداعي، تحليل، عام)
ويُرتَّب العقد المتاحة بحسب ملاءمتها للفئة — يُدمَج مع ScoringEngine التاريخي
لاتخاذ قرار التوجيه المركَّب الأمثل.

وزن التوجيه:
  65% ScoringEngine التاريخي  +  35% التحيُّز الدلالي
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ── أسماء العقد (مطابقة لـ nsm_router_bridge) ───────────────────────────
NODE_OPENROUTER  = "nsm:openrouter"
NODE_AGENT       = "nsm:agent"
NODE_FREE_ROUTER = "nsm:free_router"

# ── كلمات مفتاحية لكل فئة ────────────────────────────────────────────────
CATEGORIES: Dict[str, List[str]] = {
    "arabic_islamic": [
        # عربية وإسلامية
        "قرآن", "حديث", "إسلام", "مسلم", "سورة", "آية", "تفسير",
        "فقه", "صلاة", "زكاة", "صوم", "حج", "عقيدة", "توحيد",
        "رسول", "نبي", "صحابة", "سيرة", "شريعة", "حلال", "حرام",
        "دعاء", "ذكر", "تسبيح", "استغفار", "بسملة", "الله",
        "نحو", "صرف", "بلاغة", "لغة عربية", "جذر عربي", "اشتقاق",
        "الرحمن", "الرحيم", "بسم", "القرآن الكريم",
        # إنجليزي
        "quran", "hadith", "islam", "surah", "ayah", "tafsir",
        "salah", "zakat", "fiqh", "aqeedah", "seerah", "arabic",
        "mosque", "prayer", "ramadan", "halal", "haram",
    ],
    "code": [
        # عربي
        "كود", "برمجة", "كوّد", "خطأ برمجي", "بُغ", "سكريبت", "دالة",
        "كلاس", "مكتبة", "باثون", "جافاسكريبت", "ديباغ", "خوارزمية",
        "قاعدة بيانات", "واجهة برمجية",
        # إنجليزي
        "code", "python", "javascript", "typescript", "function",
        "class", "bug", "error", "debug", "algorithm", "api",
        "database", "sql", "html", "css", "react", "git", "github",
        "programming", "script", "library", "import", "syntax",
        "compile", "runtime", "exception", "loop", "array", "json",
        "server", "backend", "frontend", "docker", "deploy", "linux",
        "bash", "curl", "pip", "npm", "install", "module", "def",
    ],
    "creative": [
        # عربي
        "قصة", "رواية", "قصيدة", "شعر", "أدب", "خيال", "مسرحية",
        "ابتكر", "اكتب قصة", "أنشئ قصيدة", "تخيّل", "أبدع", "تأليف",
        "احكِ", "صف", "وصف أدبي", "حوار", "سرد",
        # إنجليزي
        "story", "poem", "poetry", "fiction", "creative", "write a",
        "novel", "essay", "lyrics", "script", "imagine", "invent",
        "generate a story", "compose", "create a", "roleplay",
        "character", "plot", "narrative", "fantasy",
    ],
    "analysis": [
        # عربي
        "حلّل", "قارن", "فسّر", "اشرح", "لماذا", "كيف يعمل",
        "ما الفرق", "مزايا وعيوب", "ملخص", "خلاصة", "نقد", "تقييم",
        "ما أفضل", "يقارن بين", "ما هو",
        # إنجليزي
        "analyze", "compare", "explain", "why", "how does", "difference",
        "pros and cons", "summary", "evaluate", "assess", "review",
        "advantages", "disadvantages", "what is", "what are",
        "when to use", "which is better",
    ],
}

# ── تفضيل العقد لكل فئة (الأول = الأعلى أولوية) ─────────────────────────
CATEGORY_NODE_PREFERENCE: Dict[str, List[str]] = {
    # NSM Agent أفضل بالعربية والإسلاميات (مخصَّص ومُتدرَّب عليها)
    "arabic_islamic": [NODE_AGENT, NODE_FREE_ROUTER, NODE_OPENROUTER],
    # OpenRouter أقوى في الكود (GPT-4o, Claude-3.5)
    "code":           [NODE_OPENROUTER, NODE_AGENT, NODE_FREE_ROUTER],
    # OpenRouter أفضل في الإبداع
    "creative":       [NODE_OPENROUTER, NODE_AGENT, NODE_FREE_ROUTER],
    # OpenRouter أقوى في التحليل العميق
    "analysis":       [NODE_OPENROUTER, NODE_AGENT, NODE_FREE_ROUTER],
    # الافتراضي: Agent أولاً
    "general":        [NODE_AGENT, NODE_OPENROUTER, NODE_FREE_ROUTER],
}

# وزن التحيُّز الدلالي في القرار المركَّب (0=لا تأثير، 1=حسمي)
SEMANTIC_BIAS_WEIGHT: float = 0.35

# تسميات الفئات للعرض
CATEGORY_LABELS: Dict[str, Tuple[str, str]] = {
    "arabic_islamic": ("🕌", "عربي/إسلامي"),
    "code":           ("💻", "برمجة"),
    "creative":       ("✍️", "إبداعي"),
    "analysis":       ("🔍", "تحليل"),
    "general":        ("💬", "عام"),
}


class SemanticRouter:
    """
    يُصنِّف الاستعلام ويُوجِّه العقد بناءً على الفئة الدلالية.
    """

    def classify(self, query: str) -> Tuple[str, float]:
        """
        يُعيد (category, confidence).
        confidence ∈ [0, 1] — كلما ارتفعت زادت قوة التحيُّز.
        """
        q_lower = query.lower()
        hits: Dict[str, int] = {}

        for cat, keywords in CATEGORIES.items():
            count = sum(1 for kw in keywords if kw.lower() in q_lower)
            if count:
                hits[cat] = count

        if not hits:
            return "general", 0.2

        best_cat = max(hits, key=lambda c: hits[c])
        total_kw  = len(CATEGORIES.get(best_cat, [1]))
        # confidence: نسبة الكلمات المُطابَقة مع مُعزِّز × 3 (max 1.0)
        confidence = min(hits[best_cat] / max(total_kw, 1) * 3, 1.0)

        logger.debug(
            f"SemanticRouter: «{query[:40]}» → {best_cat} "
            f"(hits={hits[best_cat]}, conf={confidence:.2f})"
        )
        return best_cat, confidence

    def bias_order(
        self,
        category: str,
        available_nodes: List[str],
        confidence: float = 0.5,
    ) -> List[str]:
        """
        يُعيد قائمة العقد مُعاد ترتيبها بحسب التفضيل الدلالي.
        إذا كانت الثقة منخفضة جداً (< 0.2) يُرجع القائمة كما هي.
        """
        if confidence < 0.2 or not available_nodes:
            return list(available_nodes)

        preferred = CATEGORY_NODE_PREFERENCE.get(category, [])
        ordered: List[str] = []
        for node in preferred:
            if node in available_nodes:
                ordered.append(node)
        # أضف ما تبقّى (ليس في التفضيل)
        for node in available_nodes:
            if node not in ordered:
                ordered.append(node)
        return ordered

    def semantic_score(self, category: str, node_id: str) -> float:
        """
        درجة الملاءمة الدلالية (0–100) للعقدة مع الفئة.
        تُستخدم لتعديل connection_score من ScoringEngine.
        """
        preferred = CATEGORY_NODE_PREFERENCE.get(category, [])
        try:
            rank   = preferred.index(node_id)
            scores = [100.0, 60.0, 30.0]
            return scores[rank] if rank < len(scores) else 20.0
        except ValueError:
            return 40.0

    def combined_score(
        self,
        category: str,
        node_id: str,
        historical_score: float,
        confidence: float = 0.5,
    ) -> float:
        """
        الدرجة المركَّبة:  65% تاريخي + 35% دلالي (مُعدَّل بالثقة).
        historical_score ∈ [0, 100]
        """
        sem   = self.semantic_score(category, node_id)
        w_sem = SEMANTIC_BIAS_WEIGHT * min(confidence, 1.0)
        w_his = 1.0 - w_sem
        return round(w_his * historical_score + w_sem * sem, 2)


# ── مفرد مُشترك (singleton) ──────────────────────────────────────────────
_router = SemanticRouter()


def classify(query: str) -> Tuple[str, float]:
    return _router.classify(query)


def bias_order(
    category: str, available_nodes: List[str], confidence: float = 0.5
) -> List[str]:
    return _router.bias_order(category, available_nodes, confidence)


def semantic_score(category: str, node_id: str) -> float:
    return _router.semantic_score(category, node_id)


def combined_score(
    category: str, node_id: str, historical_score: float, confidence: float = 0.5
) -> float:
    return _router.combined_score(category, node_id, historical_score, confidence)
