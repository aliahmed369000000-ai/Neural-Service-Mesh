"""
Web Fetcher — جالب المعرفة من الإنترنت
========================================
يجلب المعرفة الحية من:
  - ويكيبيديا العربية (API)
  - GitHub (API عام - أكواد ومشاريع)

لا يُخزَّن شيء في ملفات منفصلة — كل شيء يذهب مباشرة إلى KnowledgeTrainer.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("WebFetcher")

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False


def _get(url: str, params: Dict = None, timeout: int = 8) -> Optional[Dict]:
    if not _REQUESTS_OK:
        logger.warning("requests غير متاح")
        return None
    try:
        r = _requests.get(url, params=params, timeout=timeout,
                          headers={"User-Agent": "NeuralMesh/18.0 (knowledge-trainer)"})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"HTTP fetch failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# ويكيبيديا العربية
# ═══════════════════════════════════════════════════════════════════════════

WIKIPEDIA_AR_TOPICS = [
    # علوم
    "ذكاء اصطناعي", "تعلم آلي", "شبكة عصبية اصطناعية", "معالجة اللغة الطبيعية",
    "علم الحاسوب", "خوارزمية", "قاعدة بيانات", "تشفير",
    # فيزياء
    "ميكانيكا الكم", "نسبية عامة", "فيزياء الجسيمات", "الفيزياء الفلكية",
    "موجة كهرومغناطيسية", "الديناميكا الحرارية", "الفيزياء النووية",
    # رياضيات
    "حساب التفاضل والتكامل", "جبر خطي", "نظرية الاحتمالات", "إحصاء",
    "منطق رياضي", "نظرية المجموعات", "هندسة تفاضلية",
    # أحياء
    "تطور بيولوجي", "وراثة", "خلية", "جهاز مناعي", "علم الأعصاب",
    "نظام بيئي", "حمض نووي ريبوزي منقوص الأكسجين",
    # تاريخ وحضارات
    "الخلافة الإسلامية", "بلاد الرافدين", "الحضارة المصرية القديمة",
    "الثورة الصناعية", "الحرب العالمية الثانية", "الفتح الإسلامي",
    # فلسفة وفكر
    "فلسفة", "منطق", "أخلاق", "معرفة", "وعي",
    # طب وصحة
    "طب", "علم الصيدلة", "علم الأوبئة", "جراحة",
    # اقتصاد واجتماع
    "اقتصاد", "علم الاجتماع", "ديمقراطية", "قانون دولي",
]


def fetch_wikipedia_items(
    topics: Optional[List[str]] = None,
    max_items: int = 50,
    lang: str = "ar",
) -> List[Dict[str, Any]]:
    """
    يجلب مقالات ويكيبيديا ويحوّلها إلى عناصر قابلة للتدريب.
    يُرجع قائمة من العناصر بصيغة KnowledgeTrainer.
    """
    if topics is None:
        topics = WIKIPEDIA_AR_TOPICS[:max_items]
    else:
        topics = topics[:max_items]

    base_url = f"https://{lang}.wikipedia.org/w/api.php"
    items: List[Dict[str, Any]] = []

    for topic in topics:
        data = _get(base_url, params={
            "action":    "query",
            "titles":    topic,
            "prop":      "extracts|categories|links",
            "exintro":   True,
            "explaintext": True,
            "pllimit":   10,
            "format":    "json",
            "redirects": 1,
        })
        if not data:
            continue

        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1":
                continue
            title   = page.get("title", topic)
            extract = page.get("extract", "").strip()
            if not extract or len(extract) < 30:
                continue

            # استخراج الفئات كعلاقات
            cats = page.get("categories", [])
            relations = []
            for cat in cats[:5]:
                cat_title = cat.get("title", "").replace("تصنيف:", "").strip()
                if cat_title and len(cat_title) < 50:
                    relations.append({
                        "target": cat_title,
                        "type":   "belongs_to",
                        "weight": 0.6,
                    })

            # استخراج الروابط كعلاقات
            links = page.get("links", [])
            for lnk in links[:5]:
                lnk_title = lnk.get("title", "").strip()
                if lnk_title and len(lnk_title) < 60:
                    relations.append({
                        "target": lnk_title,
                        "type":   "links_to",
                        "weight": 0.4,
                    })

            # تحديد التجميع من الفئات
            cluster = "موسوعة"
            for cat in cats:
                ct = cat.get("title", "")
                if "علوم" in ct:      cluster = "علوم"; break
                if "فيزياء" in ct:   cluster = "فيزياء"; break
                if "رياضيات" in ct:  cluster = "رياضيات"; break
                if "أحياء" in ct:    cluster = "أحياء"; break
                if "تاريخ" in ct:    cluster = "تاريخ"; break
                if "فلسفة" in ct:    cluster = "فلسفة"; break

            items.append({
                "concept":     title,
                "text":        extract[:800],
                "cluster":     cluster,
                "importance":  0.7,
                "certainty":   0.85,
                "abstraction": 0.5,
                "relations":   relations,
            })

        time.sleep(0.1)  # احترام حدود الـ API

    logger.info(f"Wikipedia AR: جُلب {len(items)} مقالة من {len(topics)} موضوع")
    return items


# ═══════════════════════════════════════════════════════════════════════════
# GitHub — أكواد ومشاريع
# ═══════════════════════════════════════════════════════════════════════════

GITHUB_SEARCHES = [
    "machine learning",
    "neural network",
    "natural language processing",
    "computer vision",
    "data structures algorithms",
    "distributed systems",
    "operating system",
    "compiler",
    "cryptography",
    "web framework",
    "database engine",
    "quantum computing",
    "reinforcement learning",
    "transformer model",
    "graph neural network",
]


def fetch_github_items(
    queries: Optional[List[str]] = None,
    max_per_query: int = 5,
    max_total: int = 60,
) -> List[Dict[str, Any]]:
    """
    يجلب مستودعات GitHub الأكثر نجومًا لكل موضوع.
    يُرجع عناصر قابلة للتدريب.
    """
    if queries is None:
        queries = GITHUB_SEARCHES

    items: List[Dict[str, Any]] = []
    seen: set = set()

    for query in queries:
        if len(items) >= max_total:
            break

        data = _get(
            "https://api.github.com/search/repositories",
            params={
                "q":       query,
                "sort":    "stars",
                "order":   "desc",
                "per_page": max_per_query,
            },
        )
        if not data:
            continue

        repos = data.get("items", [])
        for repo in repos:
            name = repo.get("full_name", "")
            if not name or name in seen:
                continue
            seen.add(name)

            desc    = repo.get("description") or ""
            stars   = repo.get("stargazers_count", 0)
            lang    = repo.get("language") or "unknown"
            topics  = repo.get("topics", [])
            concept = repo.get("name", name.split("/")[-1])

            text = (
                f"{concept}: {desc}. "
                f"لغة: {lang}. نجوم: {stars:,}. "
                f"موضوعات: {', '.join(topics[:5])}."
            ).strip()

            # الأهمية بناءً على النجوم
            importance = min(1.0, 0.4 + (stars / 200_000))

            relations = [{"target": lang, "type": "written_in", "weight": 0.8}]
            for t in topics[:4]:
                relations.append({"target": t, "type": "tagged", "weight": 0.6})

            items.append({
                "concept":     concept,
                "text":        text[:600],
                "cluster":     f"github:{lang.lower()}",
                "importance":  round(importance, 3),
                "certainty":   0.9,
                "abstraction": 0.4,
                "relations":   relations,
            })

        time.sleep(0.15)

    logger.info(f"GitHub: جُلب {len(items)} مستودع")
    return items


# ═══════════════════════════════════════════════════════════════════════════
# واجهة موحدة
# ═══════════════════════════════════════════════════════════════════════════

def fetch_all_web(
    wiki_max: int = 40,
    github_max: int = 50,
) -> Dict[str, List[Dict[str, Any]]]:
    """يجلب جميع المصادر الإلكترونية دفعة واحدة."""
    return {
        "wikipedia": fetch_wikipedia_items(max_items=wiki_max),
        "github":    fetch_github_items(max_total=github_max),
    }
