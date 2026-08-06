"""
غلاف اختياري لـ CAMeL Tools — يفشل بهدوء إن لم تُثبَّت المكتبة/البيانات.

الفائدة لـ NSM:
  - كشف لهجة على مستوى المدن (بما فيها SAN = صنعاء/يمن)
  - تطبيع/إزالة تشكيل بأدوات CAMeL إن توفرت
  - تحليل صرفي MSA اختياري (لا يوجد DB يمني في CAMeL)

التثبيت (جهاز قوي، ليس ضرورياً للسحابة):
  pip install camel-tools
  camel_data -i dialectid-default   # لكشف اللهجة
  camel_data -i light               # للصرف MSA/EGY
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

logger = logging.getLogger("camel_optional")

# رموز يمنية/جزيرية مفيدة
YEMEN_CITY_LABEL = "SAN"  # Sana'a
GULF_LABELS = frozenset({"SAN", "JED", "RIY", "DOH", "MUS", "BAG", "BAS", "MOS"})


def camel_available() -> bool:
    try:
        import camel_tools  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _did_model():
    try:
        from camel_tools.dialectid import DialectIdentifier
        return DialectIdentifier.pretrained()
    except Exception as e:
        logger.debug("CAMeL DID غير متاح: %s", e)
        return None


def identify_dialect(text: str, level: str = "city") -> Dict[str, Any]:
    """
    يُرجع:
      available, top, scores, is_yemeni_san, yemen_score, level
    """
    out: Dict[str, Any] = {
        "available": False,
        "top": None,
        "scores": {},
        "is_yemeni_san": False,
        "yemen_score": 0.0,
        "level": level,
    }
    if not (text or "").strip():
        return out
    model = _did_model()
    if model is None:
        return out
    try:
        preds = model.predict([text], level if level in ("city", "country", "region") else "city")
        p = preds[0]
        scores = dict(getattr(p, "scores", {}) or {})
        top = getattr(p, "top", None)
        yemen = float(scores.get(YEMEN_CITY_LABEL, 0.0) or 0.0)
        # بعض الإصدارات تستخدم اسم المدينة كاملاً
        if yemen == 0.0:
            for k, v in scores.items():
                if str(k).upper() in ("SAN", "SANA'A", "SANAA") or "sana" in str(k).lower():
                    yemen = float(v)
                    break
        out.update({
            "available": True,
            "top": top,
            "scores": {str(k): float(v) for k, v in list(scores.items())[:30]},
            "is_yemeni_san": (str(top).upper() in ("SAN", "SANA'A", "SANAA")) or yemen >= 0.15,
            "yemen_score": yemen,
        })
    except Exception as e:
        logger.warning("فشل CAMeL DID: %s", e)
    return out


def camel_normalize(text: str) -> str:
    """تطبيع CAMeL إن وُجد، وإلا النص كما هو."""
    if not text:
        return ""
    try:
        from camel_tools.utils.normalize import normalize_unicode, normalize_alef_ar, normalize_teh_marbuta_ar
        from camel_tools.utils.dediac import dediac_ar
        t = normalize_unicode(text)
        t = normalize_alef_ar(t)
        t = normalize_teh_marbuta_ar(t)
        t = dediac_ar(t)
        return t
    except Exception:
        return text


def camel_analyze_msa(word: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """تحليل صرفي MSA اختياري لكلمة واحدة."""
    try:
        from camel_tools.morphology.database import MorphologyDB
        from camel_tools.morphology.analyzer import Analyzer
        db = MorphologyDB.builtin_db()
        analyzer = Analyzer(db)
        analyses = analyzer.analyze(word)[:top_k]
        out = []
        for a in analyses:
            out.append({
                "diac": a.get("diac"),
                "lex": a.get("lex"),
                "pos": a.get("pos"),
                "stem": a.get("stem"),
                "gloss": a.get("gloss"),
            })
        return out
    except Exception:
        return []
