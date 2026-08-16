"""
خبرة التوجيه التكيفية (Routing Experience) — تعلّم توجيه الطلبات.

نظام التوجيه الحالي (agent_categories.py عبر route_query_verbose) يقرر
أي وكلاء يستلمون كل طلب من تشابه دلالي آنّي فقط. لا يتذكّر أي طلبات
سابقة «توجهت بشكل خاطئ»: قد يكلّف مهمةً برمجيةً بوكيل تحليلي ثم يضطر
لمسار النسخ الاحتياطي، أو يعطّل فريقًا كاملًا على سؤال بسيط.

هذه الوحدة تبني خبرة توجيه تراكمية:
1) بعد كل جولة توجيه ناجحة (فريق/وكيل واحد) تُسجَّل بصمة للطلب
   (كلمات مفتاحية مصنفة: رقمي/برمجة/تحليل/بحث/عام...) مع route_method
   والوكلاء المختارين ونسبة نجاح جولة التوليف (تُستنتج من events
   reflect_gave_up/agent_error/reflect_resolved).
2) في كل طلب جديد تُطابَق البصمة بالدروس المحفوظة؛ إن وُجدت خبرة قريبة
   تُدرَج كتلميح إضافي في system prompt للمدير الموحّد: «مهمة مشابهة
   سابقًا نَجَحَت مع [الوكلاء]».
3) الخبرة تتقدم بالعدّاد التجريبي (successes/failures) مثل reflection_memory
   — التجارب المثبتة تاريخيًا تتفوق على التخمينات الجديدة.

التكامل: يستدعيها director الموحّد بعد كل chat() (عبر سجل turns_meta)
وعند كل طلب جديد (عبر format_routing_hints التي تُدخل تلميحاتها في
prompt المدير — استدعاء اختياري مضمن في _synthesize).
التخزين: memory/routing_experience.json — ذاكرة اختيارية لا تكسر المسار
الأصلي بأي فشل.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory")
_MEMORY_FILE = os.path.join(_MEMORY_DIR, "routing_experience.json")

_MAX_EXPERIENCES = 300
_PROVEN_THRESHOLD = 2  # عدد النجاحات حتى تُعتبر الخبرة مثبتة

# بصمات الطلبات: تصنيف نصي سريع للطلب
_FINGERPRINT_RULES: List[tuple] = [
    ("code",     r"(اكتب|عدّل|عدل|أصلح|اصلاح|إصلاح|برمج|شغّل|شغل|run_file|edit_file|git|كود|python|py)"),
    ("analysis", r"(حلل|تحليل|قارن|مقارنة|لماذا|كيف|فسّر|اشرح|ما الفرق|اقتراح|اقتراحات)"),
    ("search",   r"(ابحث عن|بحث عن|أحدث|معلومات عن|عن ماذا|جد لي|ابحث في)"),
    ("writing",  r"(اكتب مقال|اكتب تقرير|لخّص|لخص|صياغة|إعادة صياغة|ترجم)"),
    ("numbers",  r"\d"),
]


def fingerprint_request(text: str) -> List[str]:
    """يصنّف الطلب إلى بصمات ممكنة (قد يحمل أكثر من فئة)."""
    low = text.lower()
    tags: List[str] = []
    for tag, pattern in _FINGERPRINT_RULES:
        if re.search(pattern, low):
            tags.append(tag)
    return tags or ["general"]


class RoutingExperience:
    """سجل خبرات التوجيه مع مطابقة بصمات وعدّادات تجريبية."""

    def __init__(self, path: str = _MEMORY_FILE) -> None:
        self.path = path
        self.experiences: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                data = raw.get("experiences", raw) if isinstance(raw, dict) else raw
                self.experiences = [e for e in data if isinstance(e, dict)]
        except Exception as exc:  # pragma: no cover
            logger.warning("routing_experience: تعذّر التحميل: %s", exc)
            self.experiences = []

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"experiences": self.experiences[-_MAX_EXPERIENCES:]},
                          f, ensure_ascii=False, indent=1)
        except Exception as exc:  # pragma: no cover
            logger.warning("routing_experience: تعذّر الحفظ: %s", exc)

    @staticmethod
    def _fp_score(a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        return len(set(a) & set(b)) / len(set(a) | set(b))

    def record(self, request: str, route_method: str, selected_agents: List[str],
               ok: bool) -> None:
        """يسجّل نتيجة توجيه جولة (ok=True إن نجح التوليف دون استسلامات)."""
        fp = fingerprint_request(request)
        try:
            for exp in self.experiences:
                if exp.get("route_method") == route_method and exp.get("agents") == selected_agents:
                    key_match = all(set(fp) & set(exp.get("fingerprints", []))) if fp else False
                    if key_match or exp.get("fingerprints") == fp:
                        if ok:
                            exp["successes"] = int(exp.get("successes", 0)) + 1
                        else:
                            exp["failures"] = int(exp.get("failures", 0)) + 1
                        exp["last_seen"] = _now_iso()
                        self.save()
                        return
            self.experiences.append({
                "fingerprints": fp,
                "route_method": route_method,
                "agents": list(selected_agents),
                "successes": 1 if ok else 0,
                "failures": 0 if ok else 1,
                "first_seen": _now_iso(),
                "last_seen": _now_iso(),
            })
            self.save()
        except Exception as exc:  # pragma: no cover
            logger.warning("routing_experience: تعذّر التسجيل: %s", exc)

    def hints_for(self, request: str) -> str:
        """
        يولّد تلميح توجيه مبنيًا على الخبرات المثبتة الأقرب للطلب.
        نص موجّه للمدير الموحّد داخل prompt التوليف — إن لم توجد خبرة
        مثبتة قريبة يُعاد نص فارغ (لا تلميحات بلا أساس).
        """
        try:
            fp = fingerprint_request(request)
            proven = [
                e for e in self.experiences
                if int(e.get("successes", 0)) >= _PROVEN_THRESHOLD
                and self._fp_score(fp, e.get("fingerprints", [])) >= 0.5
            ]
            if not proven:
                return ""
            proven.sort(key=lambda e: int(e.get("successes", 0)), reverse=True)
            top = proven[:3]
            lines = [
                f"• مهمة مشابهة سابقًا نَجَحَت {int(e.get('successes', 0))} مرات عبر «{e.get('route_method')}» "
                f"مع {', '.join(e.get('agents', []))}"
                for e in top
            ]
            return "📊 خبرات توجيه مثبتة لمهمة مشابهة (اعتمد عليها في حسمك):\n" + "\n".join(lines)
        except Exception as exc:  # pragma: no cover
            logger.warning("routing_experience: تعذّر توليد التلميحات: %s", exc)
            return ""

    def summary(self) -> Dict[str, Any]:
        total = len(self.experiences)
        proven = len([e for e in self.experiences if int(e.get("successes", 0)) >= _PROVEN_THRESHOLD])
        return {"total": total, "proven": proven, "top": sorted(
            self.experiences, key=lambda e: int(e.get("successes", 0)), reverse=True)[:5]}


def _now_iso() -> str:
    try:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
    except Exception:  # pragma: no cover
        return ""


_instance: Optional[RoutingExperience] = None


def get_routing_experience() -> RoutingExperience:
    global _instance
    if _instance is None:
        _instance = RoutingExperience()
    return _instance
