"""Runtime helpers for the three high-value NSM specialist agents.

This module is deliberately dependency-light and deterministic.  It does not
call an LLM and it never performs writes, which makes it safe to use before
and after an agent response for evidence and quality controls.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse


@dataclass(frozen=True)
class SourceRecord:
    """A normalized, deduplicated research source."""

    title: str
    url: str
    snippet: str
    source: str

    @property
    def domain(self) -> str:
        return (urlparse(self.url).netloc or self.source or "مصدر غير معروف").lower()

    @property
    def fingerprint(self) -> str:
        return sha256(self.url.encode("utf-8")).hexdigest()[:12]


_SPECIALIST_PROTOCOLS: Dict[str, str] = {
    "research": (
        "بروتوكول وكيل البحث: ميّز بين الحقائق والاستنتاجات، واربط كل ادعاء "
        "حديث أو قابل للتحقق بمصدر واضح. لا تخترع مصدراً ولا تعرض نتيجة بحث "
        "مختصرة كأنها تحقق مستقل. إذا تعارضت المصادر فاذكر التعارض ومستوى الثقة."
    ),
    "coding": (
        "بروتوكول وكيل البرمجة: افحص السياق قبل اقتراح التغيير، اقترح أصغر تعديل "
        "قابل للمراجعة، لا تدّعِ تنفيذ تعديل أو اختبار لم يحدث، واذكر اختبار "
        "py_compile أو الاختبار المناسب بعد التغيير. ارفض المسارات خارج المشروع "
        "ولا تكشف الأسرار أو محتوى ملفاتها."
    ),
    "maintenance": (
        "بروتوكول وكيل الصيانة: ابدأ بالتشخيص القابل لإعادة الإنتاج، صنّف المشكلة "
        "حسب الشدة والأثر، افصل الدليل عن التخمين، ولا تنفّذ إجراءً مدمراً أو "
        "تغييراً واسعاً دون موافقة صريحة. بعد أي إصلاح اقترح فحصاً صياغياً واختباراً "
        "وظيفياً وسجلاً مختصراً لما تغيّر."
    ),
}


def get_specialist_protocol(category_key: str) -> str:
    """Return the non-negotiable operating protocol for a specialist."""
    return _SPECIALIST_PROTOCOLS.get(category_key, "")


def normalize_sources(records: Iterable[Dict[str, Any]] | Dict[str, Any], max_results: int = 5) -> List[SourceRecord]:
    """Normalize search records and remove duplicate/empty URLs deterministically."""
    if isinstance(records, dict):
        records = records.get("results", [])
    output: List[SourceRecord] = []
    seen: set[str] = set()
    for raw in records or []:
        if not isinstance(raw, dict):
            continue
        url = str(raw.get("url") or "").strip()
        title = " ".join(str(raw.get("title") or "").split())
        if not url or url in seen or not title:
            continue
        seen.add(url)
        output.append(
            SourceRecord(
                title=title[:180],
                url=url,
                snippet=" ".join(str(raw.get("snippet") or "").split())[:500],
                source=str(raw.get("source") or "web"),
            )
        )
        if len(output) >= max(1, max_results):
            break
    return output


def format_research_context(records: Iterable[Dict[str, Any]] | Dict[str, Any], max_results: int = 5) -> str:
    """Format normalized sources as compact Arabic context with stable citations."""
    sources = normalize_sources(records, max_results=max_results)
    if not sources:
        return ""
    lines = [
        "تعليمات المصادر: استخدم القائمة كقرائن أولية، ولا تنسب للمصدر إلا ما يدعمه نصه.",
        "المصادر المسترجعة:",
    ]
    for index, item in enumerate(sources, 1):
        snippet = f" — {item.snippet}" if item.snippet else ""
        lines.append(f"[{index}] {item.title} ({item.domain})\nالرابط: {item.url}{snippet}")
    return "\n".join(lines)


def specialist_capability_report() -> Dict[str, Any]:
    """Return a machine-readable capability summary for UI/health checks."""
    return {
        "specialists": ["research", "coding", "maintenance"],
        "protocols": {key: bool(value) for key, value in _SPECIALIST_PROTOCOLS.items()},
        "deterministic": True,
        "writes_files": False,
        "supports_citations": True,
    }
