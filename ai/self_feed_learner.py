"""
NSM Self-Feed Learner — تغذية ذاتية وتعلّم من الويب
===================================================
يسمح للوكلاء بـ:
  1) البحث العميق في الويب
  2) حفظ المعرفة المكتسبة في memory/self_feed_knowledge.jsonl
  3) تسجيل الدورات عبر LearningOrchestrator إن وُجد
  4) دورة تعلّم ذاتي: فجوة → بحث → ابتلاع → تلخيص

لا يخترع معرفة: كل سجل مرتبط بمصادر URL حقيقية من أدوات البحث.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
_STORE = ROOT / "memory" / "self_feed_knowledge.jsonl"
_LOCK = threading.Lock()
_MAX_ENTRY_CHARS = 4000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_store() -> None:
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    if not _STORE.exists():
        _STORE.write_text("", encoding="utf-8")


def ingest_text(
    topic: str,
    content: str,
    sources: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    origin: str = "manual",
) -> Dict[str, Any]:
    """حفظ قطعة معرفة مع مصادرها."""
    topic = (topic or "").strip()
    content = (content or "").strip()
    if not topic or not content:
        return {"ok": False, "msg": "مطلوب topic و content"}
    entry = {
        "ts": _now(),
        "topic": topic[:200],
        "content": content[:_MAX_ENTRY_CHARS],
        "sources": [s for s in (sources or []) if s][:12],
        "tags": [t for t in (tags or []) if t][:12],
        "origin": origin,
    }
    with _LOCK:
        _ensure_store()
        with _STORE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # محاولة ربط LearningOrchestrator
    orch_ok = False
    try:
        from ai.learning_orchestrator import get_orchestrator
        orch = get_orchestrator()
        orch.record_turn(
            query=f"[self-feed] {topic}",
            response=content[:1500],
            domain="web_knowledge",
            source="self_feed",
        )
        orch_ok = True
    except Exception:
        pass
    return {
        "ok": True,
        "topic": topic,
        "chars": len(entry["content"]),
        "sources_count": len(entry["sources"]),
        "orchestrator": orch_ok,
        "store": str(_STORE.relative_to(ROOT)),
    }


def learn_from_web(topic: str, deep: bool = True, max_results: int = 8) -> Dict[str, Any]:
    """بحث (عادي أو عميق) ثم ابتلاع النتائج كمعرفة."""
    topic = (topic or "").strip()
    if not topic:
        return {"ok": False, "msg": "مطلوب موضوع"}

    try:
        from ai.web_search_tool import deep_research, web_search_structured
    except Exception as e:
        return {"ok": False, "msg": f"تعذّر تحميل أداة البحث: {e}"}

    if deep:
        raw = deep_research(topic)
        results = raw.get("results") or []
        mode = "deep_research"
    else:
        raw = web_search_structured(topic, max_results=max_results, include_news=True)
        results = raw.get("results") or []
        mode = "web_search"

    if not results:
        return {
            "ok": False,
            "msg": raw.get("msg") or "لا نتائج للابتلاع",
            "mode": mode,
        }

    lines = []
    sources = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or ""
        snip = r.get("snippet") or ""
        url = r.get("url") or ""
        src = r.get("source") or ""
        lines.append(f"{i}. {title}\n{snip}\n({src}) {url}")
        if url:
            sources.append(url)

    content = f"ملخص معرفة مجمّع عن: {topic}\n\n" + "\n\n".join(lines)
    ingested = ingest_text(
        topic=topic,
        content=content,
        sources=sources,
        tags=["web", mode],
        origin=mode,
    )
    return {
        "ok": bool(ingested.get("ok")),
        "mode": mode,
        "topic": topic,
        "results_used": len(results),
        "ingest": ingested,
        "preview": content[:600],
    }


def self_learn_cycle(seed_topics: Optional[List[str]] = None, limit: int = 3) -> Dict[str, Any]:
    """
    دورة تعلّم ذاتي:
      - إن وُجدت فجوات من knowledge_gap_finder استخدمها
      - وإلا مواضيع رائجة أو seed_topics
      - لكل موضوع: learn_from_web
    """
    topics: List[str] = []
    if seed_topics:
        topics.extend([t.strip() for t in seed_topics if t and t.strip()])

    # فجوات المعرفة
    if len(topics) < limit:
        try:
            from ai.knowledge_gap_finder import KnowledgeGapFinder
            finder = KnowledgeGapFinder()
            gaps = finder.find_gaps() if hasattr(finder, "find_gaps") else []
            if not gaps and hasattr(finder, "get_top_gaps"):
                gaps = finder.get_top_gaps(limit)
            for g in gaps or []:
                if isinstance(g, dict):
                    c = g.get("concept") or g.get("topic") or g.get("name")
                else:
                    c = str(g)
                if c and c not in topics:
                    topics.append(c)
                if len(topics) >= limit:
                    break
        except Exception:
            pass

    # رائج
    if len(topics) < limit:
        try:
            from ai.web_search_tool import get_trending_topics
            for t in get_trending_topics(geo="SA", max_results=limit):
                title = t.get("title") or ""
                if title and title not in topics:
                    topics.append(title)
                if len(topics) >= limit:
                    break
        except Exception:
            pass

    if not topics:
        topics = ["الذكاء الاصطناعي", "التعلّم الذاتي للوكلاء", "شبكات الخدمات العصبية"]

    topics = topics[:limit]
    runs = []
    for topic in topics:
        runs.append(learn_from_web(topic, deep=True))

    ok_n = sum(1 for r in runs if r.get("ok"))
    return {
        "ok": ok_n > 0,
        "topics": topics,
        "learned": ok_n,
        "total": len(runs),
        "runs": runs,
    }


def list_knowledge(limit: int = 20, query: str = "") -> Dict[str, Any]:
    """قراءة آخر المعرفة المخزّنة."""
    _ensure_store()
    rows: List[Dict[str, Any]] = []
    q = (query or "").strip().lower()
    try:
        lines = _STORE.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return {"ok": False, "msg": str(e), "items": []}
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if q and q not in (obj.get("topic") or "").lower() and q not in (obj.get("content") or "").lower():
            continue
        rows.append({
            "ts": obj.get("ts"),
            "topic": obj.get("topic"),
            "sources_count": len(obj.get("sources") or []),
            "origin": obj.get("origin"),
            "preview": (obj.get("content") or "")[:180],
        })
        if len(rows) >= limit:
            break
    return {"ok": True, "count": len(rows), "items": rows, "store": str(_STORE.relative_to(ROOT))}


def handle_learn_command(user_input: str) -> Optional[str]:
    """أوامر محادثة للتعلّم الذاتي."""
    t = (user_input or "").strip()
    if not t:
        return None
    if not re.search(
        r"(تعل[ّم]م|غذ[ّي]ي|ابتلع|دورة\s*تعل[ّم]م|self\s*learn|learn\s*from|"
        r"deep\s*research|بحث\s*عميق|معرفة\s*مخز[ّن]نة|ما\s*تعل[ّم]مت)",
        t,
        re.I,
    ):
        return None

    low = t.lower()

    # قائمة المعرفة
    if re.search(r"(معرفة\s*مخز|ما\s*تعل[ّم]مت|list\s*knowledge)", low, re.I):
        q = re.sub(r".*?(معرفة\s*مخز[ّن]*ة|ما\s*تعل[ّم]مت|list\s*knowledge)\s*", "", t, flags=re.I).strip()
        res = list_knowledge(query=q)
        return "## 🧠 المعرفة المخزّنة\n```json\n" + json.dumps(res, ensure_ascii=False, indent=2) + "\n```"

    # دورة ذاتية
    if re.search(r"(دورة\s*تعل[ّم]م|self\s*learn|غذ[ّي]ي\s*نفسك)", low, re.I):
        res = self_learn_cycle(limit=3)
        return "## 🔄 دورة التعلّم الذاتي\n```json\n" + json.dumps(res, ensure_ascii=False, indent=2) + "\n```"

    # بحث عميق فقط
    m = re.match(r"^(بحث\s*عميق|deep\s*research)\s+(.+)$", t, re.I)
    if m:
        topic = m.group(2).strip()
        try:
            from ai.web_search_tool import deep_research
            res = deep_research(topic)
            return "## 🔬 بحث عميق\n```json\n" + json.dumps(
                {k: res[k] for k in res if k != "results"} | {
                    "results_preview": [
                        {"title": r.get("title"), "source": r.get("source"), "url": r.get("url")}
                        for r in (res.get("results") or [])[:8]
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n```"
        except Exception as e:
            return f"❌ deep_research: {e}"

    # تعلّم عن موضوع / غذِّ من الويب
    m = re.match(
        r"^(تعل[ّم]م\s*عن|تعلم\s*عن|غذ[ّي]ي|ابتلع|learn\s*from\s*web|learn)\s+(.+)$",
        t,
        re.I,
    )
    if m:
        topic = m.group(2).strip()
        res = learn_from_web(topic, deep=True)
        return "## 📥 تعلّم من الويب\n```json\n" + json.dumps(res, ensure_ascii=False, indent=2) + "\n```"

    return (
        "## 🧠 أوامر التعلّم الذاتي\n"
        "- `تعلّم عن <موضوع>` — بحث عميق + حفظ المعرفة\n"
        "- `بحث عميق <موضوع>` — بحث متعدد الزوايا بدون حفظ إجباري\n"
        "- `دورة تعلّم` / `غذِّ نفسك` — فجوات + رائج → تعلّم\n"
        "- `معرفة مخزّنة [فلتر]` — عرض ما تم ابتلاعه\n"
    )
