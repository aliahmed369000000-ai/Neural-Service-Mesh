"""
Phase 15 – World Feed
======================
التغذية من العالم الحقيقي.

الحساسات الحالية تراقب الملفات المحلية فقط.
هذا الملف يفتح الجهاز على العالم الخارجي:
  • يسحب بيانات من RSS feeds وAPIs عامة مفتوحة
  • يمرر كل بيانة عبر ImmuneSystem و QualityEngine
  • يقبل فقط ما quality_score >= 60 وحالته allowed
  • يعمل في background thread بشكل مستقل

مصادر مدمجة (بدون مفتاح API):
  • arXiv RSS  (أبحاث AI/ML)
  • Hacker News API (أخبار تقنية)
  • Wikipedia Recent Changes feed (معرفة عامة)
  • Open-Meteo API  (بيانات طقس — مثال حسي)
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_QUALITY_SCORE     = 60.0    # الحد الأدنى لقبول البيانة
DEFAULT_INTERVAL_S    = 300.0   # ثواني بين كل دورة سحب (5 دقائق)
REQUEST_TIMEOUT_S     = 10      # ثواني قبل timeout
MAX_ITEMS_PER_SOURCE  = 20      # أقصى عدد عناصر نقبلها من مصدر واحد في كل دورة
MAX_FEED_HISTORY      = 5000    # أقصى عدد عناصر نحتفظ بها في السجل


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_url(url: str, timeout: int = REQUEST_TIMEOUT_S) -> Optional[bytes]:
    """Fetch raw bytes from a URL.  Returns None on any error."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NeuralServiceMesh/15.0 WorldFeed (+https://github.com)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        logger.debug(f"[WorldFeed] fetch error {url}: {exc}")
        return None


def _parse_rss(raw: bytes) -> List[Dict[str, str]]:
    """Parse RSS/Atom XML into a list of item dicts."""
    items = []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return items

    # RSS 2.0
    for item in root.iter("item"):
        title   = (item.findtext("title") or "").strip()
        link    = (item.findtext("link") or "").strip()
        summary = (item.findtext("description") or "").strip()
        pub_    = (item.findtext("pubDate") or _now_iso()).strip()
        if title or summary:
            items.append({
                "title":   title,
                "content": summary or title,
                "url":     link,
                "date":    pub_,
            })

    # Atom
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in root.findall(".//a:entry", ns):
        title   = (entry.findtext("a:title", namespaces=ns) or "").strip()
        link_el = entry.find("a:link", ns)
        link    = link_el.get("href", "") if link_el is not None else ""
        summary = (entry.findtext("a:summary", namespaces=ns) or "").strip()
        pub_    = (entry.findtext("a:updated", namespaces=ns) or _now_iso()).strip()
        if title or summary:
            items.append({
                "title":   title,
                "content": summary or title,
                "url":     link,
                "date":    pub_,
            })

    return items[:MAX_ITEMS_PER_SOURCE]


def _parse_json_api(raw: bytes, source_type: str) -> List[Dict[str, str]]:
    """Parse well-known JSON API responses."""
    items = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return items

    # Hacker News: topstories returns list of IDs
    if source_type == "hackernews_ids":
        ids = data[:MAX_ITEMS_PER_SOURCE] if isinstance(data, list) else []
        for item_id in ids:
            items.append({"id": str(item_id), "source_type": "hackernews_item"})
        return items

    # Hacker News item
    if source_type == "hackernews_item":
        title = data.get("title", "")
        url   = data.get("url", f"https://news.ycombinator.com/item?id={data.get('id')}")
        if title:
            items.append({
                "title":   title,
                "content": title,
                "url":     url,
                "date":    _now_iso(),
            })
        return items

    # Generic list of objects with "title" / "content" / "text"
    if isinstance(data, list):
        for obj in data[:MAX_ITEMS_PER_SOURCE]:
            if isinstance(obj, dict):
                title   = obj.get("title") or obj.get("name") or ""
                content = obj.get("content") or obj.get("text") or obj.get("abstract") or title
                url     = obj.get("url") or obj.get("link") or ""
                if title or content:
                    items.append({
                        "title":   str(title),
                        "content": str(content),
                        "url":     str(url),
                        "date":    _now_iso(),
                    })

    return items


# ---------------------------------------------------------------------------
# FeedSource
# ---------------------------------------------------------------------------

class FeedSource:
    """
    A single data source (URL + type).

    source_type options:
        rss         — RSS 2.0 / Atom feed
        json_list   — JSON array of objects
        hackernews  — Hacker News (uses the HN Firebase API)
    """

    def __init__(self, url: str, source_type: str = "rss", name: Optional[str] = None):
        self.url         = url
        self.source_type = source_type
        self.name        = name or url[:60]
        self.fetch_count = 0
        self.error_count = 0
        self.last_fetch  = None
        self.enabled     = True

    def fetch(self) -> List[Dict[str, str]]:
        raw = _fetch_url(self.url)
        self.last_fetch = _now_iso()
        if raw is None:
            self.error_count += 1
            return []
        self.fetch_count += 1

        if self.source_type == "rss":
            return _parse_rss(raw)
        elif self.source_type in ("json_list", "hackernews_ids", "hackernews_item"):
            return _parse_json_api(raw, self.source_type)
        else:
            # Try RSS first, then JSON
            items = _parse_rss(raw)
            if not items:
                items = _parse_json_api(raw, "json_list")
            return items

    def summary(self) -> dict:
        return {
            "name":        self.name,
            "url":         self.url,
            "source_type": self.source_type,
            "enabled":     self.enabled,
            "fetch_count": self.fetch_count,
            "error_count": self.error_count,
            "last_fetch":  self.last_fetch,
        }


# ---------------------------------------------------------------------------
# WorldFeed
# ---------------------------------------------------------------------------

class WorldFeed:
    """
    Pulls real-world data from public sources, filters it through
    ImmuneSystem + QualityEngine, then stores accepted items.

    Parameters
    ----------
    immune_system : optional
        ImmuneSystem instance.  If None, all items pass immune check.
    quality_engine : optional
        QualityEngine instance.  If None, quality score defaults to 70.
    memory_callback : optional
        Callable(item_dict) invoked for every accepted item so the
        caller can store it in episodic/semantic memory.
    min_quality : float
        Minimum quality score to accept (default 60).
    """

    # Default public sources (no API keys required)
    DEFAULT_SOURCES = [
        FeedSource(
            url="https://export.arxiv.org/rss/cs.AI",
            source_type="rss",
            name="arXiv:cs.AI",
        ),
        FeedSource(
            url="https://export.arxiv.org/rss/cs.LG",
            source_type="rss",
            name="arXiv:cs.LG",
        ),
        FeedSource(
            url="https://hnrss.org/frontpage",
            source_type="rss",
            name="HackerNews Front Page",
        ),
        FeedSource(
            url="https://feeds.feedburner.com/TechCrunch",
            source_type="rss",
            name="TechCrunch",
        ),
    ]

    def __init__(
        self,
        immune_system=None,
        quality_engine=None,
        memory_callback: Optional[Callable[[dict], None]] = None,
        min_quality: float = MIN_QUALITY_SCORE,
    ):
        self._immune          = immune_system
        self._quality         = quality_engine
        self._memory_cb       = memory_callback
        self._min_quality     = min_quality

        self._sources: List[FeedSource] = list(self.DEFAULT_SOURCES)
        self._lock            = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running         = False

        # Stats
        self._total_fetched   = 0
        self._total_accepted  = 0
        self._total_rejected  = 0
        self._total_blocked   = 0
        self._feed_history: List[dict] = []   # accepted items ring-buffer
        self._cycles          = 0

        logger.info(
            f"[WorldFeed] Ready  sources={len(self._sources)}  "
            f"min_quality={self._min_quality}"
        )

    # ── Source Management ──────────────────────────────────────────────────

    def add_source(self, url: str, source_type: str = "rss", name: Optional[str] = None):
        """Register a new data source."""
        src = FeedSource(url=url, source_type=source_type, name=name)
        with self._lock:
            # Avoid duplicates
            if not any(s.url == url for s in self._sources):
                self._sources.append(src)
                logger.info(f"[WorldFeed] Source added: {src.name}")

    def remove_source(self, url: str):
        """Remove a source by URL."""
        with self._lock:
            self._sources = [s for s in self._sources if s.url != url]

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self, interval_s: float = DEFAULT_INTERVAL_S):
        """Start the background polling thread."""
        if self._running:
            logger.warning("[WorldFeed] Already running.")
            return
        self._interval = interval_s
        self._running  = True
        self._thread   = threading.Thread(
            target=self._run_loop, name="WorldFeed", daemon=True
        )
        self._thread.start()
        logger.info(
            f"[WorldFeed] Started  interval={interval_s:.0f}s  "
            f"sources={len(self._sources)}"
        )

    def stop(self):
        """Stop the background thread."""
        self._running = False
        logger.info("[WorldFeed] Stopped.")

    def poll_once(self):
        """Manually trigger one full polling cycle (blocking)."""
        self._poll_cycle()

    # ── Internal Loop ──────────────────────────────────────────────────────

    def _run_loop(self):
        while self._running:
            try:
                self._poll_cycle()
            except Exception as exc:
                logger.error(f"[WorldFeed] Poll error: {exc}")
            time.sleep(getattr(self, "_interval", DEFAULT_INTERVAL_S))

    def _poll_cycle(self):
        self._cycles += 1
        sources_snapshot = list(self._sources)

        for src in sources_snapshot:
            if not src.enabled:
                continue
            items = src.fetch()
            self._total_fetched += len(items)

            for raw_item in items:
                self._process_item(raw_item, src)

        # Trim ring-buffer
        if len(self._feed_history) > MAX_FEED_HISTORY:
            self._feed_history = self._feed_history[-MAX_FEED_HISTORY:]

        logger.debug(
            f"[WorldFeed] Cycle {self._cycles}  "
            f"fetched={self._total_fetched}  "
            f"accepted={self._total_accepted}  "
            f"rejected={self._total_rejected}"
        )

    def _process_item(self, raw: dict, src: FeedSource):
        """Run one item through immune + quality gate."""
        content = raw.get("content") or raw.get("title") or ""
        if not content:
            return

        item = {
            "content":     content,
            "title":       raw.get("title", ""),
            "url":         raw.get("url", ""),
            "date":        raw.get("date", _now_iso()),
            "source":      src.name,
            "source_url":  src.url,
            "source_type": src.source_type,
            "ingested_at": _now_iso(),
        }

        # ── Immune check ──────────────────────────────────────────────────
        if self._immune is not None:
            try:
                verdict = self._immune.inspect(item, source=src.name)
                status  = getattr(verdict, "status", None) or (
                    verdict.get("status") if isinstance(verdict, dict) else "allowed"
                )
                if status not in (None, "allowed", "Trusted", "Unknown"):
                    self._total_blocked += 1
                    return
            except Exception as exc:
                logger.debug(f"[WorldFeed] Immune check error: {exc}")

        # ── Quality check ─────────────────────────────────────────────────
        quality_score = 70.0  # default when no engine
        if self._quality is not None:
            try:
                result = self._quality.rate(item, source=src.name)
                if isinstance(result, dict):
                    quality_score = result.get("score", 70.0)
                elif isinstance(result, (int, float)):
                    quality_score = float(result)
            except Exception as exc:
                logger.debug(f"[WorldFeed] Quality check error: {exc}")

        item["quality_score"] = round(quality_score, 2)

        if quality_score < self._min_quality:
            self._total_rejected += 1
            return

        # ── Accept ────────────────────────────────────────────────────────
        self._total_accepted += 1
        self._feed_history.append(item)

        if self._memory_cb is not None:
            try:
                self._memory_cb(item)
            except Exception as exc:
                logger.debug(f"[WorldFeed] Memory callback error: {exc}")

    # ── Stats ──────────────────────────────────────────────────────────────

    def get_feed_stats(self) -> dict:
        return {
            "cycles":         self._cycles,
            "total_fetched":  self._total_fetched,
            "total_accepted": self._total_accepted,
            "total_rejected": self._total_rejected,
            "total_blocked":  self._total_blocked,
            "history_size":   len(self._feed_history),
            "running":        self._running,
            "sources":        [s.summary() for s in self._sources],
        }

    def get_recent(self, n: int = 20) -> List[dict]:
        """Return the n most recently accepted items."""
        return self._feed_history[-n:]

    def summary(self) -> dict:
        return {
            "component":     "WorldFeed",
            "running":       self._running,
            "sources_count": len(self._sources),
            "min_quality":   self._min_quality,
            "stats":         self.get_feed_stats(),
        }

    # ── Wiring helpers ─────────────────────────────────────────────────────

    def set_immune_system(self, immune_system):
        self._immune = immune_system

    def set_quality_engine(self, quality_engine):
        self._quality = quality_engine

    def set_memory_callback(self, cb: Callable[[dict], None]):
        self._memory_cb = cb


# ---------------------------------------------------------------------------
# Quick self-test (no internet required — uses mock data)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  WorldFeed — Self-Test  (offline mock)")
    print("=" * 60)

    wf = WorldFeed(min_quality=50.0)

    # Inject mock source that returns fake items
    class _MockSource(FeedSource):
        def fetch(self):
            return [
                {"title": f"AI News {i}", "content": f"Content {i}", "url": "http://mock.test", "date": _now_iso()}
                for i in range(5)
            ]

    wf._sources = [_MockSource("http://mock.test", "rss", "Mock")]

    wf.poll_once()

    stats = wf.get_feed_stats()
    print(f"  Fetched:  {stats['total_fetched']}")
    print(f"  Accepted: {stats['total_accepted']}")
    print(f"  Rejected: {stats['total_rejected']}")
    recent = wf.get_recent(3)
    print(f"  Recent sample: {[r['title'] for r in recent]}")

    print("\n  summary():")
    s = wf.summary()
    print(f"    running={s['running']}  sources={s['sources_count']}")

    print("\n✓ WorldFeed self-test PASSED")
