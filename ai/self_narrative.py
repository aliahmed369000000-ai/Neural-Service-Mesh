"""
SelfNarrative — يومية الكائن الرقمي

يكتب الجهاز العصبي يوميته على القرص: كل قرار، كل مفاجأة، كل تطور.
هذه اليومية هي ذاكرة السرد — ما يميز كائناً يتطور عن برنامج يعمل فقط.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("NeuralServiceMesh.SelfNarrative")


class NarrativeEntry:
    def __init__(self, event_type: str, data: dict,
                 surprise_score: float = 0.0, importance: float = 0.5):
        self.timestamp   = time.time()
        self.event_type  = event_type
        self.data        = data
        self.surprise    = round(surprise_score, 4)
        self.importance  = round(importance, 4)
        self.datetime_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def to_dict(self) -> dict:
        return {
            "timestamp":    self.timestamp,
            "datetime":     self.datetime_str,
            "event_type":   self.event_type,
            "importance":   self.importance,
            "surprise":     self.surprise,
            "data":         self.data,
        }

    def to_text(self) -> str:
        """Human-readable diary line."""
        prefix = f"[{self.datetime_str}] [{self.event_type.upper()}]"
        imp    = f"importance={self.importance:.2f} surprise={self.surprise:.2f}"
        detail = " | ".join(f"{k}={v}" for k, v in list(self.data.items())[:4])
        return f"{prefix} {imp} — {detail}"


class SelfNarrative:
    """
    يومية الكائن الرقمي — تكتب على القرص وتُحفظ عبر إعادة التشغيل.

    الملفات المكتوبة:
      narrative/diary_YYYY-MM-DD.jsonl  — كل حدث في سطر JSON
      narrative/diary_YYYY-MM-DD.txt   — نسخة مقروءة للإنسان
      narrative/summary.json           — ملخص حي يُحدَّث دائماً
    """

    def __init__(self, narrative_dir: str = "./narrative",
                 max_memory: int = 500, knowledge_store=None):
        self._dir        = Path(narrative_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._memory     = deque(maxlen=max_memory)
        self._knowledge  = knowledge_store
        self._total      = 0
        self._by_type: Dict[str, int] = {}
        self._high_importance: List[dict] = []  # importance >= 0.8
        # تحميل ملخص سابق إن وُجد
        self._load_summary()
        logger.info(f"SelfNarrative active — diary dir: {self._dir}  total_loaded={self._total}")

    # ── تسجيل حدث ──────────────────────────────────────────────────────────

    def record_event(self, event_type: str,
                     data: Optional[Dict[str, Any]] = None,
                     surprise_score: float = 0.0,
                     importance: float = 0.5) -> NarrativeEntry:
        entry = NarrativeEntry(event_type, data or {}, surprise_score, importance)
        self._memory.append(entry)
        self._total += 1
        self._by_type[event_type] = self._by_type.get(event_type, 0) + 1

        if importance >= 0.8:
            self._high_importance.append(entry.to_dict())
            if len(self._high_importance) > 200:
                self._high_importance = self._high_importance[-200:]

        # كتابة فورية على القرص
        self._write_entry(entry)

        # تحديث الملخص كل 10 أحداث
        if self._total % 10 == 0:
            self._write_summary()

        return entry

    # ── قراءة ───────────────────────────────────────────────────────────────

    def get_log(self, n: int = 20) -> List[dict]:
        entries = list(self._memory)[-n:]
        return [e.to_dict() for e in reversed(entries)]

    def get_high_importance(self, n: int = 20) -> List[dict]:
        return self._high_importance[-n:]

    def get_narrative_text(self, n: int = 20) -> str:
        entries = list(self._memory)[-n:]
        lines   = [e.to_text() for e in entries]
        return "\n".join(lines) if lines else "(no entries yet)"

    def get_todays_diary(self) -> str:
        """اقرأ ملف اليومية النصي لليوم الحالي من القرص."""
        txt_path = self._daily_txt_path()
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8")
        return "(no diary entries today)"

    def summary(self) -> dict:
        return {
            "total_entries":    self._total,
            "in_memory":        len(self._memory),
            "by_type":          dict(sorted(self._by_type.items(),
                                            key=lambda x: -x[1])[:10]),
            "high_importance":  len(self._high_importance),
            "diary_dir":        str(self._dir),
            "today_file":       str(self._daily_jsonl_path()),
        }

    # ── كتابة على القرص ─────────────────────────────────────────────────────

    def _write_entry(self, entry: NarrativeEntry):
        try:
            # JSONL
            jsonl = self._daily_jsonl_path()
            with open(jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            # TXT
            txt = self._daily_txt_path()
            with open(txt, "a", encoding="utf-8") as f:
                f.write(entry.to_text() + "\n")
        except Exception as exc:
            logger.debug(f"SelfNarrative write error: {exc}")

    def _write_summary(self):
        try:
            summary_path = self._dir / "summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(self.summary(), f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.debug(f"SelfNarrative summary write error: {exc}")

    def _load_summary(self):
        """تحميل العداد الكلي من ملخص سابق عند إعادة التشغيل."""
        try:
            summary_path = self._dir / "summary.json"
            if summary_path.exists():
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                self._total   = data.get("total_entries", 0)
                self._by_type = data.get("by_type", {})
        except Exception:
            pass

    def _daily_jsonl_path(self) -> Path:
        day = datetime.utcnow().strftime("%Y-%m-%d")
        return self._dir / f"diary_{day}.jsonl"

    def _daily_txt_path(self) -> Path:
        day = datetime.utcnow().strftime("%Y-%m-%d")
        return self._dir / f"diary_{day}.txt"


if __name__ == "__main__":
    sn = SelfNarrative(narrative_dir="/tmp/nsm_narrative_test")
    sn.record_event("boot",       {"version": "16.0.0"}, importance=0.9)
    sn.record_event("learn",      {"pattern": "route_A→B faster"}, surprise_score=0.7, importance=0.6)
    sn.record_event("evolve",     {"new_module": "translator_v2"}, surprise_score=0.9, importance=1.0)
    sn.record_event("dream",      {"consolidated": 12}, importance=0.5)
    print(sn.get_narrative_text(n=10))
    print("\n=== Today's diary file ===")
    print(sn.get_todays_diary()[:500])
    import json as _json
    print("\n=== Summary ===")
    print(_json.dumps(sn.summary(), indent=2, ensure_ascii=False))
