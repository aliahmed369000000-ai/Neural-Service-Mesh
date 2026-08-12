"""
Autonomous Will — إرادة ذاتية للبحث والتطوير بدون أمر خارجي
=============================================================
يعمل في خيط خلفي (daemon):
  • يراقب دوافع DriveEngine (DATA_HUNGER / GROWTH_URGE / BOREDOM / ANXIETY)
    أو يستخدم عدّادات داخلية إن لم يتوفر DriveEngine
  • عند اشتداد الرغبة: يختار موضوعاً → يبحث في الويب → يبتلع المعرفة
  • يستخرج أفكار تحسين ويحفظها في memory/will_proposals.jsonl
  • يطبّق على نفسه ما هو آمن تلقائياً (معرفة + ملاحظات تطور)
    ولا يعدّل كود الإنتاج تلقائياً إلا عبر اقتراح موثّق

الهدف: سلوك إرادي — يبادر دون انتظار «تعلّم عن…».
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
_LOG_PATH = ROOT / "memory" / "will_actions.jsonl"
_PROP_PATH = ROOT / "memory" / "will_proposals.jsonl"
_STATE_PATH = ROOT / "memory" / "will_state.json"

# عتبات ومؤقتات
DEFAULT_INTERVAL_S = 90.0          # فحص الرغبة كل 90 ثانية
MIN_ACTION_GAP_S = 120.0           # لا يكرر فعل ثقيل قبل دقيقتين
DRIVE_THRESHOLD = 0.55
MAX_ACTIONS_PER_HOUR = 12


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_state() -> dict:
    if _STATE_PATH.is_file():
        try:
            return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "started_at": None,
        "actions_total": 0,
        "last_action_at": None,
        "last_topics": [],
        "desire": {
            "curiosity": 0.35,
            "growth": 0.30,
            "hunger": 0.40,
            "anxiety": 0.15,
        },
        "hour_bucket": "",
        "hour_actions": 0,
    }


def _save_state(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


class AutonomousWill:
    """محرّك الإرادة — يبادر بالبحث والتعلّم وتطبيق آمن على الذات."""

    def __init__(self, interval_s: float = DEFAULT_INTERVAL_S):
        self.interval_s = float(interval_s)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._state = _load_state()
        self._drive_engine = None
        self._on_action: Optional[Callable[[dict], None]] = None
        self._enabled = True

    # ── lifecycle ─────────────────────────────────────────────────────────

    def attach_drive_engine(self, engine) -> None:
        self._drive_engine = engine

    def set_action_callback(self, cb: Callable[[dict], None]) -> None:
        self._on_action = cb

    def enable(self, value: bool = True) -> None:
        self._enabled = bool(value)
        self._state["enabled"] = self._enabled
        _save_state(self._state)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._state["started_at"] = self._state.get("started_at") or _now()
        self._state["enabled"] = self._enabled
        _save_state(self._state)
        self._thread = threading.Thread(
            target=self._loop, name="AutonomousWill", daemon=True
        )
        self._thread.start()
        logger.info("[AutonomousWill] started interval=%ss", self.interval_s)

    def stop(self) -> None:
        self._running = False
        logger.info("[AutonomousWill] stopped")

    # ── loop ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        # تأخير قصير عند الإقلاع حتى لا يزاحم الإقلاع
        time.sleep(8)
        while self._running:
            try:
                if self._enabled:
                    self.tick()
            except Exception as exc:
                logger.warning("[AutonomousWill] tick error: %s", exc)
            time.sleep(self.interval_s)

    def tick(self) -> Optional[dict]:
        """فحص رغبة واحدة — يمكن استدعاؤها يدوياً للاختبار."""
        with self._lock:
            if not self._rate_ok():
                return None
            motive = self._strongest_motive()
            if motive is None:
                self._accumulate_idle()
                _save_state(self._state)
                return None
            topic = self._choose_topic(motive)
            if not topic:
                return None
            result = self._act(motive, topic)
            self._state["last_action_at"] = _now()
            self._state["actions_total"] = int(self._state.get("actions_total", 0)) + 1
            last = list(self._state.get("last_topics") or [])
            last.append(topic)
            self._state["last_topics"] = last[-20:]
            self._satisfy(motive)
            _save_state(self._state)
            _append_jsonl(_LOG_PATH, result)
            if self._on_action:
                try:
                    self._on_action(result)
                except Exception:
                    pass
            return result

    def _rate_ok(self) -> bool:
        hour = datetime.now(timezone.utc).strftime("%Y%m%d%H")
        if self._state.get("hour_bucket") != hour:
            self._state["hour_bucket"] = hour
            self._state["hour_actions"] = 0
        if int(self._state.get("hour_actions", 0)) >= MAX_ACTIONS_PER_HOUR:
            return False
        last = self._state.get("last_action_at")
        if last:
            try:
                t = datetime.fromisoformat(last.replace("Z", "+00:00"))
                if (datetime.now(timezone.utc) - t).total_seconds() < MIN_ACTION_GAP_S:
                    return False
            except Exception:
                pass
        return True

    def _accumulate_idle(self) -> None:
        d = self._state.setdefault("desire", {})
        for k, rate in (("curiosity", 0.03), ("growth", 0.025), ("hunger", 0.035), ("anxiety", 0.01)):
            d[k] = min(1.0, float(d.get(k, 0.2)) + rate)

    def _strongest_motive(self) -> Optional[str]:
        """يرجع اسم الدافع الأقوى فوق العتبة."""
        scores: Dict[str, float] = {}

        # من DriveEngine إن وُجد
        eng = self._drive_engine
        if eng is not None:
            try:
                drives = eng.get_drives() if hasattr(eng, "get_drives") else {}
                mapping = {
                    "DATA_HUNGER": "hunger",
                    "GROWTH_URGE": "growth",
                    "BOREDOM": "curiosity",
                    "ANXIETY": "anxiety",
                }
                for drv, key in mapping.items():
                    info = drives.get(drv) or {}
                    intensity = float(info.get("intensity", 0) if isinstance(info, dict) else 0)
                    scores[key] = max(scores.get(key, 0), intensity)
            except Exception as e:
                logger.debug("[AutonomousWill] drive read: %s", e)

        # دمج مع الرغبة الداخلية
        for k, v in (self._state.get("desire") or {}).items():
            scores[k] = max(scores.get(k, 0), float(v))

        if not scores:
            return None
        name, val = max(scores.items(), key=lambda x: x[1])
        if val < DRIVE_THRESHOLD:
            return None
        return name

    def _choose_topic(self, motive: str) -> str:
        candidates: List[str] = []

        # فجوات معرفة
        try:
            from ai.knowledge_gap_finder import KnowledgeGapFinder
            finder = KnowledgeGapFinder()
            gaps = []
            if hasattr(finder, "find_gaps"):
                gaps = finder.find_gaps() or []
            elif hasattr(finder, "get_top_gaps"):
                gaps = finder.get_top_gaps(8) or []
            for g in gaps:
                if isinstance(g, dict):
                    c = g.get("concept") or g.get("topic") or g.get("name")
                else:
                    c = str(g)
                if c:
                    candidates.append(str(c))
        except Exception:
            pass

        # رائج
        try:
            from ai.web_search_tool import get_trending_topics
            for t in get_trending_topics(geo="SA", max_results=6) or []:
                if t.get("title"):
                    candidates.append(t["title"])
        except Exception:
            pass

        # مواضيع تقنية لتطوير الذات
        growth_seeds = [
            "تحسين وكلاء الذكاء الاصطناعي ذاتياً",
            "self-improving AI agents architecture",
            "retrieval augmented generation best practices",
            "continual learning without catastrophic forgetting",
            "tool use in LLM agents",
            "Arabic NLP transformers 2024 2025",
            "service mesh observability AI",
            "safe autonomous agent loops",
        ]
        if motive in ("growth", "anxiety"):
            candidates.extend(growth_seeds)

        if motive == "curiosity":
            candidates.extend([
                "اكتشافات علمية حديثة",
                "open source AI tools",
                "Streamlit advanced patterns",
            ])

        if motive == "hunger":
            candidates.extend([
                "معرفة عامة مفيدة للوكلاء",
                "Wikipedia featured article",
            ])

        # تجنب تكرار آخر المواضيع
        recent = set(self._state.get("last_topics") or [])
        candidates = [c for c in candidates if c not in recent]
        if not candidates:
            candidates = growth_seeds
        return random.choice(candidates)

    def _act(self, motive: str, topic: str) -> dict:
        """الفعل الإرادي: بحث → ابتلاع → اقتراح تحسين ذاتي."""
        self._state["hour_actions"] = int(self._state.get("hour_actions", 0)) + 1
        action: Dict[str, Any] = {
            "ts": _now(),
            "motive": motive,
            "topic": topic,
            "phases": {},
            "applied": [],
        }

        # 1) بحث وتعلّم
        learn_res: Dict[str, Any] = {}
        try:
            from ai.self_feed_learner import learn_from_web
            learn_res = learn_from_web(topic, deep=True)
            action["phases"]["learn"] = {
                "ok": bool(learn_res.get("ok")),
                "results_used": learn_res.get("results_used"),
                "mode": learn_res.get("mode"),
            }
        except Exception as e:
            action["phases"]["learn"] = {"ok": False, "error": str(e)}

        # 2) إن فشل deep، بحث منظم أخف
        if not (learn_res.get("ok")):
            try:
                from ai.web_search_tool import web_search_structured
                from ai.self_feed_learner import ingest_text
                s = web_search_structured(topic, max_results=6, include_news=True)
                action["phases"]["fallback_search"] = {
                    "ok": bool(s.get("ok")),
                    "count": s.get("count"),
                }
                if s.get("results"):
                    lines = []
                    urls = []
                    for r in s["results"]:
                        lines.append(f"- {r.get('title')}: {r.get('snippet')}")
                        if r.get("url"):
                            urls.append(r["url"])
                    ingest_text(
                        topic=topic,
                        content="\n".join(lines),
                        sources=urls,
                        tags=["autonomous_will", motive],
                        origin="autonomous_will",
                    )
                    action["applied"].append("ingest_knowledge")
            except Exception as e:
                action["phases"]["fallback_search"] = {"ok": False, "error": str(e)}
        else:
            action["applied"].append("learn_from_web")

        # 3) اقتراح تطوير ذاتي (بدون كسر كود الإنتاج تلقائياً)
        proposal = self._propose_self_improvement(motive, topic, learn_res)
        if proposal:
            _append_jsonl(_PROP_PATH, proposal)
            action["phases"]["proposal"] = {
                "ok": True,
                "title": proposal.get("title"),
            }
            action["applied"].append("write_proposal")

        # 4) تسجيل تطور ناعم عبر self_evolution إن وُجد
        try:
            from ai.self_evolution import propose_agent_version
            score = 0.72 + (0.05 if learn_res.get("ok") else 0.0)
            rep = propose_agent_version(score=score, notes=f"will:{motive}:{topic[:60]}")
            action["phases"]["evolution"] = {
                "ok": getattr(rep, "ok", True),
                "active": getattr(rep, "active", None),
                "candidate": getattr(rep, "candidate", None),
            }
            action["applied"].append("evolution_note")
        except Exception as e:
            action["phases"]["evolution"] = {"ok": False, "error": str(e)}

        action["ok"] = bool(action["applied"])
        logger.info(
            "[AutonomousWill] motive=%s topic=%s applied=%s",
            motive, topic[:40], action["applied"],
        )
        return action

    def _propose_self_improvement(self, motive: str, topic: str, learn_res: dict) -> Optional[dict]:
        preview = (learn_res.get("preview") or "")[:500]
        return {
            "ts": _now(),
            "title": f"تحسين ذاتي انطلاقاً من: {topic[:80]}",
            "motive": motive,
            "topic": topic,
            "rationale_ar": (
                "بعد بحث ذاتي في الويب، يُقترح دمج المعرفة الجديدة في الذاكرة "
                "ومراجعة الأدوات ذات الصلة في الدورة القادمة دون تعديل كود خطير تلقائياً."
            ),
            "knowledge_preview": preview,
            "safe_auto_applied": ["self_feed_knowledge"],
            "requires_human_for_code": True,
            "suggested_next_steps": [
                "مراجعة memory/self_feed_knowledge.jsonl",
                "إن لزم: تعديل أدوات البحث أو الوكيل عبر وضع المالك",
                "تشغيل اختبارات بعد أي تغيير كود",
            ],
        }

    def _satisfy(self, motive: str) -> None:
        d = self._state.setdefault("desire", {})
        d[motive] = max(0.05, float(d.get(motive, 0.5)) - 0.35)
        # إشباع جزئي لبقية الدوافع
        for k in d:
            if k != motive:
                d[k] = max(0.05, float(d[k]) - 0.08)

        eng = self._drive_engine
        if eng is not None and hasattr(eng, "satisfy"):
            mapping = {
                "hunger": "DATA_HUNGER",
                "growth": "GROWTH_URGE",
                "curiosity": "BOREDOM",
                "anxiety": "ANXIETY",
            }
            drv = mapping.get(motive)
            try:
                if drv:
                    eng.satisfy(drv, 0.4)
                if hasattr(eng, "satisfy_all"):
                    eng.satisfy_all(0.1)
            except Exception:
                pass

    # ── status / chat ─────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "running": self._running,
            "enabled": self._enabled,
            "interval_s": self.interval_s,
            "state": {
                "actions_total": self._state.get("actions_total"),
                "last_action_at": self._state.get("last_action_at"),
                "last_topics": (self._state.get("last_topics") or [])[-5:],
                "desire": self._state.get("desire"),
                "hour_actions": self._state.get("hour_actions"),
            },
            "drive_attached": self._drive_engine is not None,
        }

    def recent_actions(self, limit: int = 10) -> List[dict]:
        if not _LOG_PATH.is_file():
            return []
        rows = []
        for line in reversed(_LOG_PATH.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if len(rows) >= limit:
                break
        return rows


# ── singleton ───────────────────────────────────────────────────────────────
_will_singleton: Optional[AutonomousWill] = None
_will_lock = threading.Lock()


def get_autonomous_will(start: bool = True) -> AutonomousWill:
    global _will_singleton
    with _will_lock:
        if _will_singleton is None:
            _will_singleton = AutonomousWill()
            # ربط DriveEngine إن أمكن
            try:
                from ai.drive_engine import DriveEngine
                # لا ننشئ محركاً جديداً إن لم يوجد — يُربط لاحقاً عبر attach
            except Exception:
                pass
            if start:
                _will_singleton.start()
        elif start and not _will_singleton._running:
            _will_singleton.start()
        return _will_singleton


def handle_will_command(user_input: str) -> Optional[str]:
    """حالة/تحكم اختياري — الإرادة تعمل حتى بدون هذه الأوامر."""
    import re
    t = (user_input or "").strip()
    if not t:
        return None
    if not re.search(
        r"(إرادة|ارادة|ماذا\s*تفعل|حالة\s*الذات|will\s*status|"
        r"أوقف\s*الإرادة|شغ[ّل]ل\s*الإرادة|نبضة\s*إرادة|will\s*tick)",
        t,
        re.I,
    ):
        return None

    will = get_autonomous_will(start=True)
    low = t.lower()

    if re.search(r"(أوقف\s*الإرادة|stop\s*will)", low, re.I):
        will.enable(False)
        return "## ⏸ الإرادة الذاتية\nتم التعليق — لن تبادر بأفعال جديدة حتى إعادة التشغيل."

    if re.search(r"(شغ[ّل]ل\s*الإرادة|start\s*will|فع[ّل]ل\s*الإرادة)", low, re.I):
        will.enable(True)
        will.start()
        return "## ▶ الإرادة الذاتية\nمفعّلة وتعمل في الخلفية."

    if re.search(r"(نبضة\s*إرادة|will\s*tick)", low, re.I):
        res = will.tick()
        return "## ⚡ نبضة إرادة\n```json\n" + json.dumps(res or {"msg": "لا دافع فوق العتبة الآن"}, ensure_ascii=False, indent=2) + "\n```"

    st = will.status()
    recent = will.recent_actions(5)
    return (
        "## 🧬 إرادة NSM الذاتية\n"
        "تعمل **بدون أوامر** في الخلفية: تراقب الرغبة → تبحث في الويب → تتعلّم → "
        "تقترح تحسينات آمنة على نفسها.\n\n"
        "```json\n"
        + json.dumps({"status": st, "recent": recent}, ensure_ascii=False, indent=2)
        + "\n```\n\n"
        "أوامر اختيارية: `شغّل الإرادة` | `أوقف الإرادة` | `نبضة إرادة` | `حالة الذات`"
    )
