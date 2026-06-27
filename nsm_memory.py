"""
NSM Memory — ذاكرة المحادثة الذكية (v2 محسَّن)
=================================================
تُضاف لـ NSMChat لتحقيق:
  1) فهم الضمائر (ركعاتها، كيف يعمل، ما أهميته)
  2) تراكم السياق عبر رسائل متعددة
  3) كشف تغيير الموضوع
  4) تذكر آخر 10 رسائل (بدلاً من 5) بأوزان تناقصية
  5) ذاكرة طويلة الأمد عبر SQLite تستمر بين الجلسات
  6) تتبع الكيانات والمفاهيم الرئيسية
"""
from __future__ import annotations
import json
import logging
import re
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("NSMMemory")

# ══════════════════════════════════════════════════════════════════
# كلمات تدل على السياق (ضمائر وإشارات)
# ══════════════════════════════════════════════════════════════════
_CONTEXT_SIGNALS = [
    "ها", "هم", "هن", "ه", "ك", "كم",
    "وكم", "وكيف", "ومتى", "وأين", "ولماذا", "وما", "وهل",
    "وكذلك", "وأيضاً", "وأيضا", "وهو", "وهي", "وهما",
    "أيضاً", "أيضا", "كذلك", "بالإضافة", "فضلاً",
    "علاوة", "ومن", "وعن", "وفي", "وعلى",
    "also", "too", "additionally", "furthermore",
    "and how", "and why", "and when",
]

_PRONOUN_PATTERNS = [
    r'(ركعات|صلوات|أوقات|فوائد|أنواع|أحكام|أسباب|أهمية)(ها|هم|ه)',
    r'^(وكم|وكيف|ومتى|وأين|ولماذا|وما|وهل|وهو|وهي)\b',
    r'^(و[أا]يض[اً]|وكذلك|وبالإضافة)\b',
    r'(يعمل|تعمل|تُؤدى|يُستخدم|يُحسب)(ها|ه|هم)?\??$',
]

_ARABIC_STOPWORDS = {
    "ما","هو","هي","من","في","على","عن","إلى","هل","كيف","متى","أين",
    "لماذا","ماذا","كم","أي","أو","لكن","مع","بعد","قبل","عند","لقد",
    "قد","لم","لن","لا","نعم","هذا","هذه","ذلك","تلك","أنا","أنت","نحن",
}

# خريطة الموضوعات
_TOPIC_MAP: Dict[str, List[str]] = {
    "قرآن":       ["قرآن","آية","سورة","تلاوة","حفظ","تجويد","مصحف"],
    "حديث":       ["حديث","سنة","نبي","رسول","صحيح","بخاري","مسلم"],
    "صلاة":       ["صلاة","ركعة","وضوء","قبلة","أذان","سجود","فجر","ظهر","عصر","مغرب","عشاء"],
    "صيام":       ["صيام","صوم","رمضان","إفطار","سحور","اعتكاف"],
    "زكاة":       ["زكاة","نصاب","فقير","مسكين","مال"],
    "حج":         ["حج","عمرة","كعبة","مكة","منى","عرفة"],
    "فقه":        ["حلال","حرام","مكروه","مباح","واجب","فرض"],
    "عقيدة":      ["توحيد","إيمان","كفر","شرك","إسلام","مسلم"],
    "لغة_عربية":  ["نحو","صرف","جملة","فعل","اسم","حرف","إعراب"],
    "برمجة":      ["python","code","برمجة","كود","function","class","api","github"],
    "ذكاء_اصطناعي":["ذكاء","نموذج","تعلم","neural","llm","model","ai"],
}

_WORD_TO_TOPIC: Dict[str, str] = {}
for _t, _words in _TOPIC_MAP.items():
    for _w in _words:
        _WORD_TO_TOPIC[_w.lower()] = _t

# أوزان تناقص الرسائل (الأحدث أثقل)
_DECAY_WEIGHTS = [1.0, 0.95, 0.88, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]

# ══════════════════════════════════════════════════════════════════
# Dataclass للدور الواحد
# ══════════════════════════════════════════════════════════════════
@dataclass
class _Turn:
    user:      str
    bot:       str
    topic:     str
    entities:  List[str]
    ts:        float = field(default_factory=time.time)
    importance: float = 1.0


# ══════════════════════════════════════════════════════════════════
# Long-Term Memory — SQLite
# ══════════════════════════════════════════════════════════════════
class _LongTermStore:
    def __init__(self, db_path: str = "memory/nsm_context.db"):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self):
        with sqlite3.connect(self._path) as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS turns (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_msg   TEXT NOT NULL,
                    bot_reply  TEXT NOT NULL,
                    topic      TEXT DEFAULT '',
                    entities   TEXT DEFAULT '[]',
                    ts         REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_s ON turns(session_id, ts DESC);
                CREATE INDEX IF NOT EXISTS idx_t ON turns(topic, ts DESC);
            """)

    def save(self, session_id: str, turn: _Turn):
        try:
            with sqlite3.connect(self._path) as c:
                c.execute(
                    "INSERT INTO turns(session_id,user_msg,bot_reply,topic,entities,ts) "
                    "VALUES(?,?,?,?,?,?)",
                    (session_id, turn.user[:400], turn.bot[:800],
                     turn.topic, json.dumps(turn.entities, ensure_ascii=False), turn.ts)
                )
        except Exception as e:
            logger.debug(f"LTM save: {e}")

    def search(self, keywords: List[str], limit: int = 3) -> List[dict]:
        results = []
        try:
            with sqlite3.connect(self._path) as c:
                for kw in keywords[:3]:
                    rows = c.execute(
                        "SELECT user_msg,bot_reply,topic,ts FROM turns "
                        "WHERE user_msg LIKE ? OR bot_reply LIKE ? "
                        "ORDER BY ts DESC LIMIT ?",
                        (f"%{kw}%", f"%{kw}%", limit)
                    ).fetchall()
                    for r in rows:
                        results.append({"user": r[0], "bot": r[1][:200], "topic": r[2], "ts": r[3]})
        except Exception:
            pass
        seen, unique = set(), []
        for r in sorted(results, key=lambda x: x["ts"], reverse=True):
            k = r["user"][:40]
            if k not in seen:
                seen.add(k); unique.append(r)
        return unique[:limit]

    def stats(self) -> dict:
        try:
            with sqlite3.connect(self._path) as c:
                total    = c.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
                sessions = c.execute("SELECT COUNT(DISTINCT session_id) FROM turns").fetchone()[0]
                topics   = c.execute(
                    "SELECT topic,COUNT(*) FROM turns GROUP BY topic ORDER BY COUNT(*) DESC LIMIT 5"
                ).fetchall()
            return {"total_turns": total, "sessions": sessions,
                    "top_topics": [{"topic": t, "count": n} for t, n in topics]}
        except Exception:
            return {}


# ══════════════════════════════════════════════════════════════════
# ConversationMemory — الكلاس الرئيسي (نفس الاسم للتوافق)
# ══════════════════════════════════════════════════════════════════
class ConversationMemory:
    """
    ذاكرة المحادثة المُطوَّرة.

    الجديد مقارنةً بالإصدار القديم:
      ✅ نافذة 10 رسائل بدلاً من 5 (بأوزان تناقصية)
      ✅ ذاكرة طويلة الأمد SQLite تستمر بين الجلسات
      ✅ كشف تغيير الموضوع بـ Jaccard similarity
      ✅ تتبع الكيانات أفضل (آخر 15 كيان فريد)
      ✅ بناء سياق ذكي للـ LLM (الأهم لا الأحدث فقط)
      ✅ توافق كامل مع الواجهة القديمة
    """

    WINDOW = 10

    def __init__(self, session_id: str = "", db_path: str = "memory/nsm_context.db"):
        self._session_id        = session_id or f"s_{int(time.time())}"
        self._history           : deque = deque(maxlen=self.WINDOW)
        self._ltm               = _LongTermStore(db_path)
        self.current_topic      : str = ""
        self.current_topic_text : str = ""
        self._entities          : List[str] = []
        self._prev_keywords     : List[str] = []

    # ── إضافة رسالة ──────────────────────────────────────────────
    def add(self, user_msg: str, bot_reply: str, topic: str = ""):
        keywords = self._keywords(user_msg)
        detected = topic or self._detect_topic(keywords)

        # كشف تغيير الموضوع
        if self._prev_keywords and detected != self.current_topic:
            old = set(self._prev_keywords)
            new = set(keywords)
            union = old | new
            jaccard = len(old & new) / len(union) if union else 1.0
            if jaccard < 0.25:
                logger.debug(f"موضوع تغيّر: {self.current_topic}→{detected}")

        self.current_topic      = detected
        self.current_topic_text = user_msg
        self._prev_keywords     = keywords
        self._update_entities(keywords)

        importance = self._importance(user_msg, bot_reply)
        turn = _Turn(user=user_msg, bot=bot_reply, topic=detected,
                     entities=list(self._entities), importance=importance)
        self._history.append(turn)
        self._ltm.save(self._session_id, turn)

    # ── كشف السياق ───────────────────────────────────────────────
    def needs_context(self, user_msg: str) -> bool:
        text = user_msg.strip()
        if len(text) < 15:
            return bool(self._history)
        if text.startswith('و') and len(text) > 2:
            return True
        tl = text.lower()
        for sig in _CONTEXT_SIGNALS:
            if sig in tl:
                return True
        for pat in _PRONOUN_PATTERNS:
            if re.search(pat, text):
                return True
        return False

    # ── بناء الاستعلام المُغنى ───────────────────────────────────
    def enrich_query(self, user_msg: str) -> str:
        if not self.needs_context(user_msg) or not self._history:
            return user_msg

        # اختر أنسب رسالتين سابقتين
        kw_set = set(self._keywords(user_msg))
        scored = []
        for i, turn in enumerate(reversed(list(self._history))):
            turn_kw = set(self._keywords(turn.user))
            overlap = len(kw_set & turn_kw) / max(len(kw_set | turn_kw), 1)
            decay   = _DECAY_WEIGHTS[i] if i < len(_DECAY_WEIGHTS) else 0.1
            score   = overlap * 0.5 + decay * 0.3 + turn.importance * 0.2
            scored.append((score, turn))
        scored.sort(reverse=True)

        ctx_parts = []
        if self.current_topic:
            ctx_parts.append(self.current_topic_text[:30])
        if self._entities:
            ctx_parts.extend(self._entities[-3:])
        if scored:
            ctx_parts.append(scored[0][1].user[:40])

        context_str = " ".join(dict.fromkeys(ctx_parts))
        return f"{context_str} {user_msg}"

    # ── واجهة LLM ────────────────────────────────────────────────
    def get_llm_history(self, max_pairs: int = 4) -> List[Tuple[str, str]]:
        """يعيد أهم الرسائل للـ LLM (بالأهمية، لا فقط الأحدث)"""
        turns = list(self._history)
        if len(turns) <= max_pairs:
            return [(t.user, t.bot) for t in turns]
        # فرز بالأهمية ثم إعادة الترتيب الزمني
        indexed = sorted(enumerate(turns), key=lambda x: x[1].importance, reverse=True)
        top = sorted([i for i, _ in indexed[:max_pairs]])
        return [(turns[i].user, turns[i].bot) for i in top]

    # ── ملخص السياق ──────────────────────────────────────────────
    def context_summary(self) -> str:
        if not self._history:
            return "لا يوجد سياق سابق"
        last = self._history[-1]
        entities_str = ", ".join(self._entities[-3:]) if self._entities else "—"
        return (
            f"الموضوع: {self.current_topic} | "
            f"الكيانات: {entities_str} | "
            f"آخر سؤال: {last.user[:50]}"
        )

    # ── استرجاع من الذاكرة الطويلة ──────────────────────────────
    def recall_past(self, query: str) -> List[dict]:
        kw = self._keywords(query)
        return self._ltm.search(kw)

    def get_ltm_stats(self) -> dict:
        return self._ltm.stats()

    # ── مسح ─────────────────────────────────────────────────────
    def clear(self):
        self._history.clear()
        self.current_topic      = ""
        self.current_topic_text = ""
        self._entities          = []
        self._prev_keywords     = []

    # ── تاريخ المحادثة ───────────────────────────────────────────
    def get_history(self) -> List[Tuple[str, str]]:
        return [(t.user, t.bot) for t in self._history]

    # ── Private helpers ──────────────────────────────────────────
    def _keywords(self, text: str) -> List[str]:
        words = re.findall(r'[\u0600-\u06FF]{2,}|[a-zA-Z]{3,}', text.lower())
        return [w for w in words if w not in _ARABIC_STOPWORDS and len(w) >= 2][:15]

    def _detect_topic(self, keywords: List[str]) -> str:
        scores: Dict[str, int] = {}
        for kw in keywords:
            t = _WORD_TO_TOPIC.get(kw)
            if t:
                scores[t] = scores.get(t, 0) + 1
        return max(scores, key=scores.get) if scores else "عام"

    def _update_entities(self, keywords: List[str]):
        for kw in keywords:
            if kw not in self._entities:
                self._entities.append(kw)
        self._entities = self._entities[-15:]

    def _importance(self, user_msg: str, bot_reply: str) -> float:
        score = 0.5
        if len(user_msg) > 50:   score += 0.2
        if len(bot_reply) > 200: score += 0.1
        kw_cnt = len(self._keywords(user_msg))
        score += min(0.2, kw_cnt * 0.03)
        return min(1.0, score)

    def __len__(self):   return len(self._history)
    def __bool__(self):  return True
    def __repr__(self):
        return f"<ConversationMemory turns={len(self._history)} topic={self.current_topic}>"
