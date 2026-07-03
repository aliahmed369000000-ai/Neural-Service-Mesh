"""
NSM Memory — ذاكرة المحادثة الذكية (v3 — ذاكرة هجينة بمعايير Mem0/Zep)
=======================================================================
تُضاف لـ NSMChat لتحقيق:
  1) فهم الضمائر (ركعاتها، كيف يعمل، ما أهميته)
  2) تراكم السياق عبر رسائل متعددة
  3) كشف تغيير الموضوع
  4) تذكر آخر 10 رسائل (بدلاً من 5) بأوزان تناقصية
  5) ذاكرة طويلة الأمد عبر SQLite تستمر بين الجلسات
  6) تتبع الكيانات والمفاهيم الرئيسية

جديد في v3 (مستوحى من Mem0 / Zep — راجع البحث المرفق):
  7) ذاكرة دلالية (Semantic / Vector Memory) عبر TF-IDF + تشابه جيب التمام
     — بديل خفيف ولا يحتاج مكتبات ثقيلة (لا نموذج embedding خارجي)
     ويُحسِّن الاسترجاع كثيراً مقارنة بمطابقة LIKE النصية القديمة.
  8) ذاكرة حقائق (Fact Memory) — استخلاص واستمرار حقائق ذرية عن المستخدم
     (تفضيلاته، مشروعه، معلوماته) مع دمج/تحديث الحقائق المتكررة بدل تكرارها،
     تماماً كما تفعل Mem0 (ADD / UPDATE بدل إعادة التخزين الأعمى).
  9) دالة build_memory_context() التي تبني كتلة سياق واحدة (حقائق + محادثات
     ذات صلة دلالياً) لحقنها في الاستعلام قبل إرساله لأي LLM.

⚠️ كل الواجهة القديمة (add / needs_context / enrich_query / clear /
context_summary / get_history / recall_past / get_ltm_stats) باقية
تماماً بنفس التوقيع — لا شيء يُكسر في nsm_chat.py أو nsm_chat_plus.py.
الإضافات الجديدة كلها اختيارية (additive-only).
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

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except Exception:  # sklearn غير متاح — نتراجع تلقائياً لبحث LIKE النصي
    _HAS_SKLEARN = False

try:
    from ai.vector_backend import get_vector_backend
    _HAS_VECTOR_BACKEND = True
except Exception:  # الوحدة غير متاحة (مكتبة ناقصة/مسار مختلف) — لا مشكلة
    _HAS_VECTOR_BACKEND = False

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

                CREATE TABLE IF NOT EXISTS facts (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   TEXT NOT NULL,
                    fact_key     TEXT NOT NULL,
                    fact_text    TEXT NOT NULL,
                    category     TEXT DEFAULT 'general',
                    importance   REAL DEFAULT 0.6,
                    access_count INTEGER DEFAULT 0,
                    created_at   REAL NOT NULL,
                    updated_at   REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_f_session ON facts(session_id, updated_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_f_key ON facts(session_id, fact_key);
            """)

    # ── حفظ محادثة ────────────────────────────────────────────────
    def save(self, session_id: str, turn: _Turn):
        try:
            with sqlite3.connect(self._path) as c:
                cur = c.execute(
                    "INSERT INTO turns(session_id,user_msg,bot_reply,topic,entities,ts) "
                    "VALUES(?,?,?,?,?,?)",
                    (session_id, turn.user[:400], turn.bot[:800],
                     turn.topic, json.dumps(turn.entities, ensure_ascii=False), turn.ts)
                )
                row_id = cur.lastrowid
        except Exception as e:
            logger.debug(f"LTM save: {e}")
            return

        # مرآة اختيارية إلى Qdrant (الطبقة الأولى في سلسلة الذاكرة الدلالية)
        # — best-effort تماماً، لا تؤثر أبداً على نجاح الحفظ الأساسي أعلاه.
        if _HAS_VECTOR_BACKEND:
            try:
                vb = get_vector_backend()
                if vb.available():
                    vb.upsert(
                        key=f"turn:{session_id}:{row_id}",
                        text=f"{turn.user} {turn.bot}",
                        payload={"kind": "turn", "session_id": session_id,
                                 "user": turn.user[:400], "bot": turn.bot[:800],
                                 "topic": turn.topic, "ts": turn.ts},
                    )
            except Exception as e:
                logger.debug(f"LTM Qdrant mirror skipped: {e}")

    # ── بحث نصي بسيط (الطريقة القديمة — تبقى كـ fallback) ──────────
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

    # ── بحث دلالي — سلسلة تراجع كاملة: Qdrant → TF-IDF → (search اللي فوق) ──
    def semantic_search(self, query: str, session_id: Optional[str] = None,
                         limit: int = 5, recent_cap: int = 500) -> List[dict]:
        """
        الذاكرة الدلالية القوية — بنفس فلسفة LLMFallback بالضبط:
          1) Qdrant Cloud (embeddings حقيقية عبر bge-m3) إن كان مُعدَّاً ومتاحاً
          2) TF-IDF محلي (يعمل دائماً بدون أي اعتمادية خارجية)
        أي طبقة تفشل → تتراجع فوراً وبصمت للطبقة التالية. recall_past() في
        ConversationMemory يستدعي هذه الدالة ثم search() النصية كطبقة أخيرة.
        """
        if not query.strip():
            return []

        # الطبقة 1: Qdrant (إن كان مُعدَّاً ومتصلاً فعلياً)
        if _HAS_VECTOR_BACKEND:
            try:
                vb = get_vector_backend()
                if vb.available():
                    flt = {"session_id": session_id, "kind": "turn"} if session_id else {"kind": "turn"}
                    hits = vb.search(query, k=limit, filter_payload=flt)
                    if hits:
                        return [
                            {"user": p.get("user", ""), "bot": p.get("bot", "")[:300],
                             "topic": p.get("topic", ""), "ts": p.get("ts", 0),
                             "score": round(score, 4), "source": "qdrant"}
                            for score, p in hits
                        ]
            except Exception as e:
                logger.debug(f"Qdrant semantic_search skipped, falling back: {e}")

        # الطبقة 2: TF-IDF محلي (الافتراضي الآمن دائماً)
        if not _HAS_SKLEARN:
            return []
        try:
            with sqlite3.connect(self._path) as c:
                if session_id:
                    rows = c.execute(
                        "SELECT user_msg,bot_reply,topic,ts FROM turns "
                        "WHERE session_id=? ORDER BY ts DESC LIMIT ?",
                        (session_id, recent_cap)
                    ).fetchall()
                else:
                    rows = c.execute(
                        "SELECT user_msg,bot_reply,topic,ts FROM turns "
                        "ORDER BY ts DESC LIMIT ?", (recent_cap,)
                    ).fetchall()
            if len(rows) < 2:
                return []

            docs = [f"{r[0]} {r[1]}" for r in rows]
            now = time.time()
            vec = TfidfVectorizer(max_features=4000)
            matrix = vec.fit_transform(docs + [query])
            sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()

            scored = []
            for (row, sim) in zip(rows, sims):
                if sim <= 0.05:
                    continue
                age_days = max(0.0, (now - row[3]) / 86400.0)
                recency = 1.0 / (1.0 + age_days / 14.0)   # نصف عمر تقريبي ~14 يوماً
                final_score = 0.75 * float(sim) + 0.25 * recency
                scored.append((final_score, row))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {"user": r[0], "bot": r[1][:300], "topic": r[2], "ts": r[3],
                 "score": round(s, 4), "source": "tfidf"}
                for s, r in scored[:limit]
            ]
        except Exception as e:
            logger.debug(f"LTM semantic_search error: {e}")
            return []

    # ── ذاكرة الحقائق (Mem0-style ADD/UPDATE) ───────────────────────
    def upsert_fact(self, session_id: str, fact_key: str, fact_text: str,
                     category: str = "general", importance: float = 0.6) -> None:
        now = time.time()
        try:
            with sqlite3.connect(self._path) as c:
                c.execute("""
                    INSERT INTO facts(session_id, fact_key, fact_text, category,
                                       importance, access_count, created_at, updated_at)
                    VALUES (?,?,?,?,?,0,?,?)
                    ON CONFLICT(session_id, fact_key) DO UPDATE SET
                        fact_text=excluded.fact_text,
                        category=excluded.category,
                        importance=MAX(facts.importance, excluded.importance),
                        updated_at=excluded.updated_at
                """, (session_id, fact_key, fact_text[:300], category,
                      importance, now, now))
        except Exception as e:
            logger.debug(f"LTM upsert_fact error: {e}")
            return

        # مرآة اختيارية إلى Qdrant — SQLite يبقى دائماً مصدر الحقيقة الأساسي
        # (all_facts تقرأ منه دوماً)، Qdrant يُستخدم فقط لتحسين الترتيب/البحث.
        if _HAS_VECTOR_BACKEND:
            try:
                vb = get_vector_backend()
                if vb.available():
                    vb.upsert(
                        key=f"fact:{session_id}:{fact_key}",
                        text=fact_text,
                        payload={"kind": "fact", "session_id": session_id,
                                 "fact_key": fact_key, "category": category,
                                 "importance": importance},
                    )
            except Exception as e:
                logger.debug(f"LTM fact Qdrant mirror skipped: {e}")

    def find_similar_fact_key(self, session_id: str, fact_text: str,
                               threshold: float = 0.72) -> Optional[str]:
        """يكتشف إن كانت هناك حقيقة مشابهة مخزَّنة مسبقاً لتحديثها بدل تكرارها."""
        facts = self.all_facts(session_id)
        if not facts:
            return None
        if _HAS_SKLEARN and len(facts) >= 1:
            try:
                texts = [f["fact_text"] for f in facts] + [fact_text]
                vec = TfidfVectorizer(max_features=2000)
                matrix = vec.fit_transform(texts)
                sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
                best_i = int(sims.argmax())
                if sims[best_i] >= threshold:
                    return facts[best_i]["fact_key"]
            except Exception:
                pass
        # fallback بسيط: تطابق أول 20 حرفاً بعد التطبيع
        norm = fact_text.strip().lower()[:20]
        for f in facts:
            if f["fact_text"].strip().lower()[:20] == norm:
                return f["fact_key"]
        return None

    def all_facts(self, session_id: str) -> List[dict]:
        try:
            with sqlite3.connect(self._path) as c:
                rows = c.execute(
                    "SELECT fact_key, fact_text, category, importance, "
                    "access_count, updated_at FROM facts WHERE session_id=? "
                    "ORDER BY updated_at DESC", (session_id,)
                ).fetchall()
            return [
                {"fact_key": r[0], "fact_text": r[1], "category": r[2],
                 "importance": r[3], "access_count": r[4], "updated_at": r[5]}
                for r in rows
            ]
        except Exception:
            return []

    def search_facts(self, query: str, session_id: str, limit: int = 5) -> List[dict]:
        facts = self.all_facts(session_id)
        if not facts:
            return []
        by_key = {f["fact_key"]: f for f in facts}

        # الطبقة 1: Qdrant — ترتيب دلالي حقيقي (SQLite يبقى مصدر البيانات)
        if _HAS_VECTOR_BACKEND and query.strip():
            try:
                vb = get_vector_backend()
                if vb.available():
                    hits = vb.search(
                        query, k=limit,
                        filter_payload={"session_id": session_id, "kind": "fact"},
                    )
                    ordered = [
                        by_key[p["fact_key"]] for _, p in hits
                        if p.get("fact_key") in by_key
                    ]
                    if ordered:
                        self._bump_access([f["fact_key"] for f in ordered], session_id)
                        return ordered[:limit]
            except Exception as e:
                logger.debug(f"Qdrant search_facts skipped, falling back: {e}")

        # الطبقة 2: TF-IDF محلي
        if _HAS_SKLEARN and query.strip() and len(facts) >= 1:
            try:
                texts = [f["fact_text"] for f in facts] + [query]
                vec = TfidfVectorizer(max_features=2000)
                matrix = vec.fit_transform(texts)
                sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
                scored = sorted(
                    zip(sims, facts), key=lambda x: x[0], reverse=True
                )
                out = [f for s, f in scored if s > 0.03][:limit]
                if out:
                    self._bump_access([f["fact_key"] for f in out], session_id)
                    return out
            except Exception:
                pass
        # fallback: أهم الحقائق بالأهمية والحداثة فقط
        top = sorted(facts, key=lambda f: (f["importance"], f["updated_at"]), reverse=True)[:limit]
        return top

    def _bump_access(self, keys: List[str], session_id: str) -> None:
        try:
            with sqlite3.connect(self._path) as c:
                c.executemany(
                    "UPDATE facts SET access_count = access_count + 1 "
                    "WHERE session_id=? AND fact_key=?",
                    [(session_id, k) for k in keys]
                )
        except Exception:
            pass

    def stats(self) -> dict:
        try:
            with sqlite3.connect(self._path) as c:
                total    = c.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
                sessions = c.execute("SELECT COUNT(DISTINCT session_id) FROM turns").fetchone()[0]
                topics   = c.execute(
                    "SELECT topic,COUNT(*) FROM turns GROUP BY topic ORDER BY COUNT(*) DESC LIMIT 5"
                ).fetchall()
                n_facts  = c.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            vector_active = False
            if _HAS_VECTOR_BACKEND:
                try:
                    vector_active = get_vector_backend().available()
                except Exception:
                    vector_active = False
            return {"total_turns": total, "sessions": sessions,
                    "top_topics": [{"topic": t, "count": n} for t, n in topics],
                    "total_facts": n_facts, "semantic_search": _HAS_SKLEARN,
                    "vector_backend_active": vector_active}
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

        # ✚ حقيقة واحدة عالية الصلة من ذاكرة الحقائق (Mem0-style) إن وُجدت
        try:
            top_facts = self._ltm.search_facts(user_msg, self._session_id, limit=1)
            if top_facts:
                ctx_parts.append(top_facts[0]["fact_text"][:60])
        except Exception:
            pass

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

    # ── استرجاع من الذاكرة الطويلة (الآن دلالي بالأولوية) ────────
    def recall_past(self, query: str) -> List[dict]:
        """
        يسترجع محادثات سابقة ذات صلة. يحاول أولاً البحث الدلالي
        (TF-IDF + تشابه جيب التمام) وإن لم يُعطِ نتائج (sklearn غير
        متاح أو بيانات قليلة) يتراجع تلقائياً لبحث الكلمات المفتاحية القديم.
        """
        sem = self._ltm.semantic_search(query, session_id=self._session_id, limit=3)
        if sem:
            return sem
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

    # ══════════════════════════════════════════════════════════════
    # ذاكرة الحقائق (Fact Memory) — بمعايير Mem0: ADD / UPDATE بدل التكرار
    # ══════════════════════════════════════════════════════════════
    def remember_fact(self, fact_text: str, category: str = "general",
                       importance: float = 0.6) -> None:
        """
        يخزّن حقيقة ذرية عن المستخدم/المشروع (مثال: 'المستخدم يبني Neural
        Service Mesh بلغة Python'). إن وُجدت حقيقة مشابهة دلالياً مسبقاً
        يُحدِّثها بدل تخزين نسخة مكررة — تماماً كسلوك Mem0.
        """
        fact_text = (fact_text or "").strip()
        if len(fact_text) < 4:
            return
        try:
            existing_key = self._ltm.find_similar_fact_key(self._session_id, fact_text)
            key = existing_key or f"fk_{abs(hash(fact_text[:40])) % 10_000_000}"
            self._ltm.upsert_fact(self._session_id, key, fact_text, category, importance)
        except Exception as e:
            logger.debug(f"remember_fact error: {e}")

    def get_facts(self, query: Optional[str] = None, limit: int = 5) -> List[dict]:
        """يعيد أهم الحقائق المخزَّنة، أو الأكثر صلة بـ query إن مُرِّر."""
        if query:
            return self._ltm.search_facts(query, self._session_id, limit)
        return self._ltm.all_facts(self._session_id)[:limit]

    def extract_and_remember_facts(self, user_msg: str, bot_reply: str,
                                    llm_fallback=None) -> int:
        """
        استخلاص تلقائي للحقائق من الحوار — بمعايير Mem0 (LLM يستخرج ما
        يستحق التذكر: تفضيلات، اسم، معلومات مشروع...). آمنة تماماً:
        أي فشل (لا مفتاح API، خطأ شبكة...) يُتجاهَل بصمت ولا يوقف المحادثة.

        Args:
            llm_fallback: كائن LLMFallback اختياري (من ai/llm_fallback.py).
                          إن لم يُمرَّر، تُستخدَم قواعد بسيطة فقط (بدون LLM).
        Returns:
            عدد الحقائق الجديدة/المحدَّثة.
        """
        count_before = len(self._ltm.all_facts(self._session_id))
        # بوابة اقتصادية: لا تستدعِ LLM إضافي إلا إذا بدت الرسالة تحتوي
        # فعلاً على معلومة شخصية محتملة — لتفادي مضاعفة استهلاك الحصة
        # المجانية (Cloudflare/Groq/Gemini) على كل رسالة عادية.
        _PERSONAL_HINTS = ("اسمي", "أنا ", "أعمل", "مشروعي", "أفضل", "عندي",
                            "بنيت", "طورت", "my name", "i work", "i am",
                            "i prefer", "my project")
        looks_personal = any(h in user_msg for h in _PERSONAL_HINTS)
        try:
            if llm_fallback is not None and looks_personal:
                extraction_prompt = (
                    "استخرج فقط الحقائق الثابتة المفيدة للتذكر مستقبلاً من "
                    "هذا الحوار (اسم المستخدم، تفضيلاته، معلومات عن مشروعه أو "
                    "عمله، حقائق شخصية صرّح بها). تجاهل الأسئلة العامة أو "
                    "الأجوبة الدينية/المعرفية التي لا تخص المستخدم شخصياً. "
                    "أعد النتيجة كمصفوفة JSON فقط من نصوص قصيرة (سطر واحد لكل "
                    "حقيقة)، أو [] إن لم توجد أي حقيقة تستحق التذكر. لا تكتب "
                    "أي شيء آخر غير مصفوفة JSON."
                )
                convo = f"المستخدم: {user_msg}\nالمساعد: {bot_reply[:300]}"
                result = llm_fallback.generate(
                    query=convo, history=[], system_prompt=extraction_prompt
                )
                raw = (result.text or "").strip()
                raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
                facts = json.loads(raw) if raw.startswith("[") else []
                for f in facts[:5]:
                    if isinstance(f, str) and 4 < len(f) < 300:
                        self.remember_fact(f, category="extracted", importance=0.65)
            else:
                # بدون LLM: قواعد بسيطة لاكتشاف تصريحات هوية/تفضيل شائعة
                for pat in (r"اسمي\s+([\w\u0600-\u06FF]{2,20})",
                            r"أنا\s+([\w\u0600-\u06FF]{2,20})\s*(?:،|,|\.|$)",
                            r"أفضل\s+(.{3,40})",
                            r"أعمل\s+على\s+(.{3,60})"):
                    m = re.search(pat, user_msg)
                    if m:
                        self.remember_fact(m.group(0), category="rule_based", importance=0.55)
        except Exception as e:
            logger.debug(f"extract_and_remember_facts error: {e}")
        return max(0, len(self._ltm.all_facts(self._session_id)) - count_before)

    # ══════════════════════════════════════════════════════════════
    # الذاكرة الموحَّدة — الدالة الرئيسية لحقن السياق في أي LLM
    # ══════════════════════════════════════════════════════════════
    def build_memory_context(self, query: str, max_facts: int = 4,
                              max_turns: int = 2) -> str:
        """
        يبني كتلة سياق واحدة تجمع (حقائق + محادثات سابقة ذات صلة دلالياً)
        جاهزة للحقن في system_prompt أو بداية الاستعلام قبل إرساله لأي LLM.
        هذه هي "الذاكرة القوية" المطلوبة — تعمل بنفس مبدأ Mem0/Zep:
        استرجاع انتقائي بدل حقن كامل السجل.
        """
        parts: List[str] = []
        try:
            facts = self.get_facts(query, limit=max_facts)
            if facts:
                facts_str = "؛ ".join(f["fact_text"] for f in facts)
                parts.append(f"معلومات محفوظة عن المستخدم: {facts_str}")
        except Exception:
            pass
        try:
            past = self.recall_past(query)[:max_turns]
            if past:
                past_str = " | ".join(f"س: {p['user'][:60]} ← ج: {p['bot'][:80]}" for p in past)
                parts.append(f"محادثات سابقة ذات صلة: {past_str}")
        except Exception:
            pass
        if self.current_topic:
            parts.append(f"الموضوع الحالي: {self.current_topic}")
        return "\n".join(parts)

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
