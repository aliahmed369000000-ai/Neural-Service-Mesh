"""
SOCIAL AGENT — الوكيل الاجتماعي الموحد
==========================================
خدمة خلفية واحدة تشغّل عدة منصات تواصل بالتوازي: تنشر + ترد تلقائياً +
تراقب كلمات/إشارات مفتاحية — وتستخدم نفس محرك الشخصية الموحّدة (NSM_PERSONA_PROMPT)
عبر OpenRouter الذي يستعمله الوكيل الأساسي في app.py (نفس الشخصية والسياق، ليس نظاماً منفصلاً).

يعمل كـ singleton على مستوى العملية (process-wide) في خيط (thread) واحد
يستطلع كل منصة مُفعَّلة كل SocialAgentManager.poll_interval ثانية، بشكل
مستقل عن أي جلسة Streamlit مفتوحة — طالما أن عملية الخادم تعمل.

لا بيانات مزيّفة: أي منصة بلا بيانات اعتماد كافية تُعرض كـ "غير مُهيّأة"
في الحالة، ولا تُستَبدل نتائجها بمحتوى ملفّق.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta as _timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .social_platforms import ALL_ADAPTERS, PLATFORM_LABELS_AR, SocialItem, NotConfiguredError

try:
    from .godmode import NSM_PERSONA_PROMPT as GODMODE_SYSTEM_PROMPT
except Exception:  # pragma: no cover
    GODMODE_SYSTEM_PROMPT = "You are a helpful, direct assistant."

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_REPLY_MODEL = "google/gemini-2.5-flash"
DEFAULT_POLL_INTERVAL = 90  # ثانية — تكفي لتوفير حصة الـ API المجانية

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "memory" / "social_agent.db"
DB_PATH.parent.mkdir(exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# قاعدة البيانات — إعدادات + سجل أحداث + معرفات مُعالَجة (لمنع التكرار)
# ═════════════════════════════════════════════════════════════════════════════

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    # WAL mode: يتيح قراءات متزامنة مع الكتابة دون قفل الملف كاملاً
    conn.execute("PRAGMA journal_mode=WAL")
    # busy_timeout: انتظر 5 ثوانٍ بدل رفع OperationalError مباشرةً
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS social_config (
            key TEXT PRIMARY KEY, value TEXT
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS social_seen (
            platform TEXT, external_id TEXT, seen_at TEXT,
            PRIMARY KEY (platform, external_id)
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS social_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT, event_type TEXT, author TEXT,
            content TEXT, reply_content TEXT, created_at TEXT, ok INTEGER,
            sentiment TEXT, sentiment_score REAL
        )""")
    # ترحيل: قواعد بيانات قديمة أُنشئت قبل إضافة عمودَي المشاعر
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(social_events)")}
    if "sentiment" not in existing_cols:
        conn.execute("ALTER TABLE social_events ADD COLUMN sentiment TEXT")
    if "sentiment_score" not in existing_cols:
        conn.execute("ALTER TABLE social_events ADD COLUMN sentiment_score REAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS scheduled_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platforms TEXT, text TEXT, scheduled_at TEXT,
            status TEXT DEFAULT 'pending', created_at TEXT,
            published_at TEXT, result TEXT
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS social_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT, author TEXT, role TEXT,
            content TEXT, created_at TEXT
        )""")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conv_platform_author "
        "ON social_conversations(platform, author, id)"
    )
    # نقاط استرجاع (checkpoints) لكل منصة ضمن عملية نشر واحدة — تسمح
    # باستئناف نشر مُجدوَل توقّف عمليته منتصف الطريق (تعطّل الخادم مثلاً)
    # دون إعادة النشر على منصات نجحت فعلاً (راجع publish_to/_process_due_scheduled).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS publish_checkpoints (
            post_id INTEGER, platform TEXT, ok INTEGER,
            result TEXT, updated_at TEXT,
            PRIMARY KEY (post_id, platform)
        )""")
    conn.commit()
    return conn


def _get_checkpoint_results(post_id: int) -> Dict[str, tuple]:
    """يعيد {platform: (ok, result)} لكل منصة سُجِّلت سابقاً ضمن هذا post_id."""
    with _db() as c:
        rows = c.execute(
            "SELECT platform, ok, result FROM publish_checkpoints WHERE post_id=?",
            (post_id,),
        ).fetchall()
    return {p: (bool(ok), result) for p, ok, result in rows}


def _save_checkpoint(post_id: int, platform: str, ok: bool, result: str) -> None:
    """يحفظ نتيجة منصة واحدة فور اكتمالها — نقطة استرجاع (checkpoint) لكل
    خطوة بدل الانتظار حتى تنتهي كل المنصات، بحيث لو تعطّلت العملية بعد
    هذه النقطة، لا يُعاد تنفيذ ما سبق حفظه بنجاح."""
    with _db() as c:
        c.execute(
            "INSERT OR REPLACE INTO publish_checkpoints (post_id,platform,ok,result,updated_at) "
            "VALUES (?,?,?,?,?)",
            (post_id, platform, 1 if ok else 0, result, datetime.now(timezone.utc).isoformat()),
        )


def _clear_checkpoints(post_id: int) -> None:
    with _db() as c:
        c.execute("DELETE FROM publish_checkpoints WHERE post_id=?", (post_id,))


def get_config(key: str, default=None):
    with _db() as c:
        row = c.execute("SELECT value FROM social_config WHERE key=?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row[0])
    except (TypeError, ValueError):
        return row[0]


def set_config(key: str, value) -> None:
    with _db() as c:
        c.execute("INSERT OR REPLACE INTO social_config VALUES (?,?)", (key, json.dumps(value)))


def _is_seen(platform: str, ext_id: str) -> bool:
    with _db() as c:
        row = c.execute(
            "SELECT 1 FROM social_seen WHERE platform=? AND external_id=?", (platform, ext_id)
        ).fetchone()
    return row is not None


def _mark_seen(platform: str, ext_id: str) -> None:
    with _db() as c:
        c.execute("INSERT OR IGNORE INTO social_seen VALUES (?,?,?)",
                  (platform, ext_id, datetime.now(timezone.utc).isoformat()))


def _seen_ids_for(platform: str, limit: int = 500) -> set:
    with _db() as c:
        rows = c.execute(
            "SELECT external_id FROM social_seen WHERE platform=? ORDER BY seen_at DESC LIMIT ?",
            (platform, limit),
        ).fetchall()
    return {r[0] for r in rows}


def log_event(platform: str, event_type: str, author: str, content: str,
              reply_content: str = "", ok: bool = True,
              sentiment: Optional[str] = None, sentiment_score: Optional[float] = None) -> None:
    with _db() as c:
        c.execute(
            "INSERT INTO social_events "
            "(platform,event_type,author,content,reply_content,created_at,ok,sentiment,sentiment_score) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (platform, event_type, author, content, reply_content,
             datetime.now(timezone.utc).isoformat(), 1 if ok else 0, sentiment, sentiment_score),
        )


def get_recent_events(limit: int = 40) -> List[tuple]:
    with _db() as c:
        return c.execute(
            "SELECT platform,event_type,author,content,reply_content,created_at,ok,"
            "sentiment,sentiment_score FROM social_events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


def get_event_counts() -> Dict[str, int]:
    with _db() as c:
        rows = c.execute(
            "SELECT event_type, COUNT(*) FROM social_events GROUP BY event_type"
        ).fetchall()
    return dict(rows)


# ═════════════════════════════════════════════════════════════════════════════
# 📅 الجدولة — تقويم محتوى (نشر مؤجَّل عبر منصة أو أكثر)
# ═════════════════════════════════════════════════════════════════════════════

def schedule_post(platforms: List[str], text: str, scheduled_at: str) -> int:
    """يضيف منشوراً مجدولاً. scheduled_at بصيغة ISO 8601 (UTC يُفضَّل).
    يعيد معرّف السجل (id) لاستخدامه لاحقاً في cancel_scheduled."""
    with _db() as c:
        cur = c.execute(
            "INSERT INTO scheduled_posts (platforms,text,scheduled_at,status,created_at) "
            "VALUES (?,?,?,?,?)",
            (json.dumps(platforms), text, scheduled_at, "pending",
             datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def get_scheduled(status: Optional[str] = None) -> List[tuple]:
    """يعيد (id, platforms(list), text, scheduled_at, status, published_at, result)."""
    with _db() as c:
        if status:
            rows = c.execute(
                "SELECT id,platforms,text,scheduled_at,status,published_at,result "
                "FROM scheduled_posts WHERE status=? ORDER BY scheduled_at ASC", (status,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT id,platforms,text,scheduled_at,status,published_at,result "
                "FROM scheduled_posts ORDER BY scheduled_at ASC"
            ).fetchall()
    return [
        (rid, json.loads(plats or "[]"), text, sched_at, status, pub_at, result)
        for rid, plats, text, sched_at, status, pub_at, result in rows
    ]


def cancel_scheduled(post_id: int) -> bool:
    with _db() as c:
        cur = c.execute(
            "UPDATE scheduled_posts SET status='cancelled' WHERE id=? AND status='pending'",
            (post_id,),
        )
        return cur.rowcount > 0


def _due_scheduled_posts() -> List[tuple]:
    now_iso = datetime.now(timezone.utc).isoformat()
    with _db() as c:
        return c.execute(
            "SELECT id,platforms,text FROM scheduled_posts "
            "WHERE status='pending' AND scheduled_at<=? ORDER BY scheduled_at ASC",
            (now_iso,),
        ).fetchall()


def _mark_scheduled_result(post_id: int, status: str, result: str) -> None:
    with _db() as c:
        c.execute(
            "UPDATE scheduled_posts SET status=?, published_at=?, result=? WHERE id=?",
            (status, datetime.now(timezone.utc).isoformat(), result, post_id),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 🧠 ذاكرة المحادثة لكل شخص — ردود واعية بسياق سابق مع نفس المؤلف
# ═════════════════════════════════════════════════════════════════════════════

def append_conversation(platform: str, author: str, role: str, content: str) -> None:
    """role: 'user' (رسالة واردة من الشخص) أو 'assistant' (ردّنا عليه)."""
    with _db() as c:
        c.execute(
            "INSERT INTO social_conversations (platform,author,role,content,created_at) "
            "VALUES (?,?,?,?,?)",
            (platform, author, role, content, datetime.now(timezone.utc).isoformat()),
        )


def get_conversation_history(platform: str, author: str, limit: int = 6) -> List[Dict[str, str]]:
    """آخر limit رسالة (بالترتيب الزمني الصحيح) بين الوكيل وهذا الشخص تحديداً
    على هذه المنصة — تُستخدم كسياق إضافي عند توليد الرد، بدل معاملة كل
    رسالة بمعزل عن تاريخها."""
    with _db() as c:
        rows = c.execute(
            "SELECT role, content FROM social_conversations "
            "WHERE platform=? AND author=? ORDER BY id DESC LIMIT ?",
            (platform, author, limit),
        ).fetchall()
    return [{"role": r, "content": ct} for r, ct in reversed(rows)]


# ═════════════════════════════════════════════════════════════════════════════
# 😊 تحليل المشاعر — تصنيف عبر LLM مع احتياطي محلي بدون مفاتيح
# ═════════════════════════════════════════════════════════════════════════════

_POS_WORDS = {
    "ممتاز", "رائع", "شكرا", "شكراً", "جميل", "أحسنت", "حلو", "مذهل", "أعجبني",
    "great", "awesome", "thanks", "thank", "love", "excellent", "amazing", "good",
}
_NEG_WORDS = {
    "سيء", "فاشل", "مقرف", "أكره", "غبي", "سيئ", "خطأ", "مشكلة", "زبالة", "رديء",
    "bad", "hate", "terrible", "awful", "worst", "stupid", "broken", "sucks",
}


def _heuristic_sentiment(text: str) -> tuple[str, float]:
    """احتياطي محلي بدون أي استدعاء شبكة — يُستخدم فقط لو فشل تصنيف الـLLM."""
    words = set(text.lower().split())
    pos = len(words & _POS_WORDS)
    neg = len(words & _NEG_WORDS)
    if pos == 0 and neg == 0:
        return "neutral", 0.0
    score = (pos - neg) / max(pos + neg, 1)
    label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
    return label, round(score, 2)


def analyze_sentiment(text: str) -> tuple[str, float]:
    """يُصنّف المشاعر عبر LLM (Groq/Gemini/Cloudflare/OpenRouter — أياً توفّر)
    بصيغة JSON صارمة، ويسقط تلقائياً للتحليل المحلي البسيط لو فشلت الشبكة أو
    كل المزوّدين — لا يرفع استثناءً أبداً (تحليل المشاعر ثانوي، لا يجب أن
    يوقف المراقبة عن العمل)."""
    if not text or not text.strip():
        return "neutral", 0.0
    try:
        from .free_router import chat_free

        messages = [
            {"role": "system", "content": (
                "صنّف مشاعر النص التالي إلى positive أو negative أو neutral فقط. "
                'أجب حصراً بصيغة JSON: {"label": "...", "score": -1.0 إلى 1.0}. '
                "بدون أي نص إضافي قبل أو بعد."
            )},
            {"role": "user", "content": text[:1000]},
        ]
        raw, _ = chat_free(messages, temperature=0.0, max_tokens=60)
        cleaned = raw.strip().strip("`").replace("json\n", "").strip()
        parsed = json.loads(cleaned)
        label = str(parsed.get("label", "neutral")).lower()
        score = float(parsed.get("score", 0.0))
        if label not in ("positive", "negative", "neutral"):
            raise ValueError("label غير متوقع")
        return label, round(max(-1.0, min(1.0, score)), 2)
    except Exception:
        return _heuristic_sentiment(text)


# ═════════════════════════════════════════════════════════════════════════════
# 📊 لوحة التحليلات — ملخّص مجمّع للأداء والمشاعر عبر كل المنصات
# ═════════════════════════════════════════════════════════════════════════════

def get_analytics_summary(days: int = 7) -> Dict[str, Dict]:
    """ملخّص لكل منصة خلال آخر days يوماً: عدد كل نوع حدث + توزيع المشاعر
    على العناصر المُراقَبة (monitor_hit) التي حُلّلت مشاعرها."""
    since = (datetime.now(timezone.utc) - _timedelta(days=days)).isoformat()
    with _db() as c:
        rows = c.execute(
            "SELECT platform, event_type, sentiment, ok FROM social_events "
            "WHERE created_at >= ?", (since,),
        ).fetchall()

    summary: Dict[str, Dict] = {}
    for platform, event_type, sentiment, ok in rows:
        s = summary.setdefault(platform, {
            "monitor_hit": 0, "reply": 0, "reply_failed": 0, "publish": 0, "publish_failed": 0,
            "positive": 0, "negative": 0, "neutral": 0,
        })
        if event_type == "monitor_hit":
            s["monitor_hit"] += 1
        elif event_type == "reply":
            s["reply" if ok else "reply_failed"] += 1
        elif event_type == "publish":
            s["publish" if ok else "publish_failed"] += 1
        if sentiment in ("positive", "negative", "neutral"):
            s[sentiment] += 1
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# توليد الرد — يستخدم نفس محرك GODMODE + OpenRouter
# ═════════════════════════════════════════════════════════════════════════════

def generate_reply(item: SocialItem, persona_prompt: Optional[str] = None) -> str:
    """يولّد رداً حقيقياً عبر OpenRouter بنفس شخصية GODMODE. لو غاب مفتاح
    OpenRouter أو فشل الاتصال به، يتحوّل تلقائياً لنموذج مجاني مباشر
    (Groq/Gemini/Cloudflare) عبر ai/free_router.py. يرفع استثناء فقط لو
    فشلت كل المسارات، بدل إرجاع رد مزيّف.

    واعٍ بالسياق: يضمّ آخر رسائل هذا الشخص تحديداً (نفس platform+author)
    من social_conversations، بدل معاملة كل رسالة وكأنها أول تفاعل معه."""
    sys_prompt = persona_prompt or GODMODE_SYSTEM_PROMPT
    sys_prompt += (
        f"\n\nأنت الآن ترد نيابة عن الحساب على منصة {item.platform}. "
        "اكتب رداً قصيراً مباشراً مناسباً للسياق الاجتماعي (لا مقدمات طويلة). "
        "إن وُجد سياق محادثة سابقة مع هذا الشخص أدناه، استخدمه لرد متّسق "
        "يتذكّر ما قيل، لا رداً منعزلاً كأول مرة."
    )

    # ── سياق تعليمي: لو نص الرسالة يطابق مفاهيم من مصادر التخصصات
    # (فيزياء/كيمياء/نحو/إنجليزي/جغرافيا/حاسوب...)، نرفقها كمرجع دقيق
    # بدل الاعتماد فقط على معرفة النموذج العامة. لا يُفرض أي رد؛ فقط سياق
    # إضافي يستخدمه النموذج إن كان مناسباً.
    try:
        from knowledge_sources.domain_lookup import search_domain_concepts

        matches = search_domain_concepts(item.text, limit=3)
        if matches:
            refs = "\n".join(
                f"- [{m['domain_ar']}] {m['concept']}: {m['text']}" for m in matches
            )
            sys_prompt += (
                "\n\nمعلومات مرجعية دقيقة من قاعدة معرفة NSM التعليمية "
                "(استخدمها إن كانت ذات صلة بسؤال الشخص، ولا تذكر أنها "
                f"'قاعدة بيانات' صراحة):\n{refs}"
            )
    except Exception:
        pass  # غياب السياق التعليمي لا يجب أن يمنع توليد رد عادي

    history = get_conversation_history(item.platform, item.author, limit=6)
    messages = [{"role": "system", "content": sys_prompt}]
    for h in history:
        # رسائل الشخص السابقة تُعرض كسياق "user"، وردودنا السابقة كـ"assistant"
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": f"[{item.author}]: {item.text}"})

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if api_key:
        import requests as _requests

        try:
            resp = _requests.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": os.environ.get("SOCIAL_AGENT_MODEL", DEFAULT_REPLY_MODEL),
                    "messages": messages,
                    "max_tokens": 300,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            if text:
                return text
        except Exception:
            pass  # يسقط تلقائياً للنموذج المجاني المباشر أدناه

    # ── لا يوجد مفتاح OpenRouter صالح، أو فشل الاتصال به: نموذج مجاني مباشر ──
    try:
        from .free_router import chat_free

        text, _used_model = chat_free(messages, temperature=0.7, max_tokens=300)
        return text
    except Exception as exc:
        raise NotConfiguredError(
            "تعذّر توليد الرد: لا يوجد OPENROUTER_API_KEY صالح، وفشلت كل "
            f"النماذج المجانية المباشرة أيضاً. التفاصيل: {exc}"
        ) from exc


# ═════════════════════════════════════════════════════════════════════════════
# المدير — خيط خلفي واحد (singleton) يستطلع كل منصّة مُفعَّلة
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PlatformStatus:
    platform: str
    configured: bool
    missing_env: List[str] = field(default_factory=list)
    enabled: bool = False
    last_poll: Optional[str] = None
    last_error: Optional[str] = None


class SocialAgentManager:
    """Singleton على مستوى العملية — خيط واحد يخدم كل المنصات بالتوازي منطقياً
    (استطلاع تسلسلي سريع لكل منصة ضمن نفس الدورة، وهو كافٍ لأحمال المراقبة
    العادية ويحافظ على السيطرة على استهلاك الحصة المجانية)."""

    _instance: Optional["SocialAgentManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self.adapters = {pid: cls() for pid, cls in ALL_ADAPTERS.items()}
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        self._status: Dict[str, PlatformStatus] = {
            pid: PlatformStatus(pid, a.is_configured(), a.missing_env())
            for pid, a in self.adapters.items()
        }

    @classmethod
    def instance(cls) -> "SocialAgentManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ── الحالة/الإعدادات ─────────────────────────────────────────────────
    def refresh_status(self):
        enabled_platforms = set(get_config("enabled_platforms", []))
        for pid, a in self.adapters.items():
            st = self._status[pid]
            st.configured = a.is_configured()
            st.missing_env = a.missing_env()
            st.enabled = pid in enabled_platforms

    def status(self) -> Dict[str, PlatformStatus]:
        self.refresh_status()
        return self._status

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        with self._lock:
            if self.is_running():
                return
            self._stop_flag.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                             name="social-agent-loop")
            self._thread.start()
            set_config("agent_running", True)

    def stop(self):
        with self._lock:
            self._stop_flag.set()
            set_config("agent_running", False)

    # ── معالجة عنصر وارد واحد (مشتركة بين polling و webhook) ─────────────
    def _handle_item(self, pid: str, adapter: "PlatformAdapter", item: SocialItem,
                      keywords: List[str], auto_reply: bool) -> None:
        """يطبّق نفس منطق: تفادي التكرار → مطابقة الكلمات → تسجيل/تحليل
        مشاعر → رد تلقائي (إن فُعّل) — بغض النظر إن كان العنصر جاء من
        استطلاع دوري (fetch_new_items) أو من webhook فوري."""
        if _is_seen(pid, item.external_id):
            return
        _mark_seen(pid, item.external_id)
        matched = (not keywords) or any(k in item.text.lower() for k in keywords)
        if matched:
            label, score = analyze_sentiment(item.text)
            log_event(pid, "monitor_hit", item.author, item.text,
                      sentiment=label, sentiment_score=score)
            append_conversation(pid, item.author, "user", item.text)
        if matched and auto_reply:
            try:
                reply_text = generate_reply(item)
                adapter.reply(item, reply_text)
                log_event(pid, "reply", item.author, item.text,
                          reply_content=reply_text, ok=True)
                append_conversation(pid, item.author, "assistant", reply_text)
            except Exception as e:  # noqa: BLE001
                log_event(pid, "reply", item.author, item.text,
                          reply_content=str(e), ok=False)

    # ── استقبال عنصر فوري من webhook خارجي (api_server.py) ───────────────
    def ingest_webhook_item(self, pid: str, item: SocialItem) -> None:
        """نقطة الدخول التي يستدعيها endpoint الـwebhook (مثال: تيليجرام)
        عند وصول تحديث فوري. تُطبَّق عليه كل قواعد المراقبة (كلمات
        مفتاحية، رد تلقائي، تسجيل الأحداث) تماماً كما لو جاء من polling،
        دون انتظار دورة الاستطلاع التالية. لا تُنفَّذ إن كانت المنصة غير
        مُهيّأة أو غير مفعّلة أصلاً."""
        adapter = self.adapters.get(pid)
        if not adapter or not adapter.is_configured():
            return
        enabled_platforms = set(get_config("enabled_platforms", []))
        if pid not in enabled_platforms:
            return
        keywords = [k.strip().lower() for k in get_config("keywords", []) if k.strip()]
        auto_reply = get_config("auto_reply", False)
        try:
            self._handle_item(pid, adapter, item, keywords, auto_reply)
            st = self._status.get(pid)
            if st:
                st.last_poll = datetime.now(timezone.utc).isoformat()
                st.last_error = None
        except Exception as e:  # noqa: BLE001
            st = self._status.get(pid)
            if st:
                st.last_error = f"{e}"
            log_event(pid, "monitor_error", "-", str(e), ok=False)

    # ── دورة الاستطلاع الرئيسية ──────────────────────────────────────────
    def _loop(self):
        while not self._stop_flag.is_set():
            interval = get_config("poll_interval", DEFAULT_POLL_INTERVAL)
            enabled_platforms = set(get_config("enabled_platforms", []))
            webhook_platforms = set(get_config("webhook_enabled_platforms", []))
            keywords = [k.strip().lower() for k in get_config("keywords", []) if k.strip()]
            auto_reply = get_config("auto_reply", False)

            # 📅 معالجة المنشورات المجدولة المستحقة (بغض النظر عن المنصات المفعّلة
            # للمراقبة — الجدولة مستقلة عن الاستطلاع)
            self._process_due_scheduled()

            for pid in enabled_platforms:
                if self._stop_flag.is_set():
                    break
                # منصة بوضع webhook فعّال: تصلها التحديثات فوراً عبر
                # ingest_webhook_item من api_server.py — استطلاعها بالتوازي
                # هنا سيُعالج نفس العناصر مرتين ويُهدر حصة الـAPI بلا فائدة.
                if pid in webhook_platforms:
                    continue
                adapter = self.adapters.get(pid)
                st = self._status.get(pid)
                if not adapter or not adapter.is_configured():
                    continue
                if not adapter.supports_monitoring:
                    # مثال: Pinterest — النشر يعمل عبر publish_to، لكن لا
                    # يوجد API لقراءة تعليقات لمراقبتها أصلاً (قيد من
                    # المنصة نفسها، راجع pinterest_adapter.py). استدعاء
                    # fetch_new_items هنا سيرفع PlatformCapabilityError
                    # بلا فائدة في كل دورة — نتخطى بصمت بدل تسجيل خطأ
                    # متكرر عن شيء ليس خطأً أصلاً.
                    continue
                try:
                    since = _seen_ids_for(pid)
                    new_items = adapter.fetch_new_items(since)
                    for item in new_items:
                        self._handle_item(pid, adapter, item, keywords, auto_reply)
                    if st:
                        st.last_poll = datetime.now(timezone.utc).isoformat()
                        st.last_error = None
                except Exception as e:  # noqa: BLE001
                    if st:
                        st.last_error = f"{e}"
                    log_event(pid, "monitor_error", "-", str(e), ok=False)

            self._stop_flag.wait(timeout=interval)

    # ── تفعيل/إلغاء وضع webhook لمنصة تدعمه (مثال: تيليجرام) ─────────────
    def enable_webhook(self, pid: str, url: str, secret_token: Optional[str] = None) -> dict:
        """يفعّل webhook فعلياً لدى مزوّد المنصة (وليس فقط محلياً) عبر
        adapter.set_webhook، ثم يسجّل المنصة ضمن webhook_enabled_platforms
        كي تتوقف دورة polling عن استطلاعها. يرفع ValueError إن كانت
        المنصة لا تدعم webhook حقيقياً (راجع WEBHOOKS.md)."""
        adapter = self.adapters.get(pid)
        if not adapter:
            raise ValueError(f"منصة غير معروفة: {pid}")
        if not adapter.supports_webhook:
            raise ValueError(
                f"{pid}: لا يوفّر API الرسمي webhook حقيقياً لهذه الحالة "
                "(استقبال أحداث) — راجع ai/social_platforms/WEBHOOKS.md."
            )
        result = adapter.set_webhook(url, secret_token=secret_token)
        current = set(get_config("webhook_enabled_platforms", []))
        current.add(pid)
        set_config("webhook_enabled_platforms", sorted(current))
        return result

    def disable_webhook(self, pid: str) -> dict:
        """يلغي webhook لدى مزوّد المنصة ويعيدها لوضع polling العادي."""
        adapter = self.adapters.get(pid)
        if not adapter:
            raise ValueError(f"منصة غير معروفة: {pid}")
        result = adapter.delete_webhook() if hasattr(adapter, "delete_webhook") else {}
        current = set(get_config("webhook_enabled_platforms", []))
        current.discard(pid)
        set_config("webhook_enabled_platforms", sorted(current))
        return result

    # ── معالجة المنشورات المجدولة المستحقة ──────────────────────────────
    def _process_due_scheduled(self):
        for post_id, platforms, text in _due_scheduled_posts():
            try:
                results = self.publish_to(platforms, text, resume_key=post_id)
                failed = {p: r for p, r in results.items() if str(r).startswith("ERROR")}
                status = "failed" if failed else "published"
                _mark_scheduled_result(post_id, status, json.dumps(results, ensure_ascii=False))
                # العملية اكتملت نهائياً (نجاحاً أو فشلاً) — لم تعد هناك حاجة
                # لنقاط الاسترجاع الخاصة بهذا post_id، والحالة لم تعد 'pending'
                # فلن يُعاد التقاطه في _due_scheduled_posts لاحقاً على أي حال.
                _clear_checkpoints(post_id)
            except Exception as e:  # noqa: BLE001
                # خطأ غير متوقع قبل تجميع النتائج: أبقِ الحالة 'pending' لو
                # كانت هذه محاولة لم تُكمَل بعد أي منصة، وإلا سجّلها فاشلة —
                # لكن أبقِ نقاط الاسترجاع كي لا تُعاد المنصات الناجحة سابقاً.
                _mark_scheduled_result(post_id, "failed", f"ERROR: {e}")

    # ── نشر يدوي/برمجي فوري إلى منصة أو أكثر ────────────────────────────
    def publish_to(self, platforms: List[str], text: str,
                    resume_key: Optional[int] = None,
                    per_platform_text: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """ينشر النص على كل منصة مطلوبة بالتوازي (خيط لكل منصة). يعيد
        {platform: post_id | 'ERROR: ...'}.

        resume_key: مُعرّف اختياري (عادةً post_id من scheduled_posts) يفعّل
        الاسترجاع (checkpointing): أي منصة سبق أن نجح نشرها تحت نفس
        resume_key تُتخطَّى ولا تُعاد (يُستخدم نتيجتها المحفوظة)، ونتيجة
        كل منصة جديدة تُحفَظ فور اكتمالها بدل انتظار انتهاء كل المنصات —
        بحيث لو تعطّلت العملية (تعطّل الخادم مثلاً) منتصف النشر، لا تُكرَّر
        المنشورات على منصات نجحت فعلاً عند إعادة المحاولة.

        per_platform_text: قاموس اختياري {platform: نص مخصّص} — لو وُجدت
        منصة فيه يُستخدم نصها بدل text الموحّد (راجع ai/platform_profiles.py
        لتوليد هذا القاموس تلقائياً بحدود أحرف وهاشتاقات مناسبة لكل منصة).
        المنصات غير الموجودة في القاموس تستخدم text كما هو — لا كسر توافق."""
        results: Dict[str, str] = {}
        lock = threading.Lock()
        per_platform_text = per_platform_text or {}

        already_done = _get_checkpoint_results(resume_key) if resume_key is not None else {}
        pending_platforms = []
        for pid in platforms:
            cached = already_done.get(pid)
            if cached is not None and cached[0]:  # نجح سابقاً — لا تُعِد النشر
                results[pid] = cached[1]
            else:
                pending_platforms.append(pid)

        def _one(pid: str):
            pid_text = per_platform_text.get(pid, text)
            # Crisis / freeze gate (social_swarm)
            try:
                from ai.social_swarm import pre_publish_check
                chk = pre_publish_check(pid_text)
                if not chk.get("ok"):
                    res, ok = f"ERROR: blocked_by_crisis_gate — {chk.get('reason_ar')}", False
                    if resume_key is not None:
                        _save_checkpoint(resume_key, pid, ok, res)
                    with lock:
                        results[pid] = res
                    log_event(pid, "publish_blocked", "agent", pid_text, reply_content=str(chk), ok=False)
                    return
            except Exception:
                pass
            adapter = self.adapters.get(pid)
            if not adapter:
                res, ok = "ERROR: منصة غير معروفة", False
            elif not adapter.is_configured():
                res = f"ERROR: غير مُهيّأة — يلزم {', '.join(adapter.missing_env())}"
                ok = False
            else:
                try:
                    post_id = adapter.publish(pid_text)
                    log_event(pid, "publish", "agent", pid_text, ok=True)
                    res, ok = post_id, True
                except Exception as e:  # noqa: BLE001
                    log_event(pid, "publish", "agent", pid_text, reply_content=str(e), ok=False)
                    res, ok = f"ERROR: {e}", False
            if resume_key is not None:
                _save_checkpoint(resume_key, pid, ok, res)
            with lock:
                results[pid] = res

        TIMEOUT = 45
        threads = [(pid, threading.Thread(target=_one, args=(pid,))) for pid in pending_platforms]
        for _, t in threads:
            t.start()
        for pid, t in threads:
            t.join(timeout=TIMEOUT)
            # إذا انتهت المهلة قبل انتهاء الخيط — سجّل خطأ صريح (تبقى
            # المنصة قابلة لإعادة المحاولة في الدورة القادمة لأنها لم
            # تُحفَظ كـ ok=True في checkpoint)
            if t.is_alive():
                with lock:
                    if pid not in results:
                        err = f"ERROR: انتهت مهلة {TIMEOUT}s قبل اكتمال النشر"
                        results[pid] = err
                        log_event(pid, "publish", "agent", per_platform_text.get(pid, text),
                                  reply_content=err, ok=False)
        return results


def get_manager() -> SocialAgentManager:
    return SocialAgentManager.instance()
