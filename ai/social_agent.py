"""
SOCIAL AGENT — الوكيل الاجتماعي الموحد
==========================================
خدمة خلفية واحدة تشغّل عدة منصات تواصل بالتوازي: تنشر + ترد تلقائياً +
تراقب كلمات/إشارات مفتاحية — وتستخدم نفس محرك GODMODE/OpenRouter الذي
يستعمله الوكيل الأساسي في app.py (نفس الشخصية والسياق، ليس نظاماً منفصلاً).

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .social_platforms import ALL_ADAPTERS, PLATFORM_LABELS_AR, SocialItem, NotConfiguredError

try:
    from .godmode import GODMODE_SYSTEM_PROMPT
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
            content TEXT, reply_content TEXT, created_at TEXT, ok INTEGER
        )""")
    conn.commit()
    return conn


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
              reply_content: str = "", ok: bool = True) -> None:
    with _db() as c:
        c.execute(
            "INSERT INTO social_events (platform,event_type,author,content,reply_content,created_at,ok) "
            "VALUES (?,?,?,?,?,?,?)",
            (platform, event_type, author, content, reply_content,
             datetime.now(timezone.utc).isoformat(), 1 if ok else 0),
        )


def get_recent_events(limit: int = 40) -> List[tuple]:
    with _db() as c:
        return c.execute(
            "SELECT platform,event_type,author,content,reply_content,created_at,ok "
            "FROM social_events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


def get_event_counts() -> Dict[str, int]:
    with _db() as c:
        rows = c.execute(
            "SELECT event_type, COUNT(*) FROM social_events GROUP BY event_type"
        ).fetchall()
    return dict(rows)


# ═════════════════════════════════════════════════════════════════════════════
# توليد الرد — يستخدم نفس محرك GODMODE + OpenRouter
# ═════════════════════════════════════════════════════════════════════════════

def generate_reply(item: SocialItem, persona_prompt: Optional[str] = None) -> str:
    """يولّد رداً حقيقياً عبر OpenRouter بنفس شخصية GODMODE. لو غاب مفتاح
    OpenRouter أو فشل الاتصال به، يتحوّل تلقائياً لنموذج مجاني مباشر
    (Groq/Gemini/Cloudflare) عبر ai/free_router.py. يرفع استثناء فقط لو
    فشلت كل المسارات، بدل إرجاع رد مزيّف."""
    sys_prompt = persona_prompt or GODMODE_SYSTEM_PROMPT
    sys_prompt += (
        f"\n\nأنت الآن ترد نيابة عن الحساب على منصة {item.platform}. "
        "اكتب رداً قصيراً مباشراً مناسباً للسياق الاجتماعي (لا مقدمات طويلة)."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"[{item.author}]: {item.text}"},
    ]

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

    # ── دورة الاستطلاع الرئيسية ──────────────────────────────────────────
    def _loop(self):
        while not self._stop_flag.is_set():
            interval = get_config("poll_interval", DEFAULT_POLL_INTERVAL)
            enabled_platforms = set(get_config("enabled_platforms", []))
            keywords = [k.strip().lower() for k in get_config("keywords", []) if k.strip()]
            auto_reply = get_config("auto_reply", False)

            for pid in enabled_platforms:
                if self._stop_flag.is_set():
                    break
                adapter = self.adapters.get(pid)
                st = self._status.get(pid)
                if not adapter or not adapter.is_configured():
                    continue
                try:
                    since = _seen_ids_for(pid)
                    new_items = adapter.fetch_new_items(since)
                    for item in new_items:
                        if _is_seen(pid, item.external_id):
                            continue
                        _mark_seen(pid, item.external_id)
                        matched = (not keywords) or any(k in item.text.lower() for k in keywords)
                        if matched:
                            log_event(pid, "monitor_hit", item.author, item.text)
                        if matched and auto_reply:
                            try:
                                reply_text = generate_reply(item)
                                adapter.reply(item, reply_text)
                                log_event(pid, "reply", item.author, item.text,
                                          reply_content=reply_text, ok=True)
                            except Exception as e:  # noqa: BLE001
                                log_event(pid, "reply", item.author, item.text,
                                          reply_content=str(e), ok=False)
                    if st:
                        st.last_poll = datetime.now(timezone.utc).isoformat()
                        st.last_error = None
                except Exception as e:  # noqa: BLE001
                    if st:
                        st.last_error = f"{e}"
                    log_event(pid, "monitor_error", "-", str(e), ok=False)

            self._stop_flag.wait(timeout=interval)

    # ── نشر يدوي/برمجي فوري إلى منصة أو أكثر ────────────────────────────
    def publish_to(self, platforms: List[str], text: str) -> Dict[str, str]:
        """ينشر النص على كل منصة مطلوبة بالتوازي (خيط لكل منصة). يعيد
        {platform: post_id | 'ERROR: ...'}."""
        results: Dict[str, str] = {}
        lock = threading.Lock()

        def _one(pid: str):
            adapter = self.adapters.get(pid)
            if not adapter:
                res = "ERROR: منصة غير معروفة"
            elif not adapter.is_configured():
                res = f"ERROR: غير مُهيّأة — يلزم {', '.join(adapter.missing_env())}"
            else:
                try:
                    post_id = adapter.publish(text)
                    log_event(pid, "publish", "agent", text, ok=True)
                    res = post_id
                except Exception as e:  # noqa: BLE001
                    log_event(pid, "publish", "agent", text, reply_content=str(e), ok=False)
                    res = f"ERROR: {e}"
            with lock:
                results[pid] = res

        TIMEOUT = 45
        threads = [(pid, threading.Thread(target=_one, args=(pid,))) for pid in platforms]
        for _, t in threads:
            t.start()
        for pid, t in threads:
            t.join(timeout=TIMEOUT)
            # إذا انتهت المهلة قبل انتهاء الخيط — سجّل خطأ صريح
            if t.is_alive():
                with lock:
                    if pid not in results:
                        err = f"ERROR: انتهت مهلة {TIMEOUT}s قبل اكتمال النشر"
                        results[pid] = err
                        log_event(pid, "publish", "agent", text,
                                  reply_content=err, ok=False)
        return results


def get_manager() -> SocialAgentManager:
    return SocialAgentManager.instance()
