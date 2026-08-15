"""
Fable Engine — محرك السرد الإبداعي العربي — NSM v18.4
========================================================
يوفر طبقة سرد تفاعلي وتوليد شعري باللغة العربية، مبنية فوق LLMFallback
(نفس مزوّدي NSM: Anthropic أولاً ثم Cloudflare/Gemini/OpenRouter/Groq/CKG).

المكوّنات:
    - STORY_MODES:  أوضاع القصة الستة الأصلية + 4 أوضاع إضافية (الأكثر طلباً)
    - CHARACTERS:   شخصيات أدبية/تاريخية تُستخدم كـ"راوٍ" أسلوبي (وليس اقتباسات حقيقية)
    - ARABIC_METERS: بحور الشعر العربي الأساسية مع تفاعيلها للتوليد والشرح
    - NarrativeMemory: ذاكرة سردية خفيفة (SQLite) تحفظ فصول القصة والأدوار
    - FableEngine:  الواجهة الرئيسية — start_story / continue_story /
                    generate_poem / quick command (أنشد، صف، أضف حواراً، لخّص)

الاستخدام:
    from ai.fable_engine import FableEngine

    engine = FableEngine(llm_fallback=my_llm_fallback, db_path="memory/fable.db")
    session = engine.start_story(mode="مغامرة", character="شهرزاد")
    print(session.text)
    print(session.choices)   # 3 خيارات تفاعلية

    session = engine.continue_story(session.session_id, choice_index=0)
"""

from __future__ import annotations

import json
import logging
import re
import time
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ai.tts_engine import TTSEngine

logger = logging.getLogger("FableEngine")


# ══════════════════════════════════════════════════════════════════════════
# أوضاع القصة — 6 أصلية + 5 إضافية (رعب، خيال علمي، رومانسية، أطفال،
# وقصص إسلامية تربوية للأطفال تغرس القيم دون اختلاق أقوال منسوبة للأنبياء
# أو الصحابة أو أي شخصية دينية حقيقية)
# ══════════════════════════════════════════════════════════════════════════

STORY_MODES: Dict[str, Dict[str, str]] = {
    "مغامرة": {
        "emoji": "🗺️",
        "desc": "رحلة شائقة مليئة بالتحديات والاكتشافات",
        "tone": "حماسي، سريع الإيقاع، مليء بالمفاجآت والتحديات الجسدية والذهنية",
    },
    "شعر": {
        "emoji": "🪶",
        "desc": "سرد قصصي منظوم على بحور الشعر العربي",
        "tone": "موزون، بليغ، يستخدم الاستعارة والتشبيه بكثافة",
    },
    "تاريخية": {
        "emoji": "🏛️",
        "desc": "قصة مستوحاة من حقبة تاريخية مع دقة في التفاصيل",
        "tone": "رصين، وصفي، يحترم السياق الزمني والحضاري للحقبة",
    },
    "غموض": {
        "emoji": "🔍",
        "desc": "لغز يتكشف تدريجياً عبر أدلة واستنتاج",
        "tone": "متوتر، مشوّق، يزرع أدلة صغيرة قبل الكشف",
    },
    "حكمة": {
        "emoji": "🕊️",
        "desc": "حكاية رمزية تحمل عبرة أو درساً أخلاقياً",
        "tone": "هادئ، تأملي، أقرب لأسلوب الأمثال والحكايات الرمزية",
    },
    "إبداع حر": {
        "emoji": "🎨",
        "desc": "سرد حر بلا قيود على النوع الأدبي",
        "tone": "مرن، يتبع خيال القارئ دون التزام بنمط ثابت",
    },
    # ── الإضافات الجديدة ──
    "رعب": {
        "emoji": "🌑",
        "desc": "أجواء مشوّقة ومقلقة دون تفاصيل صادمة أو دموية",
        "tone": "غامض، متصاعد التوتر، يعتمد على الترقّب أكثر من الوصف الصريح",
    },
    "خيال علمي": {
        "emoji": "🚀",
        "desc": "عوالم مستقبلية وتقنيات متخيَّلة",
        "tone": "استكشافي، يمزج الخيال بمفاهيم علمية معقولة",
    },
    "رومانسية": {
        "emoji": "🌹",
        "desc": "قصة مشاعر واحترام متبادل بين شخصيات القصة",
        "tone": "عاطفي وراقٍ، يركّز على الحوار الداخلي والمشاعر لا التفاصيل الحسية",
    },
    "قصص أطفال": {
        "emoji": "🧸",
        "desc": "حكاية بسيطة وآمنة ومسلّية بلغة سهلة للأطفال",
        "tone": "بسيط، مرح، إيجابي، بجمل قصيرة وعبرة لطيفة في النهاية",
    },
    "قصص إسلامية تربوية": {
        "emoji": "🕌",
        "desc": "حكاية للأطفال تغرس قيمة إسلامية أو خُلقاً حميداً (الصدق، بر الوالدين، الأمانة، الصلاة، مساعدة الآخرين) عبر شخصيات خيالية من الحياة اليومية",
        "tone": (
            "بسيط ودافئ ومطمئن، بجمل قصيرة مناسبة للأطفال، يبني القصة حول شخصيات "
            "خيالية عادية (طفل، عائلة، حيوان) تتعلّم قيمة إسلامية عبر موقف يومي واقعي. "
            "لا تنسب أبداً أي قول أو حدث أو حوار مختلَق إلى الأنبياء أو الصحابة أو أي "
            "شخصية دينية حقيقية، ولا تخترع آيات أو أحاديث أو رواية تاريخية — إذا أردت "
            "الإشارة إلى قيمة قرآنية أو نبوية فاذكرها بعبارة عامة معروفة دون تفاصيل حوارية "
            "مُختلَقة. أنهِ كل فصل بعبرة لطيفة وواضحة للطفل."
        ),
    },
}

DEFAULT_MODE = "إبداع حر"

# ── قيم إسلامية مستهدفة — تُستخدم فقط مع وضع «قصص إسلامية تربوية» لبناء
# فكرة قصة موجّهة دون الحاجة لكتابة المستخدم فكرة حرة ────────────────────
ISLAMIC_VALUES: List[str] = [
    "الصدق",
    "بر الوالدين",
    "الأمانة",
    "الصلاة",
    "مساعدة الآخرين",
    "الشكر",
    "الصبر",
    "التعاون",
    "حسن الجوار",
    "النظافة",
]


# ══════════════════════════════════════════════════════════════════════════
# شخصيات أدبية — تُستخدم كأسلوب سردي/راوٍ خيالي داخل القصة التي يُنشئها
# النظام، وليست اقتباسات حقيقية منسوبة لأشخاص. الشخصيات هنا إما تراثية
# رمزية (شهرزاد) أو أدباء تاريخيون معروفون بأسلوب أدبي مميز يُحتذى في
# الأسلوب فقط (لا اقتباسات فعلية تُنسب إليهم).
# ══════════════════════════════════════════════════════════════════════════

CHARACTERS: Dict[str, Dict[str, str]] = {
    "شهرزاد": {
        "emoji": "👑",
        "style": "راوية ألف ليلة وليلة — تبدأ كل فصل بتشويق وتنهيه عند لحظة حرجة",
    },
    "ابن بطوطة": {
        "emoji": "🧭",
        "style": "رحّالة يصف الأمكنة والشعوب والعادات بعين الرحّالة الفضولي",
    },
    "المتنبي": {
        "emoji": "⚔️",
        "style": "أسلوب شعري فخم، حكمة وفخر وصور بلاغية قوية",
    },
    "ابن خلدون": {
        "emoji": "📜",
        "style": "تحليلي، يربط الأحداث بأسبابها الاجتماعية والعمرانية",
    },
    "الراوي": {
        "emoji": "📖",
        "style": "راوٍ عليم محايد، سرد كلاسيكي بضمير الغائب",
    },
}

DEFAULT_CHARACTER = "الراوي"


# ══════════════════════════════════════════════════════════════════════════
# بحور الشعر العربي الأساسية — للاستخدام في توليد الأبيات وشرحها
# ══════════════════════════════════════════════════════════════════════════

ARABIC_METERS: Dict[str, Dict[str, str]] = {
    "الطويل": {"تفعيلة": "فعولن مفاعيلن فعولن مفاعيلن", "وصف": "أشهر البحور، يُستخدم في المدح والحكمة"},
    "الكامل": {"تفعيلة": "متفاعلن متفاعلن متفاعلن", "وصف": "إيقاع متدفق، مناسب للحماسة والرثاء"},
    "البسيط": {"تفعيلة": "مستفعلن فاعلن مستفعلن فاعلن", "وصف": "واضح الإيقاع، يُستخدم في السرد والوصف"},
    "الوافر": {"تفعيلة": "مفاعلتن مفاعلتن فعولن", "وصف": "إيقاع رشيق، شائع في الغزل والفخر"},
    "الرمل": {"تفعيلة": "فاعلاتن فاعلاتن فاعلاتن", "وصف": "خفيف ومرن، يناسب الغزل الرقيق"},
}


# ══════════════════════════════════════════════════════════════════════════
# ذاكرة سردية — SQLite خفيفة تحفظ فصول كل جلسة قصة
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
# أنماط Shorts الإبداعية — توجّه النبرة والإيقاع البصري
# ══════════════════════════════════════════════════════════════════════════

SHORTS_STYLES: Dict[str, Dict[str, str]] = {
    "حقائق سريعة": {
        "emoji": "⚡",
        "tone": "أرقام صادمة، جمل قصيرة، إيقاع سريع جداً",
        "hook": "هل تعلم أن…",
    },
    "تحفيزي": {
        "emoji": "🔥",
        "tone": "طاقة عالية، مخاطبة مباشرة، خاتمة تدعو للفعل",
        "hook": "توقّف لحظة…",
    },
    "تعليمي": {
        "emoji": "🧠",
        "tone": "شرح مبسّط، خطوة بخطوة، أمثلة حسّية",
        "hook": "في أقل من دقيقة ستفهم…",
    },
    "قصصي": {
        "emoji": "📖",
        "tone": "سرد درامي قصير، شخصية وحدث وتحوّل",
        "hook": "في يومٍ ما…",
    },
    "درامي": {
        "emoji": "🎬",
        "tone": "توتر بصري، مفارقات، نهاية مفتوحة أو صادمة",
        "hook": "لم يكن أحد يتوقع…",
    },
}

DEFAULT_SHORTS_STYLE = "حقائق سريعة"


class NarrativeMemory:
    """تخزين فصول القصة لكل جلسة، مع الحفاظ على السياق عبر الأدوار."""

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fable_sessions (
                    session_id TEXT PRIMARY KEY,
                    mode       TEXT NOT NULL,
                    character  TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fable_chapters (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL,   -- 'system' | 'reader' | 'narration'
                    content    TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES fable_sessions(session_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shorts_history (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    title         TEXT NOT NULL,
                    format        TEXT NOT NULL DEFAULT 'شورت',
                    segments_json TEXT NOT NULL,
                    total_seconds INTEGER NOT NULL DEFAULT 0,
                    source_excerpt TEXT DEFAULT '',
                    created_at    REAL NOT NULL
                )
            """)
            conn.commit()

    def create_session(self, session_id: str, mode: str, character: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO fable_sessions (session_id, mode, character, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, mode, character, time.time()),
            )
            conn.commit()

    def add_chapter(self, session_id: str, role: str, content: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO fable_chapters (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, time.time()),
            )
            conn.commit()

    def get_session(self, session_id: str) -> Optional[sqlite3.Row]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM fable_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            return row

    def get_history(self, session_id: str, limit: int = 20) -> List[sqlite3.Row]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM fable_chapters WHERE session_id = ? "
                "ORDER BY id ASC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            return rows

    def list_recent_sessions(self, limit: int = 10) -> List[sqlite3.Row]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM fable_sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return rows

    def delete_session(self, session_id: str) -> None:
        """يحذف جلسة قصة وكل فصولها المحفوظة نهائياً."""
        with self._conn() as conn:
            conn.execute("DELETE FROM fable_chapters WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM fable_sessions WHERE session_id = ?", (session_id,))
            conn.commit()

    def get_narration_preview(self, session_id: str) -> tuple[str, int]:
        """استعلام خفيف لمعاينة المكتبة: أول فصل + عدد الفصول فقط، بدل تحميل
        كل نص القصة (قد يصل لمئات الفقرات) لكل قصة معروضة في القائمة —
        هذا يُستدعى لكل قصة محفوظة على كل إعادة تحميل للتبويب، فتقليل حجم
        البيانات المقروءة هنا مهم لأداء الصفحة ككل."""
        with self._conn() as conn:
            first_row = conn.execute(
                "SELECT content FROM fable_chapters WHERE session_id = ? AND role = 'narration' "
                "ORDER BY id ASC LIMIT 1",
                (session_id,),
            ).fetchone()
            count_row = conn.execute(
                "SELECT COUNT(*) AS n FROM fable_chapters WHERE session_id = ? AND role = 'narration'",
                (session_id,),
            ).fetchone()
            first_text = first_row["content"] if first_row else ""
            count = count_row["n"] if count_row else 0
            return first_text, count

    def get_last_narration(self, session_id: str) -> str:
        """يجلب آخر فصل مسرود فقط (لاستئناف القصة) دون تحميل كل التاريخ."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT content FROM fable_chapters WHERE session_id = ? AND role = 'narration' "
                "ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            return row["content"] if row else ""

    def save_short(
        self, title: str, format_: str, segments: List[dict],
        total_seconds: int, source_excerpt: str = "",
    ) -> int:
        """يحفظ سيناريو Shorts/وثائقي مولَّد (بدون الصوت/الفيديو — تلك عابرة
        وتُعاد عند الطلب) ليمكن استرجاعه لاحقاً من المكتبة بدون تكلفة LLM
        إضافية. يعيد المعرّف الداخلي للسجل المحفوظ."""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO shorts_history "
                "(title, format, segments_json, total_seconds, source_excerpt, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (title, format_, json.dumps(segments, ensure_ascii=False),
                 total_seconds, (source_excerpt or "")[:300], time.time()),
            )
            conn.commit()
            return cur.lastrowid

    def list_recent_shorts(self, limit: int = 20) -> List[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM shorts_history ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def delete_short(self, short_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM shorts_history WHERE id = ?", (short_id,))
            conn.commit()


# ══════════════════════════════════════════════════════════════════════════
# نتيجة الفصل — ما يُعاد للواجهة بعد كل استدعاء
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class FableChapter:
    session_id: str
    text:       str
    choices:    List[str] = field(default_factory=list)
    mode:       str = DEFAULT_MODE
    character:  str = DEFAULT_CHARACTER
    provider:   str = ""
    error:      Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════
# 🎬 وثائقي — مولّد سيناريو وثائقي مقسّم لمشاهد (مستوحى من فكرة
# Higgsfield Explainer: بحث تلقائي عن الموضوع + سيناريو مُقسّم إلى مشاهد
# مع سرد صوتي ووصف مرئي لكل مشهد). ملاحظة مهمة: NSM لا يملك نموذج توليد
# فيديو فعلي (لا يوجد لدينا وصول لنموذج مثل Gemini Omni Flash)، لذا هذا
# المولّد ينتج نص السيناريو والتوجيه المرئي فقط، جاهزاً لتُغذّى به أداة
# توليد فيديو خارجية (مثل Higgsfield) يدوياً.
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ExplainerSegment:
    index:        int
    narration:    str   # النص المسرود (لصوت الراوي)
    visual_notes: str    # وصف اللقطة/الصورة المقترحة لهذا المقطع
    est_seconds:  int = 30
    audio_bytes:  Optional[bytes] = None   # يُملأ بعد render_audio()
    audio_format: str = "mp3"
    audio_provider: str = ""
    # توقيت حقيقي لكل كلمة (نص, بداية بالثانية, مدة بالثانية) — يُملأ فقط
    # عند نجاح Edge TTS (المزوّد الوحيد الذي يُصدر WordBoundary فعلية عبر
    # ai/tts_engine.py)، يبقى [] لبقية المزوّدين ويتراجع VideoEngine
    # لتقدير تناسبي عند غيابه.
    word_timings: List[tuple] = field(default_factory=list)


@dataclass
class ExplainerScript:
    topic:       str
    title:       str
    segments:    List[ExplainerSegment] = field(default_factory=list)
    provider:    str = ""
    error:       Optional[str] = None
    format:      str = "وثائقي"   # "وثائقي" (Explainer) أو "شورت" (NotebookLM-style Shorts)

    @property
    def total_seconds(self) -> int:
        return sum(s.est_seconds for s in self.segments)

    @property
    def has_audio(self) -> bool:
        return bool(self.segments) and all(s.audio_bytes for s in self.segments)

    @property
    def full_narration(self) -> str:
        return "\n\n".join(s.narration for s in self.segments)


# ══════════════════════════════════════════════════════════════════════════
# FableEngine — الواجهة الرئيسية
# ══════════════════════════════════════════════════════════════════════════

class FableEngine:
    """محرك السرد الإبداعي التفاعلي. يعتمد على LLMFallback الموجود في NSM
    (Anthropic → Cloudflare → Gemini → OpenRouter → Groq → CKG synthesis)،
    فلا حاجة لأي مزوّد جديد ولا مفاتيح إضافية."""

    def __init__(self, llm_fallback, db_path: str | Path = "memory/fable.db"):
        self.llm = llm_fallback
        self.memory = NarrativeMemory(db_path)
        self.tts = TTSEngine()

    # ── تحويل سيناريو الفيديو (Shorts/Explainer) لصوت سرد فعلي ──────────

    def render_audio(self, script: "ExplainerScript", voice: str = "") -> "ExplainerScript":
        """يملأ audio_bytes لكل مقطع بالسيناريو عبر TTSEngine (Gemini →
        ElevenLabs → Edge TTS → gTTS، أول مزوّد ناجح). يُعدَّل الكائن
        في مكانه ويُعاد أيضاً للراحة في الاستخدام المتسلسل.

        استقرار: كل مقطع يُعاد محاولته حتى 3 مرات إجمالاً (محاولة + إعادتان)
        عند الفشل، قبل الاستسلام لذلك المقطع — الفشل الوحيد لمقطع واحد
        (مثلاً هفوة شبكية عابرة بالبيئة السحابية) لا يعود يُفشِل الفيديو
        بالكامل كما كان يحدث سابقاً (has_audio كانت تتطلب نجاح كل المقاطع
        من محاولة واحدة فقط لكل منها)."""
        if not script.segments:
            return script
        for seg in script.segments:
            last_error = ""
            for attempt in range(3):
                if attempt > 0:
                    time.sleep(1.5 * attempt)  # backoff بسيط قبل إعادة المحاولة
                    logger.info("إعادة محاولة توليد الصوت للمقطع %s (محاولة %d/3)", seg.index, attempt + 1)
                result = self.tts.synthesize(seg.narration, voice=voice)
                if result.ok:
                    seg.audio_bytes = result.audio_bytes
                    seg.audio_format = result.format
                    seg.audio_provider = result.provider.value
                    seg.word_timings = result.word_timings
                    break
                last_error = f"{result.error} (تُجرِّب: {', '.join(result.tried)})"
            else:
                logger.warning("فشل توليد الصوت للمقطع %s بعد 3 محاولات: %s", seg.index, last_error)
        return script

    # ── رندر الفيديو الفعلي (mp4) — يستدعي render_audio تلقائياً لو لزم ──

    def render_video(
        self, script: "ExplainerScript", voice: str = "",
        use_cinematic_backgrounds: bool = False,
        cinematic_provider: str = "higgsfield",
        use_background_music: bool = False,
        music_volume: float = 0.10,
        wan_skip_spaces: Optional[set] = None,
        professional_mode: bool = True,
        cinematic_strategy: str = "hero",
        caption_template: str = "classic_pill",
    ) -> bytes:
        """يبني mp4 فعلي (نص متحرك + صوت سرد) من ExplainerScript. يستدعي
        render_audio() تلقائياً إن لم يكن الصوت مولَّداً بعد. يرجع bytes
        الفيديو النهائي (اكتبها لملف .mp4 مباشرة).

        use_cinematic_backgrounds: اختياري (opt-in)، يستبدل الخلفية
        المتدرّجة بخلفية فيديو سينمائية حقيقية لكل مشهد. cinematic_provider
        يحدّد المصدر: "higgsfield" (يتطلب HIGGSFIELD_API_KEY مدفوع، أسرع
        وأدق) أو "wan_free" (عدة نماذج مفتوحة المصدر مجانية بالكامل —
        LTX-Video ثم Wan2.2 ثم Wan2.1، بالترتيب — عبر مساحات Hugging Face
        مجتمعية "Running on Zero"، تُجرَّب بالتتابع حتى تنجح واحدة. أبطأ
        وأقل ثباتاً من Higgsfield، HF_TOKEN اختياري).
        عند غياب المفتاح/فشل مشهد معيّن بأي من المزوّدين، يتراجع تلقائياً
        للخلفية المتدرّجة لذلك المشهد فقط دون كسر الفيديو.

        use_background_music: اختياري (opt-in)، مُعطَّل افتراضياً — يضيف
        سجادة صوتية محيطية هادئة (مولَّدة داخلياً، بدون ملف/مزوّد خارجي
        وبلا أي إشكال حقوق ملكية) تحت السرد الصوتي. music_volume يضبط
        حجمها النسبي (افتراضياً 0.10 — منخفض جداً حتى لا يطغى على السرد).

        wan_skip_spaces: اختياري — مجموعة أسماء مساحات Wan المجانية
        (مثل {"KingNish/wan2-2-fast"}) معروف عطلها مسبقاً (عادة من
        ai.video_engine.check_wan_free_space_status عبر زر «تحقّق من
        التوفّر» بالواجهة) — تُستبعَد فوراً من أول مشهد بدل انتظار فشلها
        الفعلي (مهلة قد تصل 70-110 ثانية للاكتشاف). لا تأثير له إن
        cinematic_provider != "wan_free".

        caption_template: قالب تصميم النصوص والعناوين (Shorts/TikTok) —
        أحد مفاتيح VideoEngine.CAPTION_TEMPLATES (مثل "neon" أو
        "headline")؛ افتراضياً "classic_pill". أي قيمة غير معروفة يتراجع
        إليها النظام تلقائيًا داخل المحرك دون كسر.

        الاستيراد هنا داخلي (lazy) عمداً: لو moviepy/imageio-ffmpeg غير
        مثبَّتة بعد، باقي NSM (الشات، الوكلاء، إلخ) يستمر يشتغل طبيعياً
        بدون أي كسر — فقط توليد الفيديو نفسه يفشل برسالة واضحة."""
        try:
            from ai.video_engine import VideoEngine, VideoEngineError
        except ImportError as exc:
            raise RuntimeError(
                "توليد الفيديو يحتاج حزمتي moviepy و imageio-ffmpeg. "
                "أضِفهما لـ requirements.txt: moviepy>=2.0, imageio-ffmpeg>=0.4.9"
            ) from exc

        if not script.has_audio:
            self.render_audio(script, voice=voice)
        if not script.has_audio:
            raise VideoEngineError("تعذّر توليد الصوت لكل المقاطع — لا يمكن رندر الفيديو.")

        try:
            # Shorts: الوضع الاحترافي مفعّل افتراضياً (جودة ترميز أعلى +
            # شريط تقدّم + بطاقة ختامية + انتقالات أنعم). يمكن تعطيله صراحة.
            if professional_mode and use_background_music:
                music_volume = max(music_volume, 0.08)
            # تفضيل المسار المجاني إن طُلب "free" أو تُرك فارغاً مع تفعيل السينمائي
            if use_cinematic_backgrounds and cinematic_provider in ("", "free", "auto_free"):
                cinematic_provider = "wan_free"
            return VideoEngine(
                use_cinematic_backgrounds=use_cinematic_backgrounds,
                cinematic_provider=cinematic_provider or "wan_free",
                use_background_music=use_background_music,
                music_volume=music_volume,
                wan_skip_spaces=wan_skip_spaces,
                professional_mode=professional_mode,
                cinematic_strategy=cinematic_strategy,
                caption_template=caption_template or "classic_pill",
            ).render(script)
        except ImportError as exc:
            raise RuntimeError(
                "توليد الفيديو يحتاج حزمتي moviepy و imageio-ffmpeg. "
                "أضِفهما لـ requirements.txt: moviepy>=2.0, imageio-ffmpeg>=0.4.9"
            ) from exc

    # ── 📤 تصدير الفيديو بصيغة TikTok جاهزة للرفع ──────────────────────
    def generate_tiktok_export(
        self, mp4_bytes: bytes,
        max_size_bytes: int = 287 * 1024 * 1024,
    ) -> Dict:
        """يحوّل فيديو Shorts النهائي إلى صيغة TikTok الأمثل (1080×1920
        عمودي · 30fps · H.264 High 4.0 · AAC 128k stereo · faststart)
        مع ضغط تلقائي تحت حد TikTok (287MB iOS / 72MB Android).

        لا يفشل أبداً: عند تعذّر ffmpeg يرجع الأصل كما هو مع توثيق السبب
        في "reason". لا يمسّ أي مسار آخر لتوليد الفيديو.

        يرجع dict: {bytes, reencoded, reason, original_size, exported_size,
                    fits_tiktok} — انظر ai.video_engine.export_tiktok."""
        try:
            from ai.video_engine import export_tiktok
        except ImportError as exc:
            return {
                "bytes": mp4_bytes,
                "reencoded": False,
                "reason": ("تعذّر تحميل وحدة تحويل TikTok — أضِف"
                           " imageio-ffmpeg للبيئة. " + str(exc)),
                "original_size": len(mp4_bytes),
                "exported_size": len(mp4_bytes),
                "fits_tiktok": len(mp4_bytes) <= max_size_bytes,
            }
        return export_tiktok(mp4_bytes, max_size_bytes=max_size_bytes)

    # ── بناء تعليمات النظام لكل جلسة ────────────────────────────────────

    def _system_prompt(self, mode: str, character: str) -> str:
        mode_info = STORY_MODES.get(mode, STORY_MODES[DEFAULT_MODE])
        char_info = CHARACTERS.get(character, CHARACTERS[DEFAULT_CHARACTER])
        return (
            "أنت محرك سرد إبداعي عربي ضمن NSM (Neural Service Mesh). "
            f"وضع القصة الحالي هو «{mode}»: {mode_info['desc']}. "
            f"النبرة المطلوبة: {mode_info['tone']}. "
            f"تروي القصة بأسلوب «{character}»: {char_info['style']}. "
            "اكتب فصلاً واحداً قصيراً (6-10 جمل) بالعربية الفصحى الجميلة.\n\n"
            "قواعد الجودة (مهمة، لا تتجاهلها):\n"
            "- أظهِر الحدث والمشاعر عبر فعل أو حوار أو تفصيل حسي (صوت، رائحة، ملمس، ضوء) "
            "بدل تسميتها مباشرة (\"شعر بالخوف\" ضعيف؛ \"ارتجفت يده على المقبض\" أقوى).\n"
            "- تجنّب الافتتاحيات المكرورة والمستهلكة (\"في يوم من الأيام\"، \"كان يا ما كان\"، "
            "\"في قديم الزمان\") إلا إذا كان الوضع أو الشخصية يستدعيها تحديداً؛ ابدأ من لحظة "
            "أو تفصيل ملموس يشدّ الانتباه فوراً.\n"
            "- تجنّب الحشو والصفات الفائضة (كل كلمة تخدم المشهد أو الشخصية أو الإيقاع)، "
            "ونوّع بنية الجمل بدل تكرار نفس القالب في كل سطر.\n"
            "- حافظ على تماسك الأحداث والأسماء والتفاصيل الثابتة مع الفصول السابقة، "
            "واجعل لكل خيار سابق أثراً واضحاً وملموساً في هذا الفصل بدل تجاهله.\n"
            "- تجنّب أي محتوى غير لائق أو عنيف بتفاصيل صادمة.\n\n"
            "أنهِ الفصل دائماً بثلاثة خيارات مرقّمة (1، 2، 3) لما يفعله القارئ بعد ذلك. "
            "اجعل الخيارات الثلاثة متمايزة فعلياً في اتجاه الحدث والمخاطرة (ليست صياغات "
            "مختلفة لنفس الفعل)، كل خيار جملة قصيرة واحدة. لا تكتب أي شيء بعد الخيار الثالث."
        )

    # ── تحليل الخيارات من نص الرد ────────────────────────────────────────

    @staticmethod
    def _split_choices(raw_text: str) -> tuple[str, List[str]]:
        lines = raw_text.strip().splitlines()
        story_lines: List[str] = []
        choices: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped[:2] in ("1.", "1-", "1)") or stripped.startswith("1️⃣"):
                choices.append(stripped)
            elif stripped[:2] in ("2.", "2-", "2)") or stripped.startswith("2️⃣"):
                choices.append(stripped)
            elif stripped[:2] in ("3.", "3-", "3)") or stripped.startswith("3️⃣"):
                choices.append(stripped)
            elif stripped:
                story_lines.append(stripped)
        if not choices:
            choices = ["تابع الحكاية", "غيّر مسار الأحداث", "أضف شخصية جديدة"]
        return "\n".join(story_lines), choices[:3]

    # ── بدء قصة جديدة ─────────────────────────────────────────────────

    def start_story(
        self,
        mode: str = DEFAULT_MODE,
        character: str = DEFAULT_CHARACTER,
        seed_idea: str = "",
    ) -> FableChapter:
        mode = mode if mode in STORY_MODES else DEFAULT_MODE
        character = character if character in CHARACTERS else DEFAULT_CHARACTER

        session_id = uuid.uuid4().hex[:12]
        self.memory.create_session(session_id, mode, character)

        opening_request = seed_idea.strip() or (
            "ابدأ القصة من فكرة أصيلة مناسبة للوضع المختار — تجنّب أشهر الأفكار "
            "المستهلكة في هذا النوع الأدبي، واختر زاوية أو تفصيلاً غير متوقَّع لبداية القصة."
        )
        sp = self._system_prompt(mode, character)
        self.memory.add_chapter(session_id, "system", sp)

        result = self.llm.generate(opening_request, history=[], system_prompt=sp)
        story_text, choices = self._split_choices(result.text)
        self.memory.add_chapter(session_id, "narration", story_text)

        return FableChapter(
            session_id=session_id,
            text=story_text,
            choices=choices,
            mode=mode,
            character=character,
            provider=getattr(result.provider, "value", str(result.provider)),
            error=result.error,
        )

    # ── متابعة القصة بعد اختيار القارئ ───────────────────────────────────

    def continue_story(
        self,
        session_id: str,
        choice_text: str,
    ) -> FableChapter:
        session = self.memory.get_session(session_id)
        if session is None:
            raise ValueError(f"جلسة قصة غير موجودة: {session_id}")

        mode, character = session["mode"], session["character"]
        sp = self._system_prompt(mode, character)

        history_rows = self.memory.get_history(session_id, limit=12)
        history = []
        last_narration = ""
        pending_reader = None
        for row in history_rows:
            if row["role"] == "narration":
                if pending_reader is not None:
                    history.append((pending_reader, row["content"]))
                    pending_reader = None
                else:
                    last_narration = row["content"]
            elif row["role"] == "reader":
                pending_reader = row["content"]

        self.memory.add_chapter(session_id, "reader", choice_text)
        result = self.llm.generate(choice_text, history=history[-6:], system_prompt=sp)
        story_text, choices = self._split_choices(result.text)
        self.memory.add_chapter(session_id, "narration", story_text)

        return FableChapter(
            session_id=session_id,
            text=story_text,
            choices=choices,
            mode=mode,
            character=character,
            provider=getattr(result.provider, "value", str(result.provider)),
            error=result.error,
        )

    # ── توليد الشعر ───────────────────────────────────────────────────

    def generate_poem(self, topic: str, meter: str = "الطويل") -> FableChapter:
        meter = meter if meter in ARABIC_METERS else "الطويل"
        meter_info = ARABIC_METERS[meter]
        sp = (
            "أنت شاعر عربي فصيح. اكتب قصيدة قصيرة (4-6 أبيات) عن الموضوع المطلوب "
            f"على بحر {meter} (تفعيلته: {meter_info['تفعيلة']} — {meter_info['وصف']}).\n\n"
            "قواعد الجودة (مهمة، لا تتجاهلها):\n"
            "- التزم بحرف رويّ واحد (قافية واحدة) في نهاية كل الأبيات — اختر الروي أولاً "
            "بما يخدم الموضوع، ثم ابنِ الأبيات عليه، لا العكس.\n"
            "- زِن كل بيت فعلياً على تفعيلة البحر المذكور؛ لا تُقحم كلمة أو حرفاً إضافياً "
            "لمجرد إتمام الوزن إن كسر ذلك المعنى أو السلاسة.\n"
            "- تجنّب الصور المستهلكة (القمر/الليل/الدمع كرمز عاطفي جاهز) ما لم تكن جوهر "
            "الموضوع فعلاً؛ ابحث عن صورة أو تشبيه أقل توقّعاً يخدم المعنى تحديداً.\n"
            "- اختر مفردات فصيحة دقيقة لا مجرد مفردات \"شعرية\" جاهزة؛ كل بيت يجب أن "
            "يضيف معنى جديداً لا يكرر ما سبقه بصياغة مختلفة.\n\n"
            "بعد الأبيات، اشرح في سطرين المعنى العام والصورة البلاغية الأبرز في القصيدة."
        )
        result = self.llm.generate(f"اكتب قصيدة عن: {topic}", history=[], system_prompt=sp)
        return FableChapter(
            session_id="poem-" + uuid.uuid4().hex[:8],
            text=result.text.strip(),
            choices=[],
            mode="شعر",
            character=DEFAULT_CHARACTER,
            provider=getattr(result.provider, "value", str(result.provider)),
            error=result.error,
        )

    # ── أوامر سريعة ───────────────────────────────────────────────────

    def quick_command(self, session_id: str, command: str, extra: str = "") -> FableChapter:
        """أوامر سريعة: أنشد بيتاً · صف المكان · أضف حواراً · لخّص"""
        session = self.memory.get_session(session_id)
        if session is None:
            raise ValueError(f"جلسة قصة غير موجودة: {session_id}")

        mode, character = session["mode"], session["character"]
        history_rows = self.memory.get_history(session_id, limit=12)
        narration_so_far = "\n".join(
            r["content"] for r in history_rows if r["role"] == "narration"
        )[-1500:]

        command_prompts = {
            "أنشد بيتاً": (
                "أنشئ بيتاً أو بيتين من الشعر الفصيح يلخّصان جو الأحداث الحالية "
                "في القصة التالية، ثم توقف دون إضافة خيارات جديدة:\n" + narration_so_far
            ),
            "صف المكان": (
                "صِف المكان الحالي في القصة التالية وصفاً حسياً غنياً (5-6 جمل) "
                "دون تطوير أحداث جديدة:\n" + narration_so_far
            ),
            "أضف حواراً": (
                "أضف مقطع حوار قصير (4-6 أسطر) بين شخصيتين من القصة التالية يعمّق "
                "العلاقة بينهما دون تغيير مسار الأحداث:\n" + narration_so_far
            ),
            "لخّص": (
                "لخّص أحداث القصة التالية في فقرة واحدة مختصرة (3-4 جمل):\n" + narration_so_far
            ),
            "أضف عبرة": (
                "اكتب عبرة أو درساً أخلاقياً قصيراً (2-3 جمل) ومناسباً للأطفال يُلخّص "
                "القيمة الإسلامية أو الخُلق الذي تحمله القصة التالية، بأسلوب مباشر "
                "وودود دون اختلاق أي قول أو حدث يُنسب إلى الأنبياء أو الصحابة أو أي "
                "شخصية دينية حقيقية:\n" + narration_so_far
            ),
        }
        prompt = command_prompts.get(command)
        if prompt is None:
            prompt = f"{command} {extra}\n\nسياق القصة:\n{narration_so_far}".strip()

        sp = self._system_prompt(mode, character) + " هذا أمر سريع ولا يتطلب خيارات في النهاية."
        result = self.llm.generate(prompt, history=[], system_prompt=sp)

        return FableChapter(
            session_id=session_id,
            text=result.text.strip(),
            choices=[],
            mode=mode,
            character=character,
            provider=getattr(result.provider, "value", str(result.provider)),
            error=result.error,
        )

    # ── 🎬 مولّد سيناريو وثائقي (بحث + سيناريو مُقسّم لمشاهد) ────────────

    def generate_explainer(self, topic: str, target_minutes: int = 5) -> ExplainerScript:
        """
        يولّد سيناريو وثائقياً مُقسّماً إلى مشاهد لموضوع معيّن — نص السرد
        الصوتي + توجيه مرئي مقترح لكل مشهد. هذا نص فقط (لا فيديو فعلي)؛
        الناتج جاهز لتُغذّى به أداة توليد فيديو خارجية (مثل Higgsfield
        Explainer) يدوياً من قبل المستخدم.
        """
        target_minutes = max(1, min(int(target_minutes), 10))
        n_segments = max(3, target_minutes * 2)  # ~30 ثانية تقريباً لكل مقطع

        sp = (
            "أنت باحث وكاتب سيناريو وثائقي عربي. مهمتك: أخذ موضوع من المستخدم "
            "وإنتاج سيناريو وثائقي 'faceless' (بدون ظهور شخصيات حقيقية) مبني على "
            "معلومات دقيقة ومثيرة للاهتمام، لا تختلق حقائق غير مؤكدة. "
            f"قسّم السيناريو إلى {n_segments} مشاهد تقريباً بحيث يبلغ مجموع مدة "
            f"السرد الصوتي نحو {target_minutes} دقائق (حوالي 130-150 كلمة/دقيقة). "
            "لكل مشهد اكتب بالضبط بهذا التنسيق:\n"
            "### المشهد N\n"
            "السرد: <نص السرد الصوتي بالعربية الفصحى>\n"
            "اللقطة: <وصف مختصر للمشهد المرئي المقترح: مكان/عناصر/زاوية>\n"
            "المدة: <عدد الثواني التقريبي>\n"
            "لا تكتب أي مقدمة أو خاتمة خارج تنسيق المشاهد."
        )
        result = self.llm.generate(
            f"موضوع الوثائقي: {topic}\nالمدة المستهدفة: {target_minutes} دقائق",
            history=[], system_prompt=sp,
        )

        segments = self._parse_explainer_segments(result.text)
        title = topic.strip()

        script = ExplainerScript(
            topic=topic,
            title=title,
            segments=segments,
            provider=getattr(result.provider, "value", str(result.provider)),
            error=result.error,
            format="وثائقي",
        )
        self._save_script_to_history(script, source_excerpt=topic)
        return script

    # ── ⚡ Shorts — فيديو قصير عمودي (~دقيقة واحدة) بسرد صوتي ────────────
    # مستوحى من فكرة NotebookLM: Shorts (تحويل مصدر/موضوع إلى فيديو رأسي
    # قصير بسرد صوتي ورسوم متحركة توضيحية). كما في generate_explainer، هذا
    # نص سيناريو فقط (لا فيديو فعلي فعلي) — NSM لا يملك نموذج توليد فيديو.

    def generate_short(
        self,
        source_text: str,
        target_seconds: int = 60,
        style: str = DEFAULT_SHORTS_STYLE,
        force_offline: bool = False,
    ) -> ExplainerScript:
        """
        يلخّص موضوعاً/مصدراً نصياً في سيناريو فيديو رأسي قصير (~60 ثانية)
        مع نمط إبداعي (حقائق / تحفيزي / تعليمي / قصصي / درامي).
        إن تعذّر LLM يُستخدم مولّد محلي بدون مفاتيح.
        """
        target_seconds = max(20, min(int(target_seconds), 90))
        n_beats = max(4, round(target_seconds / 7))  # لقطة كل ~7 ثوانٍ تقريباً
        style = style if style in SHORTS_STYLES else DEFAULT_SHORTS_STYLE
        style_meta = SHORTS_STYLES[style]

        if force_offline:
            return self._generate_short_offline(source_text, target_seconds, n_beats, style)

        sp = (
            "أنت كاتب سيناريو من أفضل صنّاع محتوى Shorts/Reels في العالم العربي — "
            "أسلوبك يُضاهي إنتاج NotebookLM ومنصات مثل CapCut/Submagic الاحترافية: "
            "فصحى رشيقة تنبض بالحياة، بعيدة كل البعد عن الجفاف الأكاديمي أو الحشو. "
            f"النمط الإبداعي المطلوب: {style} — {style_meta['tone']}. "
            f"ابدأ الخطّاف بأسلوب قريب من: «{style_meta['hook']}». "
            "مهمتك: تحويل النص/الموضوع المُعطى إلى سيناريو فيديو قصير عمودي "
            f"مدته نحو {target_seconds} ثانية، بجودة كتابة استثنائية:\n"
            "- ابدأ بخطّاف (hook) صادم أو مُثير للفضول يُحطّم التوقع خلال أول 3 ثوانٍ "
            "  — رقم مذهل، تناقض غريب، أو سؤال يصعب مقاومته.\n"
            "- كل جملة سرد يجب أن تكون مُبرَّرة ومكثّفة: لا حشو، لا تكرار، كل كلمة "
            "  تدفع الإيقاع للأمام.\n"
            "- إيقاع متنفّس: تناوب بين جمل قصيرة صادمة وجمل وصفية أطول قليلاً.\n"
            "- اختم بجملة ختامية تترك أثراً — مفارقة، دعوة للتفكير، أو صورة حسّية "
            "  قوية تبقى بذهن المشاهد.\n"
            f"قسّم السيناريو إلى نحو {n_beats} لقطات قصيرة جداً. "
            "لكل لقطة اكتب بالضبط بهذا التنسيق:\n"
            "### المشهد N\n"
            "السرد: <جملة سرد واحدة قصيرة وقوية بجودة أدبية عالية>\n"
            "اللقطة: <وصف بصري مُفصَّل ومحدَّد (لا 'شخص يتكلم' أو 'خريطة' فقط) — "
            "العناصر، الألوان، الحركة المقترحة، بما يناسب فيديو عمودي>\n"
            "المدة: <عدد الثواني، عادة 5-8>\n"
            "لا تكتب أي مقدمة أو خاتمة خارج تنسيق اللقطات."
        )
        result = self.llm.generate(
            f"لخّص هذا المصدر/الموضوع في فيديو قصير:\n{source_text}",
            history=[], system_prompt=sp,
        )

        segments = self._parse_explainer_segments(result.text or "")
        provider = getattr(result.provider, "value", str(result.provider))
        err = result.error

        # إن فشل التحليل أو النص فارغ → مولّد محلي إبداعي
        if not segments or (err and not (result.text or "").strip()):
            offline = self._generate_short_offline(
                source_text, target_seconds, n_beats, style
            )
            if err:
                offline.error = f"LLM: {err} · استُخدم المولّد المحلي"
            else:
                offline.error = offline.error or "مولّد محلي (fallback)"
            return offline

        _raw_title = source_text.strip().splitlines()[0] if source_text.strip() else ""
        if len(_raw_title) > 60:
            _cut = _raw_title[:60].rsplit(" ", 1)[0].rstrip("،,؛:.-")
            title = (_cut or _raw_title[:60]) + "…"
        else:
            title = _raw_title or f"شورت · {style}"

        script = ExplainerScript(
            topic=source_text,
            title=title,
            segments=segments,
            provider=provider,
            error=err,
            format="شورت",
        )
        self._save_script_to_history(script, source_excerpt=source_text)
        return script

    def _generate_short_offline(
        self,
        source_text: str,
        target_seconds: int,
        n_beats: int,
        style: str,
    ) -> "ExplainerScript":
        """مولّد Shorts محلي بدون LLM — يعمل دائماً للإبداع السريع."""
        style = style if style in SHORTS_STYLES else DEFAULT_SHORTS_STYLE
        meta = SHORTS_STYLES[style]
        text = (source_text or "").strip() or "موضوع عام للإبداع"
        # تقسيم إلى جمل
        parts = re.split(r"[.\n!?؟]+", text)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            parts = [text]
        # توسيع/تقليص لعدد اللقطات
        while len(parts) < n_beats:
            parts.append(parts[len(parts) % max(1, len(parts))])
        parts = parts[:n_beats]
        sec_each = max(4, target_seconds // max(1, len(parts)))

        hook_templates = {
            "حقائق سريعة": "هل تعلم؟ {idea}",
            "تحفيزي": "توقّف لحظة — {idea}",
            "تعليمي": "في ثوانٍ ستفهم: {idea}",
            "قصصي": "بدأت الحكاية عندما {idea}",
            "درامي": "لم يكن أحد يتوقع: {idea}",
        }
        visual_templates = {
            "حقائق سريعة": "نص كبير في منتصف الشاشة + أرقام متحركة وخلفيّة متدرجة داكنة",
            "تحفيزي": "كاميرا بطيئة على أفق مضيء + كلمات تظهر بإيقاع قوي",
            "تعليمي": "رسوم مبسّطة وخطوات مرقّمة تظهر تباعاً بخلفية هادئة",
            "قصصي": "لقطة سينمائية لشخصية ظليّة تتحرك في فضاء رمزي",
            "درامي": "تباين إضاءة حادّ، حركة كاميرا بطيئة، نص يظهر كهمس ثم ينفجر",
        }
        close_templates = {
            "حقائق سريعة": "والآن… شاركه قبل أن يختفِي من خلاصتك.",
            "تحفيزي": "الخطوة التالية عليك أنت — ابدأ اليوم.",
            "تعليمي": "هذا جوهر الفكرة — طبّقها مرة واحدة وستثبّت.",
            "قصصي": "وهكذا بقيت الحكاية… وما زال السؤال معلّقاً.",
            "درامي": "الصمت بعدها أبلغ من أي تعليق.",
        }

        segments: List[ExplainerSegment] = []
        for i, idea in enumerate(parts):
            idea_short = idea[:120]
            if i == 0:
                narr = hook_templates.get(style, "{idea}").format(idea=idea_short)
            elif i == len(parts) - 1:
                narr = f"{idea_short}. {close_templates.get(style, '')}"
            else:
                narr = idea_short
            segments.append(
                ExplainerSegment(
                    index=i + 1,
                    narration=narr,
                    visual_notes=visual_templates.get(style, "نص حركي عمودي مع خلفية متدرجة"),
                    est_seconds=sec_each,
                )
            )

        _raw = text.splitlines()[0] if text else "شورت"
        title = (_raw[:50] + "…") if len(_raw) > 50 else _raw
        title = f"{meta.get('emoji', '⚡')} {title}"
        script = ExplainerScript(
            topic=source_text,
            title=title,
            segments=segments,
            provider="offline-creative",
            error=None,
            format="شورت",
        )
        self._save_script_to_history(script, source_excerpt=source_text)
        return script

    # ── 🆕 تعديل اللقطات يدوياً + وصف وهاشتاجات النشر الاجتماعي ────────
    def rebuild_short_segments(
        self,
        segments_json: str,
        original_segments: Optional[List["ExplainerSegment"]] = None,
    ) -> List["ExplainerSegment"]:
        """يعيد بناء قائمة اللقطات من نص JSON عدّله المستخدم في الواجهة.
        يحافظ على الصوت المولَّد لكل لقطة (audio_bytes / word_timings)
        من `original_segments` عند التطابق بالرقم — فلا يحتاج المستخدم
        إعادة توليد الصوت بعد تعديل نصي فقط. يرفع ValueError عند فشل
        التحليل أو عدم وجود لقطات صالحة."""
        try:
            data = json.loads(segments_json or "[]")
            if isinstance(data, dict) and "segments" in data:
                data = data["segments"]
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"نص اللقطات غير صالح كـJSON: {exc}") from exc
        if not isinstance(data, list):
            raise ValueError("يتوقع JSON قائمة لقطات.")
        rebuilt: List["ExplainerSegment"] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"اللقطة {i + 1} يجب أن تكون كائناً (dict).")
            narration = str(item.get("narration") or "").strip()
            if not narration:
                continue
            seg = ExplainerSegment(
                index=i + 1,
                narration=narration,
                visual_notes=str(item.get("visual_notes") or ""),
                est_seconds=max(2, int(item.get("est_seconds") or 5)),
            )
            if original_segments:
                for orig in original_segments:
                    if orig.index == seg.index:
                        seg.audio_bytes = orig.audio_bytes
                        seg.audio_format = orig.audio_format
                        seg.audio_provider = orig.audio_provider
                        seg.word_timings = orig.word_timings
                        break
            rebuilt.append(seg)
        if not rebuilt:
            raise ValueError("لا توجد لقطات صالحة بعد التعديل (كل سرد فارغ).")
        return rebuilt

    def generate_short_social_description(
        self,
        script: "ExplainerScript",
    ) -> Dict[str, str]:
        """يولّد عنوانَ نشر + وصف + هاشتاجات جاهزة لرفع الفيديو على
        YouTube Shorts / TikTok. يعمل مع أي مزوّد متاح، وعند فشل LLM
        يرجع وصفاً احتياطياً مولَّداً محلياً بلا أي مفتاح."""
        narration = script.full_narration or ""
        try:
            sp = (
                "اكتب وصفاً قصيراً جذاباً لفيديو قصير عربي ثم هاشتاجات "
                "مرتفعة الوصول. أجب بصيغة JSON فقط: "
                '{"title": "...", "description": "...", "hashtags": ["#ا","#b"]}'
            )
            result = self.llm.generate(
                f"موضوع الفيديو:\n{narration[:1500]}", history=[], system_prompt=sp,
            )
            payload = json.loads(result.text or "{}")
            return {
                "title": str(payload.get("title") or script.title or ""),
                "description": str(payload.get("description") or ""),
                "hashtags": " ".join(payload.get("hashtags") or []),
                "provider": getattr(result.provider, "value", str(result.provider)),
            }
        except Exception as exc:  # فشل LLM → وصف احتياطي مولّد محلياً
            return {
                "title": script.title or "شورت · NSM",
                "description": (narration[:90] or script.title or "فيديو قصير").strip(),
                "hashtags": "#شورتس #فيديو_قصير #arabic #shorts #reels",
                "provider": "محلي",
                "fallback_error": f"تعذّر توليد الوصف بالـLLM: {exc}",
            }

    def _save_script_to_history(self, script: "ExplainerScript", source_excerpt: str = "") -> None:
        """يحفظ سيناريو Shorts/وثائقي في shorts_history — best-effort، لا
        يرفع استثناءً أبداً (فشل الحفظ لا يجوز أن يُفشل التوليد نفسه)."""
        try:
            segs = [
                {"index": s.index, "narration": s.narration,
                 "visual_notes": s.visual_notes, "est_seconds": s.est_seconds}
                for s in script.segments
            ]
            self.memory.save_short(
                title=script.title, format_=script.format, segments=segs,
                total_seconds=script.total_seconds, source_excerpt=source_excerpt,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"FableEngine: فشل حفظ السيناريو بالمكتبة: {e}")

    @staticmethod
    def _parse_explainer_segments(raw_text: str) -> List[ExplainerSegment]:
        segments: List[ExplainerSegment] = []
        blocks = re.split(r"###\s*المشهد\s*\d+", raw_text)
        idx = 0
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            idx += 1
            narration_m = re.search(r"السرد\s*:\s*(.+?)(?=\n\s*اللقطة\s*:|\Z)", block, re.S)
            visual_m    = re.search(r"اللقطة\s*:\s*(.+?)(?=\n\s*المدة\s*:|\Z)", block, re.S)
            duration_m  = re.search(r"المدة\s*:\s*(\d+)", block)

            narration = narration_m.group(1).strip() if narration_m else block
            visual    = visual_m.group(1).strip() if visual_m else ""
            duration  = int(duration_m.group(1)) if duration_m else 30

            segments.append(ExplainerSegment(
                index=idx, narration=narration, visual_notes=visual, est_seconds=duration,
            ))

        if not segments:
            # في حال لم يلتزم النموذج بالتنسيق، نعيد النص كاملاً كمقطع واحد
            segments = [ExplainerSegment(index=1, narration=raw_text.strip(),
                                          visual_notes="", est_seconds=60)]
        return segments
