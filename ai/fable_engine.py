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
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ai.tts_engine import TTSEngine

logger = logging.getLogger("FableEngine")


# ══════════════════════════════════════════════════════════════════════════
# أوضاع القصة — 6 أصلية + 4 إضافية (الأكثر طلباً حسب اتجاهات البحث العالمية
# لأدوات السرد التفاعلي بالذكاء الاصطناعي: رعب، خيال علمي، رومانسية، أطفال)
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
}

DEFAULT_MODE = "إبداع حر"


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
        في مكانه ويُعاد أيضاً للراحة في الاستخدام المتسلسل."""
        if not script.segments:
            return script
        for seg in script.segments:
            result = self.tts.synthesize(seg.narration, voice=voice)
            if result.ok:
                seg.audio_bytes = result.audio_bytes
                seg.audio_format = result.format
                seg.audio_provider = result.provider.value
            else:
                logger.warning(
                    "فشل توليد الصوت للمقطع %s: %s (تُجرِّب: %s)",
                    seg.index, result.error, ", ".join(result.tried),
                )
        return script

    # ── رندر الفيديو الفعلي (mp4) — يستدعي render_audio تلقائياً لو لزم ──

    def render_video(self, script: "ExplainerScript", voice: str = "") -> bytes:
        """يبني mp4 فعلي (نص متحرك + صوت سرد) من ExplainerScript. يستدعي
        render_audio() تلقائياً إن لم يكن الصوت مولَّداً بعد. يرجع bytes
        الفيديو النهائي (اكتبها لملف .mp4 مباشرة).

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
            return VideoEngine().render(script)
        except ImportError as exc:
            raise RuntimeError(
                "توليد الفيديو يحتاج حزمتي moviepy و imageio-ffmpeg. "
                "أضِفهما لـ requirements.txt: moviepy>=2.0, imageio-ffmpeg>=0.4.9"
            ) from exc

    # ── بناء تعليمات النظام لكل جلسة ────────────────────────────────────

    def _system_prompt(self, mode: str, character: str) -> str:
        mode_info = STORY_MODES.get(mode, STORY_MODES[DEFAULT_MODE])
        char_info = CHARACTERS.get(character, CHARACTERS[DEFAULT_CHARACTER])
        return (
            "أنت محرك سرد إبداعي عربي ضمن NSM (Neural Service Mesh). "
            f"وضع القصة الحالي هو «{mode}»: {mode_info['desc']}. "
            f"النبرة المطلوبة: {mode_info['tone']}. "
            f"تروي القصة بأسلوب «{character}»: {char_info['style']}. "
            "اكتب فصلاً واحداً قصيراً (6-10 جمل) بالعربية الفصحى الجميلة، "
            "حافظ على تماسك الأحداث والشخصيات مع الفصول السابقة، "
            "وتجنّب أي محتوى غير لائق أو عنيف بتفاصيل صادمة. "
            "أنهِ الفصل دائماً بثلاثة خيارات مرقّمة (1، 2، 3) لما يفعله القارئ بعد ذلك، "
            "كل خيار جملة قصيرة واحدة. لا تكتب أي شيء بعد الخيار الثالث."
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

        opening_request = seed_idea.strip() or "ابدأ القصة من فكرة مناسبة للوضع المختار."
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
            f"على بحر {meter} (تفعيلته: {meter_info['تفعيلة']} — {meter_info['وصف']}). "
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

        return ExplainerScript(
            topic=topic,
            title=title,
            segments=segments,
            provider=getattr(result.provider, "value", str(result.provider)),
            error=result.error,
            format="وثائقي",
        )

    # ── ⚡ Shorts — فيديو قصير عمودي (~دقيقة واحدة) بسرد صوتي ────────────
    # مستوحى من فكرة NotebookLM: Shorts (تحويل مصدر/موضوع إلى فيديو رأسي
    # قصير بسرد صوتي ورسوم متحركة توضيحية). كما في generate_explainer، هذا
    # نص سيناريو فقط (لا فيديو فعلي فعلي) — NSM لا يملك نموذج توليد فيديو.

    def generate_short(self, source_text: str, target_seconds: int = 60) -> ExplainerScript:
        """
        يلخّص موضوعاً/مصدراً نصياً في سيناريو فيديو رأسي قصير (~60 ثانية
        افتراضياً) مقسّم إلى 'لقطات' سريعة، مع سرد صوتي مكثّف ووصف رسوم
        متحركة توضيحية مقترحة لكل لقطة (بديل لصور الفيديو الوثائقي الأطول).
        """
        target_seconds = max(20, min(int(target_seconds), 90))
        n_beats = max(4, round(target_seconds / 7))  # لقطة كل ~7 ثوانٍ تقريباً

        sp = (
            "أنت كاتب سيناريو لفيديوهات قصيرة عمودية (Shorts/Reels) بالعربية "
            "الفصحى المبسّطة. مهمتك: تلخيص النص/الموضوع المُعطى في فيديو قصير "
            f"مدته نحو {target_seconds} ثانية فقط، بأسلوب سريع وجذاب يبدأ بخطّاف "
            "(hook) قوي في أول 3 ثوانٍ. "
            f"قسّم السيناريو إلى نحو {n_beats} لقطات قصيرة جداً. "
            "لكل لقطة اكتب بالضبط بهذا التنسيق:\n"
            "### المشهد N\n"
            "السرد: <جملة سرد واحدة قصيرة وقوية>\n"
            "اللقطة: <وصف رسم متحرك/رسم توضيحي مقترح للشاشة، بسيط ومناسب لفيديو عمودي>\n"
            "المدة: <عدد الثواني، عادة 5-8>\n"
            "لا تكتب أي مقدمة أو خاتمة خارج تنسيق اللقطات."
        )
        result = self.llm.generate(
            f"لخّص هذا المصدر/الموضوع في فيديو قصير:\n{source_text}",
            history=[], system_prompt=sp,
        )

        segments = self._parse_explainer_segments(result.text)
        # ضبط المدة الإجمالية لتقارب target_seconds إن أمكن (بدون تلاعب بالنص)
        title = source_text.strip().splitlines()[0][:60] if source_text.strip() else "فيديو قصير"

        return ExplainerScript(
            topic=source_text,
            title=title,
            segments=segments,
            provider=getattr(result.provider, "value", str(result.provider)),
            error=result.error,
            format="شورت",
        )

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
