"""
Quran Grounding — تثبيت الاستشهادات القرآنية على النص الموثوق
================================================================
المشكلة التي يعالجها هذا الملف:
  مسار المحادثة الحالي (nsm_chat.py) يرسل سؤال المستخدم مباشرة إلى
  LLM خارجي دون أي تحقق، رغم وجود نص القرآن الكريم الكامل والموثّق
  (knowledge_sources/quran/data/quran.json) داخل المشروع نفسه.
  نتيجة ذلك: أي آية يذكرها النموذج تأتي من "ذاكرته" الاحتمالية، بلا
  ضمان لمطابقتها للنص الفعلي — وهذا خطر مباشر في تطبيق متخصص
  بالمعرفة الإسلامية (نص مُختلَق أو معزوّ لسورة/رقم خطأ).

  هذا الملف حل خفيف الوزن (بدون تدريب أو TF-IDF أو تبعيات إضافية):
  يكتشف من نص سؤال المستخدم إشارات لآيات معيّنة (رقمياً أو باسم
  السورة) أو تشابهاً لفظياً مع نص آية، ثم يبني "سياق تحقق" من النص
  الأصلي الموثوق (trust_score=1.0) ليُرفَق مع سؤال المستخدم قبل
  إرساله للـLLM، مع تعليمات صريحة بعدم تجاوز هذا النص عند الاقتباس.

  لا يُعدَّل نص القرآن نفسه أبداً هنا (قراءة فقط) — يتوافق مع مبدأ
  المشروع القائم في quran_source.py (raw_content محمي/read-only).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

_DATA_PATH = Path(__file__).parent.parent / "knowledge_sources" / "quran" / "data" / "quran.json"

# عربي-هندي أرقام → لدعم "٢:٢٥٥" بجانب "2:255"
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# نطاق التشكيل (الحركات) في اليونيكود العربي — يُزال فقط لأغراض
# المطابقة الداخلية، لا يمس النص الأصلي المخزَّن أو المعروض.
_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u08D4-\u08E1\u08D3-\u08E5]")


def _strip_tashkeel(text: str) -> str:
    t = _TASHKEEL.sub("", text)
    t = t.replace("ـ", "")  # تطويل
    # توحيد صور الألف/الهمزة الشائعة لتحسين المطابقة
    t = re.sub(r"[إأآٱا]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه")
    return t


class QuranIndex:
    """فهرس خفيف في الذاكرة لنص القرآن — يُحمَّل مرة واحدة فقط (singleton)."""

    _instance: Optional["QuranIndex"] = None

    def __init__(self) -> None:
        self.surahs: List[dict] = []          # كما وردت في quran.json
        self.surah_name_to_num: dict[str, int] = {}
        # فهرس مسطّح: (surah_num, ayah_num, raw_text, normalized_text)
        self._flat: List[Tuple[int, int, str, str]] = []
        self._loaded = False
        self._load()

    @classmethod
    def get(cls) -> "QuranIndex":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        if not _DATA_PATH.exists():
            return
        try:
            with open(_DATA_PATH, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        raw_surahs = data.get("surahs", [])
        surahs = raw_surahs.get("references", []) if isinstance(raw_surahs, dict) else raw_surahs

        for surah in surahs:
            num = surah.get("number", 0)
            name = (surah.get("name") or "").strip()
            self.surahs.append(surah)
            if name:
                # الاسم كما ورد، وبصيغة مبسّطة بلا "سُورَةُ" وبلا تشكيل
                self.surah_name_to_num[_strip_tashkeel(name)] = num
                simple = _strip_tashkeel(name).replace("سوره", "").strip()
                if simple:
                    self.surah_name_to_num[simple] = num

            for ayah in surah.get("ayahs", []):
                text = (ayah.get("text") or "").strip()
                if not text:
                    continue
                self._flat.append((num, ayah.get("numberInSurah", 0), text, _strip_tashkeel(text)))

        self._loaded = True

    @property
    def available(self) -> bool:
        return self._loaded and bool(self._flat)

    def get_ayah(self, surah_num: int, ayah_num: int) -> Optional[Tuple[str, str]]:
        """يُرجع (النص، اسم السورة) لآية محددة، أو None إن لم توجد."""
        for surah in self.surahs:
            if surah.get("number") == surah_num:
                surah_name = (surah.get("name") or "").strip()
                for ayah in surah.get("ayahs", []):
                    if ayah.get("numberInSurah") == ayah_num:
                        return (ayah.get("text") or "").strip(), surah_name
        return None

    def search_keywords(self, query: str, max_results: int = 3, min_word_len: int = 4) -> List[Tuple[int, int, str]]:
        """بحث لفظي بسيط عن آيات تحتوي كلمات مميزة من السؤال.
        يُرجع قائمة (surah_num, ayah_num, text) مرتبة تنازلياً بعدد الكلمات المطابقة.
        """
        if not self.available:
            return []

        norm_query = _strip_tashkeel(query)
        words = [w for w in re.split(r"\W+", norm_query) if len(w) >= min_word_len]
        words = list(dict.fromkeys(words))  # إزالة التكرار مع الحفاظ على الترتيب
        if not words:
            return []

        scored: List[Tuple[int, int, int, str]] = []  # (score, surah, ayah, text)
        for surah_num, ayah_num, raw_text, norm_text in self._flat:
            matched = [w for w in words if w in norm_text]
            score = len(matched)
            # مطابقتان فأكثر تكفي، أو كلمة مميّزة واحدة طويلة (>=5 أحرف)
            # لتفادي إشعال البحث على كلمات قصيرة شائعة.
            if score >= 2 or (score == 1 and max((len(w) for w in matched), default=0) >= 5):
                scored.append((score, surah_num, ayah_num, raw_text))

        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        return [(s, a, t) for _, s, a, t in scored[:max_results]]


# ── إشارات مرجعية صريحة: "2:255" أو "٢:٢٥٥" أو "سورة البقرة آية 255" ──
_NUMERIC_REF = re.compile(r"(?<!\d)(\d{1,3})\s*[:٬,]\s*(\d{1,3})(?!\d)")
_NAMED_REF = re.compile(
    r"سور[ةه]?\s+([^\s0-9]{2,15})\s*(?:،|,)?\s*(?:آية|الآية|اية)\s*(\d{1,3})"
)


def _extract_explicit_refs(text: str) -> List[Tuple[int, int]]:
    t = text.translate(_ARABIC_DIGITS)
    refs: List[Tuple[int, int]] = []

    for m in _NUMERIC_REF.finditer(t):
        s, a = int(m.group(1)), int(m.group(2))
        if 1 <= s <= 114 and 1 <= a <= 286:
            refs.append((s, a))

    for m in _NAMED_REF.finditer(t):
        name_raw, ayah_num = m.group(1), int(m.group(2))
        idx = QuranIndex.get()
        norm_name = _strip_tashkeel(name_raw.strip())
        surah_num = idx.surah_name_to_num.get(norm_name)
        if surah_num:
            refs.append((surah_num, ayah_num))

    return refs


def build_grounding_context(user_input: str) -> Optional[str]:
    """يبني كتلة سياق نصي بآيات موثوقة ذات صلة بسؤال المستخدم، أو
    None إن لم يجد أي صلة (فلا داعي لإضافة أي شيء في الحالة العامة).
    """
    idx = QuranIndex.get()
    if not idx.available:
        return None

    found: List[Tuple[int, int, str]] = []

    # 1) إشارات صريحة (رقم سورة:آية أو اسم سورة + رقم آية)
    for surah_num, ayah_num in _extract_explicit_refs(user_input):
        result = idx.get_ayah(surah_num, ayah_num)
        if result:
            text, _name = result
            found.append((surah_num, ayah_num, text))

    # 2) تشابه لفظي — فقط إن لم توجد إشارة صريحة، ومتى بدا السؤال متعلقاً بالقرآن
    if not found and re.search(r"آية|القرآن|سورة|قرآن", user_input):
        found.extend(idx.search_keywords(user_input, max_results=3))

    if not found:
        return None

    # إزالة التكرار مع الحفاظ على الترتيب
    seen = set()
    unique: List[Tuple[int, int, str]] = []
    for s, a, t in found:
        key = (s, a)
        if key not in seen:
            seen.add(key)
            unique.append((s, a, t))

    lines = [
        "[سياق موثوق من القرآن الكريم — نص أصلي من المصحف، وليس من ذاكرة النموذج]",
        "استخدم هذا النص حصراً عند اقتباس أي آية أدناه بلفظها؛ إن كانت إجابتك",
        "تحتاج آية غير مذكورة هنا فلا تجزم بنصها الحرفي، بل أشر إلى ذلك بوضوح.",
        "",
    ]
    for surah_num, ayah_num, text in unique:
        lines.append(f"({surah_num}:{ayah_num}) {text}")

    return "\n".join(lines)
