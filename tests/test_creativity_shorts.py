"""test_creativity_shorts.py — محاكاة بدون مفاتيح API حقيقية.

يختبر:
1) rebuild_short_segments (حفظ الصوت + أخطاء JSON)
2) generate_short_social_description (fallback محلي عند غياب LLM)
3) منطق عرض جلسات المحفوظات في الواجهة (FableChapter من استئناف)
"""
from __future__ import annotations

import json
import sys
import sqlite3
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"✅ [{PASS}] {name}")
    else:
        FAIL += 1
        print(f"❌ [{FAIL}] {name} {detail}")


def main() -> int:
    from ai.fable_engine import FableEngine, ExplainerScript, ExplainerSegment
    from ai.llm_fallback import LLMFallback

    # محرك بلا مزوّد LLM حقيقي (fallback فارغ) — لا مفاتيح API
    with tempfile.TemporaryDirectory() as tmp:
        engine = FableEngine(llm_fallback=LLMFallback(), db_path=Path(tmp) / "fable.db")

        # ── 1) rebuild_short_segments: أساسيات ──
        segs = [
            ExplainerSegment(index=1, narration="لقطة أولى تجريبية",
                             visual_notes="صورة فضاء", est_seconds=5),
            ExplainerSegment(index=2, narration="لقطة ثانية عن الطاقة",
                             visual_notes="شمس", est_seconds=6),
        ]
        for s in segs:
            s.audio_bytes = b"FAKE-AUDIO"
            s.word_timings = [0.0, 0.5]

        json_text = json.dumps([
            {"narration": "لقطة أولى معدّلة", "visual_notes": "كوكب", "est_seconds": 5},
            {"narration": "لقطة ثانية عن الطاقة الشمسية", "visual_notes": "شمس", "est_seconds": 6},
        ], ensure_ascii=False)

        rebuilt = engine.rebuild_short_segments(json_text, original_segments=segs)
        check("rebuild: عدد اللقطات", len(rebuilt) == 2)
        check("rebuild: التعديل النصي طُبّق", rebuilt[0].narration == "لقطة أولى معدّلة")
        check("rebuild: الصوت محفوظ للقطات المطابقة", rebuilt[0].audio_bytes == b"FAKE-AUDIO")
        check("rebuild: توقيت الكلمات محفوظ", rebuilt[1].word_timings == [0.0, 0.5])

        # ── 2) rebuild: أخطاء ──
        failed = False
        try:
            engine.rebuild_short_segments("ليس json")
        except ValueError:
            failed = True
        check("rebuild: JSON خاطئ يرفع ValueError", failed)

        failed = False
        try:
            engine.rebuild_short_segments(json.dumps([{"narration": "  "}]))
        except ValueError:
            failed = True
        check("rebuild: سرد فارغ يُستبعد ويرفع ValueError عند الفراغ", failed)

        rebuilt_min = engine.rebuild_short_segments(
            '{"segments": [{"narration": "لقطة", "est_seconds": 3}]}')
        check("rebuild: يغلف dict segments", len(rebuilt_min) == 1 and rebuilt_min[0].narration == "لقطة")
        check("rebuild: مدة أدنى 2 ثانية", rebuilt_min[0].est_seconds >= 2)

        # ── 3) generate_short_social_description: fallback بلا LLM ──
        script = ExplainerScript(
            topic="حقائق الفضاء", title="5 حقائق عن الفضاء",
            segments=segs, format="شورت",
        )
        card = engine.generate_short_social_description(script)
        check("fallback: عنوان موجود", bool(card.get("title")))
        check("fallback: هاشتاجات عربية", "#shorts" in card.get("hashtags", ""))
        check("fallback: provider=محلي", card.get("provider") == "محلي")
        check("fallback: سبب الفشل موثّق", "fallback_error" in card)

        # ── 4) حفظ جلسات القصة واستئنافها (ما تبنيه الواجهة) ──
        sid = "testresume123"
        engine.memory.create_session(sid, "مغامرة", "شهرزاد")
        engine.memory.add_chapter(sid, "system", "sp")
        engine.memory.add_chapter(sid, "narration", "في قديم الزمان كان هناك تاجر")
        engine.memory.add_chapter(sid, "reader", "اختار الذهاب للصحراء")
        engine.memory.add_chapter(sid, "narration", "سار التاجر في الصحراء حتى وصل")

        recent = engine.memory.list_recent_sessions(limit=10)
        check("list_recent_sessions: الجلسة محفوظة", any(r["session_id"] == sid for r in recent))

        hist = engine.memory.get_history(sid, limit=200)
        narr_rows = [r for r in hist if r["role"] == "narration"]
        check("استئناف: آخر فصل سردي", narr_rows[-1]["content"] == "سار التاجر في الصحراء حتى وصل")

        # ── 5) FableChapter من الاستئناف (كما تبنيه الواجهة) ──
        from ai.fable_engine import FableChapter
        chapter = FableChapter(
            session_id=sid, text=narr_rows[-1]["content"],
            mode="مغامرة", character="شهرزاد", provider="مستأنفة من الحفظ",
        )
        check("FableChapter استئناف: نص صحيح", chapter.text == narr_rows[-1]["content"])
        check("FableChapter استئناف: session_id", chapter.session_id == sid)

        # حذف الجلسة (كما تفعله الواجهة)
        engine.memory.delete_session(sid)
        after = engine.memory.list_recent_sessions(limit=10)
        check("delete_session: حُذفت فعلياً", not any(r["session_id"] == sid for r in after))

        print(f"\n{'='*50}\nالنتيجة: {PASS} ناجحة / {FAIL} فاشلة")
        return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
