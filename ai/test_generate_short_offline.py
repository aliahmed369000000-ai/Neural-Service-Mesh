"""
اختبار FableEngine._generate_short_offline() — المولّد المحلي الاحتياطي
لسيناريوهات Shorts (يعمل بدون أي LLM/مفتاح API، يُستخدم كـfallback عند
فشل كل مزوّدي LLM الحيّين أو عبر force_offline=True).

الخلل المُصلَح: `parts[len(parts) % max(1, len(parts))]` يُعطي دائماً 0
(أي عدد % نفسه = صفر) فكان كل عنصر إضافي أثناء توسيع القائمة لعدد
اللقطات المطلوب = parts[0] حرفياً — أي أن أي مصدر بجملتين أو أكثر
(لكن أقل من n_beats) كان يُنتج سيناريو يكرر الجملة الأولى فقط لبقية
اللقطات بدل الدوران على كل الجمل الأصلية بالتناوب.
"""
import unittest

from ai.fable_engine import FableEngine


def _make_engine():
    fe = FableEngine.__new__(FableEngine)  # تجاوز __init__ — لا حاجة لـllm/قاعدة بيانات هنا
    fe.llm = None
    fe._save_script_to_history = lambda *a, **kw: None  # تعطيل الحفظ لقاعدة البيانات في الاختبار
    return fe


class GenerateShortOfflineTests(unittest.TestCase):
    def test_cycles_through_all_sentences_instead_of_repeating_first_only(self):
        fe = _make_engine()
        script = fe._generate_short_offline(
            "الذكاء الاصطناعي يغيّر العالم. الشركات تتسابق لتطوير نماذج أقوى",
            target_seconds=60, n_beats=9, style="حقائق سريعة",
        )
        narrations = [seg.narration for seg in script.segments]
        self.assertEqual(len(narrations), 9)
        # الجملة الثانية يجب أن تظهر في أكثر من لقطة (تناوب حقيقي)،
        # وليس فقط في اللقطة الثانية كنتيجة عرضية.
        second_sentence_appearances = sum(
            1 for n in narrations if "الشركات تتسابق" in n
        )
        self.assertGreaterEqual(
            second_sentence_appearances, 3,
            "الجملة الثانية يجب أن تتكرر بالتناوب مع الأولى، لا أن تظهر مرة واحدة فقط",
        )
        # لا يجوز أن تكون كل اللقطات (عدا الأولى) نسخة طبق الأصل من الأولى
        first_sentence_only_count = sum(
            1 for n in narrations if n.strip() == "الذكاء الاصطناعي يغيّر العالم"
        )
        self.assertLess(
            first_sentence_only_count, 6,
            "الخلل القديم: تكرار الجملة الأولى حرفياً في أغلب اللقطات",
        )

    def test_single_sentence_source_still_works(self):
        fe = _make_engine()
        script = fe._generate_short_offline(
            "فكرة واحدة فقط بلا نقاط فصل", target_seconds=40, n_beats=6, style="تحفيزي",
        )
        self.assertEqual(len(script.segments), 6)
        self.assertTrue(all(seg.narration.strip() for seg in script.segments))

    def test_empty_source_falls_back_to_generic_topic(self):
        fe = _make_engine()
        script = fe._generate_short_offline(
            "", target_seconds=30, n_beats=5, style="درامي",
        )
        self.assertEqual(len(script.segments), 5)
        self.assertTrue(all(seg.narration.strip() for seg in script.segments))

    def test_more_sentences_than_beats_truncates_correctly(self):
        fe = _make_engine()
        script = fe._generate_short_offline(
            "واحد. اثنان. ثلاثة. أربعة. خمسة. ستة. سبعة.",
            target_seconds=30, n_beats=4, style="تعليمي",
        )
        self.assertEqual(len(script.segments), 4)


if __name__ == "__main__":
    unittest.main()
