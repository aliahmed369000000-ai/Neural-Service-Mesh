"""
اختبارات ai/nsm_answer_verifier.py

كل هذه الاختبارات لا تحتاج مفتاح API ولا مكتبة deepeval فعلياً مثبَّتة —
نفس فلسفة CI الخفيفة الحالية (.github/workflows/tests.yml يثبّت pytest/
numpy/requests فقط). نختبر:
  1. _build_retrieval_context — منطق نقي، بلا أي اعتمادية خارجية.
  2. verify_answer_faithfulness — مسارات التدهور الآمن (deepeval غير
     مثبَّت، لا مفتاح API، لا سياق مسترجَع، لا نص إجابة) عبر monkeypatch
     بدل تثبيت deepeval الفعلي (اعتماد ثقيل اختياري — requirements-verifier.txt).
"""
import pytest

from ai import nsm_answer_verifier as verifier


# ══════════════════════════════════════════════════════════════════════════
# _build_retrieval_context — منطق نقي بلا اعتماديات
# ══════════════════════════════════════════════════════════════════════════
class TestBuildRetrievalContext:
    def test_includes_verse_text_and_reference(self):
        qa_result = {
            "verses": [
                {"surah": 2, "ayah": 183, "text": "يا أيها الذين آمنوا كتب عليكم الصيام"},
            ],
            "primary_concepts": [],
        }
        context = verifier._build_retrieval_context(qa_result)
        assert len(context) == 1
        assert "سورة 2 آية 183" in context[0]
        assert "الصيام" in context[0]

    def test_skips_verses_without_text(self):
        qa_result = {"verses": [{"surah": 2, "ayah": 183, "text": ""}], "primary_concepts": []}
        assert verifier._build_retrieval_context(qa_result) == []

    def test_includes_primary_concepts_as_extra_context(self):
        qa_result = {
            "verses": [],
            "primary_concepts": [{"name": "الصبر"}, {"name": "التقوى"}],
        }
        context = verifier._build_retrieval_context(qa_result)
        assert len(context) == 1
        assert "الصبر" in context[0] and "التقوى" in context[0]

    def test_empty_result_gives_empty_context(self):
        assert verifier._build_retrieval_context({}) == []


# ══════════════════════════════════════════════════════════════════════════
# verify_answer_faithfulness — مسارات التدهور الآمن (بلا استثناء أبداً)
# ══════════════════════════════════════════════════════════════════════════
class TestVerifyAnswerFaithfulnessGracefulDegradation:
    def test_lexical_fallback_when_deepeval_not_importable(self, monkeypatch):
        monkeypatch.setattr(verifier, "_deepeval_importable", lambda: False)
        report = verifier.verify_answer_faithfulness(
            "ما حكم الصبر؟",
            {
                "summary": "الصبر مطلوب في الصيام",
                "verses": [{"surah": 2, "ayah": 183, "text": "يا أيها الذين آمنوا كتب عليكم الصيام"}],
            },
        )
        assert report["available"] is True
        assert report.get("method") == "lexical"
        assert report["score"] is not None
        assert "معجمي" in report["reason"] or "lexical" in report["reason"].lower()

    def test_lexical_fallback_when_no_free_key(self, monkeypatch):
        monkeypatch.setattr(verifier, "_deepeval_importable", lambda: True)
        for var in ("GROQ_API_KEY", "GOOGLE_API_KEY", "CF_API_TOKEN", "CF_ACCOUNT_ID"):
            monkeypatch.delenv(var, raising=False)
        # has_any_free_key may still read env — force False via free_router if imported
        import sys
        class _FR:
            @staticmethod
            def has_any_free_key():
                return False
        monkeypatch.setitem(sys.modules, "ai.free_router", _FR)
        report = verifier.verify_answer_faithfulness(
            "ما حكم الصبر؟",
            {
                "summary": "الصبر مطلوب في الصيام",
                "verses": [{"surah": 2, "ayah": 183, "text": "يا أيها الذين آمنوا كتب عليكم الصيام"}],
            },
        )
        assert report["available"] is True
        assert report.get("method") == "lexical"

    def test_no_exception_when_missing_context(self, monkeypatch):
        monkeypatch.setattr(verifier, "_deepeval_importable", lambda: True)
        monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
        report = verifier.verify_answer_faithfulness(
            "ما حكم الصبر؟", {"summary": "إجابة ما", "verses": [], "primary_concepts": []},
        )
        assert report["available"] is True
        assert report["faithful"] is None
        assert "سياق" in report["reason"]

    def test_no_exception_when_missing_summary(self, monkeypatch):
        monkeypatch.setattr(verifier, "_deepeval_importable", lambda: True)
        monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
        report = verifier.verify_answer_faithfulness(
            "ما حكم الصبر؟",
            {"summary": "", "verses": [{"surah": 2, "ayah": 1, "text": "نص الآية"}]},
        )
        assert report["available"] is True
        assert report["faithful"] is None

    def test_never_raises_even_with_malformed_qa_result(self, monkeypatch):
        """حتى مع qa_result مشوَّه بالكامل (لا verses، لا summary)، يجب
        ألا تُرمى أي استثناءات — نفس فلسفة التدهور الآمن في كل مكان آخر
        بالمشروع (answer_question لا يجب أن ينهار بسبب فحص اختياري)."""
        monkeypatch.setattr(verifier, "_deepeval_importable", lambda: False)
        report = verifier.verify_answer_faithfulness("سؤال", {})
        assert isinstance(report, dict)
        assert report["available"] is False


# ══════════════════════════════════════════════════════════════════════════
# تكامل knowledge/qa_engine.py — include_faithfulness_check=False افتراضياً
# ══════════════════════════════════════════════════════════════════════════
class TestQAEngineIntegration:
    def test_faithfulness_check_key_present_and_none_by_default(self):
        from knowledge import qa_engine

        ckg = {"concepts": {}, "relations": {}}
        result = qa_engine.answer_question("ما حكم الصبر؟", ckg, ayat=[])
        assert "faithfulness_check" in result
        assert result["faithfulness_check"] is None

    def test_faithfulness_check_runs_gracefully_when_requested(self, monkeypatch):
        """لا مفتاح نموذج مجاني متاح في بيئة الاختبار — يجب أن يرجع تقريراً
        غير متاح (available=False) بدل رمي استثناء يكسر answer_question()."""
        for var in ("GROQ_API_KEY", "GOOGLE_API_KEY", "CF_API_TOKEN", "CF_ACCOUNT_ID"):
            monkeypatch.delenv(var, raising=False)
        from knowledge import qa_engine

        ckg = {"concepts": {}, "relations": {}}
        result = qa_engine.answer_question(
            "ما حكم الصبر؟", ckg, ayat=[], include_faithfulness_check=True,
        )
        assert result["faithfulness_check"] is not None
        assert result["faithfulness_check"]["available"] is False


class TestLexicalFaithfulness:
    def test_high_overlap_when_summary_echoes_context(self):
        summary = "كتب عليكم الصيام"
        ctx = ["سورة 2 آية 183: يا أيها الذين آمنوا كتب عليكم الصيام"]
        r = verifier._lexical_faithfulness_score(summary, ctx)
        assert r["score"] is not None and r["score"] >= 0.5
        assert r["overlap"] >= 2

    def test_low_overlap_when_unrelated(self):
        summary = "الجاذبية قانون فيزياء حديث"
        ctx = ["سورة 2 آية 183: يا أيها الذين آمنوا كتب عليكم الصيام"]
        r = verifier._lexical_faithfulness_score(summary, ctx)
        assert r["score"] is not None and r["score"] < 0.3

    def test_empty_summary(self):
        r = verifier._lexical_faithfulness_score("", ["نص ما"])
        assert r["score"] is None

    def test_alef_normalization_helps_match(self):
        # «إلى» vs «الى» after normalization should align stop/filter consistently
        summary = "الاستعانه بالصبر والصلاه"
        ctx = ["يا أيها الذين آمنوا استعينوا بالصبر والصلاة إن الله مع الصابرين"]
        r = verifier._lexical_faithfulness_score(summary, ctx)
        assert r["score"] is not None and r["score"] >= 0.3

    def test_faithful_case_passes_default_lexical_threshold(self, monkeypatch):
        monkeypatch.setenv("NSM_OFFLINE_MODE", "1")
        report = verifier.verify_answer_faithfulness(
            "ماذا عن الصبر؟",
            {
                "summary": "القرآن يحث على الاستعانة بالصبر والصلاة وأن الله مع الصابرين",
                "verses": [
                    {
                        "surah": 2,
                        "ayah": 153,
                        "text": "يا أيها الذين آمنوا استعينوا بالصبر والصلاة إن الله مع الصابرين",
                    }
                ],
                "primary_concepts": [{"name": "الصبر"}],
            },
            threshold=0.5,
        )
        assert report["method"] == "lexical"
        assert report["available"] is True
        assert report["score"] is not None
        assert report["faithful"] is True

    def test_hallucination_fails_lexical(self, monkeypatch):
        monkeypatch.setenv("NSM_OFFLINE_MODE", "1")
        report = verifier.verify_answer_faithfulness(
            "ماذا عن الصبر؟",
            {
                "summary": "نيوتن اكتشف الجاذبية وسرعة الضوء ثابتة في الفراغ",
                "verses": [
                    {
                        "surah": 2,
                        "ayah": 153,
                        "text": "يا أيها الذين آمنوا استعينوا بالصبر والصلاة إن الله مع الصابرين",
                    }
                ],
            },
            threshold=0.5,
        )
        assert report["method"] == "lexical"
        assert report["faithful"] is False


class TestFreeRouterJudge:
    def test_judge_class_callable(self):
        j = verifier._FreeRouterJudge()
        assert j.get_model_name()
        assert hasattr(j, "generate")
        assert hasattr(j, "a_generate")
        assert hasattr(j, "load_model")
