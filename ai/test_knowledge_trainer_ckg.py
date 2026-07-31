"""
اختبارات CKGManager._load — التعامل مع ملف قاعدة معرفة تالف/غير قابل للتحليل.

يغطي إصلاح استقرار: كان _load يبتلع أي استثناء تحليل JSON بصمت
(except Exception: pass) ويعيد قاعدة معرفة فارغة بدون أي تحذير —
ما يعني أن تلف ملف cognitive_graph.json (~7300 مفهوم) كان يمكن أن
يمرّ دون أي إشعار في الإنتاج. الإصلاح: تسجيل logger.error صريح قبل
الرجوع لقاعدة فارغة، مع إبقاء نفس سلوك "التعافي الآمن" (لا نرفع
استثناء، النظام يستمر بقاعدة فارغة بدل الانهيار الكامل).
"""
import json
import logging

from ai.knowledge_trainer import CKGManager


def test_load_valid_ckg_file_no_error_logged(tmp_path, caplog):
    """ملف JSON سليم: يُحمَّل بدون أي رسالة خطأ."""
    path = tmp_path / "cognitive_graph.json"
    path.write_text(json.dumps({
        "_meta": {"schema_version": "1.0.0"},
        "concepts": {"الله": {"cluster": "توحيد"}},
        "relations": {},
    }), encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        ckg = CKGManager(path=path)

    assert ckg.concept_count() == 1
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def test_load_corrupted_ckg_file_logs_error_and_falls_back_empty(tmp_path, caplog):
    """ملف JSON تالف: يجب ألا ينهار النظام، لكن يجب تسجيل خطأ صريح
    (وليس الرجوع الصامت لقاعدة فارغة كما كان الحال سابقاً)."""
    path = tmp_path / "cognitive_graph.json"
    path.write_text("{ هذا ليس JSON صالحاً على الإطلاق ", encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        ckg = CKGManager(path=path)

    # سلوك التعافي الآمن يبقى كما هو: قاعدة فارغة، لا انهيار
    assert ckg.concept_count() == 0
    # لكن الآن يجب أن يظهر تحذير/خطأ صريح في السجلات
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert any("فشل تحميل" in r.getMessage() for r in caplog.records)
