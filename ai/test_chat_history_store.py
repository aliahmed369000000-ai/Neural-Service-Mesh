"""
اختبارات ai/chat_history_store.py

تستخدم قاعدة بيانات مؤقتة (tmp_path) بدل memory/chat_history.db الفعلية
عبر monkeypatch لـ DB_PATH — لا تلمس أي بيانات حقيقية، ولا تحتاج أي
مفتاح API.
"""
from pathlib import Path

import pytest

from ai import chat_history_store as store


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """كل اختبار يستخدم قاعدة بيانات مؤقتة خاصة به، معزولة تماماً."""
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "chat_history_test.db")
    yield


@pytest.fixture()
def _unwritable_db_path(tmp_path):
    """مسار DB أبوه ملف عادي وليس مجلداً — يفشل mkdir دائماً، حتى بصلاحية
    root (بخلاف /nonexistent-dir-xyz التي تنجح إن شُغِّل الاختبار كـroot)."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    return blocker / "db.sqlite"


class TestSaveAndReadMessages:
    def test_save_then_get_session_messages_roundtrip(self):
        store.save_message("s1", "user", "ما حكم الصبر؟")
        store.save_message("s1", "nsm", "الصبر من أعلى مراتب الإيمان...")
        msgs = store.get_session_messages("s1")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "ما حكم الصبر؟"
        assert msgs[1]["role"] == "nsm"

    def test_messages_are_ordered_oldest_first(self):
        for i in range(5):
            store.save_message("s1", "user", f"رسالة {i}")
        msgs = store.get_session_messages("s1")
        assert [m["content"] for m in msgs] == [f"رسالة {i}" for i in range(5)]

    def test_sessions_are_isolated(self):
        store.save_message("s1", "user", "من الجلسة الأولى")
        store.save_message("s2", "user", "من الجلسة الثانية")
        assert len(store.get_session_messages("s1")) == 1
        assert len(store.get_session_messages("s2")) == 1
        assert store.get_session_messages("s1")[0]["content"] == "من الجلسة الأولى"

    def test_source_badge_is_persisted(self):
        store.save_message("s1", "nsm", "إجابة من الكاش", source_badge="⚡ كاش متعلَّم")
        msgs = store.get_session_messages("s1")
        assert msgs[0]["source_badge"] == "⚡ كاش متعلَّم"


class TestGetFirstMessage:
    def test_returns_first_message_in_session(self):
        store.save_message("s1", "user", "أول سؤال")
        store.save_message("s1", "nsm", "أول رد")
        store.save_message("s1", "user", "ثاني سؤال")
        first = store.get_first_message("s1")
        assert first is not None
        assert first["role"] == "user"
        assert first["content"] == "أول سؤال"

    def test_returns_none_for_unknown_session(self):
        assert store.get_first_message("no-such-session") is None

    def test_returns_none_for_empty_session_id(self):
        assert store.get_first_message("") is None


class TestListSessions:
    def test_lists_all_sessions_with_counts(self):
        store.save_message("s1", "user", "رسالة")
        store.save_message("s1", "nsm", "رد")
        store.save_message("s2", "user", "رسالة أخرى")
        sessions = store.list_sessions()
        by_id = {s["session_id"]: s for s in sessions}
        assert by_id["s1"]["message_count"] == 2
        assert by_id["s2"]["message_count"] == 1


class TestStorageStats:
    def test_empty_db_gives_zero_stats(self):
        stats = store.get_storage_stats()
        assert stats["total_messages"] == 0
        assert stats["total_sessions"] == 0

    def test_stats_count_messages_and_distinct_sessions(self):
        store.save_message("s1", "user", "١")
        store.save_message("s1", "nsm", "٢")
        store.save_message("s2", "user", "٣")
        stats = store.get_storage_stats()
        assert stats["total_messages"] == 3
        assert stats["total_sessions"] == 2
        assert stats["db_size_bytes"] > 0

    def test_returns_zeroed_stats_on_bad_db_path(self, monkeypatch, _unwritable_db_path):
        monkeypatch.setattr(store, "DB_PATH", _unwritable_db_path)
        stats = store.get_storage_stats()
        assert stats == {"total_messages": 0, "total_sessions": 0, "db_size_bytes": 0}


class TestDeleteSessionsOlderThan:
    def test_deletes_only_sessions_older_than_cutoff(self, monkeypatch):
        import ai.chat_history_store as _store_mod

        # جلسة قديمة (نزرعها يدوياً بتاريخ قديم عبر تعديل مباشر بقاعدة البيانات)
        store.save_message("old-session", "user", "قديمة")
        store.save_message("new-session", "user", "حديثة")

        old_ts = (
            _store_mod.datetime.now(_store_mod.timezone.utc)
            - _store_mod.timedelta(days=100)
        ).isoformat()
        with _store_mod._db() as c:
            c.execute(
                "UPDATE chat_messages SET created_at = ? WHERE session_id = ?",
                (old_ts, "old-session"),
            )
            c.commit()

        deleted = store.delete_sessions_older_than(30)
        assert deleted == 1
        assert store.get_session_messages("old-session") == []
        assert len(store.get_session_messages("new-session")) == 1

    def test_zero_days_deletes_nothing(self):
        store.save_message("s1", "user", "رسالة")
        assert store.delete_sessions_older_than(0) == 0
        assert len(store.get_session_messages("s1")) == 1

    def test_returns_zero_on_empty_db(self):
        assert store.delete_sessions_older_than(30) == 0

    def test_returns_zero_on_bad_db_path(self, monkeypatch, _unwritable_db_path):
        monkeypatch.setattr(store, "DB_PATH", _unwritable_db_path)
        assert store.delete_sessions_older_than(30) == 0


class TestGracefulDegradation:
    def test_save_message_never_raises_on_empty_session_id(self):
        store.save_message("", "user", "نص ما")  # لا استثناء

    def test_save_message_never_raises_on_empty_content(self):
        store.save_message("s1", "user", "   ")  # لا استثناء
        assert store.get_session_messages("s1") == []

    def test_get_session_messages_returns_empty_list_on_bad_db_path(self, monkeypatch, _unwritable_db_path):
        monkeypatch.setattr(store, "DB_PATH", _unwritable_db_path)
        assert store.get_session_messages("s1") == []

    def test_list_sessions_returns_empty_list_on_bad_db_path(self, monkeypatch, _unwritable_db_path):
        monkeypatch.setattr(store, "DB_PATH", _unwritable_db_path)
        assert store.list_sessions() == []
