"""
اختبارات تراجع: تحويل استثناءات صامتة (except Exception: pass) على مسارات
كتابة/قراءة حالة دائمة إلى تحذيرات مسجَّلة صراحةً في logger — دون تغيير
سلوك "التعافي الآمن" الأصلي (لا رفع استثناء، النظام يستمر بالعمل).

يغطي:
  - ai/safe_evolution.py  — SnapshotManager._load_index (فهرس لقطات rollback)
  - ai/task_manager.py    — update_task_status / mark_plan_status / record_checkpoint
  - ai/route_log_store.py — clear_all
"""
import logging

import pytest


# ══════════════════════════════════════════════════════════════════════════
# safe_evolution.SnapshotManager
# ══════════════════════════════════════════════════════════════════════════
def test_snapshot_manager_corrupted_index_logs_error(tmp_path, caplog):
    from ai.safe_evolution import SnapshotManager

    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "index.json").write_text("{ ليس JSON صالحاً", encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        mgr = SnapshotManager(snap_dir=snap_dir, max_snapshots=5)

    # سلوك التعافي الآمن يبقى: فهرس فارغ، لا انهيار
    assert mgr._index == []
    # لكن يجب تسجيل خطأ صريح الآن
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert any("فشل تحميل فهرس اللقطات" in r.getMessage() for r in caplog.records)


def test_snapshot_manager_valid_index_no_error(tmp_path, caplog):
    from ai.safe_evolution import SnapshotManager
    import json

    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "index.json").write_text(json.dumps([{"snapshot_id": "s1"}]), encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        mgr = SnapshotManager(snap_dir=snap_dir, max_snapshots=5)

    assert mgr._index == [{"snapshot_id": "s1"}]
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


# ══════════════════════════════════════════════════════════════════════════
# task_manager
# ══════════════════════════════════════════════════════════════════════════
@pytest.fixture()
def task_manager_module(tmp_path, monkeypatch):
    """يعزل task_manager عن memory/task_manager.db الحقيقي، ويعيد استيراده نظيفاً."""
    import importlib
    import ai.task_manager as tm

    monkeypatch.setattr(tm, "DB_PATH", tmp_path / "task_manager.db")
    monkeypatch.setattr(tm, "_SCHEMA_READY", False, raising=False)
    return tm


def test_update_task_status_failure_is_logged_not_swallowed(task_manager_module, caplog, monkeypatch):
    tm = task_manager_module

    def _boom():
        raise sqlite3.OperationalError("database is locked")

    import sqlite3
    monkeypatch.setattr(tm, "_connect", _boom)

    with caplog.at_level(logging.WARNING):
        tm.update_task_status(plan_id=1, task_id=1, status="done")  # لا يجب أن يرفع استثناء

    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    assert any("update_task_status" in r.getMessage() for r in caplog.records)


def test_mark_plan_status_failure_is_logged(task_manager_module, caplog, monkeypatch):
    tm = task_manager_module
    import sqlite3

    def _boom():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(tm, "_connect", _boom)

    with caplog.at_level(logging.WARNING):
        tm.mark_plan_status(plan_id=1, status="done")

    assert any("mark_plan_status" in r.getMessage() for r in caplog.records)


def test_record_checkpoint_failure_is_logged(task_manager_module, caplog, monkeypatch):
    tm = task_manager_module
    import sqlite3

    def _boom():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(tm, "_connect", _boom)

    with caplog.at_level(logging.WARNING):
        tm.record_checkpoint(plan_id=1, task_id=1, commit_hash="abc123")

    assert any("record_checkpoint" in r.getMessage() for r in caplog.records)


def test_task_manager_happy_path_still_works(task_manager_module):
    """التأكد من أن إضافة الـ logging لم تكسر المسار السعيد الطبيعي."""
    tm = task_manager_module

    class FakePlan:
        idea = "فكرة"
        app_name = "تطبيق"
        app_type = "web"
        description = "وصف"
        tech_stack = "python"
        tasks = []

    plan_id = tm.create_plan(FakePlan())
    assert plan_id >= 0
    tm.update_task_status(plan_id, 1, "done", "نتيجة")
    tm.mark_plan_status(plan_id, "done")
    tm.record_checkpoint(plan_id, 1, "abcdef1")
    last = tm.get_last_checkpoint(plan_id)
    assert last is not None
    assert last["commit_hash"] == "abcdef1"


# ══════════════════════════════════════════════════════════════════════════
# route_log_store
# ══════════════════════════════════════════════════════════════════════════
@pytest.fixture()
def route_log_module(tmp_path, monkeypatch):
    import ai.route_log_store as rls
    monkeypatch.setattr(rls, "DB_PATH", tmp_path / "route_log.db")
    monkeypatch.setattr(rls, "_SCHEMA_READY", False, raising=False)
    return rls


def test_clear_all_failure_is_logged_not_swallowed(route_log_module, caplog, monkeypatch):
    rls = route_log_module
    import sqlite3

    def _boom():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(rls, "_connect", _boom)

    with caplog.at_level(logging.WARNING):
        rls.clear_all()  # يجب ألا يرفع استثناء

    assert any("clear_all" in r.getMessage() for r in caplog.records)


def test_clear_all_happy_path_still_works(route_log_module):
    rls = route_log_module
    rls.append_entry({"query": "سؤال", "category": "فقه"})
    assert len(rls.get_recent(limit=10)) == 1
    rls.clear_all()
    assert len(rls.get_recent(limit=10)) == 0
