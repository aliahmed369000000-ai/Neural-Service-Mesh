"""مهمة محتوى/فيديو تقتل العملية في كل استئناف لا يجب أن تُستأنف للأبد
(نفس علّة حلقة الانهيار التي أُصلحت لاستئناف الأسرِبة)."""
import sqlite3
import time

import pytest

from ai.content_job_manager import ContentJobManager
import ai.content_job_manager as cjm
from ai.video_job_manager import VideoJobManager
import ai.video_job_manager as vjm


def _seed_content(db_path, job_id=1, attempts=0, status="running"):
    with sqlite3.connect(str(db_path)) as c:
        c.execute("""INSERT INTO content_jobs
            (job_id, status, kwargs, result, error, started_at, finished_at, resume_attempts)
            VALUES (?, ?, '{}', NULL, NULL, ?, NULL, ?)""",
            (job_id, status, time.time(), attempts))
        c.commit()


def test_content_job_abandoned_after_max_attempts(tmp_path, monkeypatch):
    mgr = ContentJobManager(db_path=tmp_path / "c.db")
    _seed_content(mgr._db_path, attempts=cjm.MAX_RESUME_ATTEMPTS)  # وصل السقف أصلاً
    launched = []
    monkeypatch.setattr(mgr, "_launch", lambda job, kwargs: launched.append(job.job_id))
    resumed = mgr.resume_interrupted()
    assert resumed == [] and launched == []
    with sqlite3.connect(str(mgr._db_path)) as c:
        row = c.execute("SELECT status FROM content_jobs WHERE job_id=1").fetchone()
    assert row[0] == "abandoned"


def test_content_job_resumes_and_increments_counter(tmp_path, monkeypatch):
    mgr = ContentJobManager(db_path=tmp_path / "c.db")
    _seed_content(mgr._db_path, attempts=1)
    launched = []
    monkeypatch.setattr(mgr, "_launch", lambda job, kwargs: launched.append(job.resume_attempts))
    resumed = mgr.resume_interrupted()
    assert resumed == [1] and launched == [2]
    with sqlite3.connect(str(mgr._db_path)) as c:
        row = c.execute("SELECT resume_attempts, status FROM content_jobs WHERE job_id=1").fetchone()
    assert row == (2, "running")  # العدّاد محفوظ حتى قبل انتهاء العمل الفعلي


def _seed_video(db_path, job_id=1, attempts=0, op_name="trim"):
    with sqlite3.connect(str(db_path)) as c:
        c.execute("""INSERT INTO video_jobs
            (job_id, label, status, op_name, kwargs, result, error, started_at, finished_at, resume_attempts)
            VALUES (?, 'قص', 'running', ?, '{}', NULL, NULL, ?, NULL, ?)""",
            (job_id, op_name, time.time(), attempts))
        c.commit()


def test_video_job_abandoned_after_max_attempts(tmp_path, monkeypatch):
    mgr = VideoJobManager(db_path=tmp_path / "v.db")
    _seed_video(mgr._db_path, attempts=vjm.MAX_RESUME_ATTEMPTS)
    launched = []
    monkeypatch.setattr(mgr, "_launch", lambda job, fn, kwargs: launched.append(job.job_id))
    resumed = mgr.resume_interrupted()
    assert resumed == [] and launched == []
    with sqlite3.connect(str(mgr._db_path)) as c:
        row = c.execute("SELECT status FROM video_jobs WHERE job_id=1").fetchone()
    assert row[0] == "abandoned"


def test_video_job_unresolvable_op_not_affected_by_cap(tmp_path, monkeypatch):
    """مسار 'fn غير معروفة' يبقى failed فوراً كما كان، بصرف النظر عن العدّاد."""
    mgr = VideoJobManager(db_path=tmp_path / "v.db")
    _seed_video(mgr._db_path, attempts=0, op_name=None)
    resumed = mgr.resume_interrupted()
    assert resumed == []
    with sqlite3.connect(str(mgr._db_path)) as c:
        row = c.execute("SELECT status FROM video_jobs WHERE job_id=1").fetchone()
    assert row[0] == "failed"


def test_healthy_job_managers_still_start_and_complete(tmp_path):
    mgr = ContentJobManager(db_path=tmp_path / "c2.db")

    def fake_pipeline(**kw):
        return {"ok": True}
    import ai.content_agent as ca_mod  # noqa
    orig = None
    import sys
    sys.modules.setdefault("ai.content_agent", type(sys)("ai.content_agent"))
    sys.modules["ai.content_agent"].run_content_pipeline = fake_pipeline

    jid = mgr.start(topic="x")
    for _ in range(50):
        if mgr.get(jid).status != "running":
            break
        time.sleep(0.05)
    assert mgr.get(jid).status == "done"
