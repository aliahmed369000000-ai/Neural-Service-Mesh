"""استئناف ExecutionEngine/مهام المحتوى/الفيديو عند الإقلاع لا يجب أن
يحجب __init__ — يعمل في خيط خلفي، تماماً مثل استئناف الأسرِبة."""
import inspect
import threading

from core.mesh_bundle import MeshBundle


def test_init_starts_background_thread_for_jobs_resume():
    src = inspect.getsource(MeshBundle.__init__)
    assert "_auto_resume_engine_and_jobs" in src
    assert 'threading.Thread(\n            target=self._auto_resume_engine_and_jobs' in src \
        or "target=self._auto_resume_engine_and_jobs" in src


def test_init_does_not_call_resume_interrupted_synchronously():
    src = inspect.getsource(MeshBundle.__init__)
    assert "engine.resume_interrupted()" not in src
    assert "get_content_job_manager().resume_interrupted()" not in src
    assert "get_video_job_manager().resume_interrupted()" not in src


def test_auto_resume_engine_and_jobs_runs_in_calling_thread_and_never_raises():
    ran_in = []

    class BoomEngine:
        def __init__(self, *a, **k): pass
        def resume_interrupted(self):
            ran_in.append(threading.current_thread().name)
            raise RuntimeError("engine boom")

    import core.mesh_bundle as mb
    stub = type("S", (), {})()
    stub.registry = stub.graph = stub.storage = stub.exec_log = stub.ai_decision = None
    orig = mb.ExecutionEngine
    mb.ExecutionEngine = BoomEngine
    try:
        MeshBundle._auto_resume_engine_and_jobs(stub)  # لا يرفع استثناء للخارج
    finally:
        mb.ExecutionEngine = orig
    assert ran_in == [threading.current_thread().name]
