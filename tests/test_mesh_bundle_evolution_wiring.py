"""
اختبار مستقل (لا يوجد ملف اختبار مخصص لـ core/mesh_bundle.py في المستودع)
للتحقق من الإصلاح: قبل هذا التعديل كانت _apply_reputation_feedback/
_apply_reputation_recovery (حجر/رفع حجر العقدة) وrun_evolution_cycle
(دورة التطور الذاتي الكاملة) مكتوبة ومختبَرة منطقياً لكن بلا أي نقطة
تشغيل فعلية في المشروع — لم يكن أي شيء يستدعي run_evolution_cycle() على
الإطلاق. هذا الاختبار يبني MeshBundle حقيقياً (بمجلد بيانات مؤقت) ويتحقق:

1. كل استدعاء لـ record_swarm_result يُشغّل فعلياً _apply_reputation_feedback
   و_apply_reputation_recovery (وليس فقط عند دورة تطور يدوية).
2. run_evolution_cycle تُستدعى تلقائياً كل EVOLUTION_CYCLE_INTERVAL نتيجة
   سرب بالضبط — لا أكثر ولا أقل.
3. لا يحدث deadlock (self._lock صار RLock) رغم أن record_swarm_result يستدعي
   الآن دوالاً تُعيد طلب نفس القفل من نفس الخيط.
"""
import shutil
import tempfile
import threading
import types

from core.mesh_bundle import MeshBundle, EVOLUTION_CYCLE_INTERVAL


def fake_swarm_result(goal="اختبار"):
    return types.SimpleNamespace(goal=goal, tasks=[])


def run_test():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_mesh_bundle_test_")
    try:
        bundle = MeshBundle(storage_dir=tmp_dir, db_path=f"{tmp_dir}/mesh.db")

        feedback_calls = {"n": 0}
        recovery_calls = {"n": 0}
        evolution_calls = {"n": 0}

        def _fake_feedback(*a, **k):
            feedback_calls["n"] += 1

        def _fake_recovery(*a, **k):
            recovery_calls["n"] += 1

        def _fake_evolution():
            evolution_calls["n"] += 1
            return {"ok": True}

        bundle._apply_reputation_feedback = _fake_feedback
        bundle._apply_reputation_recovery = _fake_recovery
        bundle.run_evolution_cycle = _fake_evolution

        # لا deadlock: كل استدعاء يجب أن يعود خلال مهلة قصيرة، ومن خيط منفصل
        # حتى لو تعطّل أي استدعاء لن يُجمّد الاختبار نفسه.
        def _call_n_times(n):
            for _ in range(n):
                bundle.record_swarm_result(fake_swarm_result())

        total_calls = EVOLUTION_CYCLE_INTERVAL * 2 + 1
        t = threading.Thread(target=_call_n_times, args=(total_calls,))
        t.start()
        t.join(timeout=15)
        assert not t.is_alive(), "record_swarm_result deadlocked (RLock fix لم يعمل)"

        assert feedback_calls["n"] == total_calls, (
            f"_apply_reputation_feedback لم تُستدعَ من كل نتيجة سرب: "
            f"{feedback_calls['n']} != {total_calls}"
        )
        assert recovery_calls["n"] == total_calls, (
            f"_apply_reputation_recovery لم تُستدعَ من كل نتيجة سرب: "
            f"{recovery_calls['n']} != {total_calls}"
        )
        expected_evolution_calls = total_calls // EVOLUTION_CYCLE_INTERVAL
        assert evolution_calls["n"] == expected_evolution_calls, (
            f"run_evolution_cycle لم تُستدعَ بالتردد الصحيح: "
            f"{evolution_calls['n']} != {expected_evolution_calls}"
        )

        print(
            f"OK: {total_calls} نتيجة سرب → feedback={feedback_calls['n']}, "
            f"recovery={recovery_calls['n']}, evolution_cycles={evolution_calls['n']} "
            f"(كل {EVOLUTION_CYCLE_INTERVAL})"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_test()
