"""
اختبارات ai/rollback_guard.py::CheckpointGuard — الوحدة المسؤولة عن حماية
حفظ LoRA (وأي checkpoint آخر مستقبلاً) من الكتابة فوق نسخة جيدة بنسخة
أسوأ بصمت (رُبطت بـ knowledge/qa_engine.py في يوليو 2026).

تستخدم tmp_path (fixture pytest قياسي) بدل مجلد ثابت — كل اختبار يعمل في
مجلد مؤقت معزول، بلا أي تداخل بين الاختبارات أو تلويث لملفات حقيقية.
"""
from ai.rollback_guard import CheckpointGuard


class TestCheckpointGuard:
    def test_first_update_accepted_without_baseline(self, tmp_path):
        """أول تحديث (لا baseline بعد) يجب أن يُقبَل تلقائياً دائماً."""
        guard = CheckpointGuard(asset="test_asset_1", root=tmp_path / "checkpoints")
        f = tmp_path / "weights.npy"

        decision = guard.guarded_update(
            [f], lambda: f.write_text("v1"), lambda: 0.9, tolerance=-0.1, label="step1",
        )
        assert decision.accepted is True
        assert f.read_text() == "v1"

    def test_regression_rejected_and_file_restored(self, tmp_path):
        """
        السيناريو الأهم: تراجع حاد في الجودة يجب أن يُرفَض، والملف يُستعاد
        فعلياً لآخر نسخة جيدة معروفة — لا مجرد رفض القبول شكلياً.
        """
        guard = CheckpointGuard(asset="test_asset_2", root=tmp_path / "checkpoints")
        f = tmp_path / "weights.npy"

        guard.guarded_update([f], lambda: f.write_text("good"), lambda: 0.9,
                              tolerance=-0.1, label="step1")
        decision = guard.guarded_update([f], lambda: f.write_text("BAD"), lambda: 0.3,
                                         tolerance=-0.1, label="step2")

        assert decision.accepted is False
        assert decision.rolled_back is True
        assert f.read_text() == "good", "الملف لم يُستعَد فعلياً بعد الرفض!"

    def test_genuine_improvement_accepted(self, tmp_path):
        """تحسّن حقيقي في الجودة يجب أن يُقبَل ويُحدَّث الملف."""
        guard = CheckpointGuard(asset="test_asset_3", root=tmp_path / "checkpoints")
        f = tmp_path / "weights.npy"

        guard.guarded_update([f], lambda: f.write_text("v1"), lambda: 0.9,
                              tolerance=-0.1, label="step1")
        decision = guard.guarded_update([f], lambda: f.write_text("v2-better"), lambda: 0.95,
                                         tolerance=-0.1, label="step2")

        assert decision.accepted is True
        assert f.read_text() == "v2-better"

    def test_small_fluctuation_within_tolerance_accepted(self, tmp_path):
        """تذبذب طفيف ضمن tolerance يجب ألا يُرفَض — وإلا يصبح النظام حساساً بلا داعٍ."""
        guard = CheckpointGuard(asset="test_asset_4", root=tmp_path / "checkpoints")
        f = tmp_path / "weights.npy"

        guard.guarded_update([f], lambda: f.write_text("v1"), lambda: 0.90,
                              tolerance=-0.1, label="step1")
        decision = guard.guarded_update([f], lambda: f.write_text("v2"), lambda: 0.85,
                                         tolerance=-0.1, label="step2")
        assert decision.accepted is True  # انخفاض 0.05 فقط، ضمن tolerance=-0.1
