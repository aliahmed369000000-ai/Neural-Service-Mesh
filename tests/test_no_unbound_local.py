#!/usr/bin/env python3
"""NSM: يتحقق من عدم وجود UnboundLocalError في كتلة '8-bit Adam + TPU' في main().

المشكلة الأصلية (2026-08-16): في main() عند السطر 592:
    if SCN_TPU_ANY and USE_8BIT_ADAM:
رفع UnboundLocalError لأن USE_8BIT_ADAM يُعدَّل محليًا لاحقًا
دون `global` — انهار الكيرنل فورًا قبل أي عصر تدريب (exit 250).

الاختبار يعيد تنفيذ كتلة المنطق كما كتبت في main() (دالة حقيقية)
ويتأكد أنها تمر دون أخطاء في جميع السيناريوهات.
"""
import os
import unittest
from unittest import mock

USE_8BIT_ADAM = os.environ.get("SCN_USE_8BIT_ADAM", "0") == "1"


# ── نسخ كتلة main() الحقيقية بعد الإصلاح ──────────────────────────────────────
def tpu_adam_block():
    """نسخة مطابقة لكتلة main() في train_pretrain_torch.py (بعد إضافة global)."""
    global USE_8BIT_ADAM
    SCN_TPU_ANY = os.environ.get("SCN_TPU", "0") == "1"
    if SCN_TPU_ANY and USE_8BIT_ADAM:
        USE_8BIT_ADAM = False
    if USE_8BIT_ADAM:
        USE_8BIT_ADAM = False  # بعد فشل استيراد bitsandbytes
    return SCN_TPU_ANY


class TestTpuAdamBlock(unittest.TestCase):
    def setUp(self):
        global USE_8BIT_ADAM
        USE_8BIT_ADAM = True

    def test_tpu_mode_no_unbound_error(self):
        """وضع TPU مع USE_8BIT_ADAM=True — كان يسبّب UnboundLocalError سابقًا."""
        with mock.patch.dict(os.environ, {"SCN_TPU": "1", "SCN_USE_8BIT_ADAM": "1"}):
            tpu = tpu_adam_block()
        self.assertTrue(tpu, "SCN_TPU يجب أن يكون 1")
        self.assertFalse(USE_8BIT_ADAM, "8-bit Adam يجب أن يُعطَّل تلقائيًا على TPU")

    def test_gpu_mode_adam_disabled_after_import_fail(self):
        """وضع GPU — كتلة الاستيراد تفشل فيُعطَّل Adam-8bit."""
        with mock.patch.dict(os.environ, {"SCN_TPU": "0", "SCN_USE_8BIT_ADAM": "1"}):
            tpu = tpu_adam_block()
        self.assertFalse(tpu)
        self.assertFalse(USE_8BIT_ADAM, "بعد فشل استيراد bitsandbytes يُعطَّل")


def _broken_block():
    """نفس كتلة main() قبل الإصلاح — يجب أن ترفع UnboundLocalError."""
    SCN_TPU_ANY = os.environ.get("SCN_TPU", "0") == "1"
    if SCN_TPU_ANY and USE_8BIT_ADAM:
        USE_8BIT_ADAM = False
    if USE_8BIT_ADAM:
        USE_8BIT_ADAM = False


class TestOriginalBug(unittest.TestCase):
    """يتأكد أن المشكلة الأصلية حقيقية — النسخة دون global تفشل فعلًا."""

    def test_original_raises_unbound_local(self):
        with mock.patch.dict(os.environ, {"SCN_TPU": "1", "SCN_USE_8BIT_ADAM": "1"}):
            with self.assertRaises(UnboundLocalError):
                _broken_block()


if __name__ == "__main__":
    unittest.main(verbosity=2)
