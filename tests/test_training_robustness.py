#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NSM — اختبارات متانة التدريب (Checkpoint-proof Training) بدون اتصال حقيقي.

يختبر:
1. معالج الموت المفاجئ SIGTERM (يحفظ checkpoint ويحاول رفعًا)
2. منطق التصنيف للرفع السريع FIRST_FAST
3. عدم انهيار السكربت عند غياب GITHUB_TOKEN (رفع يتخطى بسلام)
4. retry logic: _upload_checkpoint_once يُعاد مع backoff عند الفشل
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "surah_chain_network"))


class TestSignalHandler(unittest.TestCase):
    """معالج الموت المفاجئ — SIGTERM يحفظ ويرفع قبل الموت."""

    def test_signal_handler_registered(self):
        """بعد استيراد train_pretrain_torch handler يعمل بلا استثناء."""
        import train_pretrain_torch as tpt
        # handler لا يعتمد على torch إلا عند الموت — تسجيله سليم
        self.assertTrue(callable(tpt._handle_fatal_signal))

    def test_handler_exits_cleanly_on_sigterm(self):
        """إرسال SIGTERM فعليًا إلى عملية فرعية تستورد السكربت → exit 0.

        العملية الفرعية: تستورد السكربت ثم ترسل لنفسها SIGTERM —
        معالج _handle_fatal_signal يجب أن يلتقطها ويخرج exit(0).
        """
        script = (
            "import sys, os, signal, time, threading\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            f"sys.path.insert(0, {str(ROOT / 'experiments' / 'surah_chain_network')!r})\n"
            "os.environ.update({\n"
            "    'SCN_N': '100', 'SCN_EPOCHS': '2', 'SCN_BATCH': '8',\n"
            "    'SCN_PRESET': 'small', 'GITHUB_TOKEN': '',\n"
            "})\n"
            "import train_pretrain_torch as tpt\n"
            # محاكاة حالة crash داخل handler
            "tpt._CRASH_STATE.update({'model': None, 'train_meta': {}, 'epoch': 1})\n"
            # SIGTERM يصل بعد 2 ثانية — بعد اكتمال الاستيراد وتسجيل handler
            "import threading\n"
            "threading.Timer(2, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()\n"
            "time.sleep(5)\n"
            "print('UNREACHED')\n"
        )
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, f"stdout={r.stdout[-500:]} stderr={r.stderr[-300:]}")
        self.assertNotIn("UNREACHED", r.stdout)


class TestFirstFastEnv(unittest.TestCase):
    """SCN_FIRST_FAST مفعّل افتراضيًا + SCN_UPLOAD_RETRIES=3."""

    def test_defaults(self):
        env = os.environ.copy()
        for k in ("SCN_FIRST_FAST", "SCN_UPLOAD_RETRIES"):
            env.pop(k, None)
        with mock.patch.dict(os.environ, env, clear=False):
            # أعد تحميل الوحدة بإزالة GITHUB_TOKEN + env متغيرات
            import importlib
            for k in ("SCN_CHECKPOINT_EVERY", "SCN_FIRST_FAST", "SCN_UPLOAD_RETRIES"):
                os.environ.pop(k, None)
            import train_pretrain_torch as tpt
            importlib.reload(tpt)
            self.assertTrue(tpt.FIRST_FAST, "SCN_FIRST_FAST يجب أن يكون 1 افتراضيًا")
            self.assertEqual(tpt.UPLOAD_RETRIES, 3)
            self.assertEqual(tpt.CHECKPOINT_EVERY, 2)
            # تنظيف env
            os.environ["SCN_CHECKPOINT_EVERY"] = "2"


class TestCheckpointEveryLogic(unittest.TestCase):
    """منطق should_upload: FIRST_FAST أول عصورين + دوري."""

    def test_first_two_epochs_always_upload(self):
        import train_pretrain_torch as tpt
        # نختبر المنطق مباشرة عبر الدالة الجديدة
        # into_run = ep - start_epoch: fast يغطي 1..2، دوري every=2 يغطي الزوجي
        self.assertTrue(tpt._should_upload(ep=3, start_epoch=2, every=2))   # into_run=1 → fast
        self.assertTrue(tpt._should_upload(ep=4, start_epoch=2, every=2))   # into_run=2 → fast
        self.assertFalse(tpt._should_upload(ep=5, start_epoch=2, every=2))  # into_run=3 → لا
        self.assertTrue(tpt._should_upload(ep=6, start_epoch=2, every=2))   # into_run=4 → دوري (زوجي)
        self.assertFalse(tpt._should_upload(ep=7, start_epoch=2, every=2))  # into_run=5 → لا
        self.assertTrue(tpt._should_upload(ep=8, start_epoch=2, every=2))   # into_run=6 → دوري


class TestUploadWithoutToken(unittest.TestCase):
    """بلا GITHUB_TOKEN → الرفع يتخطى بسلام (لا استثناء، لا اتصال)."""

    def test_no_token_skips_cleanly(self):
        import train_pretrain_torch as tpt
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "", "GH_TOKEN": ""}, clear=True):
            # يجب ألا يرمي أي استثناء
            tpt._upload_checkpoint(ep=7)
            # لا نختبر stdout هنا — لا اتصال شبكي يجب أن يحدث


class TestUploadRetryOnce(unittest.TestCase):
    """retry: فشل attempt الأول → محاولة ثانية مع backoff."""

    def test_retry_on_clone_failure(self):
        import train_pretrain_torch as tpt
        attempts = []

        def fake_once(token, repo, branch, tmp, ep, attempt=1):
            attempts.append(attempt)
            return "clone failed" if attempt == 1 else None

        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "fake-token"}), \
             mock.patch.object(tpt, "_upload_checkpoint_once", fake_once), \
             mock.patch.object(tpt, "UPLOAD_RETRIES", 3), \
             mock.patch("time.sleep"):
            tpt._upload_checkpoint(ep=9)

        # الأولى failed، الثانية succeed → توقف
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], 1)


if __name__ == "__main__":
    unittest.main()
