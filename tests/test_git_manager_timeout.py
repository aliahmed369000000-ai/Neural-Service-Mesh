"""ai/git_manager.py::GitManager — لم يكن لأي subprocess.run (clone/config/
add/commit/push) أي timeout إطلاقاً. جرّبتُ هذا فعلياً: شغّلت عقدة حقيقية
عبر `python3 -m ai.node_launcher` بلا GITHUB_TOKEN/HF_TOKEN (سيناريو نشر
طبيعي تماماً، لا حالة حافة متعمَّدة) وتابعت سجلّها مباشرة — أول دورة تطوّر
ذاتي تلقائي (ai/living_mesh.py::maybe_self_evolve) علّقت عند "git clone"
بلا أي تقدّم ولا أي خطأ لأكثر من 15 ثانية. تحقّقت بمعزل عن العقدة: نفس
أمر git clone المجهول الهوية لنفس المستودع يعلّق فعلاً (> 15 ثانية بلا أي
استجابة، لا فشل سريع كما يحدث عادة مع مصادقة مرفوضة). بما أن
maybe_self_evolve تُستدعى من خيط مراقب خلفي *واحد* في حلقة، فتعليق استدعاء
git واحد يُسكِت دورة التطوّر الذاتي للعقدة بالكامل، للأبد، بصمت تام — لا
استثناء يصل أبداً لأي except لأن الخيط نفسه عالق فعلياً داخل subprocess.run.

هذا الملف (أول اختبار على الإطلاق لـGitManager) يثبت أن clone/push الآن
يفشلان بوضوح خلال ثوانٍ معدودة بدل التعليق الأبدي، باستخدام أمر git بديل
وهمي (fake git عبر PATH) يحاكي التعليق الحقيقي الذي أعيد إنتاجه فعلاً —
بلا انتظار timeout_seconds الافتراضي (60 ثانية) الذي يُبطئ الاختبار بلا داعٍ.
"""
from __future__ import annotations

import os
import shutil
import stat
import tempfile

import pytest

from ai.git_manager import GitManager, GitOperationTimeout


@pytest.fixture()
def hanging_git_on_path(monkeypatch):
    """أمر git بديل في PATH ينام إلى الأبد عملياً (أطول من أي timeout
    اختبار معقول) بدل تنفيذ أي عملية حقيقية — يحاكي بالضبط التعليق الذي
    تحقّقتُ منه فعلياً مع git clone الحقيقي بلا مصادقة."""
    bin_dir = tempfile.mkdtemp(prefix="fake_git_bin_")
    fake_git = os.path.join(bin_dir, "git")
    with open(fake_git, "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\nsleep 3600\n")
    os.chmod(fake_git, os.stat(fake_git).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    old_path = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", f"{bin_dir}:{old_path}")
    yield
    shutil.rmtree(bin_dir, ignore_errors=True)


def test_clone_raises_timeout_instead_of_hanging_forever(hanging_git_on_path):
    gm = GitManager(token=None, timeout_seconds=1)
    with pytest.raises(GitOperationTimeout):
        gm.clone("hang_test_clone")
    # لا بقايا استنساخ جزئي معلَّقة بعد التنظيف داخل clone()
    assert not os.path.exists(os.path.join(gm.base_dir, "hang_test_clone"))


def test_push_raises_timeout_instead_of_hanging_forever(hanging_git_on_path, tmp_path):
    gm = GitManager(token=None, timeout_seconds=1)
    repo_path = str(tmp_path / "fake_repo")
    os.makedirs(repo_path)
    with pytest.raises(GitOperationTimeout):
        gm.commit_and_push(repo_path, "test commit")
    # finally: self.cleanup(repo_path) يعمل حتى بعد رفع GitOperationTimeout
    assert not os.path.exists(repo_path)


def test_default_timeout_is_a_finite_positive_number():
    """تثبيت عدم الرجوع لسلوك 'بلا مهلة إطلاقاً' مستقبلاً بالخطأ."""
    gm = GitManager()
    assert 0 < gm.timeout_seconds < 600


def test_clone_uses_unique_dir_per_call_and_shallow(monkeypatch):
    """عدة عقد على جهاز واحد كانت تستنسخ إلى نفس المجلد الثابت فتحذف كل واحدة
    مجلد الأخرى (File exists / unable to write pack)، والاستنساخ الكامل لثلاث
    عقد معاً كان يتجاوز المهلة — وُجد بتشغيل 3 عقد فعلياً مع GITHUB_TOKEN."""
    import subprocess as sp
    calls = []

    class _R:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return _R()

    monkeypatch.setattr(sp, "run", fake_run)
    gm = GitManager(token=None, timeout_seconds=5)
    p1 = gm.clone("same_name")
    p2 = gm.clone("same_name")
    assert p1 != p2
    assert all("--depth" in c and "1" in c for c in calls)
