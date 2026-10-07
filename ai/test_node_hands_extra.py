# -*- coding: utf-8 -*-
"""اختبارات لإضافة جديدة لليد اليسرى (core/node_hands.py عبر
core/mesh_bundle.py._left_hand_tools): ربط أداة موجودة مسبقاً في
ai/agent_tools.py (kaggle_status) لم تكن مربوطة بأي يد — لكن عبر نسخة
"سريعة" (fetch_output=False) لأن kaggle_status الأصلية قد تستغرق حتى
600 ثانية لجلب مخرجات كيرنل منتهٍ، بينما مهلة اليد الواحدة لكل أدواتها
15 ثانية افتراضياً — ربطها كما هي كان سيُفشل كل استدعاء بمهلة اليد قبل
أن يصل لنتيجة فعلية.

ملاحظة: فكرة إضافية كانت مخطَّطة في هذه الجلسة (أداة my_status للمعرفة
الذاتية بحالة العقدة وسمعتها) أُسقطت لأن عملاً موازياً نشطاً أضاف فعلاً
node_status + reputation_detail على اليد اليسرى، تغطيان نفس الحاجة
(reputation_detail تعيد rep.to_dict() كاملة، بما فيها is_quarantined
وquarantine_checks) — تكرارها كان سيُضيف أداة زائدة بلا فائدة حقيقية.
"""
from __future__ import annotations

import shutil
import tempfile
from unittest.mock import patch

import pytest

from ai.agent_tools import kaggle_status
from core.mesh_bundle import MeshBundle
from core.node_hands import LEFT


@pytest.fixture()
def bundle():
    tmp = tempfile.mkdtemp(prefix="nsm_hands_extra_test_")
    try:
        yield MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _a_node(b):
    ids = list(b.role_node_ids.values())
    return b.registry.get(ids[0])


# ── kaggle_status (نسخة سريعة عبر اليد) ───────────────────────────────────

def test_kaggle_status_bound_to_left_hand(bundle):
    a = _a_node(bundle)
    names = {t["name"] for t in a.hands.tools()[LEFT]}
    assert "kaggle_status" in names


def test_kaggle_status_via_hand_completes_without_hand_timeout(bundle):
    """الاختبار الحقيقي هنا: حتى لو كانت حالة الكيرنل COMPLETE (تُفعِّل
    فرع جلب المخرجات البطيء في at.kaggle_status الأصلية)، النسخة المربوطة
    باليد (fetch_output=False) يجب ألا تستدعي ذلك الفرع إطلاقاً — ولا
    تُقطَع بمهلة اليد (15 ثانية افتراضياً)."""
    a = _a_node(bundle)

    def fake_run(cmd, timeout=120, cwd=None):
        if "status" in cmd:
            return 0, "COMPLETE"
        raise AssertionError("fetch_output=False يجب ألا يستدعي فرع جلب المخرجات إطلاقاً")

    with patch("shutil.which", return_value="/usr/bin/kaggle"), \
         patch("ai.agent_tools._run", side_effect=fake_run):
        res = a.use_hand(LEFT, "kaggle_status", slug="someone/some-kernel")

    assert res.ok, res.error
    assert res.output["status"] == "COMPLETE"
    assert "tail_output" not in res.output


def test_kaggle_status_quick_skips_slow_output_fetch_branch():
    """اختبار وحدة مباشر على ai.agent_tools.kaggle_status نفسها (بمعزل عن
    اليد): fetch_output=False يجب ألا يستدعي فرع 'kaggle kernels output'
    البطيء إطلاقاً، حتى لو كانت الحالة نهائية (COMPLETE/ERROR/FAILED)."""
    calls = []

    def fake_run(cmd, timeout=120, cwd=None):
        calls.append(cmd)
        if cmd[:2] == ["kaggle", "kernels"] and cmd[2] == "status":
            return 0, "ERROR"
        raise AssertionError(f"لم يكن يجب استدعاء: {cmd}")

    with patch("shutil.which", return_value="/usr/bin/kaggle"), \
         patch("ai.agent_tools._run", side_effect=fake_run):
        result = kaggle_status("someone/some-kernel", fetch_output=False)

    assert result["status"] == "ERROR"
    assert "tail_output" not in result
    assert len(calls) == 1, "استدعاء واحد فقط (status) بلا أي استدعاء لـ output"


def test_kaggle_status_default_behavior_unchanged_for_existing_callers():
    """fetch_output افتراضياً True — أي مستدعٍ قائم لـ at.kaggle_status()
    بلا هذا المعامل الجديد يجب ألا يتأثر سلوكه إطلاقاً."""
    def fake_run(cmd, timeout=120, cwd=None):
        if cmd[2] == "status":
            return 0, "COMPLETE"
        return 0, "ok"

    with patch("shutil.which", return_value="/usr/bin/kaggle"), \
         patch("ai.agent_tools._run", side_effect=fake_run), \
         patch("pathlib.Path.iterdir", return_value=iter([])):
        result = kaggle_status("someone/some-kernel")  # بلا fetch_output

    assert result["status"] == "COMPLETE"
    assert "tail_output" in result  # الفرع البطيء نُفِّذ كالمعتاد


def test_kaggle_status_denied_when_node_is_paused():
    """الضمان الثابت 'عقدة محجورة لا تستخدم أي يد إطلاقاً' يسبق أي أداة
    فردية، بما فيها kaggle_status."""
    tmp = tempfile.mkdtemp(prefix="nsm_hands_extra_test2_")
    try:
        b = MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db")
        a = _a_node(b)
        a.pause(reason="manual")
        res = a.use_hand(LEFT, "kaggle_status")
        assert res.denied
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
