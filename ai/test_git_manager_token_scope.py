# -*- coding: utf-8 -*-
"""GitManager: HF_TOKEN لا يُستخدم أبداً كتوكن GitHub + تخطّي التطوّر بلا توكن
+ تخطّي الدفع التلقائي بلا تفعيل صريح NSM_ALLOW_AUTO_EVOLUTION_PUSH=1 (حتى
مع وجود GITHUB_TOKEN) — راجع تعليق can_push في ai/git_manager.py لسبب هذه
البوابة الإضافية: GITHUB_TOKEN وحده كان كافياً سابقاً لدفع تلقائي كامل
لـmain بلا أي مراجعة بشرية، وقد حدث فعلاً (عشرات كوميتات "🧬 NSM Evolution"
في تاريخ هذا المستودع الحقيقي)."""
from unittest.mock import patch

from ai.git_manager import GitManager


def test_hf_token_never_used_for_github(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_should_not_leak")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    gm = GitManager()
    assert gm.token is None and gm.can_push is False
    assert "hf_secret_should_not_leak" not in gm._get_auth_url()


def test_github_token_alone_is_not_enough_to_push(monkeypatch):
    """توكن صالح وحده بلا NSM_ALLOW_AUTO_EVOLUTION_PUSH=1 → can_push=False
    (البوابة الجديدة: fail-closed افتراضياً للدفع التلقائي تحديداً — الاستنساخ/
    القراءة لا تتأثران، هذا خاص بـcommit_and_push عبر can_push فقط)."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    monkeypatch.delenv("NSM_ALLOW_AUTO_EVOLUTION_PUSH", raising=False)
    gm = GitManager()
    assert gm.token == "ghp_x"  # التوكن نفسه مقروء بشكل صحيح
    assert gm.can_push is False  # لكن الدفع ممنوع بلا التفعيل الصريح


def test_github_token_used_when_auto_push_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    monkeypatch.setenv("HF_TOKEN", "hf_y")
    monkeypatch.setenv("NSM_ALLOW_AUTO_EVOLUTION_PUSH", "1")
    gm = GitManager()
    assert gm.can_push is True and "ghp_x@" in gm._get_auth_url()
    assert "hf_y" not in gm._get_auth_url()


def test_apply_evolution_skips_without_token_no_clone(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("NSM_ALLOW_AUTO_EVOLUTION_PUSH", "1")
    gm = GitManager()
    with patch.object(GitManager, "clone", side_effect=AssertionError("لا استنساخ")):
        assert gm.apply_evolution("مهمة") is False


def test_apply_evolution_skips_with_token_but_auto_push_not_enabled_no_clone(monkeypatch):
    """نفس اختبار «بلا توكن أعلاه» لكن بالاتجاه المعاكس: توكن موجود وصالح،
    لكن بلا NSM_ALLOW_AUTO_EVOLUTION_PUSH=1 — يجب ألا يحدث استنساخ أو دفع
    على الإطلاق (لا فقط فشل الدفع بعد استنساخ مكلف بلا فائدة)."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    monkeypatch.delenv("NSM_ALLOW_AUTO_EVOLUTION_PUSH", raising=False)
    gm = GitManager()
    with patch.object(GitManager, "clone", side_effect=AssertionError("لا استنساخ")):
        assert gm.apply_evolution("مهمة") is False


def test_apply_evolution_runs_with_token_and_auto_push_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    monkeypatch.setenv("NSM_ALLOW_AUTO_EVOLUTION_PUSH", "1")
    gm = GitManager()
    with patch.object(GitManager, "clone", return_value=str(tmp_path)), \
         patch.object(GitManager, "commit_and_push") as cp:
        assert gm.apply_evolution("مهمة") is True
        cp.assert_called_once()


def test_living_node_skips_evolution_without_token(tmp_path, monkeypatch):
    import ai.living_mesh as lm
    from pathlib import Path
    d = Path(tmp_path)
    (d / "content").mkdir()
    monkeypatch.setattr(lm, "LIVING_MESH_DIR", d)
    monkeypatch.setattr(lm, "NETWORK_STATE", d / "network_state.json")
    monkeypatch.setattr(lm, "CONTENT_DIR", d / "content")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    node = lm.LivingMeshNode(node_id="skip_node", host="127.0.0.1", port=0)
    before = node.local_evolution_score
    import asyncio
    asyncio.run(node._execute_evolution({"task": "t"}))
    assert node.local_evolution_score == before  # لا "completed" كاذب
