# -*- coding: utf-8 -*-
"""GitManager: HF_TOKEN لا يُستخدم أبداً كتوكن GitHub + تخطّي التطوّر بلا توكن."""
from unittest.mock import patch

from ai.git_manager import GitManager


def test_hf_token_never_used_for_github(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_should_not_leak")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    gm = GitManager()
    assert gm.token is None and gm.can_push is False
    assert "hf_secret_should_not_leak" not in gm._get_auth_url()


def test_github_token_used(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    monkeypatch.setenv("HF_TOKEN", "hf_y")
    gm = GitManager()
    assert gm.can_push is True and "ghp_x@" in gm._get_auth_url()
    assert "hf_y" not in gm._get_auth_url()


def test_apply_evolution_skips_without_token_no_clone(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    gm = GitManager()
    with patch.object(GitManager, "clone", side_effect=AssertionError("لا استنساخ")):
        assert gm.apply_evolution("مهمة") is False


def test_apply_evolution_runs_with_token(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
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
