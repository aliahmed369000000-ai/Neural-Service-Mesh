# -*- coding: utf-8 -*-
"""تشديد ai/terminal_auto_policy.py: الطرفية الآمنة على يد العقدة (للقراءة فقط)
كانت تسمح بـ path traversal وقراءة .git/config (توكن الـremote) و.env وبكتابة
فروع git وبتشغيل unittest discover في أي مسار."""
from __future__ import annotations

import os

import pytest

from ai.terminal_auto_policy import decide, run_auto


@pytest.mark.parametrize("cmd", [
    "head -5 ../../etc/passwd", "ls ..", "wc -l ../../../etc/shadow",
    "head /etc/passwd", "head -5 /home/workdir/x", "head ~/.ssh/id_rsa",
    "head .git/config", "tail -5 .env", "head .env.local", "head .streamlit/secrets.toml",
    "ls .git", "head data/api_token.json", "head certs/server.pem", "head config/credentials.json",
    "python3 -m py_compile ../x.py", "python -m compileall ..",
])
def test_path_traversal_and_secret_files_blocked(cmd):
    assert not decide(cmd).allowed, cmd


@pytest.mark.parametrize("cmd", [
    "git branch -D main", "git branch -d x", "git branch pwn", "git branch -m main x",
    "git branch -f main HEAD", "git branch --set-upstream-to=origin/main",
    "git branch --delete main", "git branch --edit-description",
])
def test_git_branch_cannot_write(cmd):
    assert not decide(cmd).allowed, cmd


@pytest.mark.parametrize("cmd", [
    "git show HEAD:.env", "git show HEAD:.streamlit/secrets.toml", "git diff .env",
    "git log .env", "git diff --output=/tmp/x", "git diff --no-index /etc/passwd x",
    "git status --ignored", "git ls-files --others --exclude-from=/etc/passwd",
    "git log -p", "git show HEAD -- .env",
])
def test_git_reads_cannot_reach_secrets_or_write(cmd):
    assert not decide(cmd).allowed, cmd


@pytest.mark.parametrize("cmd", [
    "python -m unittest discover -s /tmp", "python -m unittest discover",
    "python -m unittest -s /tmp x", "python3 -m unittest ../evil",
    "pytest /tmp/test_x.py", "pytest ai/living_mesh.py", "pytest ../x/test_y.py",
])
def test_test_runners_cannot_run_arbitrary_paths(cmd):
    assert not decide(cmd).allowed, cmd


@pytest.mark.parametrize("cmd", [
    "git status", "git status -sb", "git diff --stat", "git diff --check", "git diff HEAD",
    "git log --oneline -10", "git branch -v", "git branch --show-current", "git rev-parse HEAD",
    "git show HEAD --stat", "git ls-files", "head -20 README.md", "ls tests",
    "wc -l tests/test_node_hands.py", "pytest -q tests/", "pytest tests/test_x.py::test_y -q",
    "pytest ai/test_video_job_manager_resume.py -q", "python -m py_compile core/mesh_bundle.py",
    "python -m unittest tests.test_x -v", "python3 --version",
])
def test_legitimate_read_and_check_commands_still_allowed(cmd):
    assert decide(cmd).allowed, (cmd, decide(cmd).reason)


def test_symlink_to_secret_or_outside_is_blocked_at_run_time(tmp_path):
    (tmp_path / ".env").write_text("API_KEY=hunter2\n")
    outside = tmp_path.parent / f"{tmp_path.name}_outside.txt"
    outside.write_text("outside\n")
    os.symlink(tmp_path / ".env", tmp_path / "notes.txt")
    os.symlink(outside, tmp_path / "link.txt")
    (tmp_path / "ok.txt").write_text("fine\n")

    assert "hunter2" not in run_auto("head notes.txt", cwd=str(tmp_path))
    assert run_auto("head notes.txt", cwd=str(tmp_path)).startswith("مرفوض")
    assert run_auto("head link.txt", cwd=str(tmp_path)).startswith("مرفوض")
    ok = run_auto("head ok.txt", cwd=str(tmp_path))
    assert ok.startswith("exit=0") and "fine" in ok
