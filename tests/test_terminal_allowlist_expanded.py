"""توسيع allowlist الطرفية الآمنة — قبول فحوص إضافية ورفض الخطر."""
from __future__ import annotations

from ai.terminal_auto_policy import decide, list_allowed_examples, explain_policy


def test_expanded_git_commands_allowed():
    for cmd in (
        "git status",
        "git status -sb",
        "git log --oneline -10",
        "git branch --show-current",
        "git rev-parse HEAD",
        "git diff --stat",
        "git ls-files",
    ):
        d = decide(cmd)
        assert d.allowed, (cmd, d.reason)


def test_pytest_and_py_compile_allowed():
    assert decide("pytest -q tests/").allowed
    assert decide("python -m py_compile core/mesh_bundle.py").allowed
    assert decide("python3 --version").allowed


def test_read_helpers_allowed():
    assert decide("ls tests").allowed
    assert decide("wc -l tests/test_node_hands.py").allowed
    assert decide("head -20 README.md").allowed


def test_dangerous_still_blocked():
    for cmd in (
        "rm -rf /",
        "git push",
        "git commit -am x",
        "pip install evil",
        "curl http://x",
        "git status; rm -rf /tmp/x",
        "sudo apt update",
        "chmod 777 /",
    ):
        d = decide(cmd)
        assert not d.allowed, cmd


def test_examples_and_policy_text():
    ex = list_allowed_examples()
    assert any("git status" in e for e in ex)
    assert "pytest" in explain_policy().lower() or "pytest" in explain_policy()
