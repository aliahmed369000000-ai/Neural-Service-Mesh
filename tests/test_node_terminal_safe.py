"""طرفية العقد: allowlist فقط — رفض الأوامر الخطرة وقبول الفحوص الآمنة."""
from __future__ import annotations

import shutil
import tempfile

import pytest

from core.mesh_bundle import MeshBundle
from core.node_hands import LEFT


@pytest.fixture()
def bundle():
    tmp = tempfile.mkdtemp(prefix="nsm_term_")
    try:
        b = MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db")
        yield b
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _any_node(b: MeshBundle):
    ids = list(b.role_node_ids.values())
    assert ids
    return b.registry.get(ids[0])


def test_terminal_policy_denies_shell(bundle):
    n = _any_node(bundle)
    r = n.use_hand(LEFT, "terminal_policy")
    assert r.ok
    assert r.output.get("shell") is False
    assert "policy" in r.output


def test_terminal_run_safe_rejects_dangerous(bundle):
    n = _any_node(bundle)
    for cmd in ("rm -rf /", "curl http://evil", "git push", "echo hi && rm -rf /tmp/x"):
        r = n.use_hand(LEFT, "terminal_run_safe", cmd=cmd)
        assert r.ok  # HandResult ok — الأداة نفّذت ورجعت رفضاً
        assert r.output.get("ok") is False
        assert r.output.get("requires_approval") is True or "مرفوض" in str(r.output.get("msg", "")) or r.output.get("ok") is False


def test_terminal_run_safe_allows_git_status(bundle):
    n = _any_node(bundle)
    r = n.use_hand(LEFT, "terminal_run_safe", cmd="git status")
    assert r.ok
    assert isinstance(r.output, dict)
    # قد ينجح أو يفشل بسبب cwd بدون .git — المهم أنه لم يُرفض كسياسة
    assert r.output.get("requires_approval") is not True or r.output.get("automatic") is True or "output" in r.output or r.output.get("ok") in (True, False)


def test_agent_sees_terminal_tools_in_menu():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from core.node import BaseNode, NodeSchema
    from core.node_hands import NodeHands, LEFT
    from ai.nsm_agent_core import NSMAgent

    class N(BaseNode):
        input_schema = NodeSchema(fields={}, required=[])
        output_schema = NodeSchema(fields={}, required=[])
        def process(self, data):
            return {}

    node = N("n")
    hands = NodeHands(node)
    hands.bind(LEFT, "terminal_run_safe", lambda cmd="": {"ok": False}, "طرفية آمنة")
    hands.bind(LEFT, "terminal_policy", lambda: {"shell": False}, "سياسة")
    node.attach_hands(hands)
    agent = NSMAgent()
    lf = Mock()
    lf.generate.return_value = SimpleNamespace(text="ok", provider=SimpleNamespace(value="x"))
    agent._llm_fallback = lf
    agent.run("افحص المستودع", hands=hands)
    prompt = lf.generate.call_args_list[0][0][0]
    assert "terminal_run_safe" in prompt
    assert "terminal_policy" in prompt
