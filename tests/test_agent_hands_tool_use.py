"""وكلاء السرب (AgentInstance عبر NSMAgent.run) كانوا منفصلين تماماً عن
«يدَي» العقدة (core/node_hands.py) — لا طريقة لهم لاستخدام أي أداة حقيقية
أثناء تنفيذ مهمة فعلية. هذا الاختبار يثبّت الربط: الوكيل يستخدم اليد
اليسرى «إن كان مناسباً» (استدعاءان LLM كحد أقصى)، وبلا أي يدين السلوك
مطابق تماماً لما كان قبل هذه الميزة."""
from types import SimpleNamespace
from unittest.mock import patch, Mock

from ai.agent_factory import AgentFactory
from ai.nsm_agent_core import NSMAgent
from core.node import BaseNode, NodeSchema
from core.node_hands import NodeHands, LEFT


class N(BaseNode):
    input_schema = NodeSchema(fields={}, required=[])
    output_schema = NodeSchema(fields={}, required=[])
    def process(self, data):
        return {}


def _fake_result(text):
    return SimpleNamespace(text=text, provider=SimpleNamespace(value="groq"))


def _agent_with_mock_llm(*texts):
    """llm_fallback خاصية lazy بلا setter — نضع الـMock مباشرة في الحقل
    الخاص الذي تتحقق منه الخاصية، فتتجنب أي استيراد حقيقي."""
    agent = NSMAgent()
    lf = Mock()
    lf.generate.side_effect = [_fake_result(t) for t in texts] if len(texts) > 1 else None
    if len(texts) == 1:
        lf.generate.return_value = _fake_result(texts[0])
    agent._llm_fallback = lf
    return agent, lf


def test_no_hands_is_single_call_unchanged_behavior():
    agent, lf = _agent_with_mock_llm("جواب مباشر")
    out = agent.run("مهمة بسيطة")
    assert out == "جواب مباشر" and lf.generate.call_count == 1


def test_hands_with_no_left_tools_is_single_call_unchanged():
    node = N("n")
    node.attach_hands(NodeHands(node))  # لا أدوات مربوطة
    agent, lf = _agent_with_mock_llm("جواب مباشر")
    out = agent.run("مهمة", hands=node.hands)
    assert out == "جواب مباشر" and lf.generate.call_count == 1


def test_model_requests_tool_and_gets_final_answer():
    node = N("n")
    hands = NodeHands(node)
    hands.bind(LEFT, "echo", lambda msg: {"echoed": msg}, "يكرر النص")
    node.attach_hands(hands)
    agent, lf = _agent_with_mock_llm(
        'TOOL: echo ARGS: {"msg": "hi"}', "الجواب النهائي بعد الأداة",
    )
    out = agent.run("مهمة تحتاج أداة", hands=hands)
    assert out == "الجواب النهائي بعد الأداة" and lf.generate.call_count == 2
    second_prompt = lf.generate.call_args_list[1][0][0]
    assert "echoed" in second_prompt and "hi" in second_prompt


def test_model_answers_directly_without_tool_is_single_call():
    node = N("n")
    hands = NodeHands(node)
    hands.bind(LEFT, "echo", lambda msg: msg, "يكرر النص")
    node.attach_hands(hands)
    agent, lf = _agent_with_mock_llm("لم أحتج أداة، هذا جوابي")
    out = agent.run("مهمة", hands=hands)
    assert out == "لم أحتج أداة، هذا جوابي" and lf.generate.call_count == 1


def test_paused_node_denies_tool_use_and_falls_back_gracefully():
    node = N("n")
    hands = NodeHands(node)
    hands.bind(LEFT, "echo", lambda msg: msg, "يكرر النص")
    node.attach_hands(hands)
    node.pause(reason="quarantine")
    agent, lf = _agent_with_mock_llm(
        'TOOL: echo ARGS: {"msg": "hi"}', "جواب احتياطي بلا الأداة",
    )
    out = agent.run("مهمة", hands=hands)
    assert out == "جواب احتياطي بلا الأداة" and lf.generate.call_count == 2


def test_malformed_tool_directive_falls_back_to_first_response():
    node = N("n")
    hands = NodeHands(node)
    hands.bind(LEFT, "echo", lambda msg: msg, "يكرر النص")
    node.attach_hands(hands)
    agent, lf = _agent_with_mock_llm("TOOL: echo ARGS: {not json}")
    out = agent.run("مهمة", hands=hands)
    assert out == "TOOL: echo ARGS: {not json}" and lf.generate.call_count == 1


def test_agent_factory_run_task_passes_hands_through():
    factory = AgentFactory()
    node = N("n")
    hands = NodeHands(node)
    hands.bind(LEFT, "echo", lambda msg: msg, "يكرر النص")
    node.attach_hands(hands)
    captured = {}

    class FakeAgent:
        @staticmethod
        def run(task, hands=None):
            captured["hands"] = hands
            return "ok"

    with patch("ai.nsm_agent_core.NSMAgent", return_value=FakeAgent()):
        factory.run_task(AgentFactory.available_roles()[0], "مهمة", hands=hands)
    assert captured["hands"] is hands


def test_swarm_coordinator_resolves_hands_for_role(tmp_path):
    """التكامل الكامل: SwarmCoordinator._run_task يحصل فعلياً على يد
    الدور الحقيقية من MeshBundle، لا None دائماً."""
    from core.mesh_bundle import MeshBundle
    mesh = MeshBundle(storage_dir=str(tmp_path / "s"), db_path=str(tmp_path / "m.db"))
    role = next(iter(mesh.role_node_ids))
    hands = mesh._hands_for_role(role)
    assert hands is not None
    assert hands.tools()["left"]  # أدوات حقيقية مربوطة فعلاً (search_code...)
    assert mesh._hands_for_role("role-does-not-exist") is None
    # مقارنة الدالة المربوطة نفسها (bound method) لا تُطابق بـ`is` عبر
    # وصولين منفصلين؛ المهم أنها نفس الكائن المربوط لنفس التابع.
    assert mesh.coordinator._role_hands.__func__ is mesh._hands_for_role.__func__
    assert mesh.coordinator._role_hands.__self__ is mesh
