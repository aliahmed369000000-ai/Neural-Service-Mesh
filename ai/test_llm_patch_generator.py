import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai.llm_patch_generator import LLMPatchGenerator, PatchGenerationError
from ai.nsm_agent_core import NSMAgent


class FakeLLM:
    def __init__(self, text):
        self.text = text

    def generate(self, query, system_prompt=None):
        return SimpleNamespace(text=self.text)


class LLMPatchGeneratorTests(unittest.TestCase):
    def test_generates_proposal_from_structured_model_output(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "agent.py"
            old = "VALUE = 1\n"
            new = "VALUE = 2\n"
            target.write_text(old, encoding="utf-8")
            response = json.dumps({
                "target": "agent.py", "old_text": old, "new_text": new,
                "reason": "إصلاح القيمة", "tests": ["py_compile"],
            })
            proposal = LLMPatchGenerator(FakeLLM(response)).generate(
                root, "agent.py", "تصحيح القيمة", ("py_compile",)
            )
            self.assertEqual(proposal.old_text, old)
            self.assertEqual(proposal.new_text, new)
            self.assertEqual(proposal.target, "agent.py")

    def test_rejects_model_output_for_different_source(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "agent.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            response = json.dumps({
                "target": "agent.py", "old_text": "VALUE = 0\n",
                "new_text": "VALUE = 2\n", "reason": "تعديل", "tests": [],
            })
            with self.assertRaises(PatchGenerationError):
                LLMPatchGenerator(FakeLLM(response)).generate(root, "agent.py", "تصحيح")

    def test_core_compile_verifier_is_local_only(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "agent.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            proposal = SimpleNamespace(target="agent.py")
            self.assertTrue(NSMAgent._patch_verifier(root, proposal))


if __name__ == "__main__":
    unittest.main()
