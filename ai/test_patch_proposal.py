import tempfile
import unittest
from pathlib import Path

from ai.patch_proposal import PatchRejected, PatchProposal, apply_patch_atomically


class PatchProposalTests(unittest.TestCase):
    def test_accepts_verified_patch(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "agent.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            proposal = PatchProposal.from_file(root, "agent.py", "VALUE = 2\n", "رفع قيمة الاختبار")
            result = apply_patch_atomically(root, proposal, verifier=lambda *_: True)
            self.assertEqual(result.status, "accepted")
            self.assertEqual(target.read_text(encoding="utf-8"), "VALUE = 2\n")

    def test_rejects_stale_proposal_without_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "agent.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            proposal = PatchProposal.from_file(root, "agent.py", "VALUE = 2\n", "تعديل")
            target.write_text("VALUE = 99\n", encoding="utf-8")
            with self.assertRaisesRegex(PatchRejected, "تغير الملف"):
                apply_patch_atomically(root, proposal)
            self.assertEqual(target.read_text(encoding="utf-8"), "VALUE = 99\n")

    def test_rolls_back_when_verification_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "agent.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            proposal = PatchProposal.from_file(root, "agent.py", "VALUE = broken\n", "اختبار فشل")
            result = apply_patch_atomically(root, proposal, verifier=lambda *_: False)
            self.assertEqual(result.status, "rolled_back")
            self.assertTrue(result.rolled_back)
            self.assertEqual(target.read_text(encoding="utf-8"), "VALUE = 1\n")

    def test_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(PatchRejected, "خارج"):
                PatchProposal.from_file(Path(raw), "../escape.py", "x = 1\n", "غير آمن")


if __name__ == "__main__":
    unittest.main()
