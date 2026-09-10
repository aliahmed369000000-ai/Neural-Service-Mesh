import unittest
from unittest.mock import patch

from ai.living_mesh import LivingMeshNode


class CapabilityWatchTests(unittest.TestCase):
    def _node(self):
        node = LivingMeshNode.__new__(LivingMeshNode)
        node.node_id = "node-test"
        node.node_info = {"capabilities": ["CPU"], "capability_attestation": {"attestation_id": "old"}}
        node._capability_attestation = None
        node.events = []
        node.saved = []
        node._load_state = lambda: {"nodes": {"node-test": dict(node.node_info)}}
        node._save_state = lambda state: node.saved.append(state)
        node.sync_experience = lambda kind, data: node.events.append((kind, data))
        return node

    def test_change_updates_state_and_gossips_once(self):
        node = self._node()
        reports = [
            {"attestation_id": "a", "capabilities": ["CPU"], "facts": {}, "measured_at": 1, "source": "test"},
            {"attestation_id": "a", "capabilities": ["CPU"], "facts": {}, "measured_at": 2, "source": "test"},
            {"attestation_id": "b", "capabilities": ["CPU", "GPU_LOW"], "facts": {}, "measured_at": 3, "source": "test"},
        ]
        with patch("ai.living_mesh.collect_capabilities", side_effect=reports):
            self.assertTrue(node.refresh_capability_attestation())
            self.assertFalse(node.refresh_capability_attestation())
            self.assertTrue(node.refresh_capability_attestation())
        self.assertEqual(node.node_info["capabilities"], ["CPU", "GPU_LOW"])
        self.assertEqual(len(node.events), 1)
        self.assertEqual(node.events[0][0], "capability_attestation_updated")
        self.assertEqual(node.events[0][1]["previous_id"], "a")
        self.assertEqual(node.events[0][1]["attestation_id"], "b")
        self.assertEqual(len(node.saved), 2)


if __name__ == "__main__":
    unittest.main()
