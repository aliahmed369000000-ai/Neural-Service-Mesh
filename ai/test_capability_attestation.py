import unittest

from ai.capability_attestation import collect_capabilities


class CapabilityAttestationTests(unittest.TestCase):
    def test_probe_is_structured_and_conservative(self):
        report = collect_capabilities()
        self.assertEqual(report["source"], "local_probe")
        self.assertTrue(report["attestation_id"])
        self.assertIn("CPU", report["capabilities"])
        self.assertIn("storage", report["capabilities"])
        self.assertIn("cuda", report["facts"])
        self.assertIsInstance(report["facts"]["cuda"]["available"], bool)
        if not report["facts"]["cuda"]["available"]:
            self.assertNotIn("GPU_HIGH", report["capabilities"])

    def test_attestation_id_ignores_measurement_time(self):
        first = collect_capabilities()
        second = collect_capabilities()
        self.assertEqual(first["attestation_id"], second["attestation_id"])


if __name__ == "__main__":
    unittest.main()
