import asyncio
import json
import os
import unittest

from ai.node_launcher import handle_control_task


class FakeNode:
    node_id = "coordinator"

    def _load_state(self):
        return {"nodes": {"mesh_node_05": {
            "host": "node-05.example", "port": 443,
            "status": "online", "public_key": "PUBLIC-KEY",
        }}}

    def _public_key_for_id(self, node_id):
        return b"PUBLIC-KEY" if node_id == "mesh_node_05" else None


class FakeRequest:
    def __init__(self, body, token=None):
        self.app = {"node": FakeNode()}
        self.headers = {"X-NSM-Control-Token": token} if token else {}
        self._body = body

    async def json(self):
        return self._body


class ControlEndpointTests(unittest.TestCase):
    def setUp(self):
        os.environ["NSM_CONTROL_TOKEN"] = "test-control-token"

    def tearDown(self):
        os.environ.pop("NSM_CONTROL_TOKEN", None)

    def call(self, body, token="test-control-token"):
        response = asyncio.run(handle_control_task(FakeRequest(body, token)))
        return response.status, json.loads(response.body)

    def test_requires_authentication(self):
        status, body = self.call({"kind": "capability_report", "targets": ["mesh_node_05"]}, token=None)
        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "control_auth_required")

    def test_dry_run_returns_safe_plan(self):
        status, body = self.call({
            "task_id": "control-test-1", "kind": "encrypted_rpc_roundtrip",
            "targets": ["mesh_node_05"], "parameters": {"text": "probe"},
        })
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["mode"], "dry_run")
        self.assertTrue(body["plan"]["safety"]["allowlist"])
        self.assertEqual(body["plan"]["targets"][0]["id"], "mesh_node_05")

    def test_health_and_forecast_dry_run_are_allowed(self):
        status, body = self.call({
            "kind": "mesh_health_report", "targets": ["mesh_node_05"], "dry_run": True,
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["plan"]["kind"], "mesh_health_report")

        status, body = self.call({
            "kind": "forecast_consensus_benchmark", "targets": ["mesh_node_05"],
            "parameters": {"series": [1, 2, 3, 4], "horizon": 2}, "dry_run": True,
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["plan"]["kind"], "forecast_consensus_benchmark")

    def test_forecast_parameters_are_bounded(self):
        status, body = self.call({
            "kind": "forecast_consensus_benchmark", "targets": ["mesh_node_05"],
            "parameters": {"series": [1], "horizon": 2},
        })
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "series_must_be_list_with_2_to_256_values")

        status, body = self.call({
            "kind": "forecast_consensus_benchmark", "targets": ["mesh_node_05"],
            "parameters": {"series": [1, 2, 3], "horizon": 11},
        })
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "horizon_must_be_between_1_and_10")

    def test_rejects_unknown_kind_and_unknown_target(self):
        status, body = self.call({"kind": "shell", "targets": ["mesh_node_05"]})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "unsupported_control_task")
        status, body = self.call({"kind": "capability_report", "targets": ["unknown"]})
        self.assertEqual(status, 400)
        self.assertTrue(body["error"].startswith("target_not_resolvable:"))

    def test_rejects_more_than_28_targets(self):
        targets = [f"node-{i}" for i in range(29)]
        status, body = self.call({"kind": "capability_report", "targets": targets})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "targets_must_be_non_empty_list_max_28")


if __name__ == "__main__":
    unittest.main()
