import asyncio
import json
import os
import types
import unittest

from ai import mesh_task_protocol as mt
from ai.node_launcher import CONTROL_TASK_KINDS, handle_control_task


class FakeLiveLLM:
    def generate(self, query, system_prompt):
        return types.SimpleNamespace(
            text="القرار: نفّذ الاختبار على عقدتين. الأسباب: المفتاحان متبادلان. التحقق: قِس ACK.",
            provider=types.SimpleNamespace(value="groq"),
            model="test-live-model",
            error=None,
            tried=["groq:ok"],
        )


class FakeFallbackLLM:
    def generate(self, query, system_prompt):
        return types.SimpleNamespace(
            text="إجابة تركيبية محلية",
            provider=types.SimpleNamespace(value="ckg_synthesis"),
            model="ckg-synthesis-v1",
            error="no_live_provider",
            tried=["ckg_synthesis"],
        )


class FakeMesh:
    def _load_state(self):
        return {"nodes": {"mesh_node_01": {
            "host": "node-01.example", "port": 443,
            "status": "online", "public_key": "PK",
        }}}

    def _public_key_for_id(self, node_id):
        return b"PK"


class Req:
    def __init__(self, body):
        self.app = {"node": FakeMesh()}
        self.headers = {"X-NSM-Control-Token": "reason-token"}
        self.body = body

    async def json(self):
        return self.body


class ReasoningTaskTests(unittest.TestCase):
    def setUp(self):
        os.environ["NSM_CONTROL_TOKEN"] = "reason-token"

    def tearDown(self):
        os.environ.pop("NSM_CONTROL_TOKEN", None)

    def test_live_reasoning_is_reported_as_live(self):
        out = mt.execute_reasoning_request(
            {"task_id": "r1", "prompt": "اختر اختبارًا آمنًا للشبكة", "require_live": True},
            llm_factory=FakeLiveLLM,
        )
        self.assertTrue(out["ok"])
        self.assertTrue(out["live_llm"])
        self.assertEqual(out["provider"], "groq")
        self.assertIn("ACK", out["answer"])

    def test_ckg_fallback_is_not_claimed_as_live(self):
        out = mt.execute_reasoning_request(
            {"task_id": "r2", "prompt": "حلل الحالة", "require_live": True},
            llm_factory=FakeFallbackLLM,
        )
        self.assertFalse(out["ok"])
        self.assertFalse(out["live_llm"])
        self.assertEqual(out["error"], "live_llm_unavailable")

    def test_protocol_registration_and_control_dry_run(self):
        self.assertIn(mt.KIND_REASONING, mt.ALL_TASK_KINDS)
        self.assertEqual(mt.result_kind_for(mt.KIND_REASONING), mt.KIND_REASONING_RESULT)
        self.assertIn("reasoning_benchmark", CONTROL_TASK_KINDS)
        response = asyncio.run(handle_control_task(Req({
            "kind": "reasoning_benchmark",
            "targets": ["mesh_node_01"],
            "parameters": {
                "prompt": "حلل هل نحتاج فحص E2E؟",
                "context": "لدينا عقدة ومفتاح عام متبادل.",
                "require_live": True,
            },
            "dry_run": True,
        })))
        body = json.loads(response.body)
        self.assertEqual(response.status, 200)
        self.assertEqual(body["mode"], "dry_run")
        self.assertEqual(body["plan"]["kind"], "reasoning_benchmark")

    def test_prompt_is_bounded(self):
        out = mt.execute_reasoning_request({"task_id": "r3", "prompt": ""}, llm_factory=FakeLiveLLM)
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "prompt_must_be_1_to_4000_characters")


if __name__ == "__main__":
    unittest.main()
