import unittest

from ai.mesh_task_protocol import KIND_TEMPORAL_FORECAST, dispatch_task
from ai.temporal_forecast import forecast_timeline


class TemporalForecastTests(unittest.TestCase):
    def setUp(self):
        self.observations = [
            {"date": "2026-10-01", "value": 10},
            {"date": "2026-10-05", "value": 14},
            {"date": "2026-10-10", "value": 19},
            {"date": "2026-10-12", "value": 21},
        ]

    def test_upward_forecast_and_uncertainty(self):
        result = forecast_timeline(self.observations, ["2026-10-15"], "2026-10-15")
        self.assertTrue(result["ok"])
        self.assertEqual(result["direction"], "صاعد")
        self.assertTrue(result["target_in_future"])
        point = result["forecasts"][0]
        self.assertLessEqual(point["lower_95"], point["value"])
        self.assertLessEqual(point["value"], point["upper_95"])

    def test_protocol_dispatch(self):
        result = dispatch_task(KIND_TEMPORAL_FORECAST, {
            "task_id": "t-1",
            "observations": self.observations,
            "future_dates": ["2026-10-15", "2026-10-20"],
        })
        self.assertTrue(result["ok"])
        self.assertEqual(result["task_id"], "t-1")
        self.assertEqual(len(result["forecasts"]), 2)

    def test_rejects_insufficient_data(self):
        result = dispatch_task(KIND_TEMPORAL_FORECAST, {
            "observations": self.observations[:2],
            "future_dates": ["2026-10-15"],
        })
        self.assertFalse(result["ok"])
        self.assertIn("3 observations", result["error"])


if __name__ == "__main__":
    unittest.main()
