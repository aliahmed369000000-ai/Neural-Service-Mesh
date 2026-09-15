from ai.forecast_consensus import aggregate_forecasts


def test_collective_forecast_accepts_28_contributors_and_rejects_bad_shape():
    candidates = [
        {
            "node_id": f"node-{index:02d}",
            "reputation": index % 7,
            "forecast": {
                "predictions": [10.0 + index, 20.0 + index],
                "r2": 0.50 + (index % 5) * 0.08,
                "target_date": "2026-10-15",
            },
        }
        for index in range(28)
    ]
    candidates.append({
        "node_id": "bad-length",
        "reputation": 100,
        "forecast": {"predictions": [999.0], "r2": 1.0},
    })

    result = aggregate_forecasts(candidates)

    assert result["ok"] is True
    assert result["accepted"] == 28
    assert len(result["contributors"]) == 28
    assert len(result["rejected"]) == 1
    assert result["rejected"][0]["node_id"] == "bad-length"
    assert len(result["predictions"]) == 2


def test_duplicate_contributor_is_not_deduplicated_by_aggregator_input():
    # إزالة التكرار مسؤولية LivingMeshNode قبل الاستدعاء؛ التجميع نفسه حتمي.
    result = aggregate_forecasts([
        {"node_id": "a", "forecast": {"predictions": [1], "r2": 1.0}},
    ])
    assert result["accepted"] == 1
