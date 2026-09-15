"""تجميع آمن وموزون لتنبؤات العقد داخل NSM.

هذه الوحدة حتمية ولا تتصل بالإنترنت؛ تستقبل نتائج محلية أو رسائل موثقة بعد
أن يتحقق LivingMeshNode من توقيعها ومكافحة إعادة الإرسال.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List

MAX_FORECAST_POINTS = 128
MIN_R2 = -1.0
MAX_R2 = 1.0


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result and abs(result) != float("inf") else default


def normalize_forecast(
    forecast: Dict[str, Any], *, sender_id: str, reputation: float = 0.0
) -> Dict[str, Any]:
    """يتحقق من بنية نتيجة التنبؤ ويحوّلها إلى سجل قابل للتجميع."""
    if not isinstance(forecast, dict):
        raise ValueError("forecast must be an object")
    predictions = forecast.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("predictions must be a non-empty list")
    if len(predictions) > MAX_FORECAST_POINTS:
        raise ValueError("too_many_prediction_points")
    values = [_finite_float(value) for value in predictions]
    if any(value is None for value in values):
        raise ValueError("predictions must contain finite numbers")
    r2 = _finite_float(forecast.get("r2"), 0.0)
    if r2 is None:
        r2 = 0.0
    r2 = max(MIN_R2, min(MAX_R2, r2))
    rep = _finite_float(reputation, 0.0) or 0.0
    return {
        "node_id": str(sender_id),
        "predictions": [float(value) for value in values],
        "r2": float(r2),
        "reputation": max(0.0, rep),
        "target_date": forecast.get("target_date"),
        "series_len": int(forecast.get("series_len") or 0),
    }


def aggregate_forecasts(
    forecasts: Iterable[Dict[str, Any]], *, min_r2: float = -1.0
) -> Dict[str, Any]:
    """يجمع التنبؤات بوزن جودة موجب يعتمد على R² والسمعة.

    الوزن = max(0, (R² + 1) / 2) × (1 + السمعة المحدودة إلى 100 / 100).
    لا تُقبل السجلات التي تختلف أطوال توقعاتها عن أول سجل صالح.
    """
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []
    expected_len = None
    for item in forecasts:
        try:
            normalized = normalize_forecast(
                item.get("forecast", item),
                sender_id=str(item.get("node_id") or item.get("sender_id") or "unknown"),
                reputation=item.get("reputation", 0.0),
            )
            if normalized["r2"] < min_r2:
                raise ValueError("r2_below_threshold")
            if expected_len is None:
                expected_len = len(normalized["predictions"])
            if len(normalized["predictions"]) != expected_len:
                raise ValueError("prediction_length_mismatch")
            accepted.append(normalized)
        except ValueError as exc:
            rejected.append({"node_id": str(item.get("node_id", "unknown")), "reason": str(exc)})

    if not accepted:
        return {"ok": False, "error": "no_valid_forecasts", "accepted": 0, "rejected": rejected}

    weights = []
    for item in accepted:
        quality = max(0.0, (item["r2"] + 1.0) / 2.0)
        trust = 1.0 + min(100.0, item["reputation"]) / 100.0
        weights.append(quality * trust)
    total_weight = sum(weights)
    if total_weight <= 0:
        return {"ok": False, "error": "zero_total_weight", "accepted": 0, "rejected": rejected}

    point_count = len(accepted[0]["predictions"])
    combined = [
        sum(item["predictions"][index] * weights[pos] for pos, item in enumerate(accepted))
        / total_weight
        for index in range(point_count)
    ]
    return {
        "ok": True,
        "predictions": combined,
        "accepted": len(accepted),
        "rejected": rejected,
        "contributors": [item["node_id"] for item in accepted],
        "total_weight": total_weight,
        "method": "r2_reputation_weighted_mean",
        "target_date": next((item.get("target_date") for item in accepted if item.get("target_date")), None),
    }


def merge_forecast_memory(memory: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """يحدّث ذاكرة التجميع دون تعديل الكائن الذي يملكه المستدعي."""
    updated = deepcopy(memory or {})
    updated["latest"] = deepcopy(result)
    updated["updated_at"] = result.get("updated_at")
    return updated
