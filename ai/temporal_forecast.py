"""محرك تنبؤ زمني خفيف وقابل للتدقيق، بلا شبكة أو نموذج خارجي.

يستخدم انحداراً خطياً على الزمن مع بواقي الملاحظات لبناء نطاق تقريبي.
النتيجة تقدير إحصائي، وليست معرفة بالمستقبل أو ضماناً لحدث.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return date.fromisoformat(text[:10])


def _observations(items: Iterable[Dict[str, Any]]) -> List[Tuple[date, float]]:
    parsed = []
    for item in items or []:
        if not isinstance(item, dict) or "date" not in item or "value" not in item:
            raise ValueError("each observation needs date and value")
        parsed.append((_parse_date(item["date"]), float(item["value"])))
    parsed.sort(key=lambda pair: pair[0])
    if len(parsed) < 3:
        raise ValueError("at least 3 observations are required")
    if len({day for day, _ in parsed}) != len(parsed):
        raise ValueError("observation dates must be unique")
    return parsed


def forecast_timeline(
    observations: Iterable[Dict[str, Any]],
    future_dates: Iterable[Any],
    target_date: Optional[Any] = None,
) -> Dict[str, Any]:
    obs = _observations(observations)
    origin = obs[0][0]
    xs = [float((day - origin).days) for day, _ in obs]
    ys = [value for _, value in obs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        raise ValueError("observations need different dates")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    fitted = [intercept + slope * x for x in xs]
    residuals = [y - fit for y, fit in zip(ys, fitted)]
    sigma = math.sqrt(sum(r * r for r in residuals) / max(1, len(residuals) - 2))
    sst = sum((y - mean_y) ** 2 for y in ys)
    ssr = sum(r * r for r in residuals)
    r2 = 1.0 if sst == 0 and ssr == 0 else max(0.0, min(1.0, 1.0 - ssr / sst)) if sst else 0.0
    last_day = obs[-1][0]
    horizon = max(1, (last_day - origin).days or 1)
    forecasts = []
    for raw_day in future_dates or []:
        day = _parse_date(raw_day)
        x = float((day - origin).days)
        value = intercept + slope * x
        # 1.96 تقريب تقريبي لنطاق 95%، مع اتساع بسيط خارج العينة.
        leverage = 1.0 / len(xs) + ((x - mean_x) ** 2 / sxx)
        margin = 1.96 * sigma * math.sqrt(max(1.0, 1.0 + leverage))
        forecasts.append({
            "date": day.isoformat(),
            "value": round(value, 6),
            "lower_95": round(value - margin, 6),
            "upper_95": round(value + margin, 6),
            "days_from_last_observation": (day - last_day).days,
        })
    result: Dict[str, Any] = {
        "ok": True,
        "method": "date_linear_regression",
        "observations": len(obs),
        "last_observation": {"date": last_day.isoformat(), "value": ys[-1]},
        "slope_per_day": round(slope, 9),
        "direction": "صاعد" if slope > 1e-12 else "هابط" if slope < -1e-12 else "مستقر",
        "r2": round(r2, 6),
        "residual_std": round(sigma, 6),
        "forecasts": forecasts,
        "disclaimer_ar": "تقدير إحصائي مشروط بالبيانات المدخلة، وليس ضماناً لحدث مستقبلي.",
    }
    if target_date is not None:
        target = _parse_date(target_date)
        result["target_date"] = target.isoformat()
        result["target_in_future"] = target > last_day
    return result
