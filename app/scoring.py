"""App-facing scoring helpers built on top of raw UDC engine outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

try:
    from .calibration import MetricCalibration, align_score, verdict_from_score
    from .feature_metrics import extract_feature_metrics
except ImportError:  # pragma: no cover - convenience for direct script execution
    from calibration import MetricCalibration, align_score, verdict_from_score
    from feature_metrics import extract_feature_metrics


DEFAULT_HEADLINE_METRIC = "udc_median_tok"
_EPS = 1e-12


@dataclass
class AppScoreResult:
    """UI-friendly summary derived from a raw UDC analysis payload."""

    headline_metric_name: str
    headline_metric_label: str
    raw_metric_value: float
    aligned_metric_value: float
    higher_is_more_factual: bool
    calibrated_verdict: str
    risk_score: float
    risk_bucket: str
    risk_label: str
    num_layers: int
    num_response_tokens: int
    response_tokens: list[str]
    udc_per_layer: list[float]
    udc_per_token: list[float]
    risk_scores_per_token: list[float]
    derived_features: dict[str, float]
    calibration_metric: str
    input_format: str = "plain_text"
    response_start_token: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(key, default)
    return getattr(result, key, default)


def _coerce_calibration(calibration: MetricCalibration | Mapping[str, Any]) -> MetricCalibration:
    if isinstance(calibration, MetricCalibration):
        return calibration
    if isinstance(calibration, Mapping):
        return MetricCalibration(**dict(calibration))
    raise TypeError(f"Unsupported calibration type: {type(calibration)!r}")


def _metric_label(metric_name: str) -> str:
    labels = {
        "udc_median_tok": "Median token coherence",
        "udc_mean_tok_mean": "Mean token coherence",
        "udc_scalar": "Mean token coherence",
        "tle_mean_tok_mean": "Endpoint expansion",
        "tle_scalar": "Endpoint expansion",
    }
    return labels.get(metric_name, metric_name.replace("_", " ").title())


def _risk_bucket(risk_score: float) -> tuple[str, str]:
    if risk_score < 0.25:
        return "low", "Low confabulation risk"
    if risk_score < 0.50:
        return "guarded", "Guarded confabulation risk"
    if risk_score < 0.75:
        return "elevated", "Elevated confabulation risk"
    return "high", "High confabulation risk"


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def compute_risk_score(score: float, calibration: MetricCalibration | Mapping[str, Any]) -> float:
    """Map a raw metric score into a monotonic 0-1 risk scale."""

    calibration_obj = _coerce_calibration(calibration)
    aligned = align_score(score, calibration_obj.higher_is_more_factual)
    factual_anchor = float(calibration_obj.aligned_factual_median)
    hall_anchor = float(calibration_obj.aligned_hall_median)
    span = factual_anchor - hall_anchor

    if abs(span) <= _EPS:
        if aligned >= calibration_obj.pass_threshold:
            return 0.20
        if aligned >= calibration_obj.review_threshold:
            return 0.50
        return 0.80

    normalized_factuality = (aligned - hall_anchor) / span
    risk_score = 1.0 - normalized_factuality
    return _clamp(risk_score)


def _coerce_numeric_list(values: Any) -> list[float]:
    return [float(value) for value in (values or [])]


def _ensure_derived_features(result: Any) -> dict[str, float]:
    existing = _get_value(result, "derived_features")
    if existing:
        return {str(key): float(value) for key, value in dict(existing).items()}
    derived = extract_feature_metrics(result)
    return {str(key): float(value) for key, value in derived.items()}


def resolve_metric_value(
    result: Any,
    metric_name: str = DEFAULT_HEADLINE_METRIC,
) -> float:
    """Read a metric from top-level result fields or derived features."""

    direct_value = _get_value(result, metric_name)
    if direct_value is not None:
        return float(direct_value)

    derived = _ensure_derived_features(result)
    if metric_name in derived:
        return float(derived[metric_name])

    available = sorted(set(derived) | set(getattr(result, "__dict__", {}).keys()))
    if isinstance(result, Mapping):
        available = sorted(set(available) | set(result.keys()))
    raise KeyError(f"Unknown metric '{metric_name}'. Available fields: {available}")


def adapt_analysis_result(
    result: Any,
    calibration: MetricCalibration | Mapping[str, Any] | None = None,
    *,
    headline_metric: str = DEFAULT_HEADLINE_METRIC,
) -> AppScoreResult:
    """Shape a raw engine payload into a UI-friendly scored result."""

    calibration_data = calibration or _get_value(result, "calibration")
    if calibration_data is None:
        raise ValueError("Calibration is required to build an app-facing score.")
    calibration_obj = _coerce_calibration(calibration_data)

    derived_features = _ensure_derived_features(result)
    raw_metric_value = resolve_metric_value(result, headline_metric)
    aligned_metric_value = align_score(raw_metric_value, calibration_obj.higher_is_more_factual)
    calibrated_verdict = verdict_from_score(raw_metric_value, calibration_obj)
    risk_score = compute_risk_score(raw_metric_value, calibration_obj)
    risk_bucket, risk_label = _risk_bucket(risk_score)

    response_tokens = [str(token) for token in (_get_value(result, "response_tokens") or [])]
    udc_per_layer = _coerce_numeric_list(_get_value(result, "udc_per_layer"))
    udc_per_token = _coerce_numeric_list(_get_value(result, "udc_per_token"))
    risk_scores_per_token = [
        compute_risk_score(token_score, calibration_obj)
        for token_score in udc_per_token
    ]

    return AppScoreResult(
        headline_metric_name=headline_metric,
        headline_metric_label=_metric_label(headline_metric),
        raw_metric_value=raw_metric_value,
        aligned_metric_value=float(aligned_metric_value),
        higher_is_more_factual=bool(calibration_obj.higher_is_more_factual),
        calibrated_verdict=calibrated_verdict,
        risk_score=risk_score,
        risk_bucket=risk_bucket,
        risk_label=risk_label,
        num_layers=int(_get_value(result, "num_layers", len(udc_per_layer) + 1)),
        num_response_tokens=int(_get_value(result, "num_response_tokens", len(response_tokens))),
        response_tokens=response_tokens,
        udc_per_layer=udc_per_layer,
        udc_per_token=udc_per_token,
        risk_scores_per_token=risk_scores_per_token,
        derived_features=derived_features,
        calibration_metric=str(calibration_obj.metric),
        input_format=str(_get_value(result, "input_format", "plain_text")),
        response_start_token=int(_get_value(result, "response_start_token", 0)),
    )
