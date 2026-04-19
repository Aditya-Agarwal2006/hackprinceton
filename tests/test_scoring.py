import json
from pathlib import Path

from app.scoring import adapt_analysis_result, compute_risk_score, resolve_metric_value


ROOT = Path(__file__).resolve().parents[1]


def _load_payload(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def test_adapt_analysis_result_uses_median_metric_and_calibration():
    payload = _load_payload("gemma4_calibrated_example_median.json")

    scored = adapt_analysis_result(payload)

    assert scored.headline_metric_name == "udc_median_tok"
    assert scored.headline_metric_label == "Median token coherence"
    assert scored.calibrated_verdict == "PASS"
    assert scored.higher_is_more_factual is False
    assert scored.raw_metric_value == payload["derived_features"]["udc_median_tok"]
    assert scored.aligned_metric_value > 0
    assert 0.0 <= scored.risk_score <= 1.0
    assert len(scored.risk_scores_per_token) == len(payload["response_tokens"])
    assert scored.response_tokens == payload["response_tokens"]


def test_hallucinated_example_scores_as_riskier_than_correct_example():
    factual = adapt_analysis_result(_load_payload("gemma4_calibrated_example_median.json"))
    hallucinated = adapt_analysis_result(_load_payload("gemma4_wrong_example_median.json"))

    assert factual.calibrated_verdict == "PASS"
    assert hallucinated.calibrated_verdict == "REVIEW"
    assert hallucinated.risk_score > factual.risk_score


def test_compute_risk_score_respects_lower_is_more_factual_calibration():
    payload = _load_payload("gemma4_calibrated_example_median.json")
    calibration = payload["calibration"]
    factual_score = payload["derived_features"]["udc_median_tok"]
    wrong_score = _load_payload("gemma4_wrong_example_median.json")["derived_features"]["udc_median_tok"]

    risk_score = compute_risk_score(factual_score, calibration)
    higher_risk_score = compute_risk_score(wrong_score, calibration)

    assert 0.0 <= risk_score <= 1.0
    assert 0.0 <= higher_risk_score <= 1.0
    assert higher_risk_score > risk_score


def test_resolve_metric_value_falls_back_to_derived_features():
    payload = _load_payload("gemma4_calibrated_example_median.json")

    assert resolve_metric_value(payload, "udc_median_tok") == payload["derived_features"]["udc_median_tok"]
    assert resolve_metric_value(payload, "udc_scalar") == payload["udc_scalar"]
