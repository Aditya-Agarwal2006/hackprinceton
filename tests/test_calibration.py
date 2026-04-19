from app.calibration import fit_quantile_calibration, verdict_from_score


def test_fit_quantile_calibration_higher_is_more_factual():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.8, 0.9, 1.0, 0.1, 0.2, 0.3]
    cal = fit_quantile_calibration(
        labels,
        scores,
        metric="udc",
        higher_is_more_factual=True,
        source="unit-test",
    )
    assert cal.pass_threshold >= cal.review_threshold
    assert verdict_from_score(0.95, cal) == "PASS"
    assert verdict_from_score(0.05, cal) == "FAIL"


def test_fit_quantile_calibration_lower_is_more_factual():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
    cal = fit_quantile_calibration(
        labels,
        scores,
        metric="udc",
        higher_is_more_factual=False,
        source="unit-test",
    )
    assert cal.pass_threshold >= cal.review_threshold
    assert verdict_from_score(0.05, cal) == "PASS"
    assert verdict_from_score(0.95, cal) == "FAIL"

