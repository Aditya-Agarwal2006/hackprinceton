from types import SimpleNamespace

from scripts.validate_demo_examples import (
    build_validation_row,
    rank_validation_rows,
    select_examples,
    summarize_validation_rows,
)


def _fake_scored(*, risk_score: float, verdict: str, raw_metric: float, bucket: str = "low"):
    return SimpleNamespace(
        risk_score=risk_score,
        calibrated_verdict=verdict,
        raw_metric_value=raw_metric,
        headline_metric_name="udc_median_tok",
        risk_bucket=bucket,
    )


def test_select_examples_filters_by_domain_and_id():
    examples = [
        {"id": "a", "domain": "medical"},
        {"id": "b", "domain": "scientific"},
        {"id": "c", "domain": "medical"},
    ]

    selected = select_examples(examples, domains=["medical"], example_ids=["c"])

    assert selected == [{"id": "c", "domain": "medical"}]


def test_build_validation_row_and_ranking_prioritize_correct_direction():
    example = {
        "id": "med_1",
        "domain": "medical",
        "display_name": "Medical Example",
        "priority": 2,
        "prompt": "Q",
        "factual_answer": "A",
        "hallucinated_answer": "B",
        "explanation": "why wrong",
        "source": "hand",
        "notes": "",
    }
    row_good = build_validation_row(
        example,
        _fake_scored(risk_score=0.20, verdict="PASS", raw_metric=-0.12),
        _fake_scored(risk_score=0.78, verdict="FAIL", raw_metric=-0.08, bucket="high"),
    )
    row_bad = build_validation_row(
        {**example, "id": "med_2", "display_name": "Second", "priority": 1},
        _fake_scored(risk_score=0.60, verdict="REVIEW", raw_metric=-0.10, bucket="elevated"),
        _fake_scored(risk_score=0.30, verdict="PASS", raw_metric=-0.11),
    )

    ranked = rank_validation_rows([row_bad, row_good])

    assert row_good["correct_direction"] is True
    assert row_bad["correct_direction"] is False
    assert ranked[0]["id"] == "med_1"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2


def test_summarize_validation_rows_reports_top_and_failures():
    rows = rank_validation_rows(
        [
            {
                "id": "good_one",
                "domain": "general",
                "display_name": "Good",
                "priority": 1,
                "risk_gap": 0.40,
                "correct_direction": True,
            },
            {
                "id": "bad_one",
                "domain": "medical",
                "display_name": "Bad",
                "priority": 2,
                "risk_gap": -0.10,
                "correct_direction": False,
            },
        ]
    )

    summary = summarize_validation_rows(rows)

    assert summary["total_examples"] == 2
    assert summary["correct_direction_count"] == 1
    assert summary["top_working_ids"] == ["good_one"]
    assert summary["failing_ids"] == ["bad_one"]
