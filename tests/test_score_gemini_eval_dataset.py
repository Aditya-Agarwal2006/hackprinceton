from scripts.score_gemini_eval_dataset import (
    build_pair_row,
    load_eval_pairs,
    select_pairs,
    summarize_rows,
)


class _FakeScored:
    def __init__(self, *, risk_score: float, verdict: str, raw_metric: float):
        self.risk_score = risk_score
        self.calibrated_verdict = verdict
        self.raw_metric_value = raw_metric
        self.headline_metric_name = "udc_median_tok"
        self.risk_bucket = "low"


def test_select_pairs_filters_by_domain_and_limit():
    pairs = [
        {"prompt": "a", "domain": "protocol_review"},
        {"prompt": "b", "domain": "contract_review"},
        {"prompt": "c", "domain": "protocol_review"},
    ]

    selected = select_pairs(pairs, domains=["protocol_review"], limit=1)

    assert len(selected) == 1
    assert selected[0]["prompt"] == "a"


def test_build_pair_row_tracks_direction_and_verdict_gap():
    pair = {
        "prompt": "Q",
        "factual_answer": "A",
        "hallucinated_answer": "B",
        "domain": "submission_qc",
        "generator_model": "gemini",
    }

    row = build_pair_row(
        1,
        pair,
        _FakeScored(risk_score=0.10, verdict="PASS", raw_metric=-0.12),
        _FakeScored(risk_score=0.85, verdict="FAIL", raw_metric=-0.07),
    )

    assert row["correct_direction"] is True
    assert row["strong_separation"] is True
    assert row["verdict_gap"] == 2
    assert row["headline_metric"] == "udc_median_tok"


def test_summarize_rows_reports_overall_and_domain_stats():
    rows = [
        {
            "domain": "protocol_review",
            "prompt": "p1",
            "risk_gap": 0.6,
            "correct_direction": True,
            "strong_separation": True,
        },
        {
            "domain": "protocol_review",
            "prompt": "p2",
            "risk_gap": 0.2,
            "correct_direction": True,
            "strong_separation": False,
        },
        {
            "domain": "contract_review",
            "prompt": "p3",
            "risk_gap": -0.1,
            "correct_direction": False,
            "strong_separation": False,
        },
    ]

    summary = summarize_rows(rows)

    assert summary["total_pairs"] == 3
    assert summary["correct_direction_count"] == 2
    assert summary["strong_separation_count"] == 1
    assert summary["domains"]["protocol_review"]["count"] == 2
    assert summary["domains"]["contract_review"]["correct_direction_rate"] == 0.0


def test_load_eval_pairs_accepts_metadata_wrapper(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(
        '{"pairs":[{"prompt":"Q","factual_answer":"A","hallucinated_answer":"B","domain":"general"}]}'
    )

    pairs = load_eval_pairs(path)

    assert len(pairs) == 1
    assert pairs[0]["prompt"] == "Q"
