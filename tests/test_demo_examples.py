from app.demo_examples import get_demo_examples, load_fixture_payload, load_scored_example


def test_demo_examples_load_all_fixture_payloads():
    examples = get_demo_examples()

    assert len(examples) == 2
    for example in examples:
        payload = load_fixture_payload(example)
        assert "response_tokens" in payload
        assert "calibration" in payload
        assert example.fixture_path.exists()


def test_demo_examples_adapt_into_app_scores():
    _, correct_scored, _ = load_scored_example("correct_france")
    _, wrong_scored, _ = load_scored_example("wrong_france")

    assert correct_scored.calibrated_verdict == "PASS"
    assert wrong_scored.calibrated_verdict == "REVIEW"
    assert wrong_scored.risk_score > correct_scored.risk_score
