import json
from types import SimpleNamespace

import pytest

from app.live_analysis import (
    LiveModelConfig,
    get_live_model_config,
    get_live_model_configs,
    load_live_calibration,
    run_live_analysis,
)


def test_live_model_configs_expose_calibrated_gemma_path():
    configs = get_live_model_configs()

    assert len(configs) >= 1
    assert configs[0].model_name == "google/gemma-4-e2b-it"
    assert configs[0].use_chat_template == "always"
    assert configs[0].headline_metric == "udc_median_tok"


def test_load_live_calibration_uses_fixture_backed_median_metric():
    config = get_live_model_config("gemma4_e2b")
    calibration = load_live_calibration(config)

    assert calibration.metric == "udc_median_tok"
    assert calibration.higher_is_more_factual is False
    assert calibration.pass_threshold >= calibration.review_threshold
    assert config.calibration_path.name == "49_gemma4_udc_calibration.json"


def test_load_live_calibration_rejects_metric_mismatch(tmp_path):
    bad_path = tmp_path / "bad_calibration.json"
    bad_path.write_text(
        json.dumps(
            {
                "metric": "udc_scalar",
                "higher_is_more_factual": False,
                "pass_threshold": 0.2,
                "review_threshold": 0.1,
                "aligned_hall_median": 0.09,
                "aligned_factual_median": 0.12,
                "source": "test",
                "notes": "bad metric",
            }
        )
    )
    config = LiveModelConfig(
        key="test",
        label="Test",
        model_name="test-model",
        use_chat_template="always",
        calibration_path=bad_path,
        headline_metric="udc_median_tok",
        notes="",
    )

    with pytest.raises(ValueError, match="Calibration metric mismatch"):
        load_live_calibration(config)


def test_run_live_analysis_adapts_raw_engine_result(monkeypatch):
    config = get_live_model_config("gemma4_e2b")
    calibration = load_live_calibration(config)

    fake_result = SimpleNamespace(
        udc_scalar=-0.11,
        udc_per_layer=[-0.10, -0.11, -0.09],
        udc_per_token=[-0.12, -0.11, -0.10],
        tle_scalar=1.2,
        tle_per_token=[1.0, 1.1, 1.5],
        num_layers=4,
        num_response_tokens=3,
        response_tokens=["Paris", " is", " France"],
        input_format="chat_template_offsets",
        response_start_token=12,
        verdict="REVIEW",
    )

    def fake_analyze(model, tokenizer, prompt, response, device, use_chat_template, include_geometry):
        assert prompt == "Question: What is the capital of France? Answer:"
        assert response == "Paris is the capital of France."
        assert device == "cpu"
        assert use_chat_template == "always"
        assert include_geometry is False
        return fake_result

    monkeypatch.setattr("app.live_analysis.analyze", fake_analyze)

    raw_result, scored = run_live_analysis(
        object(),
        object(),
        "Question: What is the capital of France? Answer:",
        "Paris is the capital of France.",
        device="cpu",
        config=config,
        calibration=calibration,
    )

    assert raw_result is fake_result
    assert scored.calibrated_verdict in {"PASS", "REVIEW", "FAIL"}
    assert scored.headline_metric_name == calibration.metric
    assert scored.num_response_tokens == 3
    assert len(scored.risk_scores_per_token) == 3
