import json
import os
from pathlib import Path

from app.k2_client import (
    K2ReasoningResult,
    _k2_config,
    build_fallback_reasoning,
    parse_reasoning_json,
    verify_with_reasoning,
)


def test_parse_reasoning_json_builds_structured_result():
    raw = json.dumps(
        {
            "claims": [
                {
                    "claim": "Penicillin treats viral infections.",
                    "assessment": "INACCURATE",
                    "explanation": "Penicillin is an antibiotic used for susceptible bacterial infections, not viral ones.",
                    "correction": "Say it treats certain bacterial infections.",
                }
            ],
            "overall_verdict": "LIKELY_CONFABULATED",
            "student_next_step": "Review the difference between bacterial and viral infections.",
        }
    )

    result = parse_reasoning_json(raw, model_name="K2-Think-V2")

    assert isinstance(result, K2ReasoningResult)
    assert result.model_name == "K2-Think-V2"
    assert result.overall_verdict == "LIKELY_CONFABULATED"
    assert len(result.claims) == 1
    assert result.claims[0].assessment == "INACCURATE"


def test_build_fallback_reasoning_uses_udc_context():
    result = build_fallback_reasoning(
        question="What does penicillin treat?",
        answer="Penicillin treats viral infections.",
        risk_score=0.81,
        udc_verdict="FAIL",
        layer_summary="lowest coherence around transition 11->12",
        flagged_tokens=["viral", "infections"],
    )

    assert result.used_fallback is True
    assert result.overall_verdict == "LIKELY_CONFABULATED"
    assert "viral" in result.claims[0].explanation


def test_verify_with_reasoning_falls_back_on_client_error(monkeypatch):
    def fake_post(_messages):
        raise RuntimeError("network down")

    monkeypatch.setattr("app.k2_client._post_chat_completion", fake_post)

    result = verify_with_reasoning(
        question="What is the capital of France?",
        answer="Lyon is the capital of France.",
        risk_score=0.93,
        udc_verdict="FAIL",
        layer_summary="lowest coherence around transition 14->15",
        flagged_tokens=["Lyon"],
    )

    assert result.used_fallback is True
    assert result.claims
    assert result.overall_verdict == "LIKELY_CONFABULATED"


def test_verify_with_reasoning_uses_k2_result_when_parseable(monkeypatch):
    raw = json.dumps(
        {
            "claims": [
                {
                    "claim": "Paris is the capital of France.",
                    "assessment": "ACCURATE",
                    "explanation": "Paris is the capital city of France.",
                    "correction": "",
                }
            ],
            "overall_verdict": "LIKELY_FACTUAL",
            "student_next_step": "Safe to keep studying from this answer.",
        }
    )

    def fake_post(_messages):
        return raw, "K2-Think-V2"

    monkeypatch.setattr("app.k2_client._post_chat_completion", fake_post)

    result = verify_with_reasoning(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        risk_score=0.12,
        udc_verdict="PASS",
        layer_summary="stable coherence throughout",
        flagged_tokens=[],
    )

    assert result.used_fallback is False
    assert result.model_name == "K2-Think-V2"
    assert result.overall_verdict == "LIKELY_FACTUAL"


def test_k2_config_accepts_ifm_api_key_alias(monkeypatch, tmp_path):
    monkeypatch.delenv("K2_API_KEY", raising=False)
    monkeypatch.delenv("IFM_API_KEY", raising=False)
    monkeypatch.setenv("IFM_API_KEY", "ifm-test-key")
    monkeypatch.setattr("app.k2_client._DOTENV_CACHE", None)

    config = _k2_config()

    assert config["api_key"] == "ifm-test-key"
