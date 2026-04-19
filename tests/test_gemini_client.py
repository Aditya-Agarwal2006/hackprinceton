"""Tests for app.gemini_client — parsing, fallback, and length enforcement.

These tests mock the Gemini API so they run offline without an API key.
"""

import json
from unittest import mock

import pytest

from app import gemini_client
from app.gemini_client import (
    _GeminiResult,
    _length_matched,
    _word_count,
    build_fallback_reasoning,
    extract_claims,
    generate_answer,
    generate_eval_dataset,
    generate_validation_report,
    parse_claim_lines,
    parse_eval_json,
    parse_reasoning_json,
    verify_with_reasoning,
)


# ---------------------------------------------------------------------------
# parse_claim_lines (pure function, no mocking needed)
# ---------------------------------------------------------------------------

class TestParseClaimLines:
    def test_numbered_dot(self):
        raw = "1. Paris is the capital.\n2. It has 2M people.\n3. Eiffel Tower."
        result = parse_claim_lines(raw)
        assert len(result) == 3
        assert result[0] == "Paris is the capital."
        assert not any(c[0].isdigit() for c in result)

    def test_numbered_paren(self):
        raw = "1) First claim\n2) Second claim"
        result = parse_claim_lines(raw)
        assert len(result) == 2
        assert result[0] == "First claim"

    def test_bullet_dash(self):
        raw = "- Claim one\n- Claim two\n- Claim three"
        result = parse_claim_lines(raw)
        assert len(result) == 3
        assert not any(c.startswith("-") for c in result)

    def test_bullet_star(self):
        raw = "* Alpha\n* Beta"
        result = parse_claim_lines(raw)
        assert result == ["Alpha", "Beta"]

    def test_no_numbering(self):
        raw = "First claim\nSecond claim"
        result = parse_claim_lines(raw)
        assert result == ["First claim", "Second claim"]

    def test_blank_lines_skipped(self):
        raw = "1. A\n\n\n2. B\n"
        result = parse_claim_lines(raw)
        assert len(result) == 2

    def test_empty_returns_fallback(self):
        assert parse_claim_lines("", fallback="whole text") == ["whole text"]
        assert parse_claim_lines("   \n  \n", fallback="x") == ["x"]

    def test_empty_no_fallback(self):
        assert parse_claim_lines("") == []


# ---------------------------------------------------------------------------
# parse_eval_json (pure function)
# ---------------------------------------------------------------------------

class TestParseEvalJson:
    def test_valid_json(self):
        data = [
            {"prompt": "Q?", "factual_answer": "A", "hallucinated_answer": "B", "domain": "med"}
        ]
        result = parse_eval_json(json.dumps(data))
        assert len(result) == 1
        assert result[0]["prompt"] == "Q?"

    def test_strips_code_fences(self):
        data = [{"prompt": "Q", "factual_answer": "A", "hallucinated_answer": "B", "domain": "d"}]
        raw = "```json\n" + json.dumps(data) + "\n```"
        result = parse_eval_json(raw)
        assert len(result) == 1

    def test_missing_keys_dropped(self):
        data = [
            {"prompt": "Q", "factual_answer": "A", "hallucinated_answer": "B", "domain": "d"},
            {"prompt": "Q", "factual_answer": "A"},  # missing keys
        ]
        result = parse_eval_json(json.dumps(data))
        assert len(result) == 1

    def test_empty_values_dropped(self):
        data = [{"prompt": "", "factual_answer": "A", "hallucinated_answer": "B", "domain": "d"}]
        result = parse_eval_json(json.dumps(data))
        assert len(result) == 0

    def test_invalid_json(self):
        assert parse_eval_json("not json at all") == []

    def test_non_list_json(self):
        assert parse_eval_json('{"key": "value"}') == []

    def test_non_dict_entries(self):
        assert parse_eval_json('["string", 42, null]') == []


# ---------------------------------------------------------------------------
# _length_matched
# ---------------------------------------------------------------------------

class TestLengthMatched:
    def test_exact_match(self):
        assert _length_matched("one two three", "four five six")

    def test_within_20_percent(self):
        # 10 words vs 12 words = 20% diff, should pass
        a = " ".join(["w"] * 10)
        b = " ".join(["w"] * 12)
        assert _length_matched(a, b)

    def test_over_20_percent(self):
        # 10 words vs 13 words = 30% diff, should fail
        a = " ".join(["w"] * 10)
        b = " ".join(["w"] * 13)
        assert not _length_matched(a, b)

    def test_empty_both(self):
        assert _length_matched("", "")

    def test_empty_one(self):
        assert not _length_matched("", "word")


# ---------------------------------------------------------------------------
# Fallback behavior (mocked)
# ---------------------------------------------------------------------------

def _make_mock_call(primary_fails=False, fallback_fails=False, primary_text="ok", fallback_text="fallback"):
    """Return a side_effect function that simulates primary/fallback behavior."""
    call_count = [0]

    def fake_generate_content(model, contents, config=None):
        call_count[0] += 1
        if model == gemini_client._MODEL_NAME and primary_fails:
            raise RuntimeError("primary down")
        if model == gemini_client._FALLBACK_MODEL and fallback_fails:
            raise RuntimeError("fallback down")
        text = primary_text if model == gemini_client._MODEL_NAME else fallback_text
        resp = mock.MagicMock()
        resp.text = text
        return resp

    return fake_generate_content, call_count


class TestFallback:
    def test_primary_succeeds(self):
        fake_gen, _ = _make_mock_call(primary_text="Paris")
        mock_models = mock.MagicMock()
        mock_models.generate_content = fake_gen
        mock_client = mock.MagicMock()
        mock_client.models = mock_models

        with mock.patch.object(gemini_client, "_client", mock_client):
            result = generate_answer("capital of France?")
        assert result == "Paris"

    def test_falls_back_on_primary_failure(self):
        fake_gen, call_count = _make_mock_call(primary_fails=True, fallback_text="fallback answer")
        mock_models = mock.MagicMock()
        mock_models.generate_content = fake_gen
        mock_client = mock.MagicMock()
        mock_client.models = mock_models

        with mock.patch.object(gemini_client, "_client", mock_client):
            result = generate_answer("test")
        assert result == "fallback answer"
        assert call_count[0] == 2

    def test_both_fail_returns_error_string(self):
        fake_gen, _ = _make_mock_call(primary_fails=True, fallback_fails=True)
        mock_models = mock.MagicMock()
        mock_models.generate_content = fake_gen
        mock_client = mock.MagicMock()
        mock_client.models = mock_models

        with mock.patch.object(gemini_client, "_client", mock_client):
            result = generate_answer("test")
        assert result.startswith("[GEMINI_ERROR:")

    def test_no_api_key_returns_error_string(self):
        with mock.patch.object(gemini_client, "_client", None), \
             mock.patch.object(gemini_client, "_DOTENV_CACHE", {}), \
             mock.patch.dict("os.environ", {}, clear=True):
            result = generate_answer("test")
        assert result.startswith("[GEMINI_ERROR:")

    def test_get_client_uses_dotenv_key_when_env_missing(self):
        class _FakeClient:
            def __init__(self, api_key):
                self.api_key = api_key

        fake_genai = mock.MagicMock()
        fake_genai.Client = _FakeClient

        with mock.patch.object(gemini_client, "genai", fake_genai), \
             mock.patch.object(gemini_client, "_client", None), \
             mock.patch.object(gemini_client, "_API_KEY", None), \
             mock.patch.object(gemini_client, "_DOTENV_CACHE", {"GEMINI_API_KEY": "dotenv-key"}), \
             mock.patch.dict("os.environ", {}, clear=True):
            client = gemini_client._get_client()

        assert isinstance(client, _FakeClient)
        assert client.api_key == "dotenv-key"


# ---------------------------------------------------------------------------
# extract_claims fallback
# ---------------------------------------------------------------------------

class TestExtractClaimsFallback:
    def test_api_failure_returns_original(self):
        with mock.patch.object(gemini_client, "_client", None), \
             mock.patch.object(gemini_client, "_DOTENV_CACHE", {}), \
             mock.patch.dict("os.environ", {}, clear=True):
            result = extract_claims("The sky is blue and water is wet.")
        assert result == ["The sky is blue and water is wet."]

    def test_empty_input(self):
        assert extract_claims("") == []
        assert extract_claims("   ") == []


# ---------------------------------------------------------------------------
# generate_validation_report fallback
# ---------------------------------------------------------------------------

class TestValidationReportFallback:
    def test_api_failure_returns_fallback(self):
        with mock.patch.object(gemini_client, "_client", None), \
             mock.patch.object(gemini_client, "_DOTENV_CACHE", {}), \
             mock.patch.dict("os.environ", {}, clear=True):
            result = generate_validation_report("claim", 0.5, "REVIEW", "flat pattern")
        assert "REVIEW" in result
        assert "50.0%" in result
        assert "automated coherence check" in result


class TestGeminiReasoning:
    def test_parse_reasoning_json_builds_structured_result(self):
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
                "student_next_step": "Safe to keep this answer.",
            }
        )

        result = parse_reasoning_json(raw, model_name="gemini-2.5-flash")

        assert result.model_name == "gemini-2.5-flash"
        assert result.overall_verdict == "LIKELY_FACTUAL"
        assert len(result.claims) == 1
        assert result.claims[0].assessment == "ACCURATE"

    def test_build_fallback_reasoning_uses_udc_context(self):
        result = build_fallback_reasoning(
            question="What is the capital of France?",
            answer="Lyon is the capital of France.",
            risk_score=0.91,
            udc_verdict="FAIL",
            layer_summary="lowest coherence around transition 14->15",
            flagged_tokens=["Lyon"],
        )

        assert result.used_fallback is True
        assert result.overall_verdict == "LIKELY_CONFABULATED"
        assert "Lyon" in result.claims[0].explanation

    def test_verify_with_reasoning_falls_back_when_call_fails(self):
        with mock.patch.object(gemini_client, "_call_gemini", side_effect=RuntimeError("boom")):
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

    def test_verify_with_reasoning_uses_parseable_gemini_result(self):
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
                "student_next_step": "Keep this answer.",
            }
        )

        def fake_call(_system, _content):
            return _GeminiResult(text=raw, model_used="gemini-2.5-flash")

        with mock.patch.object(gemini_client, "_call_gemini", side_effect=fake_call):
            result = verify_with_reasoning(
                question="What is the capital of France?",
                answer="Paris is the capital of France.",
                risk_score=0.12,
                udc_verdict="PASS",
                layer_summary="stable coherence throughout",
                flagged_tokens=[],
            )

        assert result.used_fallback is False
        assert result.model_name == "gemini-2.5-flash"
        assert result.overall_verdict == "LIKELY_FACTUAL"


# ---------------------------------------------------------------------------
# generate_eval_dataset length enforcement (mocked)
# ---------------------------------------------------------------------------

class TestEvalDatasetLengthEnforcement:
    def _make_pair(self, factual_words, hallucinated_words):
        return {
            "prompt": "Q?",
            "factual_answer": " ".join(f"fact{i}" for i in range(factual_words)),
            "hallucinated_answer": " ".join(f"hall{i}" for i in range(hallucinated_words)),
            "domain": "test",
        }

    def test_matched_pair_kept(self):
        pair = self._make_pair(10, 11)  # 10% diff, within 20%
        raw_json = json.dumps([pair])

        def fake_call(system, content):
            return _GeminiResult(text=raw_json, model_used="gemini-2.0-flash")

        with mock.patch.object(gemini_client, "_call_gemini", side_effect=fake_call):
            result = generate_eval_dataset("test", 1)
        assert len(result) == 1

    def test_mismatched_pair_triggers_rewrite(self):
        pair = self._make_pair(10, 20)  # 100% diff, way over 20%
        raw_json = json.dumps([pair])
        rewritten = " ".join(f"rw{i}" for i in range(11))  # 10% diff from 10

        call_num = [0]

        def fake_call(system, content):
            call_num[0] += 1
            if call_num[0] == 1:
                return _GeminiResult(text=raw_json, model_used="gemini-2.0-flash")
            else:
                return _GeminiResult(text=rewritten, model_used="gemini-2.0-flash")

        with mock.patch.object(gemini_client, "_call_gemini", side_effect=fake_call):
            result = generate_eval_dataset("test", 1)
        assert len(result) == 1
        assert "(rewrite)" in result[0]["generator_model"]

    def test_mismatched_pair_dropped_if_rewrite_fails(self):
        pair = self._make_pair(10, 20)
        raw_json = json.dumps([pair])

        call_num = [0]

        def fake_call(system, content):
            call_num[0] += 1
            if call_num[0] == 1:
                return _GeminiResult(text=raw_json, model_used="gemini-2.0-flash")
            else:
                # Rewrite still mismatched
                return _GeminiResult(
                    text=" ".join(f"rw{i}" for i in range(20)),
                    model_used="gemini-2.0-flash",
                )

        with mock.patch.object(gemini_client, "_call_gemini", side_effect=fake_call):
            result = generate_eval_dataset("test", 1)
        assert len(result) == 0

    def test_model_provenance_recorded(self):
        pair = self._make_pair(10, 10)
        raw_json = json.dumps([pair])

        def fake_call(system, content):
            return _GeminiResult(text=raw_json, model_used="gemini-2.5-flash-lite")

        with mock.patch.object(gemini_client, "_call_gemini", side_effect=fake_call):
            result = generate_eval_dataset("test", 1)
        assert result[0]["generator_model"] == "gemini-2.5-flash-lite"
