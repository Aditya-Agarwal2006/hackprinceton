import math

import pytest

from app.udc_engine import analyze, find_response_start, load_model


TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.mark.integration
def test_load_model_and_hidden_states():
    model, tokenizer = load_model(TINY_MODEL, "cpu")
    inputs = tokenizer("test", return_tensors="pt")
    outputs = model(**inputs)

    assert model.config.output_hidden_states is True
    assert tokenizer.pad_token is not None
    assert outputs.hidden_states is not None
    assert len(outputs.hidden_states) == model.config.n_layer + 1


@pytest.mark.integration
def test_find_response_start_handles_trailing_space():
    _, tokenizer = load_model(TINY_MODEL, "cpu")
    prompt = "What is 2+2? "
    response = "The answer is 4."
    full_text = f"{prompt}{response}"
    inputs = tokenizer(full_text, return_tensors="pt")

    start = find_response_start(tokenizer, prompt, response, inputs["input_ids"])
    decoded = tokenizer.decode([int(inputs["input_ids"][0, start].item())], skip_special_tokens=False)

    assert isinstance(start, int)
    assert start >= 0
    assert decoded.strip() != ""


@pytest.mark.integration
def test_analyze_is_deterministic_on_tiny_model():
    model, tokenizer = load_model(TINY_MODEL, "cpu")
    prompt = "Question: What is the capital of France? Answer:"
    response = "Paris is the capital of France."

    result_1 = analyze(model, tokenizer, prompt, response, "cpu")
    result_2 = analyze(model, tokenizer, prompt, response, "cpu")

    assert math.isfinite(result_1.udc_scalar)
    assert len(result_1.response_tokens) > 0
    assert result_1.udc_scalar == result_2.udc_scalar
    assert result_1.tle_scalar == result_2.tle_scalar
    assert result_1.response_tokens == result_2.response_tokens

