import math

import torch

from app.udc_engine import _prepare_inputs, compute_udc


def _stack_hidden_states(layer_vectors):
    """Create Hugging Face-like hidden states from per-layer token vectors.

    `layer_vectors` shape:
      [n_hidden_snapshots][seq_len][hidden_dim]
    """
    tensors = []
    for layer in layer_vectors:
        tensors.append(torch.tensor([layer], dtype=torch.float32))
    return tuple(tensors)


def test_zero_updates_return_zeroish_values():
    hidden_states = _stack_hidden_states(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )

    result = compute_udc(hidden_states, 0, 2)

    assert result.udc_scalar == 0.0
    assert result.tle_scalar == 0.0
    assert len(result.udc_per_layer) == 2
    assert len(result.udc_per_token) == 2
    assert result.verdict == "FAIL"


def test_same_direction_updates_yield_high_udc():
    hidden_states = _stack_hidden_states(
        [
            [[0.0, 0.0]],
            [[1.0, 0.0]],
            [[2.0, 0.0]],
            [[3.0, 0.0]],
        ]
    )

    result = compute_udc(hidden_states, 0, 1)

    assert result.num_layers == 3
    assert result.num_response_tokens == 1
    assert math.isclose(result.udc_scalar, 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert all(math.isclose(v, 1.0, rel_tol=1e-6, abs_tol=1e-6) for v in result.udc_per_layer)


def test_alternating_updates_yield_negative_udc():
    hidden_states = _stack_hidden_states(
        [
            [[0.0, 0.0]],
            [[1.0, 0.0]],
            [[0.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )

    result = compute_udc(hidden_states, 0, 1)

    assert result.udc_scalar < 0.0
    assert all(v < 0.0 for v in result.udc_per_layer)


def test_scalar_matches_token_mean():
    hidden_states = _stack_hidden_states(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [0.0, 3.0]],
        ]
    )

    result = compute_udc(hidden_states, 0, 2)

    assert math.isclose(
        result.udc_scalar,
        sum(result.udc_per_token) / len(result.udc_per_token),
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    assert len(result.udc_matrix) == 2
    assert len(result.udc_matrix[0]) == result.num_layers - 1


class _FakeChatTokenizer:
    chat_template = "fake-template"
    eos_token_id = None
    pad_token_id = None
    bos_token_id = None
    all_special_ids = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        return_tensors=None,
        return_dict=False,
        **_: object,
    ):
        prompt = messages[0]["content"]
        rendered = f"<user>{prompt}</user><assistant>"
        if len(messages) > 1:
            rendered += messages[1]["content"]
            rendered += "</assistant>"

        if not tokenize:
            return rendered

        encoded = self(rendered, return_tensors=return_tensors)
        if return_dict:
            return encoded
        return encoded["input_ids"]

    def __call__(self, text, *, return_offsets_mapping=False, return_tensors=None, add_special_tokens=True):
        del add_special_tokens
        input_ids = torch.tensor([[ord(ch) for ch in text]], dtype=torch.long)
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
        if return_offsets_mapping:
            offsets = torch.tensor([[(idx, idx + 1) for idx in range(len(text))]], dtype=torch.long)
            batch["offset_mapping"] = offsets
        return batch


def test_prepare_inputs_uses_chat_template_when_available():
    tokenizer = _FakeChatTokenizer()
    prompt = "Question?"
    response = "Answer."

    batch, start, end, input_format = _prepare_inputs(
        tokenizer,
        prompt,
        response,
        use_chat_template="auto",
    )

    assert input_format == "chat_template_offsets"
    assert batch["input_ids"].shape[0] == 1
    assert end > start
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False,
    )
    recovered = "".join(chr(token_id) for token_id in batch["input_ids"][0, start:end].tolist())
    assert recovered == response
