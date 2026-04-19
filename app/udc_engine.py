"""Pure UDC engine for Confab HackPrinceton Component 1.

This module is computation-only by design:

- no Streamlit import
- reusable from CLI, notebooks, tests, and the future app

It implements the four functions required by the spec:

- load_model
- find_response_start
- compute_udc
- analyze
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


EPS = 1e-12
DEFAULT_PASS_THRESHOLD = 0.15
DEFAULT_REVIEW_THRESHOLD = 0.05


@dataclass
class UDCResult:
    """Return type for all UDC engine outputs."""

    udc_scalar: float
    udc_per_layer: list[float]
    udc_per_token: list[float]
    udc_matrix: list[list[float]]
    tle_scalar: float
    tle_per_token: list[float]
    num_layers: int
    num_response_tokens: int
    response_tokens: list[str] = field(default_factory=list)
    geometry_3d: dict[str, Any] | None = None
    input_format: str = "plain_text"
    response_start_token: int = 0
    verdict: str = "REVIEW"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _preferred_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device == "cuda" else torch.float32


def _load_pretrained_model(
    model_name: str,
    *,
    dtype: torch.dtype,
) -> Any:
    kwargs = {
        "low_cpu_mem_usage": True,
        "dtype": dtype,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except TypeError:
        kwargs.pop("dtype")
        kwargs["torch_dtype"] = dtype
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as causal_error:  # pragma: no cover - depends on model class
        try:
            from transformers import AutoModelForImageTextToText
        except Exception as import_error:  # pragma: no cover
            raise RuntimeError(
                f"Could not load {model_name} as a causal LM, and "
                "AutoModelForImageTextToText is unavailable. "
                f"Original error: {causal_error}"
            ) from import_error

        kwargs = {
            "low_cpu_mem_usage": True,
            "dtype": dtype,
        }
        try:
            return AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
        except TypeError:
            kwargs.pop("dtype")
            kwargs["torch_dtype"] = dtype
            return AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)


def load_model(model_name: str, device: str) -> tuple[Any, Any]:
    """Load a decoder-style verifier model and tokenizer.

    The config is loaded explicitly so ``output_hidden_states`` is baked in
    before the model is instantiated, matching the spec.
    """

    device = _normalize_device(device)
    dtype = _preferred_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = _load_pretrained_model(model_name, dtype=dtype)
    model.config.output_hidden_states = True
    model.eval()
    model.to(device)
    return model, tokenizer


def _safe_cosine(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    norm_a = float(vec_a.norm().item())
    norm_b = float(vec_b.norm().item())
    if not math.isfinite(norm_a) or not math.isfinite(norm_b):
        return 0.0
    if norm_a <= EPS or norm_b <= EPS:
        return 0.0
    numerator = float(torch.dot(vec_a, vec_b).item())
    if not math.isfinite(numerator):
        return 0.0
    value = numerator / (norm_a * norm_b)
    return float(value) if math.isfinite(value) else 0.0


def _join_prompt_and_response(prompt: str, response: str) -> tuple[str, str]:
    """Return the full text and the exact prefix used before the response."""

    prompt = prompt or ""
    response = response or ""

    if prompt and response:
        prefix = prompt if prompt.endswith(" ") else f"{prompt} "
        return prefix + response, prefix

    return prompt + response, prompt


def _supports_chat_template(tokenizer: Any) -> bool:
    return bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")


def _coerce_batch_encoding(encoded: Any) -> dict[str, torch.Tensor]:
    if isinstance(encoded, torch.Tensor):
        return {
            "input_ids": encoded,
            "attention_mask": torch.ones_like(encoded),
        }

    if hasattr(encoded, "items"):
        batch: dict[str, torch.Tensor] = {}
        for key, value in encoded.items():
            if torch.is_tensor(value):
                batch[key] = value
        if "input_ids" not in batch and "input_ids" in encoded:
            value = encoded["input_ids"]
            if not torch.is_tensor(value):
                batch["input_ids"] = torch.tensor(value, dtype=torch.long)
        if "input_ids" not in batch:
            raise ValueError("Encoded batch did not contain input_ids.")
        if "attention_mask" not in batch:
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        return batch

    if isinstance(encoded, list):
        tensor = torch.tensor([encoded], dtype=torch.long)
        return {
            "input_ids": tensor,
            "attention_mask": torch.ones_like(tensor),
        }

    raise TypeError(f"Unsupported encoded batch type: {type(encoded)!r}")


def _extract_input_ids(encoded: Any) -> torch.Tensor:
    return _coerce_batch_encoding(encoded)["input_ids"]


def _find_response_span_from_offsets(
    offsets: list[list[int]] | list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    overlapping = [
        idx
        for idx, (tok_start, tok_end) in enumerate(offsets)
        if tok_end > char_start and tok_start < char_end
    ]
    if not overlapping:
        raise ValueError("Could not map response chars to token span.")
    return overlapping[0], overlapping[-1] + 1


def _trim_trailing_special_tokens(
    input_ids: torch.Tensor,
    response_start: int,
    tokenizer: Any,
) -> int:
    token_ids = input_ids[0].tolist()
    end = len(token_ids)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for maybe_id in [
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
    ]:
        if maybe_id is not None:
            special_ids.add(int(maybe_id))

    while end > response_start and token_ids[end - 1] in special_ids:
        end -= 1
    return end


def _prepare_plain_inputs(tokenizer: Any, prompt: str, response: str) -> tuple[dict[str, torch.Tensor], int, int, str]:
    full_text, prompt_prefix = _join_prompt_and_response(prompt, response)
    inputs = tokenizer(full_text, return_tensors="pt")
    batch = _coerce_batch_encoding(inputs)
    response_start = find_response_start(tokenizer, prompt_prefix, response, batch["input_ids"])
    response_end = int(batch["input_ids"].shape[-1])
    return batch, response_start, response_end, "plain_text"


def _prepare_chat_template_inputs(
    tokenizer: Any,
    prompt: str,
    response: str,
) -> tuple[dict[str, torch.Tensor], int, int, str]:
    messages_user = [{"role": "user", "content": prompt}]
    messages_full = messages_user + [{"role": "assistant", "content": response}]

    # Preferred path: use assistant token masks if the tokenizer/template exposes them.
    try:
        encoded = tokenizer.apply_chat_template(
            messages_full,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            return_assistant_tokens_mask=True,
        )
        batch = _coerce_batch_encoding(encoded)
        assistant_mask = None
        for key in [
            "assistant_masks",
            "assistant_mask",
            "assistant_tokens_mask",
            "assistant_token_mask",
        ]:
            if hasattr(encoded, "keys") and key in encoded:
                assistant_mask = encoded[key]
                break
        if assistant_mask is not None:
            if not torch.is_tensor(assistant_mask):
                assistant_mask = torch.tensor(assistant_mask, dtype=torch.long)
            mask_values = assistant_mask[0].tolist() if assistant_mask.ndim > 1 else assistant_mask.tolist()
            assistant_positions = [idx for idx, value in enumerate(mask_values) if int(value) == 1]
            if assistant_positions:
                start = assistant_positions[0]
                end = assistant_positions[-1] + 1
                return batch, start, end, "chat_template_mask"
    except Exception:
        pass

    # Robust fallback: render the template as text, find the literal assistant
    # response span, then map chars back to tokens via offsets.
    try:
        rendered_prefix = tokenizer.apply_chat_template(
            messages_user,
            tokenize=False,
            add_generation_prompt=True,
        )
        rendered_full = tokenizer.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(rendered_prefix, list):
            rendered_prefix = rendered_prefix[0]
        if isinstance(rendered_full, list):
            rendered_full = rendered_full[0]

        encoded = tokenizer(rendered_full, return_offsets_mapping=True, return_tensors="pt")
        prefix_start = len(rendered_prefix) if rendered_full.startswith(rendered_prefix) else 0
        relative_idx = rendered_full[prefix_start:].find(response)
        if relative_idx >= 0:
            char_start = prefix_start + relative_idx
            char_end = char_start + len(response)
            response_start, response_end = _find_response_span_from_offsets(
                encoded["offset_mapping"][0].tolist(),
                char_start,
                char_end,
            )
            batch = {key: value for key, value in encoded.items() if key != "offset_mapping"}
            batch = _coerce_batch_encoding(batch)
            return batch, response_start, response_end, "chat_template_offsets"
    except Exception:
        pass

    # Last-resort fallback: derive the response start from the user-only
    # templated prefix length. This may include a trailing EOS in the response
    # span, so we trim final special tokens.
    prefix_encoded = tokenizer.apply_chat_template(
        messages_user,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    full_encoded = tokenizer.apply_chat_template(
        messages_full,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    prefix_ids = _extract_input_ids(prefix_encoded)
    batch = _coerce_batch_encoding(full_encoded)
    response_start = int(prefix_ids.shape[-1])
    response_end = _trim_trailing_special_tokens(batch["input_ids"], response_start, tokenizer)
    return batch, response_start, response_end, "chat_template_token_diff"


def _prepare_inputs(
    tokenizer: Any,
    prompt: str,
    response: str,
    *,
    use_chat_template: str,
) -> tuple[dict[str, torch.Tensor], int, int, str]:
    if use_chat_template not in {"auto", "always", "never"}:
        raise ValueError("use_chat_template must be one of: auto, always, never")

    if use_chat_template == "never":
        return _prepare_plain_inputs(tokenizer, prompt, response)

    supports_chat = _supports_chat_template(tokenizer)
    if use_chat_template == "always":
        if not supports_chat:
            raise ValueError("Tokenizer does not expose a chat template, but use_chat_template='always' was requested.")
        return _prepare_chat_template_inputs(tokenizer, prompt, response)

    if supports_chat:
        return _prepare_chat_template_inputs(tokenizer, prompt, response)
    return _prepare_plain_inputs(tokenizer, prompt, response)


def find_response_start(tokenizer: Any, prompt: str, response: str, input_ids: torch.Tensor) -> int:
    """Locate the first token belonging to the response.

    This adapts the multi-strategy fallback logic from the existing repo's
    ``src/transformer_probes.py::_find_response_start``.

    ``prompt`` should be the exact string prefix used in the concatenated text.
    """

    n_total = int(input_ids.shape[-1])
    full_text = f"{prompt}{response}"
    prompt_char_end = len(prompt)

    # Strategy 1: offset mapping from the full text.
    try:
        enc = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc["offset_mapping"][0].tolist()
        for idx, (char_start, char_end) in enumerate(offsets):
            if char_end > 0 and char_start >= prompt_char_end:
                return idx
    except Exception:
        pass

    # Strategy 2: tokenize response alone and count backward from the end.
    try:
        response_ids = tokenizer(
            response,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]
        n_response = int(len(response_ids))
        if n_response > 0:
            candidate = n_total - n_response
            if 0 <= candidate < n_total:
                return candidate
    except Exception:
        pass

    # Strategy 3: use prompt-without-special length and infer BOS offset.
    try:
        prompt_no_special = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        full_no_special = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
        special_offset = n_total - full_no_special
        candidate = special_offset + prompt_no_special
        if 0 <= candidate < n_total:
            return candidate
    except Exception:
        pass

    # Strategy 4: prompt token length under the model's default rules.
    try:
        candidate = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        if 0 <= candidate < n_total:
            return candidate
    except Exception:
        pass

    raise ValueError("Unable to locate response start token.")


def _decode_response_tokens(tokenizer: Any, input_ids: torch.Tensor, start: int, end: int) -> list[str]:
    tokens: list[str] = []
    for token_index in range(start, end):
        token_id = int(input_ids[0, token_index].item())
        tokens.append(tokenizer.decode([token_id], skip_special_tokens=False))
    return tokens


def _compute_verdict(
    udc_scalar: float,
    *,
    pass_threshold: float,
    review_threshold: float,
) -> str:
    if pass_threshold < review_threshold:
        raise ValueError("pass_threshold must be >= review_threshold")
    if udc_scalar >= pass_threshold:
        return "PASS"
    if udc_scalar >= review_threshold:
        return "REVIEW"
    return "FAIL"


def compute_udc(
    hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
    response_start: int,
    response_end: int,
    *,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> UDCResult:
    """Compute UDC and secondary endpoint-TLE for a response span.

    ``response_end`` is exclusive.
    """

    num_layers = len(hidden_states) - 1
    if num_layers < 2:
        raise ValueError("Need at least 3 hidden-state snapshots to compute UDC.")
    if response_end <= response_start:
        raise ValueError("Empty response span.")

    udc_matrix: list[list[float]] = []
    udc_per_token: list[float] = []
    tle_per_token: list[float] = []

    for token_index in range(response_start, response_end):
        trajectory = [hidden_states[layer_index][0, token_index, :].float() for layer_index in range(num_layers + 1)]
        deltas = [trajectory[layer_index + 1] - trajectory[layer_index] for layer_index in range(num_layers)]

        row = [_safe_cosine(deltas[layer_index], deltas[layer_index + 1]) for layer_index in range(num_layers - 1)]
        udc_matrix.append(row)
        udc_per_token.append(float(np.mean(row)) if row else 0.0)

        first_norm = float(deltas[0].norm().item()) if deltas else 0.0
        last_norm = float(deltas[-1].norm().item()) if deltas else 0.0
        tle_per_token.append(float(math.log((last_norm + EPS) / (first_norm + EPS))))

    matrix = np.asarray(udc_matrix, dtype=np.float64)
    udc_per_layer = matrix.mean(axis=0).tolist() if matrix.size else [0.0] * (num_layers - 1)
    udc_scalar = float(np.mean(udc_per_token)) if udc_per_token else 0.0
    tle_scalar = float(np.mean(tle_per_token)) if tle_per_token else 0.0

    return UDCResult(
        udc_scalar=udc_scalar,
        udc_per_layer=udc_per_layer,
        udc_per_token=udc_per_token,
        udc_matrix=udc_matrix,
        tle_scalar=tle_scalar,
        tle_per_token=tle_per_token,
        num_layers=num_layers,
        num_response_tokens=response_end - response_start,
        verdict=_compute_verdict(
            udc_scalar,
            pass_threshold=pass_threshold,
            review_threshold=review_threshold,
        ),
    )


def analyze(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
    device: str,
    *,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    use_chat_template: str = "auto",
    include_geometry: bool = False,
) -> UDCResult:
    """Convenience wrapper for one end-to-end UDC analysis run."""

    device = _normalize_device(device)
    inputs, response_start, response_end, input_format = _prepare_inputs(
        tokenizer,
        prompt,
        response,
        use_chat_template=use_chat_template,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model output did not include hidden states.")

    input_ids = inputs["input_ids"]
    result = compute_udc(
        hidden_states,
        response_start,
        response_end,
        pass_threshold=pass_threshold,
        review_threshold=review_threshold,
    )
    result.response_tokens = _decode_response_tokens(tokenizer, input_ids, response_start, response_end)
    result.input_format = input_format
    result.response_start_token = response_start
    if include_geometry:
        try:
            from .geometry import project_response_update_geometry
        except ImportError:  # pragma: no cover - convenience for direct script execution
            from geometry import project_response_update_geometry

        geometry = project_response_update_geometry(
            hidden_states,
            response_start,
            response_end,
            response_tokens=result.response_tokens,
        )
        result.geometry_3d = geometry.to_dict()
    return result


__all__ = [
    "DEFAULT_PASS_THRESHOLD",
    "DEFAULT_REVIEW_THRESHOLD",
    "UDCResult",
    "analyze",
    "compute_udc",
    "find_response_start",
    "load_model",
]
