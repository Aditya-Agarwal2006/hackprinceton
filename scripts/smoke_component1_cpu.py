"""Fast CPU smoke harness for Component 1."""

from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "app").exists() else HERE
for candidate in [HERE, ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from app.udc_engine import analyze, load_model
except ModuleNotFoundError:
    from udc_engine import analyze, load_model


MODEL_NAME = "sshleifer/tiny-gpt2"
PROMPT = "Question: What is the capital of France? Answer:"
RESPONSE = "Paris is the capital of France."


def main() -> None:
    print(f"Loading {MODEL_NAME} on CPU ...")
    model, tokenizer = load_model(MODEL_NAME, "cpu")

    print("Running analyze() ...")
    result = analyze(model, tokenizer, PROMPT, RESPONSE, "cpu", use_chat_template="auto")

    preview = {
        "model": MODEL_NAME,
        "udc_scalar": result.udc_scalar,
        "tle_scalar": result.tle_scalar,
        "num_layers": result.num_layers,
        "num_response_tokens": result.num_response_tokens,
        "input_format": result.input_format,
        "verdict": result.verdict,
        "response_tokens": result.response_tokens[:10],
        "udc_per_layer_preview": result.udc_per_layer[:10],
    }
    print(json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
