"""Dataset helpers for the HackPrinceton evaluation scripts.

This file intentionally avoids the name ``datasets.py`` in flat Colab mode,
because that collides with Hugging Face's ``datasets`` package when all files
live in one directory.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parent if MODULE_DIR.name == "app" else MODULE_DIR
CACHE_DIR = ROOT / ".hf_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_halueval_qa() -> list[dict[str, Any]]:
    """Load HaluEval QA in the project's standard flat example format."""

    dataset = load_dataset("pminervini/HaluEval", "qa", cache_dir=str(CACHE_DIR))
    examples: list[dict[str, Any]] = []
    for item in dataset["data"]:
        examples.append(
            {
                "prompt": item["question"],
                "response": item["right_answer"],
                "is_hallucinated": False,
                "source": "halueval_qa",
            }
        )
        examples.append(
            {
                "prompt": item["question"],
                "response": item["hallucinated_answer"],
                "is_hallucinated": True,
                "source": "halueval_qa",
            }
        )
    return examples


def build_bench2_pairs(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_len_diff: int = 2,
) -> list[dict[str, Any]]:
    """Construct BENCH-2 style matched factual/hallucinated prompt pairs."""

    prompt_to_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        prompt_to_examples[ex["prompt"]].append(ex)

    def resp_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    matched: list[dict[str, Any]] = []
    for prompt, group in prompt_to_examples.items():
        factual = [ex for ex in group if not ex["is_hallucinated"]]
        hall = [ex for ex in group if ex["is_hallucinated"]]
        if len(factual) != 1 or len(hall) != 1:
            continue

        factual_ex = factual[0]
        hall_ex = hall[0]
        factual_len = resp_tokens(factual_ex["response"])
        hall_len = resp_tokens(hall_ex["response"])
        if abs(factual_len - hall_len) <= max_len_diff:
            matched.append(
                {
                    "prompt": prompt,
                    "factual": factual_ex,
                    "hallucinated": hall_ex,
                    "factual_len": factual_len,
                    "hallucinated_len": hall_len,
                    "len_diff": abs(factual_len - hall_len),
                }
            )

    return matched


def load_truthfulqa_first_pairs() -> list[dict[str, Any]]:
    """Load one correct/incorrect pair per TruthfulQA prompt."""

    dataset = load_dataset("truthful_qa", "generation", cache_dir=str(CACHE_DIR))
    pairs: list[dict[str, Any]] = []
    for item in dataset["validation"]:
        correct_answers = item.get("correct_answers", [])
        incorrect_answers = item.get("incorrect_answers", [])
        if not correct_answers or not incorrect_answers:
            continue
        pairs.append(
            {
                "prompt": item["question"],
                "factual": {
                    "prompt": item["question"],
                    "response": correct_answers[0],
                    "is_hallucinated": False,
                    "source": "truthfulqa",
                },
                "hallucinated": {
                    "prompt": item["question"],
                    "response": incorrect_answers[0],
                    "is_hallucinated": True,
                    "source": "truthfulqa",
                },
            }
        )
    return pairs
