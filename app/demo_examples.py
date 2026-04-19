"""Fixture-backed demo examples for the minimal Streamlit shell."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

try:
    from .scoring import AppScoreResult, adapt_analysis_result
except ImportError:  # pragma: no cover - convenience for direct script execution
    from scoring import AppScoreResult, adapt_analysis_result


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DemoExample:
    """Metadata for one precomputed demo example."""

    key: str
    label: str
    prompt: str
    answer: str
    fixture_path: Path
    notes: str


_EXAMPLES: tuple[DemoExample, ...] = (
    DemoExample(
        key="correct_france",
        label="Correct France example",
        prompt="Question: What is the capital of France? Answer:",
        answer="Paris is the capital of France.",
        fixture_path=ROOT / "gemma4_calibrated_example_median.json",
        notes="Precomputed calibrated Gemma 4 result for the factual France answer.",
    ),
    DemoExample(
        key="wrong_france",
        label="Wrong France example",
        prompt="Question: What is the capital of France? Answer:",
        answer="Lyon is the capital of France.",
        fixture_path=ROOT / "gemma4_wrong_example_median.json",
        notes="Precomputed calibrated Gemma 4 result for the hallucinated France answer.",
    ),
)


def get_demo_examples() -> tuple[DemoExample, ...]:
    """Return the demo examples in display order."""

    return _EXAMPLES


def get_demo_example(key: str) -> DemoExample:
    """Resolve one example by key."""

    for example in _EXAMPLES:
        if example.key == key:
            return example
    available = ", ".join(example.key for example in _EXAMPLES)
    raise KeyError(f"Unknown demo example '{key}'. Available: {available}")


def load_fixture_payload(example: DemoExample | str) -> dict[str, Any]:
    """Load the raw JSON payload for one fixture-backed example."""

    selected = get_demo_example(example) if isinstance(example, str) else example
    return json.loads(selected.fixture_path.read_text())


def load_scored_example(example: DemoExample | str) -> tuple[DemoExample, AppScoreResult, dict[str, Any]]:
    """Load the metadata, adapted app-facing score, and raw payload."""

    selected = get_demo_example(example) if isinstance(example, str) else example
    raw_payload = load_fixture_payload(selected)
    scored = adapt_analysis_result(raw_payload)
    return selected, scored, raw_payload
