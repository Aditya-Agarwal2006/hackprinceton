"""Helpers for live Analyze-tab inference inside the Streamlit shell."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

try:
    from .calibration import MetricCalibration
    from .scoring import AppScoreResult, adapt_analysis_result
    from .udc_engine import UDCResult, analyze
except ImportError:  # pragma: no cover - convenience for direct script execution
    from calibration import MetricCalibration
    from scoring import AppScoreResult, adapt_analysis_result
    from udc_engine import UDCResult, analyze


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LiveModelConfig:
    """Configuration for one supported live inference path."""

    key: str
    label: str
    model_name: str
    use_chat_template: str
    calibration_path: Path
    headline_metric: str
    notes: str


SUPPORTED_LIVE_MODELS: tuple[LiveModelConfig, ...] = (
    LiveModelConfig(
        key="gemma4_e2b",
        label="Gemma 4 E2B (calibrated)",
        model_name="google/gemma-4-e2b-it",
        use_chat_template="always",
        calibration_path=ROOT / "49_gemma4_udc_calibration.json",
        headline_metric="udc_median_tok",
        notes=(
            "Uses the current Gemma-first calibrated `udc_median_tok` path. "
            "Requires local Hugging Face access to the Gemma checkpoint."
        ),
    ),
)


def get_live_model_configs() -> tuple[LiveModelConfig, ...]:
    """Return the supported live model configurations."""

    return SUPPORTED_LIVE_MODELS


def get_live_model_config(key: str) -> LiveModelConfig:
    """Resolve one live model configuration by key."""

    for config in SUPPORTED_LIVE_MODELS:
        if config.key == key:
            return config
    available = ", ".join(config.key for config in SUPPORTED_LIVE_MODELS)
    raise KeyError(f"Unknown live model '{key}'. Available: {available}")


def load_live_calibration(config: LiveModelConfig | str = "gemma4_e2b") -> MetricCalibration:
    """Load the calibration object for the supported live path."""

    selected = get_live_model_config(config) if isinstance(config, str) else config
    calibration_data = json.loads(selected.calibration_path.read_text())
    calibration = MetricCalibration(**calibration_data)
    if calibration.metric != selected.headline_metric:
        raise ValueError(
            "Calibration metric mismatch for live path "
            f"'{selected.key}': expected '{selected.headline_metric}' from "
            f"{selected.calibration_path.name}, got '{calibration.metric}'."
        )
    return calibration


def run_live_analysis(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
    *,
    device: str,
    config: LiveModelConfig | str = "gemma4_e2b",
    calibration: MetricCalibration | None = None,
) -> tuple[UDCResult, AppScoreResult]:
    """Run live UDC analysis and adapt it into the app-facing format."""

    selected = get_live_model_config(config) if isinstance(config, str) else config
    calibration_obj = calibration or load_live_calibration(selected)
    raw_result = analyze(
        model,
        tokenizer,
        prompt,
        response,
        device,
        use_chat_template=selected.use_chat_template,
        include_geometry=False,
    )
    scored_result = adapt_analysis_result(
        raw_result,
        calibration_obj,
        headline_metric=selected.headline_metric,
    )
    return raw_result, scored_result
