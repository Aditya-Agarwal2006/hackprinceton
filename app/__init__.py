"""Confab HackPrinceton app package."""

from .udc_engine import (
    DEFAULT_PASS_THRESHOLD,
    DEFAULT_REVIEW_THRESHOLD,
    UDCResult,
    analyze,
    compute_udc,
    find_response_start,
    load_model,
)
from .calibration import MetricCalibration, align_score, fit_quantile_calibration, verdict_from_score
from .scoring import AppScoreResult, adapt_analysis_result, compute_risk_score, resolve_metric_value
from .live_analysis import (
    LiveModelConfig,
    get_live_model_config,
    get_live_model_configs,
    load_live_calibration,
    run_live_analysis,
)
from .gemini_client import (
    GeminiClaimAssessment,
    GeminiReasoningResult,
    build_fallback_reasoning,
    parse_reasoning_json,
    verify_with_reasoning,
)
from .geometry import GeometryProjection3D, TokenGeometry3D, project_response_update_geometry
from .text_utils import extract_claims_local
from .visualization import (
    build_focused_direction_comparison_figure,
    build_layer_chart,
    build_layer_coherence_chart,
    build_update_geometry_comparison_figure,
    build_update_geometry_figure,
    build_risk_gauge,
    extract_focused_direction_slice,
    build_token_heatmap,
    build_token_heatmap_html,
    summarize_update_geometry_comparison,
)

__all__ = [
    "AppScoreResult",
    "DEFAULT_PASS_THRESHOLD",
    "DEFAULT_REVIEW_THRESHOLD",
    "GeometryProjection3D",
    "GeminiClaimAssessment",
    "GeminiReasoningResult",
    "LiveModelConfig",
    "MetricCalibration",
    "TokenGeometry3D",
    "UDCResult",
    "adapt_analysis_result",
    "align_score",
    "analyze",
    "build_fallback_reasoning",
    "build_focused_direction_comparison_figure",
    "extract_claims_local",
    "extract_focused_direction_slice",
    "build_layer_chart",
    "build_layer_coherence_chart",
    "build_update_geometry_comparison_figure",
    "build_update_geometry_figure",
    "build_risk_gauge",
    "build_token_heatmap",
    "build_token_heatmap_html",
    "compute_udc",
    "compute_risk_score",
    "find_response_start",
    "fit_quantile_calibration",
    "get_live_model_config",
    "get_live_model_configs",
    "load_live_calibration",
    "load_model",
    "parse_reasoning_json",
    "project_response_update_geometry",
    "run_live_analysis",
    "resolve_metric_value",
    "summarize_update_geometry_comparison",
    "verify_with_reasoning",
    "verdict_from_score",
]
