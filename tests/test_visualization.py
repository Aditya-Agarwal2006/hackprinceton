import json
from pathlib import Path
import re

import plotly.graph_objects as go

from app.scoring import adapt_analysis_result
from app.visualization import (
    build_focused_direction_comparison_figure,
    build_layer_chart,
    build_layer_coherence_chart,
    build_update_geometry_comparison_figure,
    build_update_geometry_figure,
    build_risk_gauge,
    extract_focused_direction_slice,
    summarize_update_geometry_comparison,
    build_token_heatmap,
    build_token_heatmap_html,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_payload(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def test_layer_chart_accepts_app_facing_result():
    scored = adapt_analysis_result(_load_payload("gemma4_calibrated_example_median.json"))

    fig = build_layer_coherence_chart(scored, title="Gemma")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert list(fig.data[0].x) == list(range(len(scored.udc_per_layer)))
    assert fig.layout.title.text == "Gemma"


def test_layer_chart_supports_comparison_mode():
    factual = adapt_analysis_result(_load_payload("gemma4_calibrated_example_median.json"))
    hallucinated = adapt_analysis_result(_load_payload("gemma4_wrong_example_median.json"))

    fig = build_layer_chart(
        factual,
        title="Comparison",
        comparison_data=hallucinated,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_risk_gauge_uses_calibrated_risk_score():
    scored = adapt_analysis_result(_load_payload("gemma4_wrong_example_median.json"))

    fig = build_risk_gauge(scored)

    assert isinstance(fig, go.Figure)
    assert 0.0 <= fig.data[0]["value"] <= 100.0
    assert fig.data[0]["value"] == scored.risk_score * 100.0


def test_risk_gauge_accepts_raw_calibrated_payload():
    payload = _load_payload("gemma4_wrong_example_median.json")
    expected = adapt_analysis_result(payload)

    fig = build_risk_gauge(payload)

    assert fig.data[0]["value"] == expected.risk_score * 100.0


def test_token_heatmap_preserves_token_order_and_count():
    scored = adapt_analysis_result(_load_payload("gemma4_calibrated_example_median.json"))

    html = build_token_heatmap_html(scored)

    assert html.startswith("<")
    assert html.count("<span") == len(scored.response_tokens)
    indices = [int(value) for value in re.findall(r'data-token-index="(\d+)"', html)]
    assert indices == list(range(len(scored.response_tokens)))


def test_token_heatmap_accepts_raw_calibrated_payload():
    payload = _load_payload("gemma4_calibrated_example_median.json")

    html = build_token_heatmap_html(payload)

    assert html.startswith("<")
    assert html.count("<span") == len(payload["response_tokens"])


def test_token_heatmap_alias_accepts_raw_arrays():
    html = build_token_heatmap(
        ["Paris", " is", " the", " capital"],
        [-0.12, -0.11, -0.13, -0.10],
        [0.15, 0.22, 0.08, 0.30],
    )

    assert html.startswith("<")
    assert html.count("<span") == 4
    assert "title=" in html


def test_update_geometry_figure_accepts_geometry_payload():
    geometry = {
        "method": "pca",
        "num_components": 3,
        "explained_variance_ratio": [0.7, 0.2, 0.1],
        "num_layers": 3,
        "token_paths": [
            {
                "token_index": 0,
                "token": "Paris",
                "points": [[0, 0, 0], [1, 0, 0], [2, 0.5, 0], [3, 1, 0.2]],
                "deltas": [[1, 0, 0], [1, 0.5, 0], [1, 0.5, 0.2]],
                "segment_cosines": [1.0, 0.95],
            }
        ],
    }

    fig = build_update_geometry_figure(geometry, title="Geometry")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.layout.title.text == "Geometry"


def test_update_geometry_comparison_figure_shows_both_paths():
    factual = {
        "method": "pca",
        "num_components": 3,
        "explained_variance_ratio": [0.7, 0.2, 0.1],
        "num_layers": 3,
        "token_paths": [
            {
                "token_index": 0,
                "token": "Paris",
                "points": [[0, 0, 0], [1, 0, 0], [2, 0.1, 0], [3, 0.2, 0]],
                "deltas": [[1, 0, 0], [1, 0.1, 0], [1, 0.1, 0]],
                "segment_cosines": [0.99, 0.99],
            }
        ],
    }
    confabulated = {
        "method": "pca",
        "num_components": 3,
        "explained_variance_ratio": [0.7, 0.2, 0.1],
        "num_layers": 3,
        "token_paths": [
            {
                "token_index": 0,
                "token": "Lyon",
                "points": [[0, 0, 0], [1, 0, 0], [1.2, 1.0, 0], [0.4, 1.6, 0]],
                "deltas": [[1, 0, 0], [0.2, 1.0, 0], [-0.8, 0.6, 0]],
                "segment_cosines": [0.2, -0.1],
            }
        ],
    }

    fig = build_update_geometry_comparison_figure(factual, confabulated, title="Comparison")
    summary = summarize_update_geometry_comparison(factual, confabulated)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.layout.title.text == "Comparison"
    assert summary["confabulated_mean_turn_deg"] > summary["factual_mean_turn_deg"]


def test_extract_focused_direction_slice_prefers_more_aligned_factual_local_window():
    factual = {
        "method": "pca",
        "num_components": 3,
        "explained_variance_ratio": [0.7, 0.2, 0.1],
        "num_layers": 5,
        "token_paths": [
            {
                "token_index": 0,
                "token": "A",
                "points": [],
                "deltas": [[1, 0, 0], [1, 0.02, 0], [1, 0.04, 0], [1, 0.06, 0], [1, 0.08, 0]],
                "segment_cosines": [],
            },
            {
                "token_index": 1,
                "token": "B",
                "points": [],
                "deltas": [[1, 0, 0], [1, 0.01, 0], [1, 0.02, 0], [1, 0.03, 0], [1, 0.04, 0]],
                "segment_cosines": [],
            },
            {
                "token_index": 2,
                "token": "C",
                "points": [],
                "deltas": [[1, 0, 0], [1, 0.03, 0], [1, 0.05, 0], [1, 0.07, 0], [1, 0.09, 0]],
                "segment_cosines": [],
            },
            {
                "token_index": 3,
                "token": "D",
                "points": [],
                "deltas": [[1, 0, 0], [1, 0.02, 0], [1, 0.03, 0], [1, 0.04, 0], [1, 0.05, 0]],
                "segment_cosines": [],
            },
        ],
    }
    confabulated = {
        "method": "pca",
        "num_components": 3,
        "explained_variance_ratio": [0.7, 0.2, 0.1],
        "num_layers": 5,
        "token_paths": [
            {
                "token_index": 0,
                "token": "W",
                "points": [],
                "deltas": [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]],
                "segment_cosines": [],
            },
            {
                "token_index": 1,
                "token": "X",
                "points": [],
                "deltas": [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]],
                "segment_cosines": [],
            },
            {
                "token_index": 2,
                "token": "Y",
                "points": [],
                "deltas": [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]],
                "segment_cosines": [],
            },
            {
                "token_index": 3,
                "token": "Z",
                "points": [],
                "deltas": [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]],
                "segment_cosines": [],
            },
        ],
    }

    focused = extract_focused_direction_slice(
        factual,
        confabulated,
        factual_tokens=["A", "B", "C", "D"],
        confabulated_tokens=["W", "X", "Y", "Z"],
        window_sizes=(4,),
        slice_deltas=4,
    )

    assert focused["factual_mean_cosine"] > focused["confabulated_mean_cosine"]
    assert focused["factual_phrase"] == "ABCD"
    assert focused["confabulated_phrase"] == "WXYZ"


def test_focused_direction_comparison_figure_returns_plot_and_metadata():
    geometry = {
        "method": "pca",
        "num_components": 3,
        "explained_variance_ratio": [0.7, 0.2, 0.1],
        "num_layers": 5,
        "token_paths": [
            {
                "token_index": 0,
                "token": "A",
                "points": [],
                "deltas": [[1, 0, 0], [1, 0.02, 0], [1, 0.04, 0], [1, 0.06, 0], [1, 0.08, 0]],
                "segment_cosines": [],
            },
            {
                "token_index": 1,
                "token": "B",
                "points": [],
                "deltas": [[1, 0, 0], [1, 0.03, 0], [1, 0.05, 0], [1, 0.07, 0], [1, 0.09, 0]],
                "segment_cosines": [],
            },
            {
                "token_index": 2,
                "token": "C",
                "points": [],
                "deltas": [[1, 0, 0], [1, 0.04, 0], [1, 0.06, 0], [1, 0.08, 0], [1, 0.1, 0]],
                "segment_cosines": [],
            },
            {
                "token_index": 3,
                "token": "D",
                "points": [],
                "deltas": [[1, 0, 0], [1, 0.02, 0], [1, 0.03, 0], [1, 0.04, 0], [1, 0.05, 0]],
                "segment_cosines": [],
            },
        ],
    }
    confabulated = {
        "method": "pca",
        "num_components": 3,
        "explained_variance_ratio": [0.7, 0.2, 0.1],
        "num_layers": 5,
        "token_paths": [
            {
                "token_index": 0,
                "token": "W",
                "points": [],
                "deltas": [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]],
                "segment_cosines": [],
            },
            {
                "token_index": 1,
                "token": "X",
                "points": [],
                "deltas": [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]],
                "segment_cosines": [],
            },
            {
                "token_index": 2,
                "token": "Y",
                "points": [],
                "deltas": [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]],
                "segment_cosines": [],
            },
            {
                "token_index": 3,
                "token": "Z",
                "points": [],
                "deltas": [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]],
                "segment_cosines": [],
            },
        ],
    }

    fig, focused = build_focused_direction_comparison_figure(
        geometry,
        confabulated,
        factual_tokens=["A", "B", "C", "D"],
        confabulated_tokens=["W", "X", "Y", "Z"],
        title="Focused",
    )

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Focused"
    assert len(fig.data) == 8
    assert focused["factual_mean_cosine"] > focused["confabulated_mean_cosine"]
