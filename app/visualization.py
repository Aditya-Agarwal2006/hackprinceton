"""Visualization builders for app-facing UDC displays."""

from __future__ import annotations

from html import escape
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

try:
    from .scoring import adapt_analysis_result
except ImportError:  # pragma: no cover - convenience for direct script execution
    from scoring import adapt_analysis_result


_CHART_BACKGROUND = "#FFFFFF"
_TEXT_COLOR = "#0F172A"
_ACCENT_COLOR = "#2563EB"
_SECONDARY_COLOR = "#F59E0B"
_ZERO_LINE_COLOR = "#94A3B8"


def _get_value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(key, default)
    return getattr(result, key, default)


def _coerce_result_field(
    result_or_values: Any,
    field_name: str,
    values: Sequence[float] | None,
) -> list[float]:
    if values is not None:
        return [float(value) for value in values]

    extracted = _get_value(result_or_values, field_name)
    if extracted is None:
        raise ValueError(f"Missing required field '{field_name}'.")
    return [float(value) for value in extracted]


def _coerce_layer_payload(
    result_or_values: Any,
    num_layers: int | None,
) -> tuple[list[float], int]:
    if isinstance(result_or_values, Mapping) or hasattr(result_or_values, "udc_per_layer"):
        udc_per_layer = _coerce_result_field(result_or_values, "udc_per_layer", None)
        inferred_num_layers = int(_get_value(result_or_values, "num_layers", len(udc_per_layer) + 1))
        return udc_per_layer, num_layers or inferred_num_layers

    if result_or_values is None:
        raise ValueError("Layer chart needs either a result object or per-layer values.")

    udc_per_layer = [float(value) for value in result_or_values]
    if num_layers is None:
        num_layers = len(udc_per_layer) + 1
    return udc_per_layer, int(num_layers)


def _normalize_scores(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-12:
        return [0.5 for _ in values]
    return [(float(value) - lo) / (hi - lo) for value in values]


def _coerce_heatmap_payload(
    result_or_tokens: Any,
    udc_per_token: Sequence[float] | None,
    risk_scores_per_token: Sequence[float] | None,
) -> tuple[list[str], list[float], list[float]]:
    if isinstance(result_or_tokens, Mapping) or hasattr(result_or_tokens, "response_tokens"):
        tokens = [str(token) for token in (_get_value(result_or_tokens, "response_tokens") or [])]
        raw_scores = _coerce_result_field(result_or_tokens, "udc_per_token", udc_per_token)
        if risk_scores_per_token is not None:
            risk_scores = [float(value) for value in risk_scores_per_token]
        else:
            extracted_risk_scores = _get_value(result_or_tokens, "risk_scores_per_token")
            if extracted_risk_scores is not None:
                risk_scores = [float(value) for value in extracted_risk_scores]
            elif _get_value(result_or_tokens, "calibration") is not None:
                risk_scores = adapt_analysis_result(result_or_tokens).risk_scores_per_token
            else:
                risk_scores = _normalize_scores(raw_scores)
        return tokens, raw_scores, risk_scores

    if udc_per_token is None:
        raise ValueError("Token heatmap needs UDC scores when no result object is provided.")

    tokens = [str(token) for token in (result_or_tokens or [])]
    raw_scores = [float(value) for value in udc_per_token]
    risk_scores = (
        [float(value) for value in risk_scores_per_token]
        if risk_scores_per_token is not None
        else _normalize_scores(raw_scores)
    )
    return tokens, raw_scores, risk_scores


def _coerce_geometry_payload(geometry: Any) -> dict[str, Any]:
    if geometry is None:
        raise ValueError("Missing geometry payload.")
    if isinstance(geometry, Mapping):
        return dict(geometry)
    if hasattr(geometry, "to_dict"):
        return geometry.to_dict()
    raise TypeError(f"Unsupported geometry payload type: {type(geometry)!r}")


def _aggregate_geometry_snake(
    geometry: Any,
    *,
    dims: tuple[int, int] = (0, 1),
) -> dict[str, Any]:
    payload = _coerce_geometry_payload(geometry)
    token_paths = list(payload.get("token_paths") or [])
    if not token_paths:
        raise ValueError("Geometry payload did not contain any token paths.")

    num_layers = int(payload.get("num_layers", 0))
    if num_layers <= 0:
        raise ValueError("Geometry payload did not include a valid num_layers field.")

    mean_deltas: list[np.ndarray] = []
    for layer_index in range(num_layers):
        deltas = []
        for token_path in token_paths:
            token_deltas = token_path.get("deltas") or []
            if layer_index < len(token_deltas):
                deltas.append(np.asarray(token_deltas[layer_index], dtype=np.float64))
        if not deltas:
            deltas.append(np.zeros(max(dims) + 1, dtype=np.float64))
        mean_deltas.append(np.mean(np.stack(deltas, axis=0), axis=0))

    projected = [delta[list(dims)] for delta in mean_deltas]
    origin = np.zeros(2, dtype=np.float64)
    points = [origin]
    for delta in projected:
        points.append(points[-1] + delta)

    turn_angles_deg: list[float] = []
    for current_delta, next_delta in zip(projected[:-1], projected[1:]):
        denom = float(np.linalg.norm(current_delta) * np.linalg.norm(next_delta))
        if denom <= 1e-12:
            turn_angles_deg.append(0.0)
            continue
        cosine = float(np.dot(current_delta, next_delta) / denom)
        cosine = max(-1.0, min(1.0, cosine))
        turn_angles_deg.append(math.degrees(math.acos(cosine)))

    path_length = float(sum(np.linalg.norm(delta) for delta in projected))
    net_distance = float(np.linalg.norm(points[-1] - points[0]))
    path_efficiency = (net_distance / path_length) if path_length > 1e-12 else 0.0

    return {
        "points": [point.tolist() for point in points],
        "mean_deltas": [delta.tolist() for delta in projected],
        "turn_angles_deg": turn_angles_deg,
        "mean_turn_angle_deg": float(np.mean(turn_angles_deg)) if turn_angles_deg else 0.0,
        "path_efficiency": float(path_efficiency),
        "num_layers": num_layers,
        "explained_variance_ratio": list(payload.get("explained_variance_ratio") or []),
    }


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    cosine = float(np.dot(a, b) / denom)
    return max(-1.0, min(1.0, cosine))


def _unit_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.zeros_like(vec)
    return vec / norm


def _tokens_from_geometry(geometry: Any) -> list[str]:
    payload = _coerce_geometry_payload(geometry)
    return [str(path.get("token", "")) for path in payload.get("token_paths") or []]


def _mean_projected_deltas_for_window(
    geometry: Any,
    token_start: int,
    token_end: int,
) -> list[np.ndarray]:
    payload = _coerce_geometry_payload(geometry)
    token_paths = list(payload.get("token_paths") or [])
    selected_paths = token_paths[token_start:token_end]
    if not selected_paths:
        raise ValueError("Selected token window was empty.")

    num_layers = min(len(path.get("deltas") or []) for path in selected_paths)
    mean_deltas: list[np.ndarray] = []
    for layer_index in range(num_layers):
        layer_deltas = [
            np.asarray(path["deltas"][layer_index], dtype=np.float64)
            for path in selected_paths
        ]
        mean_deltas.append(np.mean(np.stack(layer_deltas, axis=0), axis=0))
    return mean_deltas


def extract_focused_direction_slice(
    factual_geometry: Any,
    confabulated_geometry: Any,
    *,
    factual_tokens: Sequence[str] | None = None,
    confabulated_tokens: Sequence[str] | None = None,
    window_sizes: Sequence[int] = (4, 5),
    slice_deltas: int = 4,
) -> dict[str, Any]:
    """Find a small local slice where factual and confabulated directions diverge most.

    The selected view intentionally normalizes arrow length because UDC measures
    direction, not magnitude. This gives the user a cleaner visual explanation
    of why one local update bundle is more coherent than the other.
    """

    factual_payload = _coerce_geometry_payload(factual_geometry)
    confabulated_payload = _coerce_geometry_payload(confabulated_geometry)
    factual_token_paths = list(factual_payload.get("token_paths") or [])
    confabulated_token_paths = list(confabulated_payload.get("token_paths") or [])
    factual_token_strings = list(factual_tokens or _tokens_from_geometry(factual_payload))
    confabulated_token_strings = list(confabulated_tokens or _tokens_from_geometry(confabulated_payload))

    max_window = min(
        len(factual_token_paths),
        len(confabulated_token_paths),
        len(factual_token_strings),
        len(confabulated_token_strings),
    )
    if max_window <= 0:
        raise ValueError("Could not select a focused geometry slice from empty token paths.")

    best: dict[str, Any] | None = None
    for window_size in window_sizes:
        if window_size > max_window:
            continue
        for token_start in range(0, max_window - window_size + 1):
            token_end = token_start + window_size
            factual_mean_deltas = _mean_projected_deltas_for_window(
                factual_payload,
                token_start,
                token_end,
            )
            confabulated_mean_deltas = _mean_projected_deltas_for_window(
                confabulated_payload,
                token_start,
                token_end,
            )
            max_slice_start = min(len(factual_mean_deltas), len(confabulated_mean_deltas)) - slice_deltas
            if max_slice_start < 0:
                continue
            for layer_start in range(0, max_slice_start + 1):
                layer_end = layer_start + slice_deltas
                factual_slice = factual_mean_deltas[layer_start:layer_end]
                confabulated_slice = confabulated_mean_deltas[layer_start:layer_end]
                factual_cosines = [
                    _safe_cosine(factual_slice[i], factual_slice[i + 1])
                    for i in range(slice_deltas - 1)
                ]
                confabulated_cosines = [
                    _safe_cosine(confabulated_slice[i], confabulated_slice[i + 1])
                    for i in range(slice_deltas - 1)
                ]
                score = float(
                    np.mean(np.asarray(factual_cosines) - np.asarray(confabulated_cosines))
                )
                candidate = {
                    "score": score,
                    "token_start": token_start,
                    "token_end": token_end,
                    "layer_start": layer_start,
                    "layer_end": layer_end,
                    "factual_deltas": factual_slice,
                    "confabulated_deltas": confabulated_slice,
                    "factual_cosines": factual_cosines,
                    "confabulated_cosines": confabulated_cosines,
                    "factual_phrase": "".join(factual_token_strings[token_start:token_end]).strip(),
                    "confabulated_phrase": "".join(confabulated_token_strings[token_start:token_end]).strip(),
                }
                if best is None or candidate["score"] > best["score"]:
                    best = candidate

    if best is None:
        raise ValueError("Could not find a valid local direction slice.")

    factual_units = [_unit_vector(np.asarray(delta, dtype=np.float64)) for delta in best["factual_deltas"]]
    confabulated_units = [
        _unit_vector(np.asarray(delta, dtype=np.float64)) for delta in best["confabulated_deltas"]
    ]

    def cumulative_points(units: list[np.ndarray]) -> list[np.ndarray]:
        points = [np.zeros(3, dtype=np.float64)]
        for unit in units:
            points.append(points[-1] + unit)
        return points

    best["factual_unit_deltas"] = factual_units
    best["confabulated_unit_deltas"] = confabulated_units
    best["factual_points"] = cumulative_points(factual_units)
    best["confabulated_points"] = cumulative_points(confabulated_units)
    best["factual_mean_cosine"] = float(np.mean(best["factual_cosines"]))
    best["confabulated_mean_cosine"] = float(np.mean(best["confabulated_cosines"]))
    best["explained_variance_ratio"] = list(
        factual_payload.get("explained_variance_ratio")
        or confabulated_payload.get("explained_variance_ratio")
        or []
    )
    return best


def _mix_channel(start: int, end: int, ratio: float) -> int:
    return int(round(start + (end - start) * ratio))


def _risk_color(risk_score: float) -> str:
    risk = max(0.0, min(1.0, float(risk_score)))
    green = (22, 163, 74)
    yellow = (245, 158, 11)
    red = (220, 38, 38)
    if risk <= 0.5:
        ratio = risk / 0.5
        rgb = tuple(_mix_channel(start, end, ratio) for start, end in zip(green, yellow))
    else:
        ratio = (risk - 0.5) / 0.5
        rgb = tuple(_mix_channel(start, end, ratio) for start, end in zip(yellow, red))
    return "#%02x%02x%02x" % rgb


def _text_color_for_hex(hex_color: str) -> str:
    red = int(hex_color[1:3], 16)
    green = int(hex_color[3:5], 16)
    blue = int(hex_color[5:7], 16)
    brightness = (red * 299 + green * 587 + blue * 114) / 1000
    return "#0F172A" if brightness >= 150 else "#FFFFFF"


def _token_html(token: str) -> str:
    escaped = escape(token).replace(" ", "&nbsp;").replace("\n", "<br/>")
    return escaped if escaped else "&nbsp;"


def build_layer_coherence_chart(
    result_or_udc_per_layer: Any,
    num_layers: int | None = None,
    title: str = "",
    *,
    comparison_data: Any | None = None,
    primary_label: str = "Primary answer",
    comparison_label: str = "Comparison answer",
) -> go.Figure:
    """Build a per-layer coherence chart from a result object or raw values."""

    udc_per_layer, num_layers = _coerce_layer_payload(result_or_udc_per_layer, num_layers)
    x_positions = list(range(len(udc_per_layer)))
    x_labels = [f"{layer}->{layer + 1}" for layer in x_positions]
    mean_value = sum(udc_per_layer) / len(udc_per_layer)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=udc_per_layer,
            mode="lines+markers",
            name=primary_label,
            line={"color": _ACCENT_COLOR, "width": 3},
            marker={"size": 6, "color": _ACCENT_COLOR},
            fill="tozeroy",
            fillcolor="rgba(94, 234, 212, 0.12)",
            hovertemplate="Transition %{text}<br>UDC %{y:.4f}<extra></extra>",
            text=x_labels,
        )
    )

    if comparison_data is not None:
        comparison_values, _ = _coerce_layer_payload(comparison_data, num_layers)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(comparison_values))),
                y=comparison_values,
                mode="lines+markers",
                name=comparison_label,
                line={"color": _SECONDARY_COLOR, "width": 3},
                marker={"size": 6, "color": _SECONDARY_COLOR},
                hovertemplate="Transition %{text}<br>UDC %{y:.4f}<extra></extra>",
                text=x_labels[: len(comparison_values)],
            )
        )

    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color=_ZERO_LINE_COLOR,
        annotation_text="orthogonal",
        annotation_position="top left",
    )
    fig.add_hline(
        y=mean_value,
        line_dash="dot",
        line_color="#cbd5e1",
        annotation_text=f"mean = {mean_value:.3f}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        template="plotly_white",
        title=title or "Layer-wise hidden-state coherence",
        paper_bgcolor=_CHART_BACKGROUND,
        plot_bgcolor=_CHART_BACKGROUND,
        font={"color": _TEXT_COLOR, "family": "Inter Tight, system-ui, sans-serif"},
        hovermode="x unified",
        height=360,
        margin={"l": 40, "r": 20, "t": 50, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(
        title_text=f"Layer transition ({num_layers} layers total)",
        tickmode="array",
        tickvals=x_positions,
        ticktext=x_labels,
        showgrid=False,
    )
    fig.update_yaxes(
        title_text="Cosine similarity",
        zeroline=False,
        gridcolor="rgba(15, 23, 42, 0.08)",
    )
    return fig


def build_layer_chart(
    result_or_udc_per_layer: Any,
    num_layers: int | None = None,
    title: str = "",
    comparison_data: Any | None = None,
) -> go.Figure:
    """Backward-compatible alias used by the original spec docs."""

    return build_layer_coherence_chart(
        result_or_udc_per_layer,
        num_layers=num_layers,
        title=title,
        comparison_data=comparison_data,
    )


def build_risk_gauge(
    result_or_risk_score: Any,
    verdict: str | None = None,
    *,
    title: str = "Confabulation Risk",
) -> go.Figure:
    """Build a compact risk gauge from an app-facing result or a raw risk score."""

    if isinstance(result_or_risk_score, Mapping) or hasattr(result_or_risk_score, "risk_score"):
        if _get_value(result_or_risk_score, "risk_score") is not None:
            risk_score = float(_get_value(result_or_risk_score, "risk_score", 0.0))
            verdict_text = str(
                _get_value(
                    result_or_risk_score,
                    "calibrated_verdict",
                    _get_value(result_or_risk_score, "verdict", verdict or "REVIEW"),
                )
            )
        elif _get_value(result_or_risk_score, "calibration") is not None:
            adapted = adapt_analysis_result(result_or_risk_score)
            risk_score = adapted.risk_score
            verdict_text = adapted.calibrated_verdict
        else:
            risk_score = 0.0
            verdict_text = verdict or "REVIEW"
    else:
        risk_score = float(result_or_risk_score)
        verdict_text = verdict or "REVIEW"

    risk_percent = max(0.0, min(100.0, risk_score * 100.0))
    bar_color = {"PASS": "#16A34A", "REVIEW": "#F59E0B", "FAIL": "#DC2626"}.get(
        verdict_text,
        "#F59E0B",
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_percent,
            number={"suffix": "%", "font": {"size": 36, "color": _TEXT_COLOR}},
            title={"text": title, "font": {"size": 20, "color": _TEXT_COLOR}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": bar_color, "thickness": 0.32},
                "bgcolor": "#F1F5F9",
                "steps": [
                    {"range": [0, 33], "color": "rgba(22, 163, 74, 0.12)"},
                    {"range": [33, 66], "color": "rgba(245, 158, 11, 0.12)"},
                    {"range": [66, 100], "color": "rgba(220, 38, 38, 0.12)"},
                ],
                "threshold": {
                    "line": {"color": _TEXT_COLOR, "width": 3},
                    "thickness": 0.8,
                    "value": risk_percent,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=_CHART_BACKGROUND,
        plot_bgcolor=_CHART_BACKGROUND,
        font={"color": _TEXT_COLOR, "family": "Inter Tight, system-ui, sans-serif"},
        height=270,
        margin={"l": 20, "r": 20, "t": 50, "b": 60},
        annotations=[
            {
                "text": f"Verdict: {escape(verdict_text)}",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.15,
                "showarrow": False,
                "font": {"size": 15, "color": "#475569"},
            }
        ],
    )
    return fig


def build_token_heatmap_html(
    result_or_response_tokens: Any,
    udc_per_token: Sequence[float] | None = None,
    risk_scores_per_token: Sequence[float] | None = None,
) -> str:
    """Render a token heatmap as HTML for Streamlit markdown."""

    response_tokens, raw_scores, risk_scores = _coerce_heatmap_payload(
        result_or_response_tokens,
        udc_per_token,
        risk_scores_per_token,
    )
    if not (len(response_tokens) == len(raw_scores) == len(risk_scores)):
        raise ValueError("Token labels, raw scores, and risk scores must have matching lengths.")

    spans: list[str] = []
    for index, (token, raw_score, risk_score) in enumerate(zip(response_tokens, raw_scores, risk_scores)):
        if token.strip() == "":
            spans.append(
                (
                    f"<span data-token-index=\"{index}\" "
                    "style=\"display:inline-block;min-width:0.6rem;height:1.6rem;"
                    "margin:1px 2px;border-bottom:1px solid #E2E8F0;\"></span>"
                )
            )
            continue

        background = _risk_color(risk_score)
        foreground = _text_color_for_hex(background)
        title = escape(f"{token} | UDC={raw_score:.4f} | risk={risk_score:.2f}")
        spans.append(
            (
                f"<span data-token-index=\"{index}\" title=\"{title}\" "
                "style=\"display:inline-block;margin:2px;padding:3px 8px;border-radius:6px;"
                f"background-color:{background};color:{foreground};"
                "font-family:'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;"
                "font-size:0.85rem;line-height:1.5;font-weight:500;"
                "outline:1.5px solid transparent;transition:outline-color 100ms ease;\""
                f">{_token_html(token)}</span>"
            )
        )

    joined_spans = "".join(spans)
    return (
        "<div style=\"background:#FFFFFF;border:1px solid #E2E8F0;"
        "border-radius:14px;padding:14px 14px;color:#0F172A;line-height:2;\">"
        "<div style=\"font-family:'JetBrains Mono',monospace;font-size:0.75rem;"
        "font-weight:600;color:#0F172A;letter-spacing:0.08em;"
        "text-transform:uppercase;margin-bottom:8px;\">"
        "Per-token coherence"
        "</div>"
        f"<div>{joined_spans}</div>"
        "</div>"
    )


def build_token_heatmap(
    result_or_response_tokens: Any,
    udc_per_token: Sequence[float] | None = None,
    risk_scores_per_token: Sequence[float] | None = None,
) -> str:
    """Backward-compatible alias used by the original spec docs."""

    return build_token_heatmap_html(
        result_or_response_tokens,
        udc_per_token=udc_per_token,
        risk_scores_per_token=risk_scores_per_token,
    )


def build_update_geometry_figure(
    geometry_or_result: Any,
    *,
    title: str = "Projected 3D Update Geometry",
    max_tokens: int = 6,
) -> go.Figure:
    """Render PCA-projected update trajectories for a real analyzed example."""

    if isinstance(geometry_or_result, Mapping) or hasattr(geometry_or_result, "geometry_3d"):
        geometry_payload = _get_value(geometry_or_result, "geometry_3d", geometry_or_result)
    else:
        geometry_payload = geometry_or_result

    geometry = _coerce_geometry_payload(geometry_payload)
    token_paths = list(geometry.get("token_paths") or [])
    if not token_paths:
        raise ValueError("Geometry payload did not contain any token paths.")

    explained = list(geometry.get("explained_variance_ratio") or [])
    explained_text = ", ".join(f"PC{i + 1}={value:.2f}" for i, value in enumerate(explained[:3])) or "n/a"

    palette = [
        "#5eead4",
        "#f97316",
        "#60a5fa",
        "#f472b6",
        "#facc15",
        "#a78bfa",
    ]

    fig = go.Figure()
    for index, path in enumerate(token_paths[:max_tokens]):
        points = path.get("points") or []
        if not points:
            continue
        x_vals = [float(point[0]) for point in points]
        y_vals = [float(point[1]) for point in points]
        z_vals = [float(point[2]) for point in points]
        token_label = str(path.get("token", f"token_{index}")).strip() or f"token_{index}"
        color = palette[index % len(palette)]

        fig.add_trace(
            go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode="lines+markers",
                name=f"{index}: {token_label}",
                line={"color": color, "width": 6},
                marker={"size": 4, "color": color},
                hovertemplate=(
                    f"Token: {escape(token_label)}<br>"
                    "Step %{text}<br>"
                    "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
                ),
                text=[f"L{layer}" for layer in range(len(points))],
            )
        )

    fig.update_layout(
        template="plotly_white",
        title={"text": title, "font": {"color": "#000000"}},
        paper_bgcolor=_CHART_BACKGROUND,
        plot_bgcolor=_CHART_BACKGROUND,
        font={"color": _TEXT_COLOR, "family": "Inter Tight, system-ui, sans-serif"},
        height=520,
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        scene={
            "xaxis": {"title": "PC1", "backgroundcolor": "#F8FAFC", "gridcolor": "rgba(15, 23, 42, 0.06)"},
            "yaxis": {"title": "PC2", "backgroundcolor": "#F8FAFC", "gridcolor": "rgba(15, 23, 42, 0.06)"},
            "zaxis": {"title": "PC3", "backgroundcolor": "#F8FAFC", "gridcolor": "rgba(15, 23, 42, 0.06)"},
            "camera": {"eye": {"x": 1.45, "y": 1.55, "z": 1.15}},
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.98, "xanchor": "left", "x": 0.01},
        annotations=[
            {
                "text": f"PCA variance: {escape(explained_text)}",
                "xref": "paper",
                "yref": "paper",
                "x": 0.01,
                "y": 0.02,
                "showarrow": False,
                "font": {"size": 12, "color": "#475569"},
            }
        ],
    )
    return fig


def build_update_geometry_comparison_figure(
    factual_geometry: Any,
    confabulated_geometry: Any,
    *,
    title: str = "Update Trajectory Comparison",
) -> go.Figure:
    """Render a simple answer comparison as two 2D snakes on one shared chart.

    Each snake is the cumulative path formed by the mean projected update vector
    at each layer. More direct paths imply steadier progress; more wandering
    implies the model keeps redirecting itself across layers.
    """

    factual = _aggregate_geometry_snake(factual_geometry)
    confab = _aggregate_geometry_snake(confabulated_geometry)

    factual_points = factual["points"]
    confab_points = confab["points"]
    factual_x = [point[0] for point in factual_points]
    factual_y = [point[1] for point in factual_points]
    confab_x = [point[0] for point in confab_points]
    confab_y = [point[1] for point in confab_points]

    factual_labels = [f"L{idx}" for idx in range(len(factual_points))]
    confab_labels = [f"L{idx}" for idx in range(len(confab_points))]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=factual_x,
            y=factual_y,
            mode="lines+markers+text",
            name="Factual answer",
            line={"color": _ACCENT_COLOR, "width": 5},
            marker={"size": 9, "color": _ACCENT_COLOR},
            text=factual_labels,
            textposition="top center",
            hovertemplate="Factual %{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=confab_x,
            y=confab_y,
            mode="lines+markers+text",
            name="Confabulated answer",
            line={"color": _SECONDARY_COLOR, "width": 5},
            marker={"size": 9, "color": _SECONDARY_COLOR},
            text=confab_labels,
            textposition="bottom center",
            hovertemplate="Confab %{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )

    variance = factual.get("explained_variance_ratio") or confab.get("explained_variance_ratio") or []
    variance_text = ", ".join(f"PC{i + 1}={value:.2f}" for i, value in enumerate(variance[:2])) or "n/a"

    fig.update_layout(
        template="plotly_white",
        title=title,
        paper_bgcolor=_CHART_BACKGROUND,
        plot_bgcolor=_CHART_BACKGROUND,
        font={"color": _TEXT_COLOR, "family": "Inter Tight, system-ui, sans-serif"},
        height=520,
        margin={"l": 40, "r": 20, "t": 60, "b": 80},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        annotations=[
            {
                "text": (
                    "Interpretation: both paths start from the same place. The easier one to trust is usually the one "
                    "that keeps moving more directly instead of wandering around."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": -0.14,
                "showarrow": False,
                "align": "left",
                "font": {"size": 13, "color": "#475569"},
            },
            {
                "text": f"These two visual axes are a compressed view of the real hidden-state updates ({escape(variance_text)})",
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": -0.22,
                "showarrow": False,
                "align": "left",
                "font": {"size": 12, "color": "#94A3B8"},
            },
        ],
    )
    fig.update_xaxes(
        title_text="Visual direction 1",
        zeroline=False,
        gridcolor="rgba(15, 23, 42, 0.08)",
    )
    fig.update_yaxes(
        title_text="Visual direction 2",
        zeroline=False,
        gridcolor="rgba(15, 23, 42, 0.08)",
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def build_focused_direction_comparison_figure(
    factual_geometry: Any,
    confabulated_geometry: Any,
    *,
    factual_tokens: Sequence[str] | None = None,
    confabulated_tokens: Sequence[str] | None = None,
    title: str = "Focused Local Direction Comparison",
) -> tuple[go.Figure, dict[str, Any]]:
    """Render a simplified 3D comparison of a short local update slice.

    The view intentionally uses unit-length arrows for a tiny token window and
    short layer span so users can see what UDC is comparing: neighboring update
    directions, not overall path length.
    """

    focused = extract_focused_direction_slice(
        factual_geometry,
        confabulated_geometry,
        factual_tokens=factual_tokens,
        confabulated_tokens=confabulated_tokens,
    )

    factual_points = list(focused["factual_points"])
    confabulated_points = list(focused["confabulated_points"])

    factual_palette = ["#93C5FD", "#60A5FA", "#3B82F6", "#2563EB"]
    confabulated_palette = ["#FDBA74", "#FB923C", "#F97316", "#EA580C"]

    fig = go.Figure()
    for index in range(len(factual_points) - 1):
        start = factual_points[index]
        end = factual_points[index + 1]
        fig.add_trace(
            go.Scatter3d(
                x=[float(start[0]), float(end[0])],
                y=[float(start[1]), float(end[1])],
                z=[float(start[2]), float(end[2])],
                mode="lines+markers",
                name="Factual",
                line={"color": factual_palette[index % len(factual_palette)], "width": 10},
                marker={
                    "size": [4, 7],
                    "color": factual_palette[index % len(factual_palette)],
                    "symbol": ["circle", "diamond"],
                },
                showlegend=index == 0,
                hovertemplate=(
                    f"Factual step {index + 1}<br>"
                    f"Layers L{focused['layer_start'] + index}→L{focused['layer_start'] + index + 1}"
                    "<extra></extra>"
                ),
            )
        )

    for index in range(len(confabulated_points) - 1):
        start = confabulated_points[index]
        end = confabulated_points[index + 1]
        fig.add_trace(
            go.Scatter3d(
                x=[float(start[0]), float(end[0])],
                y=[float(start[1]), float(end[1])],
                z=[float(start[2]), float(end[2])],
                mode="lines+markers",
                name="Confabulated",
                line={"color": confabulated_palette[index % len(confabulated_palette)], "width": 10},
                marker={
                    "size": [4, 7],
                    "color": confabulated_palette[index % len(confabulated_palette)],
                    "symbol": ["circle", "diamond"],
                },
                showlegend=index == 0,
                hovertemplate=(
                    f"Confabulated step {index + 1}<br>"
                    f"Layers L{focused['layer_start'] + index}→L{focused['layer_start'] + index + 1}"
                    "<extra></extra>"
                ),
            )
        )

    variance = focused.get("explained_variance_ratio") or []
    variance_text = ", ".join(f"PC{i + 1}={value:.2f}" for i, value in enumerate(variance[:3])) or "n/a"

    fig.update_layout(
        template="plotly_white",
        title={"text": title, "font": {"color": "#000000"}},
        paper_bgcolor=_CHART_BACKGROUND,
        plot_bgcolor=_CHART_BACKGROUND,
        font={"color": _TEXT_COLOR, "family": "Inter Tight, system-ui, sans-serif"},
        height=560,
        margin={"l": 0, "r": 0, "t": 60, "b": 30},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.98,
            "xanchor": "left",
            "x": 0.02,
            "font": {"size": 16, "color": "#000000"},
        },
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "dragmode": "orbit",
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.5, "y": 1.45, "z": 1.15}},
            "bgcolor": "#F8FAFC",
        },
        annotations=[
            {
                "text": (
                    "Zoomed to the short phrase and short layer span where the local coherence gap is largest. "
                    "All arrows are normalized to the same length so this view shows direction only."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 0.0,
                "showarrow": False,
                "align": "left",
                "font": {"size": 13, "color": "#000000"},
            },
            {
                "text": f"Compressed hidden-state view ({escape(variance_text)})",
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": -0.06,
                "showarrow": False,
                "align": "left",
                "font": {"size": 12, "color": "#000000"},
            },
        ],
    )
    return fig, focused


def summarize_update_geometry_comparison(
    factual_geometry: Any,
    confabulated_geometry: Any,
) -> dict[str, float]:
    factual = _aggregate_geometry_snake(factual_geometry)
    confab = _aggregate_geometry_snake(confabulated_geometry)
    return {
        "factual_mean_turn_deg": factual["mean_turn_angle_deg"],
        "confabulated_mean_turn_deg": confab["mean_turn_angle_deg"],
        "factual_path_efficiency": factual["path_efficiency"],
        "confabulated_path_efficiency": confab["path_efficiency"],
    }


def display_paper_figure(figure_path: str, caption: str) -> None:
    """Render a static paper figure in Streamlit when the app shell exists."""

    path = Path(figure_path)
    if not path.exists():
        raise FileNotFoundError(f"Paper figure not found: {path}")

    import streamlit as st

    st.image(str(path), caption=caption, use_container_width=True)
