"""Derived one-pass feature aggregations over UDC/TLE token scores."""

from __future__ import annotations

from typing import Any

import numpy as np


def _tail(values: np.ndarray, n: int) -> np.ndarray:
    if len(values) <= n:
        return values
    return values[-n:]


def extract_feature_metrics(result: Any) -> dict[str, float]:
    """Compute reusable derived features from a UDCResult-like object."""

    udc = np.asarray(result.udc_per_token, dtype=float)
    tle = np.asarray(result.tle_per_token, dtype=float)
    worst2 = np.sort(udc)[: min(2, len(udc))]
    tail3 = _tail(udc, 3)

    return {
        "udc_mean_tok_mean": float(result.udc_scalar),
        "udc_min_tok": float(np.min(udc)),
        "udc_p10_tok": float(np.percentile(udc, 10)),
        "udc_p25_tok": float(np.percentile(udc, 25)),
        "udc_median_tok": float(np.median(udc)),
        "udc_first_tok": float(udc[0]),
        "udc_last_tok": float(udc[-1]),
        "udc_tail3_mean": float(np.mean(tail3)),
        "udc_tail3_min": float(np.min(tail3)),
        "udc_worst2_mean": float(np.mean(worst2)),
        "udc_range_tok": float(np.max(udc) - np.min(udc)),
        "tle_mean_tok_mean": float(result.tle_scalar),
        "tle_max_tok": float(np.max(tle)),
        "tle_p75_tok": float(np.percentile(tle, 75)),
    }

