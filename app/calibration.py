"""Threshold/orientation calibration helpers for UDC/TLE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass
class MetricCalibration:
    metric: str
    higher_is_more_factual: bool
    pass_threshold: float
    review_threshold: float
    aligned_hall_median: float
    aligned_factual_median: float
    source: str
    notes: str

    def to_dict(self) -> dict:
        return asdict(self)


def align_score(score: float, higher_is_more_factual: bool) -> float:
    return float(score if higher_is_more_factual else -score)


def verdict_from_score(score: float, calibration: MetricCalibration) -> str:
    aligned = align_score(score, calibration.higher_is_more_factual)
    if aligned >= calibration.pass_threshold:
        return "PASS"
    if aligned >= calibration.review_threshold:
        return "REVIEW"
    return "FAIL"


def fit_quantile_calibration(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    metric: str,
    higher_is_more_factual: bool,
    source: str,
    notes: str = "",
) -> MetricCalibration:
    """Fit a simple two-threshold calibration from labeled benchmark rows.

    labels: 1 = hallucinated, 0 = factual
    scores: raw metric scores
    """

    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    aligned = s if higher_is_more_factual else -s

    factual = aligned[y == 0]
    hall = aligned[y == 1]
    if len(factual) == 0 or len(hall) == 0:
        raise ValueError("Need both factual and hallucinated samples for calibration.")

    pass_threshold = float(np.percentile(factual, 25))
    review_threshold = float(np.percentile(hall, 75))

    # If overlap is too large and thresholds invert, collapse the REVIEW band
    # around the midpoint between class medians.
    factual_median = float(np.median(factual))
    hall_median = float(np.median(hall))
    if pass_threshold < review_threshold:
        midpoint = (factual_median + hall_median) / 2.0
        spread = abs(factual_median - hall_median) / 6.0
        review_threshold = float(midpoint - spread)
        pass_threshold = float(midpoint + spread)

    return MetricCalibration(
        metric=metric,
        higher_is_more_factual=higher_is_more_factual,
        pass_threshold=pass_threshold,
        review_threshold=review_threshold,
        aligned_hall_median=hall_median,
        aligned_factual_median=factual_median,
        source=source,
        notes=notes or "Quantile-based calibration on labeled benchmark rows.",
    )

