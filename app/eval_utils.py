"""Evaluation helpers for benchmarking UDC/TLE across models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


@dataclass
class MetricSummary:
    metric: str
    auc: float
    ci_lo: float
    ci_hi: float
    partial_auc_len: float | None
    rho_len: float | None
    higher_is_more_factual: bool

    def to_dict(self) -> dict:
        return asdict(self)


def raw_auc(labels: Iterable[int], scores: Iterable[float]) -> tuple[float, bool]:
    """Return (direction-corrected AUC, higher_is_more_factual)."""

    y = np.asarray(labels)
    s = np.asarray(scores)
    raw = float(roc_auc_score(y, s))
    # labels: 1 = hallucinated, 0 = factual
    higher_is_more_factual = raw < 0.5
    return max(raw, 1.0 - raw), higher_is_more_factual


def bootstrap_auc(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    y = np.asarray(labels)
    s = np.asarray(scores)
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        y_boot = y[idx]
        if y_boot.sum() == 0 or y_boot.sum() == len(y_boot):
            continue
        auc, _ = raw_auc(y_boot, s[idx])
        aucs.append(auc)
    if not aucs:
        base, _ = raw_auc(y, s)
        return base, base
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def partial_auc_ols(
    labels: Iterable[int],
    scores: Iterable[float],
    confounder: Iterable[float],
) -> tuple[float, bool]:
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    c = np.asarray(confounder, dtype=float)
    X = np.column_stack([np.ones(len(c)), c])
    beta, *_ = np.linalg.lstsq(X, s, rcond=None)
    residual = s - X @ beta
    return raw_auc(y, residual)


def summarize_metric(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    lengths: Iterable[float] | None = None,
    metric_name: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> MetricSummary:
    y = np.asarray(labels)
    s = np.asarray(scores)
    auc, higher_is_more_factual = raw_auc(y, s)
    ci_lo, ci_hi = bootstrap_auc(y, s, n_bootstrap=n_bootstrap, seed=seed)
    if lengths is None:
        return MetricSummary(
            metric=metric_name,
            auc=auc,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            partial_auc_len=None,
            rho_len=None,
            higher_is_more_factual=higher_is_more_factual,
        )

    l = np.asarray(lengths, dtype=float)
    partial_auc, _ = partial_auc_ols(y, s, l)
    rho, _ = spearmanr(s, l)
    return MetricSummary(
        metric=metric_name,
        auc=auc,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        partial_auc_len=float(partial_auc),
        rho_len=float(rho),
        higher_is_more_factual=higher_is_more_factual,
    )

