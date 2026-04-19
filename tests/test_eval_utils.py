import math

from app.eval_utils import partial_auc_ols, raw_auc, summarize_metric


def test_raw_auc_reports_orientation():
    labels = [0, 0, 1, 1]
    scores = [0.9, 0.8, 0.2, 0.1]
    auc, higher_is_more_factual = raw_auc(labels, scores)
    assert math.isclose(auc, 1.0)
    assert higher_is_more_factual is True


def test_partial_auc_ols_is_bounded():
    labels = [0, 0, 1, 1]
    scores = [0.9, 0.8, 0.2, 0.1]
    lengths = [10, 11, 10, 11]
    auc, higher_is_more_factual = partial_auc_ols(labels, scores, lengths)
    assert 0.0 <= auc <= 1.0
    assert higher_is_more_factual is True


def test_summarize_metric_populates_fields():
    labels = [0, 0, 1, 1]
    scores = [0.9, 0.8, 0.2, 0.1]
    lengths = [10, 11, 10, 11]
    summary = summarize_metric(labels, scores, lengths=lengths, metric_name="udc", n_bootstrap=50)
    assert summary.metric == "udc"
    assert 0.0 <= summary.auc <= 1.0
    assert summary.partial_auc_len is not None
    assert summary.rho_len is not None

