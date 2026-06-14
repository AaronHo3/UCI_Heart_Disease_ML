"""Unit tests for DeLong test and bootstrap classification metrics."""
import numpy as np
import pytest

from stats import (
    _sensitivity_specificity,
    bootstrap_classification_metrics,
    delong_roc_test,
)


def test_delong_identical_scores_not_significant():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100)
    s = rng.uniform(size=100)
    res = delong_roc_test(y, s, s)
    assert res.diff == pytest.approx(0.0, abs=1e-9)
    assert res.p_value == pytest.approx(1.0)
    assert res.verdict() == "indistinguishable"


def test_delong_detects_clear_difference():
    rng = np.random.default_rng(1)
    n = 400
    y = rng.integers(0, 2, n)
    good = y + rng.normal(0, 0.3, n)   # strongly separates
    bad = rng.uniform(size=n)          # random
    res = delong_roc_test(y, good, bad)
    assert res.auc_a > res.auc_b
    assert res.p_value < 0.05
    assert res.verdict() == "significant"


def test_delong_requires_both_classes():
    with pytest.raises(ValueError):
        delong_roc_test(np.ones(5, dtype=int), np.arange(5.0), np.arange(5.0))


def test_sensitivity_specificity_known_confusion():
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 0, 1])  # TP=1, FN=1, TN=1, FP=1
    sens, spec = _sensitivity_specificity(y, pred)
    assert sens == pytest.approx(0.5)
    assert spec == pytest.approx(0.5)


def test_bootstrap_metrics_ci_brackets_point():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.6 + rng.uniform(size=300) * 0.4, 0, 1)
    panel = bootstrap_classification_metrics(y, p, n_boot=300)
    for est in panel.values():
        if not np.isnan(est.lower):
            assert est.lower <= est.value <= est.upper
    assert set(panel) == {"auc", "accuracy", "f1", "sensitivity", "specificity"}
