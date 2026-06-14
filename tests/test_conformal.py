"""Unit tests for split-conformal helpers + a coverage smoke test."""
import numpy as np
import pytest

from conformal import (
    _coverage,
    _set_size_breakdown,
    conformal_quantile,
    prediction_sets,
)


def test_quantile_grows_as_alpha_shrinks():
    scores = np.linspace(0, 1, 200)
    q_strict = conformal_quantile(scores, alpha=0.05)
    q_loose = conformal_quantile(scores, alpha=0.30)
    assert q_strict >= q_loose  # tighter coverage -> larger threshold -> bigger sets


def test_prediction_set_inclusion_logic():
    proba = np.array([[0.9, 0.1], [0.4, 0.6]])
    sets = prediction_sets(proba, q_hat=0.5)  # include class c if p(c) >= 0.5
    assert sets.tolist() == [[True, False], [False, True]]


def test_coverage_on_toy():
    sets = np.array([[True, False], [True, True], [False, True]])
    y = np.array([0, 1, 0])  # covered, covered, NOT covered
    assert _coverage(sets, y) == pytest.approx(2 / 3)


def test_set_size_breakdown_sums_to_one():
    # 4 rows -> clean halves/quarters that survive 3-decimal rounding.
    sets = np.array([[True, False], [True, True], [False, False], [True, False]])
    b = _set_size_breakdown(sets)
    assert b["pct_singleton"] == pytest.approx(0.5)
    assert b["pct_uncertain"] == pytest.approx(0.25)
    assert b["pct_empty"] == pytest.approx(0.25)
    assert b["pct_singleton"] + b["pct_uncertain"] + b["pct_empty"] == pytest.approx(1.0)


@pytest.mark.smoke
def test_conformal_coverage_on_separable_data():
    """End-to-end: split-conformal should roughly hit its target coverage."""
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    signal = y + rng.normal(0, 1.0, n)
    p1 = 1 / (1 + np.exp(-signal))
    proba = np.column_stack([1 - p1, p1])

    cal, test = slice(0, 1000), slice(1000, n)
    cal_scores = 1 - proba[cal][np.arange(1000), y[cal]]
    q_hat = conformal_quantile(cal_scores, alpha=0.1)
    sets = prediction_sets(proba[test], q_hat)
    cov = _coverage(sets, y[test])
    assert cov >= 0.85  # generous band for a single split
