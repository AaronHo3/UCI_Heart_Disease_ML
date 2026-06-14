"""Unit tests for metric helpers (bootstrap CI, ECE, Brier)."""
import numpy as np
import pytest

from metrics import bootstrap_auc, brier, expected_calibration_error, summarize


def test_perfect_separation_auc_is_one():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    est = bootstrap_auc(y, s, n_boot=200)
    assert est.value == pytest.approx(1.0)
    assert est.lower <= est.value <= est.upper


def test_single_class_auc_is_nan():
    est = bootstrap_auc(np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9]))
    assert np.isnan(est.value)


def test_ece_bounds_and_calibrated_is_low():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)  # perfectly calibrated by construction
    ece = expected_calibration_error(y, p, n_bins=10)
    assert 0.0 <= ece <= 1.0
    assert ece < 0.05


def test_ece_detects_miscalibration():
    y = np.zeros(1000, dtype=int)
    p = np.full(1000, 0.9)  # confidently wrong
    assert expected_calibration_error(y, p) == pytest.approx(0.9, abs=1e-6)


def test_brier_perfect_is_zero():
    y = np.array([0, 1, 0, 1])
    assert brier(y, y.astype(float)) == pytest.approx(0.0)


def test_summarize_returns_consistent_report():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(size=200)
    rep = summarize("x", y, p, n_boot=200)
    assert rep.n == 200
    assert rep.n_pos == int(y.sum())
    assert 0.0 <= rep.ece <= 1.0
