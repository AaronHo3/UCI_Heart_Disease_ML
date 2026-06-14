"""Fast end-to-end smoke test: the pipeline trains and predicts on real data."""
import numpy as np
import pytest
from sklearn.base import clone

from data import load_cohort_data
from pipeline_utils import build_logreg_pipeline


@pytest.mark.smoke
def test_pipeline_fits_and_predicts():
    data = load_cohort_data()
    # Subsample for speed; the point is that the wiring works end to end.
    X = data.X.iloc[:200]
    y = data.y.iloc[:200]
    model = clone(build_logreg_pipeline(X)).fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (200,)
    assert np.all((proba >= 0) & (proba <= 1))
