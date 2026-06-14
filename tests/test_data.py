"""Unit tests for the cohort-aware data loader."""
import numpy as np
import pandas as pd
import pytest

from data import CohortData, clean, cohort_profile, load_cohort_data


@pytest.fixture(scope="module")
def cohort() -> CohortData:
    return load_cohort_data()


def test_aligned_lengths(cohort):
    assert len(cohort.X) == len(cohort.y) == len(cohort.site)


def test_target_is_binary(cohort):
    assert set(cohort.y.unique()).issubset({0, 1})


def test_non_feature_columns_excluded(cohort):
    for col in ("id", "num", "dataset"):
        assert col not in cohort.X.columns


def test_four_sites(cohort):
    assert set(cohort.sites) == {"Cleveland", "Hungary", "Switzerland", "VA Long Beach"}


def test_chol_zero_recoded_to_nan(cohort):
    # The recoded artifact: no literal zeros remain, and NaNs now exist.
    assert (cohort.X["chol"] == 0).sum() == 0
    assert cohort.X["chol"].isna().any()


def test_clean_does_not_mutate_input():
    df = pd.DataFrame({"chol": [0.0, 200.0], "num": [0, 1], "dataset": ["A", "B"]})
    before = df["chol"].tolist()
    _ = clean(df)
    assert df["chol"].tolist() == before  # original untouched


def test_clean_recode_toggle():
    df = pd.DataFrame({"chol": [0.0, 200.0]})
    kept = clean(df, recode_zero_as_missing=False)
    assert (kept["chol"] == 0).sum() == 1


def test_for_sites_subsets_and_is_immutable(cohort):
    sub = cohort.for_sites(["Cleveland"])
    assert set(sub.site.unique()) == {"Cleveland"}
    assert len(sub.X) < len(cohort.X)
    # Original is unchanged.
    assert set(cohort.sites) == {"Cleveland", "Hungary", "Switzerland", "VA Long Beach"}


def test_cohort_profile_prevalence_matches(cohort):
    prof = cohort_profile(cohort)
    # Switzerland is the extreme high-prevalence cohort.
    assert prof.loc["Switzerland", "prevalence"] > 0.9
    assert np.isclose(prof.loc["Cleveland", "prevalence"], cohort.for_sites(["Cleveland"]).y.mean())


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cohort_data(data_path=str(tmp_path / "nope.csv"))
