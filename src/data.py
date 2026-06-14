"""Cohort-aware data loading for the multi-site heart disease study.

The UCI aggregate bundles four hospital cohorts (Cleveland, Hungary,
Switzerland, VA Long Beach) in a ``dataset`` column. The original pipeline
dropped that column "to avoid cohort bias". For the external-validation study
we keep it as a first-class *grouping* variable (never a feature) so we can
train on some sites and test on a held-out one.

Cleaning decisions are explicit and documented because they materially affect
the analysis:

* ``chol == 0`` is a known data-quality artifact (serum cholesterol of zero is
  physiologically impossible). It is the entire Switzerland cohort plus part of
  VA. We recode it to NaN so downstream imputation handles it honestly rather
  than treating 0 as a real low-cholesterol reading.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_PATH = "data/raw/heart_disease_uci.csv"
SITE_COL = "dataset"
LABEL_COL = "num"
ID_COL = "id"
# Columns that must never become model features.
NON_FEATURE_COLS = (ID_COL, LABEL_COL, SITE_COL)
# Physiologically impossible zeros that actually encode "missing".
ZERO_AS_MISSING_COLS = ("chol",)


@dataclass(frozen=True)
class CohortData:
    """Features, binary target, and site labels held in lockstep."""

    X: pd.DataFrame
    y: pd.Series
    site: pd.Series

    @property
    def sites(self) -> list[str]:
        return sorted(self.site.unique().tolist())

    def for_sites(self, names: list[str]) -> "CohortData":
        """Immutable subset: rows whose site is in ``names``."""
        mask = self.site.isin(names)
        return CohortData(
            X=self.X.loc[mask].reset_index(drop=True),
            y=self.y.loc[mask].reset_index(drop=True),
            site=self.site.loc[mask].reset_index(drop=True),
        )


def _read_raw(data_path: str) -> pd.DataFrame:
    import os

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found at {data_path}. Run `make download` first."
        )
    df = pd.read_csv(data_path)
    for required in (LABEL_COL, SITE_COL):
        if required not in df.columns:
            raise ValueError(f"Expected column '{required}' not found in dataset.")
    return df


def clean(df: pd.DataFrame, recode_zero_as_missing: bool = True) -> pd.DataFrame:
    """Return a cleaned copy (never mutates the input)."""
    out = df.copy()
    if recode_zero_as_missing:
        for col in ZERO_AS_MISSING_COLS:
            if col in out.columns:
                # np.nan (not pd.NA) keeps the column float64 for the scaler.
                out[col] = out[col].replace(0, np.nan)
    return out


def load_cohort_data(
    data_path: str = DATA_PATH,
    recode_zero_as_missing: bool = True,
) -> CohortData:
    """Load features, binary target (``num > 0``), and site labels."""
    df = clean(_read_raw(data_path), recode_zero_as_missing=recode_zero_as_missing)
    y = (df[LABEL_COL] > 0).astype(int)
    site = df[SITE_COL].astype(str)
    drop_cols = [c for c in NON_FEATURE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)
    return CohortData(
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        site=site.reset_index(drop=True),
    )


def cohort_profile(data: CohortData) -> pd.DataFrame:
    """Per-site size, positive count, and prevalence — the shift table."""
    frame = pd.DataFrame({"site": data.site, "y": data.y})
    grouped = frame.groupby("site")["y"]
    profile = pd.DataFrame(
        {
            "n": grouped.size(),
            "n_pos": grouped.sum(),
            "prevalence": grouped.mean(),
        }
    )
    return profile.sort_values("prevalence")
