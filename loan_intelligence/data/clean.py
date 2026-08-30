"""Turns the raw, messy loans.csv into an analysis-ready frame.

profile_report.py only *reports* data-quality issues; this module actually
resolves them, using exactly the issues that report surfaces:

- origination_date: mixed ISO / MM-DD-YYYY strings -> parsed to datetime.
- dti_at_origination: mixed fraction (0-1) vs percentage (0-100) units ->
  rescaled to a single fraction unit.
- interest_rate: mixed "7.88%"-string vs float -> stripped to float (no
  rescaling needed, both encode the same percentage-points scale).
- region / loan_type: casing and abbreviation variants -> mapped back to
  the canonical category set.
- domain-range violations (credit_score outside [300, 850], interest_rate
  outside [0, 40]) -> treated as missing, since the true value is not
  recoverable, then imputed downstream in features/build_features.py.
- duplicate loan_id rows -> first occurrence kept; documented rather than
  silently deduped so the profiling report and the cleaning step agree on
  what happened.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CREDIT_SCORE_RANGE = (300, 850)
INTEREST_RATE_RANGE = (0.0, 40.0)

REGION_CANONICAL = {"northeast": "Northeast", "n": "Northeast",
                    "midwest": "Midwest", "m": "Midwest",
                    "south": "South", "s": "South",
                    "west": "West", "w": "West"}
LOAN_TYPE_CANONICAL = {"mortgage": "mortgage", "auto": "auto", "personal": "personal", "heloc": "heloc"}


def _normalize_category(series: pd.Series, canonical_map: dict[str, str]) -> pd.Series:
    key = series.astype(str).str.lower().str.replace("_loan", "", regex=False).str.strip()
    return key.map(canonical_map).where(key.map(canonical_map).notna(), series)


def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    iso = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    slash = pd.to_datetime(series, format="%m/%d/%Y", errors="coerce")
    return iso.fillna(slash)


def _unify_dti(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    # anything > 1.5 is unambiguously a percentage in this domain (max true
    # fraction is ~0.68); rescale those back to a 0-1 fraction.
    return np.where(numeric > 1.5, numeric / 100.0, numeric)


def _unify_interest_rate(series: pd.Series) -> pd.Series:
    stripped = series.astype(str).str.rstrip("%")
    return pd.to_numeric(stripped, errors="coerce")


def clean_loans(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["origination_date"] = _parse_mixed_dates(df["origination_date"])
    df["dti_at_origination"] = _unify_dti(df["dti_at_origination"])
    df["interest_rate"] = _unify_interest_rate(df["interest_rate"])
    df["region"] = _normalize_category(df["region"], REGION_CANONICAL)
    df["loan_type"] = _normalize_category(df["loan_type"], LOAN_TYPE_CANONICAL)

    lo, hi = CREDIT_SCORE_RANGE
    df.loc[(df["credit_score_at_origination"] < lo) | (df["credit_score_at_origination"] > hi), "credit_score_at_origination"] = np.nan

    lo, hi = INTEREST_RATE_RANGE
    df.loc[(df["interest_rate"] < lo) | (df["interest_rate"] > hi), "interest_rate"] = np.nan

    n_before = len(df)
    df = df.drop_duplicates(subset="loan_id", keep="first").reset_index(drop=True)
    n_dupes_dropped = n_before - len(df)
    if n_dupes_dropped:
        print(f"clean_loans: dropped {n_dupes_dropped} duplicate loan_id rows (kept first occurrence)")

    return df
