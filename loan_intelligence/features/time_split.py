"""Time-aware train/test split, plus leakage-safe imputation and peer-cohort
features fit on train only.

Why split by origination_date instead of randomly:

  1. Loans originated in the same month share macro conditions (rate
     environment, regional employment shocks, underwriting-standard
     changes). A random split puts loans from the same vintage on both
     sides, so the model can partly "memorize" a vintage's outcome
     distribution instead of learning generalizable risk relationships --
     inflating validation AUC relative to what the model will see on truly
     future business.
  2. It mirrors actual deployment: a model trained today only ever scores
     loans that originate *after* its training cutoff. A time-aware split
     is the only split that measures the thing the model will actually be
     asked to do.
  3. It exposes temporal drift (e.g. underwriting standards loosening in
     later vintages) that a random split would hide by shuffling it away.

Anything requiring cross-loan statistics -- missing-value imputation,
peer-cohort z-scores -- is fit on the train split only and applied
unchanged to test, so no information about the test distribution leaks
into how train rows are represented.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from loan_intelligence import config as cfg

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"

WAS_MISSING_COLS = [
    "credit_score_at_origination",
    "dti_at_origination",
    "interest_rate",
    "borrower_income_at_origination",
]
COHORT_ZSCORE_COLS = ["dti_at_origination", "credit_score_at_origination"]
NON_FEATURE_COLS = {"loan_id", "origination_date", "default_flag", "status_at_month_24"}


def time_aware_split(tabular: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    df = tabular.copy()
    df["origination_date"] = pd.to_datetime(df["origination_date"])
    df = df.sort_values("origination_date").reset_index(drop=True)

    cutoff = df["origination_date"].quantile(1 - test_size)
    train_df = df[df["origination_date"] <= cutoff].copy()
    test_df = df[df["origination_date"] > cutoff].copy()
    return train_df, test_df, cutoff


def fit_transform_leakage_safe(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    fit_params: dict = {"medians": {}, "modes": {}, "cohort_stats": {}}

    for col in WAS_MISSING_COLS:
        train_df[f"{col}_was_missing"] = train_df[col].isna().astype(int)
        test_df[f"{col}_was_missing"] = test_df[col].isna().astype(int)

    numeric_cols = [
        c for c in train_df.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    for col in numeric_cols:
        median = float(train_df[col].median())
        fit_params["medians"][col] = median
        train_df[col] = train_df[col].fillna(median)
        test_df[col] = test_df[col].fillna(median)

    train_df["employment_status_was_missing"] = train_df["employment_status"].isna().astype(int)
    test_df["employment_status_was_missing"] = test_df["employment_status"].isna().astype(int)
    mode = train_df["employment_status"].mode().iloc[0]
    fit_params["modes"]["employment_status"] = mode
    train_df["employment_status"] = train_df["employment_status"].fillna(mode)
    test_df["employment_status"] = test_df["employment_status"].fillna(mode)

    for col in COHORT_ZSCORE_COLS:
        stats = train_df.groupby("loan_type")[col].agg(["mean", "std"])
        fit_params["cohort_stats"][col] = stats.to_dict(orient="index")
        for df_ in (train_df, test_df):
            means = df_["loan_type"].map(stats["mean"])
            stds = df_["loan_type"].map(stats["std"]).replace(0, np.nan)
            df_[f"{col}_peer_zscore"] = ((df_[col] - means) / stds).fillna(0.0)

    return train_df, test_df, fit_params


def main() -> None:
    tabular = pd.read_csv(OUTPUT_DIR / "features_tabular.csv")
    train_df, test_df, cutoff = time_aware_split(tabular)
    train_df, test_df, fit_params = fit_transform_leakage_safe(train_df, test_df)

    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"
    fit_path = OUTPUT_DIR / "feature_fit_params.json"
    split_ids_path = OUTPUT_DIR / "split_loan_ids.json"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    fit_path.write_text(json.dumps(fit_params, indent=2, default=str), encoding="utf-8")
    split_ids_path.write_text(
        json.dumps(
            {
                "cutoff_date": str(cutoff.date()),
                "train_loan_ids": train_df["loan_id"].tolist(),
                "test_loan_ids": test_df["loan_id"].tolist(),
            }
        ),
        encoding="utf-8",
    )

    print(f"cutoff date: {cutoff.date()}")
    print(f"train: {len(train_df)} loans ({train_df['default_flag'].mean():.3%} default rate)")
    print(f"test:  {len(test_df)} loans ({test_df['default_flag'].mean():.3%} default rate)")
    print(f"wrote {train_path}, {test_path}, {fit_path}, {split_ids_path}")


if __name__ == "__main__":
    main()
