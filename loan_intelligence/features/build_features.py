"""Feature engineering from the feature window (months 1-12) only.

Design rule: this module only ever touches month_index <= FEATURE_WINDOW_MONTHS
of payments.csv. The label (default_flag) is defined over months 13-24 in the
generator -- if this module read past month 12 it would leak label-window
information into the features. Every aggregate below is computed strictly
per-loan (no cross-loan statistics), so this stage can run identically on
train and test rows; anything that needs cross-loan statistics (peer-cohort
z-scores, missing-value imputation) is deliberately deferred to
features/time_split.py, which fits those on the train split only and applies
them to test -- fitting them here, before the split exists, would leak test
distribution into train.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from loan_intelligence import config as cfg
from loan_intelligence.data.clean import clean_loans

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"

SEQUENCE_CHANNELS = [
    "dti_snapshot",
    "income_snapshot",
    "payment_ratio",
    "missed_payment_flag",
    "days_past_due",
    "delinquency_flag",
]


def _loan_payment_features(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("month_index")
    months = g["month_index"].to_numpy(dtype=float)
    dti = g["dti_snapshot"].to_numpy(dtype=float)
    income = g["income_snapshot"].to_numpy(dtype=float)
    dpd = g["days_past_due"].to_numpy(dtype=float)
    delinquent = g["delinquency_flag"].to_numpy(dtype=float)
    actual = g["actual_payment_amount"].to_numpy(dtype=float)
    scheduled = g["scheduled_payment_amount"].to_numpy(dtype=float)
    balance = g["remaining_balance"].to_numpy(dtype=float)

    missed = np.isnan(actual) | (actual == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(~missed, actual / np.where(scheduled == 0, np.nan, scheduled), np.nan)

    streak = max_streak = 0
    for m in missed:
        streak = streak + 1 if m else 0
        max_streak = max(max_streak, streak)

    dti_slope = np.polyfit(months, dti, 1)[0] if len(months) > 1 else 0.0
    income_slope = np.polyfit(months, income, 1)[0] if len(months) > 1 else 0.0
    dti_change_pct = (dti[-1] - dti[0]) / dti[0] if dti[0] not in (0, np.nan) and not np.isnan(dti[0]) else 0.0
    paydown_ratio = (balance[0] - balance[-1]) / balance[0] if balance[0] not in (0,) else 0.0

    return pd.Series(
        {
            "rolling_delinquency_rate_12m": np.nanmean(delinquent),
            "max_dpd_12m": np.nanmax(dpd),
            "n_missed_payments_12m": int(missed.sum()),
            "longest_missed_streak_12m": max_streak,
            "dti_trend_12m": dti_slope,
            "dti_volatility_12m": np.nanstd(dti),
            "dti_change_pct_12m": dti_change_pct,
            "income_trend_12m": income_slope,
            "avg_payment_ratio_12m": np.nan if np.all(np.isnan(ratio)) else np.nanmean(ratio),
            "payment_volatility_12m": np.nan if np.all(np.isnan(ratio)) else np.nanstd(ratio),
            "balance_paydown_ratio_12m": paydown_ratio,
        }
    )


def build_feature_table(loans_raw: pd.DataFrame, payments_raw: pd.DataFrame) -> pd.DataFrame:
    loans = clean_loans(loans_raw)
    window = payments_raw[payments_raw["month_index"] <= cfg.FEATURE_WINDOW_MONTHS]

    payment_feats = window.groupby("loan_id").apply(_loan_payment_features, include_groups=False)
    tabular = loans.merge(payment_feats, on="loan_id", how="left")

    tabular["loan_to_income_ratio"] = tabular["loan_amount"] / tabular[
        "borrower_income_at_origination"
    ].replace(0, np.nan)

    return tabular


def build_sequence_array(payments_raw: pd.DataFrame, loan_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Returns (n_loans, FEATURE_WINDOW_MONTHS, len(SEQUENCE_CHANNELS)) aligned
    to `loan_ids`, for the Bi-LSTM / GRU sequence models. Raw values only --
    scaling/normalization is a modeling concern and belongs to whichever
    model consumes this (fit the scaler on the train split, not here)."""
    window = payments_raw[payments_raw["month_index"] <= cfg.FEATURE_WINDOW_MONTHS].copy()

    missed = window["actual_payment_amount"].isna() | (window["actual_payment_amount"] == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            ~missed,
            window["actual_payment_amount"] / window["scheduled_payment_amount"].replace(0, np.nan),
            0.0,
        )
    window["payment_ratio"] = ratio
    window["missed_payment_flag"] = missed.astype(float)

    channel_arrays = []
    for ch in SEQUENCE_CHANNELS:
        pivoted = window.pivot(index="loan_id", columns="month_index", values=ch)
        pivoted = pivoted.reindex(index=loan_ids, columns=range(1, cfg.FEATURE_WINDOW_MONTHS + 1))
        channel_arrays.append(pivoted.to_numpy(dtype=float))

    arr = np.stack(channel_arrays, axis=-1)
    arr = np.nan_to_num(arr, nan=0.0)
    return arr, SEQUENCE_CHANNELS


def main() -> None:
    loans_raw = pd.read_csv(OUTPUT_DIR / "loans.csv")
    payments_raw = pd.read_csv(OUTPUT_DIR / "payments.csv")

    tabular = build_feature_table(loans_raw, payments_raw)
    sequences, channels = build_sequence_array(payments_raw, tabular["loan_id"].to_numpy())

    tabular_path = OUTPUT_DIR / "features_tabular.csv"
    seq_path = OUTPUT_DIR / "features_sequences.npz"

    tabular.to_csv(tabular_path, index=False)
    np.savez_compressed(
        seq_path,
        loan_id=tabular["loan_id"].to_numpy(),
        X=sequences,
        channels=np.array(channels),
    )

    print(f"wrote {tabular_path} ({tabular.shape[0]} rows x {tabular.shape[1]} cols)")
    print(f"wrote {seq_path} (X shape {sequences.shape})")


if __name__ == "__main__":
    main()
