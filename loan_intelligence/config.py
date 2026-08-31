"""Shared constants for the loan intelligence pipeline.

Centralized here so the generator, feature builder, and models all agree on
schema, date conventions, and observation-window definitions.
"""
from __future__ import annotations

import datetime as _dt

# --- Reproducibility -------------------------------------------------------
RANDOM_SEED = 42

# --- Portfolio shape ---------------------------------------------------
N_LOANS = 5000

# As-of date for the synthetic portfolio snapshot (all "current" balances /
# statuses are as of this date). Kept fixed (not datetime.now()) so a
# generated dataset is fully reproducible across runs and machines.
AS_OF_DATE = _dt.date(2026, 6, 30)

# Loans originate in this window. The upper bound is deliberately
# AS_OF_DATE minus (FEATURE_WINDOW_MONTHS + LABEL_WINDOW_MONTHS) so every
# loan in the raw dataset has a *fully observed* feature window and label
# window by AS_OF_DATE -- no partially-censored loans sneak into training.
ORIGINATION_START = _dt.date(2019, 1, 1)
ORIGINATION_END = _dt.date(2024, 6, 30)

# --- Vintage-style observation windows -------------------------------------
# Prediction task: use each loan's first FEATURE_WINDOW_MONTHS of payment
# behavior to predict whether it becomes seriously delinquent (90+ DPD) or
# charged off during the following LABEL_WINDOW_MONTHS. This mirrors how
# credit risk teams validate on vintage curves and is what motivates the
# time-aware (origination-date) train/test split documented in features/.
FEATURE_WINDOW_MONTHS = 12
LABEL_WINDOW_MONTHS = 12
TOTAL_OBSERVATION_MONTHS = FEATURE_WINDOW_MONTHS + LABEL_WINDOW_MONTHS

# --- Categorical domains ----------------------------------------------------
REGIONS = ["Northeast", "Midwest", "South", "West"]
LOAN_TYPES = ["mortgage", "auto", "personal", "heloc"]

# term length (months) by loan type -- kept fixed per type for simplicity
TERM_MONTHS_BY_TYPE = {
    "mortgage": 360,
    "auto": 60,
    "personal": 36,
    "heloc": 180,
}

EMPLOYMENT_STATUSES = ["employed", "self_employed", "unemployed", "retired"]
ORIGINATION_CHANNELS = ["retail", "broker", "correspondent"]

# Delinquency staging used in the monthly payment panel.
DPD_BUCKETS = [0, 30, 60, 90, 120]

# --- Risk tiers (review/decisions.py, portfolio/summary.py) ----------------
# Bucket boundaries for the calibrated default probability. Picked from the
# actual test-set distribution (median ~3%, 97.4th percentile ~30%) so "high"
# is a genuinely small, actionable tail rather than an arbitrary round number.
# Shared by the reviewer-flag logic and the portfolio summary so a loan's
# tier never disagrees between the two views.
RISK_TIER_BOUNDS = {"low": 0.0, "medium": 0.10, "high": 0.30}


def risk_tier(calibrated_probability: float) -> str:
    if calibrated_probability >= RISK_TIER_BOUNDS["high"]:
        return "high"
    if calibrated_probability >= RISK_TIER_BOUNDS["medium"]:
        return "medium"
    return "low"

# --- Messiness injection rates (data/generate_synthetic.py) ----------------
MISSING_RATE = {
    "credit_score_at_origination": 0.04,
    "dti_at_origination": 0.05,
    "interest_rate": 0.03,
    "employment_status": 0.06,
    "borrower_income_at_origination": 0.04,
    "actual_payment_amount": 0.02,  # missed-payment months
}
OUTLIER_RATE = 0.01
DUPLICATE_RATE = 0.006
ANOMALOUS_RECORD_RATE = 0.025
