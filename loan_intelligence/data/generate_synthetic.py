"""Synthetic loan-tape generator.

Produces a realistic-but-fake loan-level dataset in two layers:

1. A *clean* internal simulation: static loan attributes drawn from
   correlated distributions, plus a 24-month monthly payment panel driven by
   a latent per-loan risk score through a simple Markov delinquency model
   (0 -> 30 -> 60 -> 90+/charged-off, with cure probabilities). A subset of
   loans get a deliberately injected structural anomaly (DTI spike or a
   payment-break/catch-up pattern) independent of their default outcome --
   this is what models/anomaly.py is meant to recover, and ground_truth.csv
   keeps the answer key out of the delivered files so it can be used purely
   for evaluation (precision @ alert budget, lift), mirroring how "The
   Watchtower" anomaly detector was scored.
2. A *messiness* pass applied only to the delivered CSVs (loans.csv,
   payments.csv): missing values, unit/format inconsistencies (dates, DTI
   as fraction vs percentage, "%"-suffixed rates, casing/typo variants of
   categoricals), a few outlier records, and a handful of duplicated
   loan_ids. The join key is preserved except for the intentional
   duplicates, so payments.csv always merges cleanly onto loans.csv.

Run directly: `python -m loan_intelligence.data.generate_synthetic`
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from loan_intelligence import config as cfg

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"


# ---------------------------------------------------------------------------
# Layer 1: clean static loan attributes
# ---------------------------------------------------------------------------

def _generate_static_loans(rng: np.random.Generator, n: int) -> pd.DataFrame:
    loan_type = rng.choice(cfg.LOAN_TYPES, size=n, p=[0.30, 0.32, 0.28, 0.10])
    term_months = np.array([cfg.TERM_MONTHS_BY_TYPE[t] for t in loan_type])

    region = rng.choice(cfg.REGIONS, size=n, p=[0.24, 0.24, 0.30, 0.22])
    # regions carry a small baseline income multiplier and macro risk effect
    region_income_mult = {"Northeast": 1.15, "Midwest": 0.95, "South": 0.90, "West": 1.10}
    region_risk_effect = {"Northeast": -0.05, "Midwest": 0.00, "South": 0.10, "West": 0.05}

    employment_status = rng.choice(
        cfg.EMPLOYMENT_STATUSES, size=n, p=[0.78, 0.12, 0.04, 0.06]
    )
    employment_risk_effect = {
        "employed": -0.05, "self_employed": 0.10, "unemployed": 0.60, "retired": -0.10,
    }

    origination_channel = rng.choice(cfg.ORIGINATION_CHANNELS, size=n, p=[0.55, 0.30, 0.15])

    origination_days = (cfg.ORIGINATION_END - cfg.ORIGINATION_START).days
    origination_date = np.array(
        [cfg.ORIGINATION_START + dt.timedelta(days=int(d)) for d in rng.integers(0, origination_days + 1, size=n)]
    )

    credit_score = np.clip(rng.normal(700, 60, size=n), 300, 850).round().astype(int)

    base_income = rng.lognormal(mean=10.9, sigma=0.45, size=n)  # ~ median ~55k
    income_mult = np.array([region_income_mult[r] for r in region])
    borrower_income_at_origination = np.round(base_income * income_mult, 2)

    loan_amount_base = {
        "mortgage": (12.6, 0.35), "auto": (10.1, 0.35), "personal": (9.0, 0.55), "heloc": (10.5, 0.45),
    }
    mu = np.array([loan_amount_base[t][0] for t in loan_type])
    sigma = np.array([loan_amount_base[t][1] for t in loan_type])
    loan_amount = np.round(rng.lognormal(mu, sigma), 2)

    # DTI anti-correlated with credit score, with independent noise
    credit_z = (credit_score - 700) / 60
    dti_at_origination = np.clip(0.36 - 0.07 * credit_z + rng.normal(0, 0.08, size=n), 0.04, 0.68)
    dti_at_origination = np.round(dti_at_origination, 4)

    base_rate = {"mortgage": 6.2, "auto": 7.0, "personal": 10.5, "heloc": 8.0}
    rate_base = np.array([base_rate[t] for t in loan_type])
    risk_premium = np.clip(-1.8 * credit_z, -1.5, 4.0) + 6.0 * (dti_at_origination - 0.35)
    interest_rate = np.round(np.clip(rate_base + risk_premium + rng.normal(0, 0.4, size=n), 1.5, 24.0), 3)

    loan_id = np.array([f"L{100000 + i}" for i in range(n)])

    df = pd.DataFrame(
        {
            "loan_id": loan_id,
            "origination_date": origination_date,
            "loan_type": loan_type,
            "term_months": term_months,
            "region": region,
            "employment_status": employment_status,
            "origination_channel": origination_channel,
            "credit_score_at_origination": credit_score,
            "borrower_income_at_origination": borrower_income_at_origination,
            "loan_amount": loan_amount,
            "dti_at_origination": dti_at_origination,
            "interest_rate": interest_rate,
        }
    )

    # latent risk score used to drive the monthly delinquency simulation
    income_to_loan = df["borrower_income_at_origination"] / df["loan_amount"].clip(lower=1)
    latent_risk = (
        -0.011 * (df["credit_score_at_origination"] - 700)
        + 2.6 * (df["dti_at_origination"] - 0.35)
        + 0.16 * (df["interest_rate"] - 7.0)
        - 0.35 * (np.log(income_to_loan) - np.log(income_to_loan).mean())
        + df["region"].map(region_risk_effect).astype(float)
        + df["employment_status"].map(employment_risk_effect).astype(float)
        + rng.normal(0, 0.35, size=n)
    )
    df["_latent_risk"] = latent_risk
    return df


# ---------------------------------------------------------------------------
# Layer 1: monthly payment panel via a simple Markov delinquency model
# ---------------------------------------------------------------------------

def _monthly_payment(principal: np.ndarray, annual_rate_pct: np.ndarray, term: np.ndarray) -> np.ndarray:
    r = (annual_rate_pct / 100.0) / 12.0
    r = np.where(r <= 0, 1e-6, r)
    return principal * r / (1 - (1 + r) ** (-term))


def _simulate_payments(rng: np.random.Generator, loans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(loans)
    months = cfg.TOTAL_OBSERVATION_MONTHS

    scheduled_payment = _monthly_payment(
        loans["loan_amount"].to_numpy(),
        loans["interest_rate"].to_numpy(),
        loans["term_months"].to_numpy(),
    )
    balance = loans["loan_amount"].to_numpy().astype(float).copy()
    dpd_state = np.zeros(n, dtype=int)  # index into DPD_BUCKETS, len-1 == charged off
    max_state = len(cfg.DPD_BUCKETS) - 1

    dti = loans["dti_at_origination"].to_numpy().astype(float).copy()
    income = loans["borrower_income_at_origination"].to_numpy().astype(float).copy()

    base_hazard = 1 / (1 + np.exp(-loans["_latent_risk"].to_numpy()))
    base_hazard = np.clip(base_hazard * 0.10, 0.002, 0.35)  # monthly worsening prob when current
    persistence_mult = {0: 1.0, 1: 1.6, 2: 1.9, 3: 2.3}
    cure_prob = {1: 0.35, 2: 0.18, 3: 0.06}

    # --- anomaly injection setup ---
    n_anom = int(round(n * cfg.ANOMALOUS_RECORD_RATE))
    anom_idx = rng.choice(n, size=n_anom, replace=False)
    anom_type = rng.choice(["dti_spike", "payment_break"], size=n_anom, p=[0.55, 0.45])
    anom_month = rng.integers(4, cfg.FEATURE_WINDOW_MONTHS - 1, size=n_anom)
    is_anomalous = np.zeros(n, dtype=bool)
    is_anomalous[anom_idx] = True
    anomaly_type_arr = np.array(["none"] * n, dtype=object)
    anomaly_type_arr[anom_idx] = anom_type
    anomaly_month_arr = np.full(n, -1, dtype=int)
    anomaly_month_arr[anom_idx] = anom_month
    payment_break_active_until = np.full(n, -1, dtype=int)

    records = []
    ever_90plus_in_label_window = np.zeros(n, dtype=bool)

    for m in range(1, months + 1):
        payment_date = [cfg.ORIGINATION_START] * 0  # placeholder, computed per-row below

        # dti / income drift
        dti = np.clip(dti + rng.normal(0, 0.006, size=n), 0.02, 0.90)
        income = np.clip(income * (1 + rng.normal(0, 0.004, size=n)), 5000, None)

        # structural anomaly: DTI spike, decays but leaves a residual
        spike_now = is_anomalous & (anomaly_type_arr == "dti_spike") & (anomaly_month_arr == m)
        if spike_now.any():
            dti[spike_now] = np.clip(dti[spike_now] * rng.uniform(1.5, 2.3, size=spike_now.sum()), 0.05, 0.95)
        spike_decay = is_anomalous & (anomaly_type_arr == "dti_spike") & (m > anomaly_month_arr) & (anomaly_month_arr > 0)
        if spike_decay.any():
            dti[spike_decay] = np.clip(dti[spike_decay] * 0.90 + 0.35 * 0.10, 0.05, 0.95)

        # payment-break trigger: force two missed months then a full catch-up
        break_now = is_anomalous & (anomaly_type_arr == "payment_break") & (anomaly_month_arr == m)
        payment_break_active_until[break_now] = m + 1

        active = dpd_state < max_state
        u = rng.uniform(size=n)
        hazard = base_hazard * np.vectorize(persistence_mult.get)(dpd_state)
        forced_miss = (payment_break_active_until >= m) & active
        worsen = active & (forced_miss | (u < hazard))

        cure_u = rng.uniform(size=n)
        cure = np.zeros(n, dtype=bool)
        for state, p in cure_prob.items():
            mask = active & (dpd_state == state) & ~worsen & (cure_u < p) & (payment_break_active_until < m)
            cure |= mask

        # catch-up cure right after a forced payment-break: unusually clean, all-at-once cure
        catchup = is_anomalous & (anomaly_type_arr == "payment_break") & (m == payment_break_active_until + 1)
        cure |= (catchup & active)

        dpd_state = np.where(worsen, np.minimum(dpd_state + 1, max_state), dpd_state)
        dpd_state = np.where(cure & ~worsen, 0, dpd_state)

        made_payment = active & ~worsen
        actual_payment = np.where(made_payment, scheduled_payment * rng.uniform(0.98, 1.02, size=n), 0.0)
        actual_payment = np.where(catchup, scheduled_payment * rng.uniform(1.8, 2.4, size=n), actual_payment)
        principal_paid = np.where(made_payment | catchup, np.minimum(actual_payment * 0.35, balance), 0.0)
        balance = np.clip(balance - principal_paid + np.where(~made_payment & ~catchup & active, balance * 0.003, 0.0), 0, None)

        days_past_due = np.array([cfg.DPD_BUCKETS[s] for s in dpd_state])
        in_label_window = m > cfg.FEATURE_WINDOW_MONTHS
        if in_label_window:
            ever_90plus_in_label_window |= days_past_due >= 90

        dates = pd.to_datetime(loans["origination_date"]) + pd.DateOffset(months=m)

        records.append(
            pd.DataFrame(
                {
                    "loan_id": loans["loan_id"].to_numpy(),
                    "month_index": m,
                    "payment_date": dates.dt.date.to_numpy(),
                    "scheduled_payment_amount": np.round(scheduled_payment, 2),
                    "actual_payment_amount": np.round(actual_payment, 2),
                    "remaining_balance": np.round(balance, 2),
                    "days_past_due": days_past_due,
                    "delinquency_flag": (days_past_due >= 30).astype(int),
                    "dti_snapshot": np.round(dti, 4),
                    "income_snapshot": np.round(income, 2),
                }
            )
        )

    payments = pd.concat(records, ignore_index=True).sort_values(["loan_id", "month_index"]).reset_index(drop=True)

    ground_truth = pd.DataFrame(
        {
            "loan_id": loans["loan_id"].to_numpy(),
            "is_anomalous": is_anomalous,
            "anomaly_type": anomaly_type_arr,
            "anomaly_month": anomaly_month_arr,
        }
    )

    default_flag = ever_90plus_in_label_window.astype(int)
    final_dpd = payments.loc[payments["month_index"] == months].set_index("loan_id")["days_past_due"]
    status_at_month_24 = np.select(
        [final_dpd.to_numpy() >= 90, final_dpd.to_numpy() >= 30, final_dpd.to_numpy() > 0],
        ["defaulted", "delinquent", "past_due"],
        default="current",
    )

    loans_out = loans.drop(columns=["_latent_risk"]).copy()
    loans_out["default_flag"] = default_flag
    loans_out["status_at_month_24"] = status_at_month_24

    return loans_out, payments, ground_truth


# ---------------------------------------------------------------------------
# Layer 2: messiness injection (delivered files only)
# ---------------------------------------------------------------------------

def _inject_messiness(rng: np.random.Generator, loans: pd.DataFrame, payments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    loans = loans.copy()
    payments = payments.copy()
    n = len(loans)

    # --- missing values -----------------------------------------------
    for col, rate in cfg.MISSING_RATE.items():
        if col == "actual_payment_amount":
            continue  # handled on the payments panel below
        mask = rng.uniform(size=n) < rate
        loans.loc[mask, col] = np.nan

    miss_mask = rng.uniform(size=len(payments)) < cfg.MISSING_RATE["actual_payment_amount"]
    payments.loc[miss_mask, "actual_payment_amount"] = np.nan

    # --- inconsistent date formats: mix ISO and MM/DD/YYYY strings ------
    dates = pd.to_datetime(loans["origination_date"])
    slash_mask = rng.uniform(size=n) < 0.30
    formatted = np.where(
        slash_mask,
        dates.dt.strftime("%m/%d/%Y"),
        dates.dt.strftime("%Y-%m-%d"),
    )
    loans["origination_date"] = formatted

    # --- DTI unit inconsistency: some rows stored as percentage (0-100) --
    pct_mask = rng.uniform(size=n) < 0.20
    dti_vals = loans["dti_at_origination"].astype(object)
    dti_vals[pct_mask & loans["dti_at_origination"].notna()] = (
        loans.loc[pct_mask & loans["dti_at_origination"].notna(), "dti_at_origination"] * 100
    ).round(2)
    loans["dti_at_origination"] = dti_vals

    # --- interest rate as "%"-suffixed strings for a subset --------------
    rate_str_mask = rng.uniform(size=n) < 0.20
    rate_vals = loans["interest_rate"].astype(object)
    rate_vals[rate_str_mask & loans["interest_rate"].notna()] = loans.loc[
        rate_str_mask & loans["interest_rate"].notna(), "interest_rate"
    ].map(lambda v: f"{v:.2f}%")
    loans["interest_rate"] = rate_vals

    # --- categorical casing / typo variants -------------------------------
    def _messy_variant(value: str, rng_local: np.random.Generator) -> str:
        choice = rng_local.integers(0, 4)
        if choice == 0:
            return value.upper()
        if choice == 1:
            return value.lower()
        if choice == 2:
            return value[:1]  # abbreviation, e.g. "W"
        return value

    region_mask = rng.uniform(size=n) < 0.12
    idx = np.where(region_mask)[0]
    loans.loc[idx, "region"] = [_messy_variant(v, rng) for v in loans.loc[idx, "region"]]

    type_mask = rng.uniform(size=n) < 0.10
    idx = np.where(type_mask)[0]
    loans.loc[idx, "loan_type"] = [
        v.upper() if rng.integers(0, 2) == 0 else f"{v}_loan" for v in loans.loc[idx, "loan_type"]
    ]

    # --- outlier records ---------------------------------------------------
    out_mask = rng.uniform(size=n) < cfg.OUTLIER_RATE
    out_idx = np.where(out_mask)[0]
    for i in out_idx:
        kind = rng.integers(0, 3)
        if kind == 0:
            loans.loc[i, "loan_amount"] = loans.loc[i, "loan_amount"] * rng.uniform(8, 15)
        elif kind == 1:
            loans.loc[i, "credit_score_at_origination"] = rng.choice([-5, 999, 0])
        else:
            loans.loc[i, "interest_rate"] = rng.choice([-3.5, 89.9])

    # --- duplicate loan_ids (re-entry, slightly perturbed) ------------------
    # Only perturb fields a re-keying data-entry error would plausibly touch;
    # identifiers, term, and the outcome label must stay exact on a "duplicate".
    n_dupe = int(round(n * cfg.DUPLICATE_RATE))
    dupe_rows = loans.sample(n=n_dupe, random_state=int(rng.integers(0, 1_000_000))).copy()
    perturbable_cols = ["loan_amount", "borrower_income_at_origination", "credit_score_at_origination"]
    for col in perturbable_cols:
        notna = dupe_rows[col].notna()
        dupe_rows.loc[notna, col] = dupe_rows.loc[notna, col] * (1 + rng.normal(0, 0.01, size=notna.sum()))
    loans = pd.concat([loans, dupe_rows], ignore_index=True)

    return loans, payments


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate(n_loans: int = cfg.N_LOANS, seed: int = cfg.RANDOM_SEED, out_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    rng = np.random.default_rng(seed)

    static_loans = _generate_static_loans(rng, n_loans)
    clean_loans, payments, ground_truth = _simulate_payments(rng, static_loans)
    messy_loans, messy_payments = _inject_messiness(rng, clean_loans, payments)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "loans": out_dir / "loans.csv",
        "payments": out_dir / "payments.csv",
        "ground_truth": out_dir / "ground_truth.csv",
    }
    messy_loans.to_csv(paths["loans"], index=False)
    messy_payments.to_csv(paths["payments"], index=False)
    ground_truth.to_csv(paths["ground_truth"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic, messy loan-level dataset.")
    parser.add_argument("--n-loans", type=int, default=cfg.N_LOANS)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    paths = generate(n_loans=args.n_loans, seed=args.seed, out_dir=args.out_dir)
    for name, path in paths.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
