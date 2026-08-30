"""Scenario simulation: shock a feature across the portfolio and see how the
calibrated default probability shifts.

Scope of the counterfactual: a shock changes an exogenous origination-time
driver (interest rate, borrower income, DTI). Features that mechanically
derive from it (loan_to_income_ratio, the peer-cohort z-scores) are
recomputed; the observed payment-history features (rolling delinquency
rate, DTI trend, payment volatility, ...) are held fixed, because this tool
asks "how does the model's risk assessment change under this new
origination condition", not "how would twelve months of payment history
have played out differently" -- re-simulating counterfactual payment
behavior is a much bigger modeling problem and out of scope here.

Peer-cohort z-scores are recomputed against the *original* (pre-shock)
train-cohort mean/std from feature_fit_params.json, not refit on the
shocked population -- refitting would silently absorb the shock into the
benchmark and mask exactly the effect we're trying to show (a shocked loan
should look unusual relative to the normal population, not relative to a
population that's now shocked too).

Run directly, e.g.:
    python -m loan_intelligence.scenario.simulate --shock interest_rate --magnitude 2.0
    python -m loan_intelligence.scenario.simulate --shock regional_income --region South --magnitude -15
    python -m loan_intelligence.scenario.simulate --shock dti --magnitude 0.15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from loan_intelligence.audit.hash_chain import AuditTrail
from loan_intelligence.models.predict import _as_xgb_frame

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

SHOCK_TYPES = ("interest_rate", "dti", "regional_income")


def apply_shock(df: pd.DataFrame, shock_type: str, magnitude: float, fit_params: dict, region: str | None = None) -> pd.DataFrame:
    df = df.copy()

    if shock_type == "interest_rate":
        df["interest_rate"] = (df["interest_rate"] + magnitude).clip(lower=0.1)
    elif shock_type == "dti":
        df["dti_at_origination"] = (df["dti_at_origination"] + magnitude).clip(0.02, 0.95)
    elif shock_type == "regional_income":
        mask = (df["region"] == region) if region else pd.Series(True, index=df.index)
        df.loc[mask, "borrower_income_at_origination"] = df.loc[mask, "borrower_income_at_origination"] * (1 + magnitude / 100.0)
    else:
        raise ValueError(f"unknown shock_type {shock_type!r}, expected one of {SHOCK_TYPES}")

    df["loan_to_income_ratio"] = df["loan_amount"] / df["borrower_income_at_origination"].replace(0, np.nan)

    for col in ["dti_at_origination", "credit_score_at_origination"]:
        stats = fit_params["cohort_stats"][col]
        means = df["loan_type"].map(lambda t: stats[t]["mean"])
        stds = df["loan_type"].map(lambda t: stats[t]["std"]).replace(0, np.nan)
        df[f"{col}_peer_zscore"] = ((df[col] - means) / stds).fillna(0.0)

    return df


def _load_portfolio() -> pd.DataFrame:
    train_df = pd.read_csv(OUTPUT_DIR / "train.csv")
    test_df = pd.read_csv(OUTPUT_DIR / "test.csv")
    return pd.concat([train_df, test_df], ignore_index=True)


def _predict(model: xgb.XGBClassifier, calibrator, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    raw = model.predict_proba(_as_xgb_frame(df, feature_cols))[:, 1]
    return calibrator.predict(raw)


def run_scenario(shock_type: str, magnitude: float, region: str | None = None, top_n: int = 10) -> dict:
    portfolio = _load_portfolio()
    fit_params = json.load(open(OUTPUT_DIR / "feature_fit_params.json"))
    feature_cols = joblib.load(MODEL_DIR / "xgboost_feature_cols.joblib")

    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / "xgboost_model.json")
    calibrator = joblib.load(MODEL_DIR / "calibrator.joblib")

    proba_before = _predict(model, calibrator, portfolio, feature_cols)
    shocked = apply_shock(portfolio, shock_type, magnitude, fit_params, region=region)
    proba_after = _predict(model, calibrator, shocked, feature_cols)

    result_df = portfolio[["loan_id", "region", "loan_type"]].copy()
    result_df["proba_before"] = proba_before
    result_df["proba_after"] = proba_after
    result_df["delta"] = proba_after - proba_before

    summary = {
        "shock_type": shock_type,
        "magnitude": magnitude,
        "region": region,
        "n_loans_in_scope": len(portfolio),
        "portfolio_mean_proba_before": round(float(proba_before.mean()), 4),
        "portfolio_mean_proba_after": round(float(proba_after.mean()), 4),
        "mean_delta": round(float((proba_after - proba_before).mean()), 4),
        "n_newly_high_risk_over_0.5": int(((proba_before < 0.5) & (proba_after >= 0.5)).sum()),
        "top_movers": result_df.sort_values("delta", ascending=False).head(top_n)[
            ["loan_id", "region", "loan_type", "proba_before", "proba_after", "delta"]
        ].to_dict(orient="records"),
    }
    return summary, result_df, proba_before, proba_after


def plot_shift(proba_before: np.ndarray, proba_after: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 40)
    ax.hist(proba_before, bins=bins, alpha=0.55, label="before shock", color="#5b8def")
    ax.hist(proba_after, bins=bins, alpha=0.55, label="after shock", color="#e0685a")
    ax.set_xlabel("calibrated default probability")
    ax.set_ylabel("number of loans")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shock a portfolio feature and see how default probability shifts.")
    parser.add_argument("--shock", choices=SHOCK_TYPES, required=True)
    parser.add_argument("--magnitude", type=float, required=True, help="+2.0 = +2pp interest rate; -15 = -15%% income; +0.15 = +0.15 DTI")
    parser.add_argument("--region", type=str, default=None, help="restrict a regional_income shock to one region")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    print(f"running scenario: shock={args.shock}, magnitude={args.magnitude}, region={args.region or 'all'}")
    summary, result_df, proba_before, proba_after = run_scenario(args.shock, args.magnitude, args.region, args.top_n)

    print("\n=== Scenario result (portfolio-level) ===")
    print(f"  mean probability before: {summary['portfolio_mean_proba_before']:.4f}")
    print(f"  mean probability after:  {summary['portfolio_mean_proba_after']:.4f}")
    print(f"  mean shift:              {summary['mean_delta']:+.4f}")
    print(f"  loans newly crossing 0.5 risk threshold: {summary['n_newly_high_risk_over_0.5']}")
    print(f"\n  top {args.top_n} most-affected loans:")
    for row in summary["top_movers"]:
        print(f"    {row['loan_id']}  ({row['region']}/{row['loan_type']})  {row['proba_before']:.3f} -> {row['proba_after']:.3f}  ({row['delta']:+.3f})")

    tag = f"{args.shock}_{args.magnitude}" + (f"_{args.region}" if args.region else "")
    plot_path = REPORT_DIR / f"scenario_{tag}.png"
    plot_shift(proba_before, proba_after, f"Scenario: {args.shock} shock ({args.magnitude:+g})", plot_path)

    result_path = OUTPUT_DIR / f"scenario_result_{tag}.csv"
    result_df.to_csv(result_path, index=False)
    summary_path = OUTPUT_DIR / f"scenario_summary_{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    AuditTrail().log("scenario_run", None, {
        "shock_type": args.shock, "magnitude": args.magnitude, "region": args.region,
        "portfolio_mean_proba_before": summary["portfolio_mean_proba_before"],
        "portfolio_mean_proba_after": summary["portfolio_mean_proba_after"],
    })

    print(f"\nwrote {plot_path}, {result_path}, {summary_path}")
    print("logged scenario run to the audit trail")


if __name__ == "__main__":
    main()
