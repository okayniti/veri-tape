"""Portfolio Command summary: portfolio-level aggregates, not another
single-loan drilldown. This is the screen a manager opens every morning.

"Portfolio" here means the current book under active servicing -- the
time-aware test split (loans originated after the training cutoff; see
features/time_split.py) -- because that's exactly what
predictions_test_calibrated.csv and anomaly_scores_test.csv already cover.
The train split is historical data the models were fit on, not loans
anyone is actively servicing today. Nothing here scores a new loan,
retrains anything, or generates data -- it's pure aggregation over
already-existing outputs/*.csv files and the audit trail.

Run directly: `python -m loan_intelligence.portfolio.summary`
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from loan_intelligence import config as cfg
from loan_intelligence.audit.hash_chain import AuditTrail

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"


def load_current_book() -> pd.DataFrame:
    """The test split (the current book), joined with calibrated
    predictions and anomaly scores. A loan is "flagged" under the same
    rule review/decisions.py uses: high risk tier, or a top-1% anomaly."""
    test_df = pd.read_csv(OUTPUT_DIR / "test.csv")
    predictions = pd.read_csv(OUTPUT_DIR / "predictions_test_calibrated.csv")
    anomalies = pd.read_csv(OUTPUT_DIR / "anomaly_scores_test.csv")

    df = test_df.merge(predictions[["loan_id", "xgb_proba", "xgb_proba_calibrated"]], on="loan_id", how="left")
    df = df.merge(anomalies[["loan_id", "anomaly_score", "flagged_top_1pct"]], on="loan_id", how="left")
    df["flagged_top_1pct"] = df["flagged_top_1pct"].fillna(False)
    df["risk_tier"] = df["xgb_proba_calibrated"].apply(cfg.risk_tier)
    df["is_flagged"] = df["flagged_top_1pct"] | (df["risk_tier"] == "high")
    return df


def _risk_tier_breakdown(df: pd.DataFrame) -> list[dict]:
    counts = df["risk_tier"].value_counts()
    total = len(df)
    return [
        {"tier": tier, "count": int(counts.get(tier, 0)), "pct_of_portfolio": round(100 * counts.get(tier, 0) / total, 2)}
        for tier in ("low", "medium", "high")
    ]


def _anomaly_breakdown(df: pd.DataFrame) -> dict:
    def _grouped(col: str) -> dict:
        g = df.groupby(col)["flagged_top_1pct"].agg(count="sum", rate="mean")
        return {k: {"count": int(v["count"]), "rate": round(float(v["rate"]), 4)} for k, v in g.iterrows()}

    return {
        "overall_rate": round(float(df["flagged_top_1pct"].mean()), 4),
        "overall_count": int(df["flagged_top_1pct"].sum()),
        "by_region": _grouped("region"),
        "by_loan_type": _grouped("loan_type"),
    }


def _reviewer_override_rate(trail: AuditTrail) -> dict:
    all_entries = trail.export_all()
    decisions = all_entries[all_entries["event_type"] == "reviewer_decision"]
    if decisions.empty:
        return {"n_reviews": 0, "n_overrides": 0, "override_rate": None}
    decisions_made = decisions["payload"].apply(lambda p: json.loads(p).get("decision"))
    n_overrides = int((decisions_made == "override").sum())
    return {
        "n_reviews": int(len(decisions)),
        "n_overrides": n_overrides,
        "override_rate": round(n_overrides / len(decisions), 4),
    }


def _latest_scenario_comparison(current_book: pd.DataFrame) -> dict | None:
    """Expected loss before/after the most recently run scenario
    (scenario/simulate.py), restricted to loans that are also in the
    current book, so the comparison uses the same portfolio definition as
    everything else in this summary."""
    scenario_files = sorted(OUTPUT_DIR.glob("scenario_result_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not scenario_files:
        return None

    latest = scenario_files[0]
    scenario_df = pd.read_csv(latest)
    merged = current_book[["loan_id", "loan_amount"]].merge(
        scenario_df[["loan_id", "proba_before", "proba_after"]], on="loan_id", how="inner"
    )
    if merged.empty:
        return None

    baseline_el = float((merged["proba_before"] * merged["loan_amount"]).sum())
    shocked_el = float((merged["proba_after"] * merged["loan_amount"]).sum())
    return {
        "scenario_file": latest.name,
        "n_loans_in_current_book": int(len(merged)),
        "baseline_expected_loss": round(baseline_el, 2),
        "shocked_expected_loss": round(shocked_el, 2),
        "delta": round(shocked_el - baseline_el, 2),
    }


def build_summary() -> dict:
    df = load_current_book()
    trail = AuditTrail()

    return {
        "n_loans": int(len(df)),
        "total_portfolio_expected_loss": round(float((df["xgb_proba_calibrated"] * df["loan_amount"]).sum()), 2),
        "average_calibrated_probability": round(float(df["xgb_proba_calibrated"].mean()), 4),
        "risk_tier_breakdown": _risk_tier_breakdown(df),
        "flagged_loan_count": int(df["is_flagged"].sum()),
        "flagged_loan_pct": round(100 * float(df["is_flagged"].mean()), 2),
        "anomaly": _anomaly_breakdown(df),
        "reviewer": _reviewer_override_rate(trail),
        "scenario_comparison": _latest_scenario_comparison(df),
    }


def main() -> None:
    summary = build_summary()
    print(json.dumps(summary, indent=2))

    out_path = OUTPUT_DIR / "portfolio_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
