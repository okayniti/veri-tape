"""Probability calibration for the XGBoost default predictor.

An AUC of 0.78 says the model *ranks* loans correctly; it says nothing
about whether "0.34" means "34% chance of default" -- gradient-boosted
trees are notoriously overconfident at the tails. This module fits a
calibrator (Platt / sigmoid, or isotonic regression) and reports Brier
score before/after plus a reliability diagram, because a reviewer-facing
probability that isn't actually a probability is worse than not showing one.

Default method is Platt (sigmoid), not isotonic, despite isotonic scoring a
touch better on this run's Brier score (0.044 vs 0.046) -- the calibration
holdout is only ~600 loans with ~40 positives, and isotonic regression's
step function ends up with ~19 plateaus at that sample size. A plateau
boundary sitting between "before" and "after" a small scenario shock (see
scenario/simulate.py) makes a probability jump by 0.3+ on a marginal input
change, which is a calibration-estimation artifact, not a real risk signal.
Platt's smooth sigmoid degrades that failure mode into a small, monotonic
shift instead.

Leakage discipline: the calibration mapping is fit on a holdout carved out
of the *train* split (via the same time-aware split used everywhere else),
using a freshly-trained XGBoost that never saw that holdout -- never on the
test set. It is then applied to the production model's (predict.py, trained
on the full train split) test-set probabilities for the before/after report.

Run directly: `python -m loan_intelligence.models.calibrate`
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from loan_intelligence.features.time_split import time_aware_split
from loan_intelligence.models.calibrators import IsotonicCalibrator, PlattCalibrator
from loan_intelligence.models.predict import get_feature_cols, predict_xgboost, train_xgboost

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"


def fit_calibrator(train_df: pd.DataFrame, method: str = "platt"):
    calib_fit_df, calib_holdout_df, _ = time_aware_split(train_df, test_size=0.15)
    feature_cols = get_feature_cols(calib_fit_df)

    calib_source_model = train_xgboost(calib_fit_df, feature_cols)
    raw_holdout_proba = predict_xgboost(calib_source_model, calib_holdout_df, feature_cols)
    y_holdout = calib_holdout_df["default_flag"].to_numpy()

    calibrator = PlattCalibrator() if method == "platt" else IsotonicCalibrator()
    calibrator.fit(raw_holdout_proba, y_holdout)
    return calibrator


def plot_reliability_diagram(y_true: np.ndarray, raw_proba: np.ndarray, calibrated_proba: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="perfectly calibrated")

    for label, proba, color in [("before calibration", raw_proba, "#d9534f"), ("after calibration", calibrated_proba, "#5cb85c")]:
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", color=color, label=label)

    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed default rate")
    ax.set_title("Reliability diagram — default probability calibration")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    train_df = pd.read_csv(OUTPUT_DIR / "train.csv")
    test_df = pd.read_csv(OUTPUT_DIR / "test.csv")
    predictions_test = pd.read_csv(OUTPUT_DIR / "predictions_test.csv")

    y_test = test_df["default_flag"].to_numpy()
    raw_proba_test = predictions_test["xgb_proba"].to_numpy()

    print("fitting calibrator on a train-only holdout (never touches test)...")
    calibrator = fit_calibrator(train_df, method="platt")
    calibrated_proba_test = calibrator.predict(raw_proba_test)

    brier_before = brier_score_loss(y_test, raw_proba_test)
    brier_after = brier_score_loss(y_test, calibrated_proba_test)

    print("\n=== Calibration report (held-out test set) ===")
    print(f"  Brier score before calibration: {brier_before:.4f}")
    print(f"  Brier score after calibration:  {brier_after:.4f}")
    improvement = 100 * (brier_before - brier_after) / brier_before
    print(f"  improvement: {improvement:+.1f}%")

    diagram_path = REPORT_DIR / "reliability_diagram.png"
    plot_reliability_diagram(y_test, raw_proba_test, calibrated_proba_test, diagram_path)
    print(f"wrote {diagram_path}")

    out = predictions_test.copy()
    out["xgb_proba_calibrated"] = calibrated_proba_test
    out_path = OUTPUT_DIR / "predictions_test_calibrated.csv"
    out.to_csv(out_path, index=False)

    report = {
        "method": "platt",
        "brier_before": round(float(brier_before), 4),
        "brier_after": round(float(brier_after), 4),
        "improvement_pct": round(float(improvement), 2),
    }
    report_path = OUTPUT_DIR / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    import joblib
    joblib.dump(calibrator, OUTPUT_DIR / "models" / "calibrator.joblib")

    print(f"wrote {out_path}, {report_path}")


if __name__ == "__main__":
    main()
