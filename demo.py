"""End-to-end demo: run one loan record through the full pipeline and print
prediction, calibrated probability, anomaly flag, SHAP explanation,
reviewer note, and audit hash -- the walkthrough for a screen recording.

Everything here is *live*: the XGBoost model, calibrator, and GRU+MLP
autoencoder are loaded and re-run on the chosen record rather than reading
back a precomputed CSV, so this is genuinely "the pipeline, on one loan",
not a lookup table dressed up as one.

Run: `python demo.py` (auto-picks an interesting record) or
     `python demo.py --loan-id L100000` (pick one yourself)
     `python demo.py --dry-run` (skip the live Gemini API call)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from loan_intelligence.audit.hash_chain import AuditTrail
from loan_intelligence.explain.shap_explain import AnomalyExplainer, PredictionExplainer
from loan_intelligence.models.anomaly import STATIC_COLS as ANOMALY_STATIC_COLS
from loan_intelligence.models.anomaly import GRUMLPAutoencoder, score_anomalies
from loan_intelligence.models.predict import _as_xgb_frame
from loan_intelligence.review.narrate import DEFAULT_MODEL, _has_gemini_key, generate_reviewer_note

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"


def load_prediction_stack():
    feature_cols = joblib.load(MODEL_DIR / "xgboost_feature_cols.joblib")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / "xgboost_model.json")
    calibrator = joblib.load(MODEL_DIR / "calibrator.joblib")
    return model, feature_cols, calibrator


def load_anomaly_stack():
    import json

    meta = json.load(open(MODEL_DIR / "anomaly_meta.json"))
    autoencoder = GRUMLPAutoencoder(n_seq_channels=meta["n_seq_channels"], n_static=len(ANOMALY_STATIC_COLS), seq_len=meta["seq_len"])
    autoencoder.load_state_dict(torch.load(MODEL_DIR / "anomaly_gru_mlp.pt"))
    autoencoder.eval()
    seq_scaler = joblib.load(MODEL_DIR / "anomaly_seq_scaler.joblib")
    static_scaler = joblib.load(MODEL_DIR / "anomaly_static_scaler.joblib")
    return autoencoder, seq_scaler, static_scaler


def pick_interesting_loan_id() -> str:
    anomalies = pd.read_csv(OUTPUT_DIR / "anomaly_scores_test.csv")
    flagged = anomalies[anomalies["flagged_top_1pct"]]
    if len(flagged):
        return flagged.sort_values("anomaly_score", ascending=False)["loan_id"].iloc[0]
    predictions = pd.read_csv(OUTPUT_DIR / "predictions_test_calibrated.csv")
    return predictions.sort_values("xgb_proba_calibrated", ascending=False)["loan_id"].iloc[0]


def _print_drivers(title: str, drivers: list[dict]) -> None:
    print(f"  {title}:")
    for d in drivers:
        print(f"    {d['feature']:<35} value={d['value']!s:<12} shap={d['shap_value']:+.4f}")


def run_demo(loan_id: str, dry_run: bool = False, model_name: str = DEFAULT_MODEL) -> dict:
    test_df = pd.read_csv(OUTPUT_DIR / "test.csv")
    row = test_df[test_df["loan_id"] == loan_id]
    if row.empty:
        train_df = pd.read_csv(OUTPUT_DIR / "train.csv")
        row = train_df[train_df["loan_id"] == loan_id]
    if row.empty:
        raise ValueError(f"loan_id {loan_id!r} not found in train.csv or test.csv")

    trail = AuditTrail()

    print(f"\n{'=' * 70}\nLOAN INTELLIGENCE -- end-to-end demo for {loan_id}\n{'=' * 70}")

    print("\n[1/5] Prediction (XGBoost, live)")
    model, feature_cols, calibrator = load_prediction_stack()
    raw_proba = float(model.predict_proba(_as_xgb_frame(row, feature_cols))[:, 1][0])
    calibrated_proba = float(calibrator.predict(np.array([raw_proba]))[0])
    print(f"  raw probability:        {raw_proba:.4f}")
    print(f"  calibrated probability: {calibrated_proba:.4f}")
    trail.log("prediction", loan_id, {"xgb_proba_raw": raw_proba, "xgb_proba_calibrated": calibrated_proba})

    print("\n[2/5] Structural anomaly check (GRU+MLP autoencoder, live)")
    autoencoder, seq_scaler, static_scaler = load_anomaly_stack()
    anomaly_score = float(score_anomalies(autoencoder, seq_scaler, static_scaler, row)[0])
    all_test_scores = score_anomalies(autoencoder, seq_scaler, static_scaler, test_df)
    threshold = float(np.quantile(all_test_scores, 0.99))
    flagged = anomaly_score >= threshold
    print(f"  anomaly score: {anomaly_score:.4f}  (top-1% threshold: {threshold:.4f})")
    print(f"  flagged: {'YES -- top 1% most structurally unusual' if flagged else 'no'}")
    trail.log("anomaly_flag", loan_id, {"anomaly_score": anomaly_score, "flagged_top_1pct": flagged})

    print("\n[3/5] SHAP explanation")
    pred_explainer = PredictionExplainer()
    top_pred = pred_explainer.explain_loan(loan_id, test_df, top_k=5)
    _print_drivers("top prediction drivers", top_pred)

    top_anomaly = None
    if flagged:
        anomaly_explainer = AnomalyExplainer()
        top_anomaly = anomaly_explainer.explain_loan(loan_id, test_df, top_k=5)
        _print_drivers("top anomaly drivers", top_anomaly)

    print("\n[4/5] Reviewer narration (LLM, explanation only)")
    facts = {
        "loan_id": loan_id,
        "raw_xgb_probability": round(raw_proba, 4),
        "calibrated_probability": round(calibrated_proba, 4),
        "top_prediction_drivers": top_pred,
        "anomaly_score": round(anomaly_score, 4),
        "anomaly_flagged_top_1pct": flagged,
    }
    if top_anomaly:
        facts["top_anomaly_drivers"] = top_anomaly
    note = generate_reviewer_note(facts, model=model_name, dry_run=dry_run)
    print(f"  {note}")
    used_live_model = model_name if (not dry_run and _has_gemini_key()) else "template_fallback"
    trail.log("llm_narration", loan_id, {"note": note, "model": used_live_model})

    print("\n[5/5] Audit trail")
    verification = trail.verify()
    latest_hash = trail.history_for_loan(loan_id)["entry_hash"].iloc[-1]
    print(f"  latest entry hash for {loan_id}: {latest_hash}")
    print(f"  full chain verification: {verification}")
    print(f"  audit DB: {trail.db_path}")

    return {
        "loan_id": loan_id, "raw_proba": raw_proba, "calibrated_proba": calibrated_proba,
        "anomaly_score": anomaly_score, "flagged": flagged, "note": note,
        "latest_audit_hash": latest_hash, "chain_valid": verification["valid"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one loan through the full loan-intelligence pipeline.")
    parser.add_argument("--loan-id", default=None, help="defaults to an auto-picked interesting record")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true", help="skip the live Gemini API call")
    args = parser.parse_args()

    loan_id = args.loan_id or pick_interesting_loan_id()
    if not args.loan_id:
        print(f"no --loan-id given; auto-selected {loan_id}")

    run_demo(loan_id, dry_run=args.dry_run, model_name=args.model)


if __name__ == "__main__":
    main()
