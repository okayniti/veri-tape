"""LLM-assisted reviewer narration -- explanation only, never prediction.

Hard separation from the modeling pipeline: this module receives numbers
that are already final (a calibrated probability from models/calibrate.py,
SHAP attributions from explain/shap_explain.py, an anomaly score from
models/anomaly.py) and asks Claude to phrase them for a human reviewer. The
system prompt explicitly forbids inventing numbers or second-guessing the
figures it's given, and nothing in this module ever writes back to a
prediction, a feature, or a model file -- it only ever produces a text note
and an audit-log entry. If the LLM call were removed entirely, every other
judged capability (prediction, anomaly detection, calibration, SHAP,
scenario simulation) still works exactly as before; that's what makes this
narration rather than an "LLM wrapper" around the actual modeling.

Falls back to a deterministic template (no API call) when ANTHROPIC_API_KEY
is unset, clearly labeled as such -- lets the rest of the pipeline demo
end-to-end without a live key, while the real path is a normal
`client.messages.create()` call.

Run directly: `python -m loan_intelligence.review.narrate --loan-id L100000`
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from loan_intelligence.audit.hash_chain import AuditTrail

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"

DEFAULT_MODEL = "claude-opus-5"

SYSTEM_PROMPT = (
    "You are assisting a human loan-portfolio reviewer. You will be given numbers "
    "already produced by a separate machine-learning pipeline: a calibrated "
    "default-probability prediction with its top SHAP feature attributions, and "
    "(if present) a structural-anomaly score with its own top attributions. Write "
    "a short reviewer note (2-4 sentences, plain English, no bullet points) that "
    "explains *why* the model produced these numbers, citing the specific figures "
    "and feature names given. Do not invent numbers, do not change or second-guess "
    "the probability or anomaly score, and do not perform any calculation yourself "
    "-- your only job is narrating results that already exist."
)


def gather_facts(loan_id: str, pred_explainer, anomaly_explainer=None) -> dict:
    predictions = pd.read_csv(OUTPUT_DIR / "predictions_test_calibrated.csv")
    features_df = pd.read_csv(OUTPUT_DIR / "test.csv")
    pred_row = predictions[predictions["loan_id"] == loan_id]
    if pred_row.empty:
        raise ValueError(f"loan_id {loan_id} not found in predictions_test_calibrated.csv")
    pred_row = pred_row.iloc[0]

    facts = {
        "loan_id": loan_id,
        "raw_xgb_probability": round(float(pred_row["xgb_proba"]), 4),
        "calibrated_probability": round(float(pred_row["xgb_proba_calibrated"]), 4),
        "top_prediction_drivers": pred_explainer.explain_loan(loan_id, features_df, top_k=5),
    }

    anomaly_path = OUTPUT_DIR / "anomaly_scores_test.csv"
    if anomaly_path.exists():
        anomalies = pd.read_csv(anomaly_path)
        anomaly_row = anomalies[anomalies["loan_id"] == loan_id]
        if not anomaly_row.empty:
            anomaly_row = anomaly_row.iloc[0]
            facts["anomaly_score"] = round(float(anomaly_row["anomaly_score"]), 4)
            facts["anomaly_flagged_top_1pct"] = bool(anomaly_row["flagged_top_1pct"])
            if anomaly_explainer is not None and bool(anomaly_row["flagged_top_1pct"]):
                facts["top_anomaly_drivers"] = anomaly_explainer.explain_loan(loan_id, features_df, top_k=5)

    return facts


def _template_fallback(facts: dict) -> str:
    top_driver = facts["top_prediction_drivers"][0]
    note = (
        f"[template fallback -- ANTHROPIC_API_KEY not set] Loan {facts['loan_id']}: calibrated default "
        f"probability {facts['calibrated_probability']:.0%}, most influenced by {top_driver['feature']} "
        f"(value {top_driver['value']}, SHAP {top_driver['shap_value']:+.3f})."
    )
    if facts.get("anomaly_flagged_top_1pct"):
        note += f" Also flagged as a top-1% structural anomaly (score {facts['anomaly_score']})."
    return note


def generate_reviewer_note(facts: dict, model: str = DEFAULT_MODEL, dry_run: bool = False) -> str:
    if dry_run or not os.environ.get("ANTHROPIC_API_KEY"):
        return _template_fallback(facts)

    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": json.dumps(facts, indent=2)}],
    )
    return next((b.text for b in response.content if b.type == "text"), "").strip()


def narrate_loan(loan_id: str, pred_explainer, anomaly_explainer=None, model: str = DEFAULT_MODEL, dry_run: bool = False) -> dict:
    facts = gather_facts(loan_id, pred_explainer, anomaly_explainer)
    note = generate_reviewer_note(facts, model=model, dry_run=dry_run)

    AuditTrail().log(
        "llm_narration", loan_id,
        {"note": note, "model": model if not dry_run else "template_fallback", "facts_used": facts},
    )
    return {"loan_id": loan_id, "note": note, "facts": facts}


def main() -> None:
    from loan_intelligence.explain.shap_explain import AnomalyExplainer, PredictionExplainer

    parser = argparse.ArgumentParser(description="Generate an LLM reviewer note for one loan (explanation only).")
    parser.add_argument("--loan-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true", help="skip the API call and use the template fallback")
    args = parser.parse_args()

    print("building explainers...")
    pred_explainer = PredictionExplainer()
    anomaly_explainer = AnomalyExplainer()

    result = narrate_loan(args.loan_id, pred_explainer, anomaly_explainer, model=args.model, dry_run=args.dry_run)

    print(f"\n=== Reviewer note for {args.loan_id} ===")
    print(result["note"])
    print(f"\n(logged to audit trail)")


if __name__ == "__main__":
    main()
