"""Reviewer decision loop: a human accepts or overrides an automated flag.

This is what turns "the model said 0.34" into an auditable workflow. A loan
is "flagged" when either automated system says so -- the anomaly detector
(models/anomaly.py) marking it top-1% unusual, or the prediction model
(models/predict.py) putting its calibrated probability in the "high" risk
tier (config.risk_tier) -- and a human reviewer can then log an "accept"
(confirms the flag) or "override" (disagrees, with a reason) decision.

Two things make this an audit trail rather than a UI convenience:

1. Every decision -- the automated flag *and* the human response -- is its
   own hash-chained entry (audit/hash_chain.py). An override's entry sets
   `references_hash` to the exact entry_hash of the flag it responds to, so
   the link between "what the model said" and "what the human decided" is
   itself tamper-evident, not just implied by matching loan_id.
2. An override never touches the stored prediction or anomaly score. It's
   an annotation on an immutable model output, not a correction to it --
   the same non-mutation principle review/narrate.py follows for LLM
   narration. What the model said and what the reviewer decided both stay
   on the record, forever, as two separate entries.

If a loan has no flag entry yet in the audit trail (nobody has looked at it
via demo.py, the API, or the CLI here), submit_decision() scores it live
first -- using the same loaded model artifacts as demo.py -- so a flag
entry exists before an override can reference it. Reviewing a loan with no
qualifying flag (calibrated probability below "high" and no anomaly flag)
is rejected: there is nothing to accept or override.

Run directly: `python -m loan_intelligence.review.decisions --loan-id L100000 --decision override --reason "..."`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from loan_intelligence import config as cfg
from loan_intelligence.audit.hash_chain import AuditTrail
from loan_intelligence.models.anomaly import score_anomalies
from loan_intelligence.models.predict import _as_xgb_frame

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"

VALID_DECISIONS = ("accept", "override")


def _load_loan_row(loan_id: str) -> pd.DataFrame:
    test_df = pd.read_csv(OUTPUT_DIR / "test.csv")
    row = test_df[test_df["loan_id"] == loan_id]
    if row.empty:
        train_df = pd.read_csv(OUTPUT_DIR / "train.csv")
        row = train_df[train_df["loan_id"] == loan_id]
    if row.empty:
        raise ValueError(f"loan_id {loan_id!r} not found in train.csv or test.csv")
    return row


def ensure_scored(loan_id: str, trail: AuditTrail) -> None:
    """Guarantees a "prediction" and an "anomaly_flag" entry exist for this
    loan in the audit trail, scoring live (via the already-trained model
    artifacts -- no retraining) if either is missing."""
    from demo import load_anomaly_stack, load_prediction_stack  # repo-root entrypoint, importable by design (see app.py)

    history = trail.history_for_loan(loan_id)
    has_prediction = (not history.empty) and (history["event_type"] == "prediction").any()
    has_anomaly = (not history.empty) and (history["event_type"] == "anomaly_flag").any()
    if has_prediction and has_anomaly:
        return

    row = _load_loan_row(loan_id)

    if not has_prediction:
        model, feature_cols, calibrator = load_prediction_stack()
        raw = float(model.predict_proba(_as_xgb_frame(row, feature_cols))[:, 1][0])
        calibrated = float(calibrator.predict(np.array([raw]))[0])
        trail.log("prediction", loan_id, {"xgb_proba_raw": raw, "xgb_proba_calibrated": calibrated})

    if not has_anomaly:
        autoencoder, seq_scaler, static_scaler = load_anomaly_stack()
        score = float(score_anomalies(autoencoder, seq_scaler, static_scaler, row)[0])
        all_scores = pd.read_csv(OUTPUT_DIR / "anomaly_scores_test.csv")
        threshold = float(np.quantile(all_scores["anomaly_score"], 0.99))
        trail.log("anomaly_flag", loan_id, {"anomaly_score": score, "flagged_top_1pct": bool(score >= threshold)})


def get_current_flags(loan_id: str, trail: AuditTrail) -> dict:
    """Returns {"anomaly": entry|None, "prediction": entry|None} -- the most
    recent audit entry of each type that currently qualifies as a flag. A
    non-qualifying entry (e.g. a "prediction" that scored "low" risk) does
    not count, even though the audit trail still has a record of it."""
    result: dict = {"anomaly": None, "prediction": None}
    history = trail.history_for_loan(loan_id)
    for _, row in history.iterrows():  # ascending id order -- last match wins, i.e. most recent
        entry = row.to_dict()
        payload = json.loads(entry["payload"])
        if entry["event_type"] == "anomaly_flag" and payload.get("flagged_top_1pct"):
            result["anomaly"] = entry
        elif entry["event_type"] == "prediction" and cfg.risk_tier(payload.get("xgb_proba_calibrated", 0.0)) == "high":
            result["prediction"] = entry
    return result


def submit_decision(loan_id: str, decision: str, reason: str | None = None, trail: AuditTrail | None = None) -> dict:
    if decision not in VALID_DECISIONS:
        raise ValueError(f"decision must be one of {VALID_DECISIONS}, got {decision!r}")

    trail = trail or AuditTrail()
    ensure_scored(loan_id, trail)

    flags = get_current_flags(loan_id, trail)
    # anomaly flag takes priority as the primary reference when a loan has both
    ordered = [f for f in (flags["anomaly"], flags["prediction"]) if f is not None]
    if not ordered:
        raise ValueError(f"loan_id {loan_id!r} is not currently flagged (risk tier below 'high' and no anomaly flag) -- nothing to review")

    primary_flag = ordered[0]
    payload = {
        "decision": decision,
        "reason": reason or "",
        "flag_event_type": primary_flag["event_type"],
        "flag_entry_id": int(primary_flag["id"]),
        "all_flag_hashes_addressed": [f["entry_hash"] for f in ordered],
    }
    entry_hash = trail.log("reviewer_decision", loan_id, payload, references_hash=primary_flag["entry_hash"])

    return {
        "loan_id": loan_id, "decision": decision, "reason": reason or "",
        "flag_event_type": primary_flag["event_type"], "references_hash": primary_flag["entry_hash"],
        "entry_hash": entry_hash,
    }


def review_history(loan_id: str, trail: AuditTrail | None = None) -> pd.DataFrame:
    """All reviewer_decision entries for a loan, oldest first."""
    trail = trail or AuditTrail()
    history = trail.history_for_loan(loan_id)
    return history[history["event_type"] == "reviewer_decision"].reset_index(drop=True)


def main() -> None:
    import sys

    parser = argparse.ArgumentParser(description="Submit a human reviewer decision on a flagged loan.")
    parser.add_argument("--loan-id", required=True)
    parser.add_argument("--decision", required=True, choices=VALID_DECISIONS)
    parser.add_argument("--reason", default=None)
    args = parser.parse_args()

    try:
        result = submit_decision(args.loan_id, args.decision, args.reason)
    except ValueError as e:
        print(f"error: {e}")
        sys.exit(1)

    print(f"\n=== Reviewer decision recorded for {result['loan_id']} ===")
    print(f"  decision: {result['decision']}")
    print(f"  reason: {result['reason'] or '(none given)'}")
    print(f"  responds to: {result['flag_event_type']} entry, hash {result['references_hash']}")
    print(f"  new entry hash: {result['entry_hash']}")

    trail = AuditTrail()
    print(f"  chain verification: {trail.verify()}")


if __name__ == "__main__":
    main()
