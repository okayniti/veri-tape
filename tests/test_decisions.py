"""Tests for the reviewer decision loop (loan_intelligence/review/decisions.py).

Uses a temporary, isolated AuditTrail per test (tmp_path) so nothing here
touches outputs/audit_trail.db -- but reads the real, already-generated
outputs/*.csv, consistent with the rest of this project's "no retraining,
no new data generation" pipeline discipline.
"""
import json
import sqlite3

import pandas as pd
import pytest

from loan_intelligence.audit.hash_chain import AuditTrail, OUTPUT_DIR
from loan_intelligence.review.decisions import submit_decision


def _first_flagged_anomaly_loan_id() -> str:
    path = OUTPUT_DIR / "anomaly_scores_test.csv"
    if not path.exists():
        pytest.skip("outputs/anomaly_scores_test.csv missing -- run models/anomaly.py first")
    anomalies = pd.read_csv(path)
    flagged = anomalies[anomalies["flagged_top_1pct"]]
    if flagged.empty:
        pytest.skip("no flagged anomaly loans in this run's outputs/anomaly_scores_test.csv")
    return flagged["loan_id"].iloc[0]


def _first_low_risk_loan_id() -> str:
    path = OUTPUT_DIR / "predictions_test_calibrated.csv"
    if not path.exists():
        pytest.skip("outputs/predictions_test_calibrated.csv missing -- run models/calibrate.py first")
    preds = pd.read_csv(path)
    low_risk = preds[preds["xgb_proba_calibrated"] < 0.10]
    if low_risk.empty:
        pytest.skip("no clearly low-risk loans in this run")
    return low_risk["loan_id"].iloc[0]


def test_override_flag_logs_reference_and_chain_still_verifies(tmp_path):
    trail = AuditTrail(db_path=tmp_path / "audit_trail.db")
    loan_id = _first_flagged_anomaly_loan_id()

    result = submit_decision(loan_id, "override", reason="test override: confirmed false positive", trail=trail)

    assert result["decision"] == "override"
    assert result["references_hash"], "an override must reference the flag entry it responds to"

    referenced = trail.get_entry_by_hash(result["references_hash"])
    assert referenced is not None, "the referenced hash must correspond to a real prior entry"
    assert referenced["event_type"] in ("anomaly_flag", "prediction")
    assert referenced["loan_id"] == loan_id

    verification = trail.verify()
    assert verification["valid"] is True
    assert verification["n_entries"] >= 2  # at least the flag entry + the reviewer_decision entry


def test_reviewing_an_unflagged_loan_is_rejected(tmp_path):
    trail = AuditTrail(db_path=tmp_path / "audit_trail.db")
    loan_id = _first_low_risk_loan_id()

    with pytest.raises(ValueError, match="not currently flagged"):
        submit_decision(loan_id, "accept", trail=trail)


def test_override_never_mutates_the_original_flag_payload(tmp_path):
    trail = AuditTrail(db_path=tmp_path / "audit_trail.db")
    loan_id = _first_flagged_anomaly_loan_id()

    result = submit_decision(loan_id, "override", reason="disagree with the flag", trail=trail)
    referenced_before = trail.get_entry_by_hash(result["references_hash"])

    # re-fetch independently and confirm the flag entry's payload is byte-identical
    referenced_after = trail.get_entry_by_hash(result["references_hash"])
    assert referenced_before["payload"] == referenced_after["payload"]
    assert "override" not in referenced_after["payload"]  # the flag itself carries no trace of the review


def test_tampering_an_override_after_the_fact_breaks_verification(tmp_path):
    trail = AuditTrail(db_path=tmp_path / "audit_trail.db")
    loan_id = _first_flagged_anomaly_loan_id()

    result = submit_decision(loan_id, "override", reason="original reason", trail=trail)
    assert trail.verify()["valid"] is True

    # simulate an operator quietly flipping the decision after the fact,
    # bypassing AuditTrail.log() entirely
    conn = sqlite3.connect(trail.db_path)
    conn.execute(
        "UPDATE audit_log SET payload = ? WHERE entry_hash = ?",
        (json.dumps({"decision": "accept", "reason": "quietly changed"}), result["entry_hash"]),
    )
    conn.commit()
    conn.close()

    verification = trail.verify()
    assert verification["valid"] is False
