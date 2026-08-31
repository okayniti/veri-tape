"""Deploy-time seeding for a fresh, empty container disk.

Render's free tier has no persistent disk: every cold start (the first
deploy, or waking up after the service spins down from 15 minutes of
inactivity) begins from an empty filesystem. Without this, a judge opening
the link right after a spin-down would land on a Portfolio Command screen
with nothing to aggregate and an Audit Trail with zero entries -- not broken,
just empty, but indistinguishable from broken to someone who didn't build it.

seed_if_needed() is called once from api/main.py's startup hook (see the
`lifespan` context manager there). It's cheap to call on every boot: if
outputs/ already has real pipeline artifacts and the audit trail already has
entries -- true for every local dev run, and every boot after the first in a
given container's lifetime -- it does nothing and returns immediately.

When it does have work to do, every step below is the *same* code
`python -m loan_intelligence.<module>` already runs locally -- this module
doesn't reimplement any modeling, feature, or scoring logic. The only choices
made here are (a) a smaller synthetic portfolio (SEED_N_LOANS) so a cold
boot finishes in seconds rather than minutes, and (b) which loans get an
audit-trail entry logged upfront via review/decisions.ensure_scored() --
the exact function a loan's first real view already uses -- instead of
waiting for someone to click into one.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

# Small enough to run data generation + XGBoost + the GRU+MLP autoencoder +
# Platt calibration end to end in seconds on a free-tier CPU, while still
# giving every Portfolio Command breakdown (4 regions x 4 loan types x 3 risk
# tiers) enough loans per bucket to be a real number instead of 0/NaN.
# Override via env var if a deploy needs to trade off differently.
SEED_N_LOANS = int(os.environ.get("VERITAPE_SEED_N_LOANS", "1000"))

# How many loans get a real prediction + anomaly-flag entry logged to the
# audit trail upfront, so /audit/verify and the Audit Trail visualization
# aren't empty on the very first request after a cold boot. Flagged loans
# are seeded first (they're the interesting ones to show), then padded with
# unflagged ones up to this count.
SEED_AUDIT_BATCH = int(os.environ.get("VERITAPE_SEED_AUDIT_BATCH", "20"))

_REQUIRED_ARTIFACTS = [
    MODEL_DIR / "xgboost_model.json",
    MODEL_DIR / "xgboost_feature_cols.joblib",
    MODEL_DIR / "calibrator.joblib",
    MODEL_DIR / "anomaly_gru_mlp.pt",
    MODEL_DIR / "anomaly_meta.json",
    MODEL_DIR / "anomaly_seq_scaler.joblib",
    MODEL_DIR / "anomaly_static_scaler.joblib",
    OUTPUT_DIR / "train.csv",
    OUTPUT_DIR / "test.csv",
    OUTPUT_DIR / "predictions_test_calibrated.csv",
    OUTPUT_DIR / "anomaly_scores_test.csv",
]


def _is_seeded() -> bool:
    """True once a full pipeline run -- local dev, or an earlier boot of
    this same container -- has already left real models and data on disk."""
    return all(p.exists() for p in _REQUIRED_ARTIFACTS)


def _audit_trail_is_empty() -> bool:
    from loan_intelligence.audit.hash_chain import AuditTrail

    return AuditTrail().verify()["n_entries"] == 0


def _run_pipeline() -> None:
    """Data generation through calibration, at SEED_N_LOANS instead of the
    full cfg.N_LOANS -- the same modules, entrypoints, and file paths every
    `python -m loan_intelligence.<module>` invocation already uses. SHAP
    explainers are deliberately not built here: api/main.py already lazily
    constructs them on first use (functools.lru_cache), and building them
    eagerly at boot would only add to a cold start's memory/time cost for
    something no request has asked for yet."""
    from loan_intelligence.data import generate_synthetic
    from loan_intelligence.features import build_features, time_split
    from loan_intelligence.models import anomaly, calibrate, predict

    generate_synthetic.generate(n_loans=SEED_N_LOANS)
    build_features.main()
    time_split.main()
    predict.main()
    anomaly.main()
    calibrate.main()


def _seed_audit_trail() -> None:
    """Logs real prediction + anomaly_flag entries for a handful of loans via
    review.decisions.ensure_scored() -- the identical function a loan's first
    GET /loans/{id} or POST .../review call already uses to score it live
    against the trained models. No separate scoring logic; just called for a
    batch upfront instead of one loan at a time, lazily, as visitors click."""
    from loan_intelligence.audit.hash_chain import AuditTrail
    from loan_intelligence.portfolio.summary import load_current_book
    from loan_intelligence.review.decisions import ensure_scored

    trail = AuditTrail()
    book = load_current_book()

    flagged = book.loc[book["is_flagged"], "loan_id"].tolist()
    unflagged = book.loc[~book["is_flagged"], "loan_id"].tolist()
    loan_ids = (flagged + unflagged)[:SEED_AUDIT_BATCH]

    for loan_id in loan_ids:
        ensure_scored(loan_id, trail)


def seed_if_needed() -> None:
    """Idempotent -- safe to call on every API startup. Does real work only
    on a genuinely empty container disk."""
    if _is_seeded() and not _audit_trail_is_empty():
        return

    started = time.monotonic()

    if not _is_seeded():
        print(f"[bootstrap] outputs/ is empty -- seeding a {SEED_N_LOANS}-loan portfolio (fresh container disk)...")
        _run_pipeline()
        print(f"[bootstrap] pipeline seeded in {time.monotonic() - started:.1f}s")

    if _audit_trail_is_empty():
        audit_started = time.monotonic()
        print(f"[bootstrap] audit trail is empty -- logging up to {SEED_AUDIT_BATCH} real prediction/anomaly entries...")
        _seed_audit_trail()
        print(f"[bootstrap] audit trail seeded in {time.monotonic() - audit_started:.1f}s")

    print(f"[bootstrap] done in {time.monotonic() - started:.1f}s total")


if __name__ == "__main__":
    seed_if_needed()
