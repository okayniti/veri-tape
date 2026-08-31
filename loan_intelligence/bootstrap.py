"""Deploy-time seeding for a fresh, empty container disk.

Render's free tier has no persistent disk: every cold start (the first
deploy, or waking up after the service spins down from 15 minutes of
inactivity) begins from an empty filesystem. Without this, a judge opening
the link right after a spin-down would land on a Portfolio Command screen
with nothing to aggregate and an Audit Trail with zero entries -- not broken,
just empty, but indistinguishable from broken to someone who didn't build it.

run_seed_with_timeout() is what api/main.py's `lifespan` startup hook
schedules (fire-and-forget, not awaited) rather than calling seed_if_needed()
directly. Two failure modes showed up the first time this ran on Render's
free tier (0.1 CPU), both worth naming:

1. A genuinely stuck or very slow step can hang indefinitely with no way to
   tell "stuck" from "just slow on 0.1 CPU" from the logs alone -- fixed by
   the stage-by-stage log lines in _run_pipeline() below, and by
   run_seed_with_timeout()'s hard ceiling (SEED_TIMEOUT_SECONDS): past that,
   seeding is declared failed even if the underlying thread is still
   grinding away (Python threads can't be force-killed; see that function's
   docstring).
2. An exception inside seed_if_needed() was being caught into
   seeding_error() correctly, but Python's stdout is fully block-buffered
   by default when it isn't attached to a terminal -- exactly Render's log
   setup -- so `print()` output, including a would-be traceback, can sit in
   an internal buffer and never reach the log tail at all. render.yaml now
   sets PYTHONUNBUFFERED=1 to fix this globally; every print() in this
   module also passes flush=True as a belt-and-suspenders measure so its
   own progress/error lines can't go missing even if that env var is ever
   unset for some reason.

is_ready() is the flag every data-dependent route (see api/main.py's
_require_seeded dependency) checks before touching outputs/ -- False until
seeding (or the no-op check that finds nothing to do, or a timeout/failure)
has resolved one way or another, so a request that lands mid-seed gets a
clear 503 instead of a crash or a confusingly empty response.

It's cheap to call on every boot: if outputs/ already has real pipeline
artifacts and the audit trail already has entries -- true for every local
dev run, and every boot after the first in a given container's lifetime --
it does nothing and flips ready almost instantly.

When it does have work to do, every step below is the *same* code
`python -m loan_intelligence.<module>` already runs locally -- this module
doesn't reimplement any modeling, feature, or scoring logic. The only
choices made here are (a) a much smaller synthetic portfolio
(SEED_LOAN_COUNT) so a cold boot finishes in seconds on a fractional-CPU
free tier rather than minutes -- the full 5,000-loan numbers already exist
locally and in the demo video; the deployed instance's only job is to prove
the pipeline runs live, not to match that scale -- and (b) which loans get
an audit-trail entry logged upfront via review/decisions.ensure_scored() --
the exact function a loan's first real view already uses -- instead of
waiting for someone to click into one.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

# Small by design: Render's free tier is 0.1 CPU / 512MB. 75-100 loans is
# enough for every Portfolio Command breakdown (4 regions x 4 loan types x 3
# risk tiers) to show a real number instead of 0/NaN, and finishes end to
# end in seconds rather than minutes. Override via env var if a deploy needs
# a different speed/realism tradeoff -- the real numbers live locally and in
# the demo video regardless.
SEED_LOAN_COUNT = int(os.environ.get("SEED_LOAN_COUNT", "100"))

# How many loans get a real prediction + anomaly-flag entry logged to the
# audit trail upfront, so /audit/verify and the Audit Trail visualization
# aren't empty on the very first request after a cold boot. Flagged loans
# are seeded first (they're the interesting ones to show), then padded with
# unflagged ones up to this count.
SEED_AUDIT_BATCH = int(os.environ.get("VERITAPE_SEED_AUDIT_BATCH", "20"))

# Hard ceiling on the whole seed attempt. Past this, run_seed_with_timeout()
# declares seeding failed rather than leaving every data route 503-ing
# forever with no way to distinguish "dead" from "slow" -- see that
# function's docstring for what this can and can't actually stop.
SEED_TIMEOUT_SECONDS = int(os.environ.get("SEED_TIMEOUT_SECONDS", "180"))

# Readiness state for the data-dependent routes to check. A plain bool +
# lock is enough here: exactly one background thread ever writes it at a
# time (see api/main.py's lifespan hook), route handlers only ever read it,
# and the GIL already makes a single bool read atomic -- the lock just makes
# the read/write of _ready and _error together non-torn, not because of any
# real contention risk.
_state_lock = threading.Lock()
_ready = False
_error: str | None = None


def is_ready() -> bool:
    """False until seeding has actually resolved -- finished, determined
    there was nothing to do, timed out, or failed -- checked by every
    data-dependent route."""
    with _state_lock:
        return _ready


def seeding_error() -> str | None:
    """None unless the seed attempt failed or timed out; surfaced so a 503
    can say more than just "still seeding" once it's actually never going to
    finish."""
    with _state_lock:
        return _error


def _mark_ready(error: str | None = None) -> None:
    global _ready, _error
    with _state_lock:
        _ready = True
        _error = error


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


def _log(msg: str) -> None:
    """print() with flush=True -- see the module docstring on why an
    unflushed print can silently never reach Render's log tail."""
    print(f"[bootstrap] {msg}", flush=True)


def _is_seeded() -> bool:
    """True once a full pipeline run -- local dev, or an earlier boot of
    this same container -- has already left real models and data on disk."""
    return all(p.exists() for p in _REQUIRED_ARTIFACTS)


def _audit_trail_is_empty() -> bool:
    from loan_intelligence.audit.hash_chain import AuditTrail

    return AuditTrail().verify()["n_entries"] == 0


def _run_pipeline() -> None:
    """Data generation through calibration, at SEED_LOAN_COUNT instead of
    the full cfg.N_LOANS -- the same modules, entrypoints, and file paths
    every `python -m loan_intelligence.<module>` invocation already uses.
    One log line before and after each stage: not for verbosity, but so a
    log tail alone shows exactly which stage is running (or which one it
    died mid-way through), instead of one line at the start and silence
    until either the next line or nothing.

    SHAP explainers are deliberately not built here: api/main.py already
    lazily constructs them on first use (functools.lru_cache), and building
    them eagerly at boot would only add to a cold start's memory/time cost
    for something no request has asked for yet."""
    from loan_intelligence.data import generate_synthetic
    from loan_intelligence.features import build_features, time_split
    from loan_intelligence.models import anomaly, calibrate, predict

    _log(f"stage 1/6: data generation ({SEED_LOAN_COUNT} loans)...")
    generate_synthetic.generate(n_loans=SEED_LOAN_COUNT)
    _log("stage 1/6: data generation done")

    _log("stage 2/6: feature engineering (includes cleaning the raw loan tape)...")
    build_features.main()
    _log("stage 2/6: feature engineering done")

    _log("stage 3/6: time-aware train/test split...")
    time_split.main()
    _log("stage 3/6: split done")

    _log("stage 4/6: prediction training (XGBoost + Bi-LSTM benchmark)...")
    predict.main()
    _log("stage 4/6: prediction training done")

    _log("stage 5/6: anomaly model training (GRU+MLP autoencoder)...")
    anomaly.main()
    _log("stage 5/6: anomaly model training done")

    _log("stage 6/6: calibration...")
    calibrate.main()
    _log("stage 6/6: calibration done")


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
    """Idempotent, blocking -- safe to call directly (CLI, tests) or from a
    background thread (run_seed_with_timeout() does the latter). Does real
    work only on a genuinely empty container disk; always flips is_ready()
    to True on the way out, success or failure, so a route waiting on it
    doesn't wait forever for a seed attempt that's already dead."""
    try:
        if _is_seeded() and not _audit_trail_is_empty():
            _mark_ready()
            return

        started = time.monotonic()

        if not _is_seeded():
            _log(f"outputs/ is empty -- seeding a {SEED_LOAN_COUNT}-loan portfolio (fresh container disk)...")
            _run_pipeline()
            _log(f"pipeline seeded in {time.monotonic() - started:.1f}s")

        if _audit_trail_is_empty():
            audit_started = time.monotonic()
            _log(f"audit trail is empty -- logging up to {SEED_AUDIT_BATCH} real prediction/anomaly entries...")
            _seed_audit_trail()
            _log(f"audit trail seeded in {time.monotonic() - audit_started:.1f}s")

        _log(f"done in {time.monotonic() - started:.1f}s total")
        _mark_ready()
    except Exception as e:
        # Loud on purpose: a bare "seeding failed: {e}" plus a traceback
        # that only ever reached stderr was effectively invisible on
        # Render's free tier the first time this shipped (see the module
        # docstring on stdout buffering). Printed via _log() -> stdout,
        # flushed, so it lands in the log tail immediately rather than
        # being inferred five minutes later from silence.
        _log(f"SEEDING FAILED: {e}")
        _log(traceback.format_exc())
        _mark_ready(error=str(e))


async def run_seed_with_timeout() -> None:
    """Runs seed_if_needed() on a background thread (so it never blocks
    Uvicorn's own startup -- see api/main.py's lifespan hook), with a hard
    ceiling of SEED_TIMEOUT_SECONDS (default 180s / 3 minutes).

    Important nuance: asyncio.wait_for's timeout cancels *waiting* on the
    background thread, not the thread itself. Python threads cannot be
    force-killed, so if a pipeline step is genuinely wedged, it keeps
    running in the background (harmlessly, just wasted CPU) even after this
    function gives up and reports a timeout. That's an intentional
    tradeoff, not an oversight: the goal is an app that stops looking hung
    to the outside world (every data route starts returning a clear
    "seeding failed" 503 instead of "still seeding" forever), not a hard
    kill of arbitrary CPU-bound code -- which would need a subprocess, not a
    thread, and is more machinery than this deploy-time seed step warrants.
    If the orphaned thread does eventually finish, it will call
    seed_if_needed()'s own _mark_ready() and can still flip the app into a
    working state after the fact."""
    try:
        await asyncio.wait_for(asyncio.to_thread(seed_if_needed), timeout=SEED_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        msg = (
            f"seeding did not finish within {SEED_TIMEOUT_SECONDS}s -- treating as failed. "
            "The pipeline stage it was on (see the last 'stage N/6' log line above) may still "
            "be running in the background; if it finishes late, the app can still recover."
        )
        _log(msg)
        _mark_ready(error=msg)


if __name__ == "__main__":
    seed_if_needed()
