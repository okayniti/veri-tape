"""FastAPI layer exposing the pipeline to a frontend.

Every route is a thin wrapper over an already-built module -- no modeling
logic lives here. Expensive objects (loaded models, the SHAP explainers)
are built once, lazily, on first use (functools.lru_cache), not per
request or at import time.

Run: `python -m uvicorn loan_intelligence.api.main:app --reload --port 8000`
(use `python -m uvicorn`, not a bare `uvicorn` command, so the current
working directory -- and with it, the repo-root `demo` module this file
and review/decisions.py both import from -- is on sys.path, the same way
every `python -m loan_intelligence...` command in this repo relies on it.)

Interactive API reference: http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from loan_intelligence.api.schemas import (
    AuditEntry,
    AuditVerifyResponse,
    HealthResponse,
    HTTPError,
    LoanDetailResponse,
    LoanListResponse,
    PortfolioSummaryResponse,
    ReviewResponse,
    ScenarioRunResponse,
)
from loan_intelligence.audit.hash_chain import AuditTrail
from loan_intelligence.portfolio.summary import build_summary, load_current_book
from loan_intelligence.review.decisions import ensure_scored, submit_decision
from loan_intelligence.review.narrate import DEFAULT_MODEL, _has_gemini_key, generate_reviewer_note
from loan_intelligence.scenario.simulate import SHOCK_TYPES, persist_scenario_result, run_scenario

BASE_DIR = Path(__file__).resolve().parents[2]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On a fresh container disk (Render free tier: no persistent disk, so
    every cold start after a spin-down begins empty) seed a small synthetic
    portfolio and a batch of real audit entries, so the deployed instance
    never opens to an empty Portfolio Command / Audit Trail. A no-op once
    outputs/ and the audit trail already have real data -- every local dev
    run, and every boot after the first in one container's lifetime. See
    bootstrap.py.

    Deliberately NOT awaited here: Render's free tier is 0.1 CPU, and a
    from-scratch seed can take long enough on that little compute that
    blocking Uvicorn's own startup on it risks Render's port-scan timing the
    deploy out -- even though nothing would actually be broken. Scheduling
    it on a background thread instead lets Uvicorn bind and start serving
    immediately; `/health` is always up regardless, and every data-dependent
    route below checks bootstrap.is_ready() first and 503s with a clear
    message rather than serving a crash or an empty response while the
    thread is still running. run_seed_with_timeout() additionally caps how
    long that can go on for (SEED_TIMEOUT_SECONDS) before declaring seeding
    failed rather than leaving those 503s indistinguishable from "just
    slow" forever -- see bootstrap.py. The task is kept on app.state so it
    isn't garbage-collected mid-run (asyncio only holds a weak reference to
    a task with no other referrer)."""
    from loan_intelligence.bootstrap import run_seed_with_timeout

    app.state.seeding_task = asyncio.create_task(run_seed_with_timeout())
    yield


app = FastAPI(
    title="VeriTape API",
    description=(
        "Auditable decision layer for loan servicing: prediction, anomaly detection, "
        "calibration, and SHAP explainability are real ML (see the model comparison table "
        "in the README) -- this API exposes their outputs, the human reviewer decision loop, "
        "the portfolio-level view, and the SHA-256 hash-chained audit trail. Nothing here "
        "retrains a model or generates new data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Local dev (Next.js default port, and 3001 when 3000 is already taken) plus
# the deployed Vercel frontend at https://veri-tape.vercel.app -- note the
# hyphen: the Vercel project is named "veri-tape", not "veritape". (An
# earlier pass used the no-hyphen placeholder before the real URL was known;
# that mismatch is exactly what made the deployed frontend see "Could not
# reach the API" even though the backend itself was healthy -- the browser's
# preflight was being rejected before any route logic ran.)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "https://veri-tape.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # Vercel gives every branch/PR preview build its own subdomain
    # (veri-tape-<hash>-<team>.vercel.app); this covers those without
    # reopening the wildcard the explicit list above deliberately avoids.
    # Matches on "veri-tape" (with the hyphen) -- the old placeholder-based
    # regex required the literal substring "veritape" with no hyphen, which
    # a real "veri-tape-git-branch.vercel.app" preview URL never contains.
    allow_origin_regex=r"https://veri-tape.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Lazily-built singletons -- loaded once, on first request, not at import time
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _prediction_stack():
    from demo import load_prediction_stack
    return load_prediction_stack()


@lru_cache(maxsize=1)
def _anomaly_stack():
    from demo import load_anomaly_stack
    return load_anomaly_stack()


@lru_cache(maxsize=1)
def _prediction_explainer():
    from loan_intelligence.explain.shap_explain import PredictionExplainer
    return PredictionExplainer()


@lru_cache(maxsize=1)
def _anomaly_explainer():
    from loan_intelligence.explain.shap_explain import AnomalyExplainer
    return AnomalyExplainer()


def _require_seeded() -> None:
    """FastAPI dependency guarding every data-dependent route: 503s while
    the background seed (see `lifespan` above) is still running, instead of
    letting a route crash reading a half-written outputs/ dir or silently
    serve empty/partial data. Cheap to check -- a lock plus a bool read."""
    from loan_intelligence.bootstrap import is_ready, seeding_error

    if not is_ready():
        raise HTTPException(
            status_code=503,
            detail="VeriTape is still seeding its demo portfolio on this fresh deploy -- try again in a few seconds.",
        )
    error = seeding_error()
    if error:
        raise HTTPException(status_code=503, detail=f"Seeding the demo portfolio failed: {error}")


def _history_to_records(history_df: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in history_df.iterrows():
        rec = row.to_dict()
        try:
            rec["payload"] = json.loads(rec["payload"])
        except (TypeError, json.JSONDecodeError):
            pass
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class ReviewRequest(BaseModel):
    decision: Literal["accept", "override"]
    reason: Optional[str] = None


class ScenarioRequest(BaseModel):
    feature: str  # "interest_rate" | "dti" | "regional_income"
    shock: float
    region: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"], summary="Health check", response_model=HealthResponse)
def root():
    """Liveness check. See /docs for the full API reference."""
    return {"service": "veritape-api", "status": "ok"}


@app.get("/health", tags=["meta"], summary="Health check (always up)")
def health():
    """Returns 200 immediately, regardless of whether the background seed
    (see bootstrap.py) is still running -- deliberately has no dependency
    on bootstrap.is_ready(). This is what Render's own health check and the
    keep-alive workflow ping: the process being up and able to serve a
    response is the thing they need to know, not whether the demo portfolio
    has finished seeding yet (GET /loans etc. report that themselves, via a
    503, if it hasn't)."""
    return {"status": "ok"}


@app.get(
    "/portfolio/summary", tags=["portfolio"], summary="Portfolio Command aggregates",
    response_model=PortfolioSummaryResponse, responses={503: {"model": HTTPError}},
    dependencies=[Depends(_require_seeded)],
)
def portfolio_summary():
    """Total expected loss, risk-tier breakdown, flagged-loan count/pct,
    anomaly rate and count by region and loan_type, reviewer override rate,
    and -- if a scenario has been run -- the baseline-vs-shocked expected
    loss comparison. See portfolio/summary.py for the exact definitions
    (in particular, what "the portfolio" means here: the current book under
    active servicing, i.e. the time-aware test split)."""
    return build_summary()


@app.get(
    "/loans", tags=["loans"], summary="Paginated, filterable loan list",
    response_model=LoanListResponse, responses={503: {"model": HTTPError}},
    dependencies=[Depends(_require_seeded)],
)
def list_loans(
    page: int = Query(1, ge=1, description="1-indexed page number"),
    page_size: int = Query(20, ge=1, le=200),
    risk_tier: Optional[Literal["low", "medium", "high"]] = Query(None),
    region: Optional[str] = Query(None),
    loan_type: Optional[str] = Query(None),
    flagged: Optional[bool] = Query(None, description="filter to loans currently flagged (high risk tier or top-1% anomaly)"),
):
    """Summary rows for the current book -- no SHAP or reviewer note here
    (that's GET /loans/{loan_id}); this endpoint is meant to back a fast,
    filterable list/table view."""
    df = load_current_book()
    if risk_tier:
        df = df[df["risk_tier"] == risk_tier]
    if region:
        df = df[df["region"] == region]
    if loan_type:
        df = df[df["loan_type"] == loan_type]
    if flagged is not None:
        df = df[df["is_flagged"] == flagged]

    total = len(df)
    start = (page - 1) * page_size
    page_df = df.iloc[start:start + page_size]

    items = (
        page_df[["loan_id", "region", "loan_type", "loan_amount", "xgb_proba_calibrated", "risk_tier", "anomaly_score", "flagged_top_1pct", "is_flagged"]]
        .rename(columns={"xgb_proba_calibrated": "calibrated_probability", "flagged_top_1pct": "anomaly_flagged"})
        .to_dict(orient="records")
    )
    return {"total": total, "page": page, "page_size": page_size, "items": items}


@app.get(
    "/loans/{loan_id}", tags=["loans"], summary="Full record for one loan",
    response_model=LoanDetailResponse, responses={404: {"model": HTTPError}, 503: {"model": HTTPError}},
    dependencies=[Depends(_require_seeded)],
)
def get_loan(loan_id: str):
    """Prediction, calibrated probability, anomaly flag, SHAP top features,
    reviewer note, and audit history for one loan.

    Scores the loan live against the already-trained models (no retraining)
    the first time it's viewed, logging that scoring to the audit trail if
    it isn't already there -- the same idempotent-on-first-view behavior
    review/decisions.py uses. The reviewer note is likewise generated once
    (via Gemini, or the template fallback with no API key set) and reused
    on subsequent views rather than re-called every request."""
    book = load_current_book()
    row = book[book["loan_id"] == loan_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"loan_id {loan_id!r} not found in the current book")
    r = row.iloc[0]

    trail = AuditTrail()
    ensure_scored(loan_id, trail)

    raw_proba, calibrated_proba = float(r["xgb_proba"]), float(r["xgb_proba_calibrated"])
    anomaly_score, anomaly_flagged = float(r["anomaly_score"]), bool(r["flagged_top_1pct"])

    top_pred = _prediction_explainer().explain_loan(loan_id, book, top_k=6)
    top_anomaly = _anomaly_explainer().explain_loan(loan_id, book, top_k=6) if anomaly_flagged else None

    history = trail.history_for_loan(loan_id)
    narration_entries = history[history["event_type"] == "llm_narration"]
    if not narration_entries.empty:
        note = json.loads(narration_entries.iloc[-1]["payload"])["note"]
    else:
        facts = {
            "loan_id": loan_id, "raw_xgb_probability": round(raw_proba, 4), "calibrated_probability": round(calibrated_proba, 4),
            "top_prediction_drivers": top_pred, "anomaly_score": round(anomaly_score, 4), "anomaly_flagged_top_1pct": anomaly_flagged,
        }
        if top_anomaly:
            facts["top_anomaly_drivers"] = top_anomaly
        note = generate_reviewer_note(facts, model=DEFAULT_MODEL)
        trail.log("llm_narration", loan_id, {"note": note, "model": DEFAULT_MODEL if _has_gemini_key() else "template_fallback"})

    return {
        "loan_id": loan_id, "region": r["region"], "loan_type": r["loan_type"], "loan_amount": float(r["loan_amount"]),
        "raw_probability": raw_proba, "calibrated_probability": calibrated_proba, "risk_tier": r["risk_tier"],
        "anomaly_score": anomaly_score, "anomaly_flagged": anomaly_flagged, "is_flagged": bool(r["is_flagged"]),
        "top_prediction_drivers": top_pred, "top_anomaly_drivers": top_anomaly,
        "reviewer_note": note,
        "audit_history": _history_to_records(trail.history_for_loan(loan_id)),
    }


@app.post(
    "/loans/{loan_id}/review", tags=["loans"], summary="Submit a reviewer decision",
    response_model=ReviewResponse, responses={400: {"model": HTTPError}, 503: {"model": HTTPError}},
    dependencies=[Depends(_require_seeded)],
)
def review_loan(loan_id: str, body: ReviewRequest):
    """Accept or override the current flag on a loan. Rejected (400) if the
    loan isn't currently flagged -- there's nothing to review. Never
    mutates the underlying prediction or anomaly score; logs a new,
    hash-chained reviewer_decision entry that references the flag entry it
    responds to. See review/decisions.py."""
    try:
        return submit_decision(loan_id, body.decision, body.reason)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/scenario/run", tags=["scenario"], summary="Run a portfolio scenario shock",
    response_model=ScenarioRunResponse, responses={400: {"model": HTTPError}, 503: {"model": HTTPError}},
    dependencies=[Depends(_require_seeded)],
)
def scenario_run(body: ScenarioRequest):
    """Shocks a feature (interest_rate: +/- percentage points; dti: +/-
    fraction e.g. 0.15; regional_income: +/-% income change, optionally
    scoped to one region) across the full portfolio and returns the
    before/after expected loss. Persists the same plot/CSV/summary
    artifacts `python -m loan_intelligence.scenario.simulate` would from
    the CLI, and logs the run to the audit trail."""
    if body.feature not in SHOCK_TYPES:
        raise HTTPException(status_code=400, detail=f"feature must be one of {SHOCK_TYPES}")

    summary, result_df, proba_before, proba_after = run_scenario(body.feature, body.shock, body.region)
    persist_scenario_result(body.feature, body.shock, body.region, summary, result_df, proba_before, proba_after)

    return {
        "shock_type": body.feature, "magnitude": body.shock, "region": body.region,
        "expected_loss_before": summary["expected_loss_before"],
        "expected_loss_after": summary["expected_loss_after"],
        "expected_loss_delta": summary["expected_loss_delta"],
        "portfolio_mean_proba_before": summary["portfolio_mean_proba_before"],
        "portfolio_mean_proba_after": summary["portfolio_mean_proba_after"],
        "top_movers": summary["top_movers"],
    }


@app.get(
    "/audit/verify", tags=["audit"], summary="Verify the hash chain",
    response_model=AuditVerifyResponse, responses={503: {"model": HTTPError}},
    dependencies=[Depends(_require_seeded)],
)
def audit_verify():
    """Re-verifies the full SHA-256 hash chain from genesis. Returns
    {"valid": true, "n_entries": N} or {"valid": false, "broken_at_id": id,
    "reason": ...} at the first link that doesn't reproduce."""
    return AuditTrail().verify()


@app.get(
    "/audit/entries", tags=["audit"], summary="List every audit entry",
    response_model=list[AuditEntry], responses={503: {"model": HTTPError}},
    dependencies=[Depends(_require_seeded)],
)
def audit_entries():
    """Every entry in the hash chain, oldest first -- the raw sequence
    verify() walks. For visualizing the chain itself (nodes, links, which
    entry references which); GET /audit/verify stays the lightweight
    valid/broken check for anything that just needs a status."""
    return _history_to_records(AuditTrail().export_all())
