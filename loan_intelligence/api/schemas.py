"""Pydantic response models for every route in api/main.py.

Without these, FastAPI's generated OpenAPI schema types every 200 response
as "unknown" (routes returned plain dicts) -- which defeats the point of
generating a frontend's TypeScript types from that schema. Adding
response_model= to each route both documents the real shape in /docs and
gives openapi-typescript something real to generate from.
"""
from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel


class HealthResponse(BaseModel):
    service: str
    status: str


# --- Portfolio summary -----------------------------------------------------

class RiskTierBucket(BaseModel):
    tier: str
    count: int
    pct_of_portfolio: float


class GroupStat(BaseModel):
    count: int
    rate: float


class AnomalyBreakdown(BaseModel):
    overall_rate: float
    overall_count: int
    by_region: dict[str, GroupStat]
    by_loan_type: dict[str, GroupStat]


class ReviewerStats(BaseModel):
    n_reviews: int
    n_overrides: int
    override_rate: Optional[float] = None


class ScenarioComparison(BaseModel):
    scenario_file: str
    n_loans_in_current_book: int
    baseline_expected_loss: float
    shocked_expected_loss: float
    delta: float


class PortfolioSummaryResponse(BaseModel):
    n_loans: int
    total_portfolio_expected_loss: float
    average_calibrated_probability: float
    risk_tier_breakdown: list[RiskTierBucket]
    flagged_loan_count: int
    flagged_loan_pct: float
    anomaly: AnomalyBreakdown
    reviewer: ReviewerStats
    scenario_comparison: Optional[ScenarioComparison] = None


# --- Loan list ---------------------------------------------------------

class LoanListItem(BaseModel):
    loan_id: str
    region: str
    loan_type: str
    loan_amount: float
    calibrated_probability: float
    risk_tier: str
    anomaly_score: float
    anomaly_flagged: bool
    is_flagged: bool


class LoanListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[LoanListItem]


# --- Loan detail ---------------------------------------------------------

class ShapDriver(BaseModel):
    feature: str
    shap_value: float
    # a feature's raw value: numeric for engineered/origination fields, or a
    # category label (e.g. loan_type="auto") for the handful of categorical
    # features XGBoost's native categorical support includes in SHAP output.
    value: Union[float, str]


class AuditEntry(BaseModel):
    id: int
    timestamp: str
    event_type: str
    loan_id: Optional[str] = None
    references_hash: str
    # shape depends on event_type (prediction / anomaly_flag / llm_narration /
    # reviewer_decision / scenario_run each carry different fields) -- see
    # audit/hash_chain.py and the event-specific .log() call sites.
    payload: dict[str, Any]
    entry_hash: str


class LoanDetailResponse(BaseModel):
    loan_id: str
    region: str
    loan_type: str
    loan_amount: float
    raw_probability: float
    calibrated_probability: float
    risk_tier: str
    anomaly_score: float
    anomaly_flagged: bool
    is_flagged: bool
    top_prediction_drivers: list[ShapDriver]
    top_anomaly_drivers: Optional[list[ShapDriver]] = None
    reviewer_note: str
    audit_history: list[AuditEntry]


# --- Review ---------------------------------------------------------------

class ReviewResponse(BaseModel):
    loan_id: str
    decision: str
    reason: str
    flag_event_type: str
    references_hash: str
    entry_hash: str


# --- Scenario ---------------------------------------------------------

class ScenarioTopMover(BaseModel):
    loan_id: str
    region: str
    loan_type: str
    proba_before: float
    proba_after: float
    delta: float


class ScenarioRunResponse(BaseModel):
    shock_type: str
    magnitude: float
    region: Optional[str] = None
    expected_loss_before: float
    expected_loss_after: float
    expected_loss_delta: float
    portfolio_mean_proba_before: float
    portfolio_mean_proba_after: float
    top_movers: list[ScenarioTopMover]


# --- Audit ---------------------------------------------------------------

class AuditVerifyResponse(BaseModel):
    valid: bool
    n_entries: Optional[int] = None
    broken_at_id: Optional[int] = None
    reason: Optional[str] = None


class HTTPError(BaseModel):
    detail: str
