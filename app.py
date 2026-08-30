"""Minimal Streamlit demo UI for the loan-intelligence pipeline.

Thin wrapper only: every number shown here comes from the same modules used
by demo.py and the CLI entrypoints (models/predict.py, models/anomaly.py,
models/calibrate.py, explain/shap_explain.py, scenario/simulate.py,
review/narrate.py, audit/hash_chain.py). This file adds no modeling logic
of its own -- it loads already-trained artifacts and already-computed
outputs and renders them.

Run: `streamlit run app.py`
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from loan_intelligence.audit.hash_chain import AuditTrail
from loan_intelligence.explain.shap_explain import AnomalyExplainer, PredictionExplainer
from loan_intelligence.models.predict import _as_xgb_frame
from loan_intelligence.review.narrate import DEFAULT_MODEL, generate_reviewer_note
from loan_intelligence.scenario.simulate import SHOCK_TYPES, apply_shock, _load_portfolio
from demo import load_anomaly_stack, load_prediction_stack, pick_interesting_loan_id
from loan_intelligence.models.anomaly import score_anomalies

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"

st.set_page_config(page_title="Loan Intelligence", layout="wide")


@st.cache_resource
def get_prediction_stack():
    return load_prediction_stack()


@st.cache_resource
def get_anomaly_stack():
    return load_anomaly_stack()


@st.cache_resource
def get_prediction_explainer():
    return PredictionExplainer()


@st.cache_resource
def get_anomaly_explainer():
    return AnomalyExplainer()


@st.cache_data
def get_test_df():
    return pd.read_csv(OUTPUT_DIR / "test.csv")


@st.cache_data
def get_model_comparison():
    return pd.read_csv(OUTPUT_DIR / "model_comparison.csv", index_col=0)


@st.cache_data
def get_anomaly_scores():
    return pd.read_csv(OUTPUT_DIR / "anomaly_scores_test.csv")


st.title("Loan Intelligence")
st.caption(
    "Prediction (XGBoost + Bi-LSTM) and anomaly detection (GRU+MLP autoencoder) are real ML. "
    "The LLM only narrates numbers computed elsewhere -- it never predicts."
)

tab_overview, tab_record, tab_scenario, tab_audit, tab_profile = st.tabs(
    ["Portfolio overview", "Record explorer", "Scenario simulation", "Audit trail", "Data profiling"]
)

# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Model comparison (held-out, time-aware test set)")
    st.dataframe(get_model_comparison(), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Calibration")
        reliability_path = REPORT_DIR / "reliability_diagram.png"
        if reliability_path.exists():
            st.image(str(reliability_path), caption="Reliability diagram: before vs. after calibration")
    with col2:
        st.subheader("Anomaly detection: precision @ alert budget")
        scores = get_anomaly_scores()
        rows = []
        for budget in (0.01, 0.025, 0.05):
            k = max(1, round(len(scores) * budget))
            flagged = scores.sort_values("anomaly_score", ascending=False).head(k)
            precision = flagged["is_anomalous"].mean()
            base_rate = scores["is_anomalous"].mean()
            rows.append({
                "alert budget": f"{budget:.1%}", "n flagged": k,
                "precision": f"{precision:.1%}", "lift": f"{precision / base_rate:.2f}x" if base_rate else "n/a",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
with tab_record:
    test_df = get_test_df()
    loan_ids = test_df["loan_id"].tolist()
    default_loan = pick_interesting_loan_id()
    loan_id = st.selectbox("Loan", loan_ids, index=loan_ids.index(default_loan) if default_loan in loan_ids else 0)

    row = test_df[test_df["loan_id"] == loan_id]

    model, feature_cols, calibrator = get_prediction_stack()
    raw_proba = float(model.predict_proba(_as_xgb_frame(row, feature_cols))[:, 1][0])
    calibrated_proba = float(calibrator.predict(np.array([raw_proba]))[0])

    autoencoder, seq_scaler, static_scaler = get_anomaly_stack()
    anomaly_score = float(score_anomalies(autoencoder, seq_scaler, static_scaler, row)[0])
    all_scores = get_anomaly_scores()
    threshold = float(np.quantile(all_scores["anomaly_score"], 0.99))
    flagged = anomaly_score >= threshold

    c1, c2, c3 = st.columns(3)
    c1.metric("Calibrated default probability", f"{calibrated_proba:.1%}", f"raw: {raw_proba:.1%}")
    c2.metric("Anomaly score", f"{anomaly_score:.3f}", f"threshold: {threshold:.3f}")
    c3.metric("Structural anomaly flag", "FLAGGED" if flagged else "clear")

    st.subheader("SHAP: top prediction drivers")
    pred_explainer = get_prediction_explainer()
    top_pred = pred_explainer.explain_loan(loan_id, test_df, top_k=6)
    st.dataframe(pd.DataFrame(top_pred), hide_index=True, width="stretch")

    top_anomaly = None
    if flagged:
        st.subheader("SHAP: top anomaly drivers")
        anomaly_explainer = get_anomaly_explainer()
        top_anomaly = anomaly_explainer.explain_loan(loan_id, test_df, top_k=6)
        st.dataframe(pd.DataFrame(top_anomaly), hide_index=True, width="stretch")

    st.subheader("Reviewer note (LLM, explanation only)")
    facts = {
        "loan_id": loan_id, "raw_xgb_probability": round(raw_proba, 4),
        "calibrated_probability": round(calibrated_proba, 4),
        "top_prediction_drivers": top_pred, "anomaly_score": round(anomaly_score, 4),
        "anomaly_flagged_top_1pct": flagged,
    }
    if top_anomaly:
        facts["top_anomaly_drivers"] = top_anomaly

    if st.button("Generate reviewer note"):
        with st.spinner("asking Claude (or using the template fallback if no API key is set)..."):
            note = generate_reviewer_note(facts, model=DEFAULT_MODEL)
        st.info(note)
        AuditTrail().log("llm_narration", loan_id, {"note": note})
        st.caption("Logged to the audit trail.")

# ---------------------------------------------------------------------------
with tab_scenario:
    st.subheader("Shock a feature across the portfolio")
    shock_type = st.selectbox("Shock type", SHOCK_TYPES)
    magnitude = st.number_input(
        "Magnitude (+2.0 = +2pp interest rate; +0.15 = +0.15 DTI; -15 = -15% income)",
        value=2.0, step=0.5,
    )
    region = None
    if shock_type == "regional_income":
        region = st.selectbox("Region", ["Northeast", "Midwest", "South", "West"])

    if st.button("Run scenario"):
        with st.spinner("rescoring the portfolio..."):
            portfolio = _load_portfolio()
            import json
            fit_params = json.load(open(OUTPUT_DIR / "feature_fit_params.json"))
            model, feature_cols, calibrator = get_prediction_stack()

            raw_before = model.predict_proba(_as_xgb_frame(portfolio, feature_cols))[:, 1]
            proba_before = calibrator.predict(raw_before)
            shocked = apply_shock(portfolio, shock_type, magnitude, fit_params, region=region)
            raw_after = model.predict_proba(_as_xgb_frame(shocked, feature_cols))[:, 1]
            proba_after = calibrator.predict(raw_after)

            result_df = portfolio[["loan_id", "region", "loan_type"]].copy()
            result_df["proba_before"] = proba_before
            result_df["proba_after"] = proba_after
            result_df["delta"] = proba_after - proba_before

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean probability before", f"{proba_before.mean():.2%}")
        c2.metric("Mean probability after", f"{proba_after.mean():.2%}", f"{(proba_after.mean() - proba_before.mean()):+.2%}")
        c3.metric("Newly crossing 50% risk", int(((proba_before < 0.5) & (proba_after >= 0.5)).sum()))

        st.subheader("Distribution shift")
        hist_df = pd.DataFrame({"before shock": proba_before, "after shock": proba_after})
        st.bar_chart(pd.DataFrame({
            "before shock": np.histogram(proba_before, bins=20, range=(0, 1))[0],
            "after shock": np.histogram(proba_after, bins=20, range=(0, 1))[0],
        }))

        st.subheader("Most-affected loans")
        st.dataframe(result_df.sort_values("delta", ascending=False).head(15), hide_index=True, width="stretch")

        AuditTrail().log("scenario_run", None, {
            "shock_type": shock_type, "magnitude": magnitude, "region": region,
            "portfolio_mean_proba_before": round(float(proba_before.mean()), 4),
            "portfolio_mean_proba_after": round(float(proba_after.mean()), 4),
        })
        st.caption("Logged to the audit trail.")

# ---------------------------------------------------------------------------
with tab_audit:
    st.subheader("Tamper-evident hash chain (SHA-256)")
    trail = AuditTrail()
    verification = trail.verify()
    if verification["valid"]:
        st.success(f"Chain valid -- {verification['n_entries']} entries.")
    else:
        st.error(f"Chain BROKEN at id={verification['broken_at_id']}: {verification['reason']}")

    st.dataframe(trail.export_all(), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
with tab_profile:
    st.subheader("Raw data-quality profile (before cleaning)")
    profile_html = REPORT_DIR / "profile_report.html"
    if profile_html.exists():
        st.iframe(profile_html.read_text(encoding="utf-8"), height=1400)
    else:
        st.warning("Run `python -m loan_intelligence.data.profile_report` first.")
