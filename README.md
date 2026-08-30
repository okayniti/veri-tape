# Loan Intelligence — Intain FinTech Challenge 2026 (AI Track)

Status: **in progress**. This README will be filled out with the full
architecture, judged-criteria mapping, and LLM-usage disclosure once the
pipeline is complete. For now:

## What exists so far

- `loan_intelligence/config.py` — shared schema/constants (date windows,
  categorical domains, messiness rates) used by every stage.
- `loan_intelligence/data/generate_synthetic.py` — generates a synthetic,
  deliberately messy loan-level dataset (`outputs/loans.csv`,
  `outputs/payments.csv`) plus a hidden `outputs/ground_truth.csv` used only
  to evaluate the anomaly detector later.
- `loan_intelligence/data/profile_report.py` — profiles the raw CSVs
  (missingness, dtype issues, domain-range violations, cardinality,
  categorical spelling variants) into `reports/profile_report.html` / `.md`.

Run:

```bash
python -m loan_intelligence.data.generate_synthetic
python -m loan_intelligence.data.profile_report
```

## Coming next (see conversation / commits for progression)

feature engineering + time-aware split -> XGBoost + Bi-LSTM prediction ->
GRU+MLP anomaly detection -> calibration -> SHAP explainability -> scenario
simulation -> LLM reviewer narration (explanation-only) -> SHA-256
hash-chained audit trail -> end-to-end demo.
