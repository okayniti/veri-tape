# Loan Intelligence — Intain FinTech Challenge 2026 (AI Track)

Profile messy loan-level data, predict delinquency/default, detect
structural anomalies, calibrate the probabilities, explain both models
per-record, simulate portfolio scenarios, and hand a human reviewer a
short plain-English note — with a tamper-evident audit trail behind every
decision.

**The LLM narrates. It does not predict.** Every number a reviewer sees —
the default probability, the calibration, the anomaly score, the SHAP
attributions — is produced by classical/deep ML (XGBoost, PyTorch, SHAP,
scikit-learn) before an LLM ever sees it. `review/narrate.py` takes those
already-final numbers and phrases them in plain English; its system prompt
explicitly forbids inventing or revising them, and nothing in that module
writes back to a prediction, a feature, or a model file. Delete the LLM
call entirely and every other capability in this repo — prediction,
anomaly detection, calibration, explainability, scenario simulation, the
audit trail — still works exactly as before. That's the difference between
narration and an "LLM wrapper."

## Architecture

```
loan_intelligence/
  config.py              shared schema, date windows, categorical domains
  data/
    generate_synthetic.py   synthetic loan tape + injected messiness + hidden anomaly ground truth
    profile_report.py       data-quality report on the RAW (uncleaned) CSVs -> reports/profile_report.html
    clean.py                resolves exactly the issues profile_report.py surfaces
  features/
    build_features.py       per-loan features from the feature window only (months 1-12)
    time_split.py            time-aware train/test split + leakage-safe imputation/cohort stats
  models/
    predict.py               XGBoost baseline + Bi-LSTM on payment sequences, vs. 2 simpler baselines
    anomaly.py                GRU+MLP autoencoder -> structural anomaly score, precision@budget / lift
    calibrate.py              Platt scaling, Brier score before/after, reliability diagram
    calibrators.py            calibrator classes (own module so joblib pickles resolve from anywhere)
  explain/
    shap_explain.py           per-record SHAP for both the prediction and anomaly models
  scenario/
    simulate.py                shock a feature across the portfolio, rescore, report the shift
  review/
    narrate.py                 LLM reviewer note from already-computed numbers (explanation only)
  audit/
    hash_chain.py               SHA-256 hash-chained SQLite log of every model decision
demo.py                        one command, one loan, the whole pipeline live
```

Each module is runnable and testable on its own (`python -m loan_intelligence.<pkg>.<module>`),
so the git history is one working stage per commit rather than one big drop.

## Why this isn't an LLM wrapper

| Stage | What actually computes it |
|---|---|
| Default probability | XGBoost (gradient-boosted trees) + a Bi-LSTM over 12-month payment sequences |
| Structural anomaly score | A GRU (sequence) + MLP (static features) autoencoder, scored by reconstruction error |
| Calibrated probability | Platt scaling fit on a held-out slice of train, evaluated by Brier score |
| Feature attributions | SHAP (`TreeExplainer` for XGBoost, `GradientExplainer` for the autoencoder) |
| Scenario shift | The same XGBoost model + calibrator, rescored on shocked features |
| Reviewer note | Gemini, given the numbers above as fixed input — narration only |
| Audit record | SHA-256 hash chain over every one of the above, in SQLite |

## Judged-criteria mapping

| Judged criterion | Where |
|---|---|
| Messy data profiling | `data/generate_synthetic.py` (injects missingness, mixed date/DTI/rate formats, outliers, duplicate loan_ids), `data/profile_report.py` (HTML/markdown report) |
| Time-aware validation | `features/time_split.py` — split by `origination_date`, not randomly; rationale documented in the module docstring |
| Real predictive ML | `models/predict.py` — XGBoost + Bi-LSTM, benchmarked against a majority-class and an origination-only logistic baseline |
| Anomaly detection | `models/anomaly.py` — GRU+MLP autoencoder, evaluated by precision@alert-budget and lift over a random baseline against hidden ground truth |
| Calibration | `models/calibrate.py` — Brier score before/after, reliability diagram (`reports/reliability_diagram.png`) |
| Explainability | `explain/shap_explain.py` — per-record top-feature SHAP for *both* models, not just global importance |
| Scenario simulation | `scenario/simulate.py` — shock interest rate / DTI / regional income, see the portfolio-level and per-loan shift |
| Human-reviewer explanation | `review/narrate.py` — LLM narration, explanation-only (see above) |
| Traceability / audit | `audit/hash_chain.py` — SHA-256 hash-chained SQLite log; `main()` demonstrates tamper detection live |

## Results (this run, seed 42, 5,000 synthetic loans)

**Prediction** (held-out, time-aware test set):

| Model | AUC | PR-AUC | Precision@0.5 | Recall@0.5 |
|---|---|---|---|---|
| Majority-class baseline | 0.500 | 0.065 | 0.000 | 0.000 |
| Logistic regression (origination fields only) | 0.675 | 0.099 | 0.111 | 0.615 |
| XGBoost | 0.781 | 0.446 | 0.303 | 0.462 |
| Bi-LSTM (payment sequences) | 0.861 | 0.565 | 0.225 | 0.723 |

Payment-history sequences carry real signal beyond static underwriting fields.

**Calibration**: Brier score 0.082 -> 0.046 (Platt scaling, **-44%**). Raw XGBoost
scores are badly overconfident at the tails — see `reports/reliability_diagram.png`.

**Anomaly detection** (unsupervised GRU+MLP autoencoder, evaluated against
hidden injected ground truth): **6.6x–8x lift** over a random baseline
across 1–5% alert budgets on the held-out test set.

## Setup

```bash
pip install -r requirements.txt
```

For live LLM narration (Google Gemini, via the `google-genai` SDK), copy
`.env.example` to `.env` and paste your key in as `GOOGLE_API_KEY=...`
(`.env` is gitignored, so it's never committed). Without a key set,
`review/narrate.py` and `demo.py` fall back to a deterministic,
clearly-labeled template so the rest of the pipeline still demos end to end.

## Running the full pipeline

```bash
python -m loan_intelligence.data.generate_synthetic
python -m loan_intelligence.data.profile_report
python -m loan_intelligence.features.build_features
python -m loan_intelligence.features.time_split
python -m loan_intelligence.models.predict
python -m loan_intelligence.models.anomaly
python -m loan_intelligence.models.calibrate
python -m loan_intelligence.explain.shap_explain
python -m loan_intelligence.scenario.simulate --shock interest_rate --magnitude 2.0
python -m loan_intelligence.audit.hash_chain
python -m loan_intelligence.review.narrate --loan-id L100000

python demo.py                       # end-to-end walkthrough on an auto-picked record
python demo.py --loan-id L100000     # or pick one yourself
```

## Scenario simulation examples

```bash
python -m loan_intelligence.scenario.simulate --shock interest_rate --magnitude 2.0
python -m loan_intelligence.scenario.simulate --shock dti --magnitude 0.15
python -m loan_intelligence.scenario.simulate --shock regional_income --region South --magnitude -15
```

Each run recomputes the features that mechanically depend on the shocked
driver (loan-to-income ratio, peer-cohort z-scores — against the
*original* cohort statistics, not refit on the shocked population),
rescores with the calibrated XGBoost model, and logs the run to the audit
trail.
