# VeriTape — Intain FinTech Challenge 2026 (AI Track)

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

## Product

VeriTape is an **auditable decision layer for loan servicing** — not a
model demo. The prediction and anomaly-detection models are real, tested
ML (see Results below), but the thing being pitched here is what wraps
them: every automated flag and every human response to it becomes a
permanent, tamper-evident record, and a manager gets a portfolio-level view
of that record instead of only single-loan drilldowns.

**Who'd use it:** loan servicers and their compliance/risk teams — the
people who have to answer "why did we flag this loan, who reviewed it, and
can you prove the record wasn't altered after the fact" during an audit or
a regulatory exam, not just "what's our default rate."

**Why the audit trail is the sellable part:** a probability score is a
commodity — every vendor has one. A SHA-256 hash-chained record of *every*
model decision and *every* human override, verifiable end to end on
demand, is the piece that turns a model into something a compliance buyer
signs off on. This mirrors the audit/compliance infrastructure companies
like Intain already build for structured finance transactions — applied
here one level down, to servicing-level loan review instead of deal-level
reporting.

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
    decisions.py                human accept/override loop on a flagged loan, hash-chained to the flag it responds to
  portfolio/
    summary.py                  Portfolio Command aggregates: expected loss, risk tiers, anomaly rate, override rate
  audit/
    hash_chain.py               SHA-256 hash-chained SQLite log of every model decision
  api/
    main.py                      FastAPI layer exposing all of the above over HTTP -- see API below
demo.py                        one command, one loan, the whole pipeline live
app.py                         minimal Streamlit UI over the same modules the API and CLI use
tests/
  test_decisions.py             pytest: override logs a correct reference and the chain still verifies
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
| Reviewer decision | A human (accept/override) — never changes the stored prediction/anomaly score, only logs a judgment against it |
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
| Human-in-the-loop review | `review/decisions.py` — accept/override a flag; the decision references the exact flag entry's hash, never mutates it (4 pytest tests, `tests/test_decisions.py`) |
| Portfolio-level view | `portfolio/summary.py` — expected loss, risk-tier mix, anomaly rate by region/loan_type, reviewer override rate, scenario comparison |
| Product API | `api/main.py` — every capability above over HTTP for a frontend; live docs at `/docs` |

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

## Reviewer decision loop

A loan is "flagged" when its calibrated probability lands in the "high"
risk tier (≥30%) or the anomaly detector marks it top-1% unusual
(`config.risk_tier`, shared by `review/decisions.py` and
`portfolio/summary.py` so a loan's tier never disagrees between views).

```bash
python -m loan_intelligence.review.decisions --loan-id L100000 --decision override --reason "confirmed false positive with borrower"
python -m loan_intelligence.review.decisions --loan-id L100000 --decision accept
```

The decision is logged as its own hash-chained entry that references the
exact `entry_hash` of the flag it responds to — reviewing a loan with no
current flag is rejected. It never changes the underlying prediction or
anomaly score. `pytest tests/test_decisions.py -v` exercises this end to
end, including that tampering an override after the fact breaks
verification.

## Portfolio Command

```bash
python -m loan_intelligence.portfolio.summary
```

The aggregate view: total expected loss, risk-tier mix, flagged-loan
count/pct, anomaly rate and count by region and loan type, the reviewer
override rate, and — once a scenario has been run — baseline vs. shocked
expected loss on the current book. Backs `GET /portfolio/summary` and the
Streamlit app's overview tab.

## API

```bash
python -m uvicorn loan_intelligence.api.main:app --reload --port 8000
```

(`python -m uvicorn`, not a bare `uvicorn` command — this keeps the repo
root, and the `demo` module this file and `review/decisions.py` import
from, on `sys.path`.) Interactive reference at `http://localhost:8000/docs`.

| Route | What it does |
|---|---|
| `GET /portfolio/summary` | Portfolio Command aggregates |
| `GET /loans` | Paginated, filterable by `risk_tier` / `region` / `loan_type` / `flagged` |
| `GET /loans/{loan_id}` | Full record: prediction, calibrated probability, anomaly flag, SHAP, reviewer note, audit history |
| `POST /loans/{loan_id}/review` | `{"decision": "accept"\|"override", "reason": "..."}` — 400 if the loan isn't currently flagged |
| `POST /scenario/run` | `{"feature": "interest_rate"\|"dti"\|"regional_income", "shock": 2.0, "region": null}` — before/after portfolio expected loss |
| `GET /audit/verify` | Re-verifies the full hash chain; `{"valid": false, "broken_at_id": ...}` on tamper |

CORS is restricted to local dev origins plus the deployed Vercel frontend
(`ALLOWED_ORIGINS` in `api/main.py`) — no auth beyond that (demo scope).
Every route is a thin wrapper over the modules above — no modeling logic
lives in `api/`. Expensive objects (loaded models, the SHAP explainers) are
built once on first request via `functools.lru_cache`: the first
`GET /loans/{id}` call takes ~20s, every call after is ~0.5s.

## Deploying (Vercel + Render)

Frontend on Vercel, backend on Render. Both are stateless-friendly except
for one wrinkle: **Render's free tier has no persistent disk**, so
`outputs/audit_trail.db` and every generated CSV/model file disappear on
every cold start (first deploy, or waking up after 15 minutes idle) — the
same regenerable artifacts `.gitignore` already keeps out of git for local
dev become a problem the moment "regenerate them" has to happen automatically
instead of by running a command yourself.

**Backend (Render)** — `render.yaml` at the repo root is a Blueprint Render
can deploy directly (New + → Blueprint), or configure the same settings by
hand:
- Build: `pip install -r requirements.txt`
- Start: `python -m uvicorn loan_intelligence.api.main:app --host 0.0.0.0 --port $PORT`
  (`python -m`, not bare `uvicorn` — see the note in `api/main.py`'s
  docstring; a bare `uvicorn` entry-point script doesn't put the repo root,
  and with it the top-level `demo` module, on `sys.path`)
- Env vars: `GOOGLE_API_KEY` (for live LLM narration; falls back to a
  labeled template without it), `PYTHON_VERSION=3.11.9`, `PYTHONUNBUFFERED=1`
  (see "Cold-start seeding" below for why this one matters)
- Health check: `/health` (always 200, independent of seeding -- see below)

**Cold-start seeding** (`loan_intelligence/bootstrap.py`): on startup, if
`outputs/` is empty or the audit trail has zero entries, the API runs the
same pipeline `python -m loan_intelligence.<module>` would locally — data
generation, features, split, XGBoost, the anomaly autoencoder, calibration
— at a much smaller `SEED_LOAN_COUNT` (default 100, vs. the full 5,000 used
locally and in the demo video: the deployed instance's job is to prove the
pipeline runs live, not to match that scale) so a cold boot finishes in
seconds even on the free tier's 0.1 CPU, then logs real prediction/anomaly
entries for `VERITAPE_SEED_AUDIT_BATCH` loans (default 20) via the exact
function (`review/decisions.ensure_scored`) a loan's first real view
already uses — so a judge opening the link right after a spin-down gets a
populated Portfolio Command and a non-empty, valid audit chain, not a blank
slate. A no-op once real data already exists (every local dev run, every
boot after the first).

Seeding runs on a background thread with a hard ceiling
(`SEED_TIMEOUT_SECONDS`, default 180s), not awaited during startup: Render's
free tier is 0.1 CPU, and blocking Uvicorn's own port bind on a from-scratch
seed risked Render's port-scan timing the deploy out. `GET /health` is up
immediately either way; every data-dependent route (`/loans`,
`/portfolio/summary`, etc.) checks a `bootstrap.is_ready()` flag first and
returns a 503 with a clear "still seeding" message while the background
thread is running, rather than crashing on a half-written `outputs/` dir or
serving something silently empty. Past the timeout, seeding is declared
failed instead of leaving those 503s indistinguishable from "just slow"
forever. `_run_pipeline()` also logs a line before/after each of its 6
stages, so a log tail alone shows which stage is running -- or which one it
died on -- rather than one line at the start and silence.

A seed failure is loud on purpose: it's logged (stage + full traceback) via
`print(..., flush=True)`, and `PYTHONUNBUFFERED=1` (set in `render.yaml`) is
what actually gets that to Render's log tail -- Python fully block-buffers
stdout by default whenever it isn't attached to a terminal, which is exactly
what made an earlier failure here look like silence rather than an error.

**Keep-alive** (`.github/workflows/keep-alive.yml`): pings the backend every
10 minutes so it doesn't spin down between now and judging (update the
placeholder URL in that file once the Render service exists). Without it,
the seeding above still makes every cold start self-healing — just slower
than a warm one.

**Frontend (Vercel)**: no code changes needed — `NEXT_PUBLIC_API_URL`
already gates every API call (`frontend/lib/api.ts`, `frontend/.env.local`
default). Set it in Vercel's Project Settings to the deployed Render URL.
See `frontend/README.md`'s Deploying section.

**Memory headroom (Render free tier, 512MB RAM)**: the live request path
loads XGBoost (prediction) + a GRU+MLP autoencoder (anomaly) + SHAP's
`TreeExplainer` and `GradientExplainer` — notably *not* the Bi-LSTM, which
`api/main.py` never imports; it exists only as `models/predict.py`'s offline
benchmark comparison point. Even so, PyTorch + XGBoost + SHAP + pandas
resident together is genuinely tight against 512MB, plausibly over it under
load — not a comfortable margin. Two mitigations are already in place:
every expensive object is lazily built on first request
(`functools.lru_cache` singletons in `api/main.py`), and `requirements.txt`
installs the CPU-only PyTorch wheel (the default PyPI wheel bundles unused
CUDA runtime libraries that inflate the resident footprint). If it still
OOMs in practice, the next lever is Render's paid tier, not trimming
anything already built.
