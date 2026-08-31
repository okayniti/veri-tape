# VeriTape — Frontend

A single continuous-scroll page over the live VeriTape API: cinematic loader
→ Portfolio Command → Loan Explorer (with the reviewer decision loop) →
Scenario Simulator → Audit Trail visualization.

Zero mock data. Every number, every SHAP bar, every audit entry is fetched
live from the backend at `NEXT_PUBLIC_API_URL`.

## Setup

```bash
npm install
cp .env.example .env.local   # points at http://127.0.0.1:8000 by default
```

The backend (`loan_intelligence/api/main.py`) must be running first:

```bash
python -m uvicorn loan_intelligence.api.main:app --reload --port 8000
```

Then:

```bash
npm run dev
```

## Regenerating API types

This is the ground-truth step -- every component's data shape comes from
these generated types, never hand-written:

```bash
npm run gen:types
```

Requires the API to be running (reads its live `/openapi.json`).

## Stack

Next.js (App Router) + TypeScript + Tailwind v4. GSAP + ScrollTrigger for
scroll-driven reveals and the portfolio-hero pin/release; Lenis for smooth
scroll. One accent color, one shared motion config (`lib/motion.ts`) --
no per-component ad hoc easing.

## Notable behavior

- `GET /loans` paginates via `page_size`, not `limit`.
- The first `GET /loans/{id}` in a fresh backend process can take up to
  ~40s (cold-start: loading the model stack and building the SHAP
  explainers). The detail panel narrates this explicitly rather than
  showing a bare spinner. Every loan after the first is near-instant.
- The audit trail's "Verify chain" button calls the real
  `GET /audit/verify` + `GET /audit/entries` -- the broken-chain animation
  was tested against an actually-tampered database entry, not just coded
  and assumed to work.
