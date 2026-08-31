"use client";

import Reveal from "./Reveal";

interface Capability {
  label: string;
  description: string;
  span: string; // Tailwind col/row span classes
}

const CAPABILITIES: Capability[] = [
  {
    label: "Reviewer Override Loop",
    description:
      "Accept or override any flagged loan. Every decision is its own hash-chained entry, referencing the exact flag it responds to -- never mutating it.",
    span: "sm:col-span-2 sm:row-span-2",
  },
  {
    label: "Tamper-Evident Audit Trail",
    description: "SHA-256 hash-chained SQLite log. Edit one entry and Verify Chain catches exactly where it broke.",
    span: "sm:col-span-2",
  },
  {
    label: "Portfolio Risk Rollup",
    description: "Expected loss, risk-tier mix, and anomaly rate across the whole book, live.",
    span: "",
  },
  {
    label: "Scenario Simulation",
    description: "Shock interest rate, DTI, or regional income and see the expected-loss shift.",
    span: "",
  },
  {
    label: "Calibrated Predictions",
    description: "Platt scaling turns raw XGBoost scores into probabilities that actually mean what they say.",
    span: "",
  },
  {
    label: "SHAP Explainability",
    description: "Per-record feature attributions for both the prediction model and the anomaly detector.",
    span: "",
  },
  {
    label: "LLM Narration",
    description: "Explanation-only, never predictive -- Gemini phrases numbers a model already computed.",
    span: "sm:col-span-2",
  },
  {
    label: "Time-Aware Validation",
    description: "Trained and tested by origination date, not a random split, so results reflect real deployment.",
    span: "sm:col-span-2",
  },
];

export default function CapabilityGrid() {
  return (
    <section id="capabilities" className="mx-auto max-w-5xl px-6 py-24">
      <Reveal>
        <h2 className="text-xs uppercase tracking-[0.3em] text-muted">Capabilities</h2>
      </Reveal>
      <Reveal delay={0.05}>
        <p className="mt-2 max-w-xl text-2xl text-foreground">What's actually running underneath.</p>
      </Reveal>

      <div className="mt-10 grid grid-cols-1 gap-4 sm:grid-cols-4 sm:[grid-auto-flow:dense] sm:auto-rows-[160px]">
        {CAPABILITIES.map((cap, i) => (
          <Reveal key={cap.label} delay={i * 0.05} className={cap.span}>
            <div className="group flex h-full flex-col justify-between rounded-lg border border-border bg-background-raised p-6 transition-all duration-300 hover:-translate-y-1 hover:border-accent/40 hover:shadow-[0_12px_30px_-12px_rgba(45,212,191,0.25)]">
              <p className="text-sm font-medium text-foreground">{cap.label}</p>
              <p className="mt-2 text-xs leading-relaxed text-muted">{cap.description}</p>
            </div>
          </Reveal>
        ))}
      </div>
    </section>
  );
}
