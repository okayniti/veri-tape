"use client";

import Reveal from "./Reveal";

const PIPELINE_TAGS = [
  "Data Profiling",
  "Feature Engineering",
  "Time-Aware Validation",
  "Prediction",
  "Calibration",
  "Anomaly Detection",
  "SHAP Explainability",
  "Scenario Simulation",
  "Audit Trail",
  "LLM Narration",
];

export default function CapabilitiesSplit() {
  return (
    <section className="mx-auto max-w-5xl px-6 py-28">
      <div className="grid grid-cols-1 gap-12 sm:grid-cols-2">
        <Reveal>
          <p className="text-3xl text-foreground sm:text-4xl">You review.</p>
          <p className="mt-1 text-3xl text-muted sm:text-4xl">We handle everything else.</p>
          <p className="mt-6 max-w-sm text-sm leading-relaxed text-muted">
            A reviewer's job is the one decision that actually needs judgment: does this specific flagged loan
            deserve an override. Everything upstream of that -- turning a messy loan tape into a calibrated,
            explained, hash-chained decision -- already ran before the loan ever reached a reviewer's screen.
          </p>
        </Reveal>
        <Reveal delay={0.1}>
          <div className="flex flex-wrap gap-2">
            {PIPELINE_TAGS.map((tag) => (
              <span key={tag} className="rounded-full border border-border bg-background-raised px-4 py-2 text-xs text-muted">
                {tag}
              </span>
            ))}
          </div>
        </Reveal>
      </div>
    </section>
  );
}
