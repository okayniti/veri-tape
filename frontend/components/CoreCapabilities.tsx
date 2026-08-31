"use client";

import Reveal from "./Reveal";

const FEATURES = [
  {
    label: "Prediction + Calibration",
    description:
      "XGBoost and a Bi-LSTM over payment sequences predict default probability, reaching 0.78 and 0.86 AUC respectively against a time-aware held-out set -- loans the model has never seen, split by origination date rather than at random. Platt scaling then recalibrates the output: Brier score improves 44%, so a predicted 30% actually behaves like one.",
  },
  {
    label: "Anomaly Detection",
    description:
      "A GRU+MLP autoencoder flags loans that are structurally unusual -- not just high-risk, but inconsistent with their own history or peers -- by reconstruction error, with no anomaly labels used during training. Evaluated against held-out ground truth, it lands a 6.6x-8x precision lift over a random baseline across alert budgets.",
  },
  {
    label: "Reviewer Override Loop",
    description:
      "Flagged loans go through a human: accept or override, with an optional reason. Every decision is its own hash-chained entry that references the exact flag it responds to -- it never rewrites the underlying prediction or anomaly score, only records a permanent judgment on top of it.",
  },
  {
    label: "Tamper-Evident Audit Trail",
    description:
      "Every model decision -- predictions, anomaly flags, reviewer overrides, scenario runs -- lands in a SHA-256 hash chain where each entry commits to the one before it. Edit any historical row and the chain visibly breaks at exactly that entry, verifiable live with the button below.",
  },
];

export default function CoreCapabilities() {
  return (
    <section className="mx-auto max-w-5xl px-6 pt-28">
      <Reveal>
        <h2 className="text-xs uppercase tracking-[0.3em] text-muted">Core Capabilities</h2>
      </Reveal>
      <Reveal delay={0.05}>
        <p className="mt-2 max-w-xl text-3xl text-foreground sm:text-4xl">The four things that actually matter.</p>
      </Reveal>

      <div className="mt-10 grid grid-cols-1 gap-6 sm:grid-cols-2">
        {FEATURES.map((f, i) => (
          <Reveal key={f.label} delay={i * 0.07}>
            <div className="h-full rounded-xl border border-border bg-background-raised p-8 shadow-[0_8px_24px_-16px_var(--shadow-color)]">
              <p className="text-lg font-medium text-foreground">{f.label}</p>
              <p className="mt-3 text-sm leading-relaxed text-muted">{f.description}</p>
            </div>
          </Reveal>
        ))}
      </div>
    </section>
  );
}
