"use client";

import Reveal from "./Reveal";

const FEATURES = [
  {
    label: "Prediction + Calibration",
    description:
      "XGBoost and a Bi-LSTM over payment sequences predict default probability, reaching 0.78 and 0.86 AUC respectively against a time-aware held-out set -- loans the model has never seen, split by origination date rather than at random. Platt scaling then recalibrates the output: Brier score improves 44%, so a predicted 30% actually behaves like one.",
    span: "lg:col-span-2",
  },
  {
    label: "Anomaly Detection",
    description:
      "A GRU+MLP autoencoder flags loans that are structurally unusual -- not just high-risk, but inconsistent with their own history or peers -- by reconstruction error, with no anomaly labels used during training. Evaluated against held-out ground truth, it lands a 6.6x-8x precision lift over a random baseline across alert budgets.",
    span: "lg:col-span-2",
  },
  {
    label: "Reviewer Override Loop",
    description:
      "Flagged loans go through a human: accept or override, with an optional reason. Every decision is its own hash-chained entry that references the exact flag it responds to -- it never rewrites the underlying prediction or anomaly score, only records a permanent judgment on top of it.",
    span: "lg:col-span-2",
  },
  {
    label: "Tamper-Evident Audit Trail",
    description:
      "Every model decision -- predictions, anomaly flags, reviewer overrides, scenario runs -- lands in a SHA-256 hash chain where each entry commits to the one before it. Edit any historical row and the chain visibly breaks at exactly that entry, verifiable live with the button below.",
    span: "lg:col-span-3",
  },
  {
    label: "LLM-Assisted Reviewer Explanations",
    description:
      "Given the SHAP output and anomaly flags already computed for a record, Gemini writes a short, human-readable reviewer note -- it never touches the prediction, the anomaly score, or the calibration; all three are already final by the time the LLM sees them. If the narration service fails or is slow, the app automatically degrades to a labeled template note, and the rest of the pipeline keeps working exactly as before. That fallback has actually been forced and verified, not just claimed.",
    example:
      "Loan L103802 was assigned a low calibrated default probability of 0.0109 (down from a raw probability of 0.0208)... Despite the low default risk, the loan flagged a high structural anomaly score of 1.5612, placing it in the top 1% of anomalies.",
    span: "lg:col-span-3",
  },
];

export default function CoreCapabilities() {
  return (
    <section className="mx-auto max-w-5xl px-6 pt-28">
      <Reveal>
        <h2 className="text-xs uppercase tracking-[0.3em] text-muted">Core Capabilities</h2>
      </Reveal>
      <Reveal delay={0.05}>
        <p className="mt-2 max-w-xl text-3xl text-foreground sm:text-4xl">The five things that actually matter.</p>
      </Reveal>

      <div className="mt-10 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-6">
        {FEATURES.map((f, i) => (
          <Reveal key={f.label} delay={i * 0.06} className={f.span}>
            <div className="flex h-full flex-col rounded-xl border border-border bg-background-raised p-8 shadow-[0_8px_24px_-16px_var(--shadow-color)]">
              <p className="text-lg font-medium text-foreground">{f.label}</p>
              <p className="mt-3 text-sm leading-relaxed text-muted">{f.description}</p>
              {f.example && (
                <div className="mt-4 border-l-2 border-accent/40 pl-3">
                  <p className="text-[11px] uppercase tracking-wide text-accent">Real reviewer note, from this app</p>
                  <p className="mt-1.5 text-xs italic leading-relaxed text-muted">&ldquo;{f.example}&rdquo;</p>
                </div>
              )}
            </div>
          </Reveal>
        ))}
      </div>
    </section>
  );
}
