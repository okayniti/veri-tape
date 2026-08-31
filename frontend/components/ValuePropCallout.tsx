"use client";

import Reveal from "./Reveal";

const BULLETS = [
  "Cuts manual review time -- only structurally unusual or high-risk loans reach a human, not the whole book.",
  "Catches structural anomalies before they compound -- flagged by reconstruction error against a loan's own history and peers, not just a bad prediction.",
  "Ships with an audit trail already built in -- every decision is hash-chained from day one, not bolted on after the fact.",
];

export default function ValuePropCallout() {
  return (
    <section className="mx-auto max-w-5xl px-6 py-28">
      <div className="grid grid-cols-1 gap-12 sm:grid-cols-2 sm:items-center">
        <div>
          <Reveal>
            <h2 className="text-xs uppercase tracking-[0.3em] text-muted">Why It's Different</h2>
          </Reveal>
          <ul className="mt-6 space-y-5">
            {BULLETS.map((b, i) => (
              <Reveal key={b} delay={i * 0.07}>
                <li className="flex gap-3 text-sm leading-relaxed text-muted">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
                  {b}
                </li>
              </Reveal>
            ))}
          </ul>
        </div>
        <Reveal delay={0.1}>
          <div className="rounded-xl border border-accent/30 bg-accent/5 p-8 shadow-[0_8px_24px_-16px_var(--shadow-color)]">
            <p className="text-xs uppercase tracking-wide text-accent">The Reviewer Override Loop</p>
            <p className="mt-3 text-lg text-foreground">A human always has the last word.</p>
            <p className="mt-3 text-sm leading-relaxed text-muted">
              Every flagged loan -- a top-1% structural anomaly, or a high calibrated risk -- goes to a person who
              accepts or overrides it, with an optional reason. The decision is hash-chained to the exact flag it
              responds to and never rewrites the model's own output. It's the one feature that turns a probability
              into an auditable decision.
            </p>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
