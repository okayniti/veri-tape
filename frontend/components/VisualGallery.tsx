"use client";

import Reveal from "./Reveal";

const ITEMS = [
  {
    src: "/gallery/risk-tier-breakdown.png",
    label: "Portfolio risk-tier breakdown",
    caption: "Live loan counts across low/medium/high risk tiers, from GET /portfolio/summary.",
  },
  {
    src: "/gallery/shap-explanation.png",
    label: "SHAP explanation",
    caption: "Per-record feature attributions for the prediction model, computed live -- not a global importance plot.",
  },
  {
    src: "/gallery/scenario-before-after.png",
    label: "Scenario simulator",
    caption: "Portfolio expected loss before and after a live interest-rate shock.",
  },
  {
    src: "/gallery/audit-chain.png",
    label: "Audit-chain visualization",
    caption: "The hash chain after a live Verify Chain call -- every link glows once confirmed.",
  },
];

export default function VisualGallery() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-28">
      <Reveal>
        <h2 className="text-xs uppercase tracking-[0.3em] text-muted">See It Running</h2>
      </Reveal>
      <Reveal delay={0.05}>
        <p className="mt-2 max-w-xl text-3xl text-foreground sm:text-4xl">Real captures from the running app.</p>
      </Reveal>

      <Reveal delay={0.1}>
        <div className="mt-10 flex snap-x snap-mandatory gap-6 overflow-x-auto pb-4">
          {ITEMS.map((item) => (
            <div key={item.src} className="w-[85vw] shrink-0 snap-center sm:w-[480px]">
              <div className="overflow-hidden rounded-xl border border-border bg-background-raised shadow-[0_8px_24px_-16px_var(--shadow-color)]">
                {/* eslint-disable-next-line @next/next/no-img-element -- static local capture, no remote optimization needed */}
                <img src={item.src} alt={item.label} className="w-full" />
              </div>
              <p className="mt-3 text-sm font-medium text-foreground">{item.label}</p>
              <p className="mt-1 text-xs text-muted">{item.caption}</p>
            </div>
          ))}
        </div>
      </Reveal>
    </section>
  );
}
