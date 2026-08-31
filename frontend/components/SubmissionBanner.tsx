"use client";

import Reveal from "./Reveal";

export default function SubmissionBanner() {
  return (
    <section className="relative mx-6 my-16 overflow-hidden rounded-xl border border-border sm:mx-auto sm:max-w-6xl">
      <div className="absolute inset-0">
        {/* eslint-disable-next-line @next/next/no-img-element -- static local capture, decorative backdrop */}
        <img src="/gallery/audit-chain.png" alt="" aria-hidden className="h-full w-full object-cover opacity-20" />
        <div className="absolute inset-0 bg-gradient-to-t from-background via-background/85 to-background/60" />
      </div>
      <div className="relative px-8 py-20 text-center sm:px-16">
        <Reveal>
          <p className="text-xs uppercase tracking-[0.3em] text-muted">Intain FinTech Challenge 2026 — AI Track</p>
        </Reveal>
        <Reveal delay={0.05}>
          <p className="mx-auto mt-4 max-w-2xl text-3xl text-foreground sm:text-4xl">
            A prediction model judges can inspect, an anomaly detector that proves its own lift, and an audit trail
            that catches its own tampering.
          </p>
        </Reveal>
        <Reveal delay={0.1}>
          <a
            href="https://github.com/okayniti/veri-tape"
            target="_blank"
            rel="noopener noreferrer"
            className="mt-8 inline-block rounded-md bg-accent px-6 py-3 text-xs uppercase tracking-wide text-accent-foreground transition hover:bg-accent-hover"
          >
            View the submission on GitHub
          </a>
        </Reveal>
      </div>
    </section>
  );
}
