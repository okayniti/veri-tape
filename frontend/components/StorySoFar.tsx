"use client";

import type { PortfolioSummary } from "@/lib/api";
import { formatCompactCurrency, formatNumber } from "@/lib/format";
import CountUp from "./CountUp";
import Reveal from "./Reveal";

export default function StorySoFar({ summary }: { summary: PortfolioSummary | null }) {
  if (!summary) return null;

  const stats: { label: string; value: number; formatter: (v: number) => string }[] = [
    { label: "Portfolio Expected Loss", value: summary.total_portfolio_expected_loss, formatter: formatCompactCurrency },
    { label: "Flagged-Loan Count", value: summary.flagged_loan_count, formatter: formatNumber },
  ];
  if (summary.anomaly.lift != null) {
    stats.splice(1, 0, {
      label: "Anomaly Detection Lift",
      value: summary.anomaly.lift,
      formatter: (v) => `${v.toFixed(1)}x`,
    });
  }

  return (
    <section className="mx-auto max-w-5xl px-6 py-20">
      <Reveal>
        <p className="text-center text-xs uppercase tracking-[0.3em] text-muted">The story so far</p>
      </Reveal>
      <div className="mt-10 grid grid-cols-1 gap-10 text-center sm:grid-cols-3">
        {stats.map((s, i) => (
          <Reveal key={s.label} delay={i * 0.08}>
            <p className="font-mono text-4xl text-foreground sm:text-5xl">
              <CountUp value={s.value} formatter={s.formatter} triggerOnScroll />
            </p>
            <p className="mt-2 text-xs uppercase tracking-wide text-muted">{s.label}</p>
          </Reveal>
        ))}
      </div>
    </section>
  );
}
