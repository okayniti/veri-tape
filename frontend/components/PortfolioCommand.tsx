"use client";

import { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import type { PortfolioSummary } from "@/lib/api";
import { formatCompactCurrency, formatCurrency, formatPercent, RISK_TIER_LABEL } from "@/lib/format";
import { prefersReducedMotion } from "@/lib/motion";
import CountUp from "./CountUp";
import PipelineMarquee from "./PipelineMarquee";
import Reveal from "./Reveal";
import { ErrorPanel, LoadingPanel } from "./Status";

gsap.registerPlugin(ScrollTrigger);

function GroupBarList({ title, data }: { title: string; data: Record<string, { count: number; rate: number }> }) {
  const entries = Object.entries(data);
  const max = Math.max(...entries.map(([, v]) => v.rate), 0.0001);
  if (entries.length === 0) return null;
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-muted">{title}</p>
      <div className="mt-4 space-y-3">
        {entries.map(([key, v]) => (
          <div key={key}>
            <div className="flex items-baseline justify-between text-sm">
              <span className="capitalize text-foreground">{key}</span>
              <span className="font-mono text-xs text-muted">
                {v.count} · {formatPercent(v.rate)}
              </span>
            </div>
            <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-border">
              <div
                className="h-full rounded-full bg-accent transition-[width]"
                style={{ width: `${max > 0 ? (v.rate / max) * 100 : 0}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function PortfolioCommand({
  summary,
  error,
  onRetry,
}: {
  summary: PortfolioSummary | null;
  error: string | null;
  onRetry: () => void;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!wrapRef.current || !contentRef.current || prefersReducedMotion()) return;
    const st = ScrollTrigger.create({
      trigger: wrapRef.current,
      start: "top top",
      end: "bottom top",
      scrub: true,
      onUpdate: (self) => {
        gsap.set(contentRef.current, {
          opacity: 1 - self.progress,
          scale: 1 - self.progress * 0.08,
        });
      },
    });
    return () => st.kill();
  }, []);

  return (
    <section id="portfolio" className="relative">
      <PipelineMarquee />
      <div ref={wrapRef} className="relative min-h-[150vh]">
        <div className="sticky top-0 flex h-screen flex-col items-center justify-center px-6 text-center">
          <div ref={contentRef}>
            <span className="inline-flex items-center gap-2 rounded-full border border-accent/40 bg-accent/10 px-4 py-1.5 text-xs text-accent">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M12 15a7 7 0 1 0 0-14 7 7 0 0 0 0 14Z" />
                <path d="m9 12 2 2 4-4" />
                <path d="M8.5 14.5 7 22l5-3 5 3-1.5-7.5" />
              </svg>
              AI Track — Intain FinTech Challenge 2026
            </span>
            <p className="mt-6 text-xs uppercase tracking-[0.35em] text-muted">Portfolio Command</p>
            {error ? (
              <div className="mx-auto mt-8 max-w-md">
                <ErrorPanel message={error} onRetry={onRetry} />
              </div>
            ) : summary ? (
              <>
                <div className="mt-6 font-mono text-5xl font-light text-foreground sm:text-7xl md:text-8xl">
                  <CountUp value={summary.total_portfolio_expected_loss} formatter={formatCompactCurrency} triggerOnScroll />
                </div>
                <p className="mt-4 max-w-md text-sm text-muted">
                  Total expected loss across {summary.n_loans.toLocaleString()} loans in the current book
                </p>
              </>
            ) : (
              <div className="mt-8">
                <LoadingPanel label="Pulling the portfolio…" />
              </div>
            )}
          </div>
        </div>
      </div>

      {summary && (
        <div className="mx-auto max-w-5xl px-6 pb-32">
          <Reveal>
            <h2 className="text-xs uppercase tracking-[0.3em] text-muted">Risk tier breakdown</h2>
          </Reveal>
          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
            {summary.risk_tier_breakdown.map((tier, i) => (
              <Reveal key={tier.tier} delay={i * 0.08}>
                <div className="rounded-xl border border-border bg-background-raised p-6">
                  <p className="text-xs uppercase tracking-wide text-muted">{RISK_TIER_LABEL[tier.tier] ?? tier.tier}</p>
                  <p className="mt-2 font-mono text-3xl text-foreground">{tier.count}</p>
                  <p className="mt-1 text-xs text-muted">{tier.pct_of_portfolio.toFixed(1)}% of portfolio</p>
                </div>
              </Reveal>
            ))}
          </div>

          <div className="mt-16 grid grid-cols-1 gap-8 sm:grid-cols-3">
            <Reveal delay={0.1}>
              <div className="rounded-xl border border-border bg-background-raised p-6">
                <p className="text-xs uppercase tracking-wide text-muted">Flagged loans</p>
                <p className="mt-2 font-mono text-3xl text-foreground">
                  <CountUp value={summary.flagged_loan_count} triggerOnScroll />
                </p>
                <p className="mt-1 text-xs text-muted">{summary.flagged_loan_pct.toFixed(1)}% of the current book</p>
              </div>
            </Reveal>
            <Reveal delay={0.13}>
              <div className="rounded-xl border border-border bg-background-raised p-6">
                <p className="text-xs uppercase tracking-wide text-muted">Anomaly rate</p>
                <p className="mt-2 font-mono text-3xl text-foreground">
                  <CountUp value={summary.anomaly.overall_rate} formatter={formatPercent} triggerOnScroll />
                </p>
                <p className="mt-1 text-xs text-muted">{summary.anomaly.overall_count} loans flagged top-1%</p>
              </div>
            </Reveal>
            <Reveal delay={0.16}>
              <div className="rounded-xl border border-border bg-background-raised p-6">
                <p className="text-xs uppercase tracking-wide text-muted">Reviewer override rate</p>
                {summary.reviewer.n_reviews === 0 ? (
                  <p className="mt-2 text-sm text-muted">No reviews logged yet</p>
                ) : (
                  <>
                    <p className="mt-2 font-mono text-3xl text-foreground">
                      <CountUp value={summary.reviewer.override_rate ?? 0} formatter={formatPercent} triggerOnScroll />
                    </p>
                    <p className="mt-1 text-xs text-muted">
                      {summary.reviewer.n_overrides} of {summary.reviewer.n_reviews} decisions
                    </p>
                  </>
                )}
              </div>
            </Reveal>
          </div>

          <div className="mt-16 grid grid-cols-1 gap-12 sm:grid-cols-2">
            <Reveal delay={0.1}>
              <GroupBarList title={`Anomaly rate by region (${formatPercent(summary.anomaly.overall_rate)} overall)`} data={summary.anomaly.by_region} />
            </Reveal>
            <Reveal delay={0.16}>
              <GroupBarList title="Anomaly rate by loan type" data={summary.anomaly.by_loan_type} />
            </Reveal>
          </div>

          {summary.scenario_comparison && (
            <Reveal delay={0.1} className="mt-16">
              <div className="rounded-xl border border-accent/25 bg-accent/5 p-6">
                <p className="text-xs uppercase tracking-wide text-muted">
                  Latest scenario · {summary.scenario_comparison.scenario_file}
                </p>
                <div className="mt-4 flex flex-wrap items-baseline gap-x-8 gap-y-2">
                  <div>
                    <p className="text-xs text-muted">Baseline expected loss</p>
                    <p className="font-mono text-xl text-foreground">
                      {formatCurrency(summary.scenario_comparison.baseline_expected_loss)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted">Shocked expected loss</p>
                    <p className="font-mono text-xl text-foreground">
                      {formatCurrency(summary.scenario_comparison.shocked_expected_loss)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted">Delta</p>
                    <p className={`font-mono text-xl ${summary.scenario_comparison.delta >= 0 ? "text-danger" : "text-success"}`}>
                      {summary.scenario_comparison.delta >= 0 ? "+" : ""}
                      {formatCurrency(summary.scenario_comparison.delta)}
                    </p>
                  </div>
                </div>
              </div>
            </Reveal>
          )}
        </div>
      )}
    </section>
  );
}
