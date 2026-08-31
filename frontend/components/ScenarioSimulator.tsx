"use client";

import { useState } from "react";
import { ApiError, runScenario, type ScenarioRunResponse } from "@/lib/api";
import { formatCurrency, formatPercent } from "@/lib/format";
import CountUp from "./CountUp";
import Reveal from "./Reveal";

const FEATURES = [
  { value: "interest_rate", label: "Interest rate", hint: "percentage points, e.g. 2.0 = +2pp" },
  { value: "dti", label: "DTI", hint: "fraction, e.g. 0.15 = +0.15 DTI" },
  { value: "regional_income", label: "Regional income", hint: "% change, e.g. -15 = -15% income" },
] as const;

const REGIONS = ["Northeast", "Midwest", "South", "West"];

export default function ScenarioSimulator() {
  const [feature, setFeature] = useState<(typeof FEATURES)[number]["value"]>("interest_rate");
  const [shock, setShock] = useState("2.0");
  const [region, setRegion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ScenarioRunResponse | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const shockValue = parseFloat(shock);
    if (Number.isNaN(shockValue)) {
      setError("Enter a numeric shock value.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await runScenario({
        feature,
        shock: shockValue,
        region: feature === "regional_income" && region ? region : undefined,
      });
      setResult(res);
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const activeFeature = FEATURES.find((f) => f.value === feature)!;
  const deltaPositive = result ? result.expected_loss_delta >= 0 : false;

  return (
    <section id="scenario" className="mx-auto max-w-4xl px-6 py-28">
      <Reveal>
        <h2 className="text-xs uppercase tracking-[0.3em] text-muted">Scenario Simulator</h2>
      </Reveal>
      <Reveal delay={0.05}>
        <p className="mt-2 max-w-xl text-3xl text-foreground sm:text-4xl">Shock a portfolio driver and see the expected-loss shift.</p>
      </Reveal>

      <Reveal delay={0.1}>
        <form onSubmit={handleSubmit} className="mt-8 flex flex-wrap items-end gap-4">
          <div>
            <label className="block text-xs uppercase tracking-wide text-muted">Feature</label>
            <select
              value={feature}
              onChange={(e) => setFeature(e.target.value as typeof feature)}
              className="mt-2 rounded-md border border-border bg-background-raised px-3 py-2 text-sm text-foreground focus:border-accent focus:outline-none"
            >
              {FEATURES.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs uppercase tracking-wide text-muted">Shock ({activeFeature.hint})</label>
            <input
              type="number"
              step="any"
              value={shock}
              onChange={(e) => setShock(e.target.value)}
              className="mt-2 w-40 rounded-md border border-border bg-background-raised px-3 py-2 text-sm text-foreground focus:border-accent focus:outline-none"
            />
          </div>
          {feature === "regional_income" && (
            <div>
              <label className="block text-xs uppercase tracking-wide text-muted">Region (optional)</label>
              <select
                value={region}
                onChange={(e) => setRegion(e.target.value)}
                className="mt-2 rounded-md border border-border bg-background-raised px-3 py-2 text-sm text-foreground focus:border-accent focus:outline-none"
              >
                <option value="">All regions</option>
                {REGIONS.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            </div>
          )}
          <button
            type="submit"
            disabled={loading}
            className="rounded-md bg-accent px-6 py-2 text-xs uppercase tracking-wide text-accent-foreground transition hover:bg-accent-hover disabled:opacity-40"
          >
            {loading ? "Running…" : "Run scenario"}
          </button>
        </form>
      </Reveal>

      {error && (
        <p className="mt-6 text-sm text-danger">
          {error}
        </p>
      )}

      {result && (
        <Reveal delay={0.05} className="mt-12">
          <div data-testid="gallery-scenario" className="grid grid-cols-1 gap-6 sm:grid-cols-3">
            <div className="rounded-xl border border-border bg-background-raised p-6">
              <p className="text-xs uppercase tracking-wide text-muted">Before</p>
              <p className="mt-2 font-mono text-2xl text-foreground">
                <CountUp value={result.expected_loss_before} formatter={formatCurrency} />
              </p>
            </div>
            <div className="rounded-xl border border-accent/30 bg-accent/5 p-6">
              <p className="text-xs uppercase tracking-wide text-accent">After</p>
              <p className="mt-2 font-mono text-2xl text-foreground">
                <CountUp value={result.expected_loss_after} formatter={formatCurrency} />
              </p>
            </div>
            <div className="rounded-xl border border-border bg-background-raised p-6">
              <p className="text-xs uppercase tracking-wide text-muted">Delta</p>
              <p className={`mt-2 font-mono text-2xl ${deltaPositive ? "text-danger" : "text-success"}`}>
                {deltaPositive ? "+" : ""}
                {formatCurrency(result.expected_loss_delta)}
              </p>
            </div>
          </div>

          <div className="mt-6 h-2 w-full overflow-hidden rounded-full bg-border">
            <div
              className="h-full rounded-full bg-accent transition-all duration-700"
              style={{
                width: `${Math.min(100, (result.expected_loss_after / Math.max(result.expected_loss_before, result.expected_loss_after)) * 100)}%`,
              }}
            />
          </div>
          <p className="mt-2 text-xs text-muted">
            Mean probability {formatPercent(result.portfolio_mean_proba_before)} → {formatPercent(result.portfolio_mean_proba_after)}
          </p>

          {result.top_movers.length > 0 && (
            <div className="mt-10">
              <p className="text-xs uppercase tracking-wide text-muted">Most-affected loans</p>
              <ul className="mt-3 space-y-2">
                {result.top_movers.slice(0, 5).map((m) => (
                  <li key={m.loan_id} className="flex items-center justify-between border-b border-border/60 pb-2 text-sm">
                    <span className="font-mono text-foreground">{m.loan_id}</span>
                    <span className="text-muted">
                      {m.region} · {m.loan_type}
                    </span>
                    <span className="font-mono text-accent">
                      {formatPercent(m.proba_before)} → {formatPercent(m.proba_after)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Reveal>
      )}
    </section>
  );
}
