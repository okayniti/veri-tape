"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { getLoans, type LoanListParams, type LoanListResponse } from "@/lib/api";
import { formatCurrency, formatPercent, RISK_TIER_LABEL } from "@/lib/format";
import Reveal from "./Reveal";
import { EmptyPanel, ErrorPanel, LoadingPanel } from "./Status";
import LoanDetailPanel from "./LoanDetailPanel";

const PAGE_SIZE = 12;

function riskBadgeClass(tier: string): string {
  if (tier === "high") return "bg-danger text-danger-foreground";
  if (tier === "medium") return "border border-accent/50 text-accent";
  return "border border-border text-muted";
}

export default function LoanExplorer({ regions, loanTypes }: { regions: string[]; loanTypes: string[] }) {
  const [page, setPage] = useState(1);
  const [riskTier, setRiskTier] = useState<"" | "low" | "medium" | "high">("");
  const [region, setRegion] = useState("");
  const [loanType, setLoanType] = useState("");
  const [flagged, setFlagged] = useState<"" | "true" | "false">("");

  const [data, setData] = useState<LoanListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedLoanId, setSelectedLoanId] = useState<string | null>(null);

  const params: LoanListParams = useMemo(
    () => ({
      page,
      page_size: PAGE_SIZE,
      risk_tier: riskTier || undefined,
      region: region || undefined,
      loan_type: loanType || undefined,
      flagged: flagged === "" ? undefined : flagged === "true",
    }),
    [page, riskTier, region, loanType, flagged]
  );

  const fetchLoans = useCallback(() => {
    setLoading(true);
    setError(null);
    getLoans(params)
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [params]);

  useEffect(() => {
    fetchLoans();
  }, [fetchLoans]);

  function resetPageAnd<T>(setter: (v: T) => void) {
    return (v: T) => {
      setter(v);
      setPage(1);
    };
  }

  const totalPages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  const selectClass =
    "rounded-md border border-border bg-background-raised px-3 py-2 text-xs uppercase tracking-wide text-foreground focus:border-accent focus:outline-none";

  return (
    <section id="loans" className="mx-auto max-w-6xl px-6 py-28">
      <Reveal>
        <h2 className="text-xs uppercase tracking-[0.3em] text-muted">Loan Explorer</h2>
      </Reveal>
      <Reveal delay={0.05}>
        <p className="mt-2 max-w-xl text-3xl text-foreground sm:text-4xl">
          Every loan in the current book, filterable by risk, region, and status.
        </p>
      </Reveal>

      <Reveal delay={0.1}>
        <div className="mt-8 flex flex-wrap gap-3">
          <select
            value={riskTier}
            onChange={(e) => resetPageAnd(setRiskTier)(e.target.value as typeof riskTier)}
            className={selectClass}
          >
            <option value="">All risk tiers</option>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
          <select value={region} onChange={(e) => resetPageAnd(setRegion)(e.target.value)} className={selectClass}>
            <option value="">All regions</option>
            {regions.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
          <select value={loanType} onChange={(e) => resetPageAnd(setLoanType)(e.target.value)} className={selectClass}>
            <option value="">All loan types</option>
            {loanTypes.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <select
            value={flagged}
            onChange={(e) => resetPageAnd(setFlagged)(e.target.value as typeof flagged)}
            className={selectClass}
          >
            <option value="">All loans</option>
            <option value="true">Flagged only</option>
            <option value="false">Not flagged</option>
          </select>
        </div>
      </Reveal>

      <div className="mt-8 min-h-[200px]">
        {error && <ErrorPanel message={error} onRetry={fetchLoans} />}
        {!error && loading && <LoadingPanel label="Loading loans…" />}
        {!error && !loading && data && data.items.length === 0 && (
          <EmptyPanel message="No loans match these filters." />
        )}
        {!error && data && data.items.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-border text-xs uppercase tracking-wide text-muted">
                  <th className="py-3 pr-4 font-normal">Loan</th>
                  <th className="py-3 pr-4 font-normal">Region</th>
                  <th className="py-3 pr-4 font-normal">Type</th>
                  <th className="py-3 pr-4 font-normal">Amount</th>
                  <th className="py-3 pr-4 font-normal">Probability</th>
                  <th className="py-3 pr-4 font-normal">Risk</th>
                  <th className="py-3 pr-4 font-normal">Anomaly</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((loan) => (
                  <tr
                    key={loan.loan_id}
                    onClick={() => setSelectedLoanId(loan.loan_id)}
                    className="cursor-pointer border-b border-border/60 transition hover:bg-background-raised"
                  >
                    <td className="py-3 pr-4 font-mono text-foreground">{loan.loan_id}</td>
                    <td className="py-3 pr-4 text-muted">{loan.region}</td>
                    <td className="py-3 pr-4 capitalize text-muted">{loan.loan_type}</td>
                    <td className="py-3 pr-4 font-mono text-foreground">{formatCurrency(loan.loan_amount)}</td>
                    <td className="py-3 pr-4 font-mono text-foreground">{formatPercent(loan.calibrated_probability)}</td>
                    <td className="py-3 pr-4">
                      <span className={`rounded-full px-2 py-0.5 text-xs ${riskBadgeClass(loan.risk_tier)}`}>
                        {RISK_TIER_LABEL[loan.risk_tier] ?? loan.risk_tier}
                      </span>
                    </td>
                    <td className="py-3 pr-4">
                      {loan.anomaly_flagged ? <span className="text-accent">flagged</span> : <span className="text-muted">—</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {data && data.total > PAGE_SIZE && (
        <div className="mt-6 flex items-center justify-between text-xs uppercase tracking-wide text-muted">
          <button
            disabled={page <= 1}
            onClick={() => setPage((p) => p - 1)}
            className="transition hover:text-accent disabled:pointer-events-none disabled:opacity-30"
          >
            ← Previous
          </button>
          <span>
            Page {page} of {totalPages} · {data.total} loans
          </span>
          <button
            disabled={page >= totalPages}
            onClick={() => setPage((p) => p + 1)}
            className="transition hover:text-accent disabled:pointer-events-none disabled:opacity-30"
          >
            Next →
          </button>
        </div>
      )}

      {selectedLoanId && (
        <LoanDetailPanel loanId={selectedLoanId} onClose={() => setSelectedLoanId(null)} onReviewed={fetchLoans} />
      )}
    </section>
  );
}
