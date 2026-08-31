"use client";

import { useCallback, useEffect, useState } from "react";
import Loader from "@/components/Loader";
import LoanExplorer from "@/components/LoanExplorer";
import PortfolioCommand from "@/components/PortfolioCommand";
import { getPortfolioSummary, type PortfolioSummary } from "@/lib/api";
import { DURATION } from "@/lib/motion";

export default function Home() {
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(false);

  const loadSummary = useCallback(() => {
    setSummaryError(null);
    getPortfolioSummary()
      .then(setSummary)
      .catch((e) => setSummaryError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => {
    loadSummary();
  }, [loadSummary]);

  const handleLoaderDone = useCallback(() => setRevealed(true), []);

  const regions = summary ? Object.keys(summary.anomaly.by_region).sort() : [];
  const loanTypes = summary ? Object.keys(summary.anomaly.by_loan_type).sort() : [];

  return (
    <>
      <Loader ready={summary !== null || summaryError !== null} onDone={handleLoaderDone} />
      <main
        className="transition-opacity"
        style={{ opacity: revealed ? 1 : 0, transitionDuration: `${DURATION.slow}s` }}
      >
        <PortfolioCommand summary={summary} error={summaryError} onRetry={loadSummary} />
        <LoanExplorer regions={regions} loanTypes={loanTypes} />
      </main>
    </>
  );
}
