"use client";

import { useCallback, useEffect, useState } from "react";
import Loader from "@/components/Loader";
import { getPortfolioSummary, type PortfolioSummary } from "@/lib/api";
import { DURATION } from "@/lib/motion";

export default function Home() {
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(false);

  useEffect(() => {
    getPortfolioSummary()
      .then(setSummary)
      .catch((e) => setSummaryError(e instanceof Error ? e.message : String(e)));
  }, []);

  const handleLoaderDone = useCallback(() => setRevealed(true), []);

  return (
    <>
      <Loader ready={summary !== null || summaryError !== null} onDone={handleLoaderDone} />
      <main
        className="flex min-h-screen items-center justify-center transition-opacity"
        style={{ opacity: revealed ? 1 : 0, transitionDuration: `${DURATION.slow}s` }}
      >
        <p className="text-xs uppercase tracking-[0.3em] text-muted">VeriTape — sections load below as they ship</p>
      </main>
    </>
  );
}
