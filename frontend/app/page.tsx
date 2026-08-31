"use client";

import { useCallback, useEffect, useState } from "react";
import AuditTrailSection from "@/components/AuditTrail";
import CapabilitiesSplit from "@/components/CapabilitiesSplit";
import CapabilityGrid from "@/components/CapabilityGrid";
import CoreCapabilities from "@/components/CoreCapabilities";
import FaqAccordion from "@/components/FaqAccordion";
import Footer from "@/components/Footer";
import Loader from "@/components/Loader";
import LoanExplorer from "@/components/LoanExplorer";
import PortfolioCommand from "@/components/PortfolioCommand";
import ScenarioSimulator from "@/components/ScenarioSimulator";
import StorySoFar from "@/components/StorySoFar";
import SubmissionBanner from "@/components/SubmissionBanner";
import ValuePropCallout from "@/components/ValuePropCallout";
import VisualGallery from "@/components/VisualGallery";
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
        className="pt-16 transition-opacity"
        style={{ opacity: revealed ? 1 : 0, transitionDuration: `${DURATION.slow}s` }}
      >
        <PortfolioCommand summary={summary} error={summaryError} onRetry={loadSummary} />
        <CoreCapabilities />
        <CapabilityGrid />
        <VisualGallery />
        <SubmissionBanner />
        <ValuePropCallout />
        <StorySoFar summary={summary} />
        <LoanExplorer regions={regions} loanTypes={loanTypes} />
        <ScenarioSimulator />
        <AuditTrailSection />
        <CapabilitiesSplit />
        <FaqAccordion />
        <Footer />
      </main>
    </>
  );
}
