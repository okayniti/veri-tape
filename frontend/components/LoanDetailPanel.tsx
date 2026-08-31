"use client";

import { useEffect, useState } from "react";
import { ApiError, getLoanDetail, submitReview, type LoanDetail, type ReviewResponse, type ShapDriver } from "@/lib/api";
import { formatCurrency, formatPercent, RISK_TIER_LABEL } from "@/lib/format";
import { ErrorPanel } from "./Status";

const EVENT_LABEL: Record<string, string> = {
  prediction: "Prediction",
  anomaly_flag: "Anomaly check",
  llm_narration: "Reviewer note generated",
  reviewer_decision: "Reviewer decision",
  scenario_run: "Scenario run",
};

function ShapBarList({ title, drivers }: { title: string; drivers: ShapDriver[] }) {
  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.shap_value)), 0.0001);
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-muted">{title}</p>
      <div className="mt-4 space-y-3">
        {drivers.map((d) => {
          const widthPct = (Math.abs(d.shap_value) / maxAbs) * 100;
          const positive = d.shap_value >= 0;
          return (
            <div key={d.feature}>
              <div className="flex items-baseline justify-between gap-2 text-xs">
                <span className="truncate text-foreground" title={d.feature}>
                  {d.feature}
                </span>
                <span className="whitespace-nowrap font-mono text-muted">
                  {positive ? "+" : ""}
                  {d.shap_value.toFixed(3)} · {typeof d.value === "number" ? d.value.toFixed(2) : d.value}
                </span>
              </div>
              <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-border">
                <div
                  className={positive ? "h-full rounded-full bg-accent" : "h-full rounded-full border border-accent/50"}
                  style={{ width: `${widthPct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function LoanDetailPanel({
  loanId,
  onClose,
  onReviewed,
}: {
  loanId: string;
  onClose: () => void;
  onReviewed: () => void;
}) {
  const [detail, setDetail] = useState<LoanDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [slowHint, setSlowHint] = useState(false);
  const [reason, setReason] = useState("");
  const [submitting, setSubmitting] = useState<"accept" | "override" | null>(null);
  const [reviewResult, setReviewResult] = useState<ReviewResponse | null>(null);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const raf = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(raf);
  }, []);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setDetail(null);
    setReviewResult(null);
    setReviewError(null);
    setSlowHint(false);

    const slowTimer = window.setTimeout(() => setSlowHint(true), 3500);

    getLoanDetail(loanId)
      .then(setDetail)
      .catch((e) => setError(e instanceof ApiError ? e.detail : e instanceof Error ? e.message : String(e)))
      .finally(() => {
        setLoading(false);
        window.clearTimeout(slowTimer);
      });

    return () => window.clearTimeout(slowTimer);
  }, [loanId]);

  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, []);

  function handleClose() {
    setMounted(false);
    window.setTimeout(onClose, 320);
  }

  async function handleDecision(decision: "accept" | "override") {
    setSubmitting(decision);
    setReviewError(null);
    try {
      const res = await submitReview(loanId, { decision, reason: reason || undefined });
      setReviewResult(res);
      onReviewed();
    } catch (e) {
      setReviewError(e instanceof ApiError ? e.detail : e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(null);
    }
  }

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <button
        aria-label="Close detail panel"
        onClick={handleClose}
        className={`absolute inset-0 bg-black/70 transition-opacity duration-300 ${mounted ? "opacity-100" : "opacity-0"}`}
      />
      <div
        className={`relative flex h-full w-full max-w-xl flex-col overflow-y-auto border-l border-border bg-background p-6 transition-transform duration-500 ease-[cubic-bezier(0.65,0,0.35,1)] sm:p-8 ${
          mounted ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between">
          <p className="font-mono text-sm text-muted">{loanId}</p>
          <button onClick={handleClose} className="text-xs uppercase tracking-wide text-muted transition hover:text-foreground">
            Close
          </button>
        </div>

        {error && (
          <div className="mt-8">
            <ErrorPanel message={error} />
          </div>
        )}

        {loading && (
          <div className="mt-16 flex flex-col items-center text-center">
            <span className="relative flex h-3 w-3">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
              <span className="relative inline-flex h-3 w-3 rounded-full bg-accent" />
            </span>
            <p className="mt-4 text-sm text-foreground">Running full explainability pass…</p>
            <p className="mt-1 text-xs text-muted">Scoring the loan, computing SHAP attributions, generating the reviewer note.</p>
            {slowHint && (
              <p className="mt-4 max-w-xs text-xs text-muted">
                First look at a loan can take up to ~40s while the model stack and SHAP explainer warm up — a cold-start
                cost, not a bug. Every loan after this one is near-instant.
              </p>
            )}
          </div>
        )}

        {detail && (
          <div className="mt-8 space-y-8">
            <div>
              <p className="text-xs uppercase tracking-wide text-muted">
                {detail.region} · {detail.loan_type}
              </p>
              <div className="mt-3 grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-muted">Calibrated probability</p>
                  <p className="mt-1 font-mono text-2xl text-foreground">{formatPercent(detail.calibrated_probability)}</p>
                  <p className="text-xs text-muted">raw {formatPercent(detail.raw_probability)}</p>
                </div>
                <div>
                  <p className="text-xs text-muted">Anomaly score</p>
                  <p className="mt-1 font-mono text-2xl text-foreground">{detail.anomaly_score.toFixed(3)}</p>
                  {detail.anomaly_flagged && <p className="text-xs text-accent">flagged top 1%</p>}
                </div>
              </div>
              <div className="mt-3 flex flex-wrap gap-2 text-xs">
                <span
                  className={`rounded-full px-2 py-0.5 ${
                    detail.risk_tier === "high"
                      ? "bg-accent text-accent-foreground"
                      : detail.risk_tier === "medium"
                        ? "border border-accent/50 text-accent"
                        : "border border-border text-muted"
                  }`}
                >
                  {RISK_TIER_LABEL[detail.risk_tier] ?? detail.risk_tier} risk
                </span>
                {detail.is_flagged && (
                  <span className="rounded-full border border-accent/50 px-2 py-0.5 text-accent">flagged</span>
                )}
              </div>
              <p className="mt-2 font-mono text-sm text-muted">{formatCurrency(detail.loan_amount)}</p>
            </div>

            <div>
              <p className="text-xs uppercase tracking-wide text-muted">Reviewer note</p>
              <p className="mt-2 text-sm leading-relaxed text-foreground">{detail.reviewer_note}</p>
            </div>

            <ShapBarList title="Top prediction drivers" drivers={detail.top_prediction_drivers} />
            {detail.top_anomaly_drivers && <ShapBarList title="Top anomaly drivers" drivers={detail.top_anomaly_drivers} />}

            {detail.is_flagged ? (
              <div className="rounded-lg border border-accent/30 bg-accent/5 p-5">
                <p className="text-xs uppercase tracking-wide text-accent">Reviewer action</p>
                {reviewResult ? (
                  <div className="mt-4 space-y-3">
                    <p className="text-sm text-foreground">
                      Decision logged: <span className="font-medium capitalize">{reviewResult.decision}</span>
                    </p>
                    <p className="text-xs text-muted">
                      This entry is hash-chained to the exact flag it responds to — the record below is what makes it
                      auditable, not the decision text alone.
                    </p>
                    <div className="space-y-1.5 rounded border border-border bg-background p-3 font-mono text-[11px] leading-relaxed text-muted">
                      <p className="break-all">
                        references_hash → <span className="text-accent">{reviewResult.references_hash}</span>
                      </p>
                      <p className="break-all">
                        entry_hash&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ <span className="text-foreground">{reviewResult.entry_hash}</span>
                      </p>
                    </div>
                  </div>
                ) : (
                  <>
                    <textarea
                      value={reason}
                      onChange={(e) => setReason(e.target.value)}
                      placeholder="Optional reason…"
                      rows={2}
                      className="mt-3 w-full resize-none rounded border border-border bg-background-raised p-2 text-sm text-foreground placeholder:text-muted focus:border-accent focus:outline-none"
                    />
                    {reviewError && <p className="mt-2 text-xs text-danger">{reviewError}</p>}
                    <div className="mt-3 flex gap-3">
                      <button
                        disabled={!!submitting}
                        onClick={() => handleDecision("accept")}
                        className="rounded border border-border px-4 py-2 text-xs uppercase tracking-wide text-foreground transition hover:border-accent hover:text-accent disabled:opacity-40"
                      >
                        {submitting === "accept" ? "Logging…" : "Accept"}
                      </button>
                      <button
                        disabled={!!submitting}
                        onClick={() => handleDecision("override")}
                        className="rounded bg-accent px-4 py-2 text-xs uppercase tracking-wide text-accent-foreground transition hover:opacity-90 disabled:opacity-40"
                      >
                        {submitting === "override" ? "Logging…" : "Override"}
                      </button>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <p className="text-xs text-muted">This loan isn&apos;t currently flagged — nothing to review.</p>
            )}

            <div>
              <p className="text-xs uppercase tracking-wide text-muted">Audit history</p>
              <ul className="mt-3 space-y-2">
                {detail.audit_history.map((entry) => (
                  <li key={entry.id} className="flex items-center justify-between border-b border-border/60 pb-2 text-xs">
                    <span className="text-foreground">{EVENT_LABEL[entry.event_type] ?? entry.event_type}</span>
                    <span className="font-mono text-muted">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
