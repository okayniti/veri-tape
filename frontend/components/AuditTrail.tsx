"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import gsap from "gsap";
import { getAuditEntries, verifyAudit, type AuditEntry, type AuditVerifyResponse } from "@/lib/api";
import Reveal from "./Reveal";
import { EmptyPanel, ErrorPanel, LoadingPanel } from "./Status";

const EVENT_LABEL: Record<string, string> = {
  prediction: "Prediction",
  anomaly_flag: "Anomaly check",
  llm_narration: "Reviewer note",
  reviewer_decision: "Reviewer decision",
  scenario_run: "Scenario run",
};

function shortHash(h: string): string {
  if (!h) return "—";
  return `${h.slice(0, 8)}…${h.slice(-6)}`;
}

export default function AuditTrailSection() {
  const [entries, setEntries] = useState<AuditEntry[] | null>(null);
  const [entriesError, setEntriesError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [verifying, setVerifying] = useState(false);
  const [verifyResult, setVerifyResult] = useState<AuditVerifyResponse | null>(null);
  const [verifyError, setVerifyError] = useState<string | null>(null);

  const nodeRefs = useRef<Map<number, HTMLDivElement>>(new Map());
  const linkRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  const fetchEntries = useCallback(() => {
    setLoading(true);
    setEntriesError(null);
    getAuditEntries()
      .then((data) => {
        nodeRefs.current.clear();
        linkRefs.current.clear();
        setEntries(data);
      })
      .catch((e) => setEntriesError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchEntries();
  }, [fetchEntries]);

  async function handleVerify() {
    setVerifying(true);
    setVerifyError(null);
    setVerifyResult(null);
    gsap.set(Array.from(linkRefs.current.values()), { backgroundColor: "var(--border)" });
    gsap.set(Array.from(nodeRefs.current.values()), { borderColor: "var(--border)", boxShadow: "none" });

    try {
      const [res, freshEntries] = await Promise.all([verifyAudit(), getAuditEntries()]);
      setEntries(freshEntries);
      setVerifyResult(res);

      const brokenIndex = res.valid ? -1 : freshEntries.findIndex((e) => e.id === res.broken_at_id);

      const tl = gsap.timeline();
      freshEntries.forEach((entry, i) => {
        if (!res.valid && brokenIndex >= 0 && i > brokenIndex) return; // animation halts at the break

        const isBroken = !res.valid && i === brokenIndex;
        const link = linkRefs.current.get(entry.id);
        const node = nodeRefs.current.get(entry.id);

        if (link) {
          tl.to(link, { backgroundColor: isBroken ? "var(--danger)" : "var(--accent)", duration: 0.25 }, i * 0.12);
        }
        if (node) {
          tl.to(
            node,
            {
              borderColor: isBroken ? "var(--danger)" : "var(--accent)",
              boxShadow: isBroken ? "0 0 0 4px rgba(241,97,88,0.15)" : "0 0 0 4px rgba(45,212,191,0.15)",
              duration: 0.25,
            },
            i * 0.12
          );
        }
      });
    } catch (e) {
      setVerifyError(e instanceof Error ? e.message : String(e));
    } finally {
      setVerifying(false);
    }
  }

  return (
    <section id="audit" className="mx-auto max-w-3xl px-6 py-24">
      <Reveal>
        <h2 className="text-xs uppercase tracking-[0.3em] text-muted">Audit Trail</h2>
      </Reveal>
      <Reveal delay={0.05}>
        <p className="mt-2 max-w-xl text-2xl text-foreground">
          Every model decision, hash-chained. Tamper with any entry and the chain visibly breaks.
        </p>
      </Reveal>

      <Reveal delay={0.1}>
        <button
          onClick={handleVerify}
          disabled={verifying}
          className="mt-8 rounded bg-accent px-6 py-2 text-xs uppercase tracking-wide text-accent-foreground transition hover:opacity-90 disabled:opacity-40"
        >
          {verifying ? "Verifying…" : "Verify chain"}
        </button>
      </Reveal>

      {verifyError && <p className="mt-4 text-sm text-danger">{verifyError}</p>}

      {verifyResult && (
        <Reveal delay={0.05} className="mt-6">
          {verifyResult.valid ? (
            <p className="text-sm text-accent">✓ Chain valid — {verifyResult.n_entries} entries verified end to end.</p>
          ) : (
            <p className="text-sm text-danger">
              ✗ Chain broken at entry #{verifyResult.broken_at_id} — {verifyResult.reason}
            </p>
          )}
        </Reveal>
      )}

      <div className="mt-12">
        {entriesError && <ErrorPanel message={entriesError} onRetry={fetchEntries} />}
        {!entriesError && loading && <LoadingPanel label="Loading the chain…" />}
        {!entriesError && !loading && entries && entries.length === 0 && (
          <EmptyPanel message="No audit entries yet — open or review a loan in the explorer above to start the chain." />
        )}
        {!entriesError && entries && entries.length > 0 && (
          <ol>
            {entries.map((entry, i) => (
              <li key={entry.id}>
                {i > 0 && (
                  <div
                    ref={(el) => {
                      if (el) linkRefs.current.set(entry.id, el);
                    }}
                    className="ml-4 h-6 w-px bg-border transition-colors duration-300"
                  />
                )}
                <div
                  ref={(el) => {
                    if (el) nodeRefs.current.set(entry.id, el);
                  }}
                  className="flex items-start gap-4 rounded-lg border border-border bg-background-raised p-4 transition-[border-color,box-shadow] duration-300"
                >
                  <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border font-mono text-xs text-muted">
                    {entry.id}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-baseline justify-between gap-2">
                      <span className="text-sm font-medium text-foreground">
                        {EVENT_LABEL[entry.event_type] ?? entry.event_type}
                      </span>
                      {entry.loan_id && <span className="font-mono text-xs text-muted">{entry.loan_id}</span>}
                    </div>
                    <p className="mt-1 truncate font-mono text-[11px] text-muted" title={entry.entry_hash}>
                      hash {shortHash(entry.entry_hash)}
                    </p>
                    {entry.references_hash && (
                      <p className="truncate font-mono text-[11px] text-accent" title={entry.references_hash}>
                        → references {shortHash(entry.references_hash)}
                      </p>
                    )}
                  </div>
                </div>
              </li>
            ))}
          </ol>
        )}
      </div>
    </section>
  );
}
