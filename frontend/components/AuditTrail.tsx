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
  const pendingAnimationRef = useRef(false);

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
      // Entries fetched here can include rows added since this section's
      // last render (e.g. from viewing a loan or running a scenario
      // elsewhere on the page) -- their <li> nodes don't exist in the DOM
      // yet, so refs for them aren't populated until React commits this
      // state update. Flag it and let the effect below (which runs after
      // commit) build the animation once every ref is guaranteed to exist.
      pendingAnimationRef.current = true;
      setEntries(freshEntries);
      setVerifyResult(res);
    } catch (e) {
      setVerifyError(e instanceof Error ? e.message : String(e));
    } finally {
      setVerifying(false);
    }
  }

  useEffect(() => {
    if (!pendingAnimationRef.current || !verifyResult || !entries) return;
    pendingAnimationRef.current = false;

    const brokenIndex = verifyResult.valid ? -1 : entries.findIndex((e) => e.id === verifyResult.broken_at_id);

    const tl = gsap.timeline();
    entries.forEach((entry, i) => {
      if (!verifyResult.valid && brokenIndex >= 0 && i > brokenIndex) return; // animation halts at the break

      const isBroken = !verifyResult.valid && i === brokenIndex;
      const link = linkRefs.current.get(entry.id);
      const node = nodeRefs.current.get(entry.id);

      if (link) {
        tl.to(link, { backgroundColor: isBroken ? "var(--danger)" : "var(--success)", duration: 0.25 }, i * 0.12);
      }
      if (node) {
        tl.to(
          node,
          {
            borderColor: isBroken ? "var(--danger)" : "var(--success)",
            boxShadow: isBroken ? "0 0 0 4px var(--shadow-danger-glow)" : "0 0 0 4px var(--shadow-success-glow)",
            duration: 0.25,
          },
          i * 0.12
        );
      }
    });
  }, [entries, verifyResult]);

  return (
    <section id="audit" className="mx-auto max-w-3xl px-6 py-28">
      <Reveal>
        <h2 className="text-xs uppercase tracking-[0.3em] text-muted">Audit Trail</h2>
      </Reveal>
      <Reveal delay={0.05}>
        <p className="mt-2 max-w-xl text-3xl text-foreground sm:text-4xl">
          Every model decision, hash-chained. Tamper with any entry and the chain visibly breaks.
        </p>
      </Reveal>

      <Reveal delay={0.1}>
        <button
          onClick={handleVerify}
          disabled={verifying}
          className="mt-8 rounded-md bg-accent px-6 py-2 text-xs uppercase tracking-wide text-accent-foreground transition hover:bg-accent-hover disabled:opacity-40"
        >
          {verifying ? "Verifying…" : "Verify chain"}
        </button>
      </Reveal>

      {verifyError && <p className="mt-4 text-sm text-danger">{verifyError}</p>}

      {verifyResult && (
        <Reveal delay={0.05} className="mt-6">
          {verifyResult.valid ? (
            <p className="text-sm text-success">✓ Chain valid — {verifyResult.n_entries} entries verified end to end.</p>
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
          <ol data-testid="gallery-audit-chain">
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
                  className="flex items-start gap-4 rounded-xl border border-border bg-background-raised p-4 transition-[border-color,box-shadow] duration-300"
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
