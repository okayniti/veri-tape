"use client";

import { scrollToId } from "@/lib/lenis";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const REPO_URL = "https://github.com/okayniti/veri-tape";

const PRODUCT_LINKS = [
  { id: "portfolio", label: "Portfolio" },
  { id: "loans", label: "Loan Explorer" },
  { id: "scenario", label: "Scenario Simulator" },
  { id: "audit", label: "Audit Trail" },
  { id: "faq", label: "FAQ" },
];

export default function Footer() {
  return (
    <footer className="border-t border-border px-6 py-16">
      <div className="mx-auto grid max-w-5xl grid-cols-2 gap-10 sm:grid-cols-4">
        <div>
          <p className="text-sm font-semibold text-foreground">VeriTape</p>
          <p className="mt-2 text-xs leading-relaxed text-muted">An auditable decision layer for loan servicing.</p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wide text-muted">Product</p>
          <ul className="mt-3 space-y-2">
            {PRODUCT_LINKS.map((l) => (
              <li key={l.id}>
                <button onClick={() => scrollToId(l.id)} className="text-xs text-muted transition hover:text-accent">
                  {l.label}
                </button>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wide text-muted">Resources</p>
          <ul className="mt-3 space-y-2 text-xs">
            <li>
              <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="text-muted transition hover:text-accent">
                GitHub Repository
              </a>
            </li>
            <li>
              <a href={`${API_URL}/docs`} target="_blank" rel="noopener noreferrer" className="text-muted transition hover:text-accent">
                API Docs
              </a>
            </li>
            <li>
              <a href={`${REPO_URL}#readme`} target="_blank" rel="noopener noreferrer" className="text-muted transition hover:text-accent">
                README
              </a>
            </li>
          </ul>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wide text-muted">Submission</p>
          <ul className="mt-3 space-y-2 text-xs">
            <li className="text-muted">AI Track — Intain FinTech Challenge 2026</li>
            <li>
              <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="text-muted transition hover:text-accent">
                Submission repository
              </a>
            </li>
          </ul>
        </div>
      </div>
      <p className="mx-auto mt-12 max-w-5xl border-t border-border pt-6 text-center text-[11px] uppercase tracking-[0.3em] text-muted">
        VeriTape — an auditable decision layer for loan servicing
      </p>
    </footer>
  );
}
