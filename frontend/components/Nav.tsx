"use client";

import { scrollToId } from "@/lib/lenis";

const LINKS = [
  { id: "portfolio", label: "Portfolio" },
  { id: "loans", label: "Explorer" },
  { id: "audit", label: "Audit Trail" },
];

export default function Nav() {
  return (
    <nav className="fixed inset-x-0 top-0 z-30 border-b border-border bg-background/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <button onClick={() => scrollToId("portfolio")} className="text-sm font-semibold tracking-wide text-foreground">
          VeriTape
        </button>
        <div className="hidden items-center gap-8 text-xs uppercase tracking-wide text-muted sm:flex">
          {LINKS.map((link) => (
            <button key={link.id} onClick={() => scrollToId(link.id)} className="transition hover:text-accent">
              {link.label}
            </button>
          ))}
        </div>
        <a
          href="https://github.com/okayniti/veri-tape"
          target="_blank"
          rel="noopener noreferrer"
          className="rounded-md bg-accent px-4 py-2 text-xs uppercase tracking-wide text-accent-foreground transition hover:bg-accent-hover"
        >
          View on GitHub
        </a>
      </div>
    </nav>
  );
}
