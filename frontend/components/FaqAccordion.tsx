"use client";

import { useRef, useState } from "react";
import gsap from "gsap";
import { DURATION, EASE, prefersReducedMotion } from "@/lib/motion";
import Reveal from "./Reveal";

const FAQS = [
  {
    q: "What happens if the LLM narration service is down?",
    a: "Nothing else stops working. review/narrate.py wraps every Gemini call in a try/except: an outage, rate limit, or empty response falls back to a deterministic template that still states the calibrated probability and top SHAP driver, clearly labeled as a fallback so it's never mistaken for a real narration. Prediction, calibration, SHAP, and the audit trail are all computed before the LLM is ever called, and none of them depend on it succeeding.",
  },
  {
    q: "How do you know the audit trail hasn't been tampered with?",
    a: "Every entry's hash is SHA-256 of the previous entry's hash plus its own timestamp, event type, and payload -- so editing any historical row, even just its stored hash, breaks every link after it. The Verify Chain button above calls the live hash-chain verification endpoint and walks the whole chain from genesis. This was tested against a real tampered row (edited directly in the database, bypassing the app entirely) and correctly reported exactly which entry broke.",
  },
  {
    q: "Does the model make the final call, or does a human?",
    a: "A human does. Flagged loans -- a top-1% structural anomaly, or a calibrated probability in the \"high\" risk tier -- go through the reviewer decision loop: a person accepts or overrides the flag, optionally with a reason. An override never changes the stored prediction or anomaly score -- it's a permanent, hash-chained annotation on top of an immutable model output, not a correction to it.",
  },
  {
    q: "How was this validated?",
    a: "Trained and tested on a time-aware split by origination date, not a random one, so the held-out set reflects loans the model has never seen and mirrors real deployment. On that split: XGBoost reaches 0.78 AUC and a Bi-LSTM over payment sequences reaches 0.86, both against simpler baselines; the anomaly detector shows a 6.6x-8x precision lift over a random baseline; Platt scaling improves the Brier score by 44%.",
  },
];

export default function FaqAccordion() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);
  const panelRefs = useRef<(HTMLDivElement | null)[]>([]);

  function closePanel(el: HTMLDivElement) {
    if (prefersReducedMotion()) {
      el.style.height = "0px";
      return;
    }
    const current = el.scrollHeight;
    gsap.fromTo(el, { height: current }, { height: 0, duration: DURATION.base, ease: EASE });
  }

  function openPanel(el: HTMLDivElement) {
    const target = el.scrollHeight;
    if (prefersReducedMotion()) {
      el.style.height = "auto";
      return;
    }
    gsap.fromTo(
      el,
      { height: 0 },
      { height: target, duration: DURATION.base, ease: EASE, onComplete: () => { el.style.height = "auto"; } }
    );
  }

  function toggle(i: number) {
    const previousIndex = openIndex;
    const willOpen = previousIndex !== i;
    setOpenIndex(willOpen ? i : null);

    if (previousIndex !== null && previousIndex !== i) {
      const prevEl = panelRefs.current[previousIndex];
      if (prevEl) closePanel(prevEl);
    }

    const el = panelRefs.current[i];
    if (!el) return;
    if (willOpen) openPanel(el);
    else closePanel(el);
  }

  return (
    <section id="faq" className="mx-auto max-w-3xl px-6 py-24">
      <Reveal>
        <h2 className="text-xs uppercase tracking-[0.3em] text-muted">FAQ</h2>
      </Reveal>
      <Reveal delay={0.05}>
        <p className="mt-2 max-w-xl text-2xl text-foreground">Questions a compliance or servicing team would actually ask.</p>
      </Reveal>

      <div className="mt-10 divide-y divide-border border-y border-border">
        {FAQS.map((item, i) => (
          <Reveal key={item.q} delay={i * 0.05}>
            <button
              onClick={() => toggle(i)}
              className="flex w-full items-center justify-between gap-4 py-5 text-left"
              aria-expanded={openIndex === i}
            >
              <span className="text-sm font-medium text-foreground">{item.q}</span>
              <span
                className={`shrink-0 text-lg text-accent transition-transform duration-300 ${openIndex === i ? "rotate-45" : ""}`}
                aria-hidden
              >
                +
              </span>
            </button>
            <div
              ref={(el) => {
                panelRefs.current[i] = el;
              }}
              data-testid={`faq-panel-${i}`}
              style={{ height: 0, overflow: "hidden" }}
            >
              <p className="pb-5 text-sm leading-relaxed text-muted">{item.a}</p>
            </div>
          </Reveal>
        ))}
      </div>
    </section>
  );
}
