"use client";

import { useEffect, useRef } from "react";
import gsap from "gsap";
import { DURATION, EASE, prefersReducedMotion } from "@/lib/motion";

/** Real pipeline actions the app actually performs somewhere -- hash-chain
 * verification, SHAP explainer construction, Platt calibration, anomaly
 * cross-referencing, the two prediction models, the portfolio fetch this
 * loader itself is waiting on. Plain and technical, not playful -- this is
 * a compliance/audit product. */
const MESSAGES = [
  "Fetching portfolio summary…",
  "Scoring loans with XGBoost…",
  "Running Bi-LSTM inference…",
  "Calibrating probabilities…",
  "Cross-referencing anomaly flags…",
  "Loading SHAP explainers…",
  "Verifying hash chain…",
];

const HOLD = 1.2;
const FADE = DURATION.fast * 0.5;

export default function LoaderTicker() {
  const ref = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    ref.current.textContent = MESSAGES[0];

    if (prefersReducedMotion()) return;

    const tl = gsap.timeline({ repeat: -1 });
    MESSAGES.forEach((_, i) => {
      tl.to({}, { duration: HOLD })
        .to(ref.current, { opacity: 0, duration: FADE, ease: EASE })
        .call(() => {
          if (ref.current) ref.current.textContent = MESSAGES[(i + 1) % MESSAGES.length];
        })
        .to(ref.current, { opacity: 1, duration: FADE, ease: EASE });
    });

    return () => {
      tl.kill();
    };
  }, []);

  return <p ref={ref} className="mt-4 h-4 text-[11px] text-muted" />;
}
