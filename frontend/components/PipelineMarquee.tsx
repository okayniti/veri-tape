"use client";

import { useEffect, useRef } from "react";
import gsap from "gsap";
import { prefersReducedMotion } from "@/lib/motion";

/** The judged-criteria checklist, doubling as a status strip: every stage
 * this pipeline actually runs, in order. Not partner logos -- there are no
 * partners here, just the pipeline itself. */
const STAGES = [
  "Data Profiling",
  "Time-Aware Validation",
  "XGBoost + Bi-LSTM",
  "Anomaly Detection",
  "Calibration",
  "SHAP Explainability",
  "Scenario Simulation",
  "LLM Narration",
  "Hash-Chained Audit",
  "Reviewer Loop",
];

export default function PipelineMarquee() {
  const trackRef = useRef<HTMLDivElement>(null);
  const tweenRef = useRef<gsap.core.Tween | null>(null);

  useEffect(() => {
    if (!trackRef.current || prefersReducedMotion()) return;
    const tween = gsap.to(trackRef.current, {
      xPercent: -50,
      duration: 34,
      ease: "none",
      repeat: -1,
    });
    tweenRef.current = tween;
    return () => {
      tween.kill();
    };
  }, []);

  return (
    <div className="overflow-hidden border-y border-border py-3" aria-label="Pipeline stages">
      <div
        ref={trackRef}
        className="flex w-max gap-3"
        onMouseEnter={() => tweenRef.current?.pause()}
        onMouseLeave={() => tweenRef.current?.resume()}
      >
        {[...STAGES, ...STAGES].map((stage, i) => (
          <span
            key={i}
            className="flex shrink-0 items-center gap-2 whitespace-nowrap rounded-full border border-border px-4 py-1.5 text-xs uppercase tracking-wide text-muted"
          >
            <span className="inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
            {stage}
          </span>
        ))}
      </div>
    </div>
  );
}
