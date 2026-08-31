"use client";

import { useEffect, useRef, useState } from "react";
import gsap from "gsap";
import { DURATION, EASE, EASE_OUT, prefersReducedMotion } from "@/lib/motion";
import LoaderTicker from "./LoaderTicker";

const MIN_DISPLAY_MS = 2000;
const CLIMB_TARGET = 92;

/**
 * 0-100% progress, held at ~92% until `ready` (the real portfolio-summary
 * fetch, not a timer) is true, minimum 2s on screen regardless of how fast
 * the API responds, then a color-wipe into the app. Skips straight to
 * content if the user prefers reduced motion.
 */
export default function Loader({ ready, onDone }: { ready: boolean; onDone: () => void }) {
  const [hidden, setHidden] = useState(false);
  const percentRef = useRef<HTMLSpanElement>(null);
  const barRef = useRef<HTMLDivElement>(null);
  const counterWrapRef = useRef<HTMLDivElement>(null);
  const wipeRef = useRef<HTMLDivElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const mountedAt = useRef(Date.now());
  const readyRef = useRef(ready);
  readyRef.current = ready;
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  useEffect(() => {
    // Deliberately runs once (empty deps): reading callbacks via refs above,
    // not as effect dependencies. onDone is an inline arrow in the caller
    // and changes identity on every parent re-render (e.g. when the
    // portfolio-summary fetch that flips `ready` resolves) -- if it were a
    // dependency, that single expected re-render would tear down and
    // restart the whole climb/wipe sequence, which can leave the wipe
    // overlay's accent fill stuck on screen mid-animation.
    if (prefersReducedMotion()) {
      setHidden(true);
      onDoneRef.current();
      return;
    }

    const progress = { value: 0 };
    const setDisplay = (v: number) => {
      const rounded = Math.round(v);
      if (percentRef.current) percentRef.current.textContent = String(rounded);
      if (barRef.current) barRef.current.style.width = `${rounded}%`;
    };

    let finished = false;

    function finish() {
      if (finished) return;
      finished = true;
      gsap.ticker.remove(checkReady);

      const elapsed = Date.now() - mountedAt.current;
      const wait = Math.max(0, MIN_DISPLAY_MS - elapsed);

      window.setTimeout(() => {
        const tl = gsap.timeline({
          onComplete: () => {
            setHidden(true);
            onDoneRef.current();
          },
        });
        tl.to(progress, {
          value: 100,
          duration: DURATION.fast,
          ease: EASE_OUT,
          onUpdate: () => setDisplay(progress.value),
        })
          .to(counterWrapRef.current, { autoAlpha: 0, duration: DURATION.fast, ease: EASE }, "-=0.05")
          .fromTo(
            wipeRef.current,
            { scaleY: 0 },
            { scaleY: 1, duration: DURATION.base, ease: EASE, transformOrigin: "bottom" }
          )
          .to(rootRef.current, { scaleY: 0, duration: DURATION.slow, ease: EASE, transformOrigin: "top" }, "+=0.05");
      }, wait);
    }

    function checkReady() {
      if (readyRef.current) finish();
    }

    const climb = gsap.to(progress, {
      value: CLIMB_TARGET,
      duration: 1.6,
      ease: EASE_OUT,
      onUpdate: () => setDisplay(progress.value),
      onComplete: () => {
        if (readyRef.current) finish();
        else gsap.ticker.add(checkReady);
      },
    });

    return () => {
      climb.kill();
      gsap.ticker.remove(checkReady);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- see comment above: intentionally runs once
  }, []);

  if (hidden) return null;

  return (
    <div
      ref={rootRef}
      data-testid="loader-root"
      className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-background"
      aria-hidden={hidden}
    >
      <div ref={counterWrapRef} className="flex flex-col items-center">
        <div className="flex items-end gap-1 font-mono text-6xl font-light tracking-tight text-foreground md:text-8xl">
          <span ref={percentRef}>0</span>
          <span className="mb-2 text-2xl text-muted md:text-3xl">%</span>
        </div>
        <div className="mt-6 h-px w-48 overflow-hidden bg-border md:w-64">
          <div ref={barRef} className="h-full bg-accent" style={{ width: "0%" }} />
        </div>
        <p className="mt-6 text-xs uppercase tracking-[0.35em] text-muted">VeriTape</p>
        <LoaderTicker />
      </div>
      <div ref={wipeRef} className="pointer-events-none absolute inset-0 origin-bottom scale-y-0 bg-accent" />
    </div>
  );
}
