"use client";

import { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { EASE_OUT, prefersReducedMotion } from "@/lib/motion";

gsap.registerPlugin(ScrollTrigger);

interface CountUpProps {
  value: number;
  duration?: number;
  formatter?: (v: number) => string;
  className?: string;
  /** Wait until this element scrolls into view before counting up, instead
   * of firing as soon as the value is available. Off by default so the
   * pinned Portfolio Command hero (already in view on load) keeps counting
   * immediately. */
  triggerOnScroll?: boolean;
}

export default function CountUp({ value, duration = 1.4, formatter, className, triggerOnScroll = false }: CountUpProps) {
  const ref = useRef<HTMLSpanElement>(null);
  const lastValue = useRef<number | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const from = lastValue.current ?? 0;
    lastValue.current = value;

    if (prefersReducedMotion()) {
      ref.current.textContent = formatter ? formatter(value) : String(Math.round(value));
      return;
    }

    const obj = { v: from };
    let tween: gsap.core.Tween | null = null;
    const run = () => {
      tween = gsap.to(obj, {
        v: value,
        duration,
        ease: EASE_OUT,
        onUpdate: () => {
          if (ref.current) ref.current.textContent = formatter ? formatter(obj.v) : String(Math.round(obj.v));
        },
      });
    };

    if (!triggerOnScroll) {
      run();
      return () => {
        tween?.kill();
      };
    }

    const st = ScrollTrigger.create({
      trigger: ref.current,
      start: "top 88%",
      once: true,
      onEnter: run,
    });
    return () => {
      st.kill();
      tween?.kill();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return (
    <span ref={ref} className={className}>
      0
    </span>
  );
}
