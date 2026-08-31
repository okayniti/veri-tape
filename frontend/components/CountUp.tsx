"use client";

import { useEffect, useRef } from "react";
import gsap from "gsap";
import { EASE_OUT, prefersReducedMotion } from "@/lib/motion";

interface CountUpProps {
  value: number;
  duration?: number;
  formatter?: (v: number) => string;
  className?: string;
}

export default function CountUp({ value, duration = 1.4, formatter, className }: CountUpProps) {
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
    const tween = gsap.to(obj, {
      v: value,
      duration,
      ease: EASE_OUT,
      onUpdate: () => {
        if (ref.current) ref.current.textContent = formatter ? formatter(obj.v) : String(Math.round(obj.v));
      },
    });
    return () => {
      tween.kill();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return (
    <span ref={ref} className={className}>
      0
    </span>
  );
}
