"use client";

import { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { DURATION, EASE, prefersReducedMotion } from "@/lib/motion";

gsap.registerPlugin(ScrollTrigger);

/** Fades + lifts children into view once scrolled near the viewport.
 * Stagger sibling Reveals with the `delay` prop -- no per-instance easing. */
export default function Reveal({
  children,
  delay = 0,
  y = 24,
  className,
  as: Tag = "div",
}: {
  children: React.ReactNode;
  delay?: number;
  y?: number;
  className?: string;
  as?: "div" | "li" | "section";
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;

    if (prefersReducedMotion()) {
      gsap.set(ref.current, { opacity: 1, y: 0 });
      return;
    }

    const ctx = gsap.context(() => {
      gsap.fromTo(
        ref.current,
        { opacity: 0, y },
        {
          opacity: 1,
          y: 0,
          duration: DURATION.base,
          ease: EASE,
          delay,
          scrollTrigger: {
            trigger: ref.current,
            start: "top 88%",
            toggleActions: "play none none reverse",
          },
        }
      );
    });
    return () => ctx.revert();
  }, [delay, y]);

  const Component = Tag as React.ElementType;
  return (
    <Component ref={ref} className={className}>
      {children}
    </Component>
  );
}
