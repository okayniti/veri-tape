/**
 * One shared motion config, used everywhere -- no per-component ad hoc
 * easing or durations. Every GSAP tween/ScrollTrigger in this app imports
 * from here.
 */
export const EASE = "power3.inOut";
export const EASE_OUT = "power3.out";

export const DURATION = {
  fast: 0.3,
  base: 0.6,
  slow: 1.1,
} as const;

/** Respect prefers-reduced-motion: callers should check this before
 * running any non-essential animation (loader, scroll reveals, pins). */
export function prefersReducedMotion(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
