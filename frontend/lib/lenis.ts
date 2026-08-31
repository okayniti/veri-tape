import type Lenis from "lenis";

/** The single Lenis instance SmoothScrollProvider creates, so other
 * components (the nav's scroll links) can trigger a properly-eased scroll
 * through the same smooth-scroll system instead of a native scrollIntoView
 * that would jump independently of Lenis's own RAF loop. */
let instance: Lenis | null = null;

export function setLenisInstance(l: Lenis | null) {
  instance = l;
}

export function scrollToId(id: string) {
  const el = document.getElementById(id);
  if (!el) return;
  if (instance) {
    instance.scrollTo(el, { offset: -72, duration: 1.2 });
  } else {
    el.scrollIntoView({ behavior: "smooth" });
  }
}
