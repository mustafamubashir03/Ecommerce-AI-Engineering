import { useEffect, useState } from "react";

/**
 * Whether a media query currently matches, and it keeps matching.
 *
 * Needed where a Tailwind `hidden` class is not enough: a component that
 * measures or animates itself, or a third party element that sets `display`
 * inline, will not respect a display utility. Mounting and unmounting is the
 * reliable way to take a component out of the layout entirely.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() =>
    typeof window === "undefined" ? false : window.matchMedia(query).matches
  );

  useEffect(() => {
    const list = window.matchMedia(query);
    const onChange = () => setMatches(list.matches);
    onChange();
    list.addEventListener("change", onChange);
    return () => list.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}

/** The desktop breakpoint, which is where the side panel replaces the drawer. */
export const DESKTOP = "(min-width: 1024px)";
