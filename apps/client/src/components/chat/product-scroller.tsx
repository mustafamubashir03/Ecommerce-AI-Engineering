import { useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { motion } from "framer-motion";

import { Button } from "@/components/ui/button";
import { GENTLE, enterUp } from "@/lib/motion";
import { cn } from "@/lib/utils";

/** Which ends of the row still have somewhere to scroll to. */
type Edges = { start: boolean; end: boolean };

/**
 * A row of cards that scrolls sideways and clips at both ends.
 *
 * Native overflow and scroll snapping rather than a carousel library, because
 * this row has to feel right under a thumb: the platform's own momentum,
 * rubber banding and keyboard handling come for free, and it still scrolls
 * with JavaScript disabled. The ends are faded and the arrows only appear when
 * there is somewhere to scroll to, so the row never advertises a scroll it
 * cannot do.
 */
export function ProductScroller({
  label,
  children,
  className,
}: {
  label: string;
  children: React.ReactNode;
  className?: string;
}) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState<Edges>({ start: false, end: false });

  useEffect(() => {
    const element = viewportRef.current;
    if (!element) return;

    const readEdges = () => {
      const max = element.scrollWidth - element.clientWidth;
      // A pixel of slack, because fractional widths make a settled row look
      // scrollable by a fraction of a pixel and would keep an arrow enabled
      // that can no longer move anything.
      const next: Edges = {
        start: element.scrollLeft > 1,
        end: element.scrollLeft < max - 1,
      };
      setEdges((prev) =>
        prev.start === next.start && prev.end === next.end ? prev : next
      );
    };

    readEdges();

    // The content is observed as well as the viewport. Watching only the
    // viewport is not enough: the cards lay out after the row mounts, so
    // `scrollWidth` grows while the row's own box stays the same size, and the
    // arrows would never appear.
    const observer = new ResizeObserver(readEdges);
    observer.observe(element);
    if (element.firstElementChild) observer.observe(element.firstElementChild);

    element.addEventListener("scroll", readEdges, { passive: true });
    return () => {
      element.removeEventListener("scroll", readEdges);
      observer.disconnect();
    };
  }, [children]);

  const nudge = (direction: 1 | -1) => {
    const element = viewportRef.current;
    if (!element) return;
    element.scrollBy({ left: direction * element.clientWidth * 0.8, behavior: "smooth" });
  };

  return (
    <div className={cn("relative", className)}>
      <div
        ref={viewportRef}
        role="group"
        aria-label={label}
        tabIndex={0}
        className={cn(
          "flex snap-x snap-mandatory gap-3 overflow-x-auto overscroll-x-contain",
          // A little room so a card's shadow is not clipped, and the scrollbar
          // itself is hidden because the fades are the affordance.
          "px-0.5 pb-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
        )}
      >
        {children}
      </div>

      {/* Edge fades. Pointer events stay off so the row underneath still drags. */}
      <div
        aria-hidden="true"
        className={cn(
          "pointer-events-none absolute inset-y-0 left-0 w-8 bg-gradient-to-r from-background to-transparent transition-opacity duration-200",
          edges.start ? "opacity-100" : "opacity-0"
        )}
      />
      <div
        aria-hidden="true"
        className={cn(
          "pointer-events-none absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-background to-transparent transition-opacity duration-200",
          edges.end ? "opacity-100" : "opacity-0"
        )}
      />

      {edges.start || edges.end ? (
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center">
          <motion.div
            initial={{ opacity: 0, x: 4 }}
            animate={{ opacity: 1, x: 0 }}
            transition={GENTLE}
            className="pointer-events-auto flex"
          >
            {edges.start ? (
              <Button
                size="icon"
                variant="secondary"
                className="mr-1 size-7 rounded-full shadow-sm"
                onClick={() => nudge(-1)}
                aria-label="Scroll products left"
              >
                <ChevronLeft className="size-4" />
              </Button>
            ) : null}
            {edges.end ? (
              <Button
                size="icon"
                variant="secondary"
                className="size-7 rounded-full shadow-sm"
                onClick={() => nudge(1)}
                aria-label="Scroll products right"
              >
                <ChevronRight className="size-4" />
              </Button>
            ) : null}
          </motion.div>
        </div>
      ) : null}
    </div>
  );
}

/** One card's width in the row: most of a phone, two on a tablet, three above. */
export function ScrollerItem({ className, ...props }: React.ComponentProps<typeof motion.div>) {
  return (
    <motion.div
      variants={enterUp}
      className={cn("w-[78%] shrink-0 snap-start sm:w-[46%] lg:w-[31%]", className)}
      {...props}
    />
  );
}
