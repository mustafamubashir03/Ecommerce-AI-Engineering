import { useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { GENTLE, enterUp } from "@/lib/motion";
import { cn } from "@/lib/utils";
type Edges = { start: boolean; end: boolean };
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
      const next: Edges = {
        start: element.scrollLeft > 1,
        end: element.scrollLeft < max - 1,
      };
      setEdges((prev) =>
        prev.start === next.start && prev.end === next.end ? prev : next
      );
    };
    readEdges();
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
          "px-0.5 pb-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
        )}
      >
        {children}
      </div>
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
export function ScrollerItem({ className, ...props }: React.ComponentProps<typeof motion.div>) {
  return (
    <motion.div
      variants={enterUp}
      className={cn("w-[78%] shrink-0 snap-start sm:w-[46%] lg:w-[31%]", className)}
      {...props}
    />
  );
}
