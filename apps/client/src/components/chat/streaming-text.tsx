import { motion, useReducedMotion } from "framer-motion";
import { cn } from "@/lib/utils";
export function StreamingText({ content, className }: { content: string; className?: string }) {
  const reduced = useReducedMotion();
  return (
    <div className={cn("text-sm leading-relaxed text-foreground", className)}>
      <p className="whitespace-pre-wrap break-words">{content}</p>
      <motion.span
        aria-hidden="true"
        className="mt-1 inline-block h-4 w-0.5 rounded-full bg-foreground align-text-bottom"
        animate={reduced ? undefined : { opacity: [1, 0.15, 1] }}
        transition={reduced ? undefined : { duration: 1.1, repeat: Infinity, ease: "easeInOut" }}
      />
    </div>
  );
}
