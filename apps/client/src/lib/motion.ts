import type { Transition, Variants } from "framer-motion";
export const EASE = [0.22, 1, 0.36, 1] as const;
export const SNAPPY: Transition = { duration: 0.18, ease: EASE };
export const GENTLE: Transition = { duration: 0.24, ease: EASE };
export const enterUp: Variants = {
  hidden: { opacity: 0, y: 8 },
  visible: { opacity: 1, y: 0, transition: GENTLE },
};
export const enterFromRight: Variants = {
  hidden: { opacity: 0, x: 24 },
  visible: { opacity: 1, x: 0, transition: GENTLE },
  exit: { opacity: 0, x: 24, transition: SNAPPY },
};
export const pressable: Variants = {
  rest: { scale: 1 },
  hover: { scale: 1.02 },
  press: { scale: 0.97 },
};
export function stagger(each = 0.03, delay = 0): Variants {
  const timing = { staggerChildren: each, delayChildren: delay };
  return {
    hidden: { transition: timing },
    visible: { transition: timing },
  };
}
