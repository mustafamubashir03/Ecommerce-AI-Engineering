import { useState } from "react";
import { Check, Copy, RefreshCw, ThumbsDown, ThumbsUp } from "lucide-react";

import { ProductCardSkeleton } from "@/components/products/product-card";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useShop } from "@/context/shop-context";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/types/ecommerce";

/** Copy, regenerate and feedback. Hidden until the row is hovered or focused. */
export function MessageActions({ message, onRetry }: { message: ChatMessage; onRetry: () => void }) {
  const [copied, setCopied] = useState(false);
  const [vote, setVote] = useState<"up" | "down" | null>(null);

  const action = (
    label: string,
    icon: React.ReactNode,
    onClick: () => void,
    tip: string,
    pressed?: boolean
  ) => (
    <Tooltip key={tip}>
      <TooltipTrigger
        render={
          <Button
            variant="ghost"
            size="icon"
            className="size-7"
            aria-label={label}
            aria-pressed={pressed}
            onClick={onClick}
          />
        }
      >
        {icon}
      </TooltipTrigger>
      <TooltipContent>{tip}</TooltipContent>
    </Tooltip>
  );

  return (
    <div className="flex items-center gap-1 text-muted-foreground opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
      {action(
        "Copy answer",
        copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />,
        () => {
          void navigator.clipboard.writeText(message.content);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        },
        copied ? "Copied" : "Copy answer"
      )}
      {action(
        "Ask the same question again",
        <RefreshCw className="size-3.5" />,
        onRetry,
        "Ask again"
      )}
      {action(
        "This answer was helpful",
        <ThumbsUp className={cn("size-3.5", vote === "up" && "text-accent-foreground")} />,
        () => setVote((prev) => (prev === "up" ? null : "up")),
        "Helpful",
        vote === "up"
      )}
      {action(
        "This answer was not helpful",
        <ThumbsDown className={cn("size-3.5", vote === "down" && "text-accent-foreground")} />,
        () => setVote((prev) => (prev === "down" ? null : "down")),
        "Not helpful",
        vote === "down"
      )}

      {message.requestId ? (
        <span className="ml-auto pr-1 font-mono text-xs">{message.requestId.slice(0, 8)}</span>
      ) : null}
    </div>
  );
}

/** At most three chips, offered after a reply. */
export function FollowUps({ products }: { products: ChatMessage["products"] }) {
  const { send, isBusy } = useShop();
  if (isBusy || products.length === 0) return null;

  const chips = [
    "Which of these is cheapest?",
    "Tell me more about the top rated one",
    "Show me something cheaper",
  ];

  return (
    <div className="flex flex-wrap gap-2">
      {chips.map((chip) => (
        <Button key={chip} variant="outline" size="sm" onClick={() => send(chip)}>
          {chip}
        </Button>
      ))}
    </div>
  );
}

/** One quiet line instead of a wall of tool chatter. */
export function AgentActivity({ message }: { message: ChatMessage }) {
  const count = message.products.length;

  return (
    <Collapsible>
      <CollapsibleTrigger className="text-xs text-muted-foreground hover:text-foreground">
        Agent retrieved {count} {count === 1 ? "product" : "products"}
        {message.requestId ? ` · request ${message.requestId.slice(0, 8)}` : ""}
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
          {message.products.map((product) => (
            <li key={product.id} className="flex items-center gap-2">
              <span className="font-mono">{product.id}</span>
              <span className="truncate">{product.title}</span>
            </li>
          ))}
        </ul>
      </CollapsibleContent>
    </Collapsible>
  );
}

export function PendingBubble() {
  return (
    <div className="space-y-4" role="status" aria-live="polite">
      <span className="sr-only">The assistant is searching the catalog</span>
      <div
        className="flex animate-pulse items-center gap-1.5 motion-reduce:animate-none"
        aria-hidden="true"
      >
        <span className="size-1.5 rounded-full bg-muted-foreground" />
        <span className="size-1.5 rounded-full bg-muted-foreground" />
        <span className="size-1.5 rounded-full bg-muted-foreground" />
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3" aria-hidden="true">
        <ProductCardSkeleton />
        <ProductCardSkeleton />
        <ProductCardSkeleton />
      </div>
    </div>
  );
}
