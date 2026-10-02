import { useState } from "react";
import { Check, Copy, MessageSquare, RefreshCw, ThumbsDown, ThumbsUp } from "lucide-react";
import { ProductCardSkeleton } from "@/components/products/product-card";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useShop } from "@/context/shop-context";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/types/ecommerce";
export function MessageActions({ message, onRetry }: { message: ChatMessage; onRetry: () => void }) {
  const { sendFeedback } = useShop();
  const [copied, setCopied] = useState(false);
  const [commentOpen, setCommentOpen] = useState(false);
  const [comment, setComment] = useState("");
  const [sending, setSending] = useState(false);
  const [sentComment, setSentComment] = useState(false);
  const vote = message.vote;
  const rateable = Boolean(message.traceId);
  const file = (feedback: { score?: number | null; text?: string }) => {
    setSending(true);
    void sendFeedback(message.id, feedback).finally(() => setSending(false));
  };
  const castVote = (next: "up" | "down") => {
    file({ score: next === "up" ? 1 : -1 });
  };
  const submitComment = () => {
    const text = comment.trim();
    if (!text) return;
    setCommentOpen(false);
    setComment("");
    setSentComment(true);
    file({ text });
  };
  const action = (
    label: string,
    icon: React.ReactNode,
    onClick: () => void,
    tip: string,
    pressed?: boolean,
    disabled?: boolean
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
            disabled={disabled}
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
    <div className="space-y-1.5">
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
          () => castVote("up"),
          "Helpful",
          vote === "up",
          sending || !rateable
        )}
        {action(
          "This answer was not helpful",
          <ThumbsDown className={cn("size-3.5", vote === "down" && "text-accent-foreground")} />,
          () => castVote("down"),
          "Not helpful",
          vote === "down",
          sending || !rateable
        )}
        {action(
          "Leave a comment about this answer",
          <MessageSquare className="size-3.5" />,
          () => setCommentOpen((open) => !open),
          "Add a comment",
          commentOpen,
          sending || !rateable
        )}
        {message.requestId ? (
          <span className="ml-auto pr-1 font-mono text-xs">{message.requestId.slice(0, 8)}</span>
        ) : null}
      </div>
      {!rateable ? (
        <p className="text-xs text-muted-foreground">
          This answer cannot be rated: it carries no trace to attach feedback to.
        </p>
      ) : null}
      {commentOpen ? (
        <div className="flex items-center gap-2">
          <Input
            value={comment}
            onChange={(event) => setComment(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") submitComment();
              if (event.key === "Escape") setCommentOpen(false);
            }}
            placeholder="What was wrong with this answer?"
            aria-label="Comment about this answer"
            className="h-8 text-sm"
            autoFocus
          />
          <Button
            size="sm"
            variant="secondary"
            onClick={submitComment}
            disabled={!comment.trim() || sending}
            aria-label="Send comment"
          >
            Send
          </Button>
        </div>
      ) : null}
      {message.feedbackError ? (
        <p role="status" className="text-xs text-destructive">
          {message.feedbackError}
        </p>
      ) : sentComment && !message.feedbackError ? (
        <p role="status" className="text-xs text-muted-foreground">
          Comment sent. Thank you.
        </p>
      ) : null}
    </div>
  );
}
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
