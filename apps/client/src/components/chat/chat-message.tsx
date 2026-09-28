import { RefreshCw } from "lucide-react";
import { motion } from "framer-motion";

import { AgentMarkdown } from "@/components/agent-markdown";
import { InlineResults } from "@/components/chat/inline-results";
import {
  AgentActivity,
  FollowUps,
  MessageActions,
  PendingBubble,
} from "@/components/chat/message-parts";
import { StreamingText } from "@/components/chat/streaming-text";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { useShop } from "@/context/shop-context";
import { EASE, enterUp } from "@/lib/motion";
import { cn } from "@/lib/utils";
import type { ChatMessage, Product } from "@/types/ecommerce";

export function ChatMessageRow({
  message,
  highlighted,
  onOpenProduct,
}: {
  message: ChatMessage;
  highlighted: boolean;
  onOpenProduct?: (product: Product) => void;
}) {
  const { showDetail, retry } = useShop();

  if (message.role === "user") {
    return (
      <motion.div
        className="flex justify-end"
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.18, ease: EASE }}
      >
        <p className="max-w-md rounded-xl rounded-br-sm bg-muted px-4 py-2.5 text-sm text-foreground">
          {message.content}
        </p>
      </motion.div>
    );
  }

  const openCitation = (id: string) => {
    const match = message.products.find((product) => product.id === id);
    if (match) showDetail(match);
  };

  return (
    <motion.div
      id={`message-${message.id}`}
      variants={enterUp}
      initial="hidden"
      animate="visible"
      className={cn(
        "group space-y-3 rounded-lg transition-colors",
        highlighted && "bg-accent px-4 py-3"
      )}
    >
      {message.status === "pending" ? (
        message.content ? (
          // The live region is the streamed text only, so a screen reader
          // announces the answer being written and not the products after it.
          <div aria-live="polite">
            <StreamingText content={message.content} />
          </div>
        ) : (
          <PendingBubble />
        )
      ) : message.status === "error" ? (
        <Alert variant="destructive">
          <AlertTitle>The assistant could not answer</AlertTitle>
          <AlertDescription className="space-y-3">
            <p>{message.error}</p>
            <Button size="sm" variant="outline" onClick={retry}>
              <RefreshCw className="size-3.5" />
              Try again
            </Button>
          </AlertDescription>
        </Alert>
      ) : (
        <>
          <AgentMarkdown content={message.content} onCitation={openCitation} />
          {message.products.length > 0 ? <AgentActivity message={message} /> : null}
          {message.products.length > 0 ? (
            <InlineResults message={message} onOpen={onOpenProduct} />
          ) : null}
          <FollowUps products={message.products} />
          <MessageActions message={message} onRetry={retry} />
        </>
      )}
    </motion.div>
  );
}
