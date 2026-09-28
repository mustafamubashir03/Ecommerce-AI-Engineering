import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowDown, PanelRight } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";

import { ChatComposer } from "@/components/chat/chat-composer";
import { ChatEmptyState } from "@/components/chat/chat-empty-state";
import { ChatMessageRow } from "@/components/chat/chat-message";
import { Button } from "@/components/ui/button";
import type { Product } from "@/types/ecommerce";
import { useShop } from "@/context/shop-context";
import { DESKTOP, useMediaQuery } from "@/hooks/use-media-query";
import { EASE } from "@/lib/motion";

const PIN_THRESHOLD = 64;

export function ChatView({
  onOpenProduct,
  onShowResults,
}: {
  onOpenProduct?: (product: Product) => void;
  onShowResults?: () => void;
}) {
  const { activeConversation, products, isBusy, openPanel } = useShop();
  const messages = activeConversation.messages;
  const viewportRef = useRef<HTMLDivElement>(null);
  const [pinned, setPinned] = useState(true);
  const isDesktop = useMediaQuery(DESKTOP);

  const scrollToLatest = useCallback((behavior: ScrollBehavior = "smooth") => {
    const element = viewportRef.current;
    if (element) element.scrollTo({ top: element.scrollHeight, behavior });
  }, []);

  useEffect(() => {
    if (pinned) scrollToLatest("auto");
  }, [messages, isBusy, pinned, scrollToLatest]);

  const onScroll = (event: React.UIEvent<HTMLDivElement>) => {
    const element = event.currentTarget;
    const distance = element.scrollHeight - element.scrollTop - element.clientHeight;
    setPinned(distance < PIN_THRESHOLD);
  };

  return (
    // h-full, not flex-1: the resizable panel is a plain element, not a flex
    // container, so a flex child here would grow to the height of the messages
    // and push the composer off the bottom of the screen. The message list has
    // to scroll inside a bounded column for the composer to stay put.
    <div className="flex h-full min-h-0 flex-col">
      {messages.length === 0 ? (
        <div className="flex min-h-0 flex-1 flex-col justify-center">
          <div className="mx-auto w-full max-w-3xl space-y-6 px-4 py-6">
            <ChatEmptyState />
            <ChatComposer embedded />
          </div>
        </div>
      ) : (
        <>
          <div className="relative min-h-0 flex-1">
            <div ref={viewportRef} className="size-full overflow-y-auto" onScroll={onScroll}>
              <div className="mx-auto w-full max-w-3xl px-4 py-6">
                <div className="space-y-6" aria-live="polite" aria-label="Conversation">
                  {messages.map((message) => (
                    <ChatMessageRow
                      key={message.id}
                      message={message}
                      onOpenProduct={onOpenProduct}
                      highlighted={products.some(
                        (product) => product.sourceMessageId === message.id
                      )}
                    />
                  ))}
                </div>
              </div>
            </div>

            {!pinned ? (
              <div className="pointer-events-none absolute inset-x-0 bottom-4 flex justify-center">
                <Button
                  size="sm"
                  variant="outline"
                  className="pointer-events-auto shadow-md"
                  onClick={() => {
                    setPinned(true);
                    scrollToLatest();
                  }}
                >
                  <ArrowDown className="size-3.5" />
                  Jump to latest
                </Button>
              </div>
            ) : null}
          </div>

          {/* On a phone the results live in a sheet, so this is the way in. It
              sits in the column above the composer rather than floating over
              it, which is what would otherwise cover the input on a short
              screen. */}
          <AnimatePresence initial={false}>
            {products.length > 0 && !isDesktop ? (
              <motion.div
                key="mobile-results"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 8 }}
                transition={{ duration: 0.18, ease: EASE }}
                className="border-t border-border bg-background px-4 pt-2"
              >
                <Button
                  variant="secondary"
                  className="w-full"
                  onClick={() => (onShowResults ? onShowResults() : openPanel("results"))}
                >
                  <PanelRight className="size-4" />
                  {products.length} {products.length === 1 ? "result" : "results"}
                </Button>
              </motion.div>
            ) : null}
          </AnimatePresence>

          <ChatComposer />
        </>
      )}
    </div>
  );
}
