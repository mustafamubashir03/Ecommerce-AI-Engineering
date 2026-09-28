import { useState } from "react";
import {
  ArrowLeft,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Heart,
  Minus,
  Plus,
  Scale,
  ShoppingCart,
  Trash2,
} from "lucide-react";
import { motion } from "framer-motion";

import { AspectRatio } from "@/components/ui/aspect-ratio";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useShop } from "@/context/shop-context";
import { formatPrice, formatRating } from "@/lib/format";
import { EASE, enterUp } from "@/lib/motion";
import { cn } from "@/lib/utils";
import type { Product } from "@/types/ecommerce";

export function ProductDetail({ product }: { product: Product }) {
  const {
    hideDetail,
    showDetail,
    addToCart,
    cart,
    setQuantity,
    removeFromCart,
    askAbout,
    toggleSaved,
    toggleCompare,
    saved,
    compare,
    products,
  } = useShop();

  const rating = formatRating(product.rating);
  const line = cart.find((item) => item.product.id === product.id);
  const isSaved = saved.some((item) => item.id === product.id);
  const isCompared = compare.some((item) => item.id === product.id);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  // Derived, so moving to another product clears the confirmation without an
  // effect that would re-render on every product change.
  const copied = copiedId === product.id;

  // Where this product sits in the current result set, so the page can step
  // through them without going back to the panel first.
  const position = products.findIndex((item) => item.id === product.id);
  const previous = position > 0 ? products[position - 1] : null;
  const next = position >= 0 && position < products.length - 1 ? products[position + 1] : null;

  const copyId = async () => {
    try {
      await navigator.clipboard.writeText(product.id);
      setCopiedId(product.id);
    } catch {
      // Clipboard access can be refused; the id is on screen either way.
    }
  };

  return (
    <motion.div
      key={product.id}
      variants={enterUp}
      initial="hidden"
      animate="visible"
      className="flex h-full min-h-0 flex-col"
    >
      <div className="flex items-center gap-2 border-b border-border px-3 py-2.5">
        <Button variant="ghost" size="icon" className="size-8" onClick={hideDetail} aria-label="Back to results">
          <ArrowLeft className="size-4" />
        </Button>

        <div className="min-w-0 flex-1">
          <p className="truncate text-xs text-muted-foreground">
            {position >= 0 && products.length > 1
              ? `Product ${position + 1} of ${products.length}`
              : "Product detail"}
          </p>
        </div>

        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="size-8"
            disabled={!previous}
            onClick={() => previous && showDetail(previous)}
            aria-label="Previous product"
          >
            <ChevronLeft className="size-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="size-8"
            disabled={!next}
            onClick={() => next && showDetail(next)}
            aria-label="Next product"
          >
            <ChevronRight className="size-4" />
          </Button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {/* Two columns once there is room, so a narrow phone panel gets one
            readable column and a wide desktop panel does not stretch a photo
            across the whole window. */}
        <div className="grid gap-5 p-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:items-start">
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.22, ease: EASE }}
            className="overflow-hidden rounded-xl bg-muted"
          >
            <AspectRatio ratio={4 / 3}>
              {product.imageUrl ? (
                <img
                  src={product.imageUrl}
                  alt=""
                  className="size-full object-cover"
                  loading="lazy"
                />
              ) : (
                <span className="flex size-full items-center justify-center text-xs text-muted-foreground">
                  No image
                </span>
              )}
            </AspectRatio>
          </motion.div>

          <div className="min-w-0 space-y-4">
            <div className="space-y-2">
              <div className="flex items-start justify-between gap-3">
                <h2 className="text-xl leading-snug font-semibold text-foreground">
                  {product.title}
                </h2>
                {line ? <Badge variant="secondary">In cart &times;{line.quantity}</Badge> : null}
              </div>

              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <span className="text-2xl font-semibold text-foreground">
                  {formatPrice(product.price)}
                </span>
                {rating ? (
                  <span className="flex items-center gap-1 text-sm text-muted-foreground">
                    <span className="flex" aria-hidden="true">
                      {Array.from({ length: 5 }, (_, index) => (
                        <span
                          key={index}
                          className={cn(
                            "size-3.5 rounded-[2px]",
                            index < Math.round(Number(rating))
                              ? "bg-primary"
                              : "bg-muted-foreground/30"
                          )}
                        />
                      ))}
                    </span>
                    {rating} out of 5
                  </span>
                ) : (
                  <span className="text-sm text-muted-foreground">No rating recorded</span>
                )}
              </div>
            </div>

            {product.reason ? (
              <div className="space-y-1.5 rounded-lg border border-border bg-accent/50 px-3 py-2.5">
                <h3 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
                  Why the agent picked this
                </h3>
                <p className="text-sm text-foreground">{product.reason}</p>
              </div>
            ) : null}

            <div className="flex flex-wrap items-center gap-2">
              <Button
                variant={isSaved ? "secondary" : "outline"}
                size="sm"
                onClick={() => toggleSaved(product)}
                aria-pressed={isSaved}
              >
                <motion.span
                  animate={isSaved ? { scale: [1, 1.2, 1] } : { scale: 1 }}
                  transition={{ duration: 0.22, ease: EASE }}
                  className="flex"
                >
                  <Heart className={cn("size-3.5", isSaved && "fill-primary text-primary")} />
                </motion.span>
                {isSaved ? "Saved" : "Save"}
              </Button>

              <Button
                variant={isCompared ? "secondary" : "outline"}
                size="sm"
                onClick={() => toggleCompare(product)}
                aria-pressed={isCompared}
              >
                <Scale className="size-3.5" />
                {isCompared ? "Comparing" : "Compare"}
              </Button>

              <Tooltip>
                <TooltipTrigger
                  render={
                    <Button variant="outline" size="sm" onClick={copyId} aria-label="Copy product ID" />
                  }
                >
                  <Copy className="size-3.5" />
                  {copied ? "Copied" : "Copy ID"}
                </TooltipTrigger>
                <TooltipContent>{product.id}</TooltipContent>
              </Tooltip>
            </div>

            <Separator />

            <div className="space-y-2 text-sm">
              <h3 className="font-semibold text-foreground">Catalog record</h3>
              <dl className="grid grid-cols-[auto_1fr] gap-y-2 text-muted-foreground">
                <dt>Product ID</dt>
                <dd className="truncate text-right font-mono text-foreground">{product.id}</dd>
                <dt>Price</dt>
                <dd className="text-right text-foreground">{formatPrice(product.price)}</dd>
                <dt>Rating</dt>
                <dd className="text-right text-foreground">{rating ?? "Not recorded"}</dd>
              </dl>
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 border-t border-border p-3">
        {line ? (
          <>
            <div className="flex items-center gap-1 rounded-lg border border-border">
              <Button
                variant="ghost"
                size="icon"
                className="size-8"
                aria-label="Decrease quantity"
                onClick={() => setQuantity(product.id, line.quantity - 1)}
              >
                <Minus className="size-3.5" />
              </Button>
              <span className="w-6 text-center text-sm tabular-nums">{line.quantity}</span>
              <Button
                variant="ghost"
                size="icon"
                className="size-8"
                aria-label="Increase quantity"
                onClick={() => setQuantity(product.id, line.quantity + 1)}
              >
                <Plus className="size-3.5" />
              </Button>
            </div>
            <Button
              variant="outline"
              size="icon"
              className="size-9"
              aria-label="Remove from cart"
              onClick={() => removeFromCart(product.id)}
            >
              <Trash2 className="size-4" />
            </Button>
            <span className="ml-auto inline-flex items-center gap-1 text-sm text-muted-foreground">
              <Check className="size-4" />
              In cart
            </span>
          </>
        ) : (
          <>
            <Button
              className="flex-1"
              onClick={() => addToCart(product)}
              disabled={product.price === null}
            >
              <ShoppingCart className="size-4" />
              {product.price === null ? "Price unavailable" : "Add to cart"}
            </Button>
            <Button variant="outline" onClick={() => askAbout(product)}>
              Ask about this
            </Button>
          </>
        )}
      </div>
    </motion.div>
  );
}
