import { Heart, Plus, Star } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";

import { AspectRatio } from "@/components/ui/aspect-ratio";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useShop } from "@/context/shop-context";
import { formatPrice, formatRating } from "@/lib/format";
import { enterUp, pressable } from "@/lib/motion";
import { cn } from "@/lib/utils";
import type { Product } from "@/types/ecommerce";

export function ProductCard({
  product,
  featured = false,
  compact = false,
  onOpen,
  className,
}: {
  product: Product;
  featured?: boolean;
  compact?: boolean;
  onOpen?: (product: Product) => void;
  className?: string;
}) {
  const { addToCart, toggleSaved, toggleCompare, saved, compare, showDetail, askAbout } = useShop();
  const reduced = useReducedMotion();
  const rating = formatRating(product.rating);
  const isSaved = saved.some((item) => item.id === product.id);
  const isCompared = compare.some((item) => item.id === product.id);
  const open = () => (onOpen ? onOpen(product) : showDetail(product));

  return (
    <motion.div
      variants={reduced ? undefined : enterUp}
      initial={reduced ? undefined : "hidden"}
      animate={reduced ? undefined : "visible"}
      className="h-full"
    >
      <Card className={cn("h-full gap-0 overflow-hidden border-border py-0 shadow-sm", className)}>
        <CardContent className="flex h-full flex-col gap-4 p-4">
          <motion.button
            type="button"
            onClick={open}
            whileTap={reduced ? undefined : { scale: 0.98 }}
            className="group relative block w-full overflow-hidden rounded-lg bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label={`View ${product.title}`}
          >
            <AspectRatio ratio={4 / 3}>
              {product.imageUrl ? (
                <img
                  src={product.imageUrl}
                  alt=""
                  loading="lazy"
                  className="size-full object-cover transition-transform duration-200 group-hover:scale-105 motion-reduce:transform-none motion-reduce:transition-none"
                />
              ) : (
                <span className="flex size-full items-center justify-center text-xs text-muted-foreground">
                  No image
                </span>
              )}
            </AspectRatio>
            {featured ? <Badge className="absolute left-2 top-2">Best pick</Badge> : null}
          </motion.button>

          <div className="space-y-1">
            <button
              type="button"
              onClick={open}
              className="line-clamp-2 w-full text-left text-sm font-medium text-foreground hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              {product.title}
            </button>

            {rating ? (
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Star className="size-3 fill-primary text-primary" aria-hidden="true" />
                <span className="font-medium text-foreground">{rating}</span>
                <span>average rating</span>
              </div>
            ) : null}

            <p className="text-base font-semibold text-foreground">{formatPrice(product.price)}</p>

            {!compact && product.reason ? (
              <p className="line-clamp-2 text-xs text-muted-foreground">{product.reason}</p>
            ) : null}
          </div>

          <div className="mt-auto flex flex-wrap items-center gap-1.5 pt-1">
            <motion.div variants={reduced ? undefined : pressable} whileTap={reduced ? undefined : "press"}>
              <Button
                size="sm"
                variant="outline"
                className="min-w-0 basis-full sm:basis-auto"
                onClick={() => addToCart(product)}
                disabled={product.price === null}
              >
                <Plus className="size-3.5" />
                Add to cart
              </Button>
            </motion.div>

            <Tooltip>
              <TooltipTrigger
                render={
                  <Button
                    size="icon"
                    variant="ghost"
                    onClick={() => toggleSaved(product)}
                    aria-label={isSaved ? "Remove from saved" : "Save for later"}
                    aria-pressed={isSaved}
                  />
                }
              >
                <motion.span
                  animate={reduced || !isSaved ? undefined : { scale: [1, 1.25, 1] }}
                  transition={reduced ? undefined : { duration: 0.25, ease: "easeOut" }}
                  className="flex"
                >
                  <Heart className={cn("size-4", isSaved && "fill-primary text-primary")} />
                </motion.span>
              </TooltipTrigger>
              <TooltipContent>{isSaved ? "Remove from saved" : "Save for later"}</TooltipContent>
            </Tooltip>

            <label className="flex cursor-pointer items-center gap-1.5 rounded-md px-1.5 py-1 text-xs text-muted-foreground hover:text-foreground">
              <Checkbox
                checked={isCompared}
                onCheckedChange={() => toggleCompare(product)}
                aria-label={`Compare ${product.title}`}
              />
              Compare
            </label>
          </div>

          <Button variant="link" size="sm" className="h-auto justify-start p-0" onClick={() => askAbout(product)}>
            Ask about this
          </Button>
        </CardContent>
      </Card>
    </motion.div>
  );
}

export function ProductCardSkeleton() {
  return (
    <Card className="gap-0 overflow-hidden border-border py-0 shadow-sm">
      <CardContent className="flex flex-col gap-4 p-4">
        <Skeleton className="aspect-4/3 w-full rounded-lg" />
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-3 w-1/3" />
        <Skeleton className="h-5 w-1/2" />
        <Skeleton className="h-8 w-full" />
      </CardContent>
    </Card>
  );
}
