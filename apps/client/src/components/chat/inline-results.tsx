import { AnimatePresence, motion } from "framer-motion";
import { ProductScroller, ScrollerItem } from "@/components/chat/product-scroller";
import { Button } from "@/components/ui/button";
import { ProductCard } from "@/components/products/product-card";
import { useShop } from "@/context/shop-context";
import type { ChatMessage, Product } from "@/types/ecommerce";
const MAX_INLINE = 8;
export function InlineResults({
  message,
  onOpen,
}: {
  message: ChatMessage;
  onOpen?: (product: Product) => void;
}) {
  const { openPanel } = useShop();
  const visible = message.products.slice(0, MAX_INLINE);
  const hidden = message.products.length - visible.length;
  if (visible.length === 0) return null;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-xs font-medium text-muted-foreground">
          {message.products.length} {message.products.length === 1 ? "product" : "products"} from this answer
        </h3>
      </div>
      <ProductScroller label="Products used in this answer">
        {visible.map((product, index) => (
          <ScrollerItem key={product.id}>
            <ProductCard
              product={product}
              compact
              featured={index === 0}
              onOpen={onOpen}
              className="h-full"
            />
          </ScrollerItem>
        ))}
      </ProductScroller>
      <AnimatePresence initial={false}>
        {message.products.length > 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.18 }}
          >
            <Button
              variant="link"
              size="sm"
              className="h-auto justify-start p-0"
              onClick={() => openPanel("results")}
            >
              {hidden > 0
                ? `View all ${message.products.length} in panel`
                : "Browse all in panel"}
            </Button>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
