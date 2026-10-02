import { useState } from "react";
import { X } from "lucide-react";
import { motion } from "framer-motion";
import { CartTab } from "@/components/products/cart-tab";
import { CompareTable } from "@/components/products/compare-table";
import { ProductCard } from "@/components/products/product-card";
import { ProductDetail } from "@/components/products/product-detail";
import { PanelToolbar } from "@/components/products/panel-toolbar";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useShop } from "@/context/shop-context";
import { usePanelProducts } from "@/hooks/use-panel-products";
import { stagger } from "@/lib/motion";
import { cn } from "@/lib/utils";
const RESULT_GRID = stagger();
export function ProductPanel() {
  const { products, findProduct, saved, compare, panel, sort, setSort, openPanel, closePanel } = useShop();
  const [maxPrice, setMaxPrice] = useState(Number.POSITIVE_INFINITY);
  const [minRating, setMinRating] = useState(0);
  const [view, setView] = useState<"grid" | "list">("grid");
  const prices = products
    .map((product) => product.price)
    .filter((price): price is number => price !== null);
  const ceiling = prices.length > 0 ? Math.ceil(Math.max(...prices)) : 100;
  const results = usePanelProducts(
    products,
    sort,
    Number.isFinite(maxPrice) ? maxPrice : ceiling,
    minRating
  );
  const detail = panel.detailId ? findProduct(panel.detailId) : null;
  const comparing = compare.length > 1 && panel.tab === "results";
  if (detail) return <ProductDetail product={detail} />;
  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col border-l border-border bg-card">
      <div className="flex items-center justify-between gap-2 border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold text-foreground">
          {comparing ? "Compare" : "Products"}
        </h2>
        <Button
          variant="ghost"
          size="icon"
          className="size-8"
          onClick={closePanel}
          aria-label="Close products panel"
        >
          <X className="size-4" />
        </Button>
      </div>
      <Tabs
        value={panel.tab}
        onValueChange={(value) => openPanel(value as "results" | "saved" | "cart")}
        className="flex min-h-0 flex-1 flex-col"
      >
        <div className="border-b border-border px-4 py-2">
          <TabsList className="w-full">
            <TabsTrigger value="results" className="flex-1">
              Results
            </TabsTrigger>
            <TabsTrigger value="saved" className="flex-1">
              Saved {saved.length > 0 ? `(${saved.length})` : ""}
            </TabsTrigger>
            <TabsTrigger value="cart" className="flex-1">
              Cart
            </TabsTrigger>
          </TabsList>
        </div>
        <TabsContent value="results" className="min-h-0 flex-1">
          {products.length === 0 ? (
            <Empty
              title="No products yet"
              body="Ask the assistant a question and the products it used will appear here."
            />
          ) : (
            <div className="flex h-full min-h-0 flex-col">
              <div className="border-b border-border px-4 py-3">
                <PanelToolbar
                  count={results.length}
                  sort={sort}
                  onSortChange={setSort}
                  maxPrice={Number.isFinite(maxPrice) ? maxPrice : ceiling}
                  onMaxPriceChange={setMaxPrice}
                  priceCeiling={ceiling}
                  minRating={minRating}
                  onMinRatingChange={setMinRating}
                  view={view}
                  onViewChange={setView}
                />
              </div>
              <ScrollArea className="min-h-0 flex-1">
                {results.length === 0 ? (
                  <Empty title="No matches" body="Widen the filters to see more products." />
                ) : comparing ? (
                  <CompareTable />
                ) : (
                  <motion.div
                    variants={RESULT_GRID}
                    initial="hidden"
                    animate="visible"
                    className={cn(
                      "grid gap-4 p-4 [grid-template-columns:repeat(auto-fill,minmax(min(15rem,100%),1fr))]",
                      view === "list" && "flex flex-col"
                    )}
                  >
                    {results.map((product, index) => (
                      <ProductCard key={product.id} product={product} featured={index === 0} />
                    ))}
                  </motion.div>
                )}
              </ScrollArea>
            </div>
          )}
        </TabsContent>
        <TabsContent value="saved" className="min-h-0 flex-1">
          <ScrollArea className="h-full">
            {saved.length === 0 ? (
              <Empty title="Nothing saved" body="Use the heart on a product to keep it here." />
            ) : (
              <div className="grid grid-cols-1 gap-4 p-4 [grid-template-columns:repeat(auto-fill,minmax(min(15rem,100%),1fr))]">
                {saved.map((product) => (
                  <ProductCard key={product.id} product={product} />
                ))}
              </div>
            )}
          </ScrollArea>
        </TabsContent>
        <TabsContent value="cart" className="min-h-0 flex-1">
          <CartTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}
function Empty({ title, body }: { title: string; body: string }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 p-8 text-center">
      <p className="text-sm font-medium text-foreground">{title}</p>
      <p className="max-w-64 text-sm text-muted-foreground">{body}</p>
    </div>
  );
}
