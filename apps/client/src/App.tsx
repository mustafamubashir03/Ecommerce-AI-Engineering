import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

import { TopBar } from "@/components/top-bar";
import { AppSidebar } from "@/components/app-sidebar";
import { ChatView } from "@/components/chat/chat-view";
import { CompareTable } from "@/components/products/compare-table";
import { ProductDetail } from "@/components/products/product-detail";
import { ProductPanel } from "@/components/products/product-panel";
import { Drawer, DrawerContent } from "@/components/ui/drawer";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { SidebarInset, SidebarProvider, useSidebar } from "@/components/ui/sidebar";
import { ShopProvider, useShop } from "@/context/shop-context";
import { DESKTOP, useMediaQuery } from "@/hooks/use-media-query";
import { enterFromRight } from "@/lib/motion";
import type { Product } from "@/types/ecommerce";

/**
 * Collapses the sidebar to its icon rail on narrow screens.
 *
 * Expanded, the sidebar is 16rem, which is a third of a tablet and leaves the
 * conversation too narrow to read. It has to live inside the provider to reach
 * the toggle. The effect only runs when the breakpoint changes, so a sidebar
 * the user expands on purpose afterwards is left alone.
 */
function CollapseSidebarOnNarrow() {
  const isDesktop = useMediaQuery(DESKTOP);
  const { setOpen } = useSidebar();

  useEffect(() => {
    if (!isDesktop) setOpen(false);
  }, [isDesktop, setOpen]);

  return null;
}

function Workspace() {
  const { products, compare, panel, showDetail } = useShop();
  const [drawerProduct, setDrawerProduct] = useState<Product | null>(null);
  // On a phone the results are a sheet, and a sheet that opens by itself covers
  // the answer the user just asked for. It opens on request there instead, so
  // the conversation and the inline row stay readable.
  const [mobileSheetOpen, setMobileSheetOpen] = useState(false);

  // The side panel is mounted only where there is room for it. On a phone a
  // resizable panel would be a sliver, so the results are a sheet instead.
  const isDesktop = useMediaQuery(DESKTOP);
  const hasResults = products.length > 0;
  const showSidePanel = isDesktop && hasResults && panel.open;

  /** Tapping a card opens the panel on desktop, a drawer on small screens. */
  const openProduct = (product: Product) => {
    if (isDesktop) showDetail(product);
    else setDrawerProduct(product);
  };

  return (
    <SidebarProvider>
      <CollapseSidebarOnNarrow />
      <AppSidebar />
      <SidebarInset className="min-h-0">
        {/* h-dvh, not h-screen: on a phone the browser chrome changes the
            usable height, and the composer has to stay above it. */}
        <div className="flex h-dvh flex-col overflow-hidden">
          <TopBar />

          <ResizablePanelGroup orientation="horizontal" className="min-h-0 flex-1">
            {/* Sizes are strings on purpose: in v4 a bare number means pixels,
                so `defaultSize={38}` would be a 38px panel. */}
            <ResizablePanel
              defaultSize={showSidePanel ? "62%" : "100%"}
              minSize="35%"
              className="min-w-0"
            >
              <ChatView
                onOpenProduct={openProduct}
                onShowResults={() => setMobileSheetOpen(true)}
              />
            </ResizablePanel>

            <AnimatePresence initial={false}>
              {showSidePanel ? (
                <>
                  <ResizableHandle />
                  <ResizablePanel
                    defaultSize="38%"
                    minSize="26%"
                    maxSize="55%"
                    className="min-w-0"
                  >
                    <motion.div
                      className="h-full min-w-0"
                      variants={enterFromRight}
                      initial="hidden"
                      animate="visible"
                      exit="exit"
                    >
                      <ProductPanel />
                    </motion.div>
                  </ResizablePanel>
                </>
              ) : null}
            </AnimatePresence>
          </ResizablePanelGroup>
        </div>
      </SidebarInset>

      <Sheet
        open={!isDesktop && mobileSheetOpen && hasResults}
        onOpenChange={setMobileSheetOpen}
      >
        <SheetContent
          side="right"
          // The width comes from a `data-[side=right]:w-3/4` variant inside the
          // sheet, which a plain `w-full` cannot override, so the variant is
          // repeated here. A three quarter sheet is right on a tablet and too
          // narrow for product cards on a phone.
          className="gap-0 p-0 data-[side=right]:w-full sm:max-w-md"
        >
          <SheetTitle className="sr-only">Products</SheetTitle>
          {/* The sheet keeps its content mounted while closed, so the panel is
              only rendered when it is the one actually on screen. Rendering it
              in both places would draw every product card twice. */}
          {!isDesktop && hasResults ? (
            compare.length > 1 ? (
              <CompareTable />
            ) : (
              <ProductPanel />
            )
          ) : null}
        </SheetContent>
      </Sheet>

      <Drawer open={drawerProduct !== null} onOpenChange={(open) => !open && setDrawerProduct(null)}>
        <DrawerContent className="max-h-screen">
          {drawerProduct ? <ProductDetail product={drawerProduct} /> : null}
        </DrawerContent>
      </Drawer>
    </SidebarProvider>
  );
}

export default function App() {
  return (
    <ShopProvider>
      <Workspace />
    </ShopProvider>
  );
}
