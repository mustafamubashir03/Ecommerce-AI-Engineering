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
  const [mobileSheetOpen, setMobileSheetOpen] = useState(false);
  const isDesktop = useMediaQuery(DESKTOP);
  const hasResults = products.length > 0;
  const showSidePanel = isDesktop && hasResults && panel.open;
  const openProduct = (product: Product) => {
    if (isDesktop) showDetail(product);
    else setDrawerProduct(product);
  };
  return (
    <SidebarProvider>
      <CollapseSidebarOnNarrow />
      <AppSidebar />
      <SidebarInset className="min-h-0">
        <div className="flex h-dvh flex-col overflow-hidden">
          <TopBar />
          <ResizablePanelGroup orientation="horizontal" className="min-h-0 flex-1">
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
          className="gap-0 p-0 data-[side=right]:w-full sm:max-w-md"
        >
          <SheetTitle className="sr-only">Products</SheetTitle>
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
