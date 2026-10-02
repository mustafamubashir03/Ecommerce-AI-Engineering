import { useEffect, useState } from "react";
import { PanelRight, Search } from "lucide-react";
import { useTheme } from "next-themes";
import { CartSheet } from "@/components/cart-sheet";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
} from "@/components/ui/command";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useShop } from "@/context/shop-context";
import { formatPrice } from "@/lib/format";
export function TopBar() {
  const {
    activeConversation,
    conversations,
    products,
    cart,
    panel,
    openPanel,
    closePanel,
    showDetail,
    newChat,
    selectConversation,
  } = useShop();
  const { resolvedTheme, setTheme } = useTheme();
  const [commandOpen, setCommandOpen] = useState(false);
  const [search, setSearch] = useState("");
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "k" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        setCommandOpen((open) => !open);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);
  const cartCount = cart.reduce((sum, line) => sum + line.quantity, 0);
  const needle = search.trim().toLowerCase();
  const productMatches = products.filter((product) =>
    `${product.title} ${product.id}`.toLowerCase().includes(needle)
  );
  const chatMatches = conversations.filter((conversation) =>
    conversation.title.toLowerCase().includes(needle)
  );
  return (
    <header className="flex h-14 shrink-0 items-center gap-2 border-b border-border px-4">
      <SidebarTrigger className="size-8" aria-label="Toggle sidebar" />
      <h1 className="min-w-0 flex-1 truncate text-sm font-semibold text-foreground">
        {activeConversation.title}
      </h1>
      <Button
        variant="outline"
        size="sm"
        className="hidden w-64 justify-start text-muted-foreground md:flex"
        onClick={() => setCommandOpen(true)}
      >
        <Search className="size-4" />
        Search products
        <CommandShortcut>⌘K</CommandShortcut>
      </Button>
      <Tooltip>
        <TooltipTrigger
          render={
            <Button
              variant="ghost"
              size="icon"
              className="md:hidden"
              aria-label="Search products"
              onClick={() => setCommandOpen(true)}
            />
          }
        >
          <Search className="size-4" />
        </TooltipTrigger>
        <TooltipContent>Search products</TooltipContent>
      </Tooltip>
      {products.length > 0 ? (
        <Button variant="ghost" size="sm" onClick={panel.open ? closePanel : () => openPanel("results")}>
          <PanelRight className="size-4" />
          <span className="hidden sm:inline">Results</span>
          <Badge variant="outline">{products.length}</Badge>
        </Button>
      ) : null}
      <CartSheet count={cartCount} />
      <CommandDialog
        open={commandOpen}
        onOpenChange={setCommandOpen}
        title="Command palette"
        description="Search chats and products, or jump to an action."
      >
        <CommandInput value={search} onValueChange={setSearch} placeholder="Search chats and products" />
        <CommandList>
          <CommandEmpty>Nothing matches that search.</CommandEmpty>
          {search.trim() ? (
            <>
              {productMatches.length > 0 ? (
                <CommandGroup heading="Products">
                  {productMatches.map((product) => (
                    <CommandItem
                      key={product.id}
                      value={product.id}
                      onSelect={() => {
                        showDetail(product);
                        setCommandOpen(false);
                      }}
                    >
                      <span className="truncate">{product.title}</span>
                      <span className="ml-auto text-xs text-muted-foreground">
                        {formatPrice(product.price)}
                      </span>
                    </CommandItem>
                  ))}
                </CommandGroup>
              ) : null}
              {chatMatches.length > 0 ? (
                <CommandGroup heading="Chats">
                  {chatMatches.map((conversation) => (
                    <CommandItem
                      key={conversation.id}
                      value={`chat-${conversation.id}`}
                      onSelect={() => {
                        selectConversation(conversation.id);
                        setCommandOpen(false);
                      }}
                    >
                      <span className="truncate">{conversation.title}</span>
                    </CommandItem>
                  ))}
                </CommandGroup>
              ) : null}
            </>
          ) : (
            <>
              <CommandGroup heading="Actions">
                <CommandItem
                  value="new-chat"
                  onSelect={() => {
                    newChat();
                    setCommandOpen(false);
                  }}
                >
                  New chat
                </CommandItem>
                <CommandItem
                  value="open-cart"
                  onSelect={() => {
                    openPanel("cart");
                    setCommandOpen(false);
                  }}
                >
                  Open cart
                  {cartCount > 0 ? <CommandShortcut>{cartCount}</CommandShortcut> : null}
                </CommandItem>
                <CommandItem
                  value="toggle-theme"
                  onSelect={() => {
                    setTheme(resolvedTheme === "dark" ? "light" : "dark");
                    setCommandOpen(false);
                  }}
                >
                  Toggle theme
                </CommandItem>
              </CommandGroup>
              <CommandSeparator />
              <CommandGroup heading="Recent chats">
                {conversations.slice(0, 5).map((conversation) => (
                  <CommandItem
                    key={conversation.id}
                    value={`chat-${conversation.id}`}
                    onSelect={() => {
                      selectConversation(conversation.id);
                      setCommandOpen(false);
                    }}
                  >
                    <span className="truncate">{conversation.title}</span>
                  </CommandItem>
                ))}
              </CommandGroup>
            </>
          )}
        </CommandList>
      </CommandDialog>
    </header>
  );
}
