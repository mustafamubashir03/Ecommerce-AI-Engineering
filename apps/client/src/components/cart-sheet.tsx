import { X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { useShop } from "@/context/shop-context";
import { formatPrice } from "@/lib/format";
export function CartSheet({ count }: { count: number }) {
  const { cart, setQuantity, removeFromCart, openPanel } = useShop();
  const total = cart.reduce((sum, line) => sum + (line.product.price ?? 0) * line.quantity, 0);
  return (
    <Sheet>
      <SheetTrigger
        render={
          <Button
            variant="outline"
            size="sm"
            className="relative hidden lg:flex"
            onClick={() => openPanel("cart")}
          >
            Cart
            {count > 0 ? (
              <Badge variant="outline" className="absolute -right-2 -top-2 px-1.5 py-0 text-xs">
                {count}
              </Badge>
            ) : null}
          </Button>
        }
      />
      <SheetContent side="right" className="flex w-full flex-col sm:max-w-md">
        <SheetTitle>Cart</SheetTitle>
        {cart.length === 0 ? (
          <p className="text-sm text-muted-foreground">Your cart is empty.</p>
        ) : (
          <div className="min-h-0 flex-1 overflow-y-auto">
            <ul className="space-y-4">
              {cart.map((line) => (
                <li key={line.product.id} className="flex gap-3">
                  <div className="size-14 shrink-0 overflow-hidden rounded-md bg-muted">
                    {line.product.imageUrl ? (
                      <img src={line.product.imageUrl} alt="" className="size-full object-cover" />
                    ) : null}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="line-clamp-2 text-sm text-foreground">{line.product.title}</p>
                    <p className="text-sm font-semibold text-foreground">
                      {formatPrice(line.product.price)}
                    </p>
                    <div className="mt-1 flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setQuantity(line.product.id, line.quantity - 1)}
                      >
                        {line.quantity === 1 ? <X className="size-3" /> : "−"}
                      </Button>
                      <span className="text-sm tabular-nums">{line.quantity}</span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setQuantity(line.product.id, line.quantity + 1)}
                      >
                        +
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="ml-auto"
                        onClick={() => removeFromCart(line.product.id)}
                      >
                        Remove
                      </Button>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
        {cart.length > 0 ? (
          <div className="mt-auto flex items-center justify-between border-t border-border pt-4">
            <span className="text-sm text-muted-foreground">Total</span>
            <span className="font-semibold text-foreground">{formatPrice(total)}</span>
          </div>
        ) : null}
      </SheetContent>
    </Sheet>
  );
}
