import { Minus, Plus, ShoppingCart, Trash2 } from "lucide-react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { useShop } from "@/context/shop-context";
import { formatPrice } from "@/lib/format";
export function CartTab() {
  const { cart, setQuantity, removeFromCart } = useShop();
  if (cart.length === 0) {
    return (
      <p className="p-8 text-center text-sm text-muted-foreground">
        Cart is empty. Add a product to review the order before paying.
      </p>
    );
  }
  const total = cart.reduce((sum, line) => sum + (line.product.price ?? 0) * line.quantity, 0);
  const count = cart.reduce((sum, line) => sum + line.quantity, 0);
  return (
    <div className="flex h-full min-h-0 flex-col">
      <ul className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        {cart.map((line) => (
          <li key={line.product.id} className="flex gap-3">
            <div className="size-16 shrink-0 overflow-hidden rounded-md bg-muted">
              {line.product.imageUrl ? (
                <img src={line.product.imageUrl} alt="" className="size-full object-cover" />
              ) : null}
            </div>
            <div className="min-w-0 flex-1 space-y-1">
              <p className="line-clamp-2 text-sm text-foreground">{line.product.title}</p>
              <p className="text-sm font-semibold text-foreground">{formatPrice(line.product.price)}</p>
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1 rounded-md border border-border">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="size-7"
                    aria-label="Decrease quantity"
                    onClick={() => setQuantity(line.product.id, line.quantity - 1)}
                  >
                    <Minus className="size-3" />
                  </Button>
                  <span className="w-5 text-center text-sm tabular-nums">{line.quantity}</span>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="size-7"
                    aria-label="Increase quantity"
                    onClick={() => setQuantity(line.product.id, line.quantity + 1)}
                  >
                    <Plus className="size-3" />
                  </Button>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-7"
                  aria-label={`Remove ${line.product.id} from cart`}
                  onClick={() => removeFromCart(line.product.id)}
                >
                  <Trash2 className="size-3" />
                </Button>
              </div>
            </div>
          </li>
        ))}
      </ul>
      <div className="space-y-3 border-t border-border p-4">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">
            {count} {count === 1 ? "item" : "items"}
          </span>
          <span className="font-semibold text-foreground">{formatPrice(total)}</span>
        </div>
        <AlertDialog>
          <AlertDialogTrigger
            render={
              <Button className="w-full">
                <ShoppingCart className="size-4" />
                Review order
              </Button>
            }
          />
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Confirm this order</AlertDialogTitle>
              <AlertDialogDescription>
                You are about to place an order for {count} {count === 1 ? "item" : "items"} totalling{" "}
                {formatPrice(total)}. The agent never changes your cart on its own, so nothing is
                charged until you confirm here.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Keep shopping</AlertDialogCancel>
              <AlertDialogAction>Place order</AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </div>
  );
}
