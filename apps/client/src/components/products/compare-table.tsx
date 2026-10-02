import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useShop } from "@/context/shop-context";
import { formatPrice, formatRating } from "@/lib/format";
import { cn } from "@/lib/utils";
export function CompareTable() {
  const { compare, toggleCompare, addToCart } = useShop();
  if (compare.length === 0) {
    return (
      <p className="p-4 text-sm text-muted-foreground">
        Tick compare on a product card to line up two to four products here.
      </p>
    );
  }
  const prices = compare.map((product) => product.price).filter((price): price is number => price !== null);
  const ratings = compare.map((product) => product.rating).filter((rating): rating is number => rating !== null);
  const bestPrice = prices.length > 0 ? Math.min(...prices) : null;
  const bestRating = ratings.length > 0 ? Math.max(...ratings) : null;
  return (
    <div className="space-y-4 p-4">
      <div className="overflow-hidden rounded-lg border border-border">
        <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Product</TableHead>
            {compare.map((product) => (
              <TableHead key={product.id} className="align-bottom">
                <div className="space-y-1">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="size-6"
                    aria-label={`Remove ${product.id} from comparison`}
                    onClick={() => toggleCompare(product)}
                  >
                    <X className="size-3" />
                  </Button>
                  <p className="line-clamp-2 max-w-40 text-xs font-normal text-foreground">
                    {product.title}
                  </p>
                </div>
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          <TableRow>
            <TableCell className="text-muted-foreground">Price</TableCell>
            {compare.map((product) => (
              <TableCell
                key={product.id}
                className={cn(
                  "rounded-lg",
                  product.price !== null && product.price === bestPrice && "bg-accent text-accent-foreground font-semibold"
                )}
              >
                {formatPrice(product.price)}
              </TableCell>
            ))}
          </TableRow>
          <TableRow>
            <TableCell className="text-muted-foreground">Rating</TableCell>
            {compare.map((product) => (
              <TableCell
                key={product.id}
                className={cn(
                  "rounded-lg",
                  product.rating !== null && product.rating === bestRating && "bg-accent text-accent-foreground font-semibold"
                )}
              >
                {formatRating(product.rating) ?? "Not recorded"}
              </TableCell>
            ))}
          </TableRow>
          <TableRow>
            <TableCell className="text-muted-foreground">Product ID</TableCell>
            {compare.map((product) => (
              <TableCell key={product.id} className="font-mono text-xs">
                {product.id}
              </TableCell>
            ))}
          </TableRow>
          <TableRow>
            <TableCell className="text-muted-foreground">Agent's reason</TableCell>
            {compare.map((product) => (
              <TableCell key={product.id} className="max-w-40 text-xs">
                <p className="line-clamp-4">{product.reason || "Not provided"}</p>
              </TableCell>
            ))}
          </TableRow>
          <TableRow>
            <TableCell />
            {compare.map((product) => (
              <TableCell key={product.id}>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => addToCart(product)}
                  disabled={product.price === null}
                >
                  Add
                </Button>
              </TableCell>
            ))}
          </TableRow>
        </TableBody>
      </Table>
      </div>
    </div>
  );
}
