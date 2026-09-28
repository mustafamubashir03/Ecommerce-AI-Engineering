import { useMemo } from "react";

import type { Product, SortKey } from "@/types/ecommerce";

/** Filter by the price ceiling, then order the way the toolbar asks for. */
export function usePanelProducts(
  source: Product[],
  sort: SortKey,
  maxPrice: number,
  minRating: number
): Product[] {
  return useMemo(() => {
    const filtered = source.filter(
      (product) =>
        (product.price === null || product.price <= maxPrice) &&
        (minRating === 0 || (product.rating !== null && product.rating >= minRating))
    );
    const sorted = [...filtered];

    if (sort === "price-asc") {
      sorted.sort((a, b) => (a.price ?? Infinity) - (b.price ?? Infinity));
    } else if (sort === "price-desc") {
      sorted.sort((a, b) => (b.price ?? -Infinity) - (a.price ?? -Infinity));
    } else if (sort === "rating") {
      sorted.sort((a, b) => (b.rating ?? -Infinity) - (a.rating ?? -Infinity));
    }

    return sorted;
  }, [source, sort, maxPrice, minRating]);
}
