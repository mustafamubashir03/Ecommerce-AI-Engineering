import type { Product, RAGUsedContext } from "@/types/ecommerce";
const currency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
});
export function formatPrice(price: number | null): string {
  return price === null || Number.isNaN(price) ? "Price unavailable" : currency.format(price);
}
export function formatRating(rating: number | null): string | null {
  if (rating === null || Number.isNaN(rating)) return null;
  return rating.toFixed(1);
}
export function toProduct(ctx: RAGUsedContext, sourceMessageId: string): Product {
  const reason = ctx.description?.trim() ?? "";
  const firstSentence = reason.split(/(?<=[.!?])\s/)[0] ?? reason;
  return {
    id: ctx.id,
    title: firstSentence || ctx.id,
    imageUrl: ctx.image_url,
    price: ctx.price,
    rating: ctx.rating,
    reason,
    sourceMessageId,
  };
}
export function latestProducts(messages: { products: Product[] }[]): Product[] {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const found = messages[index].products;
    if (found.length > 0) return found;
  }
  return [];
}
