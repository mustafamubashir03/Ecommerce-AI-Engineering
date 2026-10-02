import { Button } from "@/components/ui/button";
import { useShop } from "@/context/shop-context";
const SUGGESTED = [
  "Show me noise cancelling headphones in stock",
  "Which laptops under $800 have the best ratings?",
  "Compare mechanical keyboards for a home office",
];
export function ChatEmptyState() {
  const { send } = useShop();
  return (
    <div className="flex min-h-full flex-col items-center justify-center gap-6 px-4 py-10 text-center">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold text-foreground">What are you looking for?</h1>
        <p className="max-w-md text-sm text-muted-foreground">
          Ask about anything in the catalog. The agent searches products for you and shows you exactly
          which ones it used.
        </p>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        {SUGGESTED.map((suggestion) => (
          <Button key={suggestion} variant="outline" size="sm" onClick={() => send(suggestion)}>
            {suggestion}
          </Button>
        ))}
      </div>
    </div>
  );
}
