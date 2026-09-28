import { LayoutGrid, List, SlidersHorizontal } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { SortKey } from "@/types/ecommerce";

const SORTS: { value: SortKey; label: string }[] = [
  { value: "relevance", label: "Agent order" },
  { value: "price-asc", label: "Price: low to high" },
  { value: "price-desc", label: "Price: high to low" },
  { value: "rating", label: "Top rated" },
];

export function PanelToolbar({
  count,
  sort,
  onSortChange,
  maxPrice,
  onMaxPriceChange,
  priceCeiling,
  minRating,
  onMinRatingChange,
  view,
  onViewChange,
}: {
  count: number;
  sort: SortKey;
  onSortChange: (sort: SortKey) => void;
  maxPrice: number;
  onMaxPriceChange: (value: number) => void;
  priceCeiling: number;
  minRating: number;
  onMinRatingChange: (value: number) => void;
  view: "grid" | "list";
  onViewChange: (view: "grid" | "list") => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-sm text-muted-foreground">
        {count} {count === 1 ? "product" : "products"}
      </span>

      <Select value={sort} onValueChange={(value) => onSortChange(value as SortKey)}>
        <SelectTrigger size="sm" className="w-auto" aria-label="Sort products">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {SORTS.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Popover>
        <PopoverTrigger
          render={
            <Button variant="outline" size="icon" className="size-8" aria-label="Filter products" />
          }
        >
          <SlidersHorizontal className="size-4" />
        </PopoverTrigger>
        <PopoverContent align="end" className="w-64 space-y-4 p-4">
          <div className="space-y-3">
            <Label htmlFor="max-price">Maximum price</Label>
            <Slider
              id="max-price"
              min={0}
              max={priceCeiling}
              step={1}
              value={[maxPrice]}
              onValueChange={(value) => {
                const next = Array.isArray(value) ? value[0] : value;
                onMaxPriceChange(next ?? priceCeiling);
              }}
            />
            <p className="text-xs text-muted-foreground">Up to ${Math.round(maxPrice)}</p>
          </div>

          <Separator />

          <div className="space-y-3">
            <Label htmlFor="min-rating">Minimum rating</Label>
            <Slider
              id="min-rating"
              min={0}
              max={5}
              step={0.5}
              value={[minRating]}
              onValueChange={(value) => {
                const next = Array.isArray(value) ? value[0] : value;
                onMinRatingChange(next ?? 0);
              }}
            />
            <p className="text-xs text-muted-foreground">
              {minRating === 0 ? "Any rating" : `${minRating.toFixed(1)} and above`}
            </p>
          </div>
        </PopoverContent>
      </Popover>

      <ToggleGroup
        value={[view]}
        onValueChange={(value) => value.length > 0 && onViewChange(value[0] as "grid" | "list")}
        variant="outline"
        size="sm"
        aria-label="Result layout"
      >
        <ToggleGroupItem value="grid" aria-label="Grid view">
          <LayoutGrid className="size-4" />
        </ToggleGroupItem>
        <ToggleGroupItem value="list" aria-label="List view">
          <List className="size-4" />
        </ToggleGroupItem>
      </ToggleGroup>
    </div>
  );
}
