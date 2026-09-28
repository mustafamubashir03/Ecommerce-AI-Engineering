import { useRef, useState } from "react";
import { Send, Square, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useShop } from "@/context/shop-context";

export function ChatComposer({ embedded = false }: { embedded?: boolean }) {
  const { send, stop, isBusy, attached, attach } = useShop();
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const query = value.trim();
    if (!query || isBusy) return;
    send(query);
    setValue("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const form = (
    <form
      className="flex items-end gap-2 rounded-xl border border-border bg-card p-2 shadow-md"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      <Textarea
        ref={textareaRef}
        value={value}
        rows={1}
        placeholder="Ask about products, prices or ratings"
        aria-label="Message the shopping assistant"
        className="max-h-40 min-h-10 resize-none border-0 bg-transparent p-2 shadow-none focus-visible:ring-0"
        onChange={(event) => {
          setValue(event.target.value);
          const element = event.target;
          element.style.height = "auto";
          element.style.height = `${element.scrollHeight}px`;
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            submit();
          }
        }}
      />

      {isBusy ? (
        <Button
          type="button"
          variant="outline"
          size="icon"
          className="rounded-full"
          onClick={stop}
          aria-label="Stop generating"
        >
          <Square className="size-4" />
        </Button>
      ) : (
        <Button
          type="submit"
          size="icon"
          className="rounded-full"
          aria-label="Send message"
          disabled={!value.trim()}
        >
          <Send className="size-4" />
        </Button>
      )}
    </form>
  );

  const chip = attached ? (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <span className="rounded-md bg-accent px-2 py-1 text-accent-foreground">
        Ask about this
      </span>
      <Button
        variant="ghost"
        size="icon"
        className="size-6"
        aria-label="Remove attached product"
        onClick={() => attach(null)}
      >
        <X className="size-3" />
      </Button>
    </div>
  ) : null;

  if (embedded) {
    return (
      <div className="space-y-2">
        {chip}
        {form}
      </div>
    );
  }

  return (
    <div className="border-t border-border bg-background px-4 py-4">
      <div className="mx-auto w-full max-w-3xl space-y-2">
        {chip}
        {form}
      </div>
    </div>
  );
}
