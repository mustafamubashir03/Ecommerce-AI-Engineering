import { Fragment, type ReactNode } from "react";
import { cn } from "@/lib/utils";
const CITATION = /(\[\*{0,2}[A-Z0-9]{8,14}\*{0,2}\])/g;
const BOLD = /(\*\*[^*]+\*\*)/g;
function renderInline(text: string, onCitation: (id: string) => void): ReactNode[] {
  return text.split(CITATION).map((part, index) => {
    const citation = /^\[\*{0,2}([A-Z0-9]{8,14})\*{0,2}\]$/.exec(part);
    if (citation) {
      const id = citation[1];
      return (
        <button
          key={index}
          type="button"
          onClick={() => onCitation(id)}
          className="mx-0.5 inline-flex items-center rounded-sm bg-accent px-1 font-mono text-xs text-accent-foreground transition-colors hover:bg-primary hover:text-primary-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          title={`Open product ${id}`}
        >
          {id}
        </button>
      );
    }
    return part.split(BOLD).map((boldPart, boldIndex) => {
      if (boldPart.startsWith("**") && boldPart.endsWith("**")) {
        return (
          <strong key={`${index}-${boldIndex}`} className="font-semibold text-foreground">
            {boldPart.slice(2, -2)}
          </strong>
        );
      }
      return <Fragment key={`${index}-${boldIndex}`}>{boldPart}</Fragment>;
    });
  });
}
export function AgentMarkdown({
  content,
  onCitation,
  className,
}: {
  content: string;
  onCitation: (id: string) => void;
  className?: string;
}) {
  if (!content.trim()) return null;
  const blocks = content.split("\n");
  return (
    <div className={cn("space-y-2 text-sm leading-relaxed text-foreground", className)}>
      {blocks.map((line, index) => {
        const trimmed = line.trim();
        if (!trimmed) return <div key={index} className="h-2" aria-hidden="true" />;
        if (/^\s*[-*]\s+/.test(line)) {
          return (
            <div key={index} className="flex gap-2 pl-1">
              <span className="mt-2 size-1.5 shrink-0 rounded-full bg-muted-foreground" aria-hidden="true" />
              <p className="flex-1">{renderInline(trimmed.replace(/^[-*]\s+/, ""), onCitation)}</p>
            </div>
          );
        }
        if (trimmed.startsWith("### ")) {
          return (
            <h4 key={index} className="pt-1 text-sm font-semibold text-foreground">
              {renderInline(trimmed.slice(4), onCitation)}
            </h4>
          );
        }
        if (trimmed.startsWith("## ")) {
          return (
            <h3 key={index} className="pt-2 text-base font-semibold text-foreground">
              {renderInline(trimmed.slice(3), onCitation)}
            </h3>
          );
        }
        return <p key={index}>{renderInline(trimmed, onCitation)}</p>;
      })}
    </div>
  );
}
