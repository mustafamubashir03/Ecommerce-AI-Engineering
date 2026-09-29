import type { AgentStreamEvent, RagResponse } from "@/types/ecommerce";

const API_BASE_URL = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "http://localhost:8000";

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function checkApiHealth(signal?: AbortSignal): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/docs`, { signal });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Streaming turn. The assistant writes tokens as it thinks, then sends one
 * result event with the products it used.
 */
export async function streamAgent(
  query: string,
  threadId: string | null,
  handlers: {
    onToken: (text: string) => void;
    onResult: (response: RagResponse) => void;
  },
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/agent/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, thread_id: threadId ?? undefined }),
    signal,
  });

  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => "");
    throw new ApiError(
      `The assistant returned ${response.status}. ${detail.slice(0, 200)}`.trim(),
      response.status
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const payload = frame.split("\n").find((line) => line.startsWith("data: "));
      if (!payload) continue;

      let event: AgentStreamEvent;
      try {
        event = JSON.parse(payload.slice(6)) as AgentStreamEvent;
      } catch {
        continue;
      }

      if (event.type === "token") handlers.onToken(event.text);
      if (event.type === "result") handlers.onResult(event.payload);
      if (event.type === "error") throw new ApiError(event.message, event.status ?? 502);
      if (event.type === "done") return;
    }
  }
}
