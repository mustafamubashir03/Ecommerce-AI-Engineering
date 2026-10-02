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
export async function streamAgent(
  query: string,
  threadId: string | null,
  handlers: {
    onToken: (text: string) => void;
    onStatus: (text: string) => void;
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
  for (; ;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
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
      if (event.type === "status") handlers.onStatus(event.text);
      if (event.type === "result") handlers.onResult(event.payload);
      if (event.type === "error") throw new ApiError(event.message, event.status ?? 502);
      if (event.type === "done") return;
    }
  }
}
export interface FeedbackRequest {
  feedback_score: number | null;
  feedback_text: string;
  trace_id: string;
  thread_id: string | null;
  feedback_source_type: "api" | "model";
}
export interface FeedbackResponse {
  request_id: string;
  status: string;
}
export async function submitFeedback(
  payload: Omit<FeedbackRequest, "feedback_source_type">,
  signal?: AbortSignal
): Promise<FeedbackResponse> {
  const response = await fetch(`${API_BASE_URL}/feedback/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...payload,
      feedback_text: payload.feedback_text.trim(),
      feedback_source_type: "api",
    } satisfies FeedbackRequest),
    signal,
  });
  if (!response.ok) {
    throw new ApiError(await readError(response), response.status);
  }
  return (await response.json()) as FeedbackResponse;
}
async function readError(response: Response): Promise<string> {
  const body = await response.json().catch(() => null);
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object" && "error" in detail) {
      const message = (detail as { error: unknown }).error;
      if (typeof message === "string") return message;
    }
    if (Array.isArray(detail) && detail.length > 0) {
      const first = detail[0] as { msg?: unknown };
      if (typeof first?.msg === "string") return first.msg;
    }
  }
  return `The feedback could not be saved (${response.status}).`;
}
