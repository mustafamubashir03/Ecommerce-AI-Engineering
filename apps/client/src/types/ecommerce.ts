export interface RAGUsedContext {
  id: string;
  image_url: string;
  price: number | null;
  description: string;
  rating: number | null;
}
export interface RagResponse {
  request_id: string;
  answer: string;
  question_relevancy: boolean;
  used_context: RAGUsedContext[];
  thread_id: string | null;
  trace_id: string;
}
export type MessageStatus = "pending" | "done" | "error";
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: MessageStatus;
  requestId: string | null;
  error: string | null;
  products: Product[];
  traceId: string | null;
  vote: "up" | "down" | null;
  feedbackError: string | null;
  activity: string | null;
}
export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  threadId: string | null;
  messages: ChatMessage[];
}
export interface Product {
  id: string;
  title: string;
  imageUrl: string;
  price: number | null;
  rating: number | null;
  reason: string;
  sourceMessageId: string | null;
}
export interface CartLine {
  product: Product;
  quantity: number;
}
export type PanelTab = "results" | "saved" | "cart";
export type SortKey = "relevance" | "price-asc" | "price-desc" | "rating";
export type AgentStreamEvent =
  | { type: "token"; text: string }
  | { type: "status"; text: string }
  | { type: "result"; payload: RagResponse }
  | { type: "error"; message: string; status: number | null }
  | { type: "done" };
