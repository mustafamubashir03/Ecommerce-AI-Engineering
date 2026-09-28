import type { Conversation } from "@/types/ecommerce";

const KEYS = {
  conversations: "aether.conversations.v1",
  cart: "aether.cart.v1",
  saved: "aether.saved.v1",
} as const;

export function readStorage<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

export function writeStorage(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable, state stays in memory */
  }
}

export const STORAGE_KEYS = KEYS;

export function newConversation(): Conversation {
  return {
    id: crypto.randomUUID(),
    title: "New conversation",
    createdAt: Date.now(),
    threadId: null,
    messages: [],
  };
}

export function titleFromQuery(text: string): string {
  const clean = text.trim();
  return clean.length > 48 ? `${clean.slice(0, 48)}…` : clean;
}
