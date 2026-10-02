import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { toast } from "sonner";
import { toProduct, latestProducts } from "@/lib/format";
import { TokenBuffer } from "@/lib/token-buffer";
import { streamAgent, submitFeedback } from "@/services/api";
import type {
  CartLine,
  ChatMessage,
  Conversation,
  PanelTab,
  Product,
  SortKey,
} from "@/types/ecommerce";
const STORAGE_KEYS = {
  conversations: "aether.conversations.v1",
  cart: "aether.cart.v1",
  saved: "aether.saved.v1",
} as const;
const MAX_COMPARE = 4;
function readStorage<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}
  function writeStorage(key: string, value: unknown) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* storage unavailable, state stays in memory */
    }
  }
function withMessageDefaults(conversation: Conversation): Conversation {
  return {
    ...conversation,
    messages: (conversation.messages ?? []).map((message) => ({
      ...message,
      traceId: message.traceId ?? null,
      vote: message.vote ?? null,
      feedbackError: message.feedbackError ?? null,
      activity: message.activity ?? null,
    })),
  };
}
function newConversation(): Conversation {
  return {
    id: crypto.randomUUID(),
    title: "New conversation",
    createdAt: Date.now(),
    threadId: null,
    messages: [],
  };
}
interface ShopContextValue {
  conversations: Conversation[];
  activeConversation: Conversation;
  isBusy: boolean;
  products: Product[];
  resultsMessageId: string | null;
  findProduct: (id: string) => Product | null;
  cart: CartLine[];
  saved: Product[];
  compare: Product[];
  attached: Product | null;
  panel: { open: boolean; tab: PanelTab; detailId: string | null };
  sort: SortKey;
  send: (text: string) => void;
  stop: () => void;
  retry: (messageId?: string) => void;
  sendFeedback: (messageId: string, feedback: { score?: number | null; text?: string }) => Promise<void>;
  newChat: () => void;
  selectConversation: (id: string) => void;
  deleteConversation: (id: string) => void;
  askAbout: (product: Product) => void;
  attach: (product: Product | null) => void;
  addToCart: (product: Product, quantity?: number) => void;
  setQuantity: (productId: string, quantity: number) => void;
  removeFromCart: (productId: string) => void;
  toggleSaved: (product: Product) => void;
  toggleCompare: (product: Product) => void;
  setSort: (sort: SortKey) => void;
  openPanel: (tab?: PanelTab, detailId?: string) => void;
  closePanel: () => void;
  showDetail: (product: Product) => void;
  hideDetail: () => void;
}
const ShopContext = createContext<ShopContextValue | null>(null);
export function ShopProvider({ children }: { children: ReactNode }) {
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    const stored = readStorage<Conversation[]>(STORAGE_KEYS.conversations, []);
    return stored.length > 0 ? stored.map(withMessageDefaults) : [newConversation()];
  });
  const [activeId, setActiveId] = useState(() => conversations[0].id);
  const [cart, setCart] = useState<CartLine[]>(() => readStorage<CartLine[]>(STORAGE_KEYS.cart, []));
  const [saved, setSaved] = useState<Product[]>(() => readStorage<Product[]>(STORAGE_KEYS.saved, []));
  const [compare, setCompare] = useState<Product[]>([]);
  const [attached, setAttached] = useState<Product | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [sort, setSort] = useState<SortKey>("relevance");
  const [panelState, setPanel] = useState<{
    tab: PanelTab;
    detailId: string | null;
    dismissed: boolean;
  }>({
    tab: "results",
    detailId: null,
    dismissed: false,
  });
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => writeStorage(STORAGE_KEYS.conversations, conversations), [conversations]);
  useEffect(() => writeStorage(STORAGE_KEYS.cart, cart), [cart]);
  useEffect(() => writeStorage(STORAGE_KEYS.saved, saved), [saved]);
  const activeConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === activeId) ?? conversations[0],
    [activeId, conversations]
  );
  const products = useMemo(
    () => latestProducts(activeConversation?.messages ?? []),
    [activeConversation]
  );
  const resultsMessageId = products[0]?.sourceMessageId ?? null;
  const findProduct = useCallback(
    (id: string): Product | null => {
      const messages = activeConversation?.messages ?? [];
      for (let index = messages.length - 1; index >= 0; index -= 1) {
        const match = messages[index].products.find((product) => product.id === id);
        if (match) return match;
      }
      return null;
    },
    [activeConversation]
  );
  const panel = useMemo(
    () => ({
      open: products.length > 0 && !panelState.dismissed,
      tab: panelState.tab,
      detailId: panelState.detailId,
    }),
    [products.length, panelState.dismissed, panelState.tab, panelState.detailId]
  );
  const patchConversation = useCallback(
    (id: string, update: (conversation: Conversation) => Conversation) => {
      setConversations((prev) => prev.map((item) => (item.id === id ? update(item) : item)));
    },
    []
  );
  const patchMessage = useCallback(
    (conversationId: string, messageId: string, update: (message: ChatMessage) => ChatMessage) => {
      patchConversation(conversationId, (conversation) => ({
        ...conversation,
        messages: conversation.messages.map((message) =>
          message.id === messageId ? update(message) : message
        ),
      }));
    },
    [patchConversation]
  );
  const runQuery = useCallback(
    async (query: string, threadId: string | null) => {
      const controller = new AbortController();
      abortRef.current = controller;
      setIsBusy(true);
      const assistantId = crypto.randomUUID();
      const conversationId = activeConversation?.id;
      if (!conversationId) return;
      const userId = crypto.randomUUID();
      patchConversation(conversationId, (conversation) => ({
        ...conversation,
        title:
          conversation.title === "New conversation" ? truncateTitle(query) : conversation.title,
        messages: [
          ...conversation.messages,
          {
            id: userId,
            role: "user",
            content: query,
            status: "done",
            requestId: null,
            error: null,
            products: [],
            traceId: null,
            vote: null,
            feedbackError: null,
            activity: null,
          },
          {
            id: assistantId,
            role: "assistant",
            content: "",
            status: "pending",
            requestId: null,
            error: null,
            products: [],
            traceId: null,
            vote: null,
            feedbackError: null,
            activity: null,
          },
        ],
      }));
      const pending = new TokenBuffer((text) =>
        patchMessage(conversationId, assistantId, (message) => ({
          ...message,
          content: message.content + text,
        }))
      );
      try {
        await streamAgent(
          query,
          threadId,
          {
            onToken: pending.push,
            onStatus: (text) =>
              patchMessage(conversationId, assistantId, (message) => ({
                ...message,
                activity: message.content ? null : text,
              })),
            onResult: (response) => {
              pending.flush();
              const found = response.used_context.map((item) => toProduct(item, assistantId));
              patchConversation(conversationId, (conversation) => ({
                ...conversation,
                threadId: response.thread_id ?? conversation.threadId,
                messages: conversation.messages.map((message) =>
                  message.id === assistantId
                    ? {
                      ...message,
                      content: response.answer || message.content,
                      status: "done" as const,
                      requestId: response.request_id,
                      traceId: response.trace_id || null,
                      products: found,
                      activity: null,
                    }
                    : message
                ),
              }));
              if (found.length > 0) {
                setPanel((prev) => ({ ...prev, tab: "results", detailId: null, dismissed: false }));
              }
            },
          },
          controller.signal
        );
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          patchConversation(conversationId, (conversation) => ({
            ...conversation,
            messages: conversation.messages.filter(
              (message) => message.id !== assistantId && message.id !== userId
            ),
          }));
          return;
        }
        const message = error instanceof Error ? error.message : "Unexpected error.";
        patchConversation(conversationId, (conversation) => ({
          ...conversation,
          messages: conversation.messages.map((item) =>
            item.id === assistantId ? { ...item, status: "error" as const, error: message } : item
          ),
        }));
      } finally {
        pending.cancel();
        abortRef.current = null;
        setIsBusy(false);
      }
    },
    [activeConversation?.id, patchConversation, patchMessage]
  );
  const send = useCallback(
    (text: string) => {
      const query = text.trim();
      if (!query || isBusy) return;
      setAttached(null);
      void runQuery(query, activeConversation?.threadId ?? null);
    },
    [activeConversation?.threadId, isBusy, runQuery]
  );
  const stop = useCallback(() => abortRef.current?.abort(), []);
  const retry = useCallback(
    (messageId?: string) => {
      const messages = activeConversation?.messages ?? [];
      if (messages.length === 0 || isBusy) return;
      const target = messageId
        ? messages.findIndex((message) => message.id === messageId)
        : messages.length - 1;
      if (target < 0) return;
      let questionIndex = target;
      while (questionIndex >= 0 && messages[questionIndex].role !== "user") questionIndex -= 1;
      if (questionIndex < 0) return;
      const question = messages[questionIndex].content;
      const replaced = new Set<string>([messages[questionIndex].id]);
      if (messages[target].role === "assistant") replaced.add(messages[target].id);
      for (const message of messages) {
        if (message.status === "error") replaced.add(message.id);
      }
      patchConversation(activeConversation.id, (conversation) => ({
        ...conversation,
        messages: conversation.messages.filter((message) => !replaced.has(message.id)),
      }));
      void runQuery(question, activeConversation.threadId);
    },
    [activeConversation, isBusy, patchConversation, runQuery]
  );
  const sendFeedback = useCallback(
    async (messageId: string, feedback: { score?: number | null; text?: string }) => {
      const message = activeConversation?.messages.find((item) => item.id === messageId);
      if (!message) return;
      if (!message.traceId) {
        patchMessage(activeConversation.id, messageId, (current) => ({
          ...current,
          feedbackError: "This answer has no trace, so it cannot be rated.",
        }));
        return;
      }
      try {
        const result = await submitFeedback({
          trace_id: message.traceId,
          feedback_score: feedback.score ?? null,
          feedback_text: feedback.text ?? "",
          thread_id: activeConversation.threadId,
        });
        if (result.status !== "recorded") {
          patchMessage(activeConversation.id, messageId, (current) => ({
            ...current,
            feedbackError: `The feedback was not recorded (status: ${result.status}).`,
          }));
          return;
        }
        patchMessage(activeConversation.id, messageId, (current) => ({
          ...current,
          vote: feedback.score === 1 ? "up" : feedback.score === -1 ? "down" : current.vote,
          feedbackError: null,
        }));
      } catch (error) {
        patchMessage(activeConversation.id, messageId, (current) => ({
          ...current,
          feedbackError: error instanceof Error ? error.message : "The feedback could not be saved.",
        }));
      }
    },
    [activeConversation, patchMessage]
  );
  const newChat = useCallback(() => {
    abortRef.current?.abort();
    const conversation = newConversation();
    setConversations((prev) => [conversation, ...prev]);
    setActiveId(conversation.id);
    setPanel({ tab: "results", detailId: null, dismissed: false });
    setCompare([]);
  }, []);
  const selectConversation = useCallback(
    (id: string) => {
      abortRef.current?.abort();
      setActiveId(id);
      setCompare([]);
      setPanel((prev) => ({ ...prev, detailId: null }));
    },
    []
  );
  const deleteConversation = useCallback(
    (id: string) => {
      const remaining = conversations.filter((conversation) => conversation.id !== id);
      const next = remaining.length > 0 ? remaining : [newConversation()];
      setConversations(next);
      if (activeId === id) setActiveId(next[0].id);
    },
    [activeId, conversations]
  );
  const askAbout = useCallback(
    (product: Product) => {
      setAttached(product);
      setPanel((prev) => ({ ...prev, detailId: null }));
    },
    []
  );
  const addToCart = useCallback(
    (product: Product, quantity = 1) => {
      setCart((prev) => {
        const existing = prev.find((line) => line.product.id === product.id);
        if (existing) {
          return prev.map((line) =>
            line.product.id === product.id ? { ...line, quantity: line.quantity + quantity } : line
          );
        }
        return [...prev, { product, quantity }];
      });
      setPanel((prev) => ({ ...prev, tab: "cart" }));
      toast.success(`${product.title} added to cart`, {
        action: { label: "Undo", onClick: () => setCart((prev) => prev.filter((line) => line.product.id !== product.id)) },
      });
    },
    []
  );
  const setQuantity = useCallback((productId: string, quantity: number) => {
    setCart((prev) =>
      quantity <= 0
        ? prev.filter((line) => line.product.id !== productId)
        : prev.map((line) => (line.product.id === productId ? { ...line, quantity } : line))
    );
  }, []);
  const removeFromCart = useCallback(
    (productId: string) => {
      const line = cart.find((item) => item.product.id === productId);
      setCart((prev) => prev.filter((item) => item.product.id !== productId));
      if (!line) return;
      toast.success("Removed from cart", {
        action: {
          label: "Undo",
          onClick: () => setCart((prev) => (prev.some((i) => i.product.id === productId) ? prev : [...prev, line])),
        },
      });
    },
    [cart]
  );
  const toggleSaved = useCallback((product: Product) => {
    setSaved((prev) => {
      const exists = prev.some((item) => item.id === product.id);
      toast.success(exists ? "Removed from saved" : "Saved for later");
      return exists ? prev.filter((item) => item.id !== product.id) : [product, ...prev];
    });
  }, []);
  const toggleCompare = useCallback(
    (product: Product) => {
      setCompare((prev) => {
        const exists = prev.some((item) => item.id === product.id);
        if (exists) return prev.filter((item) => item.id !== product.id);
        if (prev.length >= MAX_COMPARE) {
          toast.error(`Compare holds ${MAX_COMPARE} products. Remove one first.`);
          return prev;
        }
        return [...prev, product];
      });
    },
    []
  );
  const openPanel = useCallback((tab: PanelTab = "results", detailId: string | null = null) => {
    setPanel({ tab, detailId, dismissed: false });
  }, []);
  const closePanel = useCallback(
    () => setPanel((prev) => ({ ...prev, dismissed: true, detailId: null })),
    []
  );
  const showDetail = useCallback(
    (product: Product) =>
      setPanel((prev) => ({ ...prev, dismissed: false, detailId: product.id })),
    []
  );
  const hideDetail = useCallback(() => setPanel((prev) => ({ ...prev, detailId: null })), []);
  const value = useMemo<ShopContextValue>(
    () => ({
      conversations,
      activeConversation,
      isBusy,
      products,
      resultsMessageId,
      findProduct,
      cart,
      saved,
      compare,
      attached,
      panel,
      sort,
      send,
      stop,
      retry,
      sendFeedback,
      newChat,
      selectConversation,
      deleteConversation,
      askAbout,
      attach: setAttached,
      addToCart,
      setQuantity,
      removeFromCart,
      toggleSaved,
      toggleCompare,
      setSort,
      openPanel,
      closePanel,
      showDetail,
      hideDetail,
    }),
    [
      conversations,
      activeConversation,
      isBusy,
      products,
      resultsMessageId,
      findProduct,
      cart,
      saved,
      compare,
      attached,
      panel,
      sort,
      send,
      stop,
      retry,
      sendFeedback,
      newChat,
      selectConversation,
      deleteConversation,
      askAbout,
      addToCart,
      setQuantity,
      removeFromCart,
      toggleSaved,
      toggleCompare,
      openPanel,
      closePanel,
      showDetail,
      hideDetail,
    ]
  );
  return <ShopContext.Provider value={value}>{children}</ShopContext.Provider>;
}
// eslint-disable-next-line react-refresh/only-export-components
export function useShop() {
  const context = useContext(ShopContext);
  if (!context) throw new Error("useShop must be used inside a ShopProvider");
  return context;
}
function truncateTitle(text: string): string {
  const clean = text.trim();
  return clean.length > 48 ? `${clean.slice(0, 48)}…` : clean;
}
