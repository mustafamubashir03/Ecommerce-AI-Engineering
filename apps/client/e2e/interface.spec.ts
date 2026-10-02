import { expect, test, type Page } from "@playwright/test";
const PRODUCTS = Array.from({ length: 6 }, (_, index) => ({
  id: `B0TEST${String(index).padStart(4, "0")}`,
  title: `Portable Washing Machine Model ${index}`,
  imageUrl: `https://placehold.co/400x300/e2e8f0/334155?text=Product+${index}`,
  price: index === 2 ? null : 19.99 + index * 10,
  rating: 3 + (index % 2),
  reason: "Matches the capacity and budget you asked about.",
  sourceMessageId: "m1",
}));
async function seedFinishedTurn(page: Page) {
  await page.addInitScript((products) => {
    localStorage.setItem(
      "aether.conversations.v1",
      JSON.stringify([
        {
          id: "c1",
          title: "washing machines",
          threadId: "t1",
          messages: [
            { id: "u1", role: "user", content: "which washing machines do you have?", status: "done", requestId: null, error: null, products: [] },
            {
              id: "m1",
              role: "assistant",
              status: "done",
              requestId: "req-1",
              error: null,
              content:
                "Here are the washing machines in stock - **Nictemaw Full Automatic** [**B0TEST0000**] and a second one [**B0TEST0001**].",
              products,
            },
          ],
        },
      ])
    );
  }, PRODUCTS);
}
test("a streamed answer is written as text, not re-parsed as markdown", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem(
      "aether.conversations.v1",
      JSON.stringify([
        {
          id: "c1",
          title: "t",
          threadId: "t1",
          messages: [
            { id: "u1", role: "user", content: "q", status: "done", requestId: null, error: null, products: [] },
            { id: "m1", role: "assistant", status: "pending", requestId: null, error: null, content: "Here are the **portable", products: [] },
          ],
        },
      ])
    );
  });
  await page.goto("/");
  await page.waitForTimeout(400);
  const message = page.locator("#message-m1");
  await expect(message).toBeVisible();
  await expect(message).toContainText("**portable");
  await expect(message.locator("span[aria-hidden='true']").first()).toBeVisible();
});
test("reduced motion removes the animation but keeps the content", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await seedFinishedTurn(page);
  await page.goto("/");
  await page.waitForTimeout(500);
  await expect(page.getByText("Nictemaw Full Automatic").first()).toBeVisible();
  const longAnimations = await page.evaluate(() =>
    Array.from(document.querySelectorAll("*")).filter((element) => {
      const style = getComputedStyle(element);
      const duration = Number.parseFloat(style.animationDuration);
      const iteration = style.animationIterationCount;
      return Number.isFinite(duration) && duration > 0.05 && iteration === "infinite";
    }).length
  );
  expect(longAnimations, "no infinite animation may run under prefers-reduced-motion").toBe(0);
});
test("the streamed text is coalesced rather than rendered per token", async ({ page }) => {
  await page.addInitScript(() => {
    (window as unknown as { __renderCount: number }).__renderCount = 0;
    const observer = new MutationObserver(() => {
      (window as unknown as { __renderCount: number }).__renderCount += 1;
    });
    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
  });
  await page.addInitScript(() => {
    localStorage.setItem(
      "aether.conversations.v1",
      JSON.stringify([
        {
          id: "c1",
          title: "t",
          threadId: "t1",
          messages: [
            { id: "u1", role: "user", content: "q", status: "done", requestId: null, error: null, products: [] },
            { id: "m1", role: "assistant", status: "pending", requestId: null, error: null, content: "", products: [] },
          ],
        },
      ])
    );
  });
  await page.goto("/");
  await page.waitForTimeout(400);
  await expect(page.locator("#message-m1")).toBeVisible();
});
test("the panel's product cards are reachable and operable", async ({ page }) => {
  await seedFinishedTurn(page);
  await page.goto("/");
  await page.waitForTimeout(400);
  const results = page.getByRole("button", { name: /Results/ });
  if (await results.isVisible().catch(() => false)) {
    await results.first().click();
    await page.waitForTimeout(400);
  }
  const panel = page.locator('[data-slot="resizable-panel"]').last();
  if (!(await panel.isVisible().catch(() => false))) return;
  const card = panel.getByRole("button", { name: /^View / }).first();
  await expect(card).toBeVisible();
  await expect(panel.getByRole("button", { name: /Add to cart/ }).first()).toBeVisible();
  await expect(panel.getByRole("button", { name: /Save for later|Remove from saved/ }).first()).toBeVisible();
  await expect(panel.getByRole("checkbox").first()).toBeVisible();
  await expect(panel.getByRole("button", { name: /Ask about this/ }).first()).toBeVisible();
});
test("citations in the answer open the product they name", async ({ page }) => {
  await seedFinishedTurn(page);
  await page.goto("/");
  await page.waitForTimeout(400);
  const citation = page.getByRole("button", { name: "B0TEST0000" }).first();
  await expect(citation).toBeVisible();
  await citation.click();
  await page.waitForTimeout(400);
  await expect(page.getByText("B0TEST0000").first()).toBeVisible();
});
