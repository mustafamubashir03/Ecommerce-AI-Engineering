import { expect, test, type Page } from "@playwright/test";
function makeProducts(count: number) {
  return Array.from({ length: count }, (_, index) => ({
    id: `B0TEST${String(index).padStart(4, "0")}`,
    title: `Portable Washing Machine Model ${index} with a long enough name to wrap`,
    imageUrl: `https://placehold.co/400x300/e2e8f0/334155?text=Product+${index}`,
    price: index === 3 ? null : 19.99 + index * 10,
    rating: 3 + (index % 2),
    reason: "Matches the capacity and budget you asked about.",
    sourceMessageId: "a1",
  }));
}
async function seedAnswer(page: Page, productCount: number) {
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
              id: "a1",
              role: "assistant",
              status: "done",
              requestId: "req-1",
              error: null,
              content: "Here are the washing machines. [**B0TEST0000**] [**B0TEST0001**]",
              products,
            },
          ],
        },
      ])
    );
  }, makeProducts(productCount));
}
const ROW = '[aria-label="Products used in this answer"]';
test("the inline product row clips and scrolls sideways", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await seedAnswer(page, 8);
  await page.goto("/");
  await page.waitForTimeout(600);
  const row = page.locator(ROW);
  await expect(row).toBeVisible();
  const box = await row.evaluate((element) => ({
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
    overflowX: getComputedStyle(element).overflowX,
  }));
  expect(box.scrollWidth, "the row must hold more than it can show").toBeGreaterThan(box.clientWidth);
  expect(box.overflowX).toBe("auto");
  await row.evaluate((element) => element.scrollTo({ left: element.scrollWidth }));
  await page.waitForTimeout(300);
  const scrolledTo = await row.evaluate((element) => element.scrollLeft);
  expect(scrolledTo, "the row must scroll to its end").toBeGreaterThan(0);
  await row.evaluate((element) => element.scrollTo({ left: 0 }));
});
test("the row shows a way to scroll, and it disappears when there is nothing to scroll", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await seedAnswer(page, 8);
  await page.goto("/");
  await page.waitForTimeout(600);
  const next = page.getByRole("button", { name: "Scroll products right" });
  await expect(next).toBeVisible();
  await next.click();
  await expect
    .poll(async () => page.locator(ROW).evaluate((element) => element.scrollLeft), { timeout: 5000 })
    .toBeGreaterThan(0);
  await expect(page.getByRole("button", { name: "Scroll products left" })).toBeVisible({
    timeout: 5000,
  });
  await page.locator(ROW).evaluate((element) => element.scrollTo({ left: element.scrollWidth }));
  await expect(next).toBeHidden({ timeout: 5000 });
  await expect(page.getByRole("button", { name: "Scroll products left" })).toBeVisible();
});
test("a single product shows no scroll controls", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await seedAnswer(page, 1);
  await page.goto("/");
  await page.waitForTimeout(600);
  await expect(page.getByRole("button", { name: "Scroll products right" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Scroll products left" })).toHaveCount(0);
});
test("the product page opens, shows the record, and adds to the cart", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await seedAnswer(page, 5);
  await page.goto("/");
  await page.waitForTimeout(600);
  await page.getByRole("button", { name: /^View / }).first().click();
  await page.waitForTimeout(500);
  const panel = page.locator('[data-slot="resizable-panel"]').nth(1);
  await expect(panel.getByText("Product 1 of 5")).toBeVisible();
  await expect(panel.getByText("Why the agent picked this")).toBeVisible();
  await expect(panel.getByText("Catalog record")).toBeVisible();
  await expect(panel.getByText("B0TEST0000").first()).toBeVisible();
  await panel.getByRole("button", { name: "Save" }).click();
  await expect(panel.getByRole("button", { name: "Saved" })).toBeVisible();
  await panel.getByRole("button", { name: "Compare" }).click();
  await expect(panel.getByRole("button", { name: "Comparing" })).toBeVisible();
  await expect(panel.getByRole("button", { name: "Copy product ID" })).toBeVisible();
  await panel.getByRole("button", { name: "Add to cart" }).click();
  await expect(panel.getByText("In cart").first()).toBeVisible();
});
test("the product page steps through the results", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await seedAnswer(page, 5);
  await page.goto("/");
  await page.waitForTimeout(600);
  await page.getByRole("button", { name: /^View / }).first().click();
  await page.waitForTimeout(500);
  const panel = page.locator('[data-slot="resizable-panel"]').nth(1);
  await expect(panel.getByText("Product 1 of 5")).toBeVisible();
  await expect(panel.getByRole("button", { name: "Previous product" })).toBeDisabled();
  await panel.getByRole("button", { name: "Next product" }).click();
  await expect(panel.getByText("Product 2 of 5")).toBeVisible();
  await expect(panel.getByRole("button", { name: "Previous product" })).toBeEnabled();
  await panel.getByRole("button", { name: "Previous product" }).click();
  await expect(panel.getByText("Product 1 of 5")).toBeVisible();
});
test("a product with no price cannot be added to the cart", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await seedAnswer(page, 5);
  await page.goto("/");
  await page.waitForTimeout(600);
  await page.getByRole("button", { name: /View .*Model 3/ }).first().click();
  await page.waitForTimeout(500);
  const panel = page.locator('[data-slot="resizable-panel"]').nth(1);
  const add = panel.getByRole("button", { name: /Price unavailable|Add to cart/ });
  await expect(add).toBeDisabled();
  await expect(panel.getByText("Not recorded").or(panel.getByText(/out of 5/)).first()).toBeVisible();
});
test("on a phone the results stay out of the way until asked for", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.addInitScript(() => {
    localStorage.setItem(
      "aether.conversations.v1",
      JSON.stringify([
        {
          id: "c1",
          title: "t",
          threadId: "t1",
          messages: [
            { id: "u1", role: "user", content: "washing machines", status: "done", requestId: null, error: null, products: [] },
          ],
        },
      ])
    );
  });
  await page.route("**/agent/stream", async (route) => {
    const frames = [
      { type: "token", text: "Here are the machines. " },
      { type: "result", payload: { answer: "Here are the machines. [**B0TEST0000**]", question_relevancy: true, request_id: "r", thread_id: "t1", used_context: makeProducts(4) } },
      { type: "done" },
    ];
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      headers: { "Access-Control-Allow-Origin": "*" },
      body: frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join(""),
    });
  });
  await page.goto("/");
  await page.waitForTimeout(500);
  await page.getByRole("textbox", { name: /Message the shopping assistant/ }).fill("washing machines");
  await page.getByRole("button", { name: "Send message" }).click();
  await page.waitForTimeout(800);
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.getByText("Here are the machines.").first()).toBeVisible();
  const showResults = page.getByRole("button", { name: /4 results/ });
  await expect(showResults).toBeVisible();
  await showResults.click();
  const sheet = page.getByRole("dialog");
  await expect(sheet).toBeVisible();
  await expect(sheet.getByRole("heading", { name: "Products" }).last()).toBeVisible();
  await expect(sheet.getByRole("button", { name: /Add to cart/ }).first()).toBeVisible();
  await sheet.getByRole("button", { name: /^View / }).first().click();
  await page.waitForTimeout(600);
  const documentScroll = await page.evaluate(() => document.documentElement.scrollHeight);
  expect(documentScroll, "the product page must not make the page scroll").toBeLessThanOrEqual(845);
  const add = await sheet.getByRole("button", { name: "Add to cart" }).boundingBox();
  expect(add!.y + add!.height, "add to cart must be reachable without scrolling").toBeLessThanOrEqual(845);
});
