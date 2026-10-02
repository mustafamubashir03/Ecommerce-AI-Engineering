import { expect, test, type Page } from "@playwright/test";
const VIEWPORTS = [
  { name: "phone-small", width: 360, height: 640 },
  { name: "phone", width: 390, height: 844 },
  { name: "tablet", width: 768, height: 1024 },
  { name: "laptop", width: 1024, height: 720 },
  { name: "desktop", width: 1440, height: 900 },
];
function makeProducts(count: number) {
  return Array.from({ length: count }, (_, index) => ({
    id: `B0TEST${String(index).padStart(4, "0")}`,
    title: `Portable Washing Machine Model ${index}`,
    imageUrl: `https://placehold.co/400x300/e2e8f0/334155?text=Product+${index}`,
    price: 19.99 + index * 10,
    rating: 3 + (index % 2),
    reason: "Matches the capacity and budget you asked about.",
    sourceMessageId: "a0",
  }));
}
async function seedLongConversation(page: Page) {
  const products = makeProducts(6);
  const messages: unknown[] = [];
  for (let turn = 0; turn < 6; turn += 1) {
    messages.push({
      id: `u${turn}`,
      role: "user",
      content: `which washing machines do you have? turn ${turn}`,
      status: "done",
      requestId: null,
      error: null,
      products: [],
    });
    messages.push({
      id: `a${turn}`,
      role: "assistant",
      status: "done",
      requestId: `req-${turn}`,
      error: null,
      content:
        "Here are the washing machines in stock. ".repeat(8) + "[**B0TEST0000**] [**B0TEST0001**]",
      products,
    });
  }
  await page.addInitScript((conversation) => {
    localStorage.setItem(
      "aether.conversations.v1",
      JSON.stringify([{ id: "c1", title: "washing machines", threadId: "t1", messages: conversation }])
    );
  }, messages);
}
async function mockStreamingTurn(page: Page, products: unknown[]) {
  await page.route("**/agent/stream", async (route) => {
    const frames = [
      { type: "token", text: "Here are the washing machines in stock. " },
      { type: "token", text: "**Nictemaw Full Automatic** [**B0TEST0000**]" },
      {
        type: "result",
        payload: {
          answer: "Here are the washing machines in stock. **Nictemaw** [**B0TEST0000**]",
          question_relevancy: true,
          request_id: "req-live",
          thread_id: "t1",
          used_context: products,
        },
      },
      { type: "done" },
    ];
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      headers: { "Access-Control-Allow-Origin": "*" },
      body: frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join(""),
    });
  });
}
for (const size of VIEWPORTS) {
  test(`the composer stays put with a long conversation at ${size.name}`, async ({ page }) => {
    await page.setViewportSize({ width: size.width, height: size.height });
    await seedLongConversation(page);
    await page.goto("/");
    await page.waitForTimeout(600);
    const composer = page.locator("form").first();
    const box = await composer.boundingBox();
    expect(box, "the composer must be rendered").not.toBeNull();
    expect(
      box!.y + box!.height,
      `composer bottom ${Math.round(box!.y + box!.height)} is below the ${size.height}px viewport`
    ).toBeLessThanOrEqual(size.height + 1);
    expect(box!.y).toBeGreaterThan(size.height * 0.4);
    const documentScroll = await page.evaluate(
      () => document.documentElement.scrollHeight
    );
    expect(documentScroll, "the document must not grow taller than the screen").toBeLessThanOrEqual(
      size.height + 1
    );
    const scrolls = await page.evaluate(() => {
      const element = document.querySelector(".overflow-y-auto");
      return element ? element.scrollHeight > element.clientHeight + 4 : false;
    });
    expect(scrolls, "a long conversation must scroll inside the chat column").toBe(true);
  });
}
test("the side panel is not mounted below the desktop breakpoint", async ({ page }) => {
  for (const size of [
    { name: "phone", width: 390, height: 844 },
    { name: "tablet", width: 768, height: 1024 },
  ]) {
    await page.setViewportSize({ width: size.width, height: size.height });
    await seedLongConversation(page);
    await page.goto("/");
    await page.waitForTimeout(500);
    const panels = page.locator('[data-slot="resizable-panel"]');
    await expect(panels, `only the chat panel belongs at ${size.name}`).toHaveCount(1);
    const sidebar = page.locator('[data-slot="sidebar"]');
    const sidebarWidth = (await sidebar.count())
      ? (await sidebar.first().boundingBox())?.width ?? 0
      : 0;
    const box = await panels.first().boundingBox();
    expect(
      box!.width,
      `chat is only ${box!.width}px of ${size.width} at ${size.name} (sidebar ${sidebarWidth}px)`
    ).toBeGreaterThan(size.width - sidebarWidth - 4);
  }
});
test("the sidebar does not take a third of a tablet", async ({ page }) => {
  await page.setViewportSize({ width: 768, height: 1024 });
  await page.goto("/");
  await page.waitForTimeout(500);
  const box = await page.locator('[data-slot="sidebar"]').first().boundingBox();
  expect(box!.width, "the sidebar must collapse to its icon rail on a tablet").toBeLessThan(80);
});
test("the panel opens on its own when a turn returns results", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
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
  await mockStreamingTurn(page, makeProducts(5));
  await page.goto("/");
  await page.waitForTimeout(500);
  await expect(page.locator('[data-slot="resizable-panel"]')).toHaveCount(1);
  await page.getByRole("textbox", { name: /Message the shopping assistant/ }).fill("washing machines");
  await page.getByRole("button", { name: "Send message" }).click();
  const panels = page.locator('[data-slot="resizable-panel"]');
  await expect(panels).toHaveCount(2, { timeout: 10_000 });
  const box = await panels.nth(1).boundingBox();
  expect(box!.x + box!.width).toBeLessThanOrEqual(1440);
  expect(box!.width, "the panel must be a usable width, not a sliver").toBeGreaterThan(280);
  await expect(page.getByRole("heading", { name: "Products" }).last()).toBeVisible();
});
test("on a phone results wait to be asked for, and the composer stays usable", async ({ page }) => {
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
  await mockStreamingTurn(page, makeProducts(4));
  await page.goto("/");
  await page.waitForTimeout(500);
  await page.getByRole("textbox", { name: /Message the shopping assistant/ }).fill("washing machines");
  await page.getByRole("button", { name: "Send message" }).click();
  await page.waitForTimeout(800);
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.getByText("Here are the washing machines in stock.").first()).toBeVisible();
  const composer = await page.locator("form").first().boundingBox();
  expect(composer!.y + composer!.height).toBeLessThanOrEqual(845);
  await page.getByRole("button", { name: /4 results/ }).click();
  const sheet = page.getByRole("dialog");
  await expect(sheet).toBeVisible();
  await expect(sheet.getByRole("heading", { name: "Products" }).last()).toBeVisible();
  await expect
    .poll(async () => (await sheet.boundingBox())?.width ?? 0, { timeout: 5000 })
    .toBeGreaterThan(320);
  await page.keyboard.press("Escape");
  await expect(sheet).toBeHidden({ timeout: 5000 });
});
test("closing the panel keeps it closed until the next turn", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await seedLongConversation(page);
  await page.goto("/");
  await page.waitForTimeout(500);
  const panels = page.locator('[data-slot="resizable-panel"]');
  await expect(panels).toHaveCount(2);
  await page.getByRole("button", { name: "Close products panel" }).click();
  await expect(panels).toHaveCount(1);
  await page.waitForTimeout(300);
  await expect(panels, "a closed panel must stay closed").toHaveCount(1);
});
