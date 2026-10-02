import { expect, test, type Page } from "@playwright/test";
const PRODUCTS = Array.from({ length: 9 }, (_, index) => ({
  id: `B0TEST${String(index).padStart(4, "0")}`,
  title:
    index === 0
      ? "Portable Washing Machine Nictemaw Full Automatic Compact 2 in 1 Laundry Washer with 10 Wash Programs and Drainage Pump"
      : `Portable Washing Machine Model ${index} with a deliberately long product name to stress min-content width`,
  imageUrl: `https://placehold.co/400x300/e2e8f0/334155?text=Product+${index}`,
  price: index === 3 ? null : 19.99 + index * 10,
  rating: 3 + (index % 2),
  reason: "Matches the capacity and budget you asked about.",
  sourceMessageId: "m1",
}));
const SIZES = [
  { name: "mobile-sm", width: 360, height: 780 },
  { name: "mobile", width: 390, height: 844 },
  { name: "tablet", width: 768, height: 1024 },
  { name: "laptop", width: 1024, height: 768 },
  { name: "laptop-wide", width: 1280, height: 800 },
  { name: "desktop", width: 1440, height: 900 },
  { name: "wide", width: 1920, height: 1080 },
];
async function seedResults(page: Page) {
  await page.addInitScript((products) => {
    const conversation = {
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
          content: "Here are the washing machines in stock - **Nictemaw Full Automatic** [**B0TEST0000**] - 1.47 cu.ft [**B0TEST0001**]",
          products,
        },
      ],
    };
    localStorage.setItem("aether.conversations.v1", JSON.stringify([conversation]));
  }, PRODUCTS);
}
async function overflowingElements(page: Page) {
  return page.evaluate(() => {
    const limit = document.documentElement.clientWidth;
    const offenders: { tag: string; cls: string; width: number; right: number }[] = [];
    for (const element of Array.from(document.querySelectorAll("*"))) {
      const rect = element.getBoundingClientRect();
      if (rect.width === 0) continue;
      if (rect.right > limit + 1) {
        offenders.push({
          tag: element.tagName.toLowerCase(),
          cls: (element.className || "").toString().slice(0, 70),
          width: Math.round(rect.width),
          right: Math.round(rect.right),
        });
      }
    }
    return {
      limit,
      scrollWidth: document.documentElement.scrollWidth,
      offenders: offenders.slice(0, 6),
    };
  });
}
for (const size of SIZES) {
  test(`no horizontal overflow at ${size.name} (${size.width}px)`, async ({ page }) => {
    await page.setViewportSize({ width: size.width, height: size.height });
    await seedResults(page);
    await page.goto("/");
    await page.waitForTimeout(400);
    const result = await overflowingElements(page);
    expect(
      result.scrollWidth,
      `document scrollWidth ${result.scrollWidth} exceeds viewport ${result.limit}; offenders: ${JSON.stringify(result.offenders, null, 2)}`
    ).toBeLessThanOrEqual(result.limit + 1);
  });
}
test("the products panel stays inside the viewport at every width", async ({ page }) => {
  for (const size of SIZES) {
    await page.setViewportSize({ width: size.width, height: size.height });
    await seedResults(page);
    await page.goto("/");
    await page.waitForTimeout(400);
    const resultsButton = page.getByRole("button", { name: /Results/ });
    if (await resultsButton.isVisible().catch(() => false)) {
      await resultsButton.first().click();
      await page.waitForTimeout(500);
    }
    const panel = page.locator('[data-slot="resizable-panel"]').last();
    if ((await panel.count()) === 0 || !(await panel.isVisible().catch(() => false))) {
      continue;
    }
    const box = await panel.boundingBox();
    if (!box) continue;
    expect(
      box.x + box.width,
      `panel right edge ${Math.round(box.x + box.width)} exceeds viewport ${size.width} at ${size.name}`
    ).toBeLessThanOrEqual(size.width + 1);
  }
});
