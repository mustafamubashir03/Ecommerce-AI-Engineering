import { expect, test, type Page } from "@playwright/test";
function makeProducts(tag: string, count: number) {
  return Array.from({ length: count }, (_, index) => ({
    id: `B0${tag}${String(index).padStart(4, "0")}`,
    image_url: `https://placehold.co/400x300/e2e8f0/334155?text=${tag}${index}`,
    price: 20 + index,
    rating: 4,
    description: `${tag} product ${index}. Retrieved for the ${tag} question.`,
  }));
}
async function mockTurns(
  page: Page,
  answers: Record<string, { answer: string; products: ReturnType<typeof makeProducts> }>
) {
  let call = 0;
  await page.route("**/agent/stream", async (route) => {
    call += 1;
    const sent = JSON.parse(route.request().postData() ?? "{}") as { query?: string };
    const question = sent.query ?? "";
    const turn = answers[question];
    const answer = turn
      ? `${turn.answer} (call ${call})`
      : `UNEXPECTED question: ${question} (call ${call})`;
    const frames = [
      { type: "token", text: answer },
      {
        type: "result",
        payload: {
          answer,
          question_relevancy: true,
          request_id: `req-${call}`,
          thread_id: `t${call}`,
          used_context: turn ? turn.products : [],
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
function resultsPanel(page: Page) {
  return page.locator("h2", { hasText: /^Products$/ }).locator("xpath=../..");
}
function conversationText(page: Page) {
  return page.locator('[aria-label="Conversation"]').innerText();
}
async function ask(page: Page, question: string) {
  const composer = page.getByRole("textbox", { name: /Message the shopping assistant/ });
  await composer.fill(question);
  await composer.press("Enter");
}
test.beforeEach(async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
});
test("a second question is answered by its own reply, not the first one", async ({ page }) => {
  await mockTurns(page, {
    "suggest washing machines": {
      answer: "FIRST_REPLY about washing machines",
      products: makeProducts("AAA", 2),
    },
    "which is cheapest": {
      answer: "SECOND_REPLY about the cheapest one",
      products: makeProducts("BBB", 3),
    },
  });
  await page.goto("/");
  await ask(page, "suggest washing machines");
  await expect(page.getByText("FIRST_REPLY about washing machines")).toBeVisible();
  await ask(page, "which is cheapest");
  await expect(page.getByText("SECOND_REPLY about the cheapest one")).toBeVisible();
  const column = await conversationText(page);
  const order = [
    "suggest washing machines",
    "FIRST_REPLY",
    "which is cheapest",
    "SECOND_REPLY",
  ].map((part) => column.indexOf(part));
  expect(order.every((index) => index >= 0), `all four should be on screen: ${column}`).toBe(true);
  expect(order, `column read as: ${column}`).toEqual([...order].sort((a, b) => a - b));
});
test("the results panel shows the newest answer's products and nothing older", async ({ page }) => {
  await mockTurns(page, {
    "suggest washing machines": { answer: "FIRST_REPLY", products: makeProducts("AAA", 2) },
    "which is cheapest": { answer: "SECOND_REPLY", products: makeProducts("BBB", 3) },
  });
  await page.goto("/");
  const card = (panel: ReturnType<typeof resultsPanel>, name: string) =>
    panel.getByRole("button", { name, exact: true });
  await ask(page, "suggest washing machines");
  await expect(card(resultsPanel(page), "AAA product 0.")).toBeVisible();
  await ask(page, "which is cheapest");
  const panel = resultsPanel(page);
  await expect(card(panel, "BBB product 0.")).toBeVisible();
  await expect(card(panel, "AAA product 0.")).toHaveCount(0);
  await expect(card(panel, "BBB product 2.")).toBeVisible();
  await expect(card(panel, "BBB product 3.")).toHaveCount(0);
});
test("asking again on an older answer re-asks that answer's own question", async ({ page }) => {
  await mockTurns(page, {
    "suggest washing machines": {
      answer: "FIRST_REPLY about washing machines",
      products: makeProducts("AAA", 2),
    },
    "which is cheapest": {
      answer: "SECOND_REPLY about the cheapest one",
      products: makeProducts("BBB", 3),
    },
  });
  await page.goto("/");
  await ask(page, "suggest washing machines");
  await expect(page.getByText("FIRST_REPLY about washing machines")).toBeVisible();
  await ask(page, "which is cheapest");
  await expect(page.getByText("SECOND_REPLY about the cheapest one")).toBeVisible();
  const firstAnswer = page.locator('[id^="message-"]', { hasText: "FIRST_REPLY about washing" });
  await firstAnswer.getByRole("button", { name: "Ask the same question again" }).click();
  const newest = page.locator('[id^="message-"]').last();
  await expect(newest).toContainText("FIRST_REPLY");
  await expect(newest).toContainText("(call 3)");
  await expect(page.getByText("UNEXPECTED question")).toHaveCount(0);
  const column = await conversationText(page);
  const asked = column.split("suggest washing machines").length - 1;
  expect(asked, `"suggest washing machines" appears ${asked} times`).toBe(1);
});
