import { expect, test, type Page } from "@playwright/test";
const RESULT = {
  answer: "Here are the machines [B0TEST0000]",
  question_relevancy: true,
  request_id: "req-1",
  thread_id: "t1",
  trace_id: "01a0f1c2-f417-7071-b203-1b05192f811c",
  used_context: [],
};
async function stream(page: Page, frames: unknown[]) {
  await page.route("**/agent/stream", async (route) => {
    const body = frames.map((f) => `data: ${JSON.stringify(f)}\n\n`).join("");
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      headers: { "Access-Control-Allow-Origin": "*" },
      body,
    });
  });
  await page.goto("/");
  await page.getByRole("textbox", { name: /Message the shopping assistant/ }).fill("washing machines");
  await page.getByRole("button", { name: "Send message" }).click();
}
test("many token events render as one answer", async ({ page }) => {
  await stream(page, [
    ...["Here ", "are ", "the ", "machines ", "[B0TEST0000]"].map((text) => ({ type: "token", text })),
    { type: "result", payload: RESULT },
    { type: "done" },
  ]);
  await expect(page.getByText("Here are the machines", { exact: false })).toBeVisible();
});
test("a status, tokens and a result in one stream all reach the page", async ({ page }) => {
  await stream(page, [
    { type: "status", text: "Checking what you are asking about" },
    { type: "token", text: "Here are the machines" },
    { type: "result", payload: RESULT },
    { type: "done" },
  ]);
  await expect(page.getByText("Here are the machines", { exact: false })).toBeVisible();
  await expect(page.getByText("req-1").first()).toBeVisible();
});
test("a status never outlives the answer it precedes", async ({ page }) => {
  await stream(page, [
    { type: "status", text: "Looking through the products in stock" },
    { type: "token", text: "Here are the machines" },
    { type: "result", payload: RESULT },
    { type: "done" },
  ]);
  await expect(page.getByText("Here are the machines", { exact: false })).toBeVisible();
  await expect(page.getByText("Looking through the products in stock")).toHaveCount(0);
});
test("a stream that ends without a result does not strand the turn", async ({ page }) => {
  await stream(page, [
    { type: "status", text: "Looking through the products in stock" },
    { type: "token", text: "Here are the machines" },
    { type: "done" },
  ]);
  const composer = page.getByRole("textbox", { name: /Message the shopping assistant/ });
  await expect(composer).toBeEditable();
  await composer.fill("a second question");
  await expect(composer).toHaveValue("a second question");
});
test("an error frame is still reported as one", async ({ page }) => {
  await stream(page, [
    { type: "status", text: "Looking through the products in stock" },
    { type: "error", message: "The model provider is rate limiting requests right now.", status: 429 },
    { type: "done" },
  ]);
  await expect(page.getByText("could not answer")).toBeVisible();
});
test("an unknown event type does not break the stream", async ({ page }) => {
  await stream(page, [
    { type: "status", text: "Looking through the products in stock" },
    { type: "something_new", payload: { anything: true } } as unknown as Record<string, unknown>,
    { type: "token", text: "Here are the machines" },
    { type: "result", payload: RESULT },
    { type: "done" },
  ]);
  await expect(page.getByText("Here are the machines", { exact: false })).toBeVisible();
});
