import { expect, test, type Page } from "@playwright/test";
const TRACE = "01a0f1c2-f417-7071-b203-1b05192f811c";
interface FeedbackBody {
  trace_id: string;
  feedback_score: number | null;
  feedback_text: string;
  thread_id: string | null;
  feedback_source_type: string;
}
async function seedAnswer(page: Page, traceId: string | null) {
  await page.addInitScript((trace) => {
    localStorage.setItem(
      "aether.conversations.v1",
      JSON.stringify([
        {
          id: "c1",
          title: "washing machines",
          threadId: "t1",
          messages: [
            {
              id: "u1",
              role: "user",
              content: "which washing machines do you have?",
              status: "done",
              requestId: null,
              error: null,
              products: [],
            },
            {
              id: "a1",
              role: "assistant",
              status: "done",
              requestId: "req-1",
              error: null,
              content: "Here are the washing machines. [**B0TEST0000**]",
              products: [],
              traceId: trace,
              vote: null,
              feedbackError: null,
            },
          ],
        },
      ])
    );
  }, traceId);
}
async function captureFeedback(page: Page, status = 200) {
  const sent: FeedbackBody[] = [];
  await page.route("**/feedback/", async (route) => {
    const body = route.request().postData();
    if (body) sent.push(JSON.parse(body) as FeedbackBody);
    await route.fulfill({
      status,
      contentType: "application/json",
      headers: { "Access-Control-Allow-Origin": "*" },
      body: JSON.stringify(
        status === 200
          ? { request_id: "req-feedback", status: "recorded" }
          : { detail: { error: "The feedback could not be recorded. Please try again." } }
      ),
    });
  });
  return sent;
}
async function captureFeedbackSaying(page: Page, reported: string) {
  const sent: FeedbackBody[] = [];
  await page.route("**/feedback/", async (route) => {
    const body = route.request().postData();
    if (body) sent.push(JSON.parse(body) as FeedbackBody);
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      headers: { "Access-Control-Allow-Origin": "*" },
      body: JSON.stringify({ request_id: "req-feedback", status: reported }),
    });
  });
  return sent;
}
async function revealActions(page: Page) {
  await page.locator("#message-a1").hover();
  await expect(page.getByRole("button", { name: "This answer was helpful" })).toBeVisible();
}
test("a thumbs up files a positive score against the answer's trace", async ({ page }) => {
  await seedAnswer(page, TRACE);
  const sent = await captureFeedback(page);
  await page.goto("/");
  await revealActions(page);
  await page.getByRole("button", { name: "This answer was helpful" }).click();
  await expect.poll(() => sent.length).toBe(1);
  expect(sent[0]).toMatchObject({
    trace_id: TRACE,
    feedback_score: 1,
    feedback_text: "",
    thread_id: "t1",
    feedback_source_type: "api",
  });
});
test("a thumbs down files a negative score", async ({ page }) => {
  await seedAnswer(page, TRACE);
  const sent = await captureFeedback(page);
  await page.goto("/");
  await revealActions(page);
  await page.getByRole("button", { name: "This answer was not helpful" }).click();
  await expect.poll(() => sent.length).toBe(1);
  expect(sent[0].feedback_score).toBe(-1);
});
test("a cast vote stays pressed", async ({ page }) => {
  await seedAnswer(page, TRACE);
  await captureFeedback(page);
  await page.goto("/");
  await revealActions(page);
  const up = page.getByRole("button", { name: "This answer was helpful" });
  await up.click();
  await expect(up).toHaveAttribute("aria-pressed", "true");
});
test("a comment is filed as text and the box closes", async ({ page }) => {
  await seedAnswer(page, TRACE);
  const sent = await captureFeedback(page);
  await page.goto("/");
  await revealActions(page);
  await page.getByRole("button", { name: "Leave a comment about this answer" }).click();
  const field = page.getByRole("textbox", { name: "Comment about this answer" });
  await field.fill("the third one is not a washing machine");
  await page.getByRole("button", { name: "Send comment" }).click();
  await expect.poll(() => sent.length).toBe(1);
  expect(sent[0]).toMatchObject({ feedback_text: "the third one is not a washing machine" });
  expect(sent[0].feedback_score).toBeNull();
  await expect(field).toBeHidden();
});
test("an empty comment is not sent", async ({ page }) => {
  await seedAnswer(page, TRACE);
  const sent = await captureFeedback(page);
  await page.goto("/");
  await revealActions(page);
  await page.getByRole("button", { name: "Leave a comment about this answer" }).click();
  await expect(page.getByRole("button", { name: "Send comment" })).toBeDisabled();
  expect(sent).toHaveLength(0);
});
test("a refused save is reported and the thumb stays unpressed", async ({ page }) => {
  await seedAnswer(page, TRACE);
  await captureFeedback(page, 502);
  await page.goto("/");
  await revealActions(page);
  const up = page.getByRole("button", { name: "This answer was helpful" });
  await up.click();
  await expect(page.getByRole("status")).toContainText("could not be recorded");
  await expect(up).toHaveAttribute("aria-pressed", "false");
});
test("a 200 that did not record anything is not treated as success", async ({ page }) => {
  await seedAnswer(page, TRACE);
  await captureFeedbackSaying(page, "queued");
  await page.goto("/");
  await revealActions(page);
  const up = page.getByRole("button", { name: "This answer was helpful" });
  await up.click();
  await expect(page.getByRole("status")).toContainText("was not recorded");
  await expect(up).toHaveAttribute("aria-pressed", "false");
});
test("a comment is confirmed once it is recorded", async ({ page }) => {
  await seedAnswer(page, TRACE);
  const sent = await captureFeedback(page);
  await page.goto("/");
  await revealActions(page);
  await page.getByRole("button", { name: "Leave a comment about this answer" }).click();
  await page.getByRole("textbox", { name: "Comment about this answer" }).fill("the third one is not a washing machine");
  await page.getByRole("button", { name: "Send comment" }).click();
  await expect.poll(() => sent.length).toBe(1);
  await expect(page.getByRole("status")).toContainText("Comment sent");
});
test("a comment that failed is not thanked for", async ({ page }) => {
  await seedAnswer(page, TRACE);
  await captureFeedback(page, 502);
  await page.goto("/");
  await revealActions(page);
  await page.getByRole("button", { name: "Leave a comment about this answer" }).click();
  await page.getByRole("textbox", { name: "Comment about this answer" }).fill("the third one is not a washing machine");
  await page.getByRole("button", { name: "Send comment" }).click();
  await expect(page.getByRole("status")).toContainText("could not be recorded");
  await expect(page.getByRole("status")).not.toContainText("Comment sent");
});
test("an answer with no trace explains itself and cannot be rated", async ({ page }) => {
  await seedAnswer(page, null);
  const sent = await captureFeedback(page);
  await page.goto("/");
  await expect(page.getByText("cannot be rated")).toBeVisible();
  await revealActions(page);
  const up = page.getByRole("button", { name: "This answer was helpful" });
  await expect(up).toBeDisabled();
  expect(sent, "nothing can be filed without a trace, so nothing should be posted").toHaveLength(0);
});
test("a conversation saved before feedback existed still opens", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem(
      "aether.conversations.v1",
      JSON.stringify([
        {
          id: "c1",
          title: "old",
          threadId: "t1",
          messages: [
            { id: "a1", role: "assistant", status: "done", requestId: "req-1", content: "An old answer.", products: [] },
          ],
        },
      ])
    );
  });
  await page.goto("/");
  await expect(page.getByText("An old answer.")).toBeVisible();
  await expect(page.getByText("cannot be rated")).toBeVisible();
  await revealActions(page);
  await expect(page.getByRole("button", { name: "This answer was helpful" })).toBeDisabled();
});
