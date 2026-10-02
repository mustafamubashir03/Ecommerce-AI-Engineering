import { expect, test } from "@playwright/test";
import { TokenBuffer } from "../src/lib/token-buffer";
async function withFakeFrames(run: (flushFrames: () => void) => Promise<void>) {
  const pending: FrameRequestCallback[] = [];
  const originalRequest = globalThis.requestAnimationFrame;
  const originalCancel = globalThis.cancelAnimationFrame;
  globalThis.requestAnimationFrame = ((callback: FrameRequestCallback) => {
    pending.push(callback);
    return pending.length;
  }) as typeof requestAnimationFrame;
  globalThis.cancelAnimationFrame = (() => {}) as typeof cancelAnimationFrame;
  try {
    const flushFrames = () => {
      const frame = pending.shift();
      if (frame) frame(0);
    };
    await run(flushFrames);
  } finally {
    globalThis.requestAnimationFrame = originalRequest;
    globalThis.cancelAnimationFrame = originalCancel;
  }
}
test("many tokens in one burst become a single update", async () => {
  await withFakeFrames(async (flushFrames) => {
    const applied: string[] = [];
    const buffer = new TokenBuffer((text) => applied.push(text));
    for (const token of ["Once", " upon", " a", " time"]) buffer.push(token);
    expect(applied, "nothing is applied before the frame runs").toHaveLength(0);
    flushFrames();
    expect(applied).toEqual(["Once upon a time"]);
  });
});
test("tokens spread over several frames are applied in order", async () => {
  await withFakeFrames(async (flushFrames) => {
    const applied: string[] = [];
    const buffer = new TokenBuffer((text) => applied.push(text));
    buffer.push("first");
    flushFrames();
    buffer.push(" second");
    flushFrames();
    buffer.push(" third");
    flushFrames();
    expect(applied).toEqual(["first", " second", " third"]);
  });
});
test("flush applies the tail immediately, so no text is lost", async () => {
  await withFakeFrames(async () => {
    const applied: string[] = [];
    const buffer = new TokenBuffer((text) => applied.push(text));
    buffer.push("partial answer");
    expect(buffer.pending).toBeGreaterThan(0);
    buffer.flush();
    expect(applied).toEqual(["partial answer"]);
    expect(buffer.pending).toBe(0);
    buffer.flush();
    expect(applied).toEqual(["partial answer"]);
  });
});
test("cancel drops the pending frame instead of applying it", async () => {
  await withFakeFrames(async (flushFrames) => {
    const applied: string[] = [];
    const buffer = new TokenBuffer((text) => applied.push(text));
    buffer.push("never delivered");
    buffer.cancel();
    flushFrames();
    expect(applied, "a cancelled turn must not write into a message that is gone").toEqual([]);
    expect(buffer.pending).toBe(0);
  });
});
test("pushing after a flush starts a new frame", async () => {
  await withFakeFrames(async (flushFrames) => {
    const applied: string[] = [];
    const buffer = new TokenBuffer((text) => applied.push(text));
    buffer.push("one");
    buffer.flush();
    buffer.push("two");
    flushFrames();
    expect(applied).toEqual(["one", "two"]);
  });
});
