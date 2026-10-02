export class TokenBuffer {
  private queued = "";
  private frame: number | null = null;
  private readonly hasFrame: boolean;
  private readonly apply: (text: string) => void;
  constructor(apply: (text: string) => void) {
    this.apply = apply;
    this.hasFrame = typeof requestAnimationFrame === "function";
  }
  push = (text: string): void => {
    this.queued += text;
    if (this.frame !== null) return;
    this.frame = this.hasFrame
      ? requestAnimationFrame(this.drain)
      : (setTimeout(this.drain, 16) as unknown as number);
  };
  flush = (): void => {
    this.clear();
    this.drain();
  };
  cancel = (): void => {
    this.queued = "";
    this.clear();
  };
  get pending(): number {
    return this.queued.length;
  }
  private drain = (): void => {
    this.frame = null;
    if (!this.queued) return;
    const text = this.queued;
    this.queued = "";
    this.apply(text);
  };
  private clear(): void {
    if (this.frame === null) return;
    if (this.hasFrame) cancelAnimationFrame(this.frame);
    else clearTimeout(this.frame as unknown as ReturnType<typeof setTimeout>);
    this.frame = null;
  }
}
