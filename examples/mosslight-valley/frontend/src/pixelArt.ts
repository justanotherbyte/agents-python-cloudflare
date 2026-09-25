import type { Art } from "./assets/pixel";

const urls = new WeakMap<Art | readonly Art[], string>();

/** CSS `url(...)` for one image, or for frames laid side by side as an animation strip. */
export function artUrl(art: Art | readonly Art[]): string {
  const cached = urls.get(art);
  if (cached) return cached;

  const frames = "rows" in art ? [art] : art;
  const width = frames[0].rows[0].length;
  const canvas = document.createElement("canvas");
  canvas.width = width * frames.length;
  canvas.height = frames[0].rows.length;
  const context = canvas.getContext("2d")!;
  frames.forEach((frame, index) => {
    frame.rows.forEach((row, y) => {
      [...row].forEach((key, x) => {
        if (key === ".") return;
        context.fillStyle = frame.palette[key];
        context.fillRect(index * width + x, y, 1, 1);
      });
    });
  });

  const url = `url(${canvas.toDataURL()})`;
  urls.set(art, url);
  return url;
}
