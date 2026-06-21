"""Generate the deterministic vision-benchmark probe image (assets/vision_probe.png).

Run: ``python assets/make_vision_probe.py``. Needs Pillow (a dev-only dependency — the committed PNG
is the artifact; the runtime never imports this). The content must match vision_probe.README.md and
the scorer in cognition/benchmark.py exactly: the word GLYPHON + three red circles on white.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = 512
WORD = "GLYPHON"
N_CIRCLES = 3
OUT = Path(__file__).with_name("vision_probe.png")


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("C:/Windows/Fonts/arialbd.ttf", "arialbd.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise SystemExit("No bold TrueType font found — install one or edit _font().")


def main() -> None:
    img = Image.new("RGB", (SIZE, SIZE), "white")
    d = ImageDraw.Draw(img)

    # The word: large, bold, black, centred horizontally in the upper third.
    font = _font(96)
    box = d.textbbox((0, 0), WORD, font=font)
    w = box[2] - box[0]
    d.text(((SIZE - w) / 2 - box[0], SIZE * 0.18 - box[1]), WORD, fill="black", font=font)

    # Exactly N solid red circles, equal size, evenly spaced in one row across the lower half.
    r = 42
    cy = int(SIZE * 0.68)
    gap = SIZE // (N_CIRCLES + 1)
    for i in range(1, N_CIRCLES + 1):
        cx = gap * i
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(220, 30, 30))

    img.save(OUT)
    print(f"wrote {OUT} ({img.size[0]}x{img.size[1]}) — word={WORD!r}, circles={N_CIRCLES}")


if __name__ == "__main__":
    main()
