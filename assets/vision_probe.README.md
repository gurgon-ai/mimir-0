# Vision probe — the deterministic test image for the `vision` benchmark dimension

Drop the generated image here as **`assets/vision_probe.png`**. The benchmark sends it to each
vision-capable model and scores the reply against the fixed answers below, so the image **must match
this spec exactly** — generate it from code (Pillow), never a diffusion render (those mangle text and
miscount shapes).

## Required content (exact)
- **Canvas:** 512×512, solid **white** background, nothing else but the two elements below.
- **Word:** the single word **`GLYPHON`** — uppercase, large, bold, black, centred horizontally in
  the upper third. (A made-up word, so a model must actually *read* it, not guess.)
- **Shapes:** exactly **three (3)** solid **red** circles, equal size, evenly spaced in one
  horizontal row across the lower half.

## Expected answers (what the scorer checks)
- "What word is written in this image?" → contains **glyphon** (case-insensitive) — tests OCR.
- "How many red circles are in the image?" → contains **3** / **three** — tests perception/counting.

Score = mean of the two checks (0, 0.5, or 1.0), multi-sampled like the other capability dims.

If you change the word or count, update the scorer in `cognition/benchmark.py` to match.
