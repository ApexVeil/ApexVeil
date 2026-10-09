# Appearance and assisted geometry experiment — 2026-10-10

This is development on the same nine-image private regression set, not a held-out
benchmark. It reuses the saved SAM masks and original geometry from the October 9
run. No new detector training, GPU inference, Photoshop or LLM review was used.

## Changes and observations

The earlier neutral black point removed the orange veiling glare in the backlit
photo. The new compositor separates a diffuse light estimate from a smooth RGB
floor. Local dark ink constrains the floor's intensity; surrounding dark pixels
supply a conservative color prior so green plate albedo is not transferred to a
black replacement. This remains a heuristic, not recovered physical materials.

A tested earlier variant estimated color solely from the old plate. It visibly
contaminated a black replacement with green, especially with partial visibility
or unresolved glyphs. The current version uses a neutral floor when less than
65% of the plane is visible or estimated image softness exceeds 1.5 px, or the projected short edge is below
48 pixels, then
applies the scene color prior. This deliberately gives up uncertain color detail.
The surrounding region can itself contain colored materials; this prior still
needs a larger, diverse evaluation set.

- **Glare (`0205`)**: visual comparison shows the orange veil continues into the
  new black face instead of stopping at its boundary. It is not a calibrated
  glare-reconstruction score.
- **Defocus (`0093`)**: retains the existing roughly 8.5 px softness estimate;
  the unreliable colored ink-floor fit is disabled. Optical edge matting is still
  missing, and the result remains a preview requiring review.
- **Near-black (`0185`)**: removing a hard minimum exposure gain prevents a new
  white logo being artificially lifted out of darkness. An explicit experimental
  preview was generated; the production image CLI still flags and skips this
  case by default.
- **Foreground and clipped geometry (`0100`, `9998`, `0287`)**: supplied manually
  estimated full-plane corners to isolate geometry from visibility. The visible
  masks stayed unchanged and occluders were not painted over. These corners are
  neither automatic recovery nor ground truth; green/soft edge residuals remain.
  The hard shadow in `9998` is not yet faithfully reproduced on the replacement.

All nine controlled renders changed zero working-buffer pixels outside their
supplied visible masks. This does not prove the masks are correct or guarantee
encoded JPEG/video pixels remain bit-identical. Per-image render calls took
about 0.43–0.85 s in this run (median about 0.57 s), excluding model inference,
image loading, output encoding and report writing. Do not call this end-to-end
latency. The default image pipeline's quality gates have not been relaxed.

## Reproduction

`scripts/evaluate_compositor.py` accepts a saved report directory, original image
directory, template, optional per-image manual corner JSON, and an output folder.
It marks every output as an experimental preview requiring review, including
cases that the production pipeline deliberately skips. It does not load a model.

```bash
python scripts/evaluate_compositor.py \
  --reports PATH_TO_SAVED_IMAGE_REPORTS \
  --images PATH_TO_ORIGINAL_IMAGES \
  --template PATH_TO_RGBA_TEMPLATE \
  --overrides PATH_TO_OPTIONAL_MANUAL_CORNERS_JSON \
  --output PATH_TO_EXPERIMENT
```

Private final comparisons and metrics are in
`outputs/image-compositor-20261010-final/`. The inputs, manually estimated
corners, templates and outputs are intentionally excluded from the public repo.

## Regression checks

Known synthetic scenes check colored veil transfer without repeated glyphs,
near-black exposure, conservative partial visibility, broad shadow contrast and
suppression of green material contamination under neutral surroundings. Existing
image tests still check protected occluders, rounded alpha and multi-target
selection. These numerical checks complement, rather than replace, visual review.

## Video feedback into the shared compositor

Small 720p video surfaces exposed two additional errors before final export:
unresolved ink raised the floor toward the old green face, and ratios of nearly
black context RGB values amplified compression noise into a purple tint. The
shared compositor now uses a low-dimensional light field for unresolved ink,
limits its black floor using nearby dark-scene evidence, and uses neutral tint
when context linear luminance is below 0.003. These bounds are explicitly
heuristic. They do not solve high-frequency moving shadows or optical matting.
Additional tests cover these observed regressions. The final still batch was
rerun after this change; earlier development outputs are retained separately.
