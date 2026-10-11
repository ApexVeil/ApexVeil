# Image pipeline (experimental)

The `image` command performs segmentation, planar geometry estimation, local
appearance matching, and either custom replacement or redaction. It does not
invoke an AI reviewer. Its outputs are previews requiring human inspection, not
certified anonymization or production-ready photorealism.

## Run

```sh
conda activate apexveil
CUDA_VISIBLE_DEVICES=0 apexveil image /path/to/photo.jpg \
  --template /path/to/custom-rgba.png --output outputs/my-photo --comparison

CUDA_VISIBLE_DEVICES=0 apexveil image /path/to/photo-directory \
  --mode blur --strength 0.12 --output outputs/private-batch
```

Use an explicitly selected GPU. The model stays loaded across a directory batch.
The existing pinned backend audits every learned tensor before inference. No
other project's source, environment, or private migration scripts are imported.

Output subdirectories contain `report.json`, visible masks, geometry previews,
local comparison sheets, declared edit-support masks and lossless PNG effects.
With `--comparison`, `index.html` shows RGB | all raw SAM masks | the latest
Replace or Blur result in full and detail views. Source files are untouched.
EXIF orientation is applied before processing and an existing ICC profile is
carried into the PNG. Carrying an ICC profile is not full HDR/wide-gamut color
management; the current compositor uses an sRGB-like transfer approximation.

## Multiple surfaces

A single detected surface becomes the automatic replacement target. More than
one candidate requires `--target-id` or a per-file override. Inspect the numbered
`geometry.jpg` and JSON before selecting. IDs are local to an image; they are not
cross-image identities. The chosen target is replaced and other non-excluded
candidates are blurred. Blur/mosaic mode processes all non-excluded detections.

```json
{
  "photo.jpg": {
    "target_id": 0,
    "exclude_ids": [2],
    "quads": {"0": [[110, 70], [410, 80], [400, 180], [100, 170]]}
  }
}
```

```sh
CUDA_VISIBLE_DEVICES=0 apexveil image /path/to/photo.jpg \
  --template /path/to/custom-rgba.png --overrides /path/to/overrides.json \
  --output outputs/selected
```

Coordinates are pixels in the EXIF-oriented source. Excluded candidates are not
edited. Detection cannot guarantee that every plate has been found.

## Geometry and visibility

SAM3.1 first detects the complete image, then automatically refines each candidate
in a local crop. Visible mask holes are retained. A separate convex boundary and
robust line fit estimate full-plane corners; image gradients refine these corners
within angle/area guards. Fitting a plane does **not** fill the compositing mask.

Missing or unsupported complete-plane geometry is rejected. A valid but small,
dark or otherwise uncertain surface can render with a `needs_review` status.
Strong perspective is not rejected merely because its projected aspect differs
from the artwork. `--allow-uncertain` may permit an existing flagged hypothesis;
it cannot invent absent corners. Supplied corners remain explicit manual input. These geometric checks
are heuristic: they do not prove that a segmentation correctly excludes an
occluder or that a partially hidden plate's full extent was recovered.

## Appearance and alpha

The transparent template is cropped to its silhouette, with its own aspect ratio
retained. Alpha values at or above 248 become opaque to prevent old glyphs
showing through near-opaque artwork; rounded transparent edges are retained.
A premultiplied linear-light warp avoids dark fringes. A low-frequency luminance
field, black point, and conservative neutral tint approximate the local light.
A bounded boundary estimator declares additional editable support when local
surface/background evidence permits it. Native visibility is retained separately.
Closed visibility holes and explicitly protected foreground remain unchanged;
pixels outside declared support are also unchanged before encoding. These exact
invariants do not prove that automatically inferred support is physically right.
Background repair abstains when nearby donor layers disagree; that can leave an
old colored rim requiring correction instead of inventing a damaged new edge.

The degradation provider estimates isotropic replacement softness and noise.
Automatic image and video composition use that same provider result. Strong
defocus estimates are flagged for review. `--blur-sigma` is an explicit numeric
override in source pixels, including zero; it is not silently increased by a
second estimator. This is not a calibrated depth-of-field or directional-motion
blur model; low contrast, nearby structure and sensor noise can bias it.
Sharp moving shadows, strong lens flare, defocused foreground matting, and
physically different reflective materials are not solved by this approximation.

Blur/mosaic use the original pixels and preserve their broad lighting. Strength
is a fraction of the detected region height. Mosaic is currently a still-image
operation; there is no stable video grid yet. None of the modes verifies OCR
unreadability.

## Verification

CPU tests use synthetic surfaces and check foreground/outside-pixel preservation,
transparent corners, multi-target policy, uncertainty handling, source-file
integrity and ICC export. No model download is needed for those tests.

Real-image experiments live in ignored `outputs/` directories. Do not infer
video quality or generalization from a small development set.
