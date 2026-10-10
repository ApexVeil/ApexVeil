# Automatic estimation with a deterministic compositor

ApexVeil uses SAM to observe visible surfaces, then estimates the quantities needed
by an artwork-preserving renderer. This version adds no neural inference after
SAM, paid API calls, or image generation. It does not claim to recover a physically
unique scene from one photograph.

## Shared image and video contract

1. Preserve each native mask and identity as diagnostic evidence.
2. Estimate a separate projective plane. A visible silhouette is not proof of a
   complete plate; frame clipping and foreground boundaries can be ambiguous.
3. For offline video, transport supported observations in both temporal
   directions using measured motion and robust consensus within one shot and
   identity. Missing masks remain missing, never silently filled or unioned.
4. Estimate local edit support, protected foreground, exposure, additive light,
   broad illumination, optical blur and noise from the current source.
5. Warp the supplied RGBA artwork and composite in linear light. The artwork's
   lettering and transparent silhouette are not synthesized by a model.
6. Record actual support and uncertainty. Pixels outside declared support and
   explicitly protected foreground are immutable before video encoding.

The last condition is an exact compositing invariant. It is not a claim that an
inferred support mask or foreground estimate is always correct. Video compression
changes pixels independently of the compositor and is checked separately.

## Replaceable estimators

`apexveil.estimation.Estimators` contains geometry, boundary, degradation and
appearance providers. Each implements `estimate()`. `get_default_estimators()`
selects stateless deterministic adapters; importing the bundle loads no neural
runtime or model weights. Image processing and video composition accept an
optional bundle through their Python APIs.

Outputs are `SurfaceGeometry`, `BoundaryEstimate`, `DegradationEstimate` and
`AppearanceEstimate`. Their arrays carry spatial information; their diagnostics
remain JSON serializable. Provider identity and whether it is learned are recorded
for reproducibility. `AppearanceEstimate` contains gain, a spatial RGB floor,
tint and an illumination field. It does not provide replacement artwork pixels.

A future learned provider can predict the same quantities without replacing the
renderer. Its evaluation must still cover geometry, foreground preservation,
source support, artwork identity and temporal consistency. Merely matching the
Python signature does not certify a model's output quality.

## Evaluation and data collection

Cache hashes bind native SAM outputs to the decoded source. Every automatic
replay recomputes geometry and appearance; target selection is allowed, but
manual corner corrections and per-image appearance parameters are excluded.
`python scripts/evaluate_cached_images.py --help` exposes this CPU-only replay.
The diagnostic export is RGB | all native SAM masks | ApexVeil Replace or Blur.
Excluded model detections remain visible in the middle panel.

Keep separate development cases and new-scene checks, including severe angle,
small plates, motion blur, defocus, clipping, foreground occlusion, deep shadows,
flare and multiple targets. Assess both complete images and native-resolution
crops; inspect clips in motion as well as individual frames. Track failures by
stage and scene, and retain unedited frames and rejected hypotheses.

A render count is neither detection recall nor a perceptual quality score. A zero
out-of-support change count does not prove a realistic boundary. A successful
synthetic temporal recovery does not establish recovery on every real occlusion.
Collect real user corrections and their provenance before deciding which learned
estimator would remove the largest remaining error source.

## Current limitations

- A person or frame edge can hide the plate's true corners. Boundary support
  alone cannot confirm complete-plane semantics.
- Local edge reconstruction approximates missing background. Strong green/blue
  contamination, rounded-corner mismatch and occluder edges may leave visible
  seams; uncertain output requires review.
- Material reflectance and illumination are not uniquely separable from one
  source. The current exposure and shadow estimates are bounded approximations.
- Degradation estimation currently models scalar blur and noise. Directional
  motion blur, rolling shutter and nonlinear camera processing remain open.
- Native SAM misses are not repaired by the offline geometry module. Reliable
  extra segmentation/tracking observations are needed before those frames can
  be edited safely.
- AI quality review and neural estimator training are not implemented by this
  change. No output is labeled reviewed by an AI that was never called.
