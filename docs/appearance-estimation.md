# Deterministic appearance and boundary estimation

ApexVeil estimates material appearance and capture degradation from the source,
then renders the approved asset deterministically. This is a photograph-derived
approximation, not recovered physical lighting or calibrated camera response.
No model, generated artwork, training or per-image color thresholds are used.

`AppearanceEstimate` contains exposure gain, a canonical linear RGB additive
floor, illumination tint and a spatial field. `estimate_appearance` returns that
object; `render_appearance(asset, estimate)` applies it. The existing temporal
parameter callback still filters gain, floor and tint without averaging spatial
shadows. A future estimator can supply those parameters while keeping the same
approved logo and renderer.

Ink removal precedes a robust diffuse exposure percentile, so isolated glints
do not determine logo brightness. The smooth floor uses dark ink when the source
has enough visibility and spatial resolution. Unresolved ink falls back to a
bounded neutral/context floor. Masked foreground does not enter the spatial fit.
Reflective license plates and a matte replacement have different material
responses, so exposure remains uncertain even when source segmentation is good.

`BoundaryEstimate` keeps local edit support, visibility and protected foreground
separate. A scale-bounded band may add pixels only when observed surface colors
fit them better than competing surrounding colors. Missing interior pixels remain
protected. Continuous correction coverage uses an inward distance ramp only in
the narrow exterior-plane band, with its width derived from capture blur. This
attenuates isolated repair dots and mask staircase artifacts without spreading
edits outside declared support. Interior occlusion boundaries retain their own
visibility and protection. Low-confidence exterior mixtures can retain some old
color rather than receiving a full-strength background reconstruction. The estimator does not infer a foreground-free full plane by filling
holes. An explicit protected mask takes precedence over all additions.

Rounded transparent artwork and blurred optical edges cannot expose the old
plate beneath them. The compositor obtains a local background approximation by
inpainting the inferred plane plus observed support, excluding protected pixels;
it uses this approximation only at editable corners and optical transitions.
The reconstruction interior is hidden by the opaque asset. This operation cannot
recover an unseen bumper, resolve a mistaken quadrilateral or reliably distinguish
surface screws from foreground. Ambiguous inferred holes, substantial expansion or a large reconstructed
background fraction are marked `needs_review` in boundary diagnostics.

The actual declared edit support and protection are available through the
`edit_support_out` dictionary (`bbox_xyxy`, local `support`, local `protected`).
Metadata remains JSON-compatible. The correct invariants are exact equality
outside declared edit support and exact equality on protected foreground. The raw
segmentation boundary is an observation and is not the edit-support invariant.

`DegradationEstimate` measures isotropic edge spread with modest presmoothing and
flat-patch high-pass MAD noise. Flat-patch variances provide a bounded encoded
signal-dependent noise approximation, together with sample count and reliability.
Noise is synthesized from an explicit seed and blended with material alpha. Video
callers should pass a different deterministic frame seed; reusing the still default
seed can cause correlated texture. Directional motion kernels, chromatic sensor
noise and physical flare decomposition are not currently recovered.

CPU synthetic checks cover known blur/noise, rounded corner decontamination,
foreground branches connected to the exterior, inferred boundary additions,
parameter injection and declared-support invariants. Private evaluation artifacts
must remain ignored and report source hash, template hash, fixed geometry source,
estimated parameters and full-frame/crop comparisons. Assisted quadrilaterals are
comparisons, never ground truth. Passing these checks is not a photorealism score.
