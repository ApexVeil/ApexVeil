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
fit them better than competing surrounding colors. Every enclosed missing component of observed visibility is protected,
including holes beside the geometric edge. Outside-connected missing components
retain conservative interior protection; only their exterior strips may receive
color-supported additions. Continuous correction coverage uses an inward distance ramp only in
the narrow exterior-plane band, with its width derived from capture blur. This
attenuates isolated repair dots and mask staircase artifacts without spreading
edits outside declared support. Interior occlusion boundaries retain their own
visibility and protection. Low-confidence exterior mixtures can retain some old
color rather than receiving a full-strength background reconstruction. The estimator does not infer a foreground-free full plane by filling
holes. An explicit protected mask takes precedence over all additions.

Rounded transparent artwork and blurred optical edges can expose the old
surface beneath them. Local background reconstruction is allowed only when two
donor choices agree: the immediate exterior of inferred plane/visible
support, and the exterior after a bounded optical donor gap. Their encoded Lab
disagreement is compared with source-derived exterior residual noise/texture and
a small quantization tolerance. A whole-surface decision avoids patchwork repair.
When they disagree, reconstruction is disabled for that surface and ordinary
source/asset alpha compositing is retained. In the same uncertain outer band,
a continuous geometric inward confidence ramp retains the observed ridge instead
of chopping it with a jagged support-distance boundary. This is explicitly
incomplete boundary replacement: the source frame and some old colored edge can
remain. The approved template alpha and artwork are not edited. The renderer reports requested,
applied and suppressed repair pixels and its donor-agreement evidence.

The donor-gap reconstruction can erase a real narrow frame/ridge and substitute
a different layer such as bright car paint. This previously caused pale erosion
dots at a jagged support boundary; soft coverage alone could not fix the wrong
background estimate. Abstaining from ambiguous reconstruction prevents this
specific failure but can retain a pre-existing colored rim. Stable-background
agreement is necessary, not proof of recovering the unseen bumper. Ambiguous
holes, unstable donors, substantial expansion or a large reconstructed fraction
are marked `needs_review`. No threshold depends on photo identity or surface hue.

The actual declared edit support and protection are available through the
`edit_support_out` dictionary (`bbox_xyxy`, local `support`, local `protected`).
Metadata remains JSON-compatible. The correct invariants are exact equality
outside declared edit support and exact equality on protected foreground. The raw
segmentation boundary is an observation and is not the edit-support invariant.

`replace_surface(..., blur_sigma=None)` delegates blur entirely to the selected
degradation estimate. An explicit number, including zero, is a user override
and is applied exactly rather than being raised to a legacy focus floor.

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
