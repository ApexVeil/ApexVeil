# Deterministic surface geometry and offline trajectory estimation

## Contract and integration

`estimate_surface_geometry(rgb, visible_mask, template_aspect_ratio,
observed_quad=None)` returns `SurfaceGeometry` with an ordered float32 quad or
None, the original visible mask, a bounded evidence score, flags, and JSON-safe
diagnostics. RGB can be None for a mask-only estimate. The mask is not repaired,
unioned, expanded, or converted to inferred visibility.

`refine_shot_geometry(frames, masks, template_aspect_ratio, cut_indices=(),
observed_quads=None, max_anchor_distance=12, identity=None)` returns an equal
length list of the same contract. Inputs are sequences and can lazily decode
frames. The caller supplies one already associated native identity. A cut index
starts a new shot. Alias association, target selection, and exclusions belong
upstream; calling this function on different IDs does not merge them.

The existing fit_quad, edge_refine, assess_geometry and
template_geometry_flags call signatures remain compatible. The latter now
rejects degenerate or ill-conditioned projections rather than comparing image
aspect to artwork aspect. A homography maps a rectangle to any nondegenerate
convex quadrilateral: apparent width alone does not establish missing surface.

## Observation estimator

The estimator intersects combinations of supported convex-hull sides, allowing
a modest missing corner where four real boundary segments remain. Candidate
planes must contain the observed hull and have bounded expansion. RGB gradient
refinement is accepted only when native boundary support does not worsen.
Separate diagnostics measure the fraction of each boundary with observed support,
visible-pixel containment, and the fraction of the projected plane currently
visible. These quantities answer different questions.

Image-boundary clipping returns no single-view plane. Unsupported sides,
collinear masks, and ill-conditioned projections return uncertainty. Small
surfaces reduce confidence. Four-sided visible fragments can remain ambiguous:
a foreground object may create a straight boundary that looks like a surface
side. Consequently confidence is explicitly an uncalibrated boundary-evidence
score; completeness_confidence is None and full_plane_status is
boundary_hypothesis_not_semantic_confirmation. It is not a calibrated probability
that all physical corners have been recovered. There is no protected foreground
mask: inferring one from arbitrary mask holes would be unjustified.

## Offline trajectory refinement

Each adjacent pair is independently tracked in both directions using the existing
LK/RANSAC projective motion implementation. Features begin on eroded native
visibility and their destinations must remain currently visible. Reverse fitting
is independent rather than merely inverting a forward fit. Corner cycle error,
current-mask containment, fit confidence, and explicit cuts reject unsupported
links. Large whole-frame changes additionally break propagation conservatively.

For each time, independently observed anchors on both sides are transported by
the measured motion chain into that time's coordinates. A bounded-influence,
medoid-initialized robust consensus jointly estimates its eight corner
coordinates from all nearby anchors. Thus the whole offline trajectory shares
observation evidence without averaging corners at different screen positions.
Real acceleration remains in the measured transforms; it is not penalized by
a constant-velocity or screen-coordinate smoothness prior. Motion confidence and
distance reduce anchor weights. All anchor frame indices, directions, weights,
fit/cycle diagnostics and consensus residuals are retained.

This is local-window batch motion-normalized trajectory estimation, not a global
bundle-adjustment camera reconstruction. It does not infer unseen pixels,
bridge native empty masks, propagate across cuts, or use a neighboring mask as
the current compositing mask. Temporally recovered clipped geometry retains
the observation's partially_out_of_frame flag and adds
temporally_supported_full_plane; consumers should examine quad presence and
provenance rather than treating the historical observation flag as a new
geometry failure.

## Reproducible verification

CPU tests in test_offline_geometry.py cover known projective foreshortening,
supported missing corners plus foreground holes, single-view clipping refusal,
accelerating projective trajectories, corrupt corners, backward clipped-plane
recovery, cut boundaries, empty visibility, collinear evidence, and deterministic
repeatability. Existing geometry-refinement and temporal tests remain applicable.
The combined geometry/offline/temporal suite passes 28 tests.

The private evaluation uses the entire reviewed 70-second shot manifest,
complete native tracks for every selected primary identity, and all nine cached
high-resolution still images. Explicit false-ID exclusions are respected.
The aliases in one distant-car shot are evaluated independently, so aggregate
frame counts are track-frame counts, not unique-car recall. Geometry uses cached
SAM3.1 native forward masks; no SAM inference, backward segmentation rerun,
neural estimator, training, GPU allocation, or manual correction is performed.

Private artifacts live under outputs/geometry-offline-vnext-20261010: an
evaluation script, log, summary, per-shot/identity JSON, and before/after
RGB-mask-quad overlays. Hashes identify input video, mask bytes, native report,
manifest and estimator sources. None of these private artifacts are committed.
Overlays show the former fitted quad and new inferred quad; native visibility
contours remain visible independently.

## Observed limits and interpretation

The previously rejected angled shots now admit supported projective geometry.
Among the deliberately difficult full shots, supported/native-visible counts
were 15/15, 58/58, 30/30 and 5/5 for the two independent distant aliases,
83/89 for the clipped sequence, 15/15 for the foliage sequence, and 27/29
for the other angled sequence. The clipped sequence obtained nine
motion-supported recoveries. Native empty detections remain empty.

Across all reviewed primary identities, the full run covers 27 track sequences
from 26 shots: 1,588 track-frames, 1,416 native-visible frames, 1,402 supported
geometry outputs, 39 temporal recoveries, and 1,303 accepted adjacent motion
links. The remaining shots have no reviewed selected primary identity and are
not silently treated as successful detections.

These counts are evidence availability, not ground-truth quality or success
rates. No automatic corner ground truth is available. The reviewed overlays show
useful stabilization in blurred angled shots, but native-mask inaccuracies and
small textures still bound precision.

Two partially occluded still targets remain failures of complete-plane
identification: the partial plate behind a person in IMG_9998, and the small
background plate behind foreground occlusion in IMG_0100. Their fitted boundary
hypotheses can still follow the occluder instead of the hidden plate edge.
Diagnostics do not claim semantic completeness. The clipped IMG_0287 plane
correctly remains unavailable without temporal evidence. A very weak incidental
candidate in IMG_0103 is rejected by boundary evidence.

A tested RGB edge-continuation heuristic was deliberately not shipped: curved
foreground boundaries did not reliably continue along fitted line extensions,
while vehicle trim sometimes continued valid surface edges. Lower warning
counts must not be interpreted as resolving these examples. Stronger automatic
single-view completeness evidence requires additional justified image/scene
constraints or a separately evaluated estimator; this implementation does not
silently manufacture hidden corners.


## Product video integration

Both run_video and render_shot accept an optional Estimators bundle. The shared
video_pipeline.estimate_geometry_track helper invokes its GeometryEstimator
exactly once per frame, then passes precomputed SurfaceGeometry objects into
refine_shot_geometry via surface_estimates. Provider implementations do not need
private candidate_quad diagnostics when they supply a valid quad.

PNG and video frame sequences decode lazily with a one-frame RGB cache. Montage
processes each declared native primary ID independently before applying its
existing alias selection rule; no mask merge or geometry state transfer occurs.
Alias switches and missing visibility reset material-parameter filtering.
Geometry is refined offline once, with no subsequent TemporalGeometry pass.
Appearance uses current-frame visible reference plus TemporalPhotometry parameter
filtering; no image-reference average is applied.

The shared render_surface_pair helper preserves the existing independent and
temporal output interfaces, injects boundary/degradation/appearance providers,
uses a deterministic frame-and-identity noise seed, and returns the compositor's
declared support separately from JSON metadata. Both product paths verify each
render independently against that declared support. The historical video field
outside_visible_masks_changed_pixels now reports its literal quantity, while
outside_edit_support_changed_pixels is the protected-output invariant.
Boundary needs_review propagates to frame review status. Historical clipping
and insufficient-boundary flags that have a supported recovered quad remain in
geometry provenance; they do not alone prevent rendering.

Default montage cache signatures include estimator metadata and all relevant
source hashes. Explicit provider injection runs in-process rather than trying
to pickle arbitrary provider objects, and bypasses existing render-cache reuse.
No learned provider or neural runtime was added.

Product tests additionally exercise provider call counts, injected estimates
without private diagnostics, alias-separated histories, deterministic varying
seeds, protected support, review propagation, and lazy frame access. A real
15-frame angled-shot montage smoke rendered all 15 frames and verified encoded
frame count and 0.5-second duration. One frame retained boundary review status.
Private evidence is under outputs/core-montage-vnext-smoke-20261010.



The bounded run_video product route was also exercised against the same complete
15-frame native cache, with separate original, independent, temporal and detail
comparison encodes. All four outputs verified 15 frames and 0.5-second duration;
all 15 target frames rendered, one retained boundary review, and the cache
provenance check passed. The corresponding private report is
outputs/core-video-vnext-smoke-20261010/report.json. These product smokes validate
the executable path, not the quality of the complete 70-second diagnostic export.
