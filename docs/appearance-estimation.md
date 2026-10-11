# Scene-layer compositing and bounded lighting guidance

The renderer distinguishes observed plate visibility, declared old-material
removal support, approved template alpha, and protected foreground. These are
not interchangeable masks. Native template alpha is projected and blurred by
the capture PSF; background uncertainty never erodes it or deliberately retains
an inward ring of old material. Outside declared edit support and on protected
foreground, source pixels remain exact.

Without semantic foreground or an explicit removal mask, the default removal
support remains observed/color-supported visibility. A quadrilateral alone does
not authorize painting a large unobserved region. `old_surface_mask` can declare
a full removal domain. `foreground_mask` supplies independently parsed semantic
foreground and can resolve conservative native-hole inference; legacy
`protected_mask` always remains an additional strict protection. A person mask
from SAM3.1 is model evidence, not ground truth, and must be reviewed against the
RGB and native plate mask. No foreground inference based on a distance cutoff
certifies a person silhouette.

Old material beneath rounded corners/optical transitions is removed using a
local background estimate. Adjacent and gap-separated donor reconstructions
provide an uncertainty diagnostic. If they disagree, the nearest exterior
approximation is used, without shrinking material coverage. This cannot recover
unseen vehicle texture with certainty. Previous donor abstention plus inward
coverage produced green rings and smaller replacement footprints; that default
has been removed rather than described as a quality success.

`AppearanceEstimate` still contains gain, canonical RGB additive field (`black`
for API compatibility), diffuse tint and multiplicative spatial shading.
Resolved ink supports a spatial RGB veil estimate. A single context dark tint
must not overwrite that field: the old green albedo is not illumination, and
warm camera flare must remain spatially colored. Sparse/unresolved ink retains
an explicitly uncertain context fallback. This is not calibrated inverse optics.

`replace_surface(..., scene_guidance=...)` accepts `SceneAppearanceGuidance` or
JSON containing optional `exposure_gain_linear`, `additive_rgb_linear`,
`flare_control_points`, `shading_control_points`, `diffuse_tint_rgb`, and
`provenance`. Control points contain normalized canonical `xy` plus
`rgb_linear` (flare) or `value` (shading). At most sixteen finite points are
accepted. Fields are interpolated continuously; diffuse shading and additive
veil are independent. Guidance never supplies replacement/logo pixels.
The deterministic renderer always uses the approved asset. Reference guidance
and reference geometry must be labelled separately from automatic estimation.

Blur `None` delegates exactly to the injected degradation estimate. A numeric
blur, including zero, is an explicit override. Capture noise remains
source-derived and seeded; video seeds should vary deterministically by frame.
Temporal smoothing of flare should operate on coefficients/levels while keeping
current-frame spatial shape rather than attaching a stale canonical glare field
to a moving plate.

Current evidence prioritizes full-image and crop comparisons for sunlight,
complete video masks and occluded reference geometry. Small synthetic checks
cover independent flare/shading, exact foreground protection, approved alpha and
explicit removal authority. These checks establish contracts, not natural image
quality. Automatic flare magnitude, incomplete foreground evidence and unknown
background behind rounded corners remain visual limitations for the guided loop.
