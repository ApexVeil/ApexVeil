# Temporal composition on native SAM3.1 tracks

The active video path remains **text initialization + SAM3.1 native video
propagation**. Its masks and object identities are shared by all compositing
comparisons. The framewise comparison independently estimates placement and
appearance; it is a diagnostic ablation, not independent per-frame SAM detection.


## Continuous placement

The image-derived quadrilateral is an observation, not a command to snap the
replacement to new coordinates. Pyramidal LK and RANSAC first predict the plane
at the current time using visible surface features. This preserves translation,
acceleration and perspective change instead of lagging behind the car with a
screen-coordinate average.

When observations disagree with motion, the estimator checks their agreement
with the current visible mask. Unsupported observations can be rejected. Other
corrections have a confidence-dependent magnitude bound relative to surface
size. A single bad corner is checked separately from the median corner error.
Disagreements and rejected/limited updates remain explicit warnings. Pure
prediction without trusted observation is still limited to five frames; the
limit was not increased to make a test clip look complete.

## Continuous appearance

Averaging reference pixels alone did not stabilize the later nonlinear exposure,
black-floor and color estimates. The shared compositor now accepts an optional
parameter-state callback. Still images omit it. Video uses `TemporalPhotometry`
to stabilize exposure in log space, additive black-field color and material tint.

A large fitted exposure change without corresponding source evidence is rate
limited. When both median and upper-middle source luminance change coherently,
the filter responds quickly for a short interval. Its time constants account
for the output frame rate. The current spatial light field is kept, so this
parameter filter does not smear moving shadow boundaries across time. Existing
softness estimation and its state remain in use.

The video path estimates material parameters from the **current** rectified
reference. It no longer applies the former averaged reference to the final
material fit. That previous reference state still supplies existing softness
state and diagnostics. All appearance states reset after a detected cut or lost
geometry. This is an approximate compositing model, not physical relighting.

## Validation and limits

Synthetic checks distinguish unwanted estimator spikes from real changes:
accelerating translation, perspective change, an isolated incorrect corner,
a prediction timeout, an isolated fitted-exposure spike, a true exposure step,
a moving shadow, and state reset between shots. Real-frame metrics measure
internal consistency and cannot substitute for manually checked alignment.

The same two private 60/75-frame clips are used for the current development
comparison, with unchanged native SAM masks, thresholds, target selection and
manual false-positive exclusion. Previous and updated temporal renders are
presented together. This is not a held-out dataset, and full occlusion/reentry,
long-shot drift, rolling shutter, directional motion blur and soft foreground
matting still need separate evaluation.
