# Bounded single-shot video workflow

The Studio calls `prepare_video` once, lets the user choose a native ID, then
calls `render_video` as often as needed without repeating SAM.
`restore_video_prepared` verifies and reopens the lossless clip and raw masks from
disk without loading a model; a changed source or requested interval requires Stage 1 again.
New caches retain exact requested and effective intervals; older caches compare
the interval using the processing frame rate.
Both functions accept
`progress(message: str, fraction: float | None)`. Progress text does not expose
advisor model IDs or backend/device configuration.

Preparation supports one continuous SDR shot, with a requested interval of at
most eight seconds. Processing is bounded to 1920×1080 and 30 fps. Resizing and
CFR resampling are reported in public warnings and source/processing metadata.
Portrait rotation metadata is accounted for. Suspected cuts or flashes are
conservatively rejected before segmentation; this heuristic does not certify
that every possible scene boundary has been detected. HDR requires an explicit
conversion outside this workflow.

The caller owns single-GPU scheduling. Pass an existing
`ImageSegmenter.predictor` to avoid another model load. Otherwise the workflow
lazily builds the same native predictor. It uses text initialization followed by
native forward propagation; no point prompt conversion, ID association, mask
union, or invented missing detections occurs. Sessions close after propagation.
Partial native failures remain explicit in the cache and warnings.

`prepared["public"]` is JSON-compatible and contains preview paths, processing
width/height/fps/duration, source size/fps, targets and timings. Each target's
bbox, area and native score refer to **that target's representative frame**.
Targets may not coexist in one frame: the UI must use the selected target's
`source_preview_path` and `mask_preview_path`, not draw its box onto another
target's preview. `color_rgb` matches the native-mask overlay palette. Private
prepared state contains a re-openable one-frame video cache, native mask paths,
records and optional reused predictor; it must not be sent to the browser.

Replacement generates scene candidates for all frames and selects one
trajectory before rendering. Each frame uses the shared `solve_scene`, with
the preselected candidate (including explicit None for unresolved frames).
The selected planes then undergo offline bidirectional, motion-compensated
robust fusion. Adjacent native surface texture constrains transport; missing
geometry can recover only where the current native mask exists. Missing visibility
breaks propagation. This is not screen-coordinate averaging, which would lag
fast motion. Isolated mask specks are excluded only from geometric fitting,
consistently across boundary and cropped-side estimation. Native masks stay intact.

`TemporalVideoAppearance` stabilizes capture softness before the ink-resolution
decision. Appearance uses matching, jointly visible canonical source samples to
distinguish light changes from crop changes, bounds exposure and the entire RGB
veil envelope, and preserves current spatial light positions. These are conservative
estimates, not physical relighting. Directional motion blur is not yet modeled. Non-target native masks are blurred without overwriting
the selected native mask. Blur and mosaic modes need neither geometry nor AI.

If enabled for replacement, the advisor has a hard budget of one call for the
continuous shot, using a representative-frame casepack and nearby-frame
evidence. Absolute advised corners apply to their representative frame;
native-visible forward/backward motion may transport them as candidates.
Failure or missing visibility stops transport, while automatic scene candidates
remain available. There is no automatic second quality-review call.

Video advisory exposure and tint act only as bounded corrections relative to the
representative frame's automatic estimate. Additive, flare and shading fields
are not copied unchanged across the clip: doing so would pin moving illumination
to artwork UV. Ignored advisory fields are reported. An advisor failure retains
automatic candidates with an explicit warning, never an AI-success claim.
Neither a valid JSON proposal nor a rendered output constitutes quality acceptance.

Rendering returns paths for the browser-ready result, RGB | raw SAM | result
diagnostic video, and JSON report, plus timings, warnings and media_type. Source
audio, when present, is explicitly converted to AAC for browser playback and is
not reported as packet-identical. A shorter audio track does not truncate video. All-target replacement failure raises an
explicit unresolved error after writing diagnostics/report, rather than
returning an original video as a successful replacement.

Timing fields distinguish preparation, model cold load, native segmentation,
geometry candidates, trajectory selection, evidence packaging, the one advisor
call, composition and encoding. Encoding timing includes writer I/O/wait and
verification, not exclusive encoder CPU utilization. Root's separately measured
model cold load must not be double-counted: when a predictor is supplied,
`sam_cold_load_seconds` is zero. Pipeline wall time excludes user target-selection
waiting; it is not a claim of realtime processing.

Existing historical evidence for the old 70-second run: native per-shot
`inference_seconds` sum to approximately 206.966 seconds across 30 tracked
shots. That excludes predictor cold load, but includes per-shot session/frame
loading and mask persistence. The 38 old render `seconds` values sum to
approximately 904.001 seconds and cover independent/temporal comparison work;
parallel per-shot wall times cannot be presented as end-to-end elapsed time.
The root historical report has no end-to-end total_seconds field. New Studio
measurements should use its actual preparation/render wall clocks.
